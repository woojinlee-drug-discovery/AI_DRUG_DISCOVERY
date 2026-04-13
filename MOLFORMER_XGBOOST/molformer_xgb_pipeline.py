from __future__ import annotations

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBRegressor

# Hugging Face / Transformers
from transformers import AutoModel, AutoTokenizer

# -----------------------------
# User settings
# -----------------------------
TRAIN_SDF = "Activity_CHEMBL4427.sdf"
MODEL_NAME = "ibm-research/MoLFormer-XL-both-10pct"
USE_STANDARD_TYPES = {"IC50", "Ki", "Kb", "EC50"}  # tighten later if needed
TEST_SIZE = 0.20
RANDOM_STATE = 42
BATCH_SIZE = 32
MAX_LENGTH = 202  # model card notes molecules longer than 202 tokens were dropped in pretraining
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# outputs
OUT_CLEAN_CSV = "dp1_training_clean_molformer.csv"
OUT_EMBED_NPZ = "dp1_molformer_embeddings.npz"
OUT_MODEL = "dp1_molformer_xgb_regressor.joblib"
OUT_META = "dp1_molformer_xgb_metadata.json"


def load_sdf_to_df(sdf_path: str) -> pd.DataFrame:
    rows = []
    suppl = Chem.SDMolSupplier(sdf_path)
    for mol in suppl:
        if mol is None:
            continue
        d = {p: mol.GetProp(p) for p in mol.GetPropNames()}
        d["mol"] = mol
        d["smiles"] = Chem.MolToSmiles(mol)
        rows.append(d)
    df = pd.DataFrame(rows)
    for col in ["pAct", "pchembl_value", "standard_value", "document_year"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def canonicalize_smiles_no_isomeric(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)


def clean_training_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df[df["relation"] == "="]
    df = df[df["pAct"].notna()]
    df = df[df["standard_type"].isin(USE_STANDARD_TYPES)]

    grouped = (
        df.groupby("molecule_chembl_id")
        .agg(
            canonical_smiles=("canonical_smiles", "first"),
            pAct=("pAct", "median"),
            n_measurements=("pAct", "size"),
            n_assays=("assay_chembl_id", "nunique"),
            standard_types=("standard_type", lambda s: ";".join(sorted(set(s)))),
        )
        .reset_index()
    )

    grouped["mol"] = grouped["canonical_smiles"].apply(Chem.MolFromSmiles)
    grouped = grouped[grouped["mol"].notna()].reset_index(drop=True)

    # Match MoLFormer pretraining assumptions as closely as practical:
    # canonicalize and remove isomeric information.
    grouped["model_smiles"] = grouped["canonical_smiles"].apply(canonicalize_smiles_no_isomeric)
    grouped = grouped[grouped["model_smiles"].notna()].reset_index(drop=True)

    return grouped


def murcko_scaffold(smiles: str) -> str:
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(smiles=smiles)
    except Exception:
        return ""


class MolFormerEmbedder:
    def __init__(self, model_name: str = MODEL_NAME, device: str = DEVICE):
        self.model_name = model_name
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_name,
            deterministic_eval=True,
            trust_remote_code=True,
        )
        self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def encode(self, smiles_list: list[str], batch_size: int = BATCH_SIZE) -> np.ndarray:
        all_embeds = []
        for start in range(0, len(smiles_list), batch_size):
            batch = smiles_list[start:start + batch_size]
            toks = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            toks = {k: v.to(self.device) for k, v in toks.items()}
            outputs = self.model(**toks)

            # Preferred path per model card example
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                emb = outputs.pooler_output
            else:
                # Fallback: masked mean pooling over last_hidden_state
                hidden = outputs.last_hidden_state
                mask = toks["attention_mask"].unsqueeze(-1)
                emb = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

            all_embeds.append(emb.detach().cpu().numpy().astype(np.float32))
        return np.vstack(all_embeds)


def main():
    print(f"Loading raw SDF: {TRAIN_SDF}")
    df_raw = load_sdf_to_df(TRAIN_SDF)

    print("Raw rows:", len(df_raw))
    print("Unique molecules:", df_raw["molecule_chembl_id"].nunique())
    print("Target:", df_raw["target_pref_name"].iloc[0], "|", df_raw["Uniprot_ID"].iloc[0])
    print("Relation counts:\n", df_raw["relation"].value_counts(dropna=False))
    print("Standard type counts:\n", df_raw["standard_type"].value_counts(dropna=False))

    df = clean_training_df(df_raw)
    print("\nAfter cleaning/dedup:")
    print("Rows (molecule-level):", len(df))
    print("pAct summary:\n", df["pAct"].describe())

    df["scaffold"] = df["canonical_smiles"].apply(murcko_scaffold)
    y = df["pAct"].values.astype(np.float32)
    groups = df["scaffold"].values
    smiles = df["model_smiles"].tolist()

    print(f"\nLoading MoLFormer on {DEVICE} ...")
    embedder = MolFormerEmbedder(model_name=MODEL_NAME, device=DEVICE)
    X = embedder.encode(smiles, batch_size=BATCH_SIZE)
    print("Embedding matrix shape:", X.shape)

    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    reg = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=700,
        learning_rate=0.03,
        max_depth=5,
        min_child_weight=2,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=4,
    )

    reg.fit(X_train, y_train)
    pred = reg.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    r2 = float(r2_score(y_test, pred))

    print("\n=== MoLFormer + XGBoost regression ===")
    print(f"Test molecules: {len(test_idx)}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R2:   {r2:.3f}")

    df_out = df.drop(columns=["mol"]).copy()
    df_out.to_csv(OUT_CLEAN_CSV, index=False)
    np.savez_compressed(
        OUT_EMBED_NPZ,
        X=X,
        y=y,
        smiles=np.array(smiles, dtype=object),
        molecule_chembl_id=df["molecule_chembl_id"].values,
        scaffold=df["scaffold"].values,
        train_idx=train_idx,
        test_idx=test_idx,
        pred_test=pred,
        y_test=y_test,
    )
    joblib.dump(reg, OUT_MODEL)

    meta = {
        "train_sdf": TRAIN_SDF,
        "hf_model": MODEL_NAME,
        "device": DEVICE,
        "batch_size": BATCH_SIZE,
        "max_length": MAX_LENGTH,
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "use_standard_types": sorted(USE_STANDARD_TYPES),
        "n_rows_raw": int(len(df_raw)),
        "n_molecules_clean": int(len(df)),
        "embedding_dim": int(X.shape[1]),
        "rmse": rmse,
        "r2": r2,
    }
    Path(OUT_META).write_text(json.dumps(meta, indent=2))

    print("\nSaved files:")
    print("-", OUT_CLEAN_CSV)
    print("-", OUT_EMBED_NPZ)
    print("-", OUT_MODEL)
    print("-", OUT_META)


if __name__ == "__main__":
    main()
