from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from transformers import AutoModel, AutoTokenizer

CANDIDATE_SDF = "dp1_125_hits.sdf"
MODEL_FILE = "dp1_molformer_xgb_regressor.joblib"
HF_MODEL = "ibm-research/MoLFormer-XL-both-10pct"
OUT_CSV = "dp1_125_scored_molformer.csv"
BATCH_SIZE = 32
MAX_LENGTH = 202
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def canonicalize_smiles_no_isomeric_from_mol(mol):
    mol = Chem.RemoveHs(mol)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)


def load_candidates_from_sdf(sdf_path: str) -> pd.DataFrame:
    rows = []
    suppl = Chem.SDMolSupplier(sdf_path, removeHs=False)
    for i, mol in enumerate(suppl):
        if mol is None:
            continue
        d = {p: mol.GetProp(p) for p in mol.GetPropNames()}
        d["mol"] = mol
        d["smiles"] = canonicalize_smiles_no_isomeric_from_mol(mol)
        d["row_index"] = i + 1
        rows.append(d)
    df = pd.DataFrame(rows)
    numeric_cols = [
        "Score", "RTCNNscore", "molLogS", "molLogP", "molWeight", "MW",
        "Tox_Class", "Tox_Score", "molPAINS", "orig_Score", "orig_Score2",
        "Hbond", "Hphob", "VwInt", "Eintl", "Dsolv", "SolEl", "mfScore",
        "dTSsc", "Natom", "Nflex", "L", "IX", "CONF", "RecConf", "O", "CO", "ord", "cl"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


class MolFormerEmbedder:
    def __init__(self, model_name: str = HF_MODEL, device: str = DEVICE):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, deterministic_eval=True, trust_remote_code=True)
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
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                emb = outputs.pooler_output
            else:
                hidden = outputs.last_hidden_state
                mask = toks["attention_mask"].unsqueeze(-1)
                emb = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            all_embeds.append(emb.detach().cpu().numpy().astype(np.float32))
        return np.vstack(all_embeds)


def main():
    reg = joblib.load(MODEL_FILE)
    df = load_candidates_from_sdf(CANDIDATE_SDF)
    if len(df) == 0:
        raise ValueError("No valid molecules were read from candidate SDF.")

    if "ID" not in df.columns:
        df["ID"] = [f"mol_{i+1}" for i in range(len(df))]

    embedder = MolFormerEmbedder(model_name=HF_MODEL, device=DEVICE)
    X = embedder.encode(df["smiles"].tolist(), batch_size=BATCH_SIZE)
    df["predicted_pAct"] = reg.predict(X)
    df["predicted_nM"] = 10 ** (9 - df["predicted_pAct"])

    df = df.sort_values("predicted_pAct", ascending=False).copy()

    preferred_cols = [
        "ID", "predicted_pAct", "predicted_nM",
        "Score", "RTCNNscore", "orig_Score", "orig_Score2",
        "molLogP", "molLogS", "MW", "molWeight",
        "Tox_Class", "Tox_Score", "molPAINS", "badGroups",
        "Nflex", "NAME", "smiles"
    ]
    existing_cols = [c for c in preferred_cols if c in df.columns]
    other_cols = [c for c in df.columns if c not in existing_cols and c != "mol"]
    df = df[existing_cols + other_cols]
    df.to_csv(OUT_CSV, index=False)

    print(f"Saved: {OUT_CSV}")
    print(df[[c for c in ["ID", "predicted_pAct", "predicted_nM", "Score", "molLogP", "molLogS", "Tox_Class", "molPAINS"] if c in df.columns]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
