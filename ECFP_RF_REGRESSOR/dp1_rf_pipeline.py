
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score, average_precision_score, classification_report
import joblib
from pathlib import Path

TRAIN_SDF = "Activity_CHEMBL4427.sdf"

# Conservative cleaning:
# - keep exact values only (relation == "=")
# - use pAct as target
# - optionally keep only selected standard types
USE_STANDARD_TYPES = {"IC50", "Ki", "Kb", "EC50"}  # tighten to {"IC50","Ki"} if you want lower assay heterogeneity

# Optional binary classifier threshold if you later want active/inactive labels
CLASSIFICATION_THRESHOLD = 6.5  # pAct >= 6.5 => active

FP_RADIUS = 2
FP_SIZE = 2048
N_ESTIMATORS = 500
RANDOM_STATE = 42

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

    # numeric conversion
    for col in ["pAct", "pchembl_value", "standard_value", "document_year"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

def clean_training_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # exact measurements only
    df = df[df["relation"] == "="]
    df = df[df["pAct"].notna()]

    # keep selected standard types
    df = df[df["standard_type"].isin(USE_STANDARD_TYPES)]

    # deduplicate at molecule level using median pAct
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

    # rebuild mol objects from canonical smiles
    grouped["mol"] = grouped["canonical_smiles"].apply(Chem.MolFromSmiles)
    grouped = grouped[grouped["mol"].notna()].reset_index(drop=True)

    return grouped

def murcko_scaffold(smiles: str) -> str:
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(smiles=smiles)
    except Exception:
        return ""

def fp_array_from_mol(mol, fpgen):
    fp = fpgen.GetFingerprint(mol)
    arr = np.zeros((fp.GetNumBits(),), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def build_features(df: pd.DataFrame):
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_SIZE)
    X = np.stack([fp_array_from_mol(m, fpgen) for m in df["mol"]])
    return X

def main():
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

    # scaffold groups
    df["scaffold"] = df["canonical_smiles"].apply(murcko_scaffold)
    X = build_features(df)
    y = df["pAct"].values
    groups = df["scaffold"].values

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # 1) Regression model
    reg = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    reg.fit(X_train, y_train)
    pred = reg.predict(X_test)

    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    r2 = float(r2_score(y_test, pred))

    print("\n=== Regression ===")
    print(f"RMSE: {rmse:.3f}")
    print(f"R2:   {r2:.3f}")

    # save cleaned dataset and model
    out_df = df.drop(columns=["mol"]).copy()
    out_df.to_csv("dp1_training_clean.csv", index=False)
    joblib.dump(reg, "dp1_rf_regressor.joblib")

    # 2) Optional classifier
    y_cls = (df["pAct"].values >= CLASSIFICATION_THRESHOLD).astype(int)
    train_idx, test_idx = next(splitter.split(X, y_cls, groups=groups))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y_cls[train_idx], y_cls[test_idx]

    clf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced"
    )
    clf.fit(X_train, y_train)
    prob = clf.predict_proba(X_test)[:, 1]
    pred_cls = (prob >= 0.5).astype(int)

    roc = float(roc_auc_score(y_test, prob))
    pr = float(average_precision_score(y_test, prob))

    print("\n=== Optional classification ===")
    print(f"Threshold for active: pAct >= {CLASSIFICATION_THRESHOLD}")
    print(f"ROC-AUC: {roc:.3f}")
    print(f"PR-AUC:  {pr:.3f}")
    print(classification_report(y_test, pred_cls, digits=3))

    joblib.dump(clf, "dp1_rf_classifier.joblib")

if __name__ == "__main__":
    main()
