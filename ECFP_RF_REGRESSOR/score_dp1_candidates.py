from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
import pandas as pd
import numpy as np
import joblib

# -----------------------------
# Settings
# -----------------------------
CANDIDATE_SDF = "dp1_125_hits.sdf"   # Multi-SDF containing 125 candidate molecules
MODEL_FILE = "dp1_rf_regressor.joblib"
OUT_CSV = "dp1_125_scored.csv"

FP_RADIUS = 2
FP_SIZE = 2048


def load_candidates_from_sdf(sdf_path: str) -> pd.DataFrame:
    rows = []
    suppl = Chem.SDMolSupplier(sdf_path, removeHs=False)

    for i, mol in enumerate(suppl):
        if mol is None:
            continue

        # Read molecule properties
        d = {p: mol.GetProp(p) for p in mol.GetPropNames()}

        # Remove hydrogens and generate canonical SMILES
        # to keep the representation as consistent as possible with training
        mol_noh = Chem.RemoveHs(mol)
        smiles = Chem.MolToSmiles(mol_noh)

        d["mol"] = mol_noh
        d["smiles"] = smiles
        d["row_index"] = i + 1

        rows.append(d)

    df = pd.DataFrame(rows)

    # Convert selected columns to numeric
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
    print("Loading model...")
    reg = joblib.load(MODEL_FILE)

    print("Reading candidate SDF...")
    df = load_candidates_from_sdf(CANDIDATE_SDF)
    print(f"Loaded molecules: {len(df)}")

    # Ensure an ID column exists
    if "ID" not in df.columns:
        df["ID"] = [f"mol_{i+1}" for i in range(len(df))]

    # Build fingerprints
    print("Building ECFP features...")
    X = build_features(df)

    # Predict pAct
    print("Predicting pAct...")
    df["predicted_pAct"] = reg.predict(X)

    # Approximate nM conversion
    # pAct = -log10(M), so nM ~= 10^(9 - pAct)
    df["predicted_nM"] = 10 ** (9 - df["predicted_pAct"])

    # Sort by predicted potency
    sort_cols = ["predicted_pAct"]
    ascending = [False]

    df_out = df.sort_values(sort_cols, ascending=ascending).copy()

    # Move preferred columns to the front
    preferred_cols = [
        "ID", "predicted_pAct", "predicted_nM",
        "Score", "RTCNNscore", "orig_Score", "orig_Score2",
        "molLogP", "molLogS", "MW", "molWeight",
        "Tox_Class", "Tox_Score", "molPAINS", "badGroups",
        "Nflex", "NAME", "smiles"
    ]
    existing_cols = [c for c in preferred_cols if c in df_out.columns]
    other_cols = [c for c in df_out.columns if c not in existing_cols and c != "mol"]
    df_out = df_out[existing_cols + other_cols]

    df_out.to_csv(OUT_CSV, index=False)

    print(f"Saved: {OUT_CSV}")
    print("\nTop 10 candidates:")
    show_cols = [c for c in ["ID", "predicted_pAct", "predicted_nM", "Score", "molLogP", "molLogS", "Tox_Class", "molPAINS"] if c in df_out.columns]
    print(df_out[show_cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
