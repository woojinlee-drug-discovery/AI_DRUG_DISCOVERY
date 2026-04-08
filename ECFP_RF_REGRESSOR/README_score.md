# DP1 Candidate Scoring with a Trained Random Forest Model

This script applies a trained random forest regression model to score candidate DP1 ligands from a multi-SDF file.

It reads candidate molecules, generates Morgan fingerprints with RDKit, predicts `pAct` values using a previously trained model, converts predicted potency into approximate nanomolar values, and exports a ranked CSV file.

## Input Files

- `dp1_125_hits.sdf`  
  Multi-SDF file containing candidate molecules

- `dp1_rf_regressor.joblib`  
  Trained random forest regression model

## Output File

- `dp1_125_scored.csv`

## What the Script Does

### 1. Load candidate molecules from SDF
The script reads a multi-SDF file and extracts:

- molecular properties from the SDF
- RDKit molecule objects
- canonical SMILES
- row index

Hydrogens are removed before generating canonical SMILES so that the representation is as consistent as possible with the training data.

### 2. Convert selected properties to numeric values
If present, the script attempts to convert the following fields into numeric format:

- `Score`
- `RTCNNscore`
- `molLogS`
- `molLogP`
- `molWeight`
- `MW`
- `Tox_Class`
- `Tox_Score`
- `molPAINS`
- `orig_Score`
- `orig_Score2`
- `Hbond`
- `Hphob`
- `VwInt`
- `Eintl`
- `Dsolv`
- `SolEl`
- `mfScore`
- `dTSsc`
- `Natom`
- `Nflex`
- `L`
- `IX`
- `CONF`
- `RecConf`
- `O`
- `CO`
- `ord`
- `cl`

### 3. Build Morgan fingerprint features
The script generates Morgan fingerprints using RDKit with:

- radius = `2`
- fingerprint size = `2048`

These fingerprints are converted into NumPy arrays and used as model input.

### 4. Predict activity
The script loads the trained regression model from:

- `dp1_rf_regressor.joblib`

It then predicts:

- `predicted_pAct`

### 5. Convert predicted pAct to approximate nM
The script estimates approximate nanomolar potency using:

```text
nM ≈ 10^(9 - predicted_pAct)
