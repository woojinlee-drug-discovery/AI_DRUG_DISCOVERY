# DP1 Random Forest Modeling from ChEMBL SDF

This script builds machine learning models for DP1 activity prediction from a ChEMBL-style SDF file.

It performs conservative dataset cleaning, generates Morgan fingerprints with RDKit, applies Murcko scaffold grouping for train/test splitting, and trains:

- a **Random Forest regressor** to predict continuous `pAct`
- an **optional Random Forest classifier** to label compounds as active/inactive based on a `pAct` threshold

## Input

- `Activity_CHEMBL4427.sdf`

## What the script does

### 1. Load the SDF into a pandas DataFrame
The script reads molecules and their properties from the SDF file and extracts:

- molecular properties from the SDF
- RDKit molecule objects
- SMILES strings

It also converts selected columns to numeric format when available, including:

- `pAct`
- `pchembl_value`
- `standard_value`
- `document_year`

### 2. Clean the training data
The cleaning workflow is intentionally conservative:

- keeps only exact measurements where `relation == "="`
- removes rows with missing `pAct`
- keeps only selected assay types:
  - `IC50`
  - `Ki`
  - `Kb`
  - `EC50`
- deduplicates at the molecule level using the **median pAct**

For each unique molecule, the script stores:

- `canonical_smiles`
- median `pAct`
- number of measurements
- number of assays
- standard types observed

### 3. Build molecular features
The script computes **Morgan fingerprints** using RDKit with:

- radius = `2`
- fingerprint size = `2048`

These fingerprints are converted into NumPy arrays for model training.

### 4. Apply scaffold-aware train/test splitting
To reduce scaffold leakage, the script computes **Bemis–Murcko scaffolds** and uses:

- `GroupShuffleSplit`
- test size = `0.2`
- random state = `42`

This means molecules sharing the same scaffold are grouped during splitting.

### 5. Train a regression model
A `RandomForestRegressor` is trained to predict continuous `pAct`.

#### Regression settings
- `n_estimators = 500`
- `random_state = 42`
- `n_jobs = -1`

#### Regression metrics reported
- RMSE
- R²

### 6. Train an optional classification model
A `RandomForestClassifier` is also trained using binary labels:

- **active** if `pAct >= 6.5`
- **inactive** otherwise

#### Classification settings
- `n_estimators = 500`
- `random_state = 42`
- `n_jobs = -1`
- `class_weight = "balanced"`

#### Classification metrics reported
- ROC-AUC
- PR-AUC
- classification report

## Output files

The script saves:

- `dp1_training_clean.csv`  
  Cleaned, molecule-level dataset

- `dp1_rf_regressor.joblib`  
  Trained random forest regression model

- `dp1_rf_classifier.joblib`  
  Trained random forest classification model

## Requirements

Install the following Python packages:

- `rdkit`
- `pandas`
- `numpy`
- `scikit-learn`
- `joblib`

## Usage

```bash
python your_script_name.py
