# MoLFormer + XGBoost Model Training

This folder contains the training workflow for building a DP1 activity prediction model using MoLFormer embeddings and XGBoost regression.

## Script

- `molformer_xgb_pipeline.py`

## Description

This script trains a regression model for predicting `pAct` values from a ChEMBL-style SDF dataset.

The workflow:

- loads `Activity_CHEMBL4427.sdf`
- keeps only exact activity measurements (`relation == "="`)
- removes rows with missing `pAct`
- keeps selected assay types:
  - `IC50`
  - `Ki`
  - `Kb`
  - `EC50`
- deduplicates molecules at the molecule level using median `pAct`
- canonicalizes SMILES and removes isomeric information to better match MoLFormer pretraining assumptions
- generates Bemis–Murcko scaffolds for grouped train/test splitting
- computes molecular embeddings using the Hugging Face model:
  - `ibm-research/MoLFormer-XL-both-10pct`
- trains an `XGBRegressor` on the embedding vectors
- evaluates the model using RMSE and R²
- saves the cleaned dataset, embeddings, trained model, and metadata

## Input

- `Activity_CHEMBL4427.sdf`

## Output

- `dp1_training_clean_molformer.csv`
- `dp1_molformer_embeddings.npz`
- `dp1_molformer_xgb_regressor.joblib`
- `dp1_molformer_xgb_metadata.json`

## Model Details

### Embedding model
- Hugging Face model: `ibm-research/MoLFormer-XL-both-10pct`
- tokenizer and encoder loaded through `transformers`
- device automatically selected:
  - `cuda` if available
  - otherwise `cpu`

### Regression model
- `XGBRegressor`
- objective: `reg:squarederror`

### Evaluation metrics
- RMSE
- R²

## Data Processing Notes

- Only exact measurements are used.
- Molecules are grouped by scaffold using `GroupShuffleSplit`.
- Test size is set to `0.20`.
- Random state is fixed at `42`.
- Canonical SMILES are converted to non-isomeric form before embedding.

## Requirements

- Python
- RDKit
- pandas
- NumPy
- PyTorch
- transformers
- scikit-learn
- xgboost
- joblib

## Usage

```bash
python molformer_xgb_pipeline.py
