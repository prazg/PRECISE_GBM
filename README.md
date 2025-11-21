#Predictive Radiomics for Evaluation of Cancer Immune SignaturE in Glioblastoma| PRECISE-GBM

<p align="center">
  <img src="PRECISE-GBM_GUI_logo%20(1).png" alt="PRECISE-GBM Logo">
</p>

Project: PRECISE-GBM - Model training & retraining helpers

Overview

This repository contains code to train models (Gaussian Mixture labelling + SVM and ensemble classifiers) and to persist all artifacts required to reproduce or retrain models on new data. It includes:

- `Scenario_heldout_final_PRECISE.py` — training pipeline producing `.joblib` models and metadata JSONs (selected features, best params, CV results).
- `retrain_helper.py` — CLI utility to rebuild pipelines, set best params and retrain using saved selected-features and params JSONs. Supports JSON/YAML config files and auto-detection of model type.
- `README_RETRAIN.md` — detailed retrain examples and a notebook cell.

This repo also includes helper files to make it ready for GitHub:
- `requirements.txt` — Python dependencies
- `.gitignore` — recommended ignores (models, caches, logs)
- `LICENSE` — MIT license
- GitHub Actions workflow for CI (pytest smoke test)

Getting started (Windows PowerShell)

1) Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2) Install dependencies

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

3) Run training (note: the training script reads data from absolute paths configured in the script — adjust them or run from an environment where those files are present)

```powershell
python Scenario_heldout_final_PRECISE.py
```

The training script will create model files under `models_LM22/` and `models_GBM/` and write metadata JSONs next to each joblib model (selected features, params, cv results) as well as group-level JSON summaries.

Retraining

See `README_RETRAIN.md` for detailed CLI and notebook examples. Short example:

```powershell
python retrain_helper.py \
  --model-prefix "models_GBM/scenario_1/GBM_scen1_Tcell" \
  --train-csv "data\new_train.csv" \
  --label-col "label"
```

Notes

- The training script contains hard-coded absolute paths to data files. Before running on another machine, update the `scenarios_*` file paths or place the datasets in the same paths.
- Retrain helper auto-detects model type when `--model-type` is omitted by looking for `{prefix}_svm_params.json` or `{prefix}_ens_params.json`.
- YAML config support for retrain requires PyYAML (`pip install pyyaml`).

CI

A basic GitHub Actions workflow runs a smoke pytest to ensure the retrain helper imports and basic pipeline construction works. It does not run heavy training.

Contributing

See `CONTRIBUTING.md` for guidance on opening issues and PRs.

License

This project is released under the MIT License — see `LICENSE`.




