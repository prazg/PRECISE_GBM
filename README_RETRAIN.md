Retrain helper — quick start

This file shows a minimal end-to-end example (CLI and notebook cell) to retrain a saved model using the `retrain_helper.py` utility.

1) Quick CLI example

From the project root (PowerShell):

```powershell
# Retrain SVM using explicit CLI args
python retrain_helper.py \
  --model-prefix "models_GBM/scenario_1/GBM_scen1_Tcell" \
  --model-type svm \
  --train-csv "data\new_train.csv" \
  --label-col "label"

# Or let the helper auto-detect model type (it looks for *_svm_params.json or *_ens_params.json)
python retrain_helper.py \
  --model-prefix "models_GBM/scenario_1/GBM_scen1_Tcell" \
  --train-csv "data\new_train.csv" \
  --label-col "label"

# Using a JSON config file (CLI args override config values)
python retrain_helper.py --config retrain_config.json
```

2) Example config files

JSON (retrain_config.json):

```json
{
  "model_prefix": "models_GBM/scenario_1/GBM_scen1_Tcell",
  "train_csv": "data/new_train.csv",
  "label_col": "label",
  "out_dir": "models_GBM/scenario_1/retrained",
  "overwrite": false
}
```

YAML (retrain_config.yml):

```yaml
model_prefix: models_GBM/scenario_1/GBM_scen1_Tcell
train_csv: data/new_train.csv
label_col: label
out_dir: models_GBM/scenario_1/retrained
overwrite: false
```

3) Notebook / Jupyter cell example (end-to-end)

This cell shows minimal steps to (A) run the CLI retrain helper using Python's subprocess, then (B) load the retrained model and run a quick prediction.

```python
# Notebook cell (Jupyter / Colab / Kaggle) - Python
import subprocess
import json
from joblib import load
import pandas as pd

# 1) Run retrain (will create a timestamped retrained model and metadata JSON)
cmd = [
    "python", "retrain_helper.py",
    "--model-prefix", "models_GBM/scenario_1/GBM_scen1_Tcell",
    "--train-csv", "data/new_train.csv",
    "--label-col", "label"
]
print('Running:', ' '.join(cmd))
subprocess.check_call(cmd)

# 2) Locate the retrain metadata file in the model-prefix folder (it has suffix _retrain_meta_YYYYMMDD_HHMMSS.json)
# For the demo, search the output folder and load the latest metadata to find the retrained model path.
import glob, os
meta_files = glob.glob(os.path.join('models_GBM','scenario_1','GBM_scen1_Tcell*_retrain_meta_*.json'))
meta_files = sorted(meta_files)
print('Found meta files:', meta_files[-3:])

meta = json.load(open(meta_files[-1]))
model_path = meta['model_file']
print('Retrained model path:', model_path)

# 3) Load retrained model and perform a smoke prediction
pipe = load(model_path)
df = pd.read_csv('data/new_train.csv', index_col=0)
sel_meta = json.load(open('models_GBM/scenario_1/GBM_scen1_Tcell_selected_features.json'))
selected_features = sel_meta.get('selected_features', sel_meta)
X = df[selected_features]
print('Predict shape', X.shape)
probs = pipe.predict_proba(X)[:5]
print('Example probs (first 5 rows):', probs)
```

4) Notes & troubleshooting

- The retrain helper expects the following files to exist next to the model prefix:
  - `{prefix}_selected_features.json` — produced by the training script and contains `selected_features` list inside metadata
  - `{prefix}_svm_params.json` or `{prefix}_ens_params.json` — best params metadata

- If parameter keys don't map to the pipeline built in `retrain_helper.py`, `pipe.set_params(**best_params)` may raise; in that case the script prints a warning and fits the pipeline with default parameter values.

- If you want to continue training from a saved estimator object instead of rebuilding the pipeline, modify the helper to `load` the .joblib file and call `.fit()` on it.

- YAML support requires PyYAML (`pip install pyyaml`).

5) Example minimal workflow to add to your notebook

- Run your `Scenario_heldout_final_PRECISE.py` training script to produce models and metadata.
- Prepare a CSV of new training data with the same column names as the original radiomics/immune features (index column required).
- Use `retrain_helper.py` through the CLI or config to retrain.
