"""retrain_helper.py
Small CLI to retrain saved SVM or Ensemble models using saved metadata.

Enhancements in this version:
- Accept a JSON or YAML config file via --config with keys: model_prefix, model_type (optional), train_csv, label_col, out_dir (optional)
- If model_type is omitted, auto-detect by checking for *_svm_params.json or *_ens_params.json next to the prefix
- CLI arguments override config values

Usage (from project root):
python retrain_helper.py --model-prefix "models_GBM/scenario_1/GBM_scen1_Tcell" --model-type svm --train-csv new_train.csv --label-col label
or using config.json/yaml:
python retrain_helper.py --config retrain_config.json

The script expects files with these suffixes next to the prefix:
- _selected_features.json  (contains metadata.selected_features list)
- _svm_params.json or _ens_params.json  (contains metadata.best_params)

It builds pipelines matching the original script, sets the best params, fits on the provided CSV using the selected features, and saves a retrained joblib model and a metadata JSON.
"""

import argparse
import json
import os
from datetime import datetime, timezone
from joblib import dump
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, VotingClassifier

# optional yaml support
try:
    import yaml
    _HAS_YAML = True
except Exception:
    _HAS_YAML = False


def load_json_meta(path):
    with open(path, 'r') as f:
        return json.load(f)


def load_config(path):
    """Load JSON or YAML config file into a dict."""
    if path.lower().endswith(('.yaml', '.yml')):
        if not _HAS_YAML:
            raise RuntimeError('PyYAML is not installed, cannot read YAML config')
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    else:
        with open(path, 'r') as f:
            return json.load(f)


def build_svm_pipeline():
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', SVC(class_weight='balanced', probability=True, random_state=42))
    ])
    return pipe


def build_ensemble_pipeline():
    # base pipe inside voting ensemble should be named and structured like in training script
    base_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', SVC(class_weight='balanced', probability=True, random_state=42))
    ])
    ensemble = VotingClassifier([
        ('svm', base_pipe),
        ('rf', RandomForestClassifier(class_weight='balanced', random_state=42)),
        ('gb', HistGradientBoostingClassifier(random_state=42))
    ], voting='soft', weights=[1, 1, 1])
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('ensemble', ensemble)
    ])
    return pipe


def _auto_detect_model_type(model_prefix):
    """Return 'svm' or 'ens' based on presence of params files next to the prefix.
    If both present, prefer 'svm' and warn."""
    svm_path = model_prefix + '_svm_params.json'
    ens_path = model_prefix + '_ens_params.json'
    svm_exists = os.path.exists(svm_path)
    ens_exists = os.path.exists(ens_path)
    if svm_exists and not ens_exists:
        return 'svm'
    if ens_exists and not svm_exists:
        return 'ens'
    if svm_exists and ens_exists:
        print('Warning: both SVM and Ensemble params found; defaulting to SVM')
        return 'svm'
    # if neither exists, raise
    raise FileNotFoundError(f'Neither {svm_path} nor {ens_path} found for auto-detection')


def retrain(model_prefix, model_type=None, train_csv=None, label_col=None, out_dir=None, overwrite=False):
    """Retrain a saved model using the saved selected-features and best-params metadata.

    model_type can be 'svm' or 'ens' (ensemble). If None, the function will try to auto-detect.
    """
    if model_type is None:
        model_type = _auto_detect_model_type(model_prefix)

    # Resolve file paths
    sel_path = model_prefix + '_selected_features.json'
    if model_type.lower() == 'svm':
        params_path = model_prefix + '_svm_params.json'
    elif model_type.lower() in ('ens', 'ensemble'):
        params_path = model_prefix + '_ens_params.json'
    else:
        raise ValueError('model_type must be "svm" or "ens"')

    if not os.path.exists(sel_path):
        raise FileNotFoundError(f'Selected-features file not found: {sel_path}')
    if not os.path.exists(params_path):
        raise FileNotFoundError(f'Params file not found: {params_path}')
    if train_csv is None or not os.path.exists(train_csv):
        raise FileNotFoundError(f'Train CSV not found: {train_csv}')

    sel_meta = load_json_meta(sel_path)
    # selected features are stored under top-level key 'selected_features' (script writes metadata)
    if isinstance(sel_meta, dict) and 'selected_features' in sel_meta:
        sel_features = sel_meta['selected_features']
    elif isinstance(sel_meta, list):
        sel_features = sel_meta
    else:
        raise ValueError('Unexpected selected features file format')

    params_meta = load_json_meta(params_path)
    # params saved under 'best_params' inside metadata
    if isinstance(params_meta, dict) and 'best_params' in params_meta:
        best_params = params_meta['best_params']
    else:
        # fallback: file may contain bare params
        best_params = params_meta

    # load training data and subset columns
    df = pd.read_csv(train_csv, index_col=0)

    missing = [c for c in sel_features if c not in df.columns]
    if missing:
        raise ValueError(f'The following selected features are missing from training CSV: {missing}')

    X = df[sel_features].values
    y = df[label_col].values

    # Build pipeline and set params
    if model_type.lower() == 'svm':
        pipe = build_svm_pipeline()
    else:
        pipe = build_ensemble_pipeline()

    # set params (keys should match the original training param names)
    try:
        pipe.set_params(**best_params)
    except Exception as e:
        print('Warning: failed to set all params on pipeline:', e)
        # continue anyway

    # Fit
    print(f'Fitting {model_type} on {X.shape[0]} samples with {X.shape[1]} features...')
    pipe.fit(X, y)

    # Save retrained model
    ts = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    if out_dir is None:
        out_dir = os.path.dirname(model_prefix) or '.'
    os.makedirs(out_dir, exist_ok=True)
    model_out_path = os.path.join(out_dir, os.path.basename(model_prefix) + f'_{model_type}_retrained_{ts}.joblib')

    # respect overwrite flag
    if os.path.exists(model_out_path) and not overwrite:
        raise FileExistsError(f'Model output already exists: {model_out_path}. Use overwrite=True to overwrite.')

    dump(pipe, model_out_path)

    # Save retrain metadata
    meta = {
        'retrained_at': datetime.now(timezone.utc).isoformat(),
        'version': ts,
        'model_type': model_type,
        'n_samples': int(X.shape[0]),
        'n_features': int(X.shape[1]),
        'selected_features_file': os.path.abspath(sel_path),
        'params_file': os.path.abspath(params_path),
        'model_file': os.path.abspath(str(model_out_path))
    }
    meta_out = os.path.join(out_dir, os.path.basename(model_prefix) + f'_{model_type}_retrain_meta_{ts}.json')
    with open(meta_out, 'w') as f:
        json.dump(meta, f, indent=2)

    print('Retrained model saved to:', model_out_path)
    print('Retrain metadata saved to:', meta_out)
    return model_out_path, meta_out


def main():
    p = argparse.ArgumentParser(description='Retrain a saved model using saved selected features and best params')
    p.add_argument('--config', required=False, help='Path to JSON or YAML config file with keys: model_prefix, model_type (optional), train_csv, label_col, out_dir')
    p.add_argument('--model-prefix', required=False, help='Path prefix to model files (without suffix). E.g. models_GBM/scenario_1/GBM_scen1_Tcell')
    p.add_argument('--model-type', required=False, choices=['svm', 'ens', 'ensemble'], help='svm or ens (if omitted, auto-detect)')
    p.add_argument('--train-csv', required=False, help='CSV with training data (index column present). Must contain selected features and label column')
    p.add_argument('--label-col', required=False, help='Name of the label column in train CSV')
    p.add_argument('--out-dir', default=None, help='Output directory (defaults to model-prefix directory)')
    p.add_argument('--overwrite', action='store_true', help='Overwrite existing output files')
    args = p.parse_args()

    cfg = {}
    if args.config:
        cfg = load_config(args.config) or {}

    # Merge config and CLI args; CLI takes precedence
    model_prefix = args.model_prefix or cfg.get('model_prefix')
    model_type = args.model_type or cfg.get('model_type')
    train_csv = args.train_csv or cfg.get('train_csv')
    label_col = args.label_col or cfg.get('label_col')
    out_dir = args.out_dir or cfg.get('out_dir')
    overwrite = args.overwrite or cfg.get('overwrite', False)

    if model_prefix is None or train_csv is None or label_col is None:
        raise ValueError('model_prefix, train_csv and label_col must be provided either via --config or CLI args')

    retrain(model_prefix, model_type=model_type, train_csv=train_csv, label_col=label_col, out_dir=out_dir, overwrite=overwrite)


if __name__ == '__main__':
    main()
