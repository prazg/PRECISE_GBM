import logging
import warnings
import pandas as pd
import numpy as np
import json
import time
from tqdm import tqdm
import os
from datetime import datetime as _dt, timezone as _tz


from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, balanced_accuracy_score, matthews_corrcoef
)
from joblib import Memory, dump

# -------------------------
# Logging & warnings
# -------------------------
logging.basicConfig(
    filename='nested_lodo_groups.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=ConvergenceWarning)

# Create directories for saving models if they don't exist
os.makedirs('models_GBM/scenario_1', exist_ok=True)
os.makedirs('models_GBM/scenario_2', exist_ok=True)
os.makedirs('models_GBM/scenario_3', exist_ok=True)
os.makedirs('models_LM22/scenario_1', exist_ok=True)
os.makedirs('models_LM22/scenario_2', exist_ok=True)
os.makedirs('models_LM22/scenario_3', exist_ok=True)

# -------------------------
# Caching for pipelines
# -------------------------
memory = Memory(location='cache_dir', verbose=0)

# Helper: convert numpy scalars/arrays and dicts into JSON-serializable Python types
import numpy as _np

def _convert_obj(o):
    """Recursively convert numpy types/arrays to native Python objects for JSON dumping."""
    # numpy arrays -> lists
    if hasattr(o, 'tolist') and not isinstance(o, (dict, list, str, bytes)):
        try:
            return o.tolist()
        except Exception:
            return str(o)
    # dict -> convert values
    if isinstance(o, dict):
        return {k: _convert_obj(v) for k, v in o.items()}
    # list/tuple -> convert items
    if isinstance(o, (list, tuple)):
        return [_convert_obj(v) for v in o]
    # numpy scalar -> python native
    if isinstance(o, (_np.integer, _np.floating, _np.bool_)):
        return o.item()
    # otherwise return as-is
    return o

def _cv_results_to_serializable(cv_dict):
    """Convert sklearn cv_results_ dict values (numpy arrays) into lists where needed."""
    out = {}
    for k, v in cv_dict.items():
        if hasattr(v, 'tolist'):
            try:
                out[k] = v.tolist()
            except Exception:
                out[k] = str(v)
        else:
            out[k] = _convert_obj(v)
    return out

# -------------------------
# Utility: two-step Lasso selection
# -------------------------
def select_features(X, y, alphas=(0.1, 0.01), cv=5, max_iter=10000, n_jobs=-1, random_state=42):
    for alpha in alphas:
        lasso = LassoCV(
            alphas=[alpha], cv=cv,
            max_iter=max_iter, n_jobs=n_jobs,
            random_state=random_state
        )
        # fit separately so static analyzers can see the correct type
        lasso.fit(X, y)
        # use flatnonzero to get selected indices as a 1-D array
        support = np.flatnonzero(lasso.coef_ != 0)
        if support.size > 0:
            return support
    raise ValueError(f"No features selected at alphas {alphas}")

# -------------------------
# Define two groups of scenarios with actual paths
# Scenario definitions_LM22
scenarios_LM22 = {
    1: {
        'train_radiomics':    r"C:/Users/radiomic_CRCT.csv",
        'train_immune':       r"C:/Users/heldout_I.csv",
        'heldout_radiomics':  r"C:/Users/R_test_IR.csv",
        'heldout_immune':     r"C:/Users/test_II.csv"
    },
    2: {
        'train_radiomics':    r"C:/Users/radiomic_CRCI.csv",
        'train_immune':       r"C:/Users/heldout_T.csv",
        'heldout_radiomics':  r"C:/Users/R_test_TR.csv",
        'heldout_immune':     r"C:/Users/test_TI.csv"
    },
    3: {
        'train_radiomics':    r"C:/Users/radiomic_CRTI.csv",
        'train_immune':       r"C:/Users/heledout_C.csv",
        'heldout_radiomics':  r"C:/Users/R_test_C.csv",
        'heldout_immune':     r"C:/Users/test_CI.csv"
    }
}
# Scenario definitions_GBM
scenarios_GBM = {
    1: {
        'train_radiomics':    r"C:/Users/radiomic_CRCT.csv",
        'train_immune':       r"C:/Users/heldout_I.csv",
        'heldout_radiomics':  r"C:/Users/R_test_IR.csv",
        'heldout_immune':     r"C:/Users/test_II.csv"
    },
    2: {
        'train_radiomics':    r"C:/Users/radiomic_CRCI.csv",
        'train_immune':       r"C:/Users/heldout_T.csv",
        'heldout_radiomics':  r"C:/Users/R_test_TR.csv",
        'heldout_immune':     r"C:/Users/test_TI.csv"
    },
    3: {
        'train_radiomics':    r"C:/Users/radiomic_CRTI.csv",
        'train_immune':       r"C:/Users/heledout_C.csv",
        'heldout_radiomics':  r"C:/Users/R_test_C.csv",
        'heldout_immune':     r"C:/Users/test_CI.csv"
    }
}

signature_groups = {
    'LM22': scenarios_LM22,
    'GBM': scenarios_GBM
}

# -------------------------
# Hyperparameter grids
# -------------------------
param_dist_svm = {
    'clf__C': [1, 10],
    'clf__gamma': [0.01, 0.1],
    'clf__kernel': ['rbf']
}
param_dist_ensemble = {
    'ensemble__svm__classifier__C': [1],
    'ensemble__svm__classifier__kernel': ['rbf'],
    'ensemble__rf__n_estimators': [100, 200],
    'ensemble__rf__max_depth': [None],
    'ensemble__gb__max_iter': [100],
    'ensemble__gb__learning_rate': [0.1]
}

# -------------------------
# Process each signature group
# -------------------------
for sig_name, scenarios in signature_groups.items():
    all_results = {}
    all_features = {}
    all_cv = {}

    for scen_id, paths in scenarios.items():
        logging.info(f"[{sig_name}] Starting {scen_id}")
        t0 = time.time()

        # Load & align training data
        rad_tr = pd.read_csv(paths['train_radiomics'], index_col=0)
        imm_tr = pd.read_csv(paths['train_immune'],    index_col=0)
        df_tr = pd.merge(rad_tr, imm_tr, left_index=True, right_index=True, how='inner')

        # Load & align held-out data
        rad_ho = pd.read_csv(paths['heldout_radiomics'], index_col=0)
        imm_ho = pd.read_csv(paths['heldout_immune'],    index_col=0)
        df_ho = pd.merge(rad_ho, imm_ho, left_index=True, right_index=True, how='inner')

        scen_results = {}
        scen_features = {}
        scen_cv = {}
        inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # Determine immune feature columns (may differ by signature)
        immune_cols = imm_tr.columns.intersection(imm_ho.columns)
        if immune_cols.empty:
            raise ValueError(f"{sig_name}:{scen_id} - no matching immune features between train and held-out")
        logging.info(f"{sig_name}:{scen_id} - {len(immune_cols)} immune features: {immune_cols.tolist()}")

        for col in tqdm(immune_cols, desc=f"{sig_name}:{scen_id}"):
            try:
                # GMM labeling on train
                gmm = GaussianMixture(n_components=2, random_state=42)
                y_tr = gmm.fit_predict(df_tr[[col]].values)
                if len(np.unique(y_tr)) < 2:
                    continue
                y_ho = gmm.predict(df_ho[[col]].values)
                # ensure label 1 = higher mean
                m0, m1 = gmm.means_.flatten()
                if m0 < m1:
                    y_tr = 1 - y_tr; y_ho = 1 - y_ho
                # save gmm model
                gmm_model_path = f'models_{sig_name}/scenario_{scen_id}/{sig_name}_scen{scen_id}_{col}_gmm_model.joblib'
                dump(gmm, gmm_model_path)
                logging.info(f"Saved GMM model to {gmm_model_path}")
                logging.info(f"GMM means for {sig_name}:{scen_id}, col {col}: {gmm.means_.flatten().tolist()}")

                # Feature selection
                X_tr = df_tr.drop(columns=[col]).values
                X_ho = df_ho.drop(columns=[col]).values
                sel = select_features(X_tr, y_tr)
                X_tr_sel, X_ho_sel = X_tr[:, sel], X_ho[:, sel]
                feat_names = df_tr.drop(columns=[col]).columns.tolist()
                sel_names = [feat_names[i] for i in sel]

                # Save selected feature names for this model so retraining can reuse them
                sel_feat_path = f'models_{sig_name}/scenario_{scen_id}/{sig_name}_scen{scen_id}_{col}_selected_features.json'
                os.makedirs(os.path.dirname(sel_feat_path), exist_ok=True)
                ts = _dt.now(_tz.utc).strftime('%Y%m%d_%H%M%S')
                meta = {'saved_at': _dt.now(_tz.utc).isoformat(), 'version': ts, 'selected_features': sel_names}
                with open(sel_feat_path, 'w') as _f:
                    json.dump(meta, _f, indent=2)

                # SVM nested CV
                pipe_svm = Pipeline([
                    ('scaler', StandardScaler()),
                    ('clf', SVC(class_weight='balanced', probability=True, random_state=42))
                ], memory=memory)
                search_svm = RandomizedSearchCV(
                    pipe_svm, param_dist_svm, n_iter=5,
                    cv=inner_cv, scoring='balanced_accuracy',
                    n_jobs=-1, refit=True, error_score='raise'
                )
                search_svm.fit(X_tr_sel, y_tr)
                y_pred_svm = search_svm.predict(X_ho_sel)
                cv_svm = {k: (v.tolist() if hasattr(v, 'tolist') else v)
                          for k, v in search_svm.cv_results_.items()}
                # save SVM model
                svm_model_path = f'models_{sig_name}/scenario_{scen_id}/{sig_name}_scen{scen_id}_{col}_svm_model.joblib'
                dump(search_svm.best_estimator_, svm_model_path)
                logging.info(f"Saved SVM model to {svm_model_path}")
                logging.info(f"SVM best params for {sig_name}:{scen_id}, col {col}: {search_svm.best_params_}")

                # Save SVM best params and cv results for reproducibility / retraining (with metadata)
                svm_params_path = f'models_{sig_name}/scenario_{scen_id}/{sig_name}_scen{scen_id}_{col}_svm_params.json'
                svm_cv_path = f'models_{sig_name}/scenario_{scen_id}/{sig_name}_scen{scen_id}_{col}_svm_cv.json'
                os.makedirs(os.path.dirname(svm_params_path), exist_ok=True)
                svm_meta = {
                    'saved_at': _dt.now(_tz.utc).isoformat(),
                    'version': _dt.now(_tz.utc).strftime('%Y%m%d_%H%M%S'),
                    'best_params': _convert_obj(search_svm.best_params_)
                }
                with open(svm_params_path, 'w') as _f:
                    json.dump(svm_meta, _f, indent=2)
                svm_cv_meta = {
                    'saved_at': _dt.now(_tz.utc).isoformat(),
                    'version': _dt.now(_tz.utc).strftime('%Y%m%d_%H%M%S'),
                    'cv_results': _cv_results_to_serializable(search_svm.cv_results_)
                }
                with open(svm_cv_path, 'w') as _f:
                    json.dump(svm_cv_meta, _f, indent=2)

                # Ensemble nested CV
                base_pipe = Pipeline([
                    ('scaler', StandardScaler()),
                    ('classifier', SVC(class_weight='balanced', probability=True, random_state=42))
                ], memory=memory)
                ensemble = VotingClassifier([
                    ('svm', base_pipe),
                    ('rf', RandomForestClassifier(class_weight='balanced', random_state=42)),
                    ('gb', HistGradientBoostingClassifier(random_state=42))
                ], voting='soft', weights=[1,1,1], n_jobs=-1)
                pipe_ens = Pipeline([
                    ('scaler', StandardScaler()),
                    ('ensemble', ensemble)
                ], memory=memory)
                search_ens = RandomizedSearchCV(
                    pipe_ens, param_dist_ensemble, n_iter=3,
                    cv=inner_cv, scoring='balanced_accuracy',
                    n_jobs=-1, refit=True, error_score='raise'
                )
                search_ens.fit(X_tr_sel, y_tr)
                y_pred_ens = search_ens.predict(X_ho_sel)
                cv_ens = {k: (v.tolist() if hasattr(v, 'tolist') else v)
                          for k, v in search_ens.cv_results_.items()}
                # save Ensemble model
                ens_model_path = f'models_{sig_name}/scenario_{scen_id}/{sig_name}_scen{scen_id}_{col}_ens_model.joblib'
                dump(search_ens.best_estimator_, ens_model_path)
                logging.info(f"Saved Ensemble model to {ens_model_path}")
                logging.info(f"Ensemble best params for {sig_name}:{scen_id}, col {col}: {search_ens.best_params_}")

                # Save Ensemble best params and cv results for reproducibility / retraining (with metadata)
                ens_params_path = f'models_{sig_name}/scenario_{scen_id}/{sig_name}_scen{scen_id}_{col}_ens_params.json'
                ens_cv_path = f'models_{sig_name}/scenario_{scen_id}/{sig_name}_scen{scen_id}_{col}_ens_cv.json'
                os.makedirs(os.path.dirname(ens_params_path), exist_ok=True)
                ens_meta = {
                    'saved_at': _dt.now(_tz.utc).isoformat(),
                    'version': _dt.now(_tz.utc).strftime('%Y%m%d_%H%M%S'),
                    'best_params': _convert_obj(search_ens.best_params_)
                }
                with open(ens_params_path, 'w') as _f:
                    json.dump(ens_meta, _f, indent=2)
                ens_cv_meta = {
                    'saved_at': _dt.now(_tz.utc).isoformat(),
                    'version': _dt.now(_tz.utc).strftime('%Y%m%d_%H%M%S'),
                    'cv_results': _cv_results_to_serializable(search_ens.cv_results_)
                }
                with open(ens_cv_path, 'w') as _f:
                    json.dump(ens_cv_meta, _f, indent=2)

                # Metrics
                def metrics(y_true, y_pred):
                    return {
                        'Accuracy': accuracy_score(y_true, y_pred),
                        'Precision': precision_score(y_true, y_pred, zero_division=1),
                        'Recall': recall_score(y_true, y_pred, zero_division=1),
                        'F1 Score': f1_score(y_true, y_pred, zero_division=1),
                        'Balanced Accuracy': balanced_accuracy_score(y_true, y_pred),
                        'MCC': matthews_corrcoef(y_true, y_pred)
                    }
                scen_results[col] = {'SVM': metrics(y_ho, y_pred_svm), 'Ensemble': metrics(y_ho, y_pred_ens)}
                scen_features[col] = sel_names
                scen_cv[col] = {'svm_cv': cv_svm, 'ensemble_cv': cv_ens}

            except Exception as e:
                logging.error(f"{sig_name}:{scen_id}, col {col}: {e}")
                print(f"[ERROR] {sig_name}:{scen_id}, column {col}: {e}")

        # Save for this scenario
        all_results[scen_id] = scen_results
        all_features[scen_id] = scen_features
        all_cv[scen_id] = scen_cv
        logging.info(f"[{sig_name}] {scen_id} done in {time.time()-t0:.1f}s")

    # Write group-level JSONs
    with open(f'nested_results111_{sig_name}.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    with open(f'nested_features111_{sig_name}.json', 'w') as f:
        json.dump(all_features, f, indent=2)
    with open(f'nested_cv111_{sig_name}.json', 'w') as f:
        json.dump(all_cv, f, indent=2)
    print(f"✅ {sig_name} group complete: scenarios={list(all_results.keys())}")

print("All signature groups processed.")

