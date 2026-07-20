"""
Leakage_Free_PRECISE_v4.py

PRECISE-GBM v4 (nested-CV corrected; picklable selector):
- Leak-free radiomics-only predictors
- Per-label radiomics-immune alignment (no X/y length mismatch)
- GMM-derived binary immune labels (fit on TRAIN, applied to held-out)
- FEATURE SELECTION INSIDE THE CV PIPELINE (truly nested inner CV)
    LASSO selection is a Pipeline step (LassoSelector), refit within every
    inner-CV fold and every calibration fold.
- LassoSelector lives in precise_selectors.py (importable) so joblib can
    pickle it during parallel CV AND reload saved models in a fresh session.
- Nested hyperparameter tuning (SVM & Ensemble)
- Post-tuning calibration with CalibratedClassifierCV on TRAIN ONLY
    SVM: probability=False, so the sigmoid is fit ONCE on the raw
    decision_function (no double-Platt). Ensemble: its SVM keeps
    probability=True because soft voting needs predict_proba; the outer
    sigmoid recalibrates the combined soft-vote score.
- Calibrated models evaluated on held-out Ivy/TCGA/CPTAC

Data-path assumptions (confirmed by user; the disjoint one is asserted below):
- ComBat harmonisation was fit on TRAINING data only; held-out never harmonised.
- Train and held-out patients are disjoint.

Requires precise_selectors.py on the PYTHONPATH (same folder is easiest).

Outputs per signature group (LM22, GBM):
- models_v4_{sig}/scenario_{1,2,3}/*_gmm_model.joblib
- models_v4_{sig}/scenario_{1,2,3}/*_svm_model.joblib        (uncalibrated)
- models_v4_{sig}/scenario_{1,2,3}/*_ens_model.joblib        (uncalibrated)
- models_v4_{sig}/scenario_{1,2,3}/*_svm_cal_model.joblib    (calibrated)
- models_v4_{sig}/scenario_{1,2,3}/*_ens_cal_model.joblib    (calibrated)
- nested_results_v4_{sig}.json  (held-out metrics for calibrated models)
- nested_features_v4_{sig}.json (selected radiomics per label/scenario/model)
- nested_cv_v4_{sig}.json       (inner CV results for uncalibrated models)
"""

import logging
import warnings
import os
import time
import json

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, balanced_accuracy_score, matthews_corrcoef,
    brier_score_loss
)
from joblib import dump

# LassoSelector must be importable (NOT defined here in __main__) so joblib can
# pickle it for parallel CV and reload saved models later.
from precise_selectors import LassoSelector

# -------------------------
# Logging & warnings
# -------------------------
logging.basicConfig(
    filename='Leakage_Free_PRECISE_v4.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=ConvergenceWarning)

# NOTE: joblib.Memory caching was removed. It hashes each pipeline step with
# standard pickle, which cannot resolve a custom transformer and raised the
# PicklingError. The parallel search path uses cloudpickle and is fine.

# -------------------------
# JSON-safe converters
# -------------------------
def _convert_obj(o):
    if hasattr(o, 'tolist') and not isinstance(o, (dict, list, str, bytes)):
        try:
            return o.tolist()
        except Exception:
            return str(o)
    if isinstance(o, dict):
        return {k: _convert_obj(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_convert_obj(v) for v in o]
    if isinstance(o, (np.integer, np.floating, np.bool_)):
        return o.item()
    return o

def _cv_results_to_serializable(cv_dict):
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
# Scenario definitions
# -------------------------
scenarios_LM22 = {
    1: {
        'train_radiomics':    r"C:/Users/pg22/Downloads/PRECISE-GBM/LOOCV_withoutHarm/Radiomics/neuro_combat_radiomic_CGGA_Rem_CP_TC.csv",
        'train_immune':       r"C:/Users/pg22/Downloads/PRECISE-GBM/LOOCV_withoutHarm/Genome/Heldout/heldout_Ivy/Cbx_LOOCV_heldout_Ivy_Lm22/CIBERSORTx_Job49_Results.csv",
        'heldout_radiomics':  r"C:/Users/pg22/Downloads/PRECISE-GBM/LOOCV_withoutHarm/Radiomics/Radiomics_LOOCV_test_Ivy.csv",
        'heldout_immune':     r"C:/Users/pg22/Downloads/PRECISE-GBM/LOOCV_withoutHarm/Genome/Testing/IvyGAP/Test_Ivy_LM22/CIBERSORTx_Job55_Results.csv"
    },
    2: {
        'train_radiomics':    r"C:/Users/pg22/Downloads/PRECISE-GBM/LOOCV_withoutHarm/Radiomics/neuro_combat_radiomic_CGGA_Rem_CP_ivy.csv",
        'train_immune':       r"C:/Users/pg22/Downloads/PRECISE-GBM/LOOCV_withoutHarm/Genome/Heldout/heldout_TCGA/Cbx_heldoutTCGA_Lm22/CIBERSORTx_Job47_Results.csv",
        'heldout_radiomics':  r"C:/Users/pg22/Downloads/PRECISE-GBM/LOOCV_withoutHarm/Radiomics/Radiomics_LOOCV_test_TCGA.csv",
        'heldout_immune':     r"C:/Users/pg22/Downloads/PRECISE-GBM/LOOCV_withoutHarm/Genome/Testing/TCGA/Cbx_TCGA_Test_LM22/CIBERSORTx_Job53_Results.csv"
    },
    3: {
        'train_radiomics':    r"C:/Users/pg22/Downloads/PRECISE-GBM/LOOCV_withoutHarm/Radiomics/neuro_combat_radiomic_CGGA_Rem_TC_ivy.csv",
        'train_immune':       r"C:/Users/pg22/Downloads/PRECISE-GBM/LOOCV_withoutHarm/Genome/Heldout/heldout_CPTAC/CBx_LOOCV_heldout_CPTAC_LM22/CIBERSORTx_Job51_Results.csv",
        'heldout_radiomics':  r"C:/Users/pg22/Downloads/PRECISE-GBM/LOOCV_withoutHarm/Radiomics/Radiomics_LOOCV_test_CPTAC.csv",
        'heldout_immune':     r"C:/Users/pg22/Downloads/PRECISE-GBM/LOOCV_withoutHarm/Genome/Testing/CPTAC/Test_CPTAC_LM22/CIBERSORTx_Job57_Results.csv"
    }
}

scenarios_GBM = {
    1: {
        'train_radiomics':    r"C:/Users/pg22/Downloads/PRECISE-GBM/LOOCV_withoutHarm/Radiomics/neuro_combat_radiomic_CGGA_Rem_CP_TC.csv",
        'train_immune':       r"C:/Users/pg22/Downloads/PRECISE-GBM/LOOCV_withoutHarm/Genome/Heldout/heldout_Ivy/Cbx_LOOCV_heldout_Ivy_GBM/CIBERSORTx_Job50_Results.csv",
        'heldout_radiomics':  r"C:/Users/pg22/Downloads/PRECISE-GBM/LOOCV_withoutHarm/Radiomics/Radiomics_LOOCV_test_Ivy.csv",
        'heldout_immune':     r"C:/Users/pg22/Downloads/PRECISE-GBM/LOOCV_withoutHarm/Genome/Testing/IvyGAP/Test_Ivy_GBM/CIBERSORTx_Job56_Results.csv"
    },
    2: {
        'train_radiomics':    r"C:/Users/pg22/Downloads/PRECISE-GBM/LOOCV_withoutHarm/Radiomics/neuro_combat_radiomic_CGGA_Rem_CP_ivy.csv",
        'train_immune':       r"C:/Users/pg22/Downloads/PRECISE-GBM/LOOCV_withoutHarm/Genome/Heldout/heldout_TCGA/Cbx_LOOCV_TCGA_heldout_GBM/CIBERSORTx_Job48_Results.csv",
        'heldout_radiomics':  r"C:/Users/pg22/Downloads/PRECISE-GBM/LOOCV_withoutHarm/Radiomics/Radiomics_LOOCV_test_TCGA.csv",
        'heldout_immune':     r"C:/Users/pg22/Downloads/PRECISE-GBM/LOOCV_withoutHarm/Genome/Testing/TCGA/TCGA_test_GBM/CIBERSORTx_Job54_Results.csv"
    },
    3: {
        'train_radiomics':    r"C:/Users/pg22/Downloads/PRECISE-GBM/LOOCV_withoutHarm/Radiomics/neuro_combat_radiomic_CGGA_Rem_TC_ivy.csv",
        'train_immune':       r"C:/Users/pg22/Downloads/PRECISE-GBM/LOOCV_withoutHarm/Genome/Heldout/heldout_CPTAC/Cbx_LOOCV_heldout_CPTAC_GBM/CIBERSORTx_Job52_Results.csv",
        'heldout_radiomics':  r"C:/Users/pg22/Downloads/PRECISE-GBM/LOOCV_withoutHarm/Radiomics/Radiomics_LOOCV_test_CPTAC.csv",
        'heldout_immune':     r"C:/Users/pg22/Downloads/PRECISE-GBM/LOOCV_withoutHarm/Genome/Testing/CPTAC/Test_CPTAC_GBM/CIBERSORTx_Job58_Results.csv"
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
# Main v3 loop
# -------------------------
for sig_name, scenarios in signature_groups.items():
    all_results = {}
    all_features = {}
    all_cv = {}

    for scen_id in [1, 2, 3]:
        os.makedirs(f"models_v4_{sig_name}/scenario_{scen_id}", exist_ok=True)

    for scen_id, paths in scenarios.items():
        logging.info(f"[v4:{sig_name}] Starting scenario {scen_id}")
        t0 = time.time()

        # Load radiomics + immune
        rad_tr_full = pd.read_csv(paths['train_radiomics'], index_col=0)
        imm_tr_full = pd.read_csv(paths['train_immune'],    index_col=0)
        rad_ho_full = pd.read_csv(paths['heldout_radiomics'], index_col=0)
        imm_ho_full = pd.read_csv(paths['heldout_immune'],    index_col=0)

        # index align WITHIN train and WITHIN held-out
        common_tr = rad_tr_full.index.intersection(imm_tr_full.index)
        rad_tr_full = rad_tr_full.loc[common_tr].sort_index()
        imm_tr_full = imm_tr_full.loc[common_tr].sort_index()

        common_ho = rad_ho_full.index.intersection(imm_ho_full.index)
        rad_ho_full = rad_ho_full.loc[common_ho].sort_index()
        imm_ho_full = imm_ho_full.loc[common_ho].sort_index()

        # --- guard: train and held-out patients must be disjoint ---
        overlap = set(rad_tr_full.index).intersection(set(rad_ho_full.index))
        if overlap:
            raise ValueError(
                f"[v4:{sig_name}:{scen_id}] Train/held-out patient overlap "
                f"({len(overlap)} IDs, e.g. {list(overlap)[:5]}). "
                f"Held-out must be disjoint from train."
            )

        # --- guard: radiomics feature columns must match train vs held-out ---
        if list(rad_tr_full.columns) != list(rad_ho_full.columns):
            common_rad = rad_tr_full.columns.intersection(rad_ho_full.columns)
            if common_rad.empty:
                raise ValueError(f"[v4:{sig_name}:{scen_id}] No shared radiomics columns train vs held-out.")
            rad_tr_full = rad_tr_full[common_rad]
            rad_ho_full = rad_ho_full[common_rad]
            logging.warning(f"[v4:{sig_name}:{scen_id}] Restricted to {len(common_rad)} shared radiomics columns.")

        scen_results = {}
        scen_features = {}
        scen_cv = {}

        inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        immune_cols = imm_tr_full.columns.intersection(imm_ho_full.columns)
        if immune_cols.empty:
            raise ValueError(f"[v4:{sig_name}:{scen_id}] No matching immune features between train and held-out")

        logging.info(f"[v4:{sig_name}:{scen_id}] {len(immune_cols)} immune features: {immune_cols.tolist()}")

        rad_cols = rad_tr_full.columns.tolist()  # radiomics-only feature space

        for col in tqdm(immune_cols, desc=f"{sig_name} v4:{scen_id}"):
            try:
                # --- per-label join to avoid X/y mismatches ---
                df_tr_label = rad_tr_full.join(imm_tr_full[[col]], how="inner").dropna(subset=[col])
                df_ho_label = rad_ho_full.join(imm_ho_full[[col]], how="inner").dropna(subset=[col])

                if df_tr_label.shape[0] == 0 or df_ho_label.shape[0] == 0:
                    logging.warning(f"[v4:{sig_name}:{scen_id}] {col}: no samples after join/dropna; skipping.")
                    continue

                # Features = radiomics ONLY; label source = this immune column ONLY
                X_tr = df_tr_label[rad_cols].values
                y_tr_vals = df_tr_label[col].values.reshape(-1, 1)
                X_ho = df_ho_label[rad_cols].values
                y_ho_vals = df_ho_label[col].values.reshape(-1, 1)

                # GMM labels: fit on TRAIN immune, apply to held-out (no held-out leakage)
                gmm = GaussianMixture(n_components=2, random_state=42)
                y_tr = gmm.fit_predict(y_tr_vals)
                if len(np.unique(y_tr)) < 2:
                    logging.warning(f"[v4:{sig_name}:{scen_id}] {col}: GMM yielded one class only; skipping.")
                    continue
                y_ho = gmm.predict(y_ho_vals)

                m0, m1 = gmm.means_.flatten()
                if m0 < m1:  # class 1 = higher-mean component
                    y_tr = 1 - y_tr
                    y_ho = 1 - y_ho

                gmm_model_path = f"models_v4_{sig_name}/scenario_{scen_id}/{sig_name}_v4_scen{scen_id}_{col}_gmm_model.joblib"
                dump(gmm, gmm_model_path)
                logging.info(f"[v4:{sig_name}:{scen_id}] Saved GMM model to {gmm_model_path}")

                # =========================================================
                # SVM: nested CV. Selection is a PIPELINE STEP -> refit inside
                # every fold, so the inner CV is truly nested. X is the FULL
                # radiomics matrix; scale -> select -> clf.
                # probability=False: the single calibrating sigmoid is applied
                # by CalibratedClassifierCV to the raw decision_function below,
                # so there is NO sigmoid-on-sigmoid. (Tuning uses predict.)
                # =========================================================
                pipe_svm = Pipeline([
                    ('scaler', StandardScaler()),
                    ('select', LassoSelector()),
                    ('clf', SVC(class_weight='balanced', probability=False, random_state=42))
                ])

                search_svm = RandomizedSearchCV(
                    pipe_svm, param_dist_svm, n_iter=5,
                    cv=inner_cv, scoring='balanced_accuracy',
                    n_jobs=-1, refit=True, error_score='raise'
                )
                search_svm.fit(X_tr, y_tr)
                cv_svm = _cv_results_to_serializable(search_svm.cv_results_)

                svm_model_path = f"models_v4_{sig_name}/scenario_{scen_id}/{sig_name}_v4_scen{scen_id}_{col}_svm_model.joblib"
                dump(search_svm.best_estimator_, svm_model_path)
                logging.info(f"[v4:{sig_name}:{scen_id}] Saved SVM model to {svm_model_path}")

                # =========================================================
                # Ensemble: nested CV (selection inside pipeline as above)
                # NOTE: soft voting needs predict_proba from every member, so this
                # SVM keeps probability=True. The outer CalibratedClassifierCV then
                # recalibrates the *combined* soft-vote score (one sigmoid on the
                # averaged output) -- not a double sigmoid on a single SVM margin.
                # =========================================================
                base_pipe = Pipeline([
                    ('scaler', StandardScaler()),
                    ('classifier', SVC(class_weight='balanced', probability=True, random_state=42))
                ])

                ensemble = VotingClassifier([
                    ('svm', base_pipe),
                    ('rf', RandomForestClassifier(class_weight='balanced', random_state=42)),
                    ('gb', HistGradientBoostingClassifier(random_state=42))
                ], voting='soft', weights=[1, 1, 1], n_jobs=-1)

                pipe_ens = Pipeline([
                    ('scaler', StandardScaler()),
                    ('select', LassoSelector()),
                    ('ensemble', ensemble)
                ])

                search_ens = RandomizedSearchCV(
                    pipe_ens, param_dist_ensemble, n_iter=3,
                    cv=inner_cv, scoring='balanced_accuracy',
                    n_jobs=-1, refit=True, error_score='raise'
                )
                search_ens.fit(X_tr, y_tr)
                cv_ens = _cv_results_to_serializable(search_ens.cv_results_)

                ens_model_path = f"models_v4_{sig_name}/scenario_{scen_id}/{sig_name}_v4_scen{scen_id}_{col}_ens_model.joblib"
                dump(search_ens.best_estimator_, ens_model_path)
                logging.info(f"[v4:{sig_name}:{scen_id}] Saved Ensemble model to {ens_model_path}")

                scen_cv[col] = {'svm_cv': cv_svm, 'ensemble_cv': cv_ens}

                # --- record features actually chosen by the final refit ---
                svm_sel = search_svm.best_estimator_.named_steps['select']
                ens_sel = search_ens.best_estimator_.named_steps['select']
                scen_features[col] = {
                    'svm':      [rad_cols[i] for i in svm_sel.get_support(indices=True)],
                    'ensemble': [rad_cols[i] for i in ens_sel.get_support(indices=True)]
                }
                if getattr(svm_sel, 'all_features_fallback_', False) or getattr(ens_sel, 'all_features_fallback_', False):
                    logging.warning(f"[v4:{sig_name}:{scen_id}] {col}: LASSO selected 0 features on final refit; kept all (check alphas).")

                # =========================================================
                # Calibration on TRAIN ONLY (internal CV -> leak-free).
                # cv=5 refits clones per calibration fold, so selection is also
                # refit inside calibration folds. Held-out never seen.
                # SVM has probability=False -> this sigmoid is fit on the raw
                # decision_function (single Platt, no double sigmoid).
                # =========================================================
                svm_cal = CalibratedClassifierCV(search_svm.best_estimator_, cv=5, method="sigmoid")
                svm_cal.fit(X_tr, y_tr)
                dump(svm_cal, f"models_v4_{sig_name}/scenario_{scen_id}/{sig_name}_v4_scen{scen_id}_{col}_svm_cal_model.joblib")

                ens_cal = CalibratedClassifierCV(search_ens.best_estimator_, cv=5, method="sigmoid")
                ens_cal.fit(X_tr, y_tr)
                dump(ens_cal, f"models_v4_{sig_name}/scenario_{scen_id}/{sig_name}_v4_scen{scen_id}_{col}_ens_cal_model.joblib")

                # --- Held-out predictions (calibrated); full radiomics in ---
                prob_svm = svm_cal.predict_proba(X_ho)[:, 1]
                y_pred_svm = (prob_svm >= 0.5).astype(int)

                prob_ens = ens_cal.predict_proba(X_ho)[:, 1]
                y_pred_ens = (prob_ens >= 0.5).astype(int)

                # --- Metrics ---
                def metrics(y_true, y_pred, y_prob):
                    return {
                        'Accuracy': accuracy_score(y_true, y_pred),
                        'Precision': precision_score(y_true, y_pred, zero_division=1),
                        'Recall': recall_score(y_true, y_pred, zero_division=1),
                        'F1 Score': f1_score(y_true, y_pred, zero_division=1),
                        'Balanced Accuracy': balanced_accuracy_score(y_true, y_pred),
                        'MCC': matthews_corrcoef(y_true, y_pred),
                        'Brier': brier_score_loss(y_true, y_prob),
                        'N': int(len(y_true))
                    }

                scen_results[col] = {
                    'SVM': metrics(y_ho, y_pred_svm, prob_svm),
                    'Ensemble': metrics(y_ho, y_pred_ens, prob_ens)
                }

            except Exception as e:
                logging.error(f"[v4:{sig_name}:{scen_id}] ERROR for column {col}: {e}")
                print(f"[ERROR v4] {sig_name}:{scen_id}, column {col}: {e}")

        all_results[scen_id] = scen_results
        all_features[scen_id] = scen_features
        all_cv[scen_id] = scen_cv

        dt = time.time() - t0
        logging.info(f"[v4:{sig_name}] scenario {scen_id} done in {dt/60:.1f} min")
        print(f"{sig_name} v4:{scen_id} complete in {dt/60:.1f} min")

    with open(f"nested_results_v4_{sig_name}.json", 'w') as f:
        json.dump(_convert_obj(all_results), f, indent=2)
    with open(f"nested_features_v4_{sig_name}.json", 'w') as f:
        json.dump(_convert_obj(all_features), f, indent=2)
    with open(f"nested_cv_v4_{sig_name}.json", 'w') as f:
        json.dump(_convert_obj(all_cv), f, indent=2)

    print(f"[OK] v4 {sig_name} group complete: scenarios={list(all_results.keys())}")

print("All v4 signature groups processed.")
