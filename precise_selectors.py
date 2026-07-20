"""
precise_selectors.py

Custom sklearn transformers for the PRECISE-GBM pipeline.

Kept in a SEPARATE, IMPORTABLE module (NOT in the training script's __main__)
so that:
  1. joblib / pickle can resolve the class by reference during parallel CV
     (a class defined in __main__ raises
     "Can't pickle <class '__main__.LassoSelector'>"), and
  2. models saved with joblib.dump() can be RELOADED in a fresh session or in
     retrain_helper.py, because the reference resolves to
     precise_selectors.LassoSelector rather than __main__.LassoSelector.

Keep this file on the PYTHONPATH (simplest: same folder as the training and
retrain scripts).
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import Lasso


class LassoSelector(BaseEstimator, TransformerMixin):
    """LASSO feature selection as a pipeline step (refit inside every CV fold).

    Two-step alpha fallback: try each alpha in order, keep the first that
    yields a non-empty support. If nothing survives any alpha for a fold,
    keep ALL features and record it via `all_features_fallback_` so frequent
    fallback (alphas too aggressive for the fold size) is auditable.
    """

    def __init__(self, alphas=(0.1, 0.01), max_iter=10000, random_state=42):
        self.alphas = alphas
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y).ravel()
        self.n_features_in_ = X.shape[1]
        support = None
        for alpha in self.alphas:
            lasso = Lasso(alpha=alpha, max_iter=self.max_iter, random_state=self.random_state, tol=1e-4)
            lasso.fit(X, y)
            idx = np.flatnonzero(lasso.coef_ != 0)
            if idx.size > 0:
                support = idx
                break
        if support is None:
            support = np.arange(self.n_features_in_)
            self.all_features_fallback_ = True
        else:
            self.all_features_fallback_ = False
        self.support_ = support
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.support_]

    def get_support(self, indices=False):
        if indices:
            return self.support_
        mask = np.zeros(self.n_features_in_, dtype=bool)
        mask[self.support_] = True
        return mask
