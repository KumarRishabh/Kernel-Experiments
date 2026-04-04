#!/usr/bin/env python
"""Test script for Cauchy kernel experiments"""

import numpy as np
from tqdm.auto import tqdm
from sklearn.svm import SVC
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, ShuffleSplit
from pyriemann.estimation import Covariances
from moabb.datasets import BNCI2014001
from moabb.paradigms import MotorImagery
import mne
import sympy as sp

# ── Butterworth paradigm ───────────────────────────────────────────────────────

class ButterworthMotorImagery(MotorImagery):
    def preprocess_raw(self, raw, dataset, fitting_config=None):
        iir_params = dict(order=5, ftype='butter')
        raw.filter(l_freq=self.fmin, h_freq=self.fmax,
                   method='iir', iir_params=iir_params, verbose=False)
        return super().preprocess_raw(raw, dataset, fitting_config)

# ── MicrovoltScaler ────────────────────────────────────────────────────────────

class MicrovoltScaler(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X): return X * 1e6

# ── Symbolic derivative cache ──────────────────────────────────────────────────

_deriv_cache = {}

def _get_derivatives(N: int, kappa: float):
    """
    Symbolically compute and cache the first N derivatives of
        gamma_tilde(s) = (kappa^2 + s^2)^{-1}
    returning a list of N numpy-callable functions.
    """
    key = (N, kappa)
    if key in _deriv_cache:
        return _deriv_cache[key]

    s = sp.Symbol('s', real=True)
    expr = 1 / (sp.Rational(kappa).limit_denominator(1000)**2 + s**2)
    fns = []
    for _ in range(N):
        fns.append(sp.lambdify(s, sp.simplify(expr), modules='numpy'))
        expr = sp.diff(expr, s)

    _deriv_cache[key] = fns
    return fns


# ── Core helpers ───────────────────────────────────────────────────────────────

def _mat_sqrt_inv(X: np.ndarray) -> np.ndarray:
    """Compute X^{-1/2} via eigendecomposition."""
    L, V = np.linalg.eigh(X)
    return V * (1.0 / np.sqrt(np.clip(L, 1e-12, None)))[None, :] @ V.T

def _cauchy_kernel_single(X: np.ndarray, Y: np.ndarray, kappa: float) -> float:
    x = _mat_sqrt_inv(X) @ Y @ _mat_sqrt_inv(X)
    N = x.shape[0]

    rho = np.sort(np.clip(np.linalg.eigvalsh(x), 1e-12, None))
    s = np.log(rho)                                       # s_i = log(rho_i)

    # ── Vandermonde in s-space: V(s) = prod_{l > k} (s_l - s_k) ──────────────
    log_abs_V = 0.0
    for k in range(N):
        for l in range(k + 1, N):
            diff = s[l] - s[k]
            if abs(diff) < 1e-10:
                diff = 1e-10
            log_abs_V += np.log(abs(diff))

    # ── det(x)^{(N-1)/2} = exp((N-1)/2 * sum(s)) ─────────────────────────────
    log_det_power = ((N - 1) / 2.0) * np.sum(s)

    # ── Derivative matrix: M[k, l] = -gamma_tilde^{(k)}(s_l) ─────────────────
    derivs = _get_derivatives(N, kappa)
    M = np.array([[-float(derivs[k](s[l])) for l in range(N)]
                  for k in range(N)])

    # Row-normalise before slogdet to handle varying derivative magnitudes
    row_max = np.max(np.abs(M), axis=1)
    row_max = np.where(row_max == 0, 1.0, row_max)
    log_row_scales = np.log(row_max)
    M_normalised = M / row_max[:, None]

    sign_M, log_abs_det_M_norm = np.linalg.slogdet(M_normalised)

    if sign_M == 0:
        return 0.0

    log_abs_det_M = log_abs_det_M_norm + np.sum(log_row_scales)

    # ── Final assembly, entirely in log-space ──────────────────────────────────
    log_val = log_det_power + log_abs_det_M - log_abs_V

    if not np.isfinite(log_val):
        return 0.0

    return float(np.exp(np.clip(log_val, -700, 700)))


# ── Scikit-learn transformer ───────────────────────────────────────────────────

class CauchyGramMatrix(BaseEstimator, TransformerMixin):
    """
    Scikit-learn transformer that builds the Gram matrix for the strictly
    positive-definite Cauchy kernel on the SPD manifold (equation 25).

        K(X, Y) = f(X^{-1/2} Y X^{-1/2})

    where f is derived via the Helgason-Fourier (spherical) transform with
    spectral density gamma(t) = (kappa/2) * exp(-kappa|t|).

    This kernel is guaranteed PD by Godement's theorem, unlike the naive
    geodesic substitution k(X,Y) = (kappa^2 + delta^2)^{-l} which fails
    conditional negative definiteness for matrix dimension N >= 2.

    Parameters
    ----------
    kappa : float, default=1.0
        Scale parameter. Must be > 0. Larger kappa = broader kernel.
    """

    def __init__(self, kappa: float = 1.0):
        if kappa <= 0:
            raise ValueError(f"kappa must be > 0, got {kappa}")
        self.kappa = kappa
        self.X_train_ = None

    def fit(self, X: np.ndarray, y=None):
        self.X_train_ = X
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        N = len(X)
        M = len(self.X_train_)
        K = np.zeros((N, M))

        for i in range(N):
            for j in range(M):
                K[i, j] = _cauchy_kernel_single(X[i], self.X_train_[j], self.kappa)

        return K


# ── Main experiment ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading dataset...")
    dataset  = BNCI2014001()
    paradigm = ButterworthMotorImagery(fmin=8.0, fmax=30.0, tmin=0.5, tmax=2.5)

    cv_strategy = ShuffleSplit(n_splits=30, test_size=0.2, random_state=42)

    param_grid_cauchy = {'svc__C': [0.1, 1, 10, 100, 1000],
                         'cauchygrammatrix__kappa': [0.1, 0.5, 1.0, 2.0, 5.0]}

    pipeline_cauchy = make_pipeline(Covariances(estimator='scm'),
                                    CauchyGramMatrix(kappa=1.0),
                                    SVC(kernel='precomputed'))

    grid_cauchy = GridSearchCV(pipeline_cauchy, param_grid_cauchy, cv=cv_strategy, n_jobs=-1)

    results_cauchy = []
    subjects = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    for subject in tqdm(subjects, desc='Processing subjects'):
        X, y, metadata = paradigm.get_data(dataset, subjects=[subject])

        train_idx = metadata['session'] == '0train'
        test_idx  = metadata['session'] == '1test'
        X_train, y_train = X[train_idx], y[train_idx]
        X_test,  y_test  = X[test_idx],  y[test_idx]

        grid_cauchy.fit(X_train, y_train)
        results_cauchy.append(grid_cauchy.score(X_test, y_test))

        print(f"Sub {subject:02d}  "
              f"Cauchy: {results_cauchy[-1]*100:5.2f}%  "
              f"[best kappa={grid_cauchy.best_params_['cauchygrammatrix__kappa']}  "
              f"C={grid_cauchy.best_params_['svc__C']}]")

    print(f"\n{'─'*60}")
    print(f"Mean  Cauchy:     {np.mean(results_cauchy)*100:.2f}%  ± {np.std(results_cauchy)*100:.2f}%")