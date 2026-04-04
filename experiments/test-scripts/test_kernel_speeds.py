import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from scipy.linalg import eigh as scipy_eigh


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _log_eigenvalues(X, Y):
    rho = scipy_eigh(Y, X, eigvals_only=True)
    return np.log(np.clip(rho, 1e-12, None))


def _cauchy_kernel_single(X, Y, kappa=1.0):
    s  = _log_eigenvalues(X, Y)
    N  = len(s)
    log_term1 = np.sum(2.0 * np.log(kappa) - np.log(kappa**2 + s**2))
    i_idx, j_idx = np.triu_indices(N, k=1)
    d = s[j_idx] - s[i_idx]
    d = d[np.abs(d) > 1e-10]
    half_d   = d / 2.0
    log_sinh = np.where(half_d > 20.0, half_d - np.log(2.0), np.log(np.sinh(half_d)))
    log_term2 = np.sum(np.log(d) - np.log(2.0) - log_sinh)
    return float(np.exp(np.clip(log_term1 + log_term2, -700, 700)))


# ─────────────────────────────────────────────
# Transformer
# ─────────────────────────────────────────────

class CauchyKernelTransformer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn transformer that maps a set of SPD matrices to a Gram-matrix
    column using the Cauchy kernel on the SPD manifold.

    Parameters
    ----------
    kappa : float, default=1.0
        Bandwidth parameter of the Cauchy kernel.  Larger values make the
        kernel broader (less discriminative); smaller values make it sharper.

    Attributes
    ----------
    X_train_ : ndarray of shape (m, p, p)
        Training SPD matrices stored during ``fit``.

    Notes
    -----
    Input arrays must contain symmetric positive definite matrices.
    A small eigenvalue clip (1e-12) is applied internally for robustness.

    Usage
    -----
    Drop-in replacement for ``SteinKernelTransformer`` inside a
    ``Pipeline`` + ``GridSearchCV`` workflow::

        pipe = Pipeline([
            ("kernel", CauchyKernelTransformer()),
            ("svm",    SVC(kernel="precomputed")),
        ])
        grid = GridSearchCV(pipe, {"kernel__kappa": [0.5, 1.0, 2.0]})
        grid.fit(X_train, y_train)
    """

    def __init__(self, kappa: float = 1.0):
        self.kappa = kappa
        self.X_train_ = None

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y=None):
        """Store training SPD matrices.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_channels, n_channels)
        y : ignored
        """
        self.X_train_ = X
        return self

    # ------------------------------------------------------------------
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Compute the Cauchy kernel matrix K[i, j] = k(X[i], X_train[j]).

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_channels, n_channels)

        Returns
        -------
        K : ndarray of shape (n_samples, n_train_samples)
        """
        if self.X_train_ is None:
            raise ValueError("Call fit() before transform().")

        N = X.shape[0]
        M = self.X_train_.shape[0]
        K = np.zeros((N, M))

        for i in range(N):
            for j in range(M):
                K[i, j] = _cauchy_kernel_single(X[i], self.X_train_[j], kappa=self.kappa)

        return K


class SteinKernelTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, beta=0.5, normalized=True):
        self.beta = beta
        self.normalized = normalized
        self.X_train_ = None

    def fit(self, X, y=None):
        # Simply store training covariance matrices
        self.X_train_ = X
        return self

    def transform(self, X):
        # X: (n_samples, n_channels, n_channels)
        # self.X_train_: (m_samples, n_channels, n_channels)
        N = X.shape[0]
        M = self.X_train_.shape[0]
        K = np.zeros((N, M))
        
        # Rigorous computation of log-determinants for Stein distance
        log_det_X = np.array([np.linalg.slogdet(x)[1] for x in X])
        log_det_train = np.array([np.linalg.slogdet(x)[1] for x in self.X_train_])

        for i in range(N):
            for j in range(M):
                # Midpoint matrix for Jensen-Bregman Log-Det Divergence
                mean_mat = 0.5 * (X[i] + self.X_train_[j])
                _, log_det_mean = np.linalg.slogdet(mean_mat)
                
                if self.normalized:
                    # Normalized Stein Kernel (S-Divergence based)
                    log_K = -self.beta * (log_det_mean - 0.5 * log_det_X[i] - 0.5 * log_det_train[j])
                else:
                    log_K = -self.beta * log_det_mean
                
                K[i, j] = np.exp(log_K)
        return K
