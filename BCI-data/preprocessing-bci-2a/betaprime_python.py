import numpy as np
import tqdm
import betaprime_cpp as betaprime
from scipy.special import gamma
import math


def cone_gamma(N, z):
    log_prod = (N * (N - 1) / 2) * np.log(2 * np.pi)
    for k in range(N):
        log_prod += math.lgamma(z - k + 1)
    return np.exp(log_prod)

def logcone_gamma_fast(N, z):
    """
    Fast computation of the cone gamma function.
    This is a more efficient version of the cone_gamma function.
    """
    log_prod = (N * (N - 1) / 2) * np.log(2 * np.pi)
    log_prod += np.sum(np.log(np.arange(z, z - N + 1, -1)))
    return log_prod

class BetaprimeKernel:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.gamma_const = gamma(alpha + 1)

    def __call__(self, X: np.ndarray, Y: np.ndarray) -> float:
        det_X = np.linalg.det(X)
        det_Y = np.linalg.det(Y)
        det_sum = np.linalg.det(X + Y)
        eps = 1e-12
        kernel_val = self.gamma_const + ((det_X * det_Y) / (det_sum**2 + eps)) ** self.alpha
        return kernel_val

    def pairwise(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Compute Beta Prime kernel matrix with minimal extra memory
        by trading off for more computation (double loop).

        This approach avoids large (n, m, d, d) allocations
        and instead uses a double loop to compute the kernel matrix
        for each pair of samples in X and Y. For large dimensions, this is more memory efficient but slower due 
        to the python loop overhead.
        """
        n, d, _ = X.shape
        m = Y.shape[0]

        # Pre‐compute individual determinants
        det_X = np.linalg.det(X)        # (n,)
        det_Y = np.linalg.det(Y)        # (m,)

        K = np.empty((n, m), dtype=X.dtype)
        eps = 1e-12

        # Double loop: no big (n, m, d, d) allocation
        for i in tqdm.tqdm(range(n), desc="Computing kernel matrix"):
            Xi = X[i]
            for j in range(m):
                # one small (d, d) alloc per inner iteration
                ds = np.linalg.det(Xi + Y[j])
                K[i, j] = self.gamma_const * ((det_X[i] * det_Y[j]) / (ds**2 + eps)) ** self.alpha
        
        return K
        
        # # for i in range(n):
        #     Xi = X[i]
        #     for j in range(m):
        #         # one small (d, d) alloc per inner iteration
        #         ds = np.linalg.det(Xi + Y[j])
        #         K[i, j] = ((det_X[i] * det_Y[j]) / (ds**2 + eps)) ** self.alpha

        # return K

class BetaprimeKernelSklearnAdapter(BetaprimeKernel):
    def __init__(self, alpha=1.0):
        # self.kernel = BetaprimeKernel(alpha=alpha)
        super().__init__(alpha=alpha)

    def __call__(self, X, Y):
        K = self.pairwise(np.array(X, dtype=np.float64), np.array(Y, dtype=np.float64))
        return K

# Fast C++ implementation using pybind11
class BetaprimeKernelFast:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def pairwise(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        assert X.ndim == 3 and Y.ndim == 3, "Expecting tensors of shape (n, d, d)"
        # return betaprime_python.pairwise_kernel(X, Y, self.alpha)
        return betaprime.pairwise_kernel(X, Y, self.alpha)
    

class logBetaPrimeKernel:
    def __init__(self, alpha: float = 1.0, normalized: bool = True):
        self.alpha = alpha
        self.loggamma_const = logcone_gamma_fast(2, alpha + 1)
        self.normalized = normalized

    def __call__(self, X: np.ndarray, Y: np.ndarray) -> float:
        # logdet_X = np.linalg.slogdet(X)[1] Use trace(log(X)) = \sum(log(eigenvalues(X)))
        (sign_X, logdet_X) = np.linalg.slogdet(X)
        (sign_Y, logdet_Y) = np.linalg.slogdet(Y)
        (sign_sum, logdet_sum) = np.linalg.slogdet(X + Y)
        eps = 1e-12
        logkernel_val = self.loggamma_const + self.alpha * (sign_X * logdet_X + sign_Y * logdet_Y - 2 * sign_sum * logdet_sum + np.log(eps))
        return logkernel_val


    def pairwise(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Compute log Beta Prime kernel matrix with minimal extra memory
        by trading off for more computation (double loop).

        This approach avoids large (n, m, d, d) allocations
        and instead uses a double loop to compute the kernel matrix
        for each pair of samples in X and Y. For large dimensions, this is more memory efficient but slower due 
        to the python loop overhead.
        """
        n, d, _ = X.shape
        m = Y.shape[0]

        # Pre‐compute individual determinants

        K = np.empty((n, m), dtype=X.dtype)
        eps = 1e-12
        for i in tqdm.tqdm(range(n), desc="Computing log kernel matrix"):
            Xi = X[i]
            logdet_Xi = np.linalg.slogdet(Xi)[1]
            for j in range(m):
                # one small (d, d) alloc per inner iteration
                logdet_Yj = np.linalg.slogdet(Y[j])[1]
                logdet_sum = np.linalg.slogdet(Xi + Y[j])[1]

                K[i, j] = self.loggamma_const + self.alpha * (logdet_Xi + logdet_Yj - 2 * logdet_sum + eps)
        if self.normalized:
            diag_K = np.diag(K)
            d = 0.5 * diag_K
            K = K - d[:, np.newaxis] - d[np.newaxis, :]
        return K
    
class logBetaPrimeKernelSklearnAdapter(logBetaPrimeKernel):
    def __init__(self, alpha=1.0, normalized=True):
        super().__init__(alpha=alpha, normalized=normalized)
        

    def __call__(self, X, Y):
        K = self.pairwise(np.array(X, dtype=np.float64), np.array(Y, dtype=np.float64))
        return K
    