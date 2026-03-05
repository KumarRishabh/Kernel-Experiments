import numpy as np
import torch
import pytest

class RiemannianCauchyKernel:
    """
    Isotropic kernel on SPD manifold:  k(X,Y) = (k^2 + delta^2(X,Y))^{-l}
    where delta is the affine-invariant geodesic distance.
    
    Corresponds to spectral measure Gamma(l, k) on the manifold.
    Special case l=1, spectral measure = (k/2)exp(-k|t|) gives the Cauchy kernel.
    """
    def __init__(self, k: float = 1.0, l: float = 1.0):
        self.k = k
        self.l = l

    def geodesic_sq(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Affine-invariant squared geodesic distance between two SPD matrices.
        delta^2(X,Y) = ||log(X^{-1/2} Y X^{-1/2})||_F^2
        """
        # Compute X^{-1/2} via eigendecomposition
        L, V = torch.linalg.eigh(X)                        # X = V diag(L) V^T
        L_invsqrt = 1.0 / torch.sqrt(L.clamp(min=1e-12))
        X_invsqrt = V * L_invsqrt.unsqueeze(-2) @ V.mT    # X^{-1/2}

        M = X_invsqrt @ Y @ X_invsqrt                      # X^{-1/2} Y X^{-1/2}
        log_eigs = torch.log(torch.linalg.eigvalsh(M).clamp(min=1e-12))
        return (log_eigs ** 2).sum()

    def __call__(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        d2 = self.geodesic_sq(X, Y)
        return (self.k ** 2 + d2) ** (-self.l)

    def pairwise(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        n, m = X.shape[0], Y.shape[0]
        K = torch.zeros(n, m, dtype=X.dtype)
        for i in range(n):
            for j in range(m):
                K[i, j] = self(X[i], Y[j])
        return K


# ── tests ─────────────────────────────────────────────────────────────────────

def make_spd(n, d, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, d, d))
    spd = A @ A.transpose(0, 2, 1) + np.eye(d) * 0.5
    return torch.from_numpy(spd)


class TestRiemannianCauchyKernel:

    def test_symmetry(self):
        """k(X,Y) == k(Y,X): geodesic distance is symmetric."""
        X = make_spd(5, 3, seed=0)
        kernel = RiemannianCauchyKernel(k=1.0, l=1.0)
        K = kernel.pairwise(X, X)
        torch.testing.assert_close(K, K.T, rtol=1e-5, atol=1e-7)

    def test_positive_values(self):
        """Kernel values must be strictly positive."""
        X = make_spd(5, 3, seed=1)
        Y = make_spd(4, 3, seed=2)
        kernel = RiemannianCauchyKernel(k=1.0, l=1.0)
        K = kernel.pairwise(X, Y)
        assert (K > 0).all()

    def test_self_similarity_is_maximum(self):
        """
        For isotropic kernels, k(X,X) >= k(X,Y) for all Y,
        since delta(X,X)=0 and the kernel is strictly decreasing in distance.
        """
        X = make_spd(5, 3, seed=3)
        kernel = RiemannianCauchyKernel(k=1.0, l=1.0)
        K = kernel.pairwise(X, X)
        diag = torch.diag(K)
        # Every off-diagonal entry should be <= diagonal of its row
        for i in range(len(X)):
            assert (K[i] <= diag[i] + 1e-6).all(), \
                f"Row {i}: off-diagonal exceeds self-similarity"

    def test_self_similarity_equals_k_power(self):
        """
        delta(X,X) = 0, so k(X,X) = (k^2)^{-l} = k^{-2l} exactly.
        """
        X = make_spd(5, 3, seed=4)
        k_param, l_param = 2.0, 1.5
        kernel = RiemannianCauchyKernel(k=k_param, l=l_param)
        expected = k_param ** (-2 * l_param)
        for i in range(len(X)):
            val = kernel(X[i], X[i]).item()
            assert abs(val - expected) < 1e-6, \
                f"Self-similarity {val} != {expected}"

    def test_affine_invariance(self):
        """
        The affine-invariant geodesic satisfies delta(AXA^T, AYA^T) = delta(X,Y),
        so the kernel must be invariant under congruence transforms.
        """
        X = make_spd(4, 3, seed=5)
        Y = make_spd(4, 3, seed=6)
        rng = np.random.default_rng(7)
        A = torch.from_numpy(rng.standard_normal((3, 3)))  # invertible w.h.p.
        AXAt = (A @ X @ A.mT)
        AYAt = (A @ Y @ A.mT)
        kernel = RiemannianCauchyKernel(k=1.0, l=1.0)
        K_orig      = kernel.pairwise(X, Y)
        K_transformed = kernel.pairwise(AXAt, AYAt)
        torch.testing.assert_close(K_orig, K_transformed, rtol=1e-4, atol=1e-6)

    def test_decreasing_in_distance(self):
        """
        Scaling Y further from X along a geodesic should decrease kernel value.
        """
        d = 3
        X = make_spd(1, d, seed=8)[0]
        kernel = RiemannianCauchyKernel(k=1.0, l=1.0)
        # Move Y away by scaling eigenvalues further from identity
        prev_val = float('inf')
        for t in [0.0, 0.5, 1.0, 2.0, 4.0]:
            Y = X @ torch.matrix_exp(t * torch.eye(d, dtype=X.dtype))
            val = kernel(X, Y).item()
            assert val < prev_val + 1e-8, \
                f"Kernel should decrease as t={t} increases"
            prev_val = val