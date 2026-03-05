"""
Positive Definite Cauchy Kernel on the SPD Manifold
=====================================================
Implements equation (25) from the paper:

    f(x) = (det(x))^{(N-1)/2} / V(rho) * det[-gamma_tilde^{(k-1)}(log(rho_l))]

where x = X^{-1/2} Y X^{-1/2}, rho_l are eigenvalues of x,
and gamma_tilde(s) = (kappa^2 + s^2)^{-1}.

The full kernel is: K(X, Y) = f(X^{-1/2} Y X^{-1/2})

This is strictly positive definite on the SPD manifold by Godement's theorem,
unlike the naive geodesic substitution k(X,Y) = (kappa^2 + delta^2(X,Y))^{-l}.
"""

import numpy as np
import torch
import sympy as sp
from functools import lru_cache
from typing import List


# ── Symbolic derivative computation ───────────────────────────────────────────

@lru_cache(maxsize=32)
def compute_gamma_tilde_derivatives(N: int, kappa: float) -> List:
    """
    Symbolically compute the first N-1 derivatives of:
        gamma_tilde(s) = (kappa^2 + s^2)^{-1}

    Returns a list of N callable functions [gamma^(0), gamma^(1), ..., gamma^(N-1)]
    evaluated numerically via sympy lambdify.
    """
    s = sp.Symbol('s', real=True)
    kappa_sym = sp.Rational(kappa).limit_denominator(1000)  # exact rational approx

    gamma_tilde = 1 / (kappa_sym**2 + s**2)

    derivatives = []
    expr = gamma_tilde
    for k in range(N):
        # Simplify before lambdifying for numerical stability
        expr_simplified = sp.simplify(expr)
        fn = sp.lambdify(s, expr_simplified, modules='numpy')
        derivatives.append(fn)
        expr = sp.diff(expr, s)

    return derivatives


# ── Core kernel components ─────────────────────────────────────────────────────

def vandermonde(rho: torch.Tensor) -> torch.Tensor:
    """
    Standard Vandermonde product: V(rho) = prod_{k < l} (rho_k - rho_l)

    Args:
        rho: eigenvalues of shape (N,)
    Returns:
        scalar Vandermonde determinant
    """
    N = rho.shape[0]
    V = torch.tensor(1.0, dtype=rho.dtype)
    for k in range(N):
        for l in range(k + 1, N):
            V = V * (rho[k] - rho[l])
    return V


def build_derivative_matrix(log_rho: torch.Tensor,
                             derivatives: List,
                             N: int) -> torch.Tensor:
    """
    Build the N x N matrix M where:
        M[k, l] = -gamma_tilde^{(k-1)}(log(rho_l))

    i.e. row k uses the (k-1)-th derivative, column l uses log(rho_l).

    Args:
        log_rho:     log-eigenvalues of shape (N,)
        derivatives: list of N callable derivative functions
        N:           matrix dimension
    Returns:
        N x N torch tensor
    """
    M = torch.zeros(N, N, dtype=log_rho.dtype)
    log_rho_np = log_rho.numpy()
    for k in range(N):
        for l in range(N):
            val = derivatives[k](log_rho_np[l])
            M[k, l] = -float(val)
    return M


def relative_matrix(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """
    Compute x = X^{-1/2} Y X^{-1/2}.

    Uses eigendecomposition for numerical stability:
        X = V diag(L) V^T  =>  X^{-1/2} = V diag(L^{-1/2}) V^T
    """
    L, V = torch.linalg.eigh(X)
    L_invsqrt = 1.0 / torch.sqrt(L.clamp(min=1e-12))
    X_invsqrt = V * L_invsqrt.unsqueeze(-2) @ V.mT
    return X_invsqrt @ Y @ X_invsqrt


# ── Main kernel class ──────────────────────────────────────────────────────────

class SPDCauchyKernel:
    """
    Strictly positive definite Cauchy kernel on the SPD manifold via
    the spherical (Helgason-Fourier) transform — equation (25).

        K(X, Y) = f(X^{-1/2} Y X^{-1/2})

    where:
        f(x) = (det(x))^{(N-1)/2} / V(rho) * det[-gamma_tilde^{(k-1)}(log(rho_l))]

    and gamma_tilde(s) = (kappa^2 + s^2)^{-1}.

    This is guaranteed PD by Godement's theorem on symmetric spaces,
    unlike the naive geodesic substitution which fails CND for N >= 2.
    """

    def __init__(self, kappa: float = 1.0):
        """
        Args:
            kappa: scale parameter, kappa > 0
        """
        assert kappa > 0, "kappa must be positive"
        self.kappa = kappa
        self._derivatives_cache = {}

    def _get_derivatives(self, N: int) -> List:
        """Retrieve or compute symbolic derivatives for given N."""
        if N not in self._derivatives_cache:
            self._derivatives_cache[N] = compute_gamma_tilde_derivatives(N, self.kappa)
        return self._derivatives_cache[N]

    def f(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate f(x) for a single SPD matrix x of shape (N, N).

        Args:
            x: SPD matrix, typically X^{-1/2} Y X^{-1/2}
        Returns:
            scalar kernel value
        """
        N = x.shape[0]
        derivatives = self._get_derivatives(N)

        # Eigenvalues rho of x  (all positive since x is SPD)
        rho = torch.linalg.eigvalsh(x).clamp(min=1e-12)   # (N,)
        log_rho = torch.log(rho)                            # (N,)

        # det(x)^{(N-1)/2}
        det_x = torch.prod(rho)
        det_power = det_x ** ((N - 1) / 2.0)

        # Vandermonde V(rho)
        V = vandermonde(rho)

        # N x N matrix M[k,l] = -gamma_tilde^{(k-1)}(log(rho_l))
        M = build_derivative_matrix(log_rho, derivatives, N)

        # det(M)
        det_M = torch.linalg.det(M)

        return det_power * det_M / V

    def __call__(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Compute K(X, Y) = f(X^{-1/2} Y X^{-1/2}).

        Args:
            X: SPD matrix of shape (N, N)
            Y: SPD matrix of shape (N, N)
        Returns:
            scalar kernel value
        """
        x = relative_matrix(X, Y)
        return self.f(x)

    def pairwise(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise kernel matrix K[i,j] = K(X[i], Y[j]).

        Args:
            X: batch of SPD matrices (n, N, N)
            Y: batch of SPD matrices (m, N, N)
        Returns:
            kernel matrix of shape (n, m)
        """
        n, m = X.shape[0], Y.shape[0]
        K = torch.zeros(n, m, dtype=X.dtype)
        for i in range(n):
            for j in range(m):
                K[i, j] = self(X[i], Y[j])
        return K


# ── Tests ─────────────────────────────────────────────────────────────────────

def make_spd(n: int, d: int, seed: int = 0) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, d, d))
    spd = A @ A.transpose(0, 2, 1) + np.eye(d) * 0.5
    return torch.from_numpy(spd)


import pytest

class TestSPDCauchyKernel:

    @pytest.fixture(params=[2, 3, 4])
    def matrices(self, request):
        d = request.param
        X = make_spd(6, d, seed=42)
        Y = make_spd(6, d, seed=99)
        return d, X, Y

    def test_positive_definite_gram_matrix(self, matrices):
        """
        THE key test the naive geodesic kernel would fail.
        All eigenvalues of the Gram matrix K[i,j] = K(X[i], X[j])
        must be >= 0 (up to numerical tolerance).
        """
        d, X, _ = matrices
        kernel = SPDCauchyKernel(kappa=1.0)
        K = kernel.pairwise(X, X)
        eigvals = torch.linalg.eigvalsh(K)
        min_eig = eigvals.min().item()
        assert min_eig >= -1e-6, \
            f"Gram matrix has negative eigenvalue {min_eig:.6f} — kernel is not PD"

    def test_symmetry(self, matrices):
        """K(X, Y) == K(Y, X): kernel must be symmetric."""
        d, X, Y = matrices
        kernel = SPDCauchyKernel(kappa=1.0)
        K_xy = kernel.pairwise(X, Y)
        K_yx = kernel.pairwise(Y, X)
        torch.testing.assert_close(K_xy, K_yx.T, rtol=1e-4, atol=1e-6)

    def test_positive_values(self, matrices):
        """Kernel values should be strictly positive."""
        d, X, Y = matrices
        kernel = SPDCauchyKernel(kappa=1.0)
        K = kernel.pairwise(X, Y)
        assert (K > 0).all(), "Kernel produced non-positive values"

    def test_affine_invariance(self, matrices):
        """
        K(AXA^T, AYA^T) == K(X, Y) for any invertible A,
        since x = X^{-1/2}YX^{-1/2} is affine-invariant.
        """
        d, X, Y = matrices
        rng = np.random.default_rng(7)
        A = torch.from_numpy(rng.standard_normal((d, d)))
        AX = (A @ X @ A.mT)
        AY = (A @ Y @ A.mT)
        kernel = SPDCauchyKernel(kappa=1.0)
        K_orig = kernel.pairwise(X, Y)
        K_transformed = kernel.pairwise(AX, AY)
        torch.testing.assert_close(K_orig, K_transformed, rtol=1e-4, atol=1e-6)

    def test_kappa_effect(self, matrices):
        """
        Larger kappa => broader kernel => values closer together.
        Specifically K(kappa=10)[i,j] should differ from K(kappa=0.1)[i,j].
        """
        d, X, Y = matrices
        k1 = SPDCauchyKernel(kappa=0.1)
        k2 = SPDCauchyKernel(kappa=10.0)
        K1 = k1.pairwise(X, Y)
        K2 = k2.pairwise(X, Y)
        assert not torch.allclose(K1, K2, rtol=1e-3), \
            "Kernel should change with kappa"

    def test_gram_matrix_is_symmetric(self, matrices):
        """Gram matrix K(X, X) must be exactly symmetric."""
        d, X, _ = matrices
        kernel = SPDCauchyKernel(kappa=1.0)
        K = kernel.pairwise(X, X)
        torch.testing.assert_close(K, K.T, rtol=1e-5, atol=1e-7)


if __name__ == "__main__":
    # Quick sanity check
    print("Computing SPD Cauchy kernel on 2x2 matrices...")
    X = make_spd(4, 2, seed=0)
    kernel = SPDCauchyKernel(kappa=1.0)
    K = kernel.pairwise(X, X)
    print("Gram matrix:\n", K)
    eigvals = torch.linalg.eigvalsh(K)
    print("Eigenvalues:", eigvals)
    print("All non-negative:", (eigvals >= -1e-6).all().item())