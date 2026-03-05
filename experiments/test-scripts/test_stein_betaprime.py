import pytest
import numpy as np
import torch
from typing import Tuple


# ── helpers ──────────────────────────────────────────────────────────────────

def make_spd(batch_size: int, d: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((batch_size, d, d))
    spd = A @ A.transpose(0, 2, 1) + np.eye(d) * 0.5   # guaranteed SPD
    return spd, torch.from_numpy(spd)


def stein_pairwise(X_np: np.ndarray, Y_np: np.ndarray,
                   beta: float, normalized: bool) -> np.ndarray:
    """Reference: mirrors SteinGramMatrix.transform logic exactly."""
    N, M = len(X_np), len(Y_np)
    K = np.zeros((N, M))
    log_det_X = np.array([np.linalg.slogdet(x)[1] for x in X_np])
    log_det_Y = np.array([np.linalg.slogdet(y)[1] for y in Y_np])
    for i in range(N):
        for j in range(M):
            mid = (X_np[i] + Y_np[j]) / 2.0
            _, log_det_M = np.linalg.slogdet(mid)
            if normalized:
                log_k = (-beta * log_det_M
                         + (beta / 2.0) * log_det_X[i]
                         + (beta / 2.0) * log_det_Y[j])
            else:
                log_k = -beta * log_det_M
            K[i, j] = np.exp(log_k)
    return K


def betaprime_pairwise(X_t: torch.Tensor, Y_t: torch.Tensor,
                       alpha: float) -> np.ndarray:
    """Vectorised beta-prime kernel (log-space, numerically stable)."""
    log_det_X = torch.linalg.slogdet(X_t).logabsdet          # (n,)
    log_det_Y = torch.linalg.slogdet(Y_t).logabsdet          # (m,)
    XpY = X_t[:, None] + Y_t[None, :]                        # (n, m, d, d)
    log_det_sum = torch.linalg.slogdet(XpY).logabsdet        # (n, m)
    log_K = alpha * (log_det_X[:, None] + log_det_Y[None, :] - 2 * log_det_sum)
    return torch.exp(log_K).numpy()


# ── tests ─────────────────────────────────────────────────────────────────────

class TestKernelEquivalence:

    @pytest.fixture(params=[2, 3, 5])
    def matrices(self, request):
        d = request.param
        X_np, X_t = make_spd(8, d, seed=42)
        Y_np, Y_t = make_spd(6, d, seed=99)
        return d, X_np, X_t, Y_np, Y_t

    def test_unnormalized_stein_ne_betaprime(self, matrices):
        """Unnormalized Stein kernel is NOT equal to beta-prime."""
        d, X_np, X_t, Y_np, Y_t = matrices
        beta = 1.0
        K_stein = stein_pairwise(X_np, Y_np, beta=beta, normalized=False)
        K_bp    = betaprime_pairwise(X_t, Y_t, alpha=beta)
        assert not np.allclose(K_stein, K_bp, rtol=1e-4), (
            "Unnormalized Stein should differ from beta-prime "
            "(it lacks the det(X) det(Y) normalization)."
        )

    def test_normalized_stein_eq_betaprime_up_to_constant(self, matrices):
        """
        Normalized Stein = beta-prime × 2^(d*beta), with alpha = beta/2.

        Stein (normalized):  det(X)^{β/2} det(Y)^{β/2} · det((X+Y)/2)^{-β}
                        = det(X)^{β/2} det(Y)^{β/2} · 2^{dβ} · det(X+Y)^{-β}

        Beta-prime (α=β/2):  det(X)^{β/2} det(Y)^{β/2} · det(X+Y)^{-β}

        So: K_stein_norm = 2^(d*beta) * K_bp(alpha=beta/2)
        """
        d, X_np, X_t, Y_np, Y_t = matrices
        beta = 1.0
        alpha = beta / 2.0          # ← corrected
        K_stein = stein_pairwise(X_np, Y_np, beta=beta, normalized=True)
        K_bp    = betaprime_pairwise(X_t, Y_t, alpha=alpha)
        scale   = 2 ** (d * beta)
        np.testing.assert_allclose(
            K_stein, scale * K_bp, rtol=1e-5,
            err_msg=f"Expected K_stein_norm = 2^(d*beta) * K_bp(alpha=beta/2) for d={d}"
        )

    def test_ratio_is_constant_across_pairs(self, matrices):
        """
        Ratio K_stein_norm / K_bp(alpha=beta/2) must equal 2^(d*beta) everywhere.
        """
        d, X_np, X_t, Y_np, Y_t = matrices
        beta = 0.75
        alpha = beta / 2.0          # ← corrected
        K_stein = stein_pairwise(X_np, Y_np, beta=beta, normalized=True)
        K_bp    = betaprime_pairwise(X_t, Y_t, alpha=alpha)
        ratio   = K_stein / K_bp
        expected_ratio = 2 ** (d * beta)
        np.testing.assert_allclose(
            ratio,
            expected_ratio * np.ones_like(ratio),
            rtol=1e-5,
            err_msg=f"Ratio should equal 2^(d*beta)={expected_ratio:.4f} for d={d}"
        )

    def test_self_similarity_is_positive(self, matrices):
        """Both kernels return strictly positive values on the diagonal."""
        d, X_np, X_t, _, _ = matrices
        beta = 1.0
        K_stein = stein_pairwise(X_np, X_np, beta=beta, normalized=True)
        K_bp    = betaprime_pairwise(X_t, X_t, alpha=beta)
        assert np.all(np.diag(K_stein) > 0)
        assert np.all(np.diag(K_bp)    > 0)

    def test_symmetry(self, matrices):
        """Both kernel matrices should be symmetric when X == Y."""
        d, X_np, X_t, _, _ = matrices
        beta = 1.0
        K_stein = stein_pairwise(X_np, X_np, beta=beta, normalized=True)
        K_bp    = betaprime_pairwise(X_t, X_t, alpha=beta)
        np.testing.assert_allclose(K_stein, K_stein.T, rtol=1e-6)
        np.testing.assert_allclose(K_bp,    K_bp.T,    rtol=1e-6)