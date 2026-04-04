import numpy as np
from scipy.linalg import eigvalsh
import pytest
import matplotlib.pyplot as plt
from pathlib import Path

# ── Helpers ────────────────────────────────────────────────────────────────────

import numpy as np
from scipy.linalg import eigvalsh
import matplotlib.pyplot as plt
from pathlib import Path


def orthogonal_matrix(d: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((d, d))
    Q, _ = np.linalg.qr(A)
    return Q


def make_pair_with_prescribed_log_spectrum(s: np.ndarray, seed: int = 0):
    """
    Construct X = I and Y = Q diag(exp(s)) Q^T so that
    log-eigenvalues of X^{-1/2} Y X^{-1/2} are exactly s.
    """
    d = len(s)
    Q = orthogonal_matrix(d, seed=seed)
    X = np.eye(d)
    Y = Q @ np.diag(np.exp(s)) @ Q.T
    return X, Y


def affine_geodesic(X: np.ndarray, Y: np.ndarray, t: float) -> np.ndarray:
    """
    Affine-invariant geodesic:
        G_t = X^{1/2} (X^{-1/2} Y X^{-1/2})^t X^{1/2}
    """
    LX, VX = np.linalg.eigh(X)
    Xhalf = VX @ np.diag(np.sqrt(np.clip(LX, 1e-12, None))) @ VX.T
    Xinvhalf = VX @ np.diag(1.0 / np.sqrt(np.clip(LX, 1e-12, None))) @ VX.T

    M = Xinvhalf @ Y @ Xinvhalf
    LM, VM = np.linalg.eigh(M)
    Mt = VM @ np.diag(np.clip(LM, 1e-12, None) ** t) @ VM.T
    return Xhalf @ Mt @ Xhalf


def effective_condition_number(K: np.ndarray, tol: float = 1e-10) -> float:
    eigs = eigvalsh(K)
    pos = eigs[eigs > tol]
    if len(pos) == 0:
        return np.inf
    return float(pos.max() / pos.min())


def offdiag_entries(K: np.ndarray) -> np.ndarray:
    n = K.shape[0]
    mask = ~np.eye(n, dtype=bool)
    return K[mask]


def kernel_summary(K: np.ndarray, name: str = "Kernel") -> dict:
    eigs = eigvalsh(K)
    off = offdiag_entries(K)
    return {
        "name": name,
        "shape": K.shape,
        "min": float(np.min(K)),
        "max": float(np.max(K)),
        "mean": float(np.mean(K)),
        "std": float(np.std(K)),
        "off_min": float(np.min(off)),
        "off_max": float(np.max(off)),
        "off_mean": float(np.mean(off)),
        "off_std": float(np.std(off)),
        "min_eig": float(np.min(eigs)),
        "max_eig": float(np.max(eigs)),
        "cond_effective": effective_condition_number(K),
    }
# ---- Made with Claude ------

def make_spd_realistic(n: int, d: int, seed: int = 0,
                       eigenvalue_range: tuple = (0.1, 10.0)) -> np.ndarray:
    """
    Generate SPD matrices with eigenvalues in a realistic EEG range.
    BNCI2014001 covariances after SCM estimation have eigenvalues
    roughly in [0.01, 100] uV^2 — eigenvalue_range controls this spread.
    """
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, d, d))
    Q, _ = np.linalg.qr(A)                                   # random orthogonals
    log_eigs = rng.uniform(np.log(eigenvalue_range[0]),
                           np.log(eigenvalue_range[1]), (n, d))
    eigs = np.exp(log_eigs)
    return np.array([Q[i] @ np.diag(eigs[i]) @ Q[i].T for i in range(n)])


def _mat_sqrt_inv(X):
    L, V = np.linalg.eigh(X)
    return V * (1.0 / np.sqrt(np.clip(L, 1e-12, None)))[None, :] @ V.T


def cauchy_kernel(X, Y, kappa=1.0):
    x = _mat_sqrt_inv(X) @ Y @ _mat_sqrt_inv(X)
    N = x.shape[0]
    rho = np.sort(np.clip(np.linalg.eigvalsh(x), 1e-12, None))
    s = np.log(rho)

    log_term1 = np.sum(np.log(kappa**2) - np.log(kappa**2 + s**2))

    log_term2 = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            d = s[j] - s[i]
            if abs(d) < 1e-10:
                log_term2 += 0.0
            else:
                half_d = d / 2.0
                log_sinh = half_d - np.log(2) if half_d > 20 else np.log(np.sinh(half_d))
                log_term2 += np.log(d) - np.log(2) - log_sinh

    log_val = log_term1 + log_term2
    return float(np.exp(np.clip(log_val, -700, 700)))


def betaprime_kernel(X, Y, alpha=1.0):
    _, ldX = np.linalg.slogdet(X)
    _, ldY = np.linalg.slogdet(Y)
    _, ldS = np.linalg.slogdet(X + Y)
    return float(np.exp(alpha * (ldX + ldY - 2 * ldS)))


def cauchy_gram(X, kappa=1.0):
    n = len(X)
    K = np.array([[cauchy_kernel(X[i], X[j], kappa) for j in range(n)]
                  for i in range(n)])
    return (K + K.T) / 2


def betaprime_gram(X, alpha=1.0):
    n = len(X)
    K = np.array([[betaprime_kernel(X[i], X[j], alpha) for j in range(n)]
                  for i in range(n)])
    return (K + K.T) / 2


def gram_stats(K):
    eigvals = eigvalsh(K)
    return {
        'min_eig':   float(eigvals.min()),
        'max_eig':   float(eigvals.max()),
        'cond':      float(eigvals.max() / (abs(eigvals.min()) + 1e-15)),
        'is_pd':     bool(eigvals.min() > -1e-6),
        'has_nan':   bool(np.any(~np.isfinite(K))),
        'is_sym':    bool(np.allclose(K, K.T, atol=1e-8)),
    }


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestAdditionalKernelProperties:

    @pytest.fixture
    def eeg_like(self):
        return make_spd_realistic(20, 22, seed=0, eigenvalue_range=(0.1, 100.0))

    @pytest.fixture
    def small(self):
        return make_spd_realistic(10, 3, seed=1, eigenvalue_range=(0.5, 5.0))

    def test_cauchy_diagonal_is_one(self, eeg_like):
        diag = np.array([cauchy_kernel(X, X, kappa=1.0) for X in eeg_like])
        assert np.allclose(diag, 1.0, atol=1e-10), \
            f"Cauchy diagonal not all ones: range [{diag.min()}, {diag.max()}]"

    def test_betaprime_diagonal_is_constant(self, eeg_like):
        d = eeg_like.shape[1]
        expected = 4.0 ** (-1.0 * d)   # alpha = 1
        diag = np.array([betaprime_kernel(X, X, alpha=1.0) for X in eeg_like])
        assert np.allclose(diag, expected, atol=1e-10), \
            f"BetaPrime diagonal not constant: expected {expected}, got range [{diag.min()}, {diag.max()}]"
        
    def test_orthogonal_congruence_invariance_cauchy(self, small):
        Q = orthogonal_matrix(small.shape[1], seed=123)
        for i in range(len(small)):
            for j in range(len(small)):
                X, Y = small[i], small[j]
                lhs = cauchy_kernel(X, Y, kappa=1.0)
                rhs = cauchy_kernel(Q @ X @ Q.T, Q @ Y @ Q.T, kappa=1.0)
                assert np.isclose(lhs, rhs, atol=1e-10), \
                    f"Cauchy not orthogonally invariant at pair {(i,j)}"

    def test_orthogonal_congruence_invariance_betaprime(self, small):
        Q = orthogonal_matrix(small.shape[1], seed=123)
        for i in range(len(small)):
            for j in range(len(small)):
                X, Y = small[i], small[j]
                lhs = betaprime_kernel(X, Y, alpha=1.0)
                rhs = betaprime_kernel(Q @ X @ Q.T, Q @ Y @ Q.T, alpha=1.0)
                assert np.isclose(lhs, rhs, atol=1e-10), \
                    f"BetaPrime not orthogonally invariant at pair {(i,j)}"
                
    @pytest.mark.parametrize("scale", [0.1, 3.0, 100.0])
    def test_simultaneous_scaling_invariance_cauchy(self, small, scale):
        for i in range(len(small)):
            for j in range(len(small)):
                X, Y = small[i], small[j]
                lhs = cauchy_kernel(X, Y, kappa=1.0)
                rhs = cauchy_kernel(scale * X, scale * Y, kappa=1.0)
                assert np.isclose(lhs, rhs, atol=1e-10), \
                    f"Cauchy not scale invariant at scale={scale}"

    @pytest.mark.parametrize("scale", [0.1, 3.0, 100.0])
    def test_simultaneous_scaling_invariance_betaprime(self, small, scale):
        for i in range(len(small)):
            for j in range(len(small)):
                X, Y = small[i], small[j]
                lhs = betaprime_kernel(X, Y, alpha=1.0)
                rhs = betaprime_kernel(scale * X, scale * Y, alpha=1.0)
                assert np.isclose(lhs, rhs, atol=1e-10), \
                    f"BetaPrime not scale invariant at scale={scale}"

    def test_cauchy_near_collision_eigenvalues_finite(self):
        s = np.array([0.0, 1e-12, 2e-12, 1e-8, 0.2])
        X, Y = make_pair_with_prescribed_log_spectrum(s, seed=7)
        val = cauchy_kernel(X, Y, kappa=1.0)
        assert np.isfinite(val), "Cauchy kernel not finite under near-colliding log-eigenvalues"
        assert val > 0, "Cauchy kernel should stay positive under near-collision"

    def test_cauchy_exact_collision_identity_case(self):
        X = np.eye(5)
        Y = np.eye(5)
        val = cauchy_kernel(X, Y, kappa=1.0)
        assert np.isclose(val, 1.0, atol=1e-12), \
            f"Cauchy self-kernel at identity should be 1, got {val}"
        
    def test_cauchy_monotone_along_geodesic(self, small):
        X, Y = small[0], small[1]
        ts = np.linspace(0.0, 1.0, 9)
        vals = np.array([
            cauchy_kernel(X, affine_geodesic(X, Y, t), kappa=1.0)
            for t in ts
        ])
        diffs = np.diff(vals)
        assert np.all(diffs <= 1e-8), \
            f"Cauchy not monotone along geodesic: vals={vals}"

    def test_betaprime_monotone_along_geodesic(self, small):
        X, Y = small[0], small[1]
        ts = np.linspace(0.0, 1.0, 9)
        vals = np.array([
            betaprime_kernel(X, affine_geodesic(X, Y, t), alpha=1.0)
            for t in ts
        ])
        diffs = np.diff(vals)
        assert np.all(diffs <= 1e-8), \
            f"BetaPrime not monotone along geodesic: vals={vals}"
        
    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_cauchy_pd_over_random_seeds(self, seed):
        X = make_spd_realistic(15, 10, seed=seed, eigenvalue_range=(0.05, 50.0))
        K = cauchy_gram(X, kappa=1.0)
        stats = gram_stats(K)
        assert stats["is_pd"], \
            f"Cauchy not PD for seed={seed}: min_eig={stats['min_eig']:.4e}"

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_betaprime_pd_over_random_seeds(self, seed):
        X = make_spd_realistic(15, 10, seed=seed, eigenvalue_range=(0.05, 50.0))
        K = betaprime_gram(X, alpha=1.0)
        stats = gram_stats(K)
        assert stats["is_pd"], \
            f"BetaPrime not PD for seed={seed}: min_eig={stats['min_eig']:.4e}"
        
    def test_cauchy_values_in_unit_interval(self, small):
        K = cauchy_gram(small, kappa=1.0)
        assert np.all(K > 0), "Cauchy should be strictly positive"
        assert np.all(K <= 1.0 + 1e-10), "Cauchy should not exceed 1"

    def test_betaprime_values_bounded_by_diagonal(self, small):
        K = betaprime_gram(small, alpha=1.0)
        d = small.shape[1]
        diag_val = 4.0 ** (-d)
        assert np.all(K > 0), "BetaPrime should be strictly positive"
        assert np.all(K <= diag_val + 1e-10), \
            "BetaPrime should not exceed the self-similarity value"
        
    def test_cauchy_better_effective_conditioning_than_betaprime(self, eeg_like):
        Kc = cauchy_gram(eeg_like, kappa=1.0)
        Kb = betaprime_gram(eeg_like, alpha=1.0)

        c_cond = effective_condition_number(Kc)
        b_cond = effective_condition_number(Kb)

        assert c_cond < b_cond, \
            f"Expected effective cond(Cauchy)={c_cond:.3e} < cond(BetaPrime)={b_cond:.3e}"
class TestNumericsVsBetaPrime:

    # --- fixtures ---

    @pytest.fixture
    def eeg_like(self):
        """22-channel, 20 trials — matches BNCI2014001 exactly."""
        return make_spd_realistic(20, 22, seed=0, eigenvalue_range=(0.1, 100.0))

    @pytest.fixture
    def small(self):
        """Small matrices for fast cross-checks."""
        return make_spd_realistic(10, 3, seed=1, eigenvalue_range=(0.5, 5.0))

    @pytest.fixture(params=[
        (0.1, 10.0),    # mild spread
        (0.01, 100.0),  # moderate — typical EEG
        (1e-3, 1e3),    # extreme spread
    ])
    def spread_matrices(self, request):
        lo, hi = request.param
        return make_spd_realistic(12, 22, seed=42,
                                  eigenvalue_range=(lo, hi)), (lo, hi)

    # --- finiteness ---

    def test_cauchy_no_nan_eeg(self, eeg_like):
        """No NaN or Inf in 22-channel Gram matrix — the overflow fix is working."""
        K = cauchy_gram(eeg_like, kappa=1.0)
        assert not np.any(~np.isfinite(K)), \
            "Cauchy Gram matrix contains NaN/Inf at N=22"

    def test_betaprime_no_nan_eeg(self, eeg_like):
        K = betaprime_gram(eeg_like, alpha=1.0)
        assert not np.any(~np.isfinite(K))

    # --- positive definiteness ---

    def test_cauchy_pd_eeg(self, eeg_like):
        """Cauchy Gram matrix must be PD at N=22 — guaranteed by Godement."""
        stats = gram_stats(cauchy_gram(eeg_like, kappa=1.0))
        assert stats['is_pd'], \
            f"Cauchy not PD at N=22: min_eig={stats['min_eig']:.4e}"

    def test_betaprime_pd_eeg(self, eeg_like):
        stats = gram_stats(betaprime_gram(eeg_like, alpha=1.0))
        assert stats['is_pd'], \
            f"BetaPrime not PD at N=22: min_eig={stats['min_eig']:.4e}"

    # --- conditioning ---

    def test_cauchy_better_conditioned_than_betaprime(self, eeg_like):
        """
        Cauchy should have lower condition number than Beta-prime
        due to the near-flat spectral structure shown in the analysis.
        """
        c_stats = gram_stats(cauchy_gram(eeg_like, kappa=1.0))
        b_stats = gram_stats(betaprime_gram(eeg_like, alpha=1.0))
        assert c_stats['cond'] < b_stats['cond'], (
            f"Expected Cauchy cond ({c_stats['cond']:.2f}) < "
            f"BetaPrime cond ({b_stats['cond']:.2f})"
        )

    # --- spread robustness ---

    def test_both_pd_across_spreads(self, spread_matrices):
        """Both kernels must stay PD across eigenvalue spreads."""
        X, (lo, hi) = spread_matrices
        c_stats = gram_stats(cauchy_gram(X, kappa=1.0))
        b_stats = gram_stats(betaprime_gram(X, alpha=1.0))
        assert c_stats['is_pd'], \
            f"Cauchy not PD for spread [{lo}, {hi}]: min_eig={c_stats['min_eig']:.4e}"
        assert b_stats['is_pd'], \
            f"BetaPrime not PD for spread [{lo}, {hi}]: min_eig={b_stats['min_eig']:.4e}"

    def test_cauchy_finite_across_spreads(self, spread_matrices):
        """Cauchy must produce finite values even at extreme eigenvalue spreads."""
        X, (lo, hi) = spread_matrices
        K = cauchy_gram(X, kappa=1.0)
        assert not np.any(~np.isfinite(K)), \
            f"Cauchy has NaN/Inf for spread [{lo}, {hi}]"

    # --- symmetry ---

    def test_symmetry_eeg(self, eeg_like):
        """K(X,Y) == K(Y,X) for both kernels."""
        Kc = cauchy_gram(eeg_like, kappa=1.0)
        Kb = betaprime_gram(eeg_like, alpha=1.0)
        assert np.allclose(Kc, Kc.T, atol=1e-8), "Cauchy Gram not symmetric"
        assert np.allclose(Kb, Kb.T, atol=1e-8), "BetaPrime Gram not symmetric"

    # --- self-similarity is maximum ---

    def test_self_similarity_maximum(self, small):
        """
        For any isotropic kernel k(X,Y) = f(X^{-1/2}YX^{-1/2}):
        k(X,X) >= k(X,Y) for all Y, since x = I at self and f is max at I.
        """
        for i in range(len(small)):
            self_val = cauchy_kernel(small[i], small[i], kappa=1.0)
            for j in range(len(small)):
                if i == j:
                    continue
                cross_val = cauchy_kernel(small[i], small[j], kappa=1.0)
                assert self_val >= cross_val - 1e-8, \
                    f"Self-similarity violated: k(X,X)={self_val:.6f} < k(X,Y)={cross_val:.6f}"

    # --- kappa sensitivity ---

    def test_kappa_sensitivity(self, small):
        """
        Larger kappa broadens the kernel — all values should increase
        since kappa^2 / (kappa^2 + s^2) is increasing in kappa.
        """
        K_small = cauchy_gram(small, kappa=0.5)
        K_large = cauchy_gram(small, kappa=5.0)
        assert np.all(K_large >= K_small - 1e-8), \
            "Larger kappa should give larger kernel values everywhere"

    # --- direct numerical comparison ---

    def test_value_range_comparison(self, eeg_like):
        """
        Print a numerical summary — not a hard assertion,
        but documents the expected dynamic range difference.
        """
        Kc = cauchy_gram(eeg_like, kappa=1.0)
        Kb = betaprime_gram(eeg_like, alpha=1.0)

        off_diag_mask = ~np.eye(len(eeg_like), dtype=bool)
        c_off = Kc[off_diag_mask]
        b_off = Kb[off_diag_mask]

        print(f"\nCauchy    off-diag: mean={c_off.mean():.4f}  "
              f"std={c_off.std():.4f}  "
              f"range=[{c_off.min():.4e}, {c_off.max():.4e}]")
        print(f"BetaPrime off-diag: mean={b_off.mean():.6f}  "
              f"std={b_off.std():.6f}  "
              f"range=[{b_off.min():.4e}, {b_off.max():.4e}]")

        # Dynamic range: max/min of off-diagonal values
        c_range = c_off.max() / (c_off.min() + 1e-15)
        b_range = b_off.max() / (b_off.min() + 1e-15)
        print(f"\nDynamic range  Cauchy: {c_range:.2f}x  BetaPrime: {b_range:.2f}x")

        # Both should have positive off-diagonal values
        assert np.all(c_off > 0), "Cauchy has non-positive off-diagonal values"
        assert np.all(b_off > 0), "BetaPrime has non-positive off-diagonal values"

def plot_kernel_diagnostics(Kc: np.ndarray,
                            Kb: np.ndarray,
                            save_dir: str = "kernel_plots",
                            suffix: str = "eeg_like"):
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    c_off = offdiag_entries(Kc)
    b_off = offdiag_entries(Kb)
    c_eigs = eigvalsh(Kc)
    b_eigs = eigvalsh(Kb)

    # 1. Heatmaps
    plt.figure(figsize=(6, 5))
    plt.imshow(Kc)
    plt.colorbar()
    plt.title("Cauchy Gram matrix")
    plt.tight_layout()
    plt.savefig(save_path / f"cauchy_gram_{suffix}.png", dpi=200)
    plt.close()

    plt.figure(figsize=(6, 5))
    plt.imshow(Kb)
    plt.colorbar()
    plt.title("Beta-prime Gram matrix")
    plt.tight_layout()
    plt.savefig(save_path / f"betaprime_gram_{suffix}.png", dpi=200)
    plt.close()

    # 2. Off-diagonal histogram
    plt.figure(figsize=(7, 4))
    plt.hist(c_off, bins=30, alpha=0.7, label="Cauchy")
    plt.hist(b_off, bins=30, alpha=0.7, label="Beta-prime")
    plt.legend()
    plt.title("Off-diagonal kernel values")
    plt.xlabel("Kernel value")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(save_path / f"offdiag_hist_{suffix}.png", dpi=200)
    plt.close()

    # 3. Log histogram
    plt.figure(figsize=(7, 4))
    plt.hist(np.log10(np.clip(c_off, 1e-300, None)), bins=30, alpha=0.7, label="Cauchy")
    plt.hist(np.log10(np.clip(b_off, 1e-300, None)), bins=30, alpha=0.7, label="Beta-prime")
    plt.legend()
    plt.title("log10 off-diagonal kernel values")
    plt.xlabel("log10(kernel value)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(save_path / f"log_offdiag_hist_{suffix}.png", dpi=200)
    plt.close()

    # 4. Eigenvalue spectrum
    plt.figure(figsize=(7, 4))
    plt.plot(np.arange(len(c_eigs)), np.clip(c_eigs, 1e-16, None), marker='o', label="Cauchy")
    plt.plot(np.arange(len(b_eigs)), np.clip(b_eigs, 1e-16, None), marker='o', label="Beta-prime")
    plt.yscale("log")
    plt.legend()
    plt.title("Gram eigenvalue spectrum")
    plt.xlabel("Eigenvalue index")
    plt.ylabel("Eigenvalue")
    plt.tight_layout()
    plt.savefig(save_path / f"eigs_{suffix}.png", dpi=200)
    plt.close()

    # 5. Sorted off-diagonal profile
    plt.figure(figsize=(7, 4))
    plt.plot(np.sort(c_off), label="Cauchy")
    plt.plot(np.sort(b_off), label="Beta-prime")
    plt.yscale("log")
    plt.legend()
    plt.title("Sorted off-diagonal entries")
    plt.xlabel("Sorted index")
    plt.ylabel("Kernel value")
    plt.tight_layout()
    plt.savefig(save_path / f"sorted_offdiag_{suffix}.png", dpi=200)
    plt.close()

def plot_geodesic_decay(X: np.ndarray,
                        Y: np.ndarray,
                        kappa: float = 1.0,
                        alpha: float = 1.0,
                        save_dir: str = "kernel_plots",
                        suffix: str = "geodesic"):
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    ts = np.linspace(0.0, 1.0, 100)
    c_vals = []
    b_vals = []

    for t in ts:
        Gt = affine_geodesic(X, Y, t)
        c_vals.append(cauchy_kernel(X, Gt, kappa=kappa))
        b_vals.append(betaprime_kernel(X, Gt, alpha=alpha))

    plt.figure(figsize=(7, 4))
    plt.plot(ts, c_vals, label="Cauchy")
    plt.plot(ts, b_vals, label="Beta-prime")
    plt.legend()
    plt.title("Kernel decay along affine-invariant geodesic")
    plt.xlabel("t")
    plt.ylabel("k(X, G_t)")
    plt.tight_layout()
    plt.savefig(save_path / f"geodesic_decay_{suffix}.png", dpi=200)
    plt.close()

def plot_parameter_sensitivity(Xs: np.ndarray,
                               kappas=(0.25, 0.5, 1.0, 2.0, 5.0),
                               alphas=(0.25, 0.5, 1.0, 2.0),
                               save_dir: str = "kernel_plots",
                               suffix: str = "params"):
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 4))
    for kappa in kappas:
        K = cauchy_gram(Xs, kappa=kappa)
        off = np.sort(offdiag_entries(K))
        plt.plot(off, label=f"kappa={kappa}")
    plt.yscale("log")
    plt.legend()
    plt.title("Cauchy off-diagonal profile vs kappa")
    plt.xlabel("Sorted off-diagonal index")
    plt.ylabel("Kernel value")
    plt.tight_layout()
    plt.savefig(save_path / f"cauchy_kappa_sensitivity_{suffix}.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 4))
    for alpha in alphas:
        K = betaprime_gram(Xs, alpha=alpha)
        off = np.sort(offdiag_entries(K))
        plt.plot(off, label=f"alpha={alpha}")
    plt.yscale("log")
    plt.legend()
    plt.title("Beta-prime off-diagonal profile vs alpha")
    plt.xlabel("Sorted off-diagonal index")
    plt.ylabel("Kernel value")
    plt.tight_layout()
    plt.savefig(save_path / f"betaprime_alpha_sensitivity_{suffix}.png", dpi=200)
    plt.close()

def main():
    X = make_spd_realistic(20, 22, seed=0, eigenvalue_range=(0.1, 100.0))

    Kc = cauchy_gram(X, kappa=1.0)
    Kb = betaprime_gram(X, alpha=1.0)

    print("\nCauchy summary")
    for k, v in kernel_summary(Kc, "Cauchy").items():
        print(f"{k}: {v}")

    print("\nBeta-prime summary")
    for k, v in kernel_summary(Kb, "Beta-prime").items():
        print(f"{k}: {v}")

    plot_kernel_diagnostics(Kc, Kb, save_dir="kernel_plots", suffix="eeg_like")
    plot_geodesic_decay(X[0], X[1], save_dir="kernel_plots", suffix="sample_pair")
    plot_parameter_sensitivity(X, save_dir="kernel_plots", suffix="eeg_like")

if __name__ == "__main__":
    main()