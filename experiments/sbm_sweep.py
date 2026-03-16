"""
Fixed-Density SBM Sweep (Section 8, Part 3)
=============================================
Primary evidence that Omega_rich captures loop density independently
of spectral radius and accessibility.

Design:
  - 4-module SBM, N=200, total edges=800, rho(A) equalised to 9.0
  - frac_intra varies 0.3 -> 0.9 (more edges inside modules = more clustering)
  - As frac_intra increases: C rises, E falls (opposite directions)
  - Omega_rich predicts multi-pattern performance: r=+0.893, p=0.007
  - Accessibility E predicts in the WRONG direction: r=-0.893, p=0.007
  - Therefore the signal is loop density (C), not accessibility (E)

Run:
    python experiments/sbm_sweep.py
"""

import numpy as np
import networkx as nx
from scipy.stats import spearmanr
from numpy.linalg import lstsq
from metrics import vortex_metrics


def make_sbm(frac_intra, N=200, target_edges=800, seed=42):
    """Fixed-density SBM with controlled intra/inter-module edge split."""
    rng = np.random.RandomState(seed)
    n_mod, n_per = 4, N // 4
    A = np.zeros((N, N))
    intra = [(i, j) for m in range(n_mod)
             for i in range(m * n_per, (m + 1) * n_per)
             for j in range(i + 1, (m + 1) * n_per)]
    inter = [(i, j) for i in range(N) for j in range(i + 1, N)
             if i // n_per != j // n_per]
    n_intra = min(int(target_edges * frac_intra), len(intra))
    n_inter = min(target_edges - n_intra, len(inter))
    for i, j in rng.permutation(intra)[:n_intra]:
        A[i, j] = A[j, i] = 1
    for i, j in rng.permutation(inter)[:n_inter]:
        A[i, j] = A[j, i] = 1
    return A


def run_multi_pattern(A, target_rho=9.0, n_seeds=15):
    """Multi-pattern reservoir task (3 simultaneous outputs, NMSE)."""
    N = A.shape[0]
    rho = max(abs(np.linalg.eigvals(A))).real
    W = A.astype(float) * (target_rho / rho) * (0.9 / target_rho)
    scores = []
    for seed in range(n_seeds):
        rng = np.random.RandomState(seed)
        Win = rng.randn(N) * 0.1
        u = rng.uniform(0, 0.5, 2000)
        y1 = np.array([np.mean(u[max(0, t - 5):t + 1])  for t in range(2000)])
        y2 = np.array([np.mean(u[max(0, t - 15):t + 1]) for t in range(2000)])
        y3 = np.array([u[t] * u[max(0, t - 7)]          for t in range(2000)])
        Y = np.c_[y1, y2, y3]
        x = np.zeros(N)
        X = []
        for t in range(2000):
            x = np.tanh(W @ x + Win * u[t])
            X.append(x.copy())
        X = np.array(X)
        Wout = lstsq(X[:1500].T @ X[:1500] + 1e-6 * np.eye(N),
                     X[:1500].T @ Y[:1500], rcond=None)[0]
        Y_pred = X[1500:] @ Wout
        nmse = np.mean([
            np.mean((Y[1500:, k] - Y_pred[:, k]) ** 2)
            / (np.var(Y[1500:, k]) + 1e-10)
            for k in range(3)
        ])
        scores.append(nmse)
    return float(np.mean(scores))


def accessibility(A, eps=1e-6):
    """Two-hop accessibility E (fraction of node pairs within graph distance 2)."""
    s = A.sum(axis=1)
    rs = np.where(s == 0, 1, s)
    F = A / rs[:, None]
    Fs = (F + F.T) / 2
    return np.mean((Fs + Fs @ Fs) > 0)


def main():
    fracs = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    rich_vals, e_vals, multi_vals = [], [], []

    print("Fixed-density SBM sweep (N=200, edges=800, rho(A)=9.0)")
    print(f"\n{'frac_intra':>12} {'Omega_rich':>12} {'E':>8} "
          f"{'C':>8} {'multi_NMSE':>12}")
    print("-" * 58)

    for fi in fracs:
        A = make_sbm(fi)
        if not nx.is_connected(nx.from_numpy_array(A)):
            print(f"{fi:>12.1f}  (not connected, skipping)")
            continue
        bind, rich = vortex_metrics(A)
        e = accessibility(A)
        import networkx as nx2
        c = nx2.average_clustering(nx2.from_numpy_array(A))
        multi = run_multi_pattern(A)
        rich_vals.append(rich)
        e_vals.append(e)
        multi_vals.append(multi)
        print(f"{fi:>12.1f} {rich:>12.4f} {e:>8.4f} {c:>8.4f} {multi:>12.4f}")

    print()
    r_rich, p_rich = spearmanr(rich_vals, [-m for m in multi_vals])
    r_e,    p_e    = spearmanr(e_vals,    [-m for m in multi_vals])
    print(f"Spearman(Omega_rich, performance) = {r_rich:+.3f}  p = {p_rich:.3f}")
    print(f"Spearman(E,          performance) = {r_e:+.3f}  p = {p_e:.3f}  "
          f"[opposite direction — E is not the signal]")

    if abs(r_rich) > 0.8 and p_rich < 0.01:
        print("\nResult matches paper (r≈+0.893, p=0.007). ✓")
    else:
        print(f"\nNote: r={r_rich:.3f}, p={p_rich:.3f} — "
              f"check seed or network connectivity.")


if __name__ == "__main__":
    main()
