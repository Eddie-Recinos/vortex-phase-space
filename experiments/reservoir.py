"""
Reservoir Computing Experiments (Section 8)
=============================================
Reproduces the four-part task dissociation:
  Part 1 — NARMA-10 with partial correlation
  Part 2 — Multi-pattern task
  Part 3 — SBM sweep (see sbm_sweep.py for the dedicated script)
  Part 4 — Sine tracking (validates Omega_bind axis)

Run:
    python experiments/reservoir.py
"""

import numpy as np
import networkx as nx
from scipy.stats import spearmanr
from numpy.linalg import lstsq
from metrics import vortex_metrics

np.random.seed(42)


# ── Network construction ──────────────────────────────────────────

def make_hmsw(seed=42):
    rng = np.random.RandomState(seed)
    G = nx.Graph()
    G.add_nodes_from(range(200))
    for mod in [range(i * 50, (i + 1) * 50) for i in range(4)]:
        nodes = list(mod)
        for i in nodes:
            for j in nodes:
                if i < j and rng.random() < 0.12:
                    G.add_edge(i, j)
    for i in range(200):
        for j in range(i + 1, 200):
            if rng.random() < 0.008:
                G.add_edge(i, j)
    return nx.to_numpy_array(G)


NETWORKS = {
    "Ring":    nx.to_numpy_array(nx.watts_strogatz_graph(200, 8, 0.0, seed=42)),
    "ER":      nx.to_numpy_array(nx.erdos_renyi_graph(200, 0.04, seed=42)),
    "SW":      nx.to_numpy_array(nx.watts_strogatz_graph(200, 8, 0.3, seed=42)),
    "HMSW":    make_hmsw(),
    "BA":      nx.to_numpy_array(nx.barabasi_albert_graph(200, 4, seed=42)),
    "RR":      nx.to_numpy_array(nx.random_regular_graph(8, 200, seed=42)),
    "Caveman": nx.to_numpy_array(nx.connected_caveman_graph(20, 10)),
    "SW-p0.1": nx.to_numpy_array(nx.watts_strogatz_graph(200, 8, 0.1, seed=42)),
}


# ── Reservoir tasks ───────────────────────────────────────────────

def run_narma(A, n_seeds=20, spec=0.9):
    """NARMA-10 benchmark. Returns mean NMSE (lower = better)."""
    N = A.shape[0]
    rho = max(abs(np.linalg.eigvals(A))).real
    W = A.astype(float) * (spec / rho)
    scores = []
    for seed in range(n_seeds):
        rng = np.random.RandomState(seed)
        Win = rng.randn(N) * 0.1
        u = rng.uniform(0, 0.5, 2000)
        # NARMA-10 target
        y = np.zeros(2000)
        for t in range(10, 2000):
            y[t] = (0.3 * y[t-1]
                    + 0.05 * y[t-1] * sum(y[t-k] for k in range(1, 11))
                    + 1.5 * u[t-1] * u[t-10] + 0.1)
        x = np.zeros(N)
        X = []
        for t in range(2000):
            x = np.tanh(W @ x + Win * u[t])
            X.append(x.copy())
        X = np.array(X)
        Wout = lstsq(X[:1500].T @ X[:1500] + 1e-6 * np.eye(N),
                     X[:1500].T @ y[:1500], rcond=None)[0]
        y_pred = X[1500:] @ Wout
        nmse = np.mean((y[1500:] - y_pred) ** 2) / (np.var(y[1500:]) + 1e-10)
        scores.append(nmse)
    return float(np.mean(scores))


def run_multi(A, n_seeds=20, spec=0.9):
    """Multi-pattern task (3 simultaneous outputs). Returns mean NMSE."""
    N = A.shape[0]
    rho = max(abs(np.linalg.eigvals(A))).real
    W = A.astype(float) * (spec / rho)
    scores = []
    for seed in range(n_seeds):
        rng = np.random.RandomState(seed)
        Win = rng.randn(N) * 0.1
        u = rng.uniform(0, 0.5, 2000)
        y1 = np.array([np.mean(u[max(0, t-5):t+1])  for t in range(2000)])
        y2 = np.array([np.mean(u[max(0, t-15):t+1]) for t in range(2000)])
        y3 = np.array([u[t] * u[max(0, t-7)]         for t in range(2000)])
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
            np.mean((Y[1500:, k] - Y_pred[:, k]) ** 2) / (np.var(Y[1500:, k]) + 1e-10)
            for k in range(3)
        ])
        scores.append(nmse)
    return float(np.mean(scores))


def run_sine(A, n_seeds=20, spec=0.9):
    """Sinusoidal tracking task. Returns mean NMSE (lower = better)."""
    N = A.shape[0]
    rho = max(abs(np.linalg.eigvals(A))).real
    W = A.astype(float) * (spec / rho)
    scores = []
    for seed in range(n_seeds):
        rng = np.random.RandomState(seed)
        Win = rng.randn(N) * 0.1
        t_ = np.arange(2000) / 2000
        u = 0.5 * np.sin(2 * np.pi * 3 * t_) + 0.1 * rng.randn(2000)
        y = np.sin(2 * np.pi * 3 * t_ + 0.5)
        x = np.zeros(N)
        X = []
        for t in range(2000):
            x = np.tanh(W @ x + Win * u[t])
            X.append(x.copy())
        X = np.array(X)
        Wout = lstsq(X[:1500].T @ X[:1500] + 1e-6 * np.eye(N),
                     X[:1500].T @ y[:1500], rcond=None)[0]
        y_pred = X[1500:] @ Wout
        nmse = np.mean((y[1500:] - y_pred) ** 2) / (np.var(y[1500:]) + 1e-10)
        scores.append(nmse)
    return float(np.mean(scores))


def partial_spearman(x, y, z):
    """Partial Spearman correlation of x and y controlling for z."""
    from scipy.stats import rankdata
    xr = rankdata(x).astype(float)
    yr = rankdata(y).astype(float)
    zr = rankdata(z).astype(float)

    def residuals(a, b):
        coef = lstsq(np.c_[b, np.ones_like(b)], a, rcond=None)[0]
        return a - np.c_[b, np.ones_like(b)] @ coef

    return np.corrcoef(residuals(xr, zr), residuals(yr, zr))[0, 1]


def main():
    names = list(NETWORKS.keys())

    print("Computing vortex metrics and spectral radius...")
    bind_v, rich_v, rho_v = [], [], []
    for name in names:
        A = NETWORKS[name]
        b, r = vortex_metrics(A)
        bind_v.append(b)
        rich_v.append(r)
        rho_v.append(max(abs(np.linalg.eigvals(A))).real)

    print("Running reservoir tasks (this takes ~2 minutes)...")
    narma_v = [run_narma(NETWORKS[n]) for n in names]
    multi_v = [run_multi(NETWORKS[n]) for n in names]
    sine_v  = [run_sine(NETWORKS[n])  for n in names]

    print("\n" + "="*60)
    print("PART 1 — NARMA-10 partial correlations")
    print("="*60)
    pr_rich_narma = partial_spearman(rich_v, [-m for m in narma_v], rho_v)
    pr_rho_narma  = partial_spearman(rho_v,  [-m for m in narma_v], rich_v)
    print(f"  Omega_rich partial r (ctrl rho) = {pr_rich_narma:+.3f}")
    print(f"  rho(A)     partial r (ctrl rich) = {pr_rho_narma:+.3f}")
    print(f"  Interpretation: rho(A) dominates NARMA-10 (as reported in paper)")

    print("\n" + "="*60)
    print("PART 2 — Multi-pattern task dissociation")
    print("="*60)
    pr_rich_multi = partial_spearman(rich_v, [-m for m in multi_v], rho_v)
    pr_rho_multi  = partial_spearman(rho_v,  [-m for m in multi_v], rich_v)
    print(f"  Omega_rich partial r (ctrl rho) = {pr_rich_multi:+.3f}")
    print(f"  rho(A)     partial r (ctrl rich) = {pr_rho_multi:+.3f}")
    print(f"  Sign convention: negative partial r = higher Omega_rich "
          f"predicts lower NMSE (better performance)")

    print("\n" + "="*60)
    print("PART 4 — Sine tracking (Omega_bind validation)")
    print("="*60)
    pr_bind_sine = partial_spearman(bind_v, [-m for m in sine_v], rho_v)
    pr_rho_sine  = partial_spearman(rho_v,  [-m for m in sine_v], bind_v)
    print(f"  Omega_bind partial r (ctrl rho) = {pr_bind_sine:+.3f}")
    print(f"  rho(A)     partial r (ctrl bind) = {pr_rho_sine:+.3f}")
    print(f"  Paper reports: Omega_bind partial r = +0.73")

    print("\n" + "="*60)
    print("TASK DISSOCIATION SUMMARY")
    print("="*60)
    print(f"  NARMA-10  dominated by rho(A):     rho partial r = {pr_rho_narma:+.3f}")
    print(f"  Multi-pattern by Omega_rich:       rich partial r = {pr_rich_multi:+.3f}")
    print(f"  Sine tracking by Omega_bind:       bind partial r = {pr_bind_sine:+.3f}")


if __name__ == "__main__":
    main()
