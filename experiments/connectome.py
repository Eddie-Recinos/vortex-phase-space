"""
C. elegans Null Model Analysis (Section 6)
===========================================
Tests whether C. elegans Omega_rich exceeds density-matched random baselines.

The actual C. elegans connectome is NOT included in this repository
due to licensing considerations. To reproduce the paper's result:

1. Download the C. elegans connectome from:
   Cook et al. (2019) Nature 571:63-71
   Available at: https://www.wormbase.org or https://wormconnectome.org

2. Place the adjacency matrix as a numpy array in data/celegans.npy
   (N=221 nodes, symmetric, unweighted)

3. Run: python experiments/connectome.py

This script also runs the null model using a WS proxy for demonstration.
"""

import numpy as np
import networkx as nx
from scipy import stats
from metrics import vortex_metrics

np.random.seed(42)


def load_connectome(path="data/celegans.npy"):
    """Load C. elegans adjacency matrix. Returns None if not found."""
    try:
        A = np.load(path)
        print(f"Loaded C. elegans connectome: N={A.shape[0]}, "
              f"edges={int(A.sum()//2)}")
        return A
    except FileNotFoundError:
        print(f"Connectome file not found at {path}.")
        print("Using WS proxy for demonstration (see docstring for data source).")
        return None


def run_null_model(N, density, n_er=30, n_sw=30, seed=42):
    """
    Generate density-matched ER and SW null models.
    Returns (er_rich_values, sw_rich_values).
    """
    er_vals, sw_vals = [], []

    for s in range(n_er):
        G = nx.erdos_renyi_graph(N, density, seed=seed * 13 + s)
        if nx.is_connected(G):
            _, r = vortex_metrics(nx.to_numpy_array(G))
            er_vals.append(r)

    for s in range(n_sw):
        k = max(2, int(density * N))
        G = nx.watts_strogatz_graph(N, k, 0.25, seed=seed * 7 + s)
        if nx.is_connected(G):
            _, r = vortex_metrics(nx.to_numpy_array(G))
            sw_vals.append(r)

    return er_vals, sw_vals


def main():
    # --- try to load real connectome, fall back to proxy ---
    A_real = load_connectome()

    if A_real is not None:
        N = A_real.shape[0]
        edges = int(A_real.sum() // 2)
        density = edges / (N * (N - 1) / 2)
        bind_cel, rich_cel = vortex_metrics(A_real)
        print(f"C. elegans: N={N}, edges={edges}, density={density:.4f}")
    else:
        # WS proxy with same N and approximate density
        N, density = 221, 0.0345
        A_proxy = nx.to_numpy_array(nx.watts_strogatz_graph(N, 8, 0.25, seed=7))
        bind_cel, rich_cel = vortex_metrics(A_proxy)
        print(f"WS proxy (N={N}, density≈{density:.4f}): "
              f"Omega_bind={bind_cel:.4f}, Omega_rich={rich_cel:.4f}")
        print("Note: proxy values differ from paper; use real connectome for exact results.")

    # --- null model comparison ---
    print(f"\nRunning null models (N={N}, density≈{density:.4f})...")
    er_vals, sw_vals = run_null_model(N, density)

    er_z = (rich_cel - np.mean(er_vals)) / np.std(er_vals)
    sw_z = (rich_cel - np.mean(sw_vals)) / np.std(sw_vals)

    print(f"\nResults:")
    print(f"  Connectome Omega_rich:    {rich_cel:.4f}")
    print(f"  ER null mean (n={len(er_vals)}):  {np.mean(er_vals):.4f}  "
          f"z = {er_z:+.1f}")
    print(f"  SW null mean (n={len(sw_vals)}):  {np.mean(sw_vals):.4f}  "
          f"z = {sw_z:+.1f}")
    print()
    print(f"Paper reports: ER z = +17.4,  SW z = +11.9")
    if A_real is not None:
        if er_z > 10 and sw_z > 8:
            print("C. elegans is substantially above both baselines. ✓")
        else:
            print(f"Note: z-scores differ from paper — check connectome preprocessing.")
    else:
        print("(Using proxy; run with real connectome for paper z-scores)")


if __name__ == "__main__":
    main()
