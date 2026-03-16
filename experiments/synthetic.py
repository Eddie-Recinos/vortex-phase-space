"""
Synthetic Network Experiments
==============================
Reproduces Table 1: vortex metrics for all eight synthetic architectures.

Run:
    python experiments/synthetic.py
"""

import numpy as np
import networkx as nx
from metrics import vortex_metrics

np.random.seed(42)


def make_hmsw(n=200, seed=42):
    """Hierarchical modular small-world network."""
    rng = np.random.RandomState(seed)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    modules = [range(i * 50, (i + 1) * 50) for i in range(4)]
    for mod in modules:
        nodes = list(mod)
        for i in nodes:
            for j in nodes:
                if i < j and rng.random() < 0.12:
                    G.add_edge(i, j)
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < 0.008:
                G.add_edge(i, j)
    return nx.to_numpy_array(G)


NETWORKS = {
    "Ring lattice":  nx.to_numpy_array(nx.watts_strogatz_graph(200, 8, 0.0, seed=42)),
    "ER random":     nx.to_numpy_array(nx.erdos_renyi_graph(200, 0.04, seed=42)),
    "Small-world":   nx.to_numpy_array(nx.watts_strogatz_graph(200, 8, 0.3, seed=42)),
    "HMSW":          make_hmsw(),
    "BA":            nx.to_numpy_array(nx.barabasi_albert_graph(200, 4, seed=42)),
    "RR":            nx.to_numpy_array(nx.random_regular_graph(8, 200, seed=42)),
    "Caveman":       nx.to_numpy_array(nx.connected_caveman_graph(20, 10)),
    "SW-p0.1":       nx.to_numpy_array(nx.watts_strogatz_graph(200, 8, 0.1, seed=42)),
}


def main():
    print(f"\n{'Network':<16} {'N':>5} {'Edges':>7} {'Density':>9} "
          f"{'Omega_bind':>12} {'Omega_rich':>12}")
    print("-" * 65)

    for name, A in NETWORKS.items():
        N = A.shape[0]
        edges = int(A.sum() // 2)
        density = edges / (N * (N - 1) / 2)
        bind, rich = vortex_metrics(A)
        print(f"{name:<16} {N:>5} {edges:>7} {density:>9.4f} "
              f"{bind:>12.3f} {rich:>12.3f}")

    print()
    print("Expected values (from paper Table 1):")
    expected = {
        "Ring lattice": (0.975, 0.401), "ER random":   (0.624, 0.303),
        "Small-world":  (0.745, 0.465), "HMSW":        (0.670, 0.366),
        "BA":           (0.393, 0.500), "RR":           (0.975, 0.290),
        "Caveman":      (0.908, 0.409), "SW-p0.1":     (0.836, 0.453),
    }
    all_ok = True
    for name, (exp_b, exp_r) in expected.items():
        b, r = vortex_metrics(NETWORKS[name])
        ok = abs(b - exp_b) < 0.005 and abs(r - exp_r) < 0.005
        if not ok:
            all_ok = False
            print(f"  MISMATCH {name}: computed ({b:.3f}, {r:.3f}), "
                  f"expected ({exp_b:.3f}, {exp_r:.3f})")
    if all_ok:
        print("  All values match Table 1 within tolerance. ✓")


if __name__ == "__main__":
    main()
