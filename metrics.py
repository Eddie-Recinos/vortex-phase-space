"""
Vortex Phase Space Metrics
==========================
Computes Omega_bind and Omega_rich for an undirected network.

Reference:
    Recinos, E. "Vortex Phase Space: A Structural Framework for Binding
    and Informational Richness in Complex Networks." Submitted to Chaos, 2026.

Usage:
    import networkx as nx
    from metrics import vortex_metrics

    G = nx.watts_strogatz_graph(200, 8, 0.3, seed=42)
    A = nx.to_numpy_array(G)
    bind, rich = vortex_metrics(A)
    print(f"Omega_bind = {bind:.4f},  Omega_rich = {rich:.4f}")
"""

import numpy as np
import networkx as nx
from scipy.linalg import eigh


def vortex_metrics(A, eta=1.1, beta=1.0, gamma=1.0, eps=1e-6):
    """
    Compute Omega_bind and Omega_rich for adjacency matrix A.

    Parameters
    ----------
    A     : (N, N) numpy array  — symmetric adjacency matrix (unweighted)
    eta   : float — susceptibility amplification parameter (default 1.1)
    beta  : float — uniform damping parameter (default 1.0)
    gamma : float — diffusion coupling parameter (default 1.0)
    eps   : float — numerical stability constant (default 1e-6)

    Returns
    -------
    omega_bind : float in [0, 1)
    omega_rich : float >= 0

    Notes
    -----
    Scope: connected, undirected graphs with average clustering C > 0.
    Isolated nodes are excluded before building the circulation operator.
    """
    N = A.shape[0]

    # --- exclude isolated nodes ---
    degrees = A.sum(axis=1)
    active = degrees > 0
    if not active.all():
        A = A[np.ix_(active, active)]
        N = A.shape[0]
        degrees = A.sum(axis=1)

    # --- flow matrix and symmetrised Laplacian ---
    row_sums = np.where(degrees == 0, 1, degrees)
    F = A / row_sums[:, None]                   # row-normalised flow matrix
    F_sym = (F + F.T) / 2                       # symmetrised (NOT row-stochastic)
    L = np.diag(F_sym.sum(axis=1)) - F_sym      # graph Laplacian from actual row sums

    # --- susceptibility profile ---
    # Linear approximation to parameter-free log normalisation
    # chi_i = 1 + log(s_i)/log(s_max), which measures normalised routing capacity.
    # Linear form used because it bounds Omega_bind within [0, 1).
    chi = 1.2 + 0.6 * degrees / (degrees.max() + eps)

    # --- circulation operator ---
    M = eta * np.diag(chi) - beta * np.eye(N) - gamma * L

    # --- leading eigenvector of M ---
    eigenvalues, eigenvectors = eigh(M)
    leading_idx = np.argmax(eigenvalues)
    v = eigenvectors[:, leading_idx]
    v = v / np.linalg.norm(v)

    # --- Omega_bind components ---
    IPR = np.sum(v ** 4)                        # inverse participation ratio
    mu = float(v @ L @ v)                       # Rayleigh quotient
    V = eta * chi.mean() / (beta + gamma * max(mu, 0))  # tipping ratio
    omega_bind = max(V - 1, 0) * (1 - IPR)

    # --- Omega_rich components ---
    E = np.mean((F_sym + F_sym @ F_sym) > 0)   # two-hop accessibility
    p = degrees / (degrees.sum() + eps)
    p = p[p > 0]
    M_deg = -np.sum(p * np.log(p)) / np.log(N)  # load-distribution entropy
    C = nx.average_clustering(nx.from_numpy_array(A))  # clustering coefficient
    H = (M_deg + eps) ** 0.35 * (E + eps) ** 0.35 * (C + eps) ** 0.30
    omega_rich = H * (1 + E)

    return omega_bind, omega_rich


def vortex_batch(networks_dict, **kwargs):
    """
    Compute vortex metrics for a dictionary of named networks.

    Parameters
    ----------
    networks_dict : dict  {name: adjacency_matrix}
    **kwargs      : passed to vortex_metrics

    Returns
    -------
    dict  {name: (omega_bind, omega_rich)}
    """
    return {
        name: vortex_metrics(A, **kwargs)
        for name, A in networks_dict.items()
    }
