# Vortex Phase Space

Code for:

> Recinos, E. "Vortex Phase Space: A Structural Framework for Binding
> and Informational Richness in Complex Networks."
> *Chaos: An Interdisciplinary Journal of Nonlinear Science*, submitted 2026.

## What this code does

Computes two structural metrics for undirected networks:

- **Ω_bind** — coherence of the dominant recurrent circulation mode  
  (amplitude × spatial delocalization of the leading eigenmode)
- **Ω_rich** — diversity of available recurrent pathways  
  (weighted geometric mean of accessibility, load entropy, clustering)

Together they define **vortex phase space**, a two-dimensional structural
map that characterises network architectures by the *type* of recurrent
circulation they support, rather than ranking them on a single axis.

## Installation

```bash
pip install numpy scipy networkx
```

Python 3.8+ required.

## Quick start

```python
import networkx as nx
from metrics import vortex_metrics

G = nx.watts_strogatz_graph(200, 8, 0.3, seed=42)
A = nx.to_numpy_array(G)
bind, rich = vortex_metrics(A)
print(f"Omega_bind = {bind:.3f},  Omega_rich = {rich:.3f}")
# Expected: Omega_bind = 0.745,  Omega_rich = 0.465
```

## Reproducing paper results

### Table 1 — all eight synthetic architectures
```bash
python experiments/synthetic.py
```

### Section 8 Part 3 — SBM sweep (primary evidence)
```bash
python experiments/sbm_sweep.py
```
This is the key experiment. It shows that Ω_rich predicts multi-pattern
reservoir performance (r = +0.893, p = 0.007) with ρ(A) exactly
equalised, and that accessibility E predicts in the *opposite* direction
in the same family — ruling it out as a confound.

### Section 8 Parts 1, 2, 4 — full task dissociation
```bash
python experiments/reservoir.py
```
Runs NARMA-10, multi-pattern, and sine tracking on all eight architectures.
Takes approximately 2 minutes.

### Section 6 — C. elegans null model
```bash
python experiments/connectome.py
```
Requires the C. elegans adjacency matrix (see `data/README.md`).

## Repository structure

```
metrics.py              # Core: vortex_metrics(A) function
experiments/
  synthetic.py          # Table 1 — eight synthetic architectures
  sbm_sweep.py          # Section 8 Part 3 — primary SBM evidence
  reservoir.py          # Section 8 Parts 1,2,4 — task dissociation
  connectome.py         # Section 6 — C. elegans null model
data/
  README.md             # Where to get connectome data
requirements.txt
.zenodo.json
```

## Key result (Section 8 Part 3)

The fixed-density SBM sweep is the paper's primary empirical contribution.
Design: 4-module SBM, N=200, edges=800. ρ(A) exactly equalised to 9.0.
As `frac_intra` increases from 0.3 to 0.9:
- Clustering C rises (more intra-module triangles)
- Accessibility E *falls* (fewer inter-module short paths)
- Ω_rich rises (C gain dominates)

Result:
```
Spearman(Omega_rich, performance) = +0.893  p = 0.007
Spearman(E,          performance) = -0.893  p = 0.007  ← opposite direction
```
E predicts in the wrong direction in the same family, ruling it out as
the confound. The signal is loop density (C), which is what Ω_rich
is designed to capture.

## Citation

If you use this code, please cite:

```
Recinos, E. (2026). Vortex Phase Space: A Structural Framework for
Binding and Informational Richness in Complex Networks.
Chaos: An Interdisciplinary Journal of Nonlinear Science. [submitted]
```

Code archived at: [Zenodo DOI — to be added after deposit]

## Licence

MIT
