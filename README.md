# Tiger POMDP: Learning Finite State Controllers with MAPSO, EM, and SGD

This project implements and benchmarks three optimization methods for learning
**Finite State Controller (FSC)** parameters from trajectory data generated in
the classic **Tiger POMDP** environment. The goal is to estimate FSC
parameters that maximize the likelihood of observed action–observation
sequences — and, more importantly, to test whether doing so well actually
recovers the *correct policy*.

A Finite State Controller is a compact policy representation that maintains a
finite amount of internal memory. Unlike reactive policies that depend only on
the current observation, FSCs use hidden memory states to summarize past
information, making them well-suited to partially observable environments.

## Methods compared

1. **MAPSO** (Modified Adaptive Particle Swarm Optimization) — a
   population-based global optimizer.
2. **EM** (Expectation-Maximization) — a likelihood-based local optimizer.
3. **SGD** (Stochastic Gradient Descent) — gradient-based local optimization
   via Fisher's identity.

Each method is also evaluated in a **MAPSO-warm-started** variant (EM/SGD
initialized from a MAPSO solution) to test whether combining global search
with local refinement helps.

## Key question

**Does near-zero training NLL imply the learned FSC recovered the correct
policy?**

Low negative log-likelihood on the training set is not sufficient evidence
that a method learned the true underlying policy. This project directly tests
policy recovery by replaying each learned model's greedy actions against the
ground-truth Bayesian agent's trajectories, and separately checks
generalization via a held-out validation split (train vs. val NLL).

## What's in the notebook

The main notebook (`Tiger_POMDP.ipynb`) walks through:

1. Ground-truth agent and dataset generation
2. Optimizer hyperparameters
3. Fitting with MAPSO
4. Fitting with EM
5. Fitting with SGD
6. Diagnostics — convergence plots across restarts
7. Warm-started EM (initialized from the MAPSO solution)
8. Warm-started SGD (initialized from the MAPSO solution)
9. Hybrid optimization — policy-match evaluation across MAPSO checkpoints
   (how much MAPSO search is needed before handing off to a local optimizer?)
10. Label-switching alignment and comparison tables (Hungarian algorithm on
    posterior overlap, since FSC node indices are arbitrary up to permutation)
11. FSC visualization
12. Train/val NLL scatter across all trials
13. Persisting results to `results_log.csv`

## Repository structure

```
tiger_pomdp.py    # Tiger POMDP environment + ground-truth agent
mapso.py          # MAPSO implementation
em.py             # EM implementation
sgd.py            # SGD implementation
val.py            # Validation-set generation and evaluation helpers
*.ipynb           # Main experiment notebook
results_log.csv   # Logged trial results (generated)
*_checkpoints/    # Cached .npz results per method, keyed by (mc, n_data, seed)
```

## Requirements

- Python 3.11+
- `numpy`, `scipy`, `matplotlib`, `pandas`, `joblib`
- `numba` (used internally by the MAPSO implementation)

Install with:

```bash
pip install numpy scipy matplotlib pandas joblib numba
```

## Usage

Open the notebook and run cells top to bottom. Fitting cells (MAPSO, EM, SGD,
and their warm-started/hybrid variants) cache results to `.npz`/`.json` files
under `*_checkpoints/` directories, keyed by `(mc, n_data, seed)` — delete the
relevant cache file to force a re-run rather than loading cached results.

Key parameters to adjust at the top of the dataset-generation cell:

- `mc` — target memory complexity of the ground-truth FSC
- `n_data` — training set size (episodes)
- `n_particles`, `n_iterations`, `n_restarts` — shared optimizer budget

## Visualizations

[https://canva.link/ntyixpf0yixi82t](https://canva.link/ri79lyqszonnfqt)
