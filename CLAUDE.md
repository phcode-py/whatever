# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch numpy pandas matplotlib requests transformers networkx
```

## Running the code

```bash
# Full Reddit scrape + train pipeline
python reddit_opinn.py --subreddit politics --topic "Iran" --n_users 50 --days 30 --epochs 80

# Run the OPINN model standalone (see __main__ at bottom of opinn.py)
python opinn.py

# Visualize a scraped dataset
python visualize.py        # edit ADJ / TS / METADATA paths at top of file

# Quick sentiment sanity check
python test_classifer_hard.py
```

## Architecture

### `opinn.py` — Core model

The model implements a Diffusion-Convection-Reaction (DCR) physics-informed ODE over a user graph. Reading order matches the section comments in the file:

1. **ODE solvers** (`ode_solve`, `_rk4_step`, `_euler_step`) — pure-NumPy-style integration, no external dep.
2. **Graph preprocessing** (`compute_normalized_adjacency`) — symmetric D^{-1/2} A D^{-1/2} normalization.
3. **DCR sub-modules** — `DiffusionModule` (graph conv, ReLU), `ConvectionModule` (attention, standard=O(N²) or linear=O(ND²)), `ReactionModule` (2-layer MLP; variants: `source`/`linear`/`nonlinear`).
4. **`NeuralOpinionDynamics`** — combines the three modules with learned scalar gates ω (diffusion/convection balance) and δ (reaction strength). This is the `f(z,t)` passed to the ODE solver.
5. **`Encoder`** (GRU) + **`Decoder`** (MLP) — map `[N, 1, context_len]` → `[N, D]` and `[N, D, horizon]` → `[N, horizon]`.
6. **`OPINN`** — top-level `nn.Module`: Encoder → `ode_solve(dynamics, ...)` → Decoder.
7. **Training** — sliding-window MSE with gradient clipping (`max_grad_norm=1.0`); evaluation uses the final window only.
8. **Weight export** — `save_weight_matrices()` writes each learned matrix to a CSV with shape `[out_dim, in_dim]` (PyTorch `nn.Linear` convention: rows = output dims).
9. **`run_experiment()`** — end-to-end helper; returns `{'model', 'MAE', 'RMSE', ...}`.
10. **`cross_predict_analysis()`** — swaps DCR dynamics (but keeps encoder/decoder) between trained models to measure weight transferability; saves `weights/cross_prediction.csv`.

### `reddit_opinn.py` — Data pipeline

Scrapes Reddit via public `.json` endpoints (no API key), scores text with DistilBERT sentiment mapped to `[-1, 1]`, builds a user interaction adjacency matrix (direct replies + thread co-participation, sparsified if density > 0.5), bins sentiment into a `[N, 1, T]` timeseries, then calls `run_experiment`.

### `visualize.py` — Analysis plots

Four plots driven by the three path constants at the top (`ADJ`, `TS`, `METADATA`). Uses `MacOSX` matplotlib backend. The cross-prediction RMSE plot reads `weights/cross_prediction.csv`.

## Data / file conventions

| Pattern | Contents |
|---|---|
| `data/{name}_adjacency.csv` | `[N+1, N+1]` CSV; first col = `user_idx`, rest = adjacency rows |
| `data/{name}_timeseries.csv` | `[N, T]` CSV; rows = users, cols = `t_0..t_{T-1}`; first col = `user_idx` |
| `data/{name}_metadata.csv` | One row per scored post/comment: `user, user_idx, time_bin, timestamp, sentiment, text, type, post_id, parent_id` |
| `results/{name}_W_D.csv` etc. | Weight matrices; rows = output dims, cols = `dim_0..dim_{D-1}` |
| `results/{name}_omega.csv` | 1×1 scalar gate |
| `weights/cross_prediction.csv` | Columns: `dynamics_from, data_from, baseline, MAE, RMSE, rel_RMSE` |
| `weights/distances.csv` | Frobenius / cosine distances between consensus and polarization weights |

## Key implementation details

- **Tensor shape convention throughout**: `[N, F, T]` for timeseries; `[N, D]` for latent states.
- `ConvectionModule` defaults to `attention='standard'` (quadratic in N). Switch to `'linear'` for N > ~500.
- `transplant_dynamics_weights(source, target)` copies only `model.dynamics` state dict; encoder/decoder stay from the target. Used by `cross_predict_analysis`.
- The `weights/` directory contains pre-trained consensus/polarization scenario weights (synthetic, not Reddit data).
