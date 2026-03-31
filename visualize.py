"""
OPINN Data Visualization
=========================
Produces four plots for a given dataset:
  1. Adjacency matrix heatmap
  2. Opinion timeseries with IQR spike detection + top posts
  3. Network graph colored by mean opinion (red=-1, blue=1)
  4. Cross-prediction RMSE table (NxN heatmap, diagonal = own / baseline)

Usage:
    python visualize.py

Configure the paths and scenario lists at the top of the file.
No training is performed — all models are loaded from saved checkpoints.
"""

import copy
import json
import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import networkx as nx
import torch

# ─── Configuration (edit these) ─────────────────────────────────────────────

RESULTS_DIR  = Path("results")
ANALYSIS_DIR = Path("analysis")

# Scenario used for plots 1–3 (must have been run with reddit_opinn.py)
SCENARIO = "politics_Israel"

# Scenarios compared in the cross-prediction table (plot 4).
# Each entry must have a corresponding subfolder in RESULTS_DIR.
CROSS_SCENARIOS = [
    "politics_Israel",
    "AskALiberal_Israel",
    "Conservative_Israel",
    "politics_Iran",
    "changemyview_Iran", 
    "politics_AI",
    "politics_Abortion",
    "politics_Climate",
    "conservative_Climate"

]

# ─── Derived paths for single-scenario plots ────────────────────────────────
_sc_dir  = RESULTS_DIR / SCENARIO
_sc_pfx  = SCENARIO   # file prefix inside the folder matches the folder name
ADJ      = str(_sc_dir / f"{_sc_pfx}_adjacency.csv")
TS       = str(_sc_dir / f"{_sc_pfx}_timeseries.csv")
METADATA = str(_sc_dir / f"{_sc_pfx}_metadata.csv")


# ─── 1. Adjacency Heatmap ──────────────────────────────────────────────────

def plot_adjacency_heatmap(adj_path: str, out_dir: Path) -> None:
    df = pd.read_csv(adj_path, index_col=0)
    adj = df.values.astype(float)
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(adj, cmap="YlOrRd", interpolation="nearest")
    ax.set_title("User Interaction Adjacency Matrix", fontsize=14)
    ax.set_xlabel("User index")
    ax.set_ylabel("User index")
    fig.colorbar(im, ax=ax, label="Edge weight")
    fig.tight_layout()
    path = out_dir / "adjacency_heatmap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path}")


# ─── 2. Opinion Timeseries with Spike Detection ────────────────────────────

def plot_opinion_spikes(ts_path: str, meta_path: str, out_dir: Path) -> None:
    ts_df = pd.read_csv(ts_path, index_col=0)
    meta_df = pd.read_csv(meta_path)

    opinions = ts_df.values  # [N, T]
    T = opinions.shape[1]
    time_steps = np.arange(T)

    mean_op = opinions.mean(axis=0)
    q25 = np.percentile(opinions, 25, axis=0)
    q75 = np.percentile(opinions, 75, axis=0)
    iqr = q75 - q25

    window = max(3, T // 5)
    iqr_diff = np.abs(np.diff(iqr, prepend=iqr[0]))
    mean_diff = np.abs(np.diff(mean_op, prepend=mean_op[0]))

    iqr_roll_std = pd.Series(iqr_diff).rolling(window, min_periods=1, center=True).std().values
    mean_roll_std = pd.Series(mean_diff).rolling(window, min_periods=1, center=True).std().values

    iqr_spikes = iqr_diff > 1.5 * (iqr_roll_std + 1e-6)
    mean_spikes = mean_diff > 1.5 * (mean_roll_std + 1e-6)
    spike_mask = iqr_spikes | mean_spikes
    spike_steps = np.where(spike_mask)[0]

    spike_posts = {}
    if "time_bin" in meta_df.columns:
        for t in spike_steps:
            bin_posts = meta_df[meta_df["time_bin"] == t].copy()
            if bin_posts.empty:
                continue
            bin_posts = bin_posts.sort_values("sentiment", key=lambda s: s.abs(), ascending=False)
            spike_posts[t] = bin_posts.head(3)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True,
                                    gridspec_kw={"height_ratios": [2, 1]})

    ax1.fill_between(time_steps, q25, q75, alpha=0.3, color="steelblue", label="IQR (25th–75th)")
    ax1.plot(time_steps, mean_op, color="navy", linewidth=2, label="Mean opinion")
    for t in spike_steps:
        ax1.axvline(t, color="red", alpha=0.4, linestyle="--", linewidth=1)
    ax1.set_ylabel("Opinion [-1, 1]")
    ax1.set_title("Opinion Timeseries with Spike Detection", fontsize=14)
    ax1.legend(loc="upper right")
    ax1.set_ylim(-1.1, 1.1)

    ax2.plot(time_steps, iqr, color="darkorange", linewidth=2, label="IQR")
    for t in spike_steps:
        ax2.axvline(t, color="red", alpha=0.4, linestyle="--", linewidth=1)
    ax2.set_xlabel("Time bin")
    ax2.set_ylabel("IQR")
    ax2.legend(loc="upper right")
    fig.tight_layout()
    path = out_dir / "opinion_spikes.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path}")

    if spike_posts:
        print("\n=== Detected Opinion Spikes ===")
        for t, posts in sorted(spike_posts.items()):
            print(f"\n--- Time bin {t} (mean={mean_op[t]:.3f}, IQR={iqr[t]:.3f}) ---")
            for _, row in posts.iterrows():
                text = str(row.get("text", ""))[:120]
                sent = row.get("sentiment", float("nan"))
                user = row.get("user", "?")
                print(f"  [{sent:+.3f}] u/{user}: {text}")
    else:
        print("\nNo significant opinion spikes detected.")


# ─── 3. Network Graph with Opinion Coloring ────────────────────────────────

def plot_opinion_graph(adj_path: str, ts_path: str, out_dir: Path) -> None:
    adj_df = pd.read_csv(adj_path, index_col=0)
    ts_df = pd.read_csv(ts_path, index_col=0)
    adj = adj_df.values.astype(float)

    mean_opinions = ts_df.values.mean(axis=1)  # [N]

    G = nx.from_numpy_array(adj)

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "opinion", ["#d62728", "#ffffff", "#1f77b4"]
    )
    norm = mcolors.Normalize(vmin=-1, vmax=1)
    node_colors = [cmap(norm(v)) for v in mean_opinions]

    degrees = np.array([G.degree(n, weight="weight") for n in G.nodes()])
    node_sizes = 100 + 400 * (degrees / (degrees.max() + 1e-9))

    fig, ax = plt.subplots(figsize=(12, 10))
    pos = nx.spring_layout(G, seed=42, k=1.5 / np.sqrt(len(G)), weight="weight")

    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.15, width=0.5)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           node_size=node_sizes, edgecolors="grey", linewidths=0.5)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=6, font_color="black")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Mean opinion", shrink=0.8)
    ax.set_title("User Interaction Graph — Opinion Clusters", fontsize=14)
    ax.axis("off")
    fig.tight_layout()
    path = out_dir / "opinion_graph.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path}")


# ─── 4. Cross-Prediction RMSE Table ────────────────────────────────────────

def _load_model_and_data(scenario: str, results_dir: Path):
    """
    Load a trained OPINN model and its timeseries from disk.
    Returns (model, X_seq [N,1,T], cfg) or None if files are missing.
    No training is performed.
    """
    from opinn import OPINN

    sc_dir = results_dir / scenario
    config_path = sc_dir / f"{scenario}_config.json"
    model_pt    = sc_dir / f"{scenario}_model.pt"
    adj_pt      = sc_dir / f"{scenario}_adj.pt"
    ts_path     = sc_dir / f"{scenario}_timeseries.csv"

    missing = [p for p in [config_path, model_pt, adj_pt, ts_path] if not p.exists()]
    if missing:
        print(f"  [warn] '{scenario}': missing files {[str(p) for p in missing]} — skipping.")
        return None

    with open(config_path) as fh:
        cfg = json.load(fh)

    adj = torch.load(adj_pt, weights_only=True)
    model = OPINN(
        adj=adj,
        input_dim=cfg["input_dim"],
        hidden_dim=cfg["hidden_dim"],
        omega_init=cfg["omega_init"],
        delta_init=cfg["delta_init"],
        attention=cfg["attention"],
        reaction_type=cfg["reaction_type"],
        ode_method=cfg["ode_method"],
    )
    model.load_state_dict(torch.load(model_pt, weights_only=True))
    model.eval()

    ts_df = pd.read_csv(ts_path, index_col=0)
    X_seq = torch.tensor(ts_df.values, dtype=torch.float32).unsqueeze(1)  # [N, 1, T]

    return model, X_seq, cfg


def _transplant_physics(source_model, target_model):
    """
    Return a deep copy of target_model with the learned physics parameters
    (W_D, W_C, W_V, omega, delta, reaction weights) replaced by those from
    source_model — but keeping target's adjacency (A_norm buffer).

    This lets source's learned opinion dynamics run on target's user graph,
    which is valid even when the two scenarios have different numbers of users.
    """
    transplanted = copy.deepcopy(target_model)
    src_sd = source_model.dynamics.state_dict()
    tgt_sd = transplanted.dynamics.state_dict()

    for key, val in src_sd.items():
        if key == "A_norm":
            # Keep target's adjacency — do not overwrite with source's graph
            continue
        if key in tgt_sd and tgt_sd[key].shape == val.shape:
            tgt_sd[key] = val.clone()
        elif key in tgt_sd:
            print(f"  [warn] shape mismatch for dynamics.{key} "
                  f"({val.shape} vs {tgt_sd[key].shape}), skipping.")

    transplanted.dynamics.load_state_dict(tgt_sd)
    return transplanted


def plot_cross_rmse_table(scenarios: list, results_dir: Path, out_dir: Path) -> None:
    """
    Load trained models from results/{scenario}/ (no training) and compute
    an NxN cross-prediction RMSE table.

    Table layout:
        rows    = "weights from" (which model's physics is used)
        columns = "data from"   (which scenario's timeseries is predicted)
        diagonal = own weights on own data (baseline)

    Cross-prediction for off-diagonal [i, j]:
        Source model i's physics (W_D, W_C, W_V, omega, delta) are transplanted
        into a copy of target model j (keeping j's adjacency and encoder/decoder),
        then evaluated on scenario j's timeseries.
    """
    from opinn import evaluate

    device = torch.device("cpu")
    n = len(scenarios)

    # Load all models and data
    loaded = {}
    for sc in scenarios:
        result = _load_model_and_data(sc, results_dir)
        if result is not None:
            loaded[sc] = result  # (model, X_seq, cfg)

    if len(loaded) < 2:
        print("  [cross-pred] Need at least 2 valid scenarios — skipping.")
        return

    # Build NxN RMSE matrix
    rmse_matrix = np.full((n, n), np.nan)

    for i, sc_weights in enumerate(scenarios):
        if sc_weights not in loaded:
            continue
        src_model, _, _ = loaded[sc_weights]

        for j, sc_data in enumerate(scenarios):
            if sc_data not in loaded:
                continue
            tgt_model, X_seq, cfg = loaded[sc_data]
            context_len = cfg["context_len"]
            horizon     = cfg["horizon"]

            if i == j:
                # Diagonal: evaluate own model on own data (no transplant)
                m = tgt_model
            else:
                # Off-diagonal: transplant source physics into target model
                m = _transplant_physics(src_model, tgt_model)
                m.eval()

            metrics = evaluate(m, X_seq, context_len, horizon, device)
            rmse_matrix[i, j] = metrics["RMSE"]
            tag = "(own)" if i == j else f"(weights from {sc_weights})"
            print(f"  [cross-pred] data={sc_data:30s}  weights={sc_weights:30s}  "
                  f"RMSE={metrics['RMSE']:.4f}  {tag}")

    # ── Plot NxN heatmap table ──────────────────────────────────────────────
    cell_size = 2.2
    fig, ax = plt.subplots(figsize=(n * cell_size + 2, n * cell_size + 1.5))

    masked = np.ma.masked_invalid(rmse_matrix)
    im = ax.imshow(masked, cmap="RdYlGn_r", aspect="auto")

    short = [s.replace("_", "\n") for s in scenarios]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(short, fontsize=10)
    ax.set_yticklabels(short, fontsize=10)
    ax.set_xlabel("Data from  (evaluated on)", fontsize=12, labelpad=10)
    ax.set_ylabel("Weights from", fontsize=12, labelpad=10)
    ax.set_title(
        "Cross-Prediction RMSE\n"
        "diagonal = own weights (baseline)   |   off-diagonal = transplanted physics",
        fontsize=12,
    )

    # Annotate each cell
    for i in range(n):
        for j in range(n):
            val = rmse_matrix[i, j]
            if np.isnan(val):
                continue
            text = f"{val:.4f}"
            if i == j:
                text += "\n(own)"
            ax.text(
                j, i, text,
                ha="center", va="center",
                fontsize=10,
                fontweight="bold" if i == j else "normal",
                color="black",
            )

    # Highlight diagonal cells with a border
    for k in range(n):
        ax.add_patch(plt.Rectangle(
            (k - 0.5, k - 0.5), 1, 1,
            fill=False, edgecolor="black", linewidth=2.5,
        ))

    fig.colorbar(im, ax=ax, label="RMSE", shrink=0.7, pad=0.02)
    fig.tight_layout()
    path = out_dir / "cross_prediction_rmse.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path}")


# ─── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    print("1/4  Adjacency heatmap …")
    plot_adjacency_heatmap(ADJ, ANALYSIS_DIR)

    print("2/4  Opinion spikes …")
    plot_opinion_spikes(TS, METADATA, ANALYSIS_DIR)

    print("3/4  Opinion graph …")
    plot_opinion_graph(ADJ, TS, ANALYSIS_DIR)

    print("4/4  Cross-prediction RMSE table …")
    plot_cross_rmse_table(CROSS_SCENARIOS, RESULTS_DIR, ANALYSIS_DIR)

    print(f"\nDone. All plots saved in '{ANALYSIS_DIR}/'.")
