# OPINN: Physics-Informed Neural Opinion Dynamics

A Python implementation of the architecture from ["Advancing Opinion Dynamics Modeling with Neural Diffusion-Convection-Reaction Equation"](https://arxiv.org/abs/2602.05403), with a Reddit data pipeline for real-world opinion tracking.

The model combines graph neural networks with physics-informed ODEs to forecast how opinions evolve across a social network. A Diffusion-Convection-Reaction (DCR) framework captures three forces shaping opinion change: **diffusion** (consensus via neighbor averaging), **convection** (directional drift), and **reaction** (nonlinear individual dynamics). These are integrated using classical ODE solvers (RK4/Euler) over a learned latent space.

## Project Structure

```
.
├── opinn.py            # Core OPINN model (ODE solvers, DCR modules, encoder/decoder, training)
├── reddit_opinn.py     # Reddit scraping + sentiment analysis + OPINN pipeline
├── data/               # Scraped datasets (timeseries, adjacency, metadata CSVs)
├── results/            # Trained weight matrices, loss summaries, forecast plots
└── weights/            # Pre-trained weights for consensus/polarization scenarios
```

## How It Works

1. **Scrape** a subreddit for posts/comments matching a topic keyword
2. **Score** each text with DistilBERT sentiment (mapped to [-1, 1])
3. **Build** a user interaction graph (reply edges + thread co-participation)
4. **Bin** sentiment into a timeseries per user, forward-fill gaps
5. **Train** the OPINN model on the resulting `[N, 1, T]` tensor + adjacency matrix
6. **Forecast** future opinion trajectories and compare against a train-mean baseline

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch numpy pandas matplotlib requests transformers
```

## Usage

### Run the full Reddit pipeline

```bash
python reddit_opinn.py --subreddit technology --topic "AI" \
    --n_users 30 --days 30 --epochs 80
```

Key arguments:

| Flag | Default | Description |
|------|---------|-------------|
| `--subreddit` | `technology` | Subreddit to scrape |
| `--topic` | `"AI"` | Search keyword |
| `--n_users` | `30` | Top N most-active users to model |
| `--days` | `30` | Look-back window in days |
| `--epochs` | `50` | Training epochs |
| `--hidden_dim` | `32` | Latent dimension |
| `--context_len` | `10` | Encoder context window |
| `--horizon` | `5` | Forecast steps |
| `--lr` | `0.005` | Learning rate |

### Use the OPINN model directly

```python
from opinn import OPINN, run_experiment, evaluate

result = run_experiment(
    dataset_name="my_dataset",
    X_seq=X_seq,       # [N, 1, T] opinion timeseries
    adj=adj,            # [N, N] adjacency matrix
    hidden_dim=32,
    context_len=10,
    horizon=5,
    n_epochs=100,
)

model = result["model"]
metrics = evaluate(model, X_test, context_len=10, horizon=5)
print(f"MAE={metrics['MAE']:.4f}  RMSE={metrics['RMSE']:.4f}")
```

## Outputs

- **`results/<name>_opinion_dynamics.png`** -- Forecast plot with actual vs predicted mean opinion, percentile bands, and train/test split
- **`results/<name>_W_D.csv`, `W_C.csv`, `W_V.csv`** -- Learned diffusion, convection, and gating weight matrices
- **`results/<name>_omega.csv`, `delta.csv`** -- DCR mixing coefficients
- **`results/summary.csv`** -- Frobenius norms of all weight matrices across experiments
- **`data/<name>_timeseries.csv`** -- Per-user sentiment timeseries
- **`data/<name>_adjacency.csv`** -- User interaction graph
- **`data/<name>_metadata.csv`** -- Raw text, timestamps, and sentiment scores

## Notes

- Reddit scraping uses public `.json` endpoints (no API key needed) with a 2-second rate limit
- Sentiment analysis runs on CPU via `distilbert-base-uncased-finetuned-sst-2-english`
- The `weights/` directory contains pre-trained weights for consensus and polarization scenarios, along with cross-prediction distances
