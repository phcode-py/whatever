"""
Reddit → OPINN Pipeline
========================
Scrapes a subreddit for posts matching a topic keyword, extracts per-user
sentiment timeseries, builds a user interaction adjacency matrix, and feeds
everything into the OPINN opinion-dynamics model.

Usage:
    python reddit_opinn.py --subreddit technology --topic "AI" \
        --n_users 30 --days 30 --epochs 80

Uses Reddit's public .json endpoints (no API key required).
Sentiment via HuggingFace distilbert-base-uncased-finetuned-sst-2-english.
"""

import argparse
import time
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from opinn import run_experiment, evaluate
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import torch

from transformers import pipeline as hf_pipeline



# ─── Reddit Scraper ─────────────────────────────────────────────────────────

HEADERS = {"User-Agent": "opinion-dynamics-research/0.1 (academic project)"}
RATE_LIMIT = 2.0  # seconds between requests


def _reddit_get(url: str, params: Optional[dict] = None) -> dict:
    """GET a Reddit .json endpoint with rate-limiting and retries."""
    time.sleep(RATE_LIMIT)
    for attempt in range(3):
        try:
            resp = requests.get(url, headers=HEADERS, params=params, timeout=15)
            if resp.status_code == 429:
                wait = float(resp.headers.get("Retry-After", 10))
                print(f"  [rate-limit] waiting {wait}s ...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, json.JSONDecodeError) as e:
            print(f"  [retry {attempt+1}/3] {e}")
            time.sleep(5 * (attempt + 1))
    return {"data": {"children": []}}


def scrape_subreddit(
    subreddit: str,
    topic: str,
    max_pages: int = 10,
    days: int = 30,
) -> List[dict]:
    """
    Search a subreddit for posts matching `topic` within the last `days` days.
    Returns a flat list of submission dicts.
    """
    base_url = f"https://www.reddit.com/r/{subreddit}/search.json"
    cutoff_utc = time.time() - days * 86400
    posts = []
    after = None
    hit_cutoff = False

    for page in range(max_pages):
        params = {
            "q": topic,
            "restrict_sr": "on",
            "sort": "new",
            "t": "all",
            "limit": 100,
        }
        if after:
            params["after"] = after

        data = _reddit_get(base_url, params)
        children = data.get("data", {}).get("children", [])
        if not children:
            break

        page_all_old = True
        for child in children:
            d = child["data"]
            created = d.get("created_utc", 0)
            if created < cutoff_utc:
                continue  # skip posts older than cutoff
            page_all_old = False
            posts.append({
                "id": d["id"],
                "title": d.get("title", ""),
                "selftext": d.get("selftext", ""),
                "author": d.get("author", "[deleted]"),
                "created_utc": created,
                "num_comments": d.get("num_comments", 0),
                "permalink": d.get("permalink", ""),
                "subreddit": subreddit,
            })

        # Since sort=new, once all posts on a page are old, we're done
        if page_all_old:
            hit_cutoff = True
            break

        after = data.get("data", {}).get("after")
        if not after:
            break
        print(f"  [scrape] page {page+1}: {len(posts)} posts so far")

    if not hit_cutoff and after is not None:
        from datetime import datetime, timezone
        oldest = min((p["created_utc"] for p in posts), default=0)
        oldest_date = datetime.fromtimestamp(oldest, tz=timezone.utc).strftime("%Y-%m-%d")
        print(f"  [warning] pagination limit reached before covering full "
              f"{days}-day window. Oldest post: {oldest_date}")

    print(f"  [scrape] total posts fetched: {len(posts)} "
          f"(last {days} days)")
    return posts


def scrape_comments(permalink: str) -> List[dict]:
    """
    Fetch comments for a single submission.
    Returns list of {author, body, created_utc, parent_id, id}.
    """
    url = f"https://www.reddit.com{permalink}.json"
    data = _reddit_get(url)
    comments = []

    if not isinstance(data, list) or len(data) < 2:
        return comments

    def _walk(node: dict):
        if node.get("kind") != "t1":
            return
        d = node["data"]
        comments.append({
            "id": d.get("name", ""),          # fullname e.g. t1_abc123
            "author": d.get("author", "[deleted]"),
            "body": d.get("body", ""),
            "created_utc": d.get("created_utc", 0),
            "parent_id": d.get("parent_id", ""),   # t1_xxx or t3_xxx
            "link_id": d.get("link_id", ""),        # t3_xxx (the submission)
        })
        replies = d.get("replies")
        if isinstance(replies, dict):
            for child in replies.get("data", {}).get("children", []):
                _walk(child)

    for child in data[1].get("data", {}).get("children", []):
        _walk(child)

    return comments


# ─── Sentiment Analysis ─────────────────────────────────────────────────────

def build_sentiment_classifier():
    """Load distilbert SST-2 sentiment classifier."""
    print("  [sentiment] loading distilbert-sst2 classifier ...")
    return hf_pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=-1,  # CPU
        truncation=True,
        max_length=512,
    )


def score_sentiment(classifier, text: str) -> float:
    """
    Return sentiment in [-1, 1].
    POSITIVE → +score, NEGATIVE → -score.
    """
    if not text or not text.strip():
        return 0.0
    # Truncate very long text to avoid issues
    text = text[:1500]
    try:
        result = classifier(text)[0]
        score = result["score"]
        if result["label"] == "NEGATIVE":
            score = -score
        return score
    except Exception:
        return 0.0


# ─── Data Processing Pipeline ───────────────────────────────────────────────

def _collect_user_items(
    posts: List[dict],
    comments_by_post: Dict[str, List[dict]],
) -> Dict[str, List[dict]]:
    """Aggregate all texts (posts + comments) per user."""
    user_items: Dict[str, List[dict]] = defaultdict(list)

    for p in posts:
        author = p["author"]
        if author in ("[deleted]", "AutoModerator"):
            continue
        text = f"{p['title']} {p['selftext']}".strip()
        user_items[author].append({
            "text": text,
            "created_utc": p["created_utc"],
            "type": "post",
            "post_id": p["id"],
            "parent_id": None,
        })

    for post_id, cmts in comments_by_post.items():
        for c in cmts:
            author = c["author"]
            if author in ("[deleted]", "AutoModerator"):
                continue
            user_items[author].append({
                "text": c["body"],
                "created_utc": c["created_utc"],
                "type": "comment",
                "post_id": post_id,
                "parent_id": c.get("parent_id", ""),
            })

    return user_items


def build_adjacency(
    posts: List[dict],
    comments_by_post: Dict[str, List[dict]],
    user_to_idx: Dict[str, int],
    megathread_threshold: int = 500,
) -> np.ndarray:
    """
    Build adjacency matrix from ALL scraped data (including posts that may
    be pruned from the sentiment timeseries later).

    Returns: [N, N] binary adjacency matrix.
    """
    N = len(user_to_idx)
    adj_np = np.zeros((N, N), dtype=np.float32)

    # Map comment fullnames to authors for reply detection
    comment_author_map: Dict[str, str] = {}
    for post_id, cmts in comments_by_post.items():
        for c in cmts:
            comment_author_map[c["id"]] = c["author"]

    # Connection type 1: direct replies between users
    for post_id, cmts in comments_by_post.items():
        for c in cmts:
            author = c["author"]
            if author not in user_to_idx:
                continue
            parent_id = c.get("parent_id", "")
            parent_author = comment_author_map.get(parent_id)
            if parent_author and parent_author in user_to_idx and parent_author != author:
                i, j = user_to_idx[author], user_to_idx[parent_author]
                adj_np[i, j] += 1.0
                adj_np[j, i] += 1.0

    # Connection type 2: co-participation in same thread (non-megathread)
    for post_id, cmts in comments_by_post.items():
        post_match = [p for p in posts if p["id"] == post_id]
        if post_match and post_match[0].get("num_comments", 0) > megathread_threshold:
            continue

        thread_users = set()
        if post_match:
            pa = post_match[0]["author"]
            if pa in user_to_idx:
                thread_users.add(pa)
        for c in cmts:
            if c["author"] in user_to_idx:
                thread_users.add(c["author"])

        thread_users_list = list(thread_users)
        for i_idx in range(len(thread_users_list)):
            for j_idx in range(i_idx + 1, len(thread_users_list)):
                u_i = user_to_idx[thread_users_list[i_idx]]
                u_j = user_to_idx[thread_users_list[j_idx]]
                adj_np[u_i, u_j] += 0.5
                adj_np[u_j, u_i] += 0.5

    np.fill_diagonal(adj_np, 0.0)

    # Sparsify if too dense
    density = (adj_np > 0).sum() / (N * N)
    if density > 0.5 and N > 10:
        print(f"  [adjacency] density={density:.2f}, sparsifying ...")
        k = max(int(N * 0.3), 3)
        for i in range(N):
            row = adj_np[i]
            if (row > 0).sum() > k:
                threshold = np.sort(row)[-k]
                row[row < threshold] = 0.0

    # Fallback: random sparse edges if no interactions found
    if adj_np.sum() == 0:
        print("  [adjacency] no interactions found, adding random sparse edges")
        for i in range(N):
            neighbors = np.random.choice(
                [j for j in range(N) if j != i],
                size=min(3, N - 1),
                replace=False,
            )
            for j in neighbors:
                adj_np[i, j] = 1.0
                adj_np[j, i] = 1.0

    adj_np = (adj_np > 0).astype(np.float32)
    return adj_np


def subsample_uniform(
    user_items: Dict[str, List[dict]],
    top_users: List[str],
    days: int,
    t_bins: int,
) -> Dict[str, List[dict]]:
    """
    Subsample items per user for uniform temporal coverage.

    For each user, divides the time window into `t_bins` equal intervals
    and keeps at most `max(1, ceil(t_bins / days))` items per interval,
    evenly spaced. Ensures at least 1 item per interval where data exists.

    Returns a new user_items dict with subsampled items.
    """
    import math

    # Find global time range across selected users
    all_times = [
        item["created_utc"]
        for u in top_users
        for item in user_items[u]
        if item["created_utc"] > 0
    ]
    if not all_times:
        return {u: user_items[u] for u in top_users}

    t_min, t_max = min(all_times), max(all_times)
    span = max(t_max - t_min, 1)
    bin_seconds = span / t_bins

    result = {}
    for u in top_users:
        items = sorted(user_items[u], key=lambda x: x["created_utc"])
        # Group into time bins
        bins: Dict[int, List[dict]] = defaultdict(list)
        for item in items:
            if item["created_utc"] <= 0:
                continue
            b = min(int((item["created_utc"] - t_min) / bin_seconds), t_bins - 1)
            bins[b].append(item)

        # Keep evenly spaced items per bin (at least 1, at most ceil(total/bins))
        target_per_bin = max(1, math.ceil(len(items) / t_bins))
        sampled = []
        for b_idx in sorted(bins.keys()):
            bin_items = bins[b_idx]
            if len(bin_items) <= target_per_bin:
                sampled.extend(bin_items)
            else:
                # Evenly spaced selection
                indices = np.linspace(0, len(bin_items) - 1, target_per_bin, dtype=int)
                sampled.extend(bin_items[i] for i in indices)

        result[u] = sampled

    original_count = sum(len(user_items[u]) for u in top_users)
    sampled_count = sum(len(v) for v in result.values())
    print(f"  [subsample] {original_count} items → {sampled_count} "
          f"({sampled_count/max(original_count,1)*100:.0f}% retained)")

    return result


def process_data(
    posts: List[dict],
    comments_by_post: Dict[str, List[dict]],
    classifier,
    n_users: int,
    days: int,
    megathread_threshold: int = 500,
    time_bin: str = "auto",
    adj_override: Optional[np.ndarray] = None,
    top_users_override: Optional[List[str]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, pd.DataFrame, float, float]:
    """
    Process scraped data into OPINN-ready tensors.

    Args:
        adj_override:       If provided, use this pre-built adjacency matrix.
        top_users_override: If provided, use this user list+ordering (must
                            match adj_override rows).

    Returns:
        X_seq:       [N, 1, T]  sentiment timeseries
        adj:         [N, N]      adjacency matrix
        df:          DataFrame   with all metadata
        t_min:       float       epoch timestamp of first time bin
        bin_seconds: float       width of each time bin in seconds
    """
    # ── Collect all texts (posts + comments) per user ────────────────────────
    user_items = _collect_user_items(posts, comments_by_post)

    # ── Select top N users by activity ───────────────────────────────────────
    if top_users_override is not None:
        top_users = top_users_override
    else:
        user_counts = {u: len(items) for u, items in user_items.items()}
        top_users = sorted(user_counts, key=user_counts.get, reverse=True)[:n_users]
    user_to_idx = {u: i for i, u in enumerate(top_users)}
    N = len(top_users)

    if N == 0:
        print("  [warning] no users found, returning empty tensors")
        return torch.zeros(1, 1, 1), torch.zeros(1, 1), pd.DataFrame(), 0.0, 1.0

    print(f"  [process] selected {N} users (requested {n_users})")

    # ── Determine time range from ALL posts, not just top users ──────────────
    all_times = [p["created_utc"] for p in posts if p["created_utc"] > 0]
    for cmts in comments_by_post.values():
        for c in cmts:
            if c["created_utc"] > 0:
                all_times.append(c["created_utc"])

    if not all_times:
        print("  [warning] no valid timestamps")
        adj_t = torch.eye(N) if adj_override is None else torch.tensor(adj_override, dtype=torch.float32)
        return torch.zeros(N, 1, 1), adj_t, pd.DataFrame(), 0.0, 1.0

    t_min, t_max = min(all_times), max(all_times)
    span = t_max - t_min

    # Auto-detect bin size based on data span
    if time_bin == "auto":
        if span < 7 * 86400:
            bin_seconds = 3600        # hourly
            bin_label = "hourly"
        elif span < 90 * 86400:
            bin_seconds = 86400       # daily
            bin_label = "daily"
        else:
            bin_seconds = 7 * 86400   # weekly
            bin_label = "weekly"
    else:
        bin_map = {"hourly": 3600, "daily": 86400, "weekly": 7 * 86400}
        bin_seconds = bin_map.get(time_bin, 86400)
        bin_label = time_bin
    
    T = max(int(span / bin_seconds) + 1, 1)

    # Subsample items for uniform coverage before sentiment scoring
    sub_items = subsample_uniform(user_items, top_users, days, T)

    print(f"  [process] time span: {span/86400:.1f} days, "
          f"binning: {bin_label}, T={T} bins")

    # ── Score sentiment and fill timeseries ───────────────────────────────────
    print("  [sentiment] scoring texts ...")
    sentiment_grid = np.full((N, T), np.nan)
    metadata_rows = []

    for u in top_users:
        idx = user_to_idx[u]
        for item in sub_items[u]:
            ts = item["created_utc"]
            if ts <= 0:
                continue
            t_bin = min(int((ts - t_min) / bin_seconds), T - 1)
            score = score_sentiment(classifier, item["text"])

            if np.isnan(sentiment_grid[idx, t_bin]):
                sentiment_grid[idx, t_bin] = score
            else:
                sentiment_grid[idx, t_bin] = (
                    sentiment_grid[idx, t_bin] + score
                ) / 2.0

            metadata_rows.append({
                "user": u,
                "user_idx": idx,
                "time_bin": t_bin,
                "timestamp": ts,
                "sentiment": score,
                "text": item["text"],
                "type": item["type"],
                "post_id": item["post_id"],
                "parent_id": item["parent_id"],
            })

    # Forward-fill NaNs per user, then fill remaining with 0
    for i in range(N):
        arr = sentiment_grid[i]
        last_valid = np.nan
        for t in range(T):
            if np.isnan(arr[t]):
                if not np.isnan(last_valid):
                    arr[t] = last_valid
            else:
                last_valid = arr[t]
        first_valid_idx = None
        for t in range(T):
            if not np.isnan(arr[t]):
                first_valid_idx = t
                break
        if first_valid_idx is not None and first_valid_idx > 0:
            arr[:first_valid_idx] = arr[first_valid_idx]
        arr[np.isnan(arr)] = 0.0

    X_seq = torch.tensor(sentiment_grid, dtype=torch.float32).unsqueeze(1)  # [N, 1, T]

    # ── Adjacency matrix ─────────────────────────────────────────────────────
    if adj_override is not None:
        adj_np = adj_override
        print(f"  [adjacency] using pre-built adjacency (from full dataset)")
    else:
        print("  [adjacency] building user interaction graph ...")
        adj_np = build_adjacency(posts, comments_by_post, user_to_idx,
                                 megathread_threshold)

    adj = torch.tensor(adj_np, dtype=torch.float32)

    # ── Build metadata DataFrame ─────────────────────────────────────────────
    df = pd.DataFrame(metadata_rows)

    print(f"  [process] X_seq shape: {list(X_seq.shape)}, "
          f"adj shape: {list(adj.shape)}, "
          f"edges: {int(adj_np.sum() / 2)}")

    return X_seq, adj, df, t_min, bin_seconds


# ─── CSV Export ──────────────────────────────────────────────────────────────

def save_dataset(
    subreddit: str,
    topic: str,
    X_seq: torch.Tensor,
    adj: torch.Tensor,
    df: pd.DataFrame,
    output_dir: str = "data",
):
    """Save timeseries, adjacency matrix, and metadata to CSV files."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", f"{subreddit}_{topic}")

    # Timeseries: each row is a user, columns are time bins
    N, _, T = X_seq.shape
    ts_data = X_seq[:, 0, :].numpy()  # [N, T]
    ts_cols = [f"t_{i}" for i in range(T)]
    df_ts = pd.DataFrame(ts_data, columns=ts_cols)
    df_ts.index.name = "user_idx"
    ts_path = out / f"{safe_name}_timeseries.csv"
    df_ts.to_csv(ts_path)
    print(f"  [save] timeseries → {ts_path}")

    # Adjacency matrix
    adj_data = adj.numpy()
    adj_cols = [f"user_{i}" for i in range(adj_data.shape[0])]
    df_adj = pd.DataFrame(adj_data, columns=adj_cols)
    df_adj.index.name = "user_idx"
    adj_path = out / f"{safe_name}_adjacency.csv"
    df_adj.to_csv(adj_path)
    print(f"  [save] adjacency → {adj_path}")

    # Full metadata
    if not df.empty:
        meta_path = out / f"{safe_name}_metadata.csv"
        df.to_csv(meta_path, index=False)
        print(f"  [save] metadata  → {meta_path}")

    return safe_name


# ─── Train / Test Split + OPINN ─────────────────────────────────────────────

def run_pipeline(
    X_seq: torch.Tensor,
    adj: torch.Tensor,
    dataset_name: str,
    n_epochs: int = 50,
    hidden_dim: int = 32,
    context_len: int = 10,
    horizon: int = 5,
    lr: float = 5e-3,
    test_ratio: float = 0.2,
    output_dir: str = "results",
    t_min: float = 0.0,
    bin_seconds: float = 1.0,
):
    """
    Temporal train/test split → train OPINN → baseline → report loss → plot.
    """
    from datetime import datetime, timezone

    N, F, T = X_seq.shape

    # Need at least context_len + 2*horizon for train + test
    min_T = context_len + 2 * horizon
    if T < min_T:
        context_len = max(T // 4, 2)
        horizon = max(T // 6, 1)
        min_T = context_len + 2 * horizon
        print(f"  [pipeline] T={T} too small, adjusted context_len={context_len}, "
              f"horizon={horizon}")

    if T < min_T:
        print(f"  [pipeline] T={T} still too small (need {min_T}). Cannot train.")
        return

    # Temporal split
    split_idx = int(T * (1 - test_ratio))
    split_idx = min(split_idx, T - context_len - horizon)
    split_idx = max(split_idx, context_len + horizon)

    X_train = X_seq[:, :, :split_idx]
    X_test = X_seq

    print(f"\n  [split] train: t=[0, {split_idx}), test: t=[{split_idx}, {T})")
    print(f"  [split] train shape: {list(X_train.shape)}, "
          f"full shape: {list(X_test.shape)}")

    # ── Data summary statistics ──────────────────────────────────────────────
    opinions = X_seq[:, 0, :].numpy()  # [N, T]
    print(f"\n  {'='*50}")
    print(f"  DATA SUMMARY: {dataset_name}")
    print(f"  {'─'*50}")
    print(f"  Nodes (users):    {N}")
    print(f"  Time steps:       {T}")
    print(f"  Global mean:      {opinions.mean():.4f}")
    print(f"  Global std:       {opinions.std():.4f}")
    print(f"  Global median:    {np.median(opinions):.4f}")
    print(f"  Global min:       {opinions.min():.4f}")
    print(f"  Global max:       {opinions.max():.4f}")
    print(f"  IQR:              [{np.percentile(opinions, 25):.4f}, "
          f"{np.percentile(opinions, 75):.4f}]")
    print(f"  Sparsity (zeros): {(opinions == 0).sum() / opinions.size:.1%}")
    print(f"  Adj density:      {(adj.numpy() > 0).sum() / (N * N):.3f}")
    print(f"  {'='*50}")

    # ── Baseline: per-node train mean ────────────────────────────────────────
    train_opinions = X_seq[:, 0, :split_idx].numpy()  # [N, split_idx]
    per_node_mean = train_opinions.mean(axis=1)       # [N]

    # Evaluate baseline on the same test window as OPINN's evaluate()
    test_start = T - context_len - horizon
    test_target = X_seq[:, 0, test_start + context_len:].numpy()  # [N, h]
    baseline_pred = np.tile(per_node_mean[:, None], (1, horizon))  # [N, h]
    baseline_mae = float(np.abs(baseline_pred - test_target).mean())
    baseline_rmse = float(np.sqrt(((baseline_pred - test_target) ** 2).mean()))
    print(f"\n  [baseline] Per-node train-mean predictor:")
    print(f"  [baseline] Test MAE={baseline_mae:.4f}  RMSE={baseline_rmse:.4f}")

    # ── Train OPINN ──────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    result = run_experiment(
        dataset_name=dataset_name,
        X_seq=X_train,
        adj=adj,
        hidden_dim=hidden_dim,
        context_len=context_len,
        horizon=horizon,
        n_epochs=n_epochs,
        lr=lr,
        output_dir=output_dir,
        device=device,
        print_every=max(n_epochs // 10, 1),
    )

    # ── Evaluate on test ─────────────────────────────────────────────────────
    model = result["model"]
    test_metrics = evaluate(model, X_test, context_len, horizon, device)
    print(f"\n  [test] OPINN   MAE={test_metrics['MAE']:.4f}  "
          f"RMSE={test_metrics['RMSE']:.4f}")
    print(f"  [test] Baseline MAE={baseline_mae:.4f}  "
          f"RMSE={baseline_rmse:.4f}")

    # ── Generate predictions for plotting ──────────────────────────────────
    model.eval()
    with torch.no_grad():
        pred_accum = np.zeros((N, T))
        pred_count = np.zeros(T)

        for start in range(0, T - context_len - horizon + 1):
            end_ctx = start + context_len
            X_ctx = X_seq[:, :, start:end_ctx].to(device)
            X_hat = model(X_ctx, horizon).cpu().numpy()  # [N, h]
            for h_i in range(horizon):
                t_idx = end_ctx + h_i
                if t_idx < T:
                    pred_accum[:, t_idx] += X_hat[:, h_i]
                    pred_count[t_idx] += 1

        pred_mask = pred_count > 0
        pred_accum[:, pred_mask] /= pred_count[pred_mask]
        mean_pred = pred_accum.mean(axis=0)  # [T]

    # ── Build real datetime x-axis ───────────────────────────────────────────
    timestamps = [t_min + i * bin_seconds for i in range(T)]
    dates = [datetime.fromtimestamp(ts, tz=timezone.utc) for ts in timestamps]
    import matplotlib.dates as mdates

    # ── Plot ─────────────────────────────────────────────────────────────────
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 7))

    # Sample ~5% of users for trajectory hints (declutter)
    n_sample = max(int(N * 0.05), 1)
    sample_idx = np.random.choice(N, size=n_sample, replace=False)
    for i in sample_idx:
        ax.plot(dates, opinions[i], alpha=0.15, linewidth=0.6, color="steelblue")

    # Percentile bands for trend visualization
    p25 = np.percentile(opinions, 25, axis=0)
    p75 = np.percentile(opinions, 75, axis=0)
    p10 = np.percentile(opinions, 10, axis=0)
    p90 = np.percentile(opinions, 90, axis=0)
    ax.fill_between(dates, p10, p90, alpha=0.08, color="steelblue",
                    label="10th-90th percentile")
    ax.fill_between(dates, p25, p75, alpha=0.18, color="steelblue",
                    label="25th-75th percentile (IQR)")

    # Mean opinion (actual)
    mean_opinion = opinions.mean(axis=0)
    ax.plot(dates, mean_opinion, color="darkred", linewidth=2.5,
            label="Mean opinion (actual)")

    # Mean opinion (OPINN predicted)
    pred_dates = [dates[i] for i in range(T) if pred_mask[i]]
    pred_vals = mean_pred[pred_mask]
    ax.plot(pred_dates, pred_vals, color="limegreen", linewidth=2.5,
            label="Mean opinion (OPINN predicted)")

    # Baseline: flat line at global train mean
    global_train_mean = float(per_node_mean.mean())
    ax.axhline(y=global_train_mean, color="orange", linestyle=":",
               linewidth=1.5, label=f"Baseline (train mean = {global_train_mean:.3f})")

    # Train/test split
    split_date = dates[split_idx] if split_idx < T else dates[-1]
    ax.axvline(x=split_date, color="black", linestyle="--", linewidth=1,
               label="Train/test split")

    # Format x-axis with real dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate(rotation=35, ha="right")

    ax.set_xlabel("Date")
    ax.set_ylabel("Sentiment (opinion)")
    ax.set_title(
        f"Opinion Dynamics: r/{dataset_name}\n"
        f"OPINN  Test MAE={test_metrics['MAE']:.4f}  RMSE={test_metrics['RMSE']:.4f}   |   "
        f"Baseline Test MAE={baseline_mae:.4f}  RMSE={baseline_rmse:.4f}\n"
        f"N={N} users, T={T} bins   "
        f"mean={opinions.mean():.3f}  std={opinions.std():.3f}  "
        f"median={np.median(opinions):.3f}",
        fontsize=11,
    )
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    plot_path = out_path / f"{dataset_name}_opinion_dynamics.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] saved → {plot_path}")

    # ── Print final summary ──────────────────────────────────────────────────
    g = model.get_gating_info()
    print(f"\n{'='*55}")
    print(f"  RESULTS: {dataset_name}")
    print(f"  {'─'*53}")
    print(f"  Data:  N={N}  T={T}  mean={opinions.mean():.4f}  "
          f"std={opinions.std():.4f}  median={np.median(opinions):.4f}")
    print(f"  {'─'*53}")
    print(f"  OPINN    Train MAE: {result['MAE']:.4f}")
    print(f"  OPINN    Test  MAE: {test_metrics['MAE']:.4f}  "
          f"RMSE: {test_metrics['RMSE']:.4f}")
    print(f"  Baseline Test  MAE: {baseline_mae:.4f}  "
          f"RMSE: {baseline_rmse:.4f}")
    print(f"  {'─'*53}")
    print(f"  Learned omega (diffusion/convection): {g['omega_learned']:.4f}")
    print(f"  Learned delta (reaction strength):    {g['delta_learned']:.4f}")
    print(f"{'='*55}")

    return result, test_metrics


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Reddit → OPINN opinion dynamics pipeline"
    )
    parser.add_argument("--subreddit", type=str, required=True,
                        help="Subreddit to scrape (e.g. 'technology')")
    parser.add_argument("--topic", type=str, required=True,
                        help="Topic keyword to search for (e.g. 'AI')")
    parser.add_argument("--n_users", type=int, default=30,
                        help="Number of top users to include (N)")
    parser.add_argument("--days", type=int, required=True,
                        help="Historical lookback window in days")
    parser.add_argument("--max_pages", type=int, default=10,
                        help="Max search result pages to fetch")
    parser.add_argument("--max_threads", type=int, default=50,
                        help="Max threads to fetch comments from")
    parser.add_argument("--megathread_threshold", type=int, default=500,
                        help="Filter threads with more comments than this")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Training epochs")
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--context_len", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--time_bin", type=str, default="auto",
                        choices=["auto", "hourly", "daily", "weekly"])
    parser.add_argument("--results_dir", type=str, default="results",
                        help="Root results directory; each run gets its own subfolder")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  Reddit → OPINN Pipeline")
    print(f"  Subreddit: r/{args.subreddit}")
    print(f"  Topic:     {args.topic}")
    print(f"  N={args.n_users} users, lookback={args.days} days")
    print(f"{'='*60}\n")

    # ── Step 1: Scrape posts ─────────────────────────────────────────────────
    print("[1/6] Scraping posts ...")
    posts = scrape_subreddit(args.subreddit, args.topic,
                             max_pages=args.max_pages, days=args.days)

    if not posts:
        print("  No posts found. Try a different subreddit/topic.")
        return

    # ── Step 2: Scrape comments from top threads ─────────────────────────────
    print("\n[2/6] Scraping comments ...")
    posts_sorted = sorted(posts, key=lambda p: p["num_comments"], reverse=True)
    threads_to_fetch = posts_sorted[:args.max_threads]

    comments_by_post: Dict[str, List[dict]] = {}
    for i, p in enumerate(threads_to_fetch):
        print(f"  [comments] thread {i+1}/{len(threads_to_fetch)}: "
              f"{p['num_comments']} comments - {p['title'][:50]}...")
        cmts = scrape_comments(p["permalink"])
        comments_by_post[p["id"]] = cmts

    total_comments = sum(len(c) for c in comments_by_post.values())
    print(f"  [comments] total comments fetched: {total_comments}")

    # ── Step 3: Build adjacency from ALL data ────────────────────────────────
    print("\n[3/6] Building adjacency matrix (from full dataset) ...")
    all_user_items = _collect_user_items(posts, comments_by_post)
    user_counts = {u: len(items) for u, items in all_user_items.items()}
    top_users = sorted(user_counts, key=user_counts.get, reverse=True)[:args.n_users]
    user_to_idx = {u: i for i, u in enumerate(top_users)}

    adj_np = build_adjacency(posts, comments_by_post, user_to_idx,
                             args.megathread_threshold)
    print(f"  [adjacency] {len(top_users)} users, "
          f"{int(adj_np.sum() / 2)} edges "
          f"(density={((adj_np > 0).sum() / max(adj_np.size, 1)):.3f})")

    # ── Step 4: Process data (subsample + sentiment) ─────────────────────────
    print("\n[4/6] Processing data (subsample + sentiment analysis) ...")
    classifier = build_sentiment_classifier()

    X_seq, adj, df, t_min, bin_seconds = process_data(
        posts=posts,
        comments_by_post=comments_by_post,
        classifier=classifier,
        n_users=args.n_users,
        days=args.days,
        megathread_threshold=args.megathread_threshold,
        time_bin=args.time_bin,
        adj_override=adj_np,
        top_users_override=top_users,
    )

    if X_seq.shape[0] <= 1 or X_seq.shape[2] <= 1:
        print("  Insufficient data to train. Try different parameters.")
        return

    # ── One folder per run ───────────────────────────────────────────────────
    dataset_name = f"{args.subreddit}_{re.sub(r'[^a-zA-Z0-9_-]', '_', args.topic)}"
    run_dir = str(Path(args.results_dir) / dataset_name)

    # ── Step 5: Save dataset ─────────────────────────────────────────────────
    print("\n[5/6] Saving dataset ...")
    save_dataset(
        args.subreddit, args.topic, X_seq, adj, df, run_dir
    )

    # ── Step 6: Train OPINN ──────────────────────────────────────────────────
    print("\n[6/6] Training OPINN ...")
    run_pipeline(
        X_seq=X_seq,
        adj=adj,
        dataset_name=dataset_name,
        n_epochs=args.epochs,
        hidden_dim=args.hidden_dim,
        context_len=args.context_len,
        horizon=args.horizon,
        lr=args.lr,
        test_ratio=args.test_ratio,
        output_dir=run_dir,
        t_min=t_min,
        bin_seconds=bin_seconds,
    )

    print(f"\nDone! All outputs saved in '{run_dir}/'.")


if __name__ == "__main__":
    main()
