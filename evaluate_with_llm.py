"""
Full Pipeline Evaluation — Layer 6 (LightGBM) + Layer 7 (LLM Reasoning)

Evaluates the complete 7-layer pipeline on the held-out 20% test set.

  Layer 6 evaluation: runs on the full test set (667 samples) using cached
  features — fast, no API calls required.

  Layer 7 evaluation: runs on a configurable sample of test tweets through
  InferencePipeline with use_llm_reasoning=True and llm_use_tavily=True.
  Compares GPT-4o directions against ground-truth labels and LightGBM predictions.

Usage:
    python evaluate_with_llm.py
    python evaluate_with_llm.py --llm-samples 100 --timeframe 5m
    python evaluate_with_llm.py --llm-samples 0   # Layer 6 only, skip LLM
    python evaluate_with_llm.py --no-cache        # re-extract Layer 6 features

Outputs (written to --output-dir, default saved_models/):
    llm_eval_metrics.json   full metrics dict
    llm_eval_report.md      readable comparison report
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Load .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

sys.path.insert(0, str(Path(__file__).parent))

from train_pipeline import split_train_test_df
from models.market_impact.impact_predictor import (
    MarketImpactPredictor, ASSETS, TIMEFRAMES, ASSET_TICKERS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("evaluate_with_llm")

DIRECTION_LABEL = {-1: "BEARISH", 0: "NEUTRAL", 1: "BULLISH"}


# ─────────────────────────────────────────────────────────────────────────────
# Metrics helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute classification metrics for direction prediction."""
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score,
        f1_score, precision_score, recall_score,
        matthews_corrcoef, cohen_kappa_score,
        confusion_matrix,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        "n_samples": int(len(y_true)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[-1, 0, 1]).tolist(),
        "class_dist": {
            "bearish(-1)": int((y_true == -1).sum()),
            "flat(0)": int((y_true == 0).sum()),
            "bullish(1)": int((y_true == 1).sum()),
        },
    }


def compute_relevance_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                               y_proba: np.ndarray = None) -> dict:
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
    )
    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "n_samples": int(len(y_true)),
    }
    if y_proba is not None and len(np.unique(y_true)) > 1:
        from sklearn.metrics import roc_auc_score
        result["roc_auc"] = float(roc_auc_score(y_true, y_proba))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Layer 6 evaluation (LightGBM, full test set)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_layer6(
    predictor: MarketImpactPredictor,
    test_features: list,
    test_df: pd.DataFrame,
    timeframes: list,
) -> dict:
    """Evaluate LightGBM models on the full held-out test set."""
    logger.info(f"Evaluating Layer 6 (LightGBM) on {len(test_features)} test samples...")
    results = {"per_asset": {}, "relevance": None, "global_summary": {}}
    rows = []

    for tf in timeframes:
        logger.info(f"  Timeframe: {tf}")
        preds_list = []
        for fd in test_features:
            preds, _, _ = predictor.predict(fd, timeframe=tf)
            preds_list.append(preds)

        for asset in ASSETS:
            label_col = f"{asset}_{ASSET_TICKERS[asset]}_actual_dir_{tf}"
            if label_col not in test_df.columns:
                continue

            y_true = test_df[label_col].astype(int).to_numpy()
            y_pred = np.array([p[asset].direction for p in preds_list], dtype=int)

            m = compute_metrics(y_true, y_pred)
            key = f"{asset}_{tf}"
            results["per_asset"][key] = {"asset": asset, "timeframe": tf, **m}
            rows.append({"asset": asset, "timeframe": tf, **m})

    # Relevance
    if predictor.relevance_model is not None and "is_market_relevant" in test_df.columns:
        X_test = predictor.assemble_features(test_features)
        rel_true = test_df["is_market_relevant"].astype(int).to_numpy()
        rel_proba = predictor.relevance_model.predict_proba(X_test)[:, 1]
        rel_pred = (rel_proba > 0.5).astype(int)
        results["relevance"] = compute_relevance_metrics(rel_true, rel_pred, rel_proba)

    # Global summary
    if rows:
        df_rows = pd.DataFrame(rows)
        results["global_summary"] = {
            "mean_accuracy": float(df_rows["accuracy"].mean()),
            "mean_balanced_accuracy": float(df_rows["balanced_accuracy"].mean()),
            "mean_f1_macro": float(df_rows["f1_macro"].mean()),
            "mean_f1_weighted": float(df_rows["f1_weighted"].mean()),
            "mean_precision_macro": float(df_rows["precision_macro"].mean()),
            "mean_recall_macro": float(df_rows["recall_macro"].mean()),
            "mean_mcc": float(df_rows["mcc"].mean()),
        }

    logger.info("Layer 6 evaluation complete.")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Thread context builder
# ─────────────────────────────────────────────────────────────────────────────

def build_thread_contexts(df: pd.DataFrame, window: int = 5) -> list:
    """
    For each tweet in df (sorted by created_at), build a list of surrounding
    tweets (±window) to provide escalation / topic context to the LLM.

    Returns a list of thread_context lists, one per row in df.
    Each thread_context entry: {"content": str, "created_at": str, "offset": int}
    Offset 0 = target tweet, negative = before, positive = after.
    """
    # Sort by timestamp for correct ordering
    df = df.copy()
    if "created_at" in df.columns:
        df["_sort_ts"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
        df = df.sort_values("_sort_ts").reset_index(drop=True)

    contents = df["content"].fillna("").astype(str).tolist()
    timestamps = df["created_at"].fillna("").astype(str).tolist() \
        if "created_at" in df.columns else [""] * len(df)

    thread_contexts = []
    for i in range(len(df)):
        ctx = []
        for offset in range(-window, window + 1):
            j = i + offset
            if j < 0 or j >= len(df):
                continue
            ctx.append({
                "content": contents[j][:200],
                "created_at": timestamps[j][:16],
                "offset": offset,
            })
        thread_contexts.append(ctx)

    return thread_contexts


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def _save_checkpoint(path: Path, results_per_asset, relevance_true,
                     relevance_pred_lgbm, relevance_pred_llm, skipped, timings, completed):
    """Persist current evaluation state so a crash can be resumed."""
    ckpt = {
        "completed": completed,
        "skipped": skipped,
        "timings": timings,
        "results_per_asset": results_per_asset,
        "relevance_true": relevance_true,
        "relevance_pred_lgbm": relevance_pred_lgbm,
        "relevance_pred_llm": relevance_pred_llm,
    }
    with open(path, "w") as f:
        json.dump(ckpt, f)


# ─────────────────────────────────────────────────────────────────────────────
# Layer 7 evaluation (LLM, sampled test set)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_layer7(
    sample_df: pd.DataFrame,
    model_dir: str,
    timeframe: str,
    use_gpu: bool = True,
    checkpoint_path: Path = None,
    checkpoint_every: int = 10,
    use_thread_context: bool = True,
    architecture_label: str = "new",
    use_tavily: bool = True,
) -> dict:
    """
    Evaluate Layer 7 (GPT-4o) on test tweets.

    Runs the full InferencePipeline (all 7 layers) on each tweet.
    Saves a checkpoint every `checkpoint_every` tweets so a crash mid-run
    can be resumed by re-running with the same --checkpoint-path.
    """
    from inference_pipeline import InferencePipeline

    n = len(sample_df)
    logger.info(f"Evaluating Layer 7 (LLM) on {n} test tweets (timeframe={timeframe}) ...")
    logger.info("Loading full pipeline (may take 30-60s for transformers)...")

    # Respect LLM_USE_TAVILY env var; can be overridden by caller via use_tavily param
    env_tavily = os.getenv("LLM_USE_TAVILY", "true").lower() == "true"
    effective_tavily = use_tavily and env_tavily
    tavily_key = os.getenv("TAVILY_API_KEY") if effective_tavily else None
    logger.info(f"Tavily: {'enabled' if effective_tavily else 'disabled'}")

    pipeline = InferencePipeline.load(
        model_dir,
        use_gpu=use_gpu,
        target_timeframe=timeframe,
        use_tavily_context=effective_tavily,
        tavily_api_key=tavily_key,
        use_llm_reasoning=True,
        llm_use_tavily=effective_tavily,
    )

    results_per_asset = {asset: {"y_true": [], "y_pred_lgbm": [], "y_pred_llm": []}
                         for asset in ASSETS}
    relevance_true = []
    relevance_pred_lgbm = []
    relevance_pred_llm = []
    skipped = 0
    timings = []

    # ── Resume from checkpoint if it exists ──────────────────────────────────
    start_i = 0
    if checkpoint_path is not None and checkpoint_path.exists():
        logger.info(f"Resuming from checkpoint: {checkpoint_path}")
        with open(checkpoint_path) as f:
            ckpt = json.load(f)
        results_per_asset = ckpt["results_per_asset"]
        relevance_true = ckpt["relevance_true"]
        relevance_pred_lgbm = ckpt["relevance_pred_lgbm"]
        relevance_pred_llm = ckpt["relevance_pred_llm"]
        skipped = ckpt["skipped"]
        timings = ckpt["timings"]
        start_i = ckpt["completed"]
        logger.info(f"Resumed at tweet {start_i}/{n} ({skipped} previously skipped)")

    texts = sample_df["content"].fillna("").astype(str).tolist()
    timestamps = sample_df["created_at"].fillna("").astype(str).tolist() \
        if "created_at" in sample_df.columns else [""] * n

    # Build tweet thread context (±5 surrounding tweets for escalation detection)
    if use_thread_context:
        logger.info("Building tweet thread contexts (±5 surrounding tweets per tweet)...")
        thread_contexts = build_thread_contexts(sample_df, window=5)
    else:
        thread_contexts = [None] * n

    for i in range(start_i, n):
        text, ts = texts[i], timestamps[i]
        elapsed_so_far = sum(timings)
        remaining = n - i
        eta_min = (elapsed_so_far / max(len(timings), 1)) * remaining / 60 if timings else 0
        logger.info(
            f"  [{i+1}/{n}] predicting... "
            f"(avg {np.mean(timings):.1f}s/tweet, ETA ~{eta_min:.0f}min)"
            if timings else f"  [{i+1}/{n}] predicting..."
        )
        t0 = time.time()

        try:
            result = pipeline.predict(
                text,
                created_at=ts,
                thread_context=thread_contexts[i],
            )
        except Exception as e:
            logger.warning(f"  Skipping row {i}: {e}")
            skipped += 1
            # Still checkpoint on skips so index advances
            if checkpoint_path is not None and (i + 1) % checkpoint_every == 0:
                _save_checkpoint(checkpoint_path, results_per_asset, relevance_true,
                                 relevance_pred_lgbm, relevance_pred_llm, skipped, timings, i + 1)
            continue

        elapsed = time.time() - t0
        timings.append(elapsed)

        # Ground-truth direction labels
        for asset in ASSETS:
            label_col = f"{asset}_{ASSET_TICKERS[asset]}_actual_dir_{timeframe}"
            if label_col not in sample_df.columns:
                continue
            true_dir = int(sample_df.iloc[i][label_col])
            lgbm_dir = result.predictions.get(asset)
            llm_result = result.llm_reasoning

            if lgbm_dir is None:
                continue

            results_per_asset[asset]["y_true"].append(true_dir)
            results_per_asset[asset]["y_pred_lgbm"].append(lgbm_dir.direction)

            if llm_result is not None and not llm_result.error and asset in llm_result.per_asset:
                results_per_asset[asset]["y_pred_llm"].append(
                    llm_result.per_asset[asset].direction
                )
            else:
                results_per_asset[asset]["y_pred_llm"].append(lgbm_dir.direction)

        # Relevance
        if "is_market_relevant" in sample_df.columns:
            relevance_true.append(int(sample_df.iloc[i]["is_market_relevant"]))
            relevance_pred_lgbm.append(int(result.is_market_relevant))
            if result.llm_reasoning is not None and not result.llm_reasoning.error:
                llm_rel = result.llm_reasoning.llm_is_market_relevant
                if llm_rel is not None:
                    # Use the direct relevance classification from the LLM
                    relevance_pred_llm.append(int(llm_rel))
                else:
                    # Fallback: infer from high-confidence non-zero directions
                    any_high_conf = any(
                        a.confidence >= 0.6 and a.direction != 0
                        for a in result.llm_reasoning.per_asset.values()
                    )
                    relevance_pred_llm.append(1 if any_high_conf else 0)
            else:
                relevance_pred_llm.append(int(result.is_market_relevant))

        # ── Save checkpoint every N tweets ────────────────────────────────
        if checkpoint_path is not None and (i + 1) % checkpoint_every == 0:
            _save_checkpoint(checkpoint_path, results_per_asset, relevance_true,
                             relevance_pred_lgbm, relevance_pred_llm, skipped, timings, i + 1)
            logger.info(
                f"  Checkpoint saved ({i+1}/{n} done, "
                f"{sum(timings)/60:.1f}min elapsed)"
            )

    # Final checkpoint
    if checkpoint_path is not None:
        _save_checkpoint(checkpoint_path, results_per_asset, relevance_true,
                         relevance_pred_lgbm, relevance_pred_llm, skipped, timings, n)

    if timings:
        logger.info(
            f"Layer 7 done: {n - skipped}/{n} succeeded, "
            f"avg latency {np.mean(timings):.1f}s/tweet, "
            f"total {sum(timings)/60:.1f}min"
        )

    # Compile metrics
    asset_metrics = {}
    agreement_rates = {}
    for asset in ASSETS:
        d = results_per_asset[asset]
        if len(d["y_true"]) < 3:
            continue

        y_true = np.array(d["y_true"])
        y_lgbm = np.array(d["y_pred_lgbm"])
        y_llm = np.array(d["y_pred_llm"])

        asset_metrics[asset] = {
            "lgbm": compute_metrics(y_true, y_lgbm),
            "llm": compute_metrics(y_true, y_llm),
        }
        agreement_rates[asset] = float((y_lgbm == y_llm).mean())

    rel_metrics = {}
    if relevance_true:
        rel_metrics["lgbm"] = compute_relevance_metrics(
            np.array(relevance_true), np.array(relevance_pred_lgbm)
        )
        rel_metrics["llm"] = compute_relevance_metrics(
            np.array(relevance_true), np.array(relevance_pred_llm)
        )

    # Global LLM summary
    llm_accs = [v["llm"]["accuracy"] for v in asset_metrics.values()]
    lgbm_accs = [v["lgbm"]["accuracy"] for v in asset_metrics.values()]
    llm_f1s = [v["llm"]["f1_macro"] for v in asset_metrics.values()]
    lgbm_f1s = [v["lgbm"]["f1_macro"] for v in asset_metrics.values()]

    global_summary = {}
    if llm_accs:
        global_summary = {
            "n_samples_evaluated": n - skipped,
            "n_samples_skipped": skipped,
            "mean_latency_s": float(np.mean(timings)) if timings else 0,
            "lgbm_mean_accuracy": float(np.mean(lgbm_accs)),
            "lgbm_mean_f1_macro": float(np.mean(lgbm_f1s)),
            "llm_mean_accuracy": float(np.mean(llm_accs)),
            "llm_mean_f1_macro": float(np.mean(llm_f1s)),
            "mean_lgbm_llm_agreement": float(np.mean(list(agreement_rates.values()))) if agreement_rates else 0,
        }

    return {
        "timeframe": timeframe,
        "architecture": architecture_label,
        "use_thread_context": use_thread_context,
        "per_asset": asset_metrics,
        "agreement_rates": agreement_rates,
        "relevance": rel_metrics,
        "global_summary": global_summary,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Report writer
# ─────────────────────────────────────────────────────────────────────────────

def write_report(
    layer6: dict,
    layer7: dict,
    output_path: Path,
    timeframe: str,
    llm_samples: int,
    total_test: int = 667,
    report_path: Path = None,
    old_layer7: dict = None,
):
    lines = [
        "# Full Pipeline Evaluation Report",
        f"",
        f"Evaluation of the complete 7-layer pipeline on the held-out test set.",
        f"",
        f"- **Layer 6 (LightGBM)** evaluated on: **{layer6.get('per_asset', {}) and next(iter(layer6['per_asset'].values()), {}).get('n_samples', '?')} samples** (full test set)",
        f"- **Layer 7 (GPT-4o LLM)** evaluated on: **{layer7['global_summary'].get('n_samples_evaluated', 0)} samples** "
        f"({'full test set — same 667 as Layer 6' if llm_samples >= total_test else 'random sample'})",
        f"- **Timeframe for LLM evaluation**: `{timeframe}`",
        f"- **Average LLM latency**: {layer7['global_summary'].get('mean_latency_s', 0):.1f}s / tweet",
        f"",
        "---",
        "",
        "## Layer 6 — LightGBM Relevance Classifier (Full Test Set)",
        "",
    ]

    rel6 = layer6.get("relevance", {})
    if rel6:
        lines += [
            f"| Metric | Value |",
            f"|---|---|",
            f"| Accuracy | **{rel6.get('accuracy', 0):.2%}** |",
            f"| ROC-AUC | **{rel6.get('roc_auc', 0):.2%}** |",
            f"| Recall | **{rel6.get('recall', 0):.2%}** |",
            f"| F1 (Weighted) | **{rel6.get('f1_weighted', 0):.2%}** |",
            f"| F1 (Macro) | **{rel6.get('f1_macro', 0):.2%}** |",
            f"| Samples | {rel6.get('n_samples', '?')} |",
            "",
        ]

    lines += [
        "## Layer 6 — LightGBM Direction Classifiers (Full Test Set)",
        "",
        f"Global averages across all {len(ASSETS)} assets and {len(TIMEFRAMES)} timeframes:",
        "",
    ]
    gs6 = layer6.get("global_summary", {})
    if gs6:
        lines += [
            f"| Metric | Value |",
            f"|---|---|",
            f"| Mean Accuracy | {gs6.get('mean_accuracy', 0):.2%} |",
            f"| Mean Balanced Accuracy | {gs6.get('mean_balanced_accuracy', 0):.2%} |",
            f"| Mean F1 (Macro) | {gs6.get('mean_f1_macro', 0):.2%} |",
            f"| Mean F1 (Weighted) | {gs6.get('mean_f1_weighted', 0):.2%} |",
            f"| Mean MCC | {gs6.get('mean_mcc', 0):.4f} |",
            "",
        ]

    lines += [
        "### Per-Asset Results (all timeframes)",
        "",
        "| Asset | TF | Accuracy | Bal. Acc | F1 Macro | F1 Weighted | MCC |",
        "|---|---|---|---|---|---|---|",
    ]
    for key, m in sorted(layer6.get("per_asset", {}).items()):
        lines.append(
            f"| {m['asset']} | {m['timeframe']} | {m['accuracy']:.2%} | "
            f"{m['balanced_accuracy']:.2%} | {m['f1_macro']:.2%} | "
            f"{m['f1_weighted']:.2%} | {m['mcc']:.4f} |"
        )

    lines += [
        "",
        "---",
        "",
        f"## Layer 7 — GPT-4o LLM Reasoning vs LightGBM ({timeframe} horizon, {layer7['global_summary'].get('n_samples_evaluated', 0)} samples)",
        "",
        "### Overall Comparison",
        "",
        f"| Metric | LightGBM (Layer 6) | GPT-4o (Layer 7) |",
        f"|---|---|---|",
    ]

    gs7 = layer7.get("global_summary", {})
    lines += [
        f"| Mean Accuracy | {gs7.get('lgbm_mean_accuracy', 0):.2%} | **{gs7.get('llm_mean_accuracy', 0):.2%}** |",
        f"| Mean F1 (Macro) | {gs7.get('lgbm_mean_f1_macro', 0):.2%} | **{gs7.get('llm_mean_f1_macro', 0):.2%}** |",
        f"| LightGBM–LLM Agreement | — | {gs7.get('mean_lgbm_llm_agreement', 0):.2%} |",
        "",
    ]

    if layer7.get("relevance"):
        rel7_lgbm = layer7["relevance"].get("lgbm", {})
        rel7_llm = layer7["relevance"].get("llm", {})
        lines += [
            "### Relevance Classification (LLM sample)",
            "",
            "| Metric | LightGBM | GPT-4o |",
            "|---|---|---|",
            f"| Accuracy | {rel7_lgbm.get('accuracy', 0):.2%} | **{rel7_llm.get('accuracy', 0):.2%}** |",
            f"| Recall | {rel7_lgbm.get('recall', 0):.2%} | **{rel7_llm.get('recall', 0):.2%}** |",
            f"| F1 (Weighted) | {rel7_lgbm.get('f1_weighted', 0):.2%} | **{rel7_llm.get('f1_weighted', 0):.2%}** |",
            "",
        ]

    lines += [
        "### Per-Asset Direction Comparison",
        "",
    ]

    has_old = old_layer7 is not None and old_layer7.get("per_asset")
    new_arch = layer7.get("architecture", "new (thread + yfinance)")
    old_arch = old_layer7.get("architecture", "old LLM") if has_old else "old LLM"

    if has_old:
        lines += [
            f"3-way comparison: **LGBM** vs **{old_arch}** vs **{new_arch}**",
            "",
            f"| Asset | LGBM Acc | {old_arch} Acc | {new_arch} Acc | LGBM F1 | {old_arch} F1 | {new_arch} F1 | Δ Acc (new-old) |",
            f"|---|---|---|---|---|---|---|---|",
        ]
        for asset in ASSETS:
            if asset not in layer7.get("per_asset", {}):
                continue
            m_new = layer7["per_asset"][asset]
            m_old = old_layer7["per_asset"].get(asset, {})
            lgbm_acc   = m_new["lgbm"]["accuracy"]
            new_acc    = m_new["llm"]["accuracy"]
            old_acc    = m_old.get("llm", {}).get("accuracy", float("nan"))
            lgbm_f1    = m_new["lgbm"]["f1_macro"]
            new_f1     = m_new["llm"]["f1_macro"]
            old_f1     = m_old.get("llm", {}).get("f1_macro", float("nan"))
            import math
            delta      = new_acc - old_acc if not math.isnan(old_acc) else float("nan")
            delta_str  = f"{delta:+.2%}" if not math.isnan(delta) else "—"
            old_acc_s  = f"{old_acc:.2%}" if not math.isnan(old_acc) else "—"
            old_f1_s   = f"{old_f1:.2%}" if not math.isnan(old_f1) else "—"
            lines.append(
                f"| {asset} | {lgbm_acc:.2%} | {old_acc_s} | **{new_acc:.2%}** | "
                f"{lgbm_f1:.2%} | {old_f1_s} | **{new_f1:.2%}** | {delta_str} |"
            )
        # Average row
        new_accs = [layer7["per_asset"][a]["llm"]["accuracy"] for a in ASSETS if a in layer7["per_asset"]]
        old_accs = [old_layer7["per_asset"].get(a, {}).get("llm", {}).get("accuracy", float("nan"))
                    for a in ASSETS if a in layer7["per_asset"]]
        lgbm_accs_row = [layer7["per_asset"][a]["lgbm"]["accuracy"] for a in ASSETS if a in layer7["per_asset"]]
        new_f1s = [layer7["per_asset"][a]["llm"]["f1_macro"] for a in ASSETS if a in layer7["per_asset"]]
        old_f1s_list = [old_layer7["per_asset"].get(a, {}).get("llm", {}).get("f1_macro", float("nan"))
                        for a in ASSETS if a in layer7["per_asset"]]
        import math
        avg_old_acc = float(np.nanmean(old_accs)) if old_accs else float("nan")
        avg_new_acc = float(np.mean(new_accs)) if new_accs else float("nan")
        avg_lgbm_acc = float(np.mean(lgbm_accs_row)) if lgbm_accs_row else float("nan")
        avg_old_f1  = float(np.nanmean(old_f1s_list)) if old_f1s_list else float("nan")
        avg_new_f1  = float(np.mean(new_f1s)) if new_f1s else float("nan")
        avg_lgbm_f1 = float(np.mean([layer7["per_asset"][a]["lgbm"]["f1_macro"] for a in ASSETS if a in layer7["per_asset"]])) if new_accs else float("nan")
        avg_delta   = avg_new_acc - avg_old_acc if not math.isnan(avg_old_acc) else float("nan")
        delta_avg_s = f"{avg_delta:+.2%}" if not math.isnan(avg_delta) else "—"
        old_avg_s   = f"{avg_old_acc:.2%}" if not math.isnan(avg_old_acc) else "—"
        old_f1_avg_s = f"{avg_old_f1:.2%}" if not math.isnan(avg_old_f1) else "—"
        lines.append(
            f"| **Average** | **{avg_lgbm_acc:.2%}** | {old_avg_s} | **{avg_new_acc:.2%}** | "
            f"**{avg_lgbm_f1:.2%}** | {old_f1_avg_s} | **{avg_new_f1:.2%}** | {delta_avg_s} |"
        )
    else:
        lines += [
            f"| Asset | LGBM Acc | LLM Acc | LGBM F1 | LLM F1 | Agreement |",
            f"|---|---|---|---|---|---|",
        ]
        for asset in ASSETS:
            if asset not in layer7.get("per_asset", {}):
                continue
            m = layer7["per_asset"][asset]
            agr = layer7["agreement_rates"].get(asset, 0)
            lines.append(
                f"| {asset} | {m['lgbm']['accuracy']:.2%} | **{m['llm']['accuracy']:.2%}** | "
                f"{m['lgbm']['f1_macro']:.2%} | **{m['llm']['f1_macro']:.2%}** | {agr:.2%} |"
            )

    lines += [
        "",
        "---",
        "",
        "## Notes",
        "",
        "- Layer 6 (LightGBM) evaluates all 21 classifiers (7 assets × 3 timeframes) on the full 667-sample test set.",
        f"- Layer 7 (GPT-4o) evaluates on {layer7['global_summary'].get('n_samples_evaluated', 0)} "
        f"{'test tweets (same full set as Layer 6, enabling direct comparison).' if llm_samples >= total_test else 'randomly sampled test tweets.'}",
        "- LLM accuracy reflects GPT-4o's final direction verdict compared to ground-truth labels.",
        "- Agreement rate measures how often LLM and LightGBM predict the same direction (not necessarily correct).",
        "- LLM results are for the specified timeframe only; LightGBM is evaluated across all three timeframes.",
        "- Where the LLM call failed for a sample, LightGBM's prediction is used as fallback.",
    ]

    path = report_path if report_path is not None else output_path / "llm_eval_report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    logger.info(f"Report written to {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate full pipeline (Layer 6 + Layer 7) on held-out test set"
    )
    parser.add_argument("--data", default="data/train.csv",
                        help="Original training CSV (used to reconstruct test split)")
    parser.add_argument("--model-dir", default="saved_models",
                        help="Trained model directory")
    parser.add_argument("--output-dir", default="saved_models",
                        help="Directory to write evaluation outputs")
    parser.add_argument("--timeframe", default="5m", choices=["1m", "5m", "10m"],
                        help="Timeframe for Layer 7 (LLM) evaluation")
    parser.add_argument("--llm-samples", type=int, default=50,
                        help="Number of test samples to evaluate with Layer 7 LLM (0 = skip LLM)")
    parser.add_argument("--llm-use-tavily", action="store_true",
                        help="Fetch real-time Tavily context for Layer 7 LLM (already on by default in eval)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for the markdown report (overrides --output-dir default location)")
    parser.add_argument("--no-gpu", action="store_true", help="Force CPU")
    parser.add_argument("--random-seed", type=int, default=42,
                        help="Random seed for sampling LLM evaluation subset")
    parser.add_argument("--split-method", default="time", choices=["time", "random"])
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--no-cache", action="store_true",
                        help="Re-extract Layer 6 features instead of using cache")
    parser.add_argument("--checkpoint-path", type=str,
                        default="saved_models/llm_eval_checkpoint.json",
                        help="Path to checkpoint file (auto-saved every 10 tweets, "
                             "re-run same command to resume if interrupted)")
    parser.add_argument("--checkpoint-every", type=int, default=10,
                        help="Save checkpoint every N tweets (default: 10)")
    parser.add_argument("--old-metrics", type=str, default=None,
                        help="Path to a previous llm_eval_metrics.json to show side-by-side comparison "
                             "of old LLM (without thread/yfinance) vs new LLM in the report")
    parser.add_argument("--no-thread-context", action="store_true",
                        help="Disable tweet thread context (runs old-style LLM evaluation for comparison baseline)")
    parser.add_argument("--no-tavily", action="store_true",
                        help="Disable Tavily for this run (overrides LLM_USE_TAVILY=true in .env)")
    args = parser.parse_args()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load old metrics for comparison (optional)
    old_layer7_results = None
    if args.old_metrics:
        old_metrics_path = Path(args.old_metrics)
        if old_metrics_path.exists():
            with open(old_metrics_path) as f:
                old_all = json.load(f)
            old_layer7_results = old_all.get("layer7", None)
            if old_layer7_results:
                old_layer7_results.setdefault("architecture", "old LLM (no thread/yfinance)")
            logger.info(f"Loaded old metrics from {old_metrics_path} for comparison")
        else:
            logger.warning(f"--old-metrics path not found: {old_metrics_path}")

    # ── Load and split data ──────────────────────────────────────────────────
    logger.info(f"Loading data from {args.data}")
    df = pd.read_csv(args.data)
    logger.info(f"Loaded {len(df)} rows")

    _, test_df = split_train_test_df(
        df=df,
        test_size=args.test_size,
        split_method=args.split_method,
    )
    logger.info(f"Test set: {len(test_df)} rows")

    # ── Load Layer 6 (LightGBM) models ──────────────────────────────────────
    logger.info("Loading LightGBM models from saved_models/impact_predictor/...")
    predictor = MarketImpactPredictor()
    predictor.load(str(Path(args.model_dir) / "impact_predictor"))

    # ── Layer 6 features ────────────────────────────────────────────────────
    cache_path = Path(args.model_dir) / "features_cache_test.json"
    test_features = None

    if not args.no_cache and cache_path.exists():
        logger.info(f"Loading cached test features from {cache_path}")
        with open(cache_path) as f:
            test_features = json.load(f)
        if len(test_features) != len(test_df):
            logger.warning(
                f"Cache size mismatch ({len(test_features)} vs {len(test_df)}). "
                "Re-extracting features..."
            )
            test_features = None

    if test_features is None:
        logger.info("Extracting Layer 6 features from scratch (this takes ~9 minutes)...")
        from train_pipeline import extract_feature_dicts
        from models.ner.entity_extractor import EntityExtractor
        from models.sentiment.finbert_sentiment import FinBERTSentiment
        from models.events.event_detector import EventDetector
        from models.events.context_enrichment import ContextEnricher
        from models.graph_reasoning.entity_graph import EntityGraph

        ner = EntityExtractor(use_gpu=not args.no_gpu); ner.load()
        sentiment = FinBERTSentiment(use_gpu=not args.no_gpu); sentiment.load()
        event_det = EventDetector(use_gpu=not args.no_gpu); event_det.load()
        event_head = Path(args.model_dir) / "event_head.pt"
        if event_head.exists():
            event_det.load_head(str(event_head))
        context_enricher = ContextEnricher()
        graph_reasoner = EntityGraph()

        test_features = extract_feature_dicts(
            texts=test_df["content"].fillna("").astype(str).tolist(),
            timestamps=test_df["created_at"].fillna("").astype(str).tolist()
            if "created_at" in test_df.columns else [""] * len(test_df),
            ner=ner, sentiment=sentiment, event_det=event_det,
            context_enricher=context_enricher, graph_reasoner=graph_reasoner,
            use_trained_event_head=True,
            cache_path=cache_path,
            cache_features=True,
            desc="Feature extraction (test)",
        )

    # ── Layer 6 Evaluation ───────────────────────────────────────────────────
    layer6_results = evaluate_layer6(
        predictor=predictor,
        test_features=test_features,
        test_df=test_df,
        timeframes=TIMEFRAMES,
    )

    # Print Layer 6 summary
    print("\n" + "=" * 80)
    print("LAYER 6 (LIGHTGBM) — FULL TEST SET RESULTS")
    print("=" * 80)
    rel6 = layer6_results.get("relevance", {})
    if rel6:
        print(f"\nRelevance Classifier ({rel6.get('n_samples', '?')} samples):")
        print(f"  Accuracy:  {rel6['accuracy']:.4f}  ({rel6['accuracy']:.2%})")
        print(f"  Recall:    {rel6['recall']:.4f}  ({rel6['recall']:.2%})")
        print(f"  F1 Wtd:    {rel6['f1_weighted']:.4f}  ({rel6['f1_weighted']:.2%})")
        if "roc_auc" in rel6:
            print(f"  ROC-AUC:   {rel6['roc_auc']:.4f}  ({rel6['roc_auc']:.2%})")

    gs6 = layer6_results.get("global_summary", {})
    if gs6:
        print(f"\nDirection classifiers (global averages across {len(ASSETS)} assets × {len(TIMEFRAMES)} timeframes):")
        print(f"  Mean Accuracy:          {gs6['mean_accuracy']:.4f}  ({gs6['mean_accuracy']:.2%})")
        print(f"  Mean Balanced Accuracy: {gs6['mean_balanced_accuracy']:.4f}  ({gs6['mean_balanced_accuracy']:.2%})")
        print(f"  Mean F1 (Macro):        {gs6['mean_f1_macro']:.4f}  ({gs6['mean_f1_macro']:.2%})")
        print(f"  Mean F1 (Weighted):     {gs6['mean_f1_weighted']:.4f}  ({gs6['mean_f1_weighted']:.2%})")
        print(f"  Mean MCC:               {gs6['mean_mcc']:.4f}")

    print(f"\n{'Asset':<14} {'TF':<4} {'Accuracy':>10} {'Bal.Acc':>10} {'F1(Macro)':>11} {'F1(Wtd)':>10} {'MCC':>8}")
    print("-" * 80)
    for key in sorted(layer6_results["per_asset"].keys()):
        m = layer6_results["per_asset"][key]
        print(f"{m['asset']:<14} {m['timeframe']:<4} {m['accuracy']:>10.4f} "
              f"{m['balanced_accuracy']:>10.4f} {m['f1_macro']:>11.4f} "
              f"{m['f1_weighted']:>10.4f} {m['mcc']:>8.4f}")

    # ── Layer 7 Evaluation ───────────────────────────────────────────────────
    layer7_results = {"global_summary": {}, "per_asset": {}, "agreement_rates": {}, "relevance": {}}

    if args.llm_samples > 0:
        if args.llm_samples >= len(test_df):
            # Full test set — preserve temporal order
            sample_df = test_df.reset_index(drop=True)
            logger.info(f"\nUsing full test set ({len(sample_df)} tweets) for Layer 7 LLM evaluation.")
        else:
            random.seed(args.random_seed)
            sample_indices = random.sample(range(len(test_df)), args.llm_samples)
            sample_df = test_df.iloc[sample_indices].reset_index(drop=True)
            logger.info(f"\nSampled {len(sample_df)} tweets for Layer 7 LLM evaluation.")

        logger.info(f"Estimated time: {len(sample_df) * 4 / 60:.0f}–{len(sample_df) * 6 / 60:.0f} minutes")

        use_thread = not args.no_thread_context
        use_tavily = not getattr(args, "no_tavily", False)
        arch_parts = ["thread", "yfinance"] if use_thread else ["yfinance"]
        if use_tavily:
            arch_parts.append("tavily")
        arch_label = "new (" + " + ".join(arch_parts) + ")" if use_thread else "old LLM (no thread/yfinance)"
        layer7_results = evaluate_layer7(
            sample_df=sample_df,
            model_dir=args.model_dir,
            timeframe=args.timeframe,
            use_gpu=not args.no_gpu,
            checkpoint_path=Path(args.checkpoint_path),
            checkpoint_every=args.checkpoint_every,
            use_thread_context=use_thread,
            architecture_label=arch_label,
            use_tavily=use_tavily,
        )

        # Print Layer 7 summary
        print("\n" + "=" * 80)
        print(f"LAYER 7 (GPT-4o LLM) vs LAYER 6 — {args.timeframe.upper()} HORIZON")
        print(f"Evaluated on {layer7_results['global_summary'].get('n_samples_evaluated', 0)} sampled test tweets")
        print("=" * 80)

        gs7 = layer7_results.get("global_summary", {})
        if gs7:
            print(f"\n  {'Metric':<32} {'LightGBM':>12} {'GPT-4o':>12}")
            print(f"  {'-'*56}")
            print(f"  {'Mean Accuracy':<32} {gs7['lgbm_mean_accuracy']:>12.2%} {gs7['llm_mean_accuracy']:>12.2%}")
            print(f"  {'Mean F1 (Macro)':<32} {gs7['lgbm_mean_f1_macro']:>12.2%} {gs7['llm_mean_f1_macro']:>12.2%}")
            print(f"  {'LightGBM–LLM Agreement':<32} {'—':>12} {gs7['mean_lgbm_llm_agreement']:>12.2%}")

        print(f"\n{'Asset':<14} {'LGBM Acc':>10} {'LLM Acc':>10} {'LGBM F1':>10} {'LLM F1':>10} {'Agreement':>12}")
        print("-" * 70)
        for asset in ASSETS:
            if asset not in layer7_results.get("per_asset", {}):
                continue
            m = layer7_results["per_asset"][asset]
            agr = layer7_results["agreement_rates"].get(asset, 0)
            print(f"{asset:<14} {m['lgbm']['accuracy']:>10.4f} {m['llm']['accuracy']:>10.4f} "
                  f"{m['lgbm']['f1_macro']:>10.4f} {m['llm']['f1_macro']:>10.4f} {agr:>12.2%}")
    else:
        logger.info("--llm-samples 0: skipping Layer 7 evaluation.")

    # ── Save outputs ─────────────────────────────────────────────────────────
    all_metrics = {
        "layer6": layer6_results,
        "layer7": layer7_results,
        "config": {
            "data": args.data,
            "model_dir": args.model_dir,
            "timeframe": args.timeframe,
            "llm_samples": args.llm_samples,
            "test_size": args.test_size,
            "split_method": args.split_method,
        },
    }

    metrics_path = output_path / "llm_eval_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)
    logger.info(f"Metrics saved to {metrics_path}")

    write_report(
        layer6=layer6_results,
        layer7=layer7_results,
        output_path=output_path,
        timeframe=args.timeframe,
        llm_samples=args.llm_samples,
        total_test=len(test_df),
        report_path=Path(args.output) if args.output else None,
        old_layer7=old_layer7_results,
    )

    print(f"\nOutputs saved to {output_path}/")
    print(f"  llm_eval_metrics.json")
    print(f"  llm_eval_report.md")


if __name__ == "__main__":
    main()
