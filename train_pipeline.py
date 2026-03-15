"""
Trump Tweet Market Impact Classifier — Training Script.

Loads train.csv, runs all feature extraction layers, trains LightGBM
classifiers for each asset/timeframe, evaluates, and saves models.

Usage:
    python train_pipeline.py --data data/train.csv --output saved_models/
    python train_pipeline.py --data data/train.csv --output saved_models/ --no-gpu
    python train_pipeline.py --data data/train.csv --output saved_models/ --test-size 0.2 --split-method time
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from models.ner.entity_extractor import EntityExtractor
from models.sentiment.finbert_sentiment import FinBERTSentiment
from models.events.event_detector import EventDetector, EVENT_TYPES, EVENT_PATTERNS
from models.events.context_enrichment import ContextEnricher
from models.graph_reasoning.entity_graph import EntityGraph
from models.market_impact.impact_predictor import (
    MarketImpactPredictor, ASSETS, TIMEFRAMES, ASSET_TICKERS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_pipeline")


def split_train_test_df(
    df: pd.DataFrame,
    test_size: float = 0.2,
    split_method: str = "time",
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into train/test with strict holdout.

    split_method:
      - "time": oldest rows for train, newest rows for test (preferred)
      - "random": shuffled split (fallback)
    """
    if not (0.0 < test_size < 1.0):
        raise ValueError(f"test_size must be in (0,1), got {test_size}")

    n = len(df)
    if n < 2:
        raise ValueError("Need at least 2 rows to create train/test split")

    n_test = max(1, int(round(n * test_size)))
    n_test = min(n_test, n - 1)

    if split_method == "time":
        if "created_at" in df.columns:
            tmp = df.copy()
            tmp["_sort_ts"] = pd.to_datetime(tmp["created_at"], errors="coerce", utc=True)
            tmp = tmp.sort_values("_sort_ts", kind="mergesort", na_position="last")
            tmp = tmp.drop(columns=["_sort_ts"])
        else:
            logger.warning("split_method='time' requested but 'created_at' not found; using current row order")
            tmp = df.copy()

        train_df = tmp.iloc[:-n_test].reset_index(drop=True)
        test_df = tmp.iloc[-n_test:].reset_index(drop=True)
        return train_df, test_df

    if split_method == "random":
        shuffled = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
        train_df = shuffled.iloc[:-n_test].reset_index(drop=True)
        test_df = shuffled.iloc[-n_test:].reset_index(drop=True)
        return train_df, test_df

    raise ValueError(f"Unknown split_method: {split_method}")


def extract_feature_dicts(
    texts: list,
    timestamps: list,
    ner: EntityExtractor,
    sentiment: FinBERTSentiment,
    event_det: EventDetector,
    context_enricher: ContextEnricher,
    graph_reasoner: EntityGraph,
    use_trained_event_head: bool,
    cache_path: Optional[Path] = None,
    cache_features: bool = False,
    desc: str = "Feature extraction",
) -> list:
    """Extract full multi-layer feature dicts for each text."""
    if cache_features and cache_path is not None and cache_path.exists():
        logger.info(f"Loading cached features from {cache_path}")
        with open(cache_path) as f:
            cached = json.load(f)
        if len(cached) == len(texts):
            logger.info(f"Loaded {len(cached)} cached feature dicts")
            return cached
        logger.warning(
            f"Ignoring cache {cache_path} due to row mismatch: "
            f"cached={len(cached)} current={len(texts)}"
        )

    t0 = time.time()
    try:
        from tqdm.auto import tqdm
        iterator = tqdm(range(len(texts)), desc=desc)
    except ImportError:
        iterator = range(len(texts))

    all_features = []
    for i in iterator:
        text = texts[i]
        ts = timestamps[i]
        prev_ts = timestamps[i - 1] if i > 0 else None

        features = {}

        ner_result = ner.extract(text)
        features.update(ner_result.to_feature_dict())

        sent_result = sentiment.analyze(text)
        features.update(sent_result.to_feature_dict())

        if use_trained_event_head:
            event_result = event_det.detect(text)
        else:
            event_result = event_det.detect_rules(text)
        features.update(event_result.to_feature_dict())

        ctx_result = context_enricher.enrich(
            text=text, created_at=ts, prev_tweet_time=prev_ts,
        )
        features.update(ctx_result.to_feature_dict())

        try:
            graph_result = graph_reasoner.build_and_reason(
                entities=ner_result.entities,
                events=event_result.events,
                sentiment_compound=sent_result.compound,
            )
            features.update(graph_result.to_feature_dict())
        except Exception as e:
            logger.debug(f"Graph reasoning failed for row {i}: {e}")

        all_features.append(features)

    elapsed = time.time() - t0
    logger.info(f"{desc}: {elapsed:.1f}s ({elapsed/len(texts)*1000:.0f}ms/tweet)")

    if cache_features and cache_path is not None:
        with open(cache_path, "w") as f:
            json.dump(all_features, f)
        logger.info(f"Features cached to {cache_path}")

    return all_features


def evaluate_on_test_set(
    predictor: MarketImpactPredictor,
    test_features: list,
    test_df: pd.DataFrame,
    target_timeframes: list,
) -> tuple[dict, pd.DataFrame]:
    """Evaluate trained predictor on strict held-out test set."""
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        confusion_matrix,
        balanced_accuracy_score,
        matthews_corrcoef,
        cohen_kappa_score,
        classification_report,
    )

    rows = []
    report = {
        "n_test": int(len(test_df)),
        "timeframes": {},
    }

    for tf in target_timeframes:
        tf_preds = []
        for fd in test_features:
            preds, _, _ = predictor.predict(fd, timeframe=tf)
            tf_preds.append(preds)

        report["timeframes"][tf] = {}
        for asset in ASSETS:
            label_col = f"{asset}_{ASSET_TICKERS[asset]}_actual_dir_{tf}"
            if label_col not in test_df.columns:
                continue

            y_true = test_df[label_col].astype(int).to_numpy()
            y_pred = np.array([preds[asset].direction for preds in tf_preds], dtype=int)

            acc = float(accuracy_score(y_true, y_pred))
            bal_acc = float(balanced_accuracy_score(y_true, y_pred))
            f1_macro = float(f1_score(y_true, y_pred, average="macro"))
            f1_weighted = float(f1_score(y_true, y_pred, average="weighted"))
            precision_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
            precision_weighted = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
            recall_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
            recall_weighted = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
            mcc = float(matthews_corrcoef(y_true, y_pred))
            kappa = float(cohen_kappa_score(y_true, y_pred))
            cm = confusion_matrix(y_true, y_pred, labels=[-1, 0, 1]).tolist()
            class_report = classification_report(
                y_true,
                y_pred,
                labels=[-1, 0, 1],
                target_names=["bearish(-1)", "flat(0)", "bullish(1)"],
                zero_division=0,
                output_dict=True,
            )
            class_dist = {
                "-1": int((y_true == -1).sum()),
                "0": int((y_true == 0).sum()),
                "1": int((y_true == 1).sum()),
            }

            report["timeframes"][tf][asset] = {
                "accuracy": acc,
                "balanced_accuracy": bal_acc,
                "f1_macro": f1_macro,
                "f1_weighted": f1_weighted,
                "precision_macro": precision_macro,
                "precision_weighted": precision_weighted,
                "recall_macro": recall_macro,
                "recall_weighted": recall_weighted,
                "mcc": mcc,
                "kappa": kappa,
                "class_dist": class_dist,
                "confusion_matrix": cm,
                "classification_report": class_report,
            }

            rows.append({
                "asset": asset,
                "timeframe": tf,
                "accuracy": acc,
                "balanced_accuracy": bal_acc,
                "f1_macro": f1_macro,
                "f1_weighted": f1_weighted,
                "precision_macro": precision_macro,
                "precision_weighted": precision_weighted,
                "recall_macro": recall_macro,
                "recall_weighted": recall_weighted,
                "mcc": mcc,
                "kappa": kappa,
                "support": int(len(y_true)),
                "class_dist": str(class_dist),
            })

    if predictor.relevance_model is not None and "is_market_relevant" in test_df.columns:
        try:
            from sklearn.metrics import roc_auc_score
        except ImportError:
            roc_auc_score = None

        X_test = predictor.assemble_features(test_features)
        rel_true = test_df["is_market_relevant"].astype(int).to_numpy()
        rel_proba = predictor.relevance_model.predict_proba(X_test)[:, 1]
        rel_pred = (rel_proba > 0.5).astype(int)

        rel_report = {
            "accuracy": float(accuracy_score(rel_true, rel_pred)),
            "precision": float(precision_score(rel_true, rel_pred, zero_division=0)),
            "recall": float(recall_score(rel_true, rel_pred, zero_division=0)),
            "f1_weighted": float(f1_score(rel_true, rel_pred, average="weighted")),
            "f1_macro": float(f1_score(rel_true, rel_pred, average="macro")),
        }
        if roc_auc_score is not None and len(np.unique(rel_true)) > 1:
            rel_report["roc_auc"] = float(roc_auc_score(rel_true, rel_proba))

        report["relevance"] = rel_report

    report_df = pd.DataFrame(rows)

    if not report_df.empty:
        report["global_summary"] = {
            "mean_accuracy": float(report_df["accuracy"].mean()),
            "mean_balanced_accuracy": float(report_df["balanced_accuracy"].mean()),
            "mean_f1_macro": float(report_df["f1_macro"].mean()),
            "mean_f1_weighted": float(report_df["f1_weighted"].mean()),
            "mean_precision_macro": float(report_df["precision_macro"].mean()),
            "mean_recall_macro": float(report_df["recall_macro"].mean()),
            "mean_mcc": float(report_df["mcc"].mean()),
            "mean_kappa": float(report_df["kappa"].mean()),
        }

    return report, report_df


def write_test_report_markdown(
    test_report: dict,
    test_report_df: pd.DataFrame,
    output_path: Path,
    min_accuracy: float = 0.70,
):
    """Write a readable markdown report for held-out test metrics."""
    lines = [
        "# Held-out Test Report",
        "",
        f"- Test samples: {test_report.get('n_test', 0)}",
        "",
        "## Global Summary",
    ]

    summary = test_report.get("global_summary", {})
    if summary:
        lines.extend([
            f"- Mean Accuracy: {summary.get('mean_accuracy', 0.0):.4f}",
            f"- Mean Balanced Accuracy: {summary.get('mean_balanced_accuracy', 0.0):.4f}",
            f"- Mean F1 Macro: {summary.get('mean_f1_macro', 0.0):.4f}",
            f"- Mean F1 Weighted: {summary.get('mean_f1_weighted', 0.0):.4f}",
            f"- Mean Precision Macro: {summary.get('mean_precision_macro', 0.0):.4f}",
            f"- Mean Recall Macro: {summary.get('mean_recall_macro', 0.0):.4f}",
            f"- Mean MCC: {summary.get('mean_mcc', 0.0):.4f}",
            f"- Mean Kappa: {summary.get('mean_kappa', 0.0):.4f}",
        ])
    else:
        lines.append("- No test metrics found.")

    lines.extend([
        "",
        f"## Per Asset / Timeframe (Acc >= {min_accuracy:.2f})",
        "",
        "| Asset | TF | Acc | Bal Acc | F1 Macro | F1 Weighted | Prec Macro | Recall Macro | MCC | Kappa |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])

    filtered_df = test_report_df[test_report_df["accuracy"] >= min_accuracy].copy()
    if not filtered_df.empty:
        for _, row in filtered_df.iterrows():
            lines.append(
                f"| {row['asset']} | {row['timeframe']} | {row['accuracy']:.4f} | "
                f"{row['balanced_accuracy']:.4f} | {row['f1_macro']:.4f} | {row['f1_weighted']:.4f} | "
                f"{row['precision_macro']:.4f} | {row['recall_macro']:.4f} | {row['mcc']:.4f} | {row['kappa']:.4f} |"
            )
    else:
        lines.append("| None | - | - | - | - | - | - | - | - | - |")

    relevance = test_report.get("relevance")
    if relevance and relevance.get("accuracy", 0.0) >= min_accuracy:
        lines.extend([
            "",
            "## Relevance Classifier",
            "",
            f"- Accuracy: {relevance.get('accuracy', 0.0):.4f}",
            f"- Precision: {relevance.get('precision', 0.0):.4f}",
            f"- Recall: {relevance.get('recall', 0.0):.4f}",
            f"- F1 Weighted: {relevance.get('f1_weighted', 0.0):.4f}",
            f"- F1 Macro: {relevance.get('f1_macro', 0.0):.4f}",
        ])
        if "roc_auc" in relevance:
            lines.append(f"- ROC-AUC: {relevance['roc_auc']:.4f}")

    report_path = output_path / "test_report.md"
    report_path.write_text("\n".join(lines))


def generate_event_labels(texts: list) -> np.ndarray:
    """
    Generate multi-label event annotations from rule-based patterns.
    Used to train the event detection head.

    Returns: (N, num_events) binary array.
    """
    import re
    labels = np.zeros((len(texts), len(EVENT_TYPES)), dtype=np.float32)

    for i, text in enumerate(texts):
        text_lower = text.lower()
        any_event = False
        for j, event_type in enumerate(EVENT_TYPES):
            if event_type == "no_event":
                continue
            patterns = EVENT_PATTERNS.get(event_type, [])
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    labels[i, j] = 1.0
                    any_event = True
                    break
        if not any_event:
            labels[i, EVENT_TYPES.index("no_event")] = 1.0

    return labels


def train(
    data_path: str,
    output_dir: str,
    use_gpu: bool = True,
    target_timeframes: list = None,
    cache_features: bool = True,
    test_size: float = 0.2,
    split_method: str = "time",
    random_state: int = 42,
    tavily_api_key: Optional[str] = None,
    use_tavily_context: bool = False,
):
    """
    Full training pipeline.

    Steps:
        1. Load and validate training data
        2. Load models (GLiNER, FinBERT, DistilBERT)
        3. Extract features from all layers
        4. Train event detection head
        5. Re-extract event features with trained head
        6. Train LightGBM market impact classifiers
        7. Evaluate and save
    """
    if target_timeframes is None:
        target_timeframes = TIMEFRAMES

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ─────────────────────────────────────────────
    # Step 1: Load training data
    # ─────────────────────────────────────────────
    logger.info(f"Loading training data from {data_path}")
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    # Validate required columns
    required_label_cols = []
    for asset in ASSETS:
        ticker = ASSET_TICKERS[asset]
        for tf in target_timeframes:
            col = f"{asset}_{ticker}_actual_dir_{tf}"
            required_label_cols.append(col)
            if col not in df.columns:
                logger.error(f"Missing label column: {col}")
                raise ValueError(f"Missing label column: {col}")

    texts = df["content"].fillna("").astype(str).tolist()
    timestamps = df["created_at"].fillna("").astype(str).tolist()
    train_df, test_df = split_train_test_df(
        df=df,
        test_size=test_size,
        split_method=split_method,
        random_state=random_state,
    )

    logger.info(
        f"Split complete ({split_method}): train={len(train_df)} ({len(train_df)/len(df):.1%}), "
        f"test={len(test_df)} ({len(test_df)/len(df):.1%})"
    )

    split_summary = {
        "split_method": split_method,
        "test_size": test_size,
        "random_state": random_state,
        "n_total": int(len(df)),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
    }
    with open(output_path / "data_split_summary.json", "w") as f:
        json.dump(split_summary, f, indent=2)

    texts = train_df["content"].fillna("").astype(str).tolist()
    timestamps = train_df["created_at"].fillna("").astype(str).tolist()
    if timestamps:
        logger.info(f"Train date range: {timestamps[0][:10]} to {timestamps[-1][:10]}")

    # ─────────────────────────────────────────────
    # Step 2: Load base models
    # ─────────────────────────────────────────────
    logger.info("Loading base models...")

    ner = EntityExtractor(use_gpu=use_gpu)
    ner.load()

    sentiment = FinBERTSentiment(use_gpu=use_gpu)
    sentiment.load()

    event_det = EventDetector(use_gpu=use_gpu)
    event_det.load()

    context_enricher = ContextEnricher(
        tavily_api_key=tavily_api_key,
        enable_live_search=use_tavily_context,
    )
    graph_reasoner = EntityGraph()

    cache_suffix = "_tavily" if use_tavily_context else ""

    # ─────────────────────────────────────────────
    # Step 3: Extract train features
    # ─────────────────────────────────────────────
    all_features = extract_feature_dicts(
        texts=texts,
        timestamps=timestamps,
        ner=ner,
        sentiment=sentiment,
        event_det=event_det,
        context_enricher=context_enricher,
        graph_reasoner=graph_reasoner,
        use_trained_event_head=False,
        cache_path=output_path / f"features_cache_train{cache_suffix}.json",
        cache_features=cache_features,
        desc="Feature extraction (train)",
    )

    # ─────────────────────────────────────────────
    # Step 4: Train event detection head
    # ─────────────────────────────────────────────
    logger.info("Generating event labels for head training...")
    event_labels = generate_event_labels(texts)
    event_dist = {EVENT_TYPES[j]: int(event_labels[:, j].sum()) for j in range(len(EVENT_TYPES))}
    logger.info(f"Event label distribution: {event_dist}")

    logger.info("Training event detection head...")
    event_det.train_head(texts, event_labels, epochs=15, lr=2e-4)
    event_det.save(str(output_path / "event_head.pt"))

    # ─────────────────────────────────────────────
    # Step 5: Re-extract event features with trained head
    # ─────────────────────────────────────────────
    logger.info("Re-extracting event features with trained head...")
    try:
        from tqdm.auto import tqdm
        iterator = tqdm(range(len(texts)), desc="Re-extract events")
    except ImportError:
        iterator = range(len(texts))

    for i in iterator:
        event_result = event_det.detect(texts[i])
        # Update event features in the feature dicts
        event_features = event_result.to_feature_dict()
        all_features[i].update(event_features)

    # ─────────────────────────────────────────────
    # Step 6: Train market impact classifiers
    # ─────────────────────────────────────────────
    logger.info("Training market impact classifiers...")
    predictor = MarketImpactPredictor(target_timeframe="5m")

    metrics = predictor.train(
        feature_dicts=all_features,
        labels_df=train_df,
        timeframes=target_timeframes,
        n_splits=5,
    )

    # ─────────────────────────────────────────────
    # Step 7: Evaluate on held-out test set
    # ─────────────────────────────────────────────
    logger.info("Extracting test features and running held-out evaluation...")
    test_texts = test_df["content"].fillna("").astype(str).tolist()
    test_timestamps = test_df["created_at"].fillna("").astype(str).tolist()

    test_features = extract_feature_dicts(
        texts=test_texts,
        timestamps=test_timestamps,
        ner=ner,
        sentiment=sentiment,
        event_det=event_det,
        context_enricher=context_enricher,
        graph_reasoner=graph_reasoner,
        use_trained_event_head=True,
        cache_path=output_path / f"features_cache_test{cache_suffix}.json",
        cache_features=cache_features,
        desc="Feature extraction (test)",
    )

    test_report, test_report_df = evaluate_on_test_set(
        predictor=predictor,
        test_features=test_features,
        test_df=test_df,
        target_timeframes=target_timeframes,
    )

    # ─────────────────────────────────────────────
    # Step 8: Save and report
    # ─────────────────────────────────────────────
    predictor.save(str(output_path / "impact_predictor"))

    # Print results
    print("\n" + "=" * 75)
    print("TRAINING RESULTS")
    print("=" * 75)

    if "relevance" in metrics:
        m = metrics["relevance"]
        print(f"\nRelevance Classifier:")
        print(f"  Accuracy: {m['accuracy']:.4f}")
        print(f"  F1:       {m['f1']:.4f}")

    print(f"\n{'Asset':<14} {'TF':<4} {'CV Acc':>8} {'CV F1':>8} {'±Std':>8} {'Train Acc':>10} {'Class Dist':>15}")
    print("-" * 75)

    for asset in ASSETS:
        for tf in target_timeframes:
            key = f"{asset}_{tf}"
            if key in metrics:
                m = metrics[key]
                print(f"{asset:<14} {tf:<4} {m['cv_accuracy']:>8.4f} {m['cv_f1']:>8.4f} "
                      f"{m['cv_accuracy_std']:>8.4f} {m['train_accuracy']:>10.4f} "
                      f"{str(m['class_dist']):>15}")

    print(f"\n{'─'*75}")
    print("HELD-OUT TEST RESULTS")
    print(f"{'─'*75}")
    if not test_report_df.empty:
        print(f"{'Asset':<14} {'TF':<4} {'Test Acc':>10} {'Test F1(macro)':>16} {'Test F1(weighted)':>18}")
        print("-" * 75)
        for _, r in test_report_df.iterrows():
            print(f"{r['asset']:<14} {r['timeframe']:<4} {r['accuracy']:>10.4f} "
                  f"{r['f1_macro']:>16.4f} {r['f1_weighted']:>18.4f}")

    if "relevance" in test_report:
        rel = test_report["relevance"]
        print("\nRelevance (held-out):")
        print(f"  Accuracy: {rel['accuracy']:.4f}")
        print(f"  F1:       {rel['f1_weighted']:.4f}")
        if "roc_auc" in rel:
            print(f"  ROC-AUC:  {rel['roc_auc']:.4f}")

    # Top features per asset
    print(f"\n{'─'*75}")
    print("TOP 10 FEATURES PER ASSET (5m timeframe):")
    print(f"{'─'*75}")
    for asset in ASSETS:
        top = predictor.get_top_features(asset, "5m", top_k=10)
        if top:
            print(f"\n  {asset.upper()}:")
            for fname, imp in top:
                print(f"    {fname:<40} importance={imp:.1f}")

    # Save metrics
    metrics_path = output_path / "training_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.info(f"\nMetrics saved to {metrics_path}")

    test_metrics_path = output_path / "test_metrics.json"
    with open(test_metrics_path, "w") as f:
        json.dump(test_report, f, indent=2, default=str)
    logger.info(f"Test metrics saved to {test_metrics_path}")

    test_metrics_csv = output_path / "test_metrics_by_asset.csv"
    test_report_df.to_csv(test_metrics_csv, index=False)
    logger.info(f"Test metrics table saved to {test_metrics_csv}")

    write_test_report_markdown(
        test_report=test_report,
        test_report_df=test_report_df,
        output_path=output_path,
    )
    logger.info(f"Markdown report saved to {output_path / 'test_report.md'}")

    print(f"\n✅ Training complete! Models saved to {output_path}/")
    print(f"   Total models: {len(predictor.models)} classifiers + 1 relevance + 1 event head")

    return {
        "train_metrics": metrics,
        "test_metrics": test_report,
        "split": split_summary,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Trump Tweet Market Impact Classifier")
    parser.add_argument("--data", default="data/train.csv", help="Training CSV path")
    parser.add_argument("--output", default="saved_models", help="Output directory")
    parser.add_argument("--no-gpu", action="store_true", help="Force CPU")
    parser.add_argument("--timeframes", nargs="+", default=["1m", "5m", "10m"],
                        choices=["1m", "5m", "10m"])
    parser.add_argument("--no-cache", action="store_true", help="Don't cache features")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Fraction reserved for strict held-out test set")
    parser.add_argument("--split-method", choices=["time", "random"], default="time",
                        help="How to split train/test")
    parser.add_argument("--random-state", type=int, default=42,
                        help="Random seed for random split")
    parser.add_argument("--tavily-api-key", type=str, default=None,
                        help="Tavily API key (or set TAVILY_API_KEY env var)")
    parser.add_argument("--use-tavily-context", action="store_true",
                        help="Auto-fetch macro context via Tavily during feature extraction")
    args = parser.parse_args()

    tavily_api_key = args.tavily_api_key or os.getenv("TAVILY_API_KEY")

    train(
        data_path=args.data,
        output_dir=args.output,
        use_gpu=not args.no_gpu,
        target_timeframes=args.timeframes,
        cache_features=not args.no_cache,
        test_size=args.test_size,
        split_method=args.split_method,
        random_state=args.random_state,
        tavily_api_key=tavily_api_key,
        use_tavily_context=args.use_tavily_context,
    )
