"""
Trump Tweet Market Impact Classifier — Training Script.

Loads train.csv, runs all feature extraction layers, trains LightGBM
classifiers for each asset/timeframe, evaluates, and saves models.

Usage:
    python train_pipeline.py --data data/train.csv --output saved_models/
    python train_pipeline.py --data data/train.csv --output saved_models/ --no-gpu
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

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
    logger.info(f"Texts: {len(texts)}, Date range: {timestamps[0][:10]} to {timestamps[-1][:10]}")

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

    context_enricher = ContextEnricher()
    graph_reasoner = EntityGraph()

    # ─────────────────────────────────────────────
    # Step 3: Extract features
    # ─────────────────────────────────────────────
    features_cache_path = output_path / "features_cache.json"

    if cache_features and features_cache_path.exists():
        logger.info(f"Loading cached features from {features_cache_path}")
        with open(features_cache_path) as f:
            all_features = json.load(f)
        logger.info(f"Loaded {len(all_features)} cached feature dicts")
    else:
        logger.info("Extracting features for all tweets...")
        t0 = time.time()

        try:
            from tqdm.auto import tqdm
            iterator = tqdm(range(len(texts)), desc="Feature extraction")
        except ImportError:
            iterator = range(len(texts))

        all_features = []
        for i in iterator:
            text = texts[i]
            ts = timestamps[i]
            prev_ts = timestamps[i - 1] if i > 0 else None

            features = {}

            # Layer 1: NER
            ner_result = ner.extract(text)
            features.update(ner_result.to_feature_dict())

            # Layer 2: Sentiment
            sent_result = sentiment.analyze(text)
            features.update(sent_result.to_feature_dict())

            # Layer 3: Events (rule-based initially)
            event_result = event_det.detect_rules(text)
            features.update(event_result.to_feature_dict())

            # Layer 4: Context
            ctx_result = context_enricher.enrich(
                text=text, created_at=ts, prev_tweet_time=prev_ts,
            )
            features.update(ctx_result.to_feature_dict())

            # Layer 5: Graph
            try:
                graph_result = graph_reasoner.build_and_reason(
                    entities=ner_result.entities,
                    events=event_result.events,
                    sentiment_compound=sent_result.compound,
                )
                features.update(graph_result.to_feature_dict())
            except Exception as e:
                logger.debug(f"Graph reasoning failed for tweet {i}: {e}")

            all_features.append(features)

        feature_time = time.time() - t0
        logger.info(f"Feature extraction: {feature_time:.1f}s ({feature_time/len(texts)*1000:.0f}ms/tweet)")

        # Cache features
        if cache_features:
            with open(features_cache_path, "w") as f:
                json.dump(all_features, f)
            logger.info(f"Features cached to {features_cache_path}")

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
        labels_df=df,
        timeframes=target_timeframes,
        n_splits=5,
    )

    # ─────────────────────────────────────────────
    # Step 7: Save and report
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

    print(f"\n✅ Training complete! Models saved to {output_path}/")
    print(f"   Total models: {len(predictor.models)} classifiers + 1 relevance + 1 event head")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Trump Tweet Market Impact Classifier")
    parser.add_argument("--data", default="data/train.csv", help="Training CSV path")
    parser.add_argument("--output", default="saved_models", help="Output directory")
    parser.add_argument("--no-gpu", action="store_true", help="Force CPU")
    parser.add_argument("--timeframes", nargs="+", default=["1m", "5m", "10m"],
                        choices=["1m", "5m", "10m"])
    parser.add_argument("--no-cache", action="store_true", help="Don't cache features")
    args = parser.parse_args()

    train(
        data_path=args.data,
        output_dir=args.output,
        use_gpu=not args.no_gpu,
        target_timeframes=args.timeframes,
        cache_features=not args.no_cache,
    )
