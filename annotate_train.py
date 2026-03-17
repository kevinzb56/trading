"""
Annotated inference on train.csv.

Outputs one row per tweet with:
  - Original columns
  - Layer 1 (NER)     : entities, entity types, asset mappings
  - Layer 2 (Sentiment): label, compound, positive, negative, neutral
  - Layer 3 (Events)  : primary event + confidence, all detected events
  - Layer 5 (Graph)   : per-asset graph signal and weight
  - Layer 6 (LightGBM): is_market_relevant, relevance_score,
                         per-asset direction + confidence
"""

import json
import logging
import os
import sys
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

# ── Bootstrap ────────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("annotate")

# ── Pipeline imports ──────────────────────────────────────────────────────────
from inference_pipeline import InferencePipeline
from models.market_impact.impact_predictor import ASSETS

ASSETS_LIST = list(ASSETS)

INPUT_CSV  = "train.csv"
OUTPUT_CSV = "train_annotated.csv"
MODEL_DIR  = "saved_models"

# ── Load pipeline ─────────────────────────────────────────────────────────────
log.info("Loading pipeline …")
pipeline = InferencePipeline(use_gpu=True, use_llm_reasoning=False)
pipeline.load_models()
pipeline.load_trained(MODEL_DIR)
log.info("Pipeline ready.")

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(INPUT_CSV)
log.info(f"Loaded {len(df)} rows from {INPUT_CSV}")

texts      = df["content"].astype(str).tolist()
timestamps = df["created_at"].astype(str).tolist() if "created_at" in df.columns else [""] * len(df)
tweet_ids  = df["tweet_id"].astype(str).tolist()   if "tweet_id"   in df.columns else [str(i) for i in range(len(df))]

# ── Per-row annotation ────────────────────────────────────────────────────────
rows = []

for i in tqdm(range(len(texts)), desc="Annotating"):
    text = texts[i]
    ts   = timestamps[i]
    prev_ts = timestamps[i - 1] if i > 0 else None

    # ── Run all layers, capture intermediates ──────────────────────────────
    ner_result       = pipeline.entity_extractor.extract(text)
    sentiment_result = pipeline.sentiment_analyzer.analyze(text)
    event_result     = pipeline.event_detector.detect(text)
    context_result   = pipeline.context_enricher.enrich(
        text=text, created_at=ts, prev_tweet_time=prev_ts, macro_context=""
    )
    graph_result = pipeline.graph_reasoner.build_and_reason(
        entities=ner_result.entities,
        events=event_result.events,
        sentiment_compound=sentiment_result.compound,
    )

    # ── LightGBM prediction ────────────────────────────────────────────────
    features = {}
    features.update(ner_result.to_feature_dict())
    features.update(sentiment_result.to_feature_dict())
    features.update(event_result.to_feature_dict())
    features.update(context_result.to_feature_dict())
    features.update(graph_result.to_feature_dict())

    preds, is_relevant, relevance_score = pipeline.impact_predictor.predict(
        features, timeframe=pipeline.target_timeframe
    )

    # ── Layer 1: NER ──────────────────────────────────────────────────────
    entities_summary = "; ".join(
        f"{e.text} [{e.label}, {e.score:.2f}]" for e in ner_result.entities
    ) if ner_result.entities else "none"

    entity_types = ", ".join(sorted(set(e.label for e in ner_result.entities))) or "none"

    asset_mentions = ", ".join(
        a for e in ner_result.entities for a in e.mapped_assets
    ) or "none"

    # ── Layer 3: Events ───────────────────────────────────────────────────
    non_null_events = [(et, c) for et, c in event_result.events if et != "no_event"]
    all_events_str  = "; ".join(f"{et}({c:.2f})" for et, c in non_null_events) or "no_event"

    # ── Build annotated row ───────────────────────────────────────────────
    row = {
        # originals
        "tweet_id":   tweet_ids[i],
        "created_at": ts,
        "content":    text,

        # ── Layer 1 ─────────────────────────────────────────────────────
        "ner_entities":     entities_summary,
        "ner_entity_types": entity_types,
        "ner_asset_mentions": asset_mentions,
        "ner_entity_count": ner_result.entity_count,

        # ── Layer 2 ─────────────────────────────────────────────────────
        "sentiment_label":    sentiment_result.label,
        "sentiment_compound": round(sentiment_result.compound, 4),
        "sentiment_positive": round(sentiment_result.positive, 4),
        "sentiment_negative": round(sentiment_result.negative, 4),
        "sentiment_neutral":  round(sentiment_result.neutral,  4),

        # ── Layer 3 ─────────────────────────────────────────────────────
        "event_primary":     event_result.primary_event,
        "event_primary_conf": round(event_result.primary_confidence, 4),
        "event_all":         all_events_str,
        "event_count":       event_result.event_count,

        # ── Layer 6 (final) ──────────────────────────────────────────────
        "is_market_relevant": is_relevant,
        "relevance_score":    round(relevance_score, 4),
    }

    # ── Layer 5: graph signal per asset ──────────────────────────────────
    for asset in ASSETS_LIST:
        row[f"graph_signal_{asset}"] = round(graph_result.asset_graph_signal.get(asset, 0.0), 4)
        row[f"graph_weight_{asset}"] = round(graph_result.asset_graph_weight.get(asset, 0.0), 4)

    # ── Layer 6: LightGBM per-asset ───────────────────────────────────────
    for asset in ASSETS_LIST:
        pred = preds.get(asset)
        if pred:
            row[f"{asset}_direction"]   = pred.direction
            row[f"{asset}_confidence"]  = round(pred.confidence, 4)
        else:
            row[f"{asset}_direction"]  = 0
            row[f"{asset}_confidence"] = 0.0

    rows.append(row)

# ── Assemble & save ───────────────────────────────────────────────────────────
df_out = pd.DataFrame(rows)

# Append original actual-direction columns if present
actual_cols = [c for c in df.columns if "_actual_dir_" in c]
if actual_cols:
    df_out = df_out.merge(
        df[["tweet_id"] + actual_cols].astype({"tweet_id": str}),
        on="tweet_id", how="left"
    )

df_out.to_csv(OUTPUT_CSV, index=False)
log.info(f"Saved {len(df_out)} annotated rows to {OUTPUT_CSV}")
log.info(f"Columns ({len(df_out.columns)}): {list(df_out.columns)}")
