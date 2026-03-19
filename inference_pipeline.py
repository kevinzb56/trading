"""
Trump Tweet Market Impact Classifier — Main Inference Pipeline.

Orchestrates all 6 layers:
  1. Entity Extraction (GLiNER)
  2. Sentiment Analysis (FinBERT)
  3. Event Detection (DistilBERT + rules)
  4. Context Enrichment (temporal + Tavily)
  5. Graph Reasoning (entity-event-asset graph)
  6. Market Impact Prediction (LightGBM ensemble)

Usage:
    # Single tweet
    pipeline = InferencePipeline.load("saved_models/")
    result = pipeline.predict("TARIFFS on China! 50% immediately!")

    # Batch
    results = pipeline.predict_batch(df["content"].tolist(), df["created_at"].tolist())
"""

import logging
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Load .env file if present (so API keys can be set there instead of shell)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import numpy as np
import pandas as pd

from models.ner.entity_extractor import EntityExtractor
from models.sentiment.finbert_sentiment import FinBERTSentiment
from models.events.event_detector import EventDetector
from models.events.context_enrichment import ContextEnricher
from models.graph_reasoning.entity_graph import EntityGraph
from models.market_impact.impact_predictor import (
    MarketImpactPredictor, TweetPrediction, ASSETS,
)

# Layer 7: optional LLM reasoning (requires openai>=1.0)
try:
    from models.llm_reasoning.llm_reasoner import LLMReasoner, LLMReasoningResult
    _LLM_AVAILABLE = True
except ImportError:
    _LLM_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class IntermediateLayerOutputs:
    """
    Sidecar container for raw intermediate objects from layers 1-5.
    Used only by predict() when Layer 7 (LLM Reasoning) is enabled.
    Not used by extract_features(), extract_features_batch(), or predict_batch().
    """
    ner_result: object       # EntityExtractionResult
    sentiment_result: object # SentimentResult
    event_result: object     # EventDetectionResult
    graph_result: object     # GraphFeatures
    context_result: object   # ContextFeatures


class InferencePipeline:
    """
    End-to-end inference pipeline for Trump tweet market impact prediction.

    Layers:
        1. EntityExtractor (GLiNER) → entities, asset mapping
        2. FinBERTSentiment → sentiment scores + embeddings
        3. EventDetector (DistilBERT) → event classification
        4. ContextEnricher → temporal + macro features
        5. EntityGraph → graph-based reasoning features
        6. MarketImpactPredictor → final direction + confidence
    """

    def __init__(
        self,
        use_gpu: bool = True,
        tavily_api_key: Optional[str] = None,
        use_tavily_context: bool = False,
        target_timeframe: str = "5m",
        # Layer 7 options
        use_llm_reasoning: bool = False,
        llm_use_tavily: bool = False,
    ):
        self.use_gpu = use_gpu
        self.target_timeframe = target_timeframe

        # Initialize layers 1-5
        self.entity_extractor = EntityExtractor(use_gpu=use_gpu)
        self.sentiment_analyzer = FinBERTSentiment(use_gpu=use_gpu)
        self.event_detector = EventDetector(use_gpu=use_gpu)
        self.context_enricher = ContextEnricher(
            tavily_api_key=tavily_api_key,
            enable_live_search=use_tavily_context,
        )
        self.graph_reasoner = EntityGraph()
        self.impact_predictor = MarketImpactPredictor(target_timeframe=target_timeframe)

        # Layer 7: LLM Reasoning (optional)
        self.use_llm_reasoning = use_llm_reasoning
        self.llm_reasoner: Optional["LLMReasoner"] = None
        if use_llm_reasoning:
            if not _LLM_AVAILABLE:
                raise ImportError(
                    "openai package is required for Layer 7. Install with: pip install openai>=1.0"
                )
            self.llm_reasoner = LLMReasoner.from_env(
                use_tavily_for_llm=llm_use_tavily if llm_use_tavily else None
            )
            logger.info("Layer 7 (LLM Reasoning via Azure OpenAI GPT-4o) initialized")

        self._loaded = False

    def load_models(self):
        """Load all model weights."""
        logger.info("Loading all pipeline models...")
        t0 = time.time()

        self.entity_extractor.load()
        self.sentiment_analyzer.load()
        self.event_detector.load()

        self._loaded = True
        logger.info(f"All models loaded in {time.time() - t0:.1f}s")

    def extract_features(
        self,
        text: str,
        created_at: str = "",
        prev_tweet_time: Optional[str] = None,
        macro_context: str = "",
    ) -> dict:
        """
        Run all feature extraction layers on a single tweet.

        Returns flat feature dictionary suitable for the predictor.
        """
        if not self._loaded:
            self.load_models()

        features = {}

        # Layer 1: Entity Extraction
        ner_result = self.entity_extractor.extract(text)
        features.update(ner_result.to_feature_dict())

        # Layer 2: Sentiment Analysis
        sentiment_result = self.sentiment_analyzer.analyze(text)
        features.update(sentiment_result.to_feature_dict())

        # Layer 3: Event Detection
        event_result = self.event_detector.detect(text)
        features.update(event_result.to_feature_dict())

        # Layer 4: Context Enrichment
        context_result = self.context_enricher.enrich(
            text=text,
            created_at=created_at,
            prev_tweet_time=prev_tweet_time,
            macro_context=macro_context,
        )
        features.update(context_result.to_feature_dict())

        # Layer 5: Graph Reasoning
        graph_result = self.graph_reasoner.build_and_reason(
            entities=ner_result.entities,
            events=event_result.events,
            sentiment_compound=sentiment_result.compound,
        )
        features.update(graph_result.to_feature_dict())

        return features

    def _extract_features_with_intermediates(
        self,
        text: str,
        created_at: str = "",
        prev_tweet_time: Optional[str] = None,
        macro_context: str = "",
    ) -> tuple:
        """
        Like extract_features() but also returns the raw intermediate layer objects.

        Called only from predict() when Layer 7 (LLM Reasoning) is enabled.
        Returns the same flat feature dict as extract_features(), plus an
        IntermediateLayerOutputs sidecar containing the rich objects needed to
        build the LLM prompt. Never called by extract_features_batch() or
        predict_batch() — those paths are unaffected.
        """
        if not self._loaded:
            self.load_models()

        features = {}

        ner_result = self.entity_extractor.extract(text)
        features.update(ner_result.to_feature_dict())

        sentiment_result = self.sentiment_analyzer.analyze(text)
        features.update(sentiment_result.to_feature_dict())

        event_result = self.event_detector.detect(text)
        features.update(event_result.to_feature_dict())

        context_result = self.context_enricher.enrich(
            text=text,
            created_at=created_at,
            prev_tweet_time=prev_tweet_time,
            macro_context=macro_context,
        )
        features.update(context_result.to_feature_dict())

        graph_result = self.graph_reasoner.build_and_reason(
            entities=ner_result.entities,
            events=event_result.events,
            sentiment_compound=sentiment_result.compound,
        )
        features.update(graph_result.to_feature_dict())

        intermediates = IntermediateLayerOutputs(
            ner_result=ner_result,
            sentiment_result=sentiment_result,
            event_result=event_result,
            graph_result=graph_result,
            context_result=context_result,
        )
        return features, intermediates

    def predict(
        self,
        text: str,
        created_at: str = "",
        prev_tweet_time: Optional[str] = None,
        macro_context: str = "",
        tweet_id: str = "",
        thread_context: Optional[list] = None,
    ) -> TweetPrediction:
        """
        Full prediction for a single tweet.

        When Layer 7 (LLM Reasoning) is enabled, runs all 5 feature layers via
        _extract_features_with_intermediates() to also capture the rich layer
        objects for the LLM prompt, then calls the Azure OpenAI GPT-4o reasoner
        after the LightGBM predictions are complete.

        Returns TweetPrediction with per-asset direction and confidence, plus an
        optional llm_reasoning field when Layer 7 is active.
        """
        # Choose extraction path based on whether Layer 7 needs intermediates
        if self.use_llm_reasoning:
            features, intermediates = self._extract_features_with_intermediates(
                text=text,
                created_at=created_at,
                prev_tweet_time=prev_tweet_time,
                macro_context=macro_context,
            )
        else:
            features = self.extract_features(
                text=text,
                created_at=created_at,
                prev_tweet_time=prev_tweet_time,
                macro_context=macro_context,
            )
            intermediates = None

        predictions, is_relevant, relevance_score = self.impact_predictor.predict(
            features, timeframe=self.target_timeframe
        )

        # Layer 7: LLM Reasoning (optional, non-blocking)
        llm_result = None
        if self.use_llm_reasoning and self.llm_reasoner is not None and intermediates is not None:
            try:
                llm_result = self.llm_reasoner.reason(
                    text=text,
                    created_at=created_at,
                    timeframe=self.target_timeframe,
                    intermediates=intermediates,
                    lgbm_predictions=predictions,
                    is_market_relevant=is_relevant,
                    relevance_score=relevance_score,
                    thread_context=thread_context,
                )
            except Exception as e:
                logger.error(f"Layer 7 LLM reasoning failed, continuing without it: {e}")

        return TweetPrediction(
            tweet_id=tweet_id,
            content=text,
            is_market_relevant=is_relevant,
            relevance_score=relevance_score,
            predictions=predictions,
            llm_reasoning=llm_result,
        )

    def predict_batch(
        self,
        texts: list,
        timestamps: Optional[list] = None,
        tweet_ids: Optional[list] = None,
        macro_contexts: Optional[list] = None,
        thread_contexts: Optional[list] = None,
        show_progress: bool = True,
    ) -> list:
        """
        Predict market impact for a batch of tweets.

        Returns list of TweetPrediction objects.
        """
        if not self._loaded:
            self.load_models()

        n = len(texts)
        if timestamps is None:
            timestamps = [""] * n
        if tweet_ids is None:
            tweet_ids = [str(i) for i in range(n)]
        if macro_contexts is None:
            macro_contexts = [""] * n
        if thread_contexts is None:
            thread_contexts = [None] * n

        # Extract features for all tweets
        logger.info(f"Extracting features for {n} tweets...")
        t0 = time.time()

        all_features = []
        iterator = range(n)
        if show_progress:
            try:
                from tqdm.auto import tqdm
                iterator = tqdm(iterator, desc="Feature extraction")
            except ImportError:
                pass

        all_intermediates = []
        for i in iterator:
            prev_ts = timestamps[i - 1] if i > 0 else None
            if self.use_llm_reasoning:
                features, intermediates = self._extract_features_with_intermediates(
                    text=texts[i],
                    created_at=timestamps[i],
                    prev_tweet_time=prev_ts,
                    macro_context=macro_contexts[i],
                )
                all_intermediates.append(intermediates)
            else:
                features = self.extract_features(
                    text=texts[i],
                    created_at=timestamps[i],
                    prev_tweet_time=prev_ts,
                    macro_context=macro_contexts[i],
                )
                all_intermediates.append(None)
            all_features.append(features)

        feature_time = time.time() - t0
        logger.info(f"Feature extraction: {feature_time:.1f}s ({feature_time/n*1000:.0f}ms/tweet)")

        # Predict
        logger.info("Running market impact predictions...")
        t0 = time.time()
        batch_results = self.impact_predictor.predict_batch(
            all_features, timeframe=self.target_timeframe
        )
        predict_time = time.time() - t0
        logger.info(f"Prediction: {predict_time:.1f}s ({predict_time/n*1000:.0f}ms/tweet)")

        # Assemble results, running Layer 7 LLM reasoning per tweet if enabled
        if self.use_llm_reasoning and self.llm_reasoner is not None:
            logger.info(f"Running Layer 7 LLM reasoning for {n} tweets...")

        results = []
        for i, (preds, is_rel, rel_score) in enumerate(batch_results):
            llm_result = None
            if self.use_llm_reasoning and self.llm_reasoner is not None and all_intermediates[i] is not None:
                try:
                    llm_result = self.llm_reasoner.reason(
                        text=texts[i],
                        created_at=timestamps[i],
                        timeframe=self.target_timeframe,
                        intermediates=all_intermediates[i],
                        lgbm_predictions=preds,
                        is_market_relevant=is_rel,
                        relevance_score=rel_score,
                        thread_context=thread_contexts[i],
                    )
                except Exception as e:
                    logger.error(f"Layer 7 LLM reasoning failed for tweet {tweet_ids[i]}, continuing without it: {e}")
            results.append(TweetPrediction(
                tweet_id=tweet_ids[i],
                content=texts[i],
                is_market_relevant=is_rel,
                relevance_score=rel_score,
                predictions=preds,
                llm_reasoning=llm_result,
            ))

        return results

    def extract_features_batch(
        self,
        texts: list,
        timestamps: Optional[list] = None,
        macro_contexts: Optional[list] = None,
        show_progress: bool = True,
    ) -> list:
        """
        Extract features only (for training). Returns list of feature dicts.
        """
        if not self._loaded:
            self.load_models()

        n = len(texts)
        if timestamps is None:
            timestamps = [""] * n
        if macro_contexts is None:
            macro_contexts = [""] * n

        all_features = []
        iterator = range(n)
        if show_progress:
            try:
                from tqdm.auto import tqdm
                iterator = tqdm(iterator, desc="Feature extraction")
            except ImportError:
                pass

        for i in iterator:
            prev_ts = timestamps[i - 1] if i > 0 else None
            features = self.extract_features(
                text=texts[i],
                created_at=timestamps[i],
                prev_tweet_time=prev_ts,
                macro_context=macro_contexts[i],
            )
            all_features.append(features)

        return all_features

    def save(self, path: str):
        """Save trained models."""
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)
        self.impact_predictor.save(str(save_dir / "impact_predictor"))
        self.event_detector.save(str(save_dir / "event_head.pt"))
        logger.info(f"Pipeline saved to {save_dir}")

    def load_trained(self, path: str):
        """Load trained models (impact predictor + event head)."""
        save_dir = Path(path)
        self.impact_predictor.load(str(save_dir / "impact_predictor"))
        event_head_path = save_dir / "event_head.pt"
        if event_head_path.exists():
            self.event_detector.load_head(str(event_head_path))
        logger.info(f"Trained pipeline loaded from {save_dir}")

    @classmethod
    def load(
        cls,
        path: str,
        use_gpu: bool = True,
        target_timeframe: str = "5m",
        tavily_api_key: Optional[str] = None,
        use_tavily_context: bool = False,
        use_llm_reasoning: bool = False,
        llm_use_tavily: bool = False,
    ):
        """Class method to create and load a pre-trained pipeline."""
        pipeline = cls(
            use_gpu=use_gpu,
            tavily_api_key=tavily_api_key,
            use_tavily_context=use_tavily_context,
            target_timeframe=target_timeframe,
            use_llm_reasoning=use_llm_reasoning,
            llm_use_tavily=llm_use_tavily,
        )
        pipeline.load_models()
        pipeline.load_trained(path)
        return pipeline


def results_to_dataframe(results: list) -> pd.DataFrame:
    """Convert list of TweetPrediction to a DataFrame."""
    rows = []
    for r in results:
        row = {
            "tweet_id": r.tweet_id,
            "content": r.content[:200],
            "is_market_relevant": r.is_market_relevant,
            "relevance_score": r.relevance_score,
        }
        for asset, pred in r.predictions.items():
            row[f"{asset}_dir"] = pred.direction
            row[f"{asset}_conf"] = pred.confidence
            row[f"{asset}_reasoning"] = "; ".join(pred.reasoning[:3])
        if r.llm_reasoning is not None:
            row["llm_overall_assessment"] = r.llm_reasoning.overall_assessment
            row["llm_is_market_relevant"] = r.llm_reasoning.llm_is_market_relevant
            row["llm_relevance_reasoning"] = r.llm_reasoning.llm_relevance_reasoning
            row["llm_model_used"] = r.llm_reasoning.model_used
            row["llm_latency_ms"] = r.llm_reasoning.latency_ms
            row["llm_tavily_used"] = r.llm_reasoning.tavily_context_used
            for asset, ar in r.llm_reasoning.per_asset.items():
                row[f"{asset}_llm_dir"] = ar.direction
                row[f"{asset}_llm_conf"] = ar.confidence
                row[f"{asset}_llm_reasoning"] = ar.reasoning
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Trump Tweet Market Impact Predictor")
    parser.add_argument("--model-dir", default="saved_models", help="Trained model directory")
    parser.add_argument("--tweet", type=str, help="Single tweet to predict")
    parser.add_argument("--csv", type=str, help="CSV file with tweets to predict")
    parser.add_argument("--output", type=str, default="predictions_output.csv", help="Output CSV")
    parser.add_argument("--timeframe", default="5m", choices=["1m", "5m", "10m"])
    parser.add_argument("--no-gpu", action="store_true")
    parser.add_argument("--tavily-api-key", type=str, default=None,
                        help="Tavily API key (or set TAVILY_API_KEY env var)")
    parser.add_argument("--use-tavily-context", action="store_true",
                        help="Auto-fetch macro context from Tavily when macro_context is empty")
    parser.add_argument("--use-llm-reasoning", action="store_true",
                        help="Enable Layer 7: Azure OpenAI GPT-4o reasoning over all layer outputs")
    parser.add_argument("--llm-use-tavily", action="store_true",
                        help="Fetch real-time Tavily context for the LLM prompt (Layer 7 only)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    tavily_api_key = args.tavily_api_key or os.getenv("TAVILY_API_KEY")

    if args.tweet:
        pipeline = InferencePipeline.load(
            args.model_dir,
            use_gpu=not args.no_gpu,
            target_timeframe=args.timeframe,
            tavily_api_key=tavily_api_key,
            use_tavily_context=args.use_tavily_context,
            use_llm_reasoning=args.use_llm_reasoning,
            llm_use_tavily=args.llm_use_tavily,
        )
        result = pipeline.predict(args.tweet)
        print(json.dumps(result.to_dict(), indent=2))

    elif args.csv:
        df = pd.read_csv(args.csv)
        pipeline = InferencePipeline.load(
            args.model_dir,
            use_gpu=not args.no_gpu,
            target_timeframe=args.timeframe,
            tavily_api_key=tavily_api_key,
            use_tavily_context=args.use_tavily_context,
            use_llm_reasoning=args.use_llm_reasoning,
            llm_use_tavily=args.llm_use_tavily,
        )
        results = pipeline.predict_batch(
            texts=df["content"].tolist(),
            timestamps=df.get("created_at", pd.Series([""] * len(df))).tolist(),
            tweet_ids=df.get("tweet_id", pd.Series(range(len(df)))).astype(str).tolist(),
        )
        df_out = results_to_dataframe(results)
        df_out.to_csv(args.output, index=False)
        print(f"Saved {len(df_out)} predictions to {args.output}")
    else:
        print("Provide --tweet or --csv argument")
