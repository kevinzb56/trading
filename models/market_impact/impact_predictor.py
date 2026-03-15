"""
Layer 6: Market Impact Predictor.

Combines features from all layers (NER, sentiment, events, context, graph)
to predict per-asset direction and confidence.

Architecture:
  1. Feature assembly from all layers
  2. LightGBM classifier per asset (3-class: -1, 0, 1)
  3. Confidence calibration via Platt scaling
  4. Explainability via SHAP or feature importance
"""

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ASSETS = ["gold", "equities", "btc", "cl", "wheat", "eurodollar", "treasury_2y"]
TIMEFRAMES = ["1m", "5m", "10m"]
ASSET_TICKERS = {
    "gold": "GC", "equities": "ES", "btc": "BTC", "cl": "CL",
    "wheat": "ZW", "eurodollar": "6E", "treasury_2y": "ZT",
}


@dataclass
class MarketImpactPrediction:
    """Prediction for a single asset."""
    asset: str
    direction: int  # -1, 0, 1
    confidence: float  # 0.0 to 1.0
    reasoning: list = field(default_factory=list)


@dataclass
class TweetPrediction:
    """Full prediction for a single tweet across all assets."""
    tweet_id: str
    content: str
    is_market_relevant: bool
    relevance_score: float
    predictions: dict = field(default_factory=dict)   # asset -> MarketImpactPrediction
    llm_reasoning: Optional[Any] = field(default=None)    # LLMReasoningResult, if Layer 7 ran

    def to_dict(self) -> dict:
        """Convert to serializable dictionary."""
        result = {
            "tweet_id": self.tweet_id,
            "content": self.content[:200],
            "is_market_relevant": self.is_market_relevant,
            "relevance_score": self.relevance_score,
            "assets": {},
        }
        for asset, pred in self.predictions.items():
            result["assets"][asset] = {
                "direction": pred.direction,
                "confidence": pred.confidence,
                "reasoning": pred.reasoning,
            }
        if self.llm_reasoning is not None:
            result["llm_reasoning"] = self.llm_reasoning.to_dict()
        return result


class MarketImpactPredictor:
    """
    Multi-asset market impact predictor using LightGBM.

    Trains one classifier per (asset, timeframe) combination.
    Also trains a relevance classifier.
    """

    def __init__(self, target_timeframe: str = "5m"):
        self.target_timeframe = target_timeframe
        self.models = {}  # (asset, timeframe) -> trained model
        self.relevance_model = None
        self.feature_names = None
        self.feature_importance = {}

    def _get_label_col(self, asset: str, timeframe: str) -> str:
        """Get the column name for ground truth labels."""
        ticker = ASSET_TICKERS[asset]
        return f"{asset}_{ticker}_actual_dir_{timeframe}"

    def assemble_features(self, feature_dicts: list) -> pd.DataFrame:
        """
        Assemble feature matrix from list of per-tweet feature dicts.

        Each dict should contain keys from:
        - NER: ner_*
        - Sentiment: sentiment_*
        - Events: event_*
        - Context: ctx_*
        - Graph: graph_*
        """
        df = pd.DataFrame(feature_dicts)
        # Fill NaN with 0
        df = df.fillna(0.0)
        # Ensure consistent column ordering
        if self.feature_names is not None:
            for col in self.feature_names:
                if col not in df.columns:
                    df[col] = 0.0
            df = df[self.feature_names]
        return df

    def train(
        self,
        feature_dicts: list,
        labels_df: pd.DataFrame,
        timeframes: Optional[list] = None,
        n_splits: int = 5,
    ) -> dict:
        """
        Train classifiers for all assets and timeframes.

        Args:
            feature_dicts: list of dicts from feature extraction
            labels_df: DataFrame with actual direction columns
            timeframes: which timeframes to train (default: all)
            n_splits: number of CV folds

        Returns:
            Dictionary of evaluation metrics.
        """
        try:
            import lightgbm as lgb
        except ImportError:
            logger.error("LightGBM not installed. pip install lightgbm")
            raise

        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import accuracy_score, f1_score, classification_report

        if timeframes is None:
            timeframes = TIMEFRAMES

        X = self.assemble_features(feature_dicts)
        self.feature_names = list(X.columns)
        logger.info(f"Feature matrix: {X.shape} ({len(self.feature_names)} features)")

        metrics = {}

        # Train relevance classifier
        logger.info("Training relevance classifier...")
        if "is_market_relevant" in labels_df.columns:
            y_rel = labels_df["is_market_relevant"].astype(int).values
            self.relevance_model = lgb.LGBMClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                class_weight="balanced", random_state=42, verbose=-1,
            )
            self.relevance_model.fit(X, y_rel)
            rel_pred = self.relevance_model.predict(X)
            metrics["relevance"] = {
                "accuracy": accuracy_score(y_rel, rel_pred),
                "f1": f1_score(y_rel, rel_pred, average="weighted"),
            }
            logger.info(f"  Relevance acc={metrics['relevance']['accuracy']:.3f}")

        # Train per-asset direction classifiers
        for asset in ASSETS:
            for tf in timeframes:
                label_col = self._get_label_col(asset, tf)
                if label_col not in labels_df.columns:
                    logger.warning(f"  Skipping {asset}/{tf}: column {label_col} not found")
                    continue

                y = labels_df[label_col].values
                # Map -1, 0, 1 to 0, 1, 2 for LightGBM
                y_mapped = y + 1  # {-1,0,1} -> {0,1,2}

                logger.info(f"Training {asset}/{tf}: {len(y)} samples, "
                            f"class dist={np.bincount(y_mapped, minlength=3)}")

                # Compute class weights
                class_counts = np.bincount(y_mapped, minlength=3)
                total = len(y_mapped)
                class_weights = {i: total / (3 * max(c, 1)) for i, c in enumerate(class_counts)}
                sample_weights = np.array([class_weights[yi] for yi in y_mapped])

                model = lgb.LGBMClassifier(
                    n_estimators=300,
                    max_depth=7,
                    learning_rate=0.03,
                    subsample=0.8,
                    colsample_bytree=0.7,
                    min_child_samples=20,
                    reg_alpha=0.1,
                    reg_lambda=0.1,
                    num_leaves=63,
                    random_state=42,
                    verbose=-1,
                    n_jobs=-1,
                )

                # Cross-validation
                skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
                cv_accs, cv_f1s = [], []
                for train_idx, val_idx in skf.split(X, y_mapped):
                    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
                    y_tr, y_val = y_mapped[train_idx], y_mapped[val_idx]
                    sw_tr = sample_weights[train_idx]

                    model_cv = lgb.LGBMClassifier(
                        n_estimators=300, max_depth=7, learning_rate=0.03,
                        subsample=0.8, colsample_bytree=0.7, min_child_samples=20,
                        reg_alpha=0.1, reg_lambda=0.1, num_leaves=63,
                        random_state=42, verbose=-1, n_jobs=-1,
                    )
                    model_cv.fit(X_tr, y_tr, sample_weight=sw_tr)
                    y_pred = model_cv.predict(X_val)
                    cv_accs.append(accuracy_score(y_val, y_pred))
                    cv_f1s.append(f1_score(y_val, y_pred, average="weighted"))

                # Train final model on all data
                model.fit(X, y_mapped, sample_weight=sample_weights)
                self.models[(asset, tf)] = model

                # Store feature importance
                importance = model.feature_importances_
                self.feature_importance[(asset, tf)] = dict(
                    zip(self.feature_names, importance)
                )

                # Full-data metrics (overfit metric, for reference)
                y_pred_full = model.predict(X)
                metrics[f"{asset}_{tf}"] = {
                    "cv_accuracy": np.mean(cv_accs),
                    "cv_f1": np.mean(cv_f1s),
                    "cv_accuracy_std": np.std(cv_accs),
                    "train_accuracy": accuracy_score(y_mapped, y_pred_full),
                    "train_f1": f1_score(y_mapped, y_pred_full, average="weighted"),
                    "class_dist": class_counts.tolist(),
                }
                logger.info(f"  {asset}/{tf}: CV_acc={np.mean(cv_accs):.3f}±{np.std(cv_accs):.3f}, "
                            f"CV_f1={np.mean(cv_f1s):.3f}")

        return metrics

    def predict(self, feature_dict: dict, timeframe: Optional[str] = None) -> dict:
        """
        Predict market impact for a single tweet.

        Returns: dict of asset -> MarketImpactPrediction
        """
        if timeframe is None:
            timeframe = self.target_timeframe

        X = self.assemble_features([feature_dict])
        predictions = {}

        # Relevance prediction
        is_relevant = True
        relevance_score = 0.5
        if self.relevance_model is not None:
            rel_proba = self.relevance_model.predict_proba(X)[0]
            is_relevant = bool(rel_proba[1] > 0.5)
            relevance_score = float(rel_proba[1])

        for asset in ASSETS:
            key = (asset, timeframe)
            if key not in self.models:
                predictions[asset] = MarketImpactPrediction(
                    asset=asset, direction=0, confidence=0.0,
                    reasoning=["No trained model available"],
                )
                continue

            model = self.models[key]
            proba = model.predict_proba(X)[0]  # [P(-1), P(0), P(1)]
            pred_class = int(np.argmax(proba))
            direction = pred_class - 1  # map back: {0,1,2} -> {-1,0,1}
            confidence = float(proba[pred_class])

            # Scale confidence down for irrelevant tweets
            if not is_relevant:
                confidence *= 0.3

            # Generate reasoning from feature importance
            reasoning = self._get_reasoning(feature_dict, asset, timeframe)

            predictions[asset] = MarketImpactPrediction(
                asset=asset,
                direction=direction,
                confidence=round(confidence, 3),
                reasoning=reasoning,
            )

        return predictions, is_relevant, relevance_score

    def predict_batch(self, feature_dicts: list, timeframe: Optional[str] = None) -> list:
        """Predict for a batch of tweets."""
        results = []
        for fd in feature_dicts:
            preds, is_rel, rel_score = self.predict(fd, timeframe)
            results.append((preds, is_rel, rel_score))
        return results

    def _get_reasoning(self, feature_dict: dict, asset: str, timeframe: str,
                       top_k: int = 5) -> list:
        """Generate reasoning from top feature importances."""
        key = (asset, timeframe)
        if key not in self.feature_importance:
            return []

        importance = self.feature_importance[key]
        # Get features that are nonzero in this sample
        active_features = {
            k: v for k, v in importance.items()
            if k in feature_dict and feature_dict.get(k, 0) != 0
        }
        # Sort by importance
        sorted_features = sorted(active_features.items(), key=lambda x: x[1], reverse=True)

        reasons = []
        for feat_name, imp in sorted_features[:top_k]:
            val = feature_dict.get(feat_name, 0)
            # Create human-readable reason
            reason = self._feature_to_reason(feat_name, val)
            if reason:
                reasons.append(reason)

        return reasons

    @staticmethod
    def _feature_to_reason(feature_name: str, value) -> str:
        """Convert a feature name + value to human-readable reasoning."""
        if feature_name.startswith("sentiment_"):
            if "compound" in feature_name:
                if value > 0.3:
                    return "positive financial sentiment"
                elif value < -0.3:
                    return "negative financial sentiment"
            return ""
        elif feature_name.startswith("ner_"):
            if "policy" in feature_name and value:
                return "policy-related entity detected"
            elif "country" in feature_name and value:
                return "country/geopolitical entity mentioned"
            elif "commodity" in feature_name and value:
                return "commodity entity mentioned"
            elif "asset_relevance" in feature_name and value > 0.5:
                asset = feature_name.split("_")[-1]
                return f"high entity relevance to {asset}"
            return ""
        elif feature_name.startswith("event_"):
            if feature_name.endswith("_conf"):
                evt = feature_name.replace("event_", "").replace("_conf", "")
                if value > 0.3:
                    return f"{evt.replace('_', ' ')} event detected"
            elif value > 0:
                evt = feature_name.replace("event_", "")
                return f"{evt.replace('_', ' ')} event"
            return ""
        elif feature_name.startswith("graph_"):
            if "signal" in feature_name and abs(value) > 0.2:
                asset = feature_name.split("_")[-1]
                direction = "bullish" if value > 0 else "bearish"
                return f"graph propagation: {direction} signal for {asset}"
            return ""
        elif feature_name.startswith("ctx_"):
            if "market_hours" in feature_name and value:
                return "posted during market hours"
            elif "rapid_fire" in feature_name and value:
                return "rapid-fire posting pattern"
            elif "all_caps" in feature_name and value > 0.5:
                return "aggressive tone (high caps ratio)"
            return ""
        return ""

    def save(self, path: str):
        """Save all trained models."""
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save models
        for (asset, tf), model in self.models.items():
            model_path = save_dir / f"model_{asset}_{tf}.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(model, f)

        # Save relevance model
        if self.relevance_model is not None:
            with open(save_dir / "model_relevance.pkl", "wb") as f:
                pickle.dump(self.relevance_model, f)

        # Save metadata
        metadata = {
            "feature_names": self.feature_names,
            "feature_importance": {
                f"{a}_{t}": imp for (a, t), imp in self.feature_importance.items()
            },
            "target_timeframe": self.target_timeframe,
        }
        with open(save_dir / "metadata.pkl", "wb") as f:
            pickle.dump(metadata, f)

        logger.info(f"Models saved to {save_dir}")

    def load(self, path: str):
        """Load trained models."""
        save_dir = Path(path)

        # Load metadata
        with open(save_dir / "metadata.pkl", "rb") as f:
            metadata = pickle.load(f)
        self.feature_names = metadata["feature_names"]
        self.target_timeframe = metadata["target_timeframe"]
        self.feature_importance = {
            tuple(k.rsplit("_", 1)): v
            for k, v in metadata["feature_importance"].items()
        }

        # Load models
        for asset in ASSETS:
            for tf in TIMEFRAMES:
                model_path = save_dir / f"model_{asset}_{tf}.pkl"
                if model_path.exists():
                    with open(model_path, "rb") as f:
                        self.models[(asset, tf)] = pickle.load(f)

        # Load relevance model
        rel_path = save_dir / "model_relevance.pkl"
        if rel_path.exists():
            with open(rel_path, "rb") as f:
                self.relevance_model = pickle.load(f)

        logger.info(f"Loaded {len(self.models)} models from {save_dir}")

    def get_top_features(self, asset: str, timeframe: str, top_k: int = 20) -> list:
        """Get top features by importance for an asset/timeframe."""
        key = (asset, timeframe)
        if key not in self.feature_importance:
            return []
        imp = self.feature_importance[key]
        sorted_feats = sorted(imp.items(), key=lambda x: x[1], reverse=True)
        return sorted_feats[:top_k]
