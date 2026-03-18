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

    # Ensemble configs: diverse hyperparameter sets trained together, soft-voted at prediction time
    _ENSEMBLE_CONFIGS = [
        dict(n_estimators=800, max_depth=6,  learning_rate=0.02, num_leaves=47,
             subsample=0.8, colsample_bytree=0.7, min_child_samples=15,
             reg_alpha=0.05, reg_lambda=0.1,  boosting_type="gbdt"),
        dict(n_estimators=600, max_depth=8,  learning_rate=0.03, num_leaves=63,
             subsample=0.7, colsample_bytree=0.8, min_child_samples=10,
             reg_alpha=0.1,  reg_lambda=0.05, boosting_type="gbdt"),
        dict(n_estimators=500, max_depth=5,  learning_rate=0.05, num_leaves=31,
             subsample=0.9, colsample_bytree=0.6, min_child_samples=20,
             reg_alpha=0.2,  reg_lambda=0.2,  boosting_type="gbdt"),
        dict(n_estimators=700, max_depth=7,  learning_rate=0.025, num_leaves=55,
             subsample=0.75, colsample_bytree=0.75, min_child_samples=12,
             reg_alpha=0.08, reg_lambda=0.08, boosting_type="dart",
             drop_rate=0.1,  skip_drop=0.5),
        dict(n_estimators=400, max_depth=4,  learning_rate=0.08, num_leaves=24,
             subsample=0.85, colsample_bytree=0.65, min_child_samples=25,
             reg_alpha=0.15, reg_lambda=0.15, boosting_type="gbdt"),
    ]

    def _compute_sample_weights(self, y_mapped: np.ndarray,
                                neutral_boost: float = 5.0) -> np.ndarray:
        """
        Compute per-sample weights with an extra boost for the neutral class,
        which is heavily under-represented (~10% of data).
        """
        class_counts = np.bincount(y_mapped, minlength=3)
        total = len(y_mapped)
        base_weights = {i: total / (3 * max(c, 1)) for i, c in enumerate(class_counts)}
        # Apply extra multiplier to neutral (class index 1 = original label 0)
        base_weights[1] = base_weights[1] * neutral_boost
        return np.array([base_weights[yi] for yi in y_mapped])

    def train(
        self,
        feature_dicts: list,
        labels_df: pd.DataFrame,
        timeframes: Optional[list] = None,
        n_splits: int = 5,
        tune: bool = False,
        tune_trials: int = 30,
    ) -> dict:
        """
        Train classifiers for all assets and timeframes.

        Uses a soft-voting ensemble of diverse LightGBM configs with boosted
        neutral-class weighting, early stopping, and optional Optuna tuning.

        Args:
            feature_dicts:  list of dicts from feature extraction
            labels_df:      DataFrame with actual direction columns
            timeframes:     which timeframes to train (default: all)
            n_splits:       number of CV folds
            tune:           run Optuna hyperparameter search (slower)
            tune_trials:    number of Optuna trials per model when tune=True

        Returns:
            Dictionary of evaluation metrics.
        """
        try:
            import lightgbm as lgb
        except ImportError:
            logger.error("LightGBM not installed. pip install lightgbm")
            raise

        from sklearn.model_selection import StratifiedKFold, train_test_split
        from sklearn.metrics import accuracy_score, f1_score

        if timeframes is None:
            timeframes = TIMEFRAMES

        X = self.assemble_features(feature_dicts)
        self.feature_names = list(X.columns)
        logger.info(f"Feature matrix: {X.shape} ({len(self.feature_names)} features)")

        metrics = {}

        # ── Relevance classifier ──────────────────────────────────────────────
        logger.info("Training relevance classifier...")
        if "is_market_relevant" in labels_df.columns:
            y_rel = labels_df["is_market_relevant"].astype(int).values
            self.relevance_model = lgb.LGBMClassifier(
                n_estimators=400, max_depth=6, learning_rate=0.04,
                subsample=0.8, colsample_bytree=0.8, num_leaves=47,
                class_weight="balanced", random_state=42, verbose=-1, n_jobs=-1,
            )
            self.relevance_model.fit(X, y_rel)
            rel_pred = self.relevance_model.predict(X)
            metrics["relevance"] = {
                "accuracy": accuracy_score(y_rel, rel_pred),
                "f1": f1_score(y_rel, rel_pred, average="weighted"),
            }
            logger.info(f"  Relevance acc={metrics['relevance']['accuracy']:.3f}")

        # ── Direction classifiers ─────────────────────────────────────────────
        for asset in ASSETS:
            for tf in timeframes:
                label_col = self._get_label_col(asset, tf)
                if label_col not in labels_df.columns:
                    logger.warning(f"  Skipping {asset}/{tf}: column {label_col} not found")
                    continue

                y = labels_df[label_col].values
                y_mapped = y + 1  # {-1,0,1} -> {0,1,2}
                class_counts = np.bincount(y_mapped, minlength=3)
                sample_weights = self._compute_sample_weights(y_mapped)

                logger.info(f"Training {asset}/{tf}: {len(y)} samples, "
                            f"class dist={class_counts}")

                # ── Optional Optuna tuning ────────────────────────────────────
                if tune:
                    best_params = self._tune_hyperparams(
                        X, y_mapped, sample_weights, n_splits, tune_trials, asset, tf
                    )
                    ensemble_configs = [best_params] + self._ENSEMBLE_CONFIGS[:3]
                else:
                    ensemble_configs = self._ENSEMBLE_CONFIGS

                # ── Cross-validation with ensemble ────────────────────────────
                skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
                cv_accs, cv_f1s = [], []

                for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_mapped)):
                    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
                    y_tr, y_val = y_mapped[train_idx], y_mapped[val_idx]
                    sw_tr = sample_weights[train_idx]

                    # Sub-split train for early stopping
                    X_train_es, X_es, y_train_es, y_es, sw_es, _ = train_test_split(
                        X_tr, y_tr, sw_tr, test_size=0.15,
                        stratify=y_tr, random_state=fold,
                    )

                    fold_probas = []
                    for cfg in ensemble_configs:
                        es_rounds = 50 if cfg.get("boosting_type") != "dart" else None
                        m = lgb.LGBMClassifier(
                            **cfg, random_state=42, verbose=-1, n_jobs=-1,
                        )
                        fit_kw = dict(sample_weight=sw_es)
                        if es_rounds:
                            fit_kw.update(
                                eval_set=[(X_es, y_es)],
                                callbacks=[lgb.early_stopping(es_rounds, verbose=False),
                                           lgb.log_evaluation(-1)],
                            )
                        m.fit(X_train_es, y_train_es, **fit_kw)
                        fold_probas.append(m.predict_proba(X_val))

                    avg_proba = np.mean(fold_probas, axis=0)
                    y_pred = np.argmax(avg_proba, axis=1)
                    cv_accs.append(accuracy_score(y_val, y_pred))
                    cv_f1s.append(f1_score(y_val, y_pred, average="weighted"))

                # ── Train final ensemble on all data ──────────────────────────
                # Sub-split for early stopping on full dataset
                X_main, X_es_full, y_main, y_es_full, sw_main, _ = train_test_split(
                    X, y_mapped, sample_weights, test_size=0.12,
                    stratify=y_mapped, random_state=99,
                )

                final_models = []
                for cfg in ensemble_configs:
                    es_rounds = 50 if cfg.get("boosting_type") != "dart" else None
                    m = lgb.LGBMClassifier(**cfg, random_state=42, verbose=-1, n_jobs=-1)
                    fit_kw = dict(sample_weight=sw_main)
                    if es_rounds:
                        fit_kw.update(
                            eval_set=[(X_es_full, y_es_full)],
                            callbacks=[lgb.early_stopping(es_rounds, verbose=False),
                                       lgb.log_evaluation(-1)],
                        )
                    m.fit(X_main, y_main, **fit_kw)
                    final_models.append(m)

                self.models[(asset, tf)] = final_models

                # Feature importance: average across ensemble
                avg_importance = np.mean(
                    [m.feature_importances_ for m in final_models], axis=0
                )
                self.feature_importance[(asset, tf)] = dict(
                    zip(self.feature_names, avg_importance)
                )

                # Full-data ensemble accuracy (in-sample reference)
                full_probas = np.mean(
                    [m.predict_proba(X) for m in final_models], axis=0
                )
                y_pred_full = np.argmax(full_probas, axis=1)
                metrics[f"{asset}_{tf}"] = {
                    "cv_accuracy": float(np.mean(cv_accs)),
                    "cv_f1": float(np.mean(cv_f1s)),
                    "cv_accuracy_std": float(np.std(cv_accs)),
                    "train_accuracy": float(accuracy_score(y_mapped, y_pred_full)),
                    "train_f1": float(f1_score(y_mapped, y_pred_full, average="weighted")),
                    "class_dist": class_counts.tolist(),
                    "n_ensemble_models": len(final_models),
                }
                logger.info(
                    f"  {asset}/{tf}: CV_acc={np.mean(cv_accs):.3f}±{np.std(cv_accs):.3f}, "
                    f"CV_f1={np.mean(cv_f1s):.3f}  [{len(final_models)}-model ensemble]"
                )

        return metrics

    def _tune_hyperparams(
        self,
        X: pd.DataFrame,
        y_mapped: np.ndarray,
        sample_weights: np.ndarray,
        n_splits: int,
        n_trials: int,
        asset: str,
        tf: str,
    ) -> dict:
        """Run Optuna to find best hyperparameters for one asset/timeframe."""
        try:
            import optuna
            import lightgbm as lgb
            from sklearn.model_selection import StratifiedKFold
            from sklearn.metrics import accuracy_score
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            logger.warning("optuna not installed; skipping tuning. pip install optuna")
            return self._ENSEMBLE_CONFIGS[0]

        def objective(trial):
            params = dict(
                n_estimators=trial.suggest_int("n_estimators", 300, 1000),
                max_depth=trial.suggest_int("max_depth", 4, 10),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                num_leaves=trial.suggest_int("num_leaves", 20, 80),
                subsample=trial.suggest_float("subsample", 0.6, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
                min_child_samples=trial.suggest_int("min_child_samples", 5, 30),
                reg_alpha=trial.suggest_float("reg_alpha", 0.01, 0.5, log=True),
                reg_lambda=trial.suggest_float("reg_lambda", 0.01, 0.5, log=True),
            )
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            accs = []
            for train_idx, val_idx in skf.split(X, y_mapped):
                X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_tr, y_val = y_mapped[train_idx], y_mapped[val_idx]
                m = lgb.LGBMClassifier(**params, random_state=42, verbose=-1, n_jobs=-1)
                m.fit(X_tr, y_tr, sample_weight=sample_weights[train_idx])
                accs.append(accuracy_score(y_val, m.predict(X_val)))
            return float(np.mean(accs))

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        logger.info(f"  Optuna {asset}/{tf}: best CV acc={study.best_value:.3f} "
                    f"params={study.best_params}")
        return study.best_params

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

            models = self.models[key]
            # Support both legacy single model and new ensemble list
            if not isinstance(models, list):
                models = [models]
            proba = np.mean([m.predict_proba(X)[0] for m in models], axis=0)
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

        # Save models (ensemble list or legacy single model)
        for (asset, tf), model in self.models.items():
            model_path = save_dir / f"model_{asset}_{tf}.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(model, f)   # saves list or single model transparently

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
