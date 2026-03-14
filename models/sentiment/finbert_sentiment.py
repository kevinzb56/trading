"""
Layer 2: Financial Sentiment Analysis using FinBERT.

Outputs: positive / negative / neutral + confidence scores.
Also extracts text embeddings for downstream use.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logger = logging.getLogger(__name__)

FINBERT_MODEL = "ProsusAI/finbert"


@dataclass
class SentimentResult:
    """Sentiment analysis result for a single tweet."""
    positive: float
    negative: float
    neutral: float
    label: str  # "positive", "negative", "neutral"
    score: float  # confidence of the predicted label
    compound: float  # positive - negative (range: -1 to 1)

    def to_feature_dict(self) -> dict:
        """Convert to flat feature dictionary."""
        return {
            "sentiment_positive": self.positive,
            "sentiment_negative": self.negative,
            "sentiment_neutral": self.neutral,
            "sentiment_compound": self.compound,
            "sentiment_label_pos": int(self.label == "positive"),
            "sentiment_label_neg": int(self.label == "negative"),
            "sentiment_label_neu": int(self.label == "neutral"),
            "sentiment_confidence": self.score,
        }


class FinBERTSentiment:
    """
    FinBERT-based financial sentiment analyzer.

    Uses ProsusAI/finbert — pre-trained on financial text.
    """

    def __init__(self, model_name: str = FINBERT_MODEL, use_gpu: bool = True):
        self.model_name = model_name
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None
        self._label_map = {0: "positive", 1: "negative", 2: "neutral"}

    def load(self):
        """Load FinBERT model and tokenizer."""
        logger.info(f"Loading FinBERT: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"FinBERT loaded on {self.device}")

    def analyze(self, text: str) -> SentimentResult:
        """Analyze sentiment of a single text."""
        if self.model is None:
            self.load()

        # Truncate long texts
        text = text[:512]

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

        probs_np = probs.cpu().numpy()[0]
        positive, negative, neutral = float(probs_np[0]), float(probs_np[1]), float(probs_np[2])

        label_idx = int(np.argmax(probs_np))
        label = self._label_map[label_idx]
        score = float(probs_np[label_idx])
        compound = positive - negative

        return SentimentResult(
            positive=positive,
            negative=negative,
            neutral=neutral,
            label=label,
            score=score,
            compound=compound,
        )

    def batch_analyze(self, texts: list, batch_size: int = 32) -> list:
        """Analyze sentiment for a batch of texts."""
        if self.model is None:
            self.load()

        results = []
        for i in range(0, len(texts), batch_size):
            batch_texts = [t[:512] for t in texts[i:i + batch_size]]
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

            for j in range(probs.shape[0]):
                probs_np = probs[j].cpu().numpy()
                positive, negative, neutral = (
                    float(probs_np[0]),
                    float(probs_np[1]),
                    float(probs_np[2]),
                )
                label_idx = int(np.argmax(probs_np))
                label = self._label_map[label_idx]
                score = float(probs_np[label_idx])
                compound = positive - negative

                results.append(SentimentResult(
                    positive=positive,
                    negative=negative,
                    neutral=neutral,
                    label=label,
                    score=score,
                    compound=compound,
                ))

        return results

    def get_embeddings(self, texts: list, batch_size: int = 32) -> np.ndarray:
        """
        Extract CLS token embeddings from FinBERT for downstream use.
        Returns: (N, hidden_dim) numpy array.
        """
        if self.model is None:
            self.load()

        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = [t[:512] for t in texts[i:i + batch_size]]
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.bert(**{k: v for k, v in inputs.items()
                                             if k in ("input_ids", "attention_mask", "token_type_ids")})
                # CLS token embedding
                cls_embeddings = outputs.last_hidden_state[:, 0, :]

            all_embeddings.append(cls_embeddings.cpu().numpy())

        return np.vstack(all_embeddings)
