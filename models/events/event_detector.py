"""
Layer 3: Event Detection.

Classifies tweets into event categories relevant to market impact:
- trade_war, sanctions, geopolitical_conflict, monetary_policy,
  supply_chain, fiscal_policy, regulation, election, diplomacy,
  military, energy_policy, crypto_policy, no_event

Uses a trainable DistilBERT classifier with rule-based fallback.
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)

EVENT_TYPES = [
    "trade_war",
    "sanctions",
    "geopolitical_conflict",
    "monetary_policy",
    "supply_chain",
    "fiscal_policy",
    "regulation",
    "election",
    "diplomacy",
    "military",
    "energy_policy",
    "crypto_policy",
    "no_event",
]

# Rule-based keyword patterns for event detection (fallback & feature)
EVENT_PATTERNS = {
    "trade_war": [
        r"\btariff", r"\btrade\s+war", r"\btrade\s+deal", r"\btrade\s+barrier",
        r"\bimport\s+dut", r"\bexport\s+ban", r"\btrade\s+deficit",
        r"\breciprocal", r"\bretaliati",
    ],
    "sanctions": [
        r"\bsanction", r"\bembargo", r"\bfreeze\s+asset", r"\bblacklist",
        r"\bban\s+on", r"\brestriction",
    ],
    "geopolitical_conflict": [
        r"\bwar\b", r"\binvasion", r"\bconflict", r"\btension",
        r"\bescalat", r"\bnuclear", r"\bmissile", r"\bbomb",
        r"\battack", r"\baggressi",
    ],
    "monetary_policy": [
        r"\bfed\b", r"\bfederal\s+reserve", r"\binterest\s+rate",
        r"\brate\s+cut", r"\brate\s+hike", r"\bmonetary", r"\bquantitative",
        r"\binflation", r"\bdeflation", r"\bcpi\b", r"\bppi\b",
    ],
    "supply_chain": [
        r"\bsupply\s+chain", r"\bshipping", r"\bport\b", r"\blogistic",
        r"\bshortage", r"\bsupply\s+disrupt", r"\bchip\s+short",
    ],
    "fiscal_policy": [
        r"\btax\s+cut", r"\btax\s+hike", r"\btax\s+reform", r"\bfiscal",
        r"\bspending\b", r"\bbudget", r"\bdebt\s+ceil", r"\bshutdown",
        r"\bstimulus", r"\binfrastructure\s+bill",
    ],
    "regulation": [
        r"\bregulat", r"\bderegulat", r"\bexecutive\s+order",
        r"\banti-?trust", r"\bbreak\s+up",
    ],
    "election": [
        r"\belection", r"\bvote\b", r"\bvoting", r"\bcampaign",
        r"\bpoll\b", r"\bballot", r"\bdemocrat", r"\brepublican",
        r"\bcandidat",
    ],
    "diplomacy": [
        r"\bdeal\b", r"\bagreement", r"\bnegotiat", r"\bsummit",
        r"\bdiploma", r"\btreaty", r"\bpeace\s+deal", r"\bally\b",
        r"\ballianc",
    ],
    "military": [
        r"\bmilitary", r"\bdefense\b", r"\barmy\b", r"\bnavy\b",
        r"\bair\s+force", r"\bdeploy", r"\btroops?\b", r"\bnato\b",
        r"\bweapon",
    ],
    "energy_policy": [
        r"\boil\b", r"\bgas\b", r"\benergy", r"\bdrill",
        r"\bopec\b", r"\bpipeline", r"\bfracking", r"\bpetroleum",
        r"\brenewable",
    ],
    "crypto_policy": [
        r"\bbitcoin", r"\bcrypto", r"\bblockchain", r"\bdigital\s+curr",
        r"\bstablecoin", r"\bcbdc\b",
    ],
}


@dataclass
class EventDetectionResult:
    """Result of event detection for a single tweet."""
    events: list  # list of (event_type, confidence) tuples
    primary_event: str
    primary_confidence: float
    event_count: int

    def to_feature_dict(self) -> dict:
        """Convert to flat feature dictionary."""
        features = {
            "event_count": self.event_count,
            "event_primary_confidence": self.primary_confidence,
        }
        # One-hot for each event type
        detected_events = {e[0] for e in self.events}
        for evt in EVENT_TYPES:
            features[f"event_{evt}"] = int(evt in detected_events)
        # Confidence per event
        event_conf = {e[0]: e[1] for e in self.events}
        for evt in EVENT_TYPES:
            features[f"event_{evt}_conf"] = event_conf.get(evt, 0.0)
        return features


class EventDetectorHead(nn.Module):
    """Small classification head on top of a transformer encoder."""

    def __init__(self, hidden_size: int = 768, num_events: int = len(EVENT_TYPES),
                 dropout: float = 0.3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_events),
        )

    def forward(self, x):
        return self.classifier(x)


class EventDetector:
    """
    Multi-label event detector.

    Primary: trainable DistilBERT classifier.
    Fallback: rule-based keyword matching.
    """

    def __init__(self, model_name: str = "distilbert-base-uncased",
                 use_gpu: bool = True):
        self.model_name = model_name
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.encoder = None
        self.head = None
        self._trained = False

    def load(self):
        """Load base transformer encoder."""
        logger.info(f"Loading event detector encoder: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.encoder = AutoModel.from_pretrained(self.model_name)
        self.encoder.to(self.device)
        self.encoder.eval()

        self.head = EventDetectorHead(
            hidden_size=self.encoder.config.hidden_size,
            num_events=len(EVENT_TYPES),
        ).to(self.device)
        logger.info("Event detector loaded")

    def detect_rules(self, text: str) -> EventDetectionResult:
        """Rule-based event detection (always available)."""
        text_lower = text.lower()
        events = []

        for event_type, patterns in EVENT_PATTERNS.items():
            matches = 0
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    matches += 1
            if matches > 0:
                confidence = min(1.0, matches * 0.3 + 0.2)
                events.append((event_type, round(confidence, 3)))

        if not events:
            events = [("no_event", 0.9)]

        events.sort(key=lambda x: x[1], reverse=True)
        return EventDetectionResult(
            events=events,
            primary_event=events[0][0],
            primary_confidence=events[0][1],
            event_count=len([e for e in events if e[0] != "no_event"]),
        )

    def detect(self, text: str) -> EventDetectionResult:
        """Detect events using trained model or fallback to rules."""
        if not self._trained or self.encoder is None:
            return self.detect_rules(text)
        return self._model_detect(text)

    def _model_detect(self, text: str) -> EventDetectionResult:
        """Detect events using the trained classifier head."""
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True,
            max_length=256, padding=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.encoder(**inputs)
            cls_emb = outputs.last_hidden_state[:, 0, :]
            logits = self.head(cls_emb)
            probs = torch.sigmoid(logits).cpu().numpy()[0]

        events = []
        for idx, (evt, prob) in enumerate(zip(EVENT_TYPES, probs)):
            if prob > 0.3 or evt == "no_event":
                events.append((evt, round(float(prob), 3)))

        # Also combine with rule-based for robustness
        rule_result = self.detect_rules(text)
        rule_events = {e[0]: e[1] for e in rule_result.events}

        # Merge: take max confidence from model and rules
        merged = {}
        for evt, conf in events:
            merged[evt] = conf
        for evt, conf in rule_events.items():
            if evt in merged:
                merged[evt] = max(merged[evt], conf)
            else:
                merged[evt] = conf * 0.5  # discount pure rule matches

        events = [(k, v) for k, v in merged.items() if v > 0.2]
        events.sort(key=lambda x: x[1], reverse=True)

        if not events:
            events = [("no_event", 0.9)]

        return EventDetectionResult(
            events=events,
            primary_event=events[0][0],
            primary_confidence=events[0][1],
            event_count=len([e for e in events if e[0] != "no_event"]),
        )

    def get_embeddings(self, texts: list, batch_size: int = 32) -> np.ndarray:
        """Extract CLS embeddings for training."""
        if self.encoder is None:
            self.load()

        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = [t[:256] for t in texts[i:i + batch_size]]
            inputs = self.tokenizer(
                batch, return_tensors="pt", truncation=True,
                max_length=256, padding=True,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.encoder(**inputs)
                cls = outputs.last_hidden_state[:, 0, :]
            all_embeddings.append(cls.cpu().numpy())

        return np.vstack(all_embeddings)

    def train_head(self, texts: list, labels: np.ndarray,
                   epochs: int = 15, lr: float = 2e-4, batch_size: int = 32):
        """
        Train the classification head on labeled data.

        labels: (N, num_events) binary multi-label array
        """
        if self.encoder is None:
            self.load()

        logger.info(f"Training event detection head on {len(texts)} samples")

        # Get embeddings from frozen encoder
        embeddings = self.get_embeddings(texts, batch_size=batch_size)
        X = torch.tensor(embeddings, dtype=torch.float32)
        Y = torch.tensor(labels, dtype=torch.float32)

        dataset = torch.utils.data.TensorDataset(X, Y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(self.head.parameters(), lr=lr, weight_decay=0.01)
        criterion = nn.BCEWithLogitsLoss()

        self.head.train()
        for epoch in range(epochs):
            total_loss = 0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                logits = self.head(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if (epoch + 1) % 5 == 0:
                logger.info(f"  Epoch {epoch+1}/{epochs}, loss={total_loss/len(loader):.4f}")

        self.head.eval()
        self._trained = True
        logger.info("Event detection head training complete")

    def save(self, path: str):
        """Save the trained head."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.head.state_dict(), path)
        logger.info(f"Event detection head saved to {path}")

    def load_head(self, path: str):
        """Load a trained head."""
        if self.head is None:
            if self.encoder is None:
                self.load()
        self.head.load_state_dict(torch.load(path, map_location=self.device))
        self.head.eval()
        self._trained = True
        logger.info(f"Event detection head loaded from {path}")

    def batch_detect(self, texts: list) -> list:
        """Detect events for a batch of texts."""
        return [self.detect(t) for t in texts]
