"""
Layer 1: Entity Extraction using GLiNER (zero-shot NER).

Extracts: people, countries, commodities, currencies, organizations,
geopolitical entities, policy terms from tweets.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import torch

logger = logging.getLogger(__name__)

# Entity labels for zero-shot NER
ENTITY_LABELS = [
    "person",
    "country",
    "commodity",
    "currency",
    "organization",
    "geopolitical entity",
    "policy term",
    "financial instrument",
    "trade agreement",
    "sanction target",
]

# Mapping from extracted entities to canonical asset classes
ENTITY_TO_ASSET_MAP = {
    # Commodities
    "oil": ["cl"],
    "crude": ["cl"],
    "petroleum": ["cl"],
    "opec": ["cl"],
    "wheat": ["wheat"],
    "grain": ["wheat"],
    "corn": ["wheat"],
    "agriculture": ["wheat"],
    "gold": ["gold"],
    "silver": ["gold"],
    "precious metal": ["gold"],
    # Currencies / FX
    "dollar": ["eurodollar"],
    "euro": ["eurodollar"],
    "eur": ["eurodollar"],
    "usd": ["eurodollar"],
    "fx": ["eurodollar"],
    "currency": ["eurodollar"],
    # Crypto
    "bitcoin": ["btc"],
    "btc": ["btc"],
    "crypto": ["btc"],
    "cryptocurrency": ["btc"],
    # Equities
    "stock": ["equities"],
    "market": ["equities"],
    "s&p": ["equities"],
    "nasdaq": ["equities"],
    "dow": ["equities"],
    "wall street": ["equities"],
    # Rates
    "treasury": ["treasury_2y"],
    "bond": ["treasury_2y"],
    "yield": ["treasury_2y"],
    "fed": ["treasury_2y", "equities", "gold"],
    "interest rate": ["treasury_2y", "equities"],
    "federal reserve": ["treasury_2y", "equities", "gold"],
    # Countries -> multiple assets
    "china": ["equities", "gold", "eurodollar"],
    "russia": ["cl", "wheat", "gold"],
    "ukraine": ["wheat", "cl", "gold"],
    "iran": ["cl", "gold"],
    "saudi": ["cl"],
    "saudi arabia": ["cl"],
    "japan": ["equities", "eurodollar"],
    "europe": ["eurodollar", "equities"],
    "eu": ["eurodollar"],
    "canada": ["cl", "equities"],
    "mexico": ["equities"],
    "india": ["equities"],
    "korea": ["equities", "gold"],
    "north korea": ["gold", "equities"],
    # Policy terms
    "tariff": ["equities", "gold", "eurodollar"],
    "sanction": ["cl", "gold"],
    "trade war": ["equities", "gold", "eurodollar"],
    "trade deal": ["equities", "gold", "eurodollar"],
    "tax": ["equities", "treasury_2y"],
    "regulation": ["equities", "btc"],
    "deregulation": ["equities", "btc"],
    "executive order": ["equities"],
    "border": ["equities"],
    "immigration": ["equities"],
    "shutdown": ["equities", "treasury_2y"],
    "debt ceiling": ["treasury_2y", "equities"],
    "stimulus": ["equities", "gold", "treasury_2y"],
    "inflation": ["gold", "treasury_2y", "eurodollar"],
    "military": ["gold", "cl"],
    "war": ["gold", "cl", "wheat"],
    "peace": ["equities"],
    "nato": ["eurodollar", "gold"],
}


@dataclass
class Entity:
    """A single extracted entity."""
    text: str
    label: str
    score: float
    start: int = 0
    end: int = 0
    mapped_assets: list = field(default_factory=list)


@dataclass
class EntityExtractionResult:
    """Result of entity extraction for a single tweet."""
    entities: list  # List[Entity]
    asset_relevance: dict  # asset -> relevance score
    entity_count: int = 0
    has_policy_entity: bool = False
    has_country_entity: bool = False
    has_commodity_entity: bool = False
    has_person_entity: bool = False

    def to_feature_dict(self) -> dict:
        """Convert to flat feature dictionary for ML pipeline."""
        features = {
            "ner_entity_count": self.entity_count,
            "ner_has_policy": int(self.has_policy_entity),
            "ner_has_country": int(self.has_country_entity),
            "ner_has_commodity": int(self.has_commodity_entity),
            "ner_has_person": int(self.has_person_entity),
            "ner_unique_labels": len(set(e.label for e in self.entities)),
        }
        # Asset relevance scores
        for asset in ["gold", "equities", "btc", "cl", "wheat", "eurodollar", "treasury_2y"]:
            features[f"ner_asset_relevance_{asset}"] = self.asset_relevance.get(asset, 0.0)
        return features


class EntityExtractor:
    """
    GLiNER-based zero-shot entity extractor.
    Falls back to keyword matching if GLiNER unavailable.
    """

    def __init__(self, model_name: str = "urchade/gliner_medium-v2.1", use_gpu: bool = True):
        self.model_name = model_name
        self.model = None
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        self._use_fallback = False

    def load(self):
        """Load the GLiNER model."""
        try:
            from gliner import GLiNER
            logger.info(f"Loading GLiNER model: {self.model_name}")
            self.model = GLiNER.from_pretrained(self.model_name)
            if self.device == "cuda":
                self.model = self.model.to(self.device)
            logger.info(f"GLiNER loaded on {self.device}")
        except Exception as e:
            logger.warning(f"GLiNER load failed ({e}), using keyword fallback")
            self._use_fallback = True

    def extract(self, text: str, threshold: float = 0.3) -> EntityExtractionResult:
        """Extract entities from tweet text."""
        if self._use_fallback or self.model is None:
            return self._keyword_fallback(text)
        return self._gliner_extract(text, threshold)

    def _gliner_extract(self, text: str, threshold: float) -> EntityExtractionResult:
        """Extract entities using GLiNER model."""
        try:
            raw_entities = self.model.predict_entities(
                text, ENTITY_LABELS, threshold=threshold
            )
        except Exception as e:
            logger.warning(f"GLiNER inference failed: {e}")
            return self._keyword_fallback(text)

        entities = []
        asset_relevance = {}

        for ent in raw_entities:
            mapped_assets = self._map_entity_to_assets(ent["text"])
            entity = Entity(
                text=ent["text"],
                label=ent["label"],
                score=ent["score"],
                start=ent.get("start", 0),
                end=ent.get("end", 0),
                mapped_assets=mapped_assets,
            )
            entities.append(entity)

            # Accumulate asset relevance
            for asset in mapped_assets:
                asset_relevance[asset] = max(
                    asset_relevance.get(asset, 0.0), ent["score"]
                )

        labels = set(e.label for e in entities)
        return EntityExtractionResult(
            entities=entities,
            asset_relevance=asset_relevance,
            entity_count=len(entities),
            has_policy_entity="policy term" in labels or "trade agreement" in labels or "sanction target" in labels,
            has_country_entity="country" in labels or "geopolitical entity" in labels,
            has_commodity_entity="commodity" in labels,
            has_person_entity="person" in labels,
        )

    def _keyword_fallback(self, text: str) -> EntityExtractionResult:
        """Rule-based fallback when GLiNER is unavailable."""
        text_lower = text.lower()
        entities = []
        asset_relevance = {}

        for keyword, assets in ENTITY_TO_ASSET_MAP.items():
            if keyword in text_lower:
                # Determine label
                label = "policy term"
                if keyword in ("china", "russia", "ukraine", "iran", "saudi", "japan",
                               "europe", "eu", "canada", "mexico", "india", "korea",
                               "north korea", "saudi arabia"):
                    label = "country"
                elif keyword in ("oil", "crude", "wheat", "grain", "corn", "gold", "silver",
                                 "petroleum"):
                    label = "commodity"
                elif keyword in ("dollar", "euro", "eur", "usd", "currency", "bitcoin",
                                 "btc", "crypto", "cryptocurrency"):
                    label = "currency"
                elif keyword in ("opec", "fed", "federal reserve", "nato"):
                    label = "organization"

                entity = Entity(
                    text=keyword, label=label, score=0.8,
                    mapped_assets=assets,
                )
                entities.append(entity)

                for asset in assets:
                    asset_relevance[asset] = max(asset_relevance.get(asset, 0.0), 0.8)

        labels = set(e.label for e in entities)
        return EntityExtractionResult(
            entities=entities,
            asset_relevance=asset_relevance,
            entity_count=len(entities),
            has_policy_entity="policy term" in labels,
            has_country_entity="country" in labels,
            has_commodity_entity="commodity" in labels,
            has_person_entity="person" in labels,
        )

    @staticmethod
    def _map_entity_to_assets(entity_text: str) -> list:
        """Map an entity string to relevant asset classes."""
        text_lower = entity_text.lower().strip()
        mapped = set()
        for keyword, assets in ENTITY_TO_ASSET_MAP.items():
            if keyword in text_lower or text_lower in keyword:
                mapped.update(assets)
        return list(mapped)

    def batch_extract(self, texts: list, threshold: float = 0.3) -> list:
        """Extract entities from a batch of tweets."""
        return [self.extract(t, threshold) for t in texts]
