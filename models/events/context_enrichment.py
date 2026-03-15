"""
Layer 4: Context Enrichment.

Adds temporal context to tweets using:
1. Timestamp-based features (hour, day of week, market hours)
2. Optional Tavily news API search for recent events
3. Thread context from surrounding tweets
"""

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# US market hours (ET)
MARKET_OPEN_HOUR = 9   # 9:30 AM ET
MARKET_CLOSE_HOUR = 16  # 4:00 PM ET

# Key policy keywords for context search
CONTEXT_KEYWORDS = {
    "tariff", "trade", "china", "russia", "iran", "sanction", "tax",
    "fed", "rate", "oil", "bitcoin", "crypto", "nato", "military",
    "war", "peace", "deal", "shutdown", "debt", "inflation",
    "executive order", "regulation", "deregulation",
}


@dataclass
class ContextFeatures:
    """Temporal and contextual features for a tweet."""
    # Temporal
    hour: int = 0
    day_of_week: int = 0
    is_weekend: bool = False
    is_market_hours: bool = False
    is_pre_market: bool = False
    is_post_market: bool = False

    # Tweet features
    tweet_length: int = 0
    has_url: bool = False
    has_exclamation: bool = False
    is_all_caps_ratio: float = 0.0
    word_count: int = 0
    has_numbers: bool = False

    # Thread features
    thread_position: int = 0  # position in conversation thread
    minutes_since_last_tweet: float = 0.0
    is_rapid_fire: bool = False  # < 5 minutes since last tweet

    # Macro context (from search or cache)
    macro_context_length: int = 0
    macro_has_tariff: bool = False
    macro_has_conflict: bool = False
    macro_has_rate: bool = False

    def to_feature_dict(self) -> dict:
        """Convert to flat feature dictionary."""
        return {
            "ctx_hour": self.hour,
            "ctx_day_of_week": self.day_of_week,
            "ctx_is_weekend": int(self.is_weekend),
            "ctx_is_market_hours": int(self.is_market_hours),
            "ctx_is_pre_market": int(self.is_pre_market),
            "ctx_is_post_market": int(self.is_post_market),
            "ctx_tweet_length": self.tweet_length,
            "ctx_has_url": int(self.has_url),
            "ctx_has_exclamation": int(self.has_exclamation),
            "ctx_all_caps_ratio": self.is_all_caps_ratio,
            "ctx_word_count": self.word_count,
            "ctx_has_numbers": int(self.has_numbers),
            "ctx_thread_position": self.thread_position,
            "ctx_minutes_since_last": self.minutes_since_last_tweet,
            "ctx_is_rapid_fire": int(self.is_rapid_fire),
            "ctx_macro_context_length": self.macro_context_length,
            "ctx_macro_has_tariff": int(self.macro_has_tariff),
            "ctx_macro_has_conflict": int(self.macro_has_conflict),
            "ctx_macro_has_rate": int(self.macro_has_rate),
        }


class ContextEnricher:
    """
    Enriches tweets with temporal, textual, and macro context features.

    Optionally uses Tavily API for real-time macro context.
    """

    def __init__(
        self,
        tavily_api_key: Optional[str] = None,
        enable_live_search: bool = False,
    ):
        self.tavily_client = None
        self.enable_live_search = enable_live_search
        self._search_cache = {}
        if tavily_api_key:
            try:
                from tavily import TavilyClient
                self.tavily_client = TavilyClient(api_key=tavily_api_key)
                logger.info("Tavily client initialized")
            except ImportError:
                logger.warning("tavily-python not installed, skipping search context")

    def enrich(
        self,
        text: str,
        created_at: str = "",
        prev_tweet_time: Optional[str] = None,
        macro_context: str = "",
        thread_position: int = 0,
    ) -> ContextFeatures:
        """Compute context features for a single tweet."""
        features = ContextFeatures()

        # --- Temporal features ---
        try:
            dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            features.hour = dt.hour
            features.day_of_week = dt.weekday()
            features.is_weekend = dt.weekday() >= 5
            features.is_market_hours = MARKET_OPEN_HOUR <= dt.hour < MARKET_CLOSE_HOUR and not features.is_weekend
            features.is_pre_market = 4 <= dt.hour < MARKET_OPEN_HOUR and not features.is_weekend
            features.is_post_market = MARKET_CLOSE_HOUR <= dt.hour < 20 and not features.is_weekend
        except (ValueError, AttributeError):
            pass

        # --- Tweet text features ---
        features.tweet_length = len(text)
        features.word_count = len(text.split())
        features.has_url = bool(re.search(r"https?://", text))
        features.has_exclamation = "!" in text
        features.has_numbers = bool(re.search(r"\d+%?", text))

        # All-caps ratio
        alpha_chars = [c for c in text if c.isalpha()]
        if alpha_chars:
            features.is_all_caps_ratio = sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)

        # --- Thread features ---
        features.thread_position = thread_position
        if prev_tweet_time:
            try:
                dt_prev = datetime.fromisoformat(prev_tweet_time.replace("Z", "+00:00"))
                dt_curr = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                delta = (dt_curr - dt_prev).total_seconds() / 60.0
                features.minutes_since_last_tweet = max(0, delta)
                features.is_rapid_fire = delta < 5
            except (ValueError, AttributeError):
                pass

        # --- Optional live macro context via Tavily ---
        if not macro_context and self.enable_live_search and self.tavily_client:
            date_str = ""
            if created_at:
                try:
                    dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                    date_str = dt.strftime("%Y-%m-%d")
                except (ValueError, AttributeError):
                    date_str = ""
            macro_context = self.search_context(text=text, date_str=date_str)

        # --- Macro context ---
        if macro_context:
            mc_lower = macro_context.lower()
            features.macro_context_length = len(macro_context)
            features.macro_has_tariff = "tariff" in mc_lower or "trade war" in mc_lower
            features.macro_has_conflict = any(w in mc_lower for w in ["war", "conflict", "tension", "military"])
            features.macro_has_rate = any(w in mc_lower for w in ["rate", "fed", "monetary", "inflation"])

        return features

    def search_context(self, text: str, date_str: str = "") -> str:
        """Fetch macro context via Tavily search."""
        if not self.tavily_client:
            return ""

        # Extract keywords from tweet
        text_lower = text.lower()
        found_keywords = [kw for kw in CONTEXT_KEYWORDS if kw in text_lower]

        if not found_keywords:
            return ""

        query = f"US markets {' '.join(found_keywords[:3])} {date_str}"
        query = query.strip()

        if query in self._search_cache:
            return self._search_cache[query]

        try:
            result = self.tavily_client.search(
                query=query,
                search_depth="basic",
                max_results=3,
                include_answer=True,
            )
            context_parts = []
            if result.get("answer"):
                context_parts.append(result["answer"])
            for r in result.get("results", [])[:2]:
                snippet = r.get("content", "")[:300]
                if snippet:
                    context_parts.append(snippet)
            context = "\n".join(context_parts)[:2000]
            self._search_cache[query] = context
            return context
        except Exception as e:
            logger.warning(f"Tavily search failed: {e}")
            return ""

    def batch_enrich(
        self,
        texts: list,
        timestamps: list,
        macro_contexts: Optional[list] = None,
    ) -> list:
        """Enrich a batch of tweets."""
        if macro_contexts is None:
            macro_contexts = [""] * len(texts)

        results = []
        for i, (text, ts) in enumerate(zip(texts, timestamps)):
            prev_ts = timestamps[i - 1] if i > 0 else None
            ctx = self.enrich(
                text=text,
                created_at=ts,
                prev_tweet_time=prev_ts,
                macro_context=macro_contexts[i] if i < len(macro_contexts) else "",
                thread_position=i,
            )
            results.append(ctx)

        return results
