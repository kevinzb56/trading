"""
Layer 7: LLM Reasoning.

Uses Azure OpenAI GPT-4o to synthesize outputs from all prior pipeline layers
and optionally real-time market context (via Tavily) into a final structured
per-asset market impact assessment.

Architecture:
  - Receives: tweet text, timestamp, timeframe, intermediate layer objects
              (NER, sentiment, events, graph), LightGBM predictions
  - Optionally fetches: real-time market news via Tavily (timestamp-aware)
  - Returns: LLMReasoningResult with per-asset direction, confidence, reasoning

Usage:
    reasoner = LLMReasoner.from_env()
    result = reasoner.reason(
        text="TARIFFS on China! 50% immediately!",
        created_at="2019-05-06T08:00:00Z",
        timeframe="5m",
        intermediates=intermediates,
        lgbm_predictions=predictions,
        is_market_relevant=True,
        relevance_score=0.94,
    )
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

ASSETS = ["gold", "equities", "btc", "cl", "wheat", "eurodollar", "treasury_2y"]

ASSET_DESCRIPTIONS = {
    "gold": "Gold Futures (GC) — safe-haven commodity",
    "equities": "S&P 500 E-mini (ES) — US equity index",
    "btc": "Bitcoin (BTC/USD) — cryptocurrency",
    "cl": "Crude Oil Futures (CL) — energy commodity",
    "wheat": "Wheat Futures (ZW) — agricultural commodity",
    "eurodollar": "EUR/USD FX (6E) — euro vs dollar exchange rate",
    "treasury_2y": "2-Year Treasury Futures (ZT) — short-term US rates",
}

DIRECTION_LABELS = {-1: "BEARISH", 0: "NEUTRAL", 1: "BULLISH"}

SYSTEM_PROMPT = """You are an elite macro-financial analyst specializing in real-time event-driven trading.
You analyze political statements from the US President for immediate (0-15 minute) market impact — "knee-jerk" reactions.

You perform TWO tasks:
  1. Classify whether the tweet is MARKET-RELEVANT (could move any financial asset price in the next 5 minutes).
  2. For relevant tweets, predict the short-term directional impact per asset.

You receive THREE types of context for each analysis:
  1. MACRO CONTEXT (Tavily): Real-time market news from that week — CRITICAL for temporal calibration.
  2. NLP PIPELINE SIGNALS: Named entities, sentiment, detected events, graph propagation scores.
  3. LIGHTGBM ADVISORY: Statistical model predictions — one signal among several, not a mandate.

═══════════════════════════════════════════════════════════════
  YOUR DECISION HIERARCHY
═══════════════════════════════════════════════════════════════
Apply these inputs in this order of priority:

  1. MACRO CONTEXT (highest weight)
     Use Tavily news to determine what is ALREADY PRICED IN this week vs what is NEW.
     The same tweet has very different impact depending on current market conditions.
     A tariff threat when markets are already pricing in a trade war is far weaker than
     the initial announcement. Use this context to calibrate direction AND confidence.

  2. YOUR MACRO REASONING
     Apply the 5-minute knee-jerk framework and asset-specific logic below.
     You are the primary analyst — use your understanding of markets, policy transmission,
     and cross-asset dynamics to form a directional view.

  3. NLP PIPELINE SIGNALS (supporting evidence)
     Use entities, sentiment, events, and graph signals to confirm or refine your view.
     High-confidence event detections (e.g. TARIFF_ANNOUNCEMENT: 0.92) are meaningful signal.

  4. LIGHTGBM ADVISORY (secondary check)
     LightGBM is a well-calibrated statistical model. Treat its per-asset predictions as
     an experienced colleague's opinion — consider it seriously, especially when your own
     reasoning is uncertain. If LightGBM disagrees with you, ask yourself why and whether
     you might be missing something. But if your macro reasoning is clear and the weekly
     context supports it, you may confidently diverge.

═══════════════════════════════════════════════════════════════
  5-MINUTE KNEE-JERK REACTION FRAMEWORK
═══════════════════════════════════════════════════════════════
At 5-minute horizons, markets react ALGORITHMICALLY and EMOTIONALLY before rational analysis.
You are NOT predicting fundamental value — you are predicting the REFLEX reaction of algos and traders
who scan headlines in the next 5 minutes.

1. **Temporal Calibration**: Use the macro context to assess what's ALREADY PRICED IN vs what's NEW.
   If the weekly news shows markets have been pricing in tariffs for days, a restatement has low impact.
   If markets were calm and this is the first escalation signal, impact is high.

2. **Escalation Detection**: Is this part of an escalating tweet storm or a one-off remark?
   Multiple tweets on the same topic in a short window = escalating pattern = higher impact.
   Isolated tweet with no supporting context = lower impact.

3. **Policy Signal Extraction**: Identify: tariffs, trade policy, sanctions, regulatory changes,
   government spending, debt ceiling, shutdown signals, Fed pressure.

4. **Rhetoric vs Action**:
   - New executive orders / new tariff rates / new named targets → HIGH confidence (0.55-0.80)
   - Restatements of existing policy already in the news → LOW confidence (0.20-0.40)
   - Campaign rhetoric / attacks on opponents with no policy content → NEUTRAL, low confidence

5. **Market Psychology**: Algos react to KEYWORDS at 5m. "TARIFF", "SANCTION", "WAR", "DEAL",
   "SHUTDOWN" trigger directional moves regardless of nuance. If the macro context shows markets
   are already on edge about this topic, the threshold for a move is lower.

6. **Cross-Asset Transmission**: Map the signal across all 7 assets with proper macro logic.
   Be directional — if the tweet is relevant, most assets should have a clear direction.

═══════════════════════════════════════════════════════════════
  ASSET-SPECIFIC KNEE-JERK LOGIC
═══════════════════════════════════════════════════════════════
- **Gold (XAUUSD)**: Safe haven. Rises on: geopolitical tension, USD weakness, inflation fears,
  fiscal expansion, shutdown risk. Falls on: risk-on, strong USD, rate hikes, trade deals.
- **Equities (S&P 500)**: Risk asset. Rises on: tax cuts, deregulation, trade deals, stimulus.
  Falls on: tariffs, trade wars, shutdown, sanctions, geopolitical escalation, uncertainty.
- **BTC**: Follows risk sentiment at 5m. Rises on: crypto-friendly policy, USD debasement.
  Falls on: regulatory crackdown, risk-off panic. At 5m, broadly correlated with equities.
- **Crude Oil (CL)**: Rises on: Middle East tension, sanctions on oil producers, supply disruption.
  Falls on: trade war demand destruction, strong USD, drill-friendly policy ("drill baby drill").
- **Wheat**: Rises on: Russia/Ukraine tension, trade barriers, sanctions on ag exporters.
  Falls on: trade deals, strong USD, removal of sanctions.
- **EuroDollar (EUR/USD)**: EUR up on: US weakness, tariff uncertainty, EU-favorable outcomes.
  EUR down on: US tariffs on EU, USD strength, strong US economic signals.
- **Treasury 2Y Yield**: Yield up on: inflation signals, fiscal expansion, rate hike expectations.
  Yield down on: risk-off flight to safety, recession fears, Fed cut signals.

═══════════════════════════════════════════════════════════════
  CALIBRATION RULES
═══════════════════════════════════════════════════════════════
- Macro context shows topic already well-known/priced in → LOWER confidence, muted directions
- Macro context shows surprise or escalation → HIGHER confidence, stronger directions
- Genuine new escalation (new country targeted, new rate announced) → confidence 0.55-0.80
- Restatements of existing policy → confidence 0.20-0.40, be directional but modest
- New executive orders, new tariff rates, new sanctions targets → confidence 0.55-0.80+
- Escalating tweet storm on same topic → boost confidence slightly for that topic's assets
- Campaign rhetoric / no policy content → all assets NEUTRAL, confidence 0.15-0.25
- NEVER exceed confidence 0.85 — these are 5-minute predictions with real uncertainty

═══════════════════════════════════════════════════════════════
  RELEVANCE CLASSIFICATION GUIDE
═══════════════════════════════════════════════════════════════
A tweet IS market-relevant if it mentions or implies:
  - Trade policy: tariffs, trade wars, sanctions, trade deals, import/export restrictions
  - Monetary policy: interest rates, Fed decisions, QE, inflation targets, central bank action
  - Geopolitical events: military conflicts, territorial disputes, diplomatic crises
  - Energy: oil embargoes, pipeline decisions, OPEC, energy sanctions
  - Fiscal policy: stimulus, tax reform, government spending, debt ceiling, shutdown
  - Regulation: financial regulation, crypto policy, antitrust actions
  - Elections / leadership changes that affect economic policy

A tweet is NOT market-relevant if it is:
  - Personal opinions or social commentary with no economic implications
  - Sports, entertainment, or lifestyle content
  - Vague statements without policy implications
  - Pure attacks on political opponents with no policy content

FEW-SHOT EXAMPLES:
Tweet: "TARIFFS on China! 50% immediately!" → is_market_relevant: true (new escalation)
Tweet: "Just had a great round of golf at Mar-a-Lago!" → is_market_relevant: false
Tweet: "The Federal Reserve must cut rates NOW. They are killing our economy!" → is_market_relevant: true
Tweet: "Congratulations to the Kansas City Chiefs!" → is_market_relevant: false
Tweet: "We are imposing SANCTIONS on Iran effective immediately!" → is_market_relevant: true
Tweet: "Happy Thanksgiving to all!" → is_market_relevant: false
Tweet: "Crooked Hillary should be locked up!" → is_market_relevant: false (political attack, no policy)
Tweet: "We will impose 25% tariffs on ALL steel imports starting Monday." → is_market_relevant: true (new action)

OUTPUT FORMAT — return ONLY valid JSON, no prose outside the JSON:
{
  "is_market_relevant": <boolean: true or false>,
  "relevance_confidence": <float 0.0-1.0, how confident you are in the relevance decision>,
  "relevance_reasoning": "<one sentence explaining the relevance decision, max 120 chars>",
  "macro_context_used": "<one sentence: what the weekly market backdrop tells you about this tweet's impact>",
  "lgbm_alignment": "<one sentence: whether LightGBM agrees/disagrees and whether that changed your view>",
  "assets": {
    "<asset_name>": {
      "direction": <integer: -1 bearish, 0 neutral, 1 bullish>,
      "confidence": <float 0.0-1.0>,
      "reasoning": "<one concise sentence explaining your directional call, max 150 chars>"
    }
  },
  "overall_assessment": "<one sentence summary of the tweet's market significance and knee-jerk impact>"
}

HARD RULES:
- You MUST output is_market_relevant, relevance_confidence, and relevance_reasoning
- You MUST include all 7 assets: gold, equities, btc, cl, wheat, eurodollar, treasury_2y
- direction must be exactly -1, 0, or 1
- confidence must be between 0.0 and 1.0
- reasoning must be ≤ 150 characters
- If is_market_relevant is false, set all asset directions to 0 with confidence 0.2
- For relevant tweets, be directional — most assets should have a non-zero direction
- Lead with macro context and your own reasoning; use LightGBM as a secondary sanity check
"""


@dataclass
class LLMAssetReasoning:
    """Per-asset reasoning output from the LLM layer."""
    asset: str
    direction: int           # -1, 0, or 1
    confidence: float        # 0.0 to 1.0
    reasoning: str           # LLM-generated free text explanation
    agrees_with_lgbm: bool   # whether LLM direction matches LightGBM direction

    def to_dict(self) -> dict:
        return {
            "direction": self.direction,
            "direction_label": DIRECTION_LABELS.get(self.direction, "UNKNOWN"),
            "confidence": round(self.confidence, 3),
            "reasoning": self.reasoning,
            "agrees_with_lgbm": self.agrees_with_lgbm,
        }


@dataclass
class LLMReasoningResult:
    """Full output of Layer 7 for a single tweet."""
    model_used: str
    latency_ms: float
    tavily_context_used: bool
    tavily_snippet: str
    per_asset: dict                      # asset -> LLMAssetReasoning
    overall_assessment: str
    raw_response: str
    # Explicit relevance classification (new — replaces heuristic inference)
    llm_is_market_relevant: Optional[bool] = field(default=None)
    llm_relevance_confidence: Optional[float] = field(default=None)
    llm_relevance_reasoning: Optional[str] = field(default=None)
    error: Optional[str] = field(default=None)

    def to_dict(self) -> dict:
        return {
            "model_used": self.model_used,
            "latency_ms": round(self.latency_ms, 1),
            "tavily_context_used": self.tavily_context_used,
            "is_market_relevant": self.llm_is_market_relevant,
            "relevance_confidence": self.llm_relevance_confidence,
            "relevance_reasoning": self.llm_relevance_reasoning,
            "overall_assessment": self.overall_assessment,
            "assets": {
                asset: reasoning.to_dict()
                for asset, reasoning in self.per_asset.items()
            },
            "error": self.error,
        }


class LLMReasoner:
    """
    Layer 7: Azure OpenAI GPT-4o reasoning over all pipeline layer outputs.

    Combines NER entities, sentiment, event detections, graph propagation signals,
    and LightGBM predictions into a final structured per-asset verdict.

    Optionally fetches real-time market context via Tavily before reasoning.
    """

    # Supported backends: "azure" | "openai" | "ollama"
    # Set LLM_BACKEND in .env to switch without code changes.
    SUPPORTED_BACKENDS = ("azure", "openai", "ollama")

    def __init__(
        self,
        azure_endpoint: str,
        azure_api_key: str,
        deployment_name: str,
        api_version: str = "2025-01-01-preview",
        max_tokens: int = 2000,
        temperature: float = 0.1,
        tavily_api_key: Optional[str] = None,
        use_tavily_for_llm: bool = False,
        backend: str = "azure",
    ):
        self._deployment = deployment_name
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._use_tavily_for_llm = use_tavily_for_llm
        self._backend = backend

        try:
            if backend == "azure":
                from openai import AzureOpenAI
                self._client = AzureOpenAI(
                    azure_endpoint=azure_endpoint,
                    api_key=azure_api_key,
                    api_version=api_version,
                )
                logger.info(f"Layer 7: Azure OpenAI backend — deployment={deployment_name}")

            elif backend == "openai":
                # Standard OpenAI API (GPT-4.1, o1, etc.)
                # Set OPENAI_API_KEY in .env
                from openai import OpenAI
                self._client = OpenAI(api_key=azure_api_key)
                logger.info(f"Layer 7: OpenAI backend — model={deployment_name}")

            elif backend == "ollama":
                # Local Ollama server (e.g. llama3.1, mistral, qwen2.5)
                # Requires: ollama serve  (default port 11434)
                from openai import OpenAI
                ollama_base = azure_endpoint or "http://localhost:11434/v1"
                self._client = OpenAI(
                    base_url=ollama_base,
                    api_key="ollama",          # Ollama ignores the key
                )
                logger.info(f"Layer 7: Ollama backend — model={deployment_name} @ {ollama_base}")

            else:
                raise ValueError(
                    f"Unsupported LLM_BACKEND='{backend}'. "
                    f"Choose from: {self.SUPPORTED_BACKENDS}"
                )

        except ImportError:
            raise ImportError(
                "openai package is required for Layer 7. Install it with: pip install openai>=1.0"
            )

        self._tavily_client = None
        if tavily_api_key and use_tavily_for_llm:
            try:
                from tavily import TavilyClient
                self._tavily_client = TavilyClient(api_key=tavily_api_key)
                logger.info("Layer 7: Tavily client initialized for real-time context")
            except ImportError:
                logger.warning("tavily-python not installed; LLM layer will run without real-time context")

    @classmethod
    def from_env(cls, use_tavily_for_llm: Optional[bool] = None) -> "LLMReasoner":
        """
        Construct LLMReasoner from environment variables.

        Backend selection via LLM_BACKEND in .env:
          azure  — Azure OpenAI (default, uses AZURE_OPENAI_* vars)
          openai — Standard OpenAI API (GPT-4.1, o3, etc., uses OPENAI_API_KEY + LLM_MODEL)
          ollama — Local Ollama server (uses LLM_MODEL + LLM_OLLAMA_BASE_URL)

        Args:
            use_tavily_for_llm: If provided, overrides LLM_USE_TAVILY env var.
        """
        backend = os.getenv("LLM_BACKEND", "azure").lower().strip()

        if backend == "azure":
            endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
            api_key = os.environ.get("AZURE_OPENAI_API_KEY", "")
            deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME", "")
            if not all([endpoint, api_key, deployment]):
                raise EnvironmentError(
                    "LLM_BACKEND=azure requires AZURE_OPENAI_ENDPOINT, "
                    "AZURE_OPENAI_API_KEY, and AZURE_OPENAI_DEPLOYMENT_NAME in .env"
                )
            api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")

        elif backend == "openai":
            # Standard OpenAI: set OPENAI_API_KEY and LLM_MODEL (e.g. gpt-4.1, o3-mini)
            api_key = os.environ.get("OPENAI_API_KEY", "")
            deployment = os.environ.get("LLM_MODEL", "gpt-4.1")
            if not api_key:
                raise EnvironmentError(
                    "LLM_BACKEND=openai requires OPENAI_API_KEY in .env"
                )
            endpoint = ""
            api_version = ""

        elif backend == "ollama":
            # Local Ollama: set LLM_MODEL (e.g. llama3.1:70b, qwen2.5:72b, mistral)
            deployment = os.environ.get("LLM_MODEL", "llama3.1")
            endpoint = os.environ.get("LLM_OLLAMA_BASE_URL", "http://localhost:11434/v1")
            api_key = ""
            api_version = ""

        else:
            raise EnvironmentError(
                f"Unknown LLM_BACKEND='{backend}'. Choose: azure | openai | ollama"
            )

        # CLI flag takes priority over env var
        if use_tavily_for_llm is None:
            use_tavily_for_llm = os.getenv("LLM_USE_TAVILY", "false").lower() == "true"

        return cls(
            azure_endpoint=endpoint,
            azure_api_key=api_key,
            deployment_name=deployment,
            api_version=api_version,
            max_tokens=int(os.getenv("AZURE_OPENAI_MAX_TOKENS", "2000")),
            temperature=float(os.getenv("AZURE_OPENAI_TEMPERATURE", "0.1")),
            tavily_api_key=os.getenv("TAVILY_API_KEY"),
            use_tavily_for_llm=use_tavily_for_llm,
            backend=backend,
        )

    def reason(
        self,
        text: str,
        created_at: str,
        timeframe: str,
        intermediates: object,           # IntermediateLayerOutputs
        lgbm_predictions: dict,          # asset -> MarketImpactPrediction
        is_market_relevant: bool,
        relevance_score: float,
    ) -> "LLMReasoningResult":
        """
        Run LLM reasoning over all pipeline layer outputs.

        Args:
            text:               Raw tweet text.
            created_at:         ISO timestamp string of the tweet.
            timeframe:          Prediction horizon (e.g. "5m", "10m").
            intermediates:      IntermediateLayerOutputs with all raw layer objects.
            lgbm_predictions:   Dict of asset -> MarketImpactPrediction from Layer 6.
            is_market_relevant: Boolean from relevance classifier.
            relevance_score:    Float relevance probability from Layer 6.

        Returns:
            LLMReasoningResult with per-asset verdicts, or an error result if the
            API call fails. Never raises — the pipeline continues regardless.
        """
        t0 = time.time()
        tavily_snippet = ""
        tavily_used = False

        if self._use_tavily_for_llm:
            tavily_snippet = self._fetch_tavily_context(text, created_at)
            tavily_used = bool(tavily_snippet)

        messages = self._build_prompt(
            text=text,
            created_at=created_at,
            timeframe=timeframe,
            intermediates=intermediates,
            lgbm_predictions=lgbm_predictions,
            is_market_relevant=is_market_relevant,
            relevance_score=relevance_score,
            tavily_snippet=tavily_snippet,
        )

        try:
            response = self._client.chat.completions.create(
                model=self._deployment,
                messages=messages,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                response_format={"type": "json_object"},
            )
            raw_text = response.choices[0].message.content
            parsed = self._parse_response(raw_text)
            latency_ms = (time.time() - t0) * 1000

            per_asset = {}
            for asset in ASSETS:
                asset_data = parsed.get("assets", {}).get(asset, {})
                llm_dir = int(asset_data.get("direction", 0))
                llm_dir = max(-1, min(1, llm_dir))  # clamp to valid range

                lgbm_dir = lgbm_predictions.get(asset)
                lgbm_direction = lgbm_dir.direction if lgbm_dir else 0

                per_asset[asset] = LLMAssetReasoning(
                    asset=asset,
                    direction=llm_dir,
                    confidence=float(asset_data.get("confidence", 0.5)),
                    reasoning=str(asset_data.get("reasoning", ""))[:200],
                    agrees_with_lgbm=(llm_dir == lgbm_direction),
                )

            overall = parsed.get("overall_assessment", "")

            # Extract direct relevance classification
            llm_relevant = parsed.get("is_market_relevant", None)
            if llm_relevant is not None:
                llm_relevant = bool(llm_relevant)
            llm_rel_conf = parsed.get("relevance_confidence", None)
            if llm_rel_conf is not None:
                llm_rel_conf = float(llm_rel_conf)
            llm_rel_reason = parsed.get("relevance_reasoning", None)

            # Compute agreement rate for monitoring
            n_agree = sum(1 for a in per_asset.values() if a.agrees_with_lgbm)
            agree_rate = n_agree / len(per_asset) if per_asset else 0.0
            llm_agree_rate = parsed.get("lgbm_agreement_rate", agree_rate)

            if agree_rate < 0.5:
                logger.warning(
                    f"Layer 7: LOW LGBM agreement rate {agree_rate:.0%} "
                    f"({n_agree}/{len(per_asset)} assets) — LLM may be over-overriding"
                )
            else:
                logger.info(
                    f"Layer 7 LLM reasoning complete in {latency_ms:.0f}ms "
                    f"(Tavily: {tavily_used}, relevant: {llm_relevant}, "
                    f"LGBM agreement: {agree_rate:.0%})"
                )

            return LLMReasoningResult(
                model_used=self._deployment,
                latency_ms=latency_ms,
                tavily_context_used=tavily_used,
                tavily_snippet=tavily_snippet,
                per_asset=per_asset,
                overall_assessment=overall,
                raw_response=raw_text,
                llm_is_market_relevant=llm_relevant,
                llm_relevance_confidence=llm_rel_conf,
                llm_relevance_reasoning=llm_rel_reason,
            )

        except Exception as e:
            latency_ms = (time.time() - t0) * 1000
            logger.error(f"Layer 7 LLM API call failed: {e}")
            return LLMReasoningResult(
                model_used=self._deployment,
                latency_ms=latency_ms,
                tavily_context_used=tavily_used,
                tavily_snippet=tavily_snippet,
                per_asset={},
                overall_assessment="",
                raw_response="",
                error=str(e),
            )

    def _fetch_tavily_context(self, text: str, created_at: str) -> str:
        """
        Fetch real-time market context via Tavily, scoped to the tweet timestamp.

        Distinct from Layer 4's Tavily usage — this query is richer and aims to
        surface market-moving news contemporary with the tweet for LLM grounding.
        """
        if not self._tavily_client:
            return ""

        date_str = ""
        if created_at:
            try:
                dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                date_str = dt.strftime("%B %d, %Y")
            except (ValueError, AttributeError):
                pass

        # Build a focused financial market query from tweet text
        snippet = text[:180].strip()
        query = f"financial market impact {snippet}"
        if date_str:
            query += f" around {date_str}"

        try:
            result = self._tavily_client.search(
                query=query,
                search_depth="basic",
                max_results=3,
                include_answer=True,
            )
            parts = []
            if result.get("answer"):
                parts.append(result["answer"])
            for r in result.get("results", [])[:2]:
                content = r.get("content", "")[:400]
                url = r.get("url", "")
                if content:
                    parts.append(f"[{url}] {content}" if url else content)
            context = "\n\n".join(parts)[:3000]
            logger.debug(f"Layer 7 Tavily context fetched ({len(context)} chars)")
            return context
        except Exception as e:
            logger.warning(f"Layer 7 Tavily search failed: {e}")
            return ""

    def _build_prompt(
        self,
        text: str,
        created_at: str,
        timeframe: str,
        intermediates: object,
        lgbm_predictions: dict,
        is_market_relevant: bool,
        relevance_score: float,
        tavily_snippet: str,
    ) -> list:
        """Build the OpenAI messages list for the reasoning call."""

        sections = []

        # --- [1] Macro Context (Tavily) — placed FIRST so the model calibrates before seeing the tweet ---
        if tavily_snippet:
            sections.append(
                "=== MACRO CONTEXT — WEEKLY MARKET BACKGROUND (Tavily) ===\n"
                "Use this to determine what is ALREADY PRICED IN vs what is NEW INFORMATION.\n"
                "A tweet that confirms existing market fears has less impact than a surprise escalation.\n\n"
                + tavily_snippet
            )
        else:
            sections.append(
                "=== MACRO CONTEXT — WEEKLY MARKET BACKGROUND ===\n"
                "No real-time market context available. Rely on your macro reasoning and NLP signals.\n"
                "Be conservative with confidence — without knowing what's priced in, uncertainty is higher."
            )

        # --- [2] Tweet ---
        sections.append(f"=== TARGET TWEET ===\nText: {text}\nTimestamp: {created_at or 'unknown'}\nPrediction horizon: {timeframe}")

        # --- Layer 1: NER ---
        ner = getattr(intermediates, "ner_result", None)
        if ner is not None:
            entities = getattr(ner, "entities", [])
            asset_rel = getattr(ner, "asset_relevance", {})
            entity_lines = []
            for e in entities[:10]:
                text_e = getattr(e, "text", str(e))
                label_e = getattr(e, "label", "")
                score_e = getattr(e, "score", 0.0)
                entity_lines.append(f"  {text_e} ({label_e}, score={score_e:.2f})")
            asset_rel_str = ", ".join(
                f"{a}: {v:.2f}" for a, v in asset_rel.items() if v > 0
            ) or "none"
            ner_block = (
                "=== LAYER 1 — EXTRACTED ENTITIES ===\n"
                f"Entities detected: {len(entities)}\n"
                + ("\n".join(entity_lines) if entity_lines else "  (none)") + "\n"
                + f"Asset relevance scores: {asset_rel_str}"
            )
            sections.append(ner_block)

        # --- Layer 2: Sentiment ---
        sent = getattr(intermediates, "sentiment_result", None)
        if sent is not None:
            sections.append(
                "=== LAYER 2 — SENTIMENT (FinBERT) ===\n"
                f"Label: {getattr(sent, 'label', 'unknown')}\n"
                f"Positive: {getattr(sent, 'positive', 0):.3f}  "
                f"Negative: {getattr(sent, 'negative', 0):.3f}  "
                f"Neutral: {getattr(sent, 'neutral', 0):.3f}\n"
                f"Compound score: {getattr(sent, 'compound', 0):.3f}  "
                f"(range -1 very negative → +1 very positive)"
            )

        # --- Layer 3: Events ---
        evt = getattr(intermediates, "event_result", None)
        if evt is not None:
            events = getattr(evt, "events", [])
            primary = getattr(evt, "primary_event", "none")
            primary_conf = getattr(evt, "primary_confidence", 0.0)
            event_lines = []
            for e in events:
                e_type = getattr(e, "event_type", str(e))
                e_conf = getattr(e, "confidence", 0.0)
                if e_type != "no_event":
                    event_lines.append(f"  {e_type}: {e_conf:.3f}")
            sections.append(
                "=== LAYER 3 — DETECTED EVENTS ===\n"
                f"Primary event: {primary} (confidence: {primary_conf:.3f})\n"
                + ("\n".join(event_lines) if event_lines else "  (no events detected)")
            )

        # --- Layer 5: Graph Signals ---
        graph = getattr(intermediates, "graph_result", None)
        if graph is not None:
            graph_fd = graph.to_feature_dict() if hasattr(graph, "to_feature_dict") else {}
            signal_lines = []
            for asset in ASSETS:
                sig = graph_fd.get(f"graph_signal_{asset}", 0.0)
                wt = graph_fd.get(f"graph_weight_{asset}", 0.0)
                direction_hint = "↑ bullish" if sig > 0.15 else ("↓ bearish" if sig < -0.15 else "→ neutral")
                signal_lines.append(f"  {asset:15s} signal={sig:+.3f}  weight={wt:.3f}  {direction_hint}")
            sections.append(
                "=== LAYER 5 — GRAPH PROPAGATION SIGNALS ===\n"
                + "\n".join(signal_lines)
            )

        # --- Layer 6: LightGBM Advisory ---
        if lgbm_predictions:
            lgbm_lines = [
                f"Market relevance: {'YES' if is_market_relevant else 'NO'} "
                f"(score: {relevance_score:.3f})",
                "",
                "These are statistical model predictions — treat as an informed advisory signal.",
                "Consider them seriously when your own reasoning is uncertain, but your macro",
                "analysis and the weekly context above should drive your final answer.",
                "",
                f"{'Asset':15s} {'Direction':10s} {'Confidence':12s} Top Reason",
                "-" * 70,
            ]
            for asset in ASSETS:
                pred = lgbm_predictions.get(asset)
                if pred:
                    dir_label = DIRECTION_LABELS.get(pred.direction, "?")
                    reasons = getattr(pred, "reasoning", [])
                    top_reason = reasons[0] if reasons else "—"
                    lgbm_lines.append(
                        f"  {asset:13s} {dir_label:10s} {pred.confidence:<12.3f} {top_reason}"
                    )
            sections.append(
                "=== LAYER 6 — LIGHTGBM ADVISORY PREDICTIONS ===\n"
                + "\n".join(lgbm_lines)
            )

        # --- Asset reference ---
        asset_ref_lines = [f"  {a}: {desc}" for a, desc in ASSET_DESCRIPTIONS.items()]
        sections.append(
            "=== ASSET REFERENCE ===\n" + "\n".join(asset_ref_lines)
        )

        sections.append(
            "FINAL CHECKLIST BEFORE YOU ANSWER:\n"
            "1. Did you use the MACRO CONTEXT above to judge whether this tweet is new info or already priced in?\n"
            "2. You are predicting a 5-MINUTE KNEE-JERK reaction — algo/headline scanners, not fundamentals.\n"
            "3. If the tweet is relevant, be DIRECTIONAL — most assets should have a non-zero direction.\n"
            "4. Did you check whether LightGBM's advisory agrees with your view? If it disagrees, have you considered why?\n"
            "5. Is your confidence calibrated to how much the macro context shows this topic is already priced in?\n"
            "Now produce your JSON assessment covering all 7 assets:"
        )

        user_content = "\n\n".join(sections)

        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    @staticmethod
    def _parse_response(raw: str) -> dict:
        """Parse LLM JSON response, handling markdown code fences defensively."""
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            # Remove first line (```json or ```) and last line (```)
            cleaned = "\n".join(lines[1:-1]).strip()
        return json.loads(cleaned)
