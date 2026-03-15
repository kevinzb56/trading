# General Market Impact Classifier

A 7-layer NLP and LLM pipeline that processes political tweets in real time,
determines their market relevance, and produces a directional impact verdict
across seven global financial asset classes at configurable time horizons.

---

## What the System Does

Given a tweet posted by a major political figure, the system answers three questions:

1. Is this tweet market-relevant at all?
2. For each of the seven target assets, does it predict a bullish, bearish, or neutral short-term price move?
3. Why — what evidence from the tweet text, detected events, and current market conditions drove that conclusion?

The final answer is produced by an LLM (Azure OpenAI GPT-4o) that synthesizes
outputs from five upstream NLP layers, twenty-two trained statistical classifiers,
and real-time market news fetched from Tavily at prediction time.

---

## Target Assets

| Asset | Instrument | Description |
|---|---|---|
| `gold` | GC (Gold Futures) | Classic safe-haven commodity |
| `equities` | ES (S&P 500 E-mini) | Broad US equity market |
| `btc` | BTC/USD | Bitcoin cryptocurrency |
| `cl` | CL (Crude Oil Futures) | Global energy benchmark |
| `wheat` | ZW (Wheat Futures) | Agricultural commodity |
| `eurodollar` | 6E (EUR/USD FX) | Euro vs. US dollar exchange rate |
| `treasury_2y` | ZT (2-Year Treasury Futures) | Short-term US interest rate proxy |

Prediction labels: **+1** (bullish), **0** (flat/neutral), **-1** (bearish)
Prediction horizons: **1 minute**, **5 minutes**, **10 minutes** post-tweet

---

## Pipeline Architecture

The system processes each tweet through seven sequential layers. The first five
extract features of increasing abstraction. Layer 6 applies trained statistical
models. Layer 7 applies an LLM to reason over everything and produce the final output.

```
  Tweet Text + Timestamp
         │
         ▼
  ┌──────────────────────────────────────────────────────────┐
  │  Layer 1: Named Entity Recognition (GLiNER)              │
  │  Zero-shot NER across 10 entity types                    │
  │  Maps entities to assets via curated lookup table        │
  │  Output: 14 features                                     │
  └────────────────────────────┬─────────────────────────────┘
                               │
                               ▼
  ┌──────────────────────────────────────────────────────────┐
  │  Layer 2: Financial Sentiment (FinBERT)                  │
  │  Pre-trained on financial news corpora                   │
  │  Positive / negative / neutral + compound score          │
  │  Output: 8 features                                      │
  └────────────────────────────┬─────────────────────────────┘
                               │
                               ▼
  ┌──────────────────────────────────────────────────────────┐
  │  Layer 3: Event Detection (DistilBERT + Trainable Head)  │
  │  13 event categories: trade war, sanctions, conflict,    │
  │  monetary policy, election, regulation, energy, etc.     │
  │  Hybrid: neural model + 100+ keyword rule patterns       │
  │  Output: 27 features (one-hot + confidence per event)    │
  └────────────────────────────┬─────────────────────────────┘
                               │
                               ▼
  ┌──────────────────────────────────────────────────────────┐
  │  Layer 4: Context Enrichment                             │
  │  Temporal: trading session, hour-of-day, day-of-week     │
  │  Stylistic: urgency, caps ratio, exclamations, length    │
  │  Live news: Tavily search (timestamp-aware, required)    │
  │  Output: 18 features                                     │
  └────────────────────────────┬─────────────────────────────┘
                               │
                               ▼
  ┌──────────────────────────────────────────────────────────┐
  │  Layer 5: Entity-Event-Asset Graph Reasoning             │
  │  Directed weighted graph with 97 transmission rules      │
  │  Signal: entity → event → asset (direction + weight)     │
  │  Modulated by FinBERT compound score (±30%)              │
  │  Output: 21 features (per-asset signals + topology)      │
  └────────────────────────────┬─────────────────────────────┘
                               │
                      ~130 total features
                               │
                               ▼
  ┌──────────────────────────────────────────────────────────┐
  │  Layer 6: LightGBM Ensemble                              │
  │  1 binary relevance classifier                           │
  │  21 direction classifiers (7 assets × 3 timeframes)      │
  │  5-fold CV training, class-weight balanced               │
  │  Output: is_relevant + per-asset direction + confidence  │
  └────────────────────────────┬─────────────────────────────┘
                               │
         LightGBM predictions + all layer intermediate objects
                               │
                               ▼
  ┌──────────────────────────────────────────────────────────┐
  │  Layer 7: LLM Reasoning (Azure OpenAI GPT-4o)           │
  │                                                          │
  │  Step 1 — Real-Time Context (Tavily, required)           │
  │    Timestamp-aware financial news query                  │
  │    Grounds GPT-4o in current market conditions           │
  │                                                          │
  │  Step 2 — Structured Prompt Assembly                     │
  │    Extracted entities (Layer 1)                          │
  │    Sentiment scores (Layer 2)                            │
  │    Detected events + confidence (Layer 3)                │
  │    Graph propagation signals per asset (Layer 5)         │
  │    LightGBM predictions per asset (Layer 6)              │
  │    Live Tavily news snippet                              │
  │                                                          │
  │  Step 3 — GPT-4o Call (response_format: json_object)    │
  │    Reasons over all evidence                             │
  │    May agree with or override LightGBM per asset         │
  │    Returns: direction + confidence + reasoning per asset │
  │             + overall market assessment                  │
  └────────────────────────────┬─────────────────────────────┘
                               │
                               ▼
  Final Output: TweetPrediction
  {
    is_market_relevant: bool,
    relevance_score: float,
    assets (LightGBM): { direction, confidence, reasoning },
    llm_reasoning: {
      model_used, latency_ms, tavily_context_used,
      overall_assessment,
      assets: { direction, confidence, reasoning, agrees_with_lgbm }
    }
  }
```

---

## Real-Time Data: Why Tavily is Required

Tavily is the system's window into current market conditions. It is used in two
distinct ways and is required for the pipeline to function at full capability:

**Layer 4 — Feature Engineering**
A keyword-based query is built from the tweet text and submitted to Tavily.
The returned snippets are parsed for binary macro flags (tariffs, conflict, rate
policy) that become numerical features fed into the LightGBM models.
This allows the statistical models to condition on whether macro conditions
align with the tweet's signal at the time of prediction.

**Layer 7 — LLM Grounding**
A richer, timestamp-aware query is submitted specifically for the GPT-4o prompt.
The returned news snippets are injected verbatim into the LLM context.
This allows GPT-4o to reason about current conditions rather than relying solely
on its training-time knowledge, which may not reflect the market environment at
the moment of prediction.

Without Tavily, the LLM reasons from the tweet and static NLP layer outputs only.
With Tavily, it grounds its verdicts in what is actually happening in markets today.

**Configuration:**
Set `TAVILY_API_KEY` in your `.env` file. No export commands needed.

```
TAVILY_API_KEY="your-tavily-key-here"
LLM_USE_TAVILY=true
```

---

## Timestamp and Timeframe Handling

The system uses timestamps in two ways:

**Timestamp-aware Tavily queries**
When a tweet timestamp is provided (e.g. `created_at="2019-05-06T08:00:00Z"`),
both the Layer 4 and Layer 7 Tavily queries append a formatted date string to
the search query. This biases Tavily's results toward content that was
contemporaneous with the tweet, avoiding misleading results from later periods.

**Timeframe selection**
The `timeframe` parameter (`1m`, `5m`, `10m`) selects which set of trained
LightGBM models to use for direction prediction. Each timeframe has seven
separately trained models (one per asset). The selected timeframe is also
included in the LLM prompt so GPT-4o calibrates its confidence appropriately
for the prediction horizon.

**Market session context**
Layer 4 extracts whether the tweet was posted during regular market hours
(09:30–16:00 ET), pre-market (04:00–09:30 ET), or post-market (16:00–20:00 ET).
This is included in the LLM prompt as context for how immediately the predicted
move would materialize.

---

## Setup

**Step 1 — Install dependencies**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Step 2 — Create a `.env` file in the project root**

```
# Tavily (required — used in both Layer 4 features and Layer 7 LLM context)
TAVILY_API_KEY="your-tavily-key-here"
LLM_USE_TAVILY=true

# Azure OpenAI (required for Layer 7 LLM Reasoning)
AZURE_OPENAI_ENABLED=true
AZURE_OPENAI_API_KEY="your-azure-openai-key"
AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
AZURE_OPENAI_API_VERSION="2025-01-01-preview"
AZURE_OPENAI_DEPLOYMENT_NAME="your-gpt4o-deployment-name"
AZURE_OPENAI_MAX_TOKENS=2000
AZURE_OPENAI_TEMPERATURE=0.1
```

The pipeline loads this file automatically at startup. No shell `export` commands needed.

**Step 3 — Run inference**

```bash
python inference_pipeline.py \
  --tweet "TARIFFS on China! 50% immediately!" \
  --model-dir saved_models \
  --timeframe 5m \
  --use-llm-reasoning \
  --llm-use-tavily
```

---

## Results

### Relevance Detection (held-out test set, 667 samples)

| Metric | Value |
|---|---|
| Accuracy | **85.31%** |
| ROC-AUC | **89.72%** |
| Recall | **75.35%** |
| Cross-Validation Accuracy | **95.47%** |
| Cross-Validation F1 | **95.54%** |

The relevance classifier correctly identifies market-moving tweets with strong
precision and recall. This gate is the highest-confidence component of the
statistical pipeline and is the primary signal for downstream routing.

### Layer 7 LLM Output (live inference example)

Tweet: `"TARIFFS on China! 50% immediately!"`

```
LLM overall assessment:
  "Significant trade war escalation signal — equities and eurodollar
   most directly impacted; gold and treasuries benefit as safe havens."

Per-asset verdicts (GPT-4o):
  gold        →  BULLISH   (0.75)  "Trade war escalation boosts safe-haven demand for gold."
  equities    →  BEARISH   (0.85)  "Tariffs on China likely hurt corporate earnings and growth."
  btc         →  NEUTRAL   (0.50)  "Bitcoin shows limited sensitivity to trade war events."
  cl          →  BEARISH   (0.70)  "Trade war fears reduce global demand expectations for crude."
  wheat       →  NEUTRAL   (0.50)  "Agricultural commodities show limited immediate tariff impact."
  eurodollar  →  BEARISH   (0.80)  "Tariffs strengthen USD as safe-haven, weakening EUR/USD."
  treasury_2y →  BULLISH   (0.80)  "Trade war escalation increases demand for short-term Treasuries."

Latency: ~3,800ms  |  Tavily context used: true
```

---

## Key Design Choices

**Why seven layers?**
Each layer contributes a distinct signal type that no single model can capture alone.
NER provides entity grounding. FinBERT provides domain-calibrated sentiment. The
event detector provides categorical classification. The graph encodes structured
financial domain knowledge. LightGBM provides calibrated statistical predictions
from labeled historical data. GPT-4o synthesizes everything with live context.

**Why LightGBM before LLM?**
The statistical models are trained on labeled historical data and capture
empirical relationships that the LLM may not reliably reproduce. Including
their predictions in the LLM prompt gives GPT-4o a calibrated statistical prior
to reason against — it can confirm, adjust, or override on a per-asset basis,
with its reasoning visible in the output.

**Why Tavily is required for Layer 7?**
A major political tweet often references an evolving situation. Without real-time
context, the LLM can only reason from the tweet text and its training-time
knowledge. With Tavily, it grounds its verdict in what is actually happening
in markets today — improving accuracy on events that evolved after the LLM's
knowledge cutoff and providing specificity that static NLP features cannot.

**Why separate models per asset?**
Each asset responds differently to the same event. Sanctions news is more
relevant to oil than to Bitcoin. Trade war news is more relevant to equities
than to wheat. Separate models let each classifier learn asset-specific feature
weights from historical data without cross-asset interference.
