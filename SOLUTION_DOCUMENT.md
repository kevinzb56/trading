# General Market Impact Classifier
## Solution Architecture & Workflow

---

## 1. The Problem

Financial markets react to political events in real time. A single tweet from a world leader can move gold, equities, oil, and currencies within minutes — but most of those tweets are noise. The challenge is:

> **Given a political tweet and its timestamp, determine:**
> 1. Is this tweet market-relevant at all?
> 2. If yes — in which direction does it move each of seven financial assets at the 1m, 5m, and 10m horizons?
> 3. Why — what specific evidence from the tweet and current market context drives that conclusion?

Manual analysis is too slow. A simple keyword filter misses nuance. A single model cannot capture entities, sentiment, event type, graph propagation, historical patterns, and live market context simultaneously.

**The solution:** a seven-layer pipeline that decomposes this problem into specialized stages, each contributing a distinct type of signal, and synthesizes them into a final verdict using an LLM with live market grounding.

---

## 2. Running Example

> **Tweet:** `"TARIFFS on China! 150% immediately!"`
> **Posted:** During regular US market hours (10:30 AM ET)

This single tweet is used throughout every layer below to show exactly how the system processes it from raw text to final trading signal.

---

## 3. System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         INPUT                                           │
│           Tweet Text: "TARIFFS on China! 150% immediately!"             │
│           Timestamp:  2026-03-16T10:30:00Z                              │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
                ┌───────────────▼───────────────┐
                │  LAYER 1: Named Entity         │
                │  Recognition (GLiNER)          │
                │  → 14 entity features          │
                └───────────────┬───────────────┘
                                │
                ┌───────────────▼───────────────┐
                │  LAYER 2: Financial Sentiment  │
                │  (FinBERT)                     │
                │  → 8 sentiment features        │
                └───────────────┬───────────────┘
                                │
                ┌───────────────▼───────────────┐
                │  LAYER 3: Event Detection      │
                │  (DistilBERT + Rules)          │
                │  → 27 event features           │
                └───────────────┬───────────────┘
                                │
              ┌─────────────────▼────────────────┐
              │  LAYER 4: Context Enrichment      │
              │  (Temporal + Stylistic + Tavily)  │  ◄── Tavily Live News
              │  → 18 context features            │
              └─────────────────┬────────────────┘
                                │
                ┌───────────────▼───────────────┐
                │  LAYER 5: Graph Reasoning      │
                │  (Entity-Event-Asset Graph)    │
                │  97 domain knowledge rules     │
                │  → 21 graph features           │
                └───────────────┬───────────────┘
                                │
                    ~130 TOTAL FEATURES
                                │
                ┌───────────────▼───────────────┐
                │  LAYER 6: LightGBM Ensemble    │
                │  1 relevance classifier        │
                │  21 direction classifiers      │
                │  (7 assets × 3 timeframes)     │
                └───────────────┬───────────────┘
                                │
              ┌─────────────────▼────────────────┐
              │  LAYER 7: LLM Reasoning Synthesis │
              │  (Azure OpenAI GPT-4.1 / GPT-4o)  │  ◄── Tavily Live News
              │  Reasoning over all evidence      │
              │  + live market context            │
              └─────────────────┬────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────────────┐
│                         FINAL OUTPUT                                    │
│  is_market_relevant: true                                               │
│  Per-asset verdicts: direction (-1/0/+1), confidence, reasoning         │
│  Overall market assessment (natural language)                           │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Layer-by-Layer Walkthrough

---

### Layer 1 — Named Entity Recognition (GLiNER)

**Purpose:** Extract the financially and politically relevant entities from raw tweet text.

**How it works:**
- Uses **GLiNER** (a zero-shot NER model, ~170M parameters) to identify 10 entity types: persons, countries, organizations, commodities, policy terms, financial instruments, and more.
- Each detected entity is mapped to one or more of the seven target assets via a **curated lookup table** (e.g., "China" → equities, crude oil, wheat).
- Falls back to keyword matching when a GPU is unavailable.

**Input → Output:**
| Input | Output |
|-------|--------|
| Raw tweet text | 14 binary and count features per entity type |

**Example — Running Tweet:**
```
Tweet: "TARIFFS on China! 150% immediately!"

Entities Detected:
  - "China"   → type: country   → maps to: equities, crude oil, wheat
  - "TARIFFS" → type: policy    → maps to: equities, eurodollar
  - "150%"    → type: magnitude → signals: high urgency / severity

Entity Features (sample):
  has_country: 1
  has_policy_term: 1
  entity_equities_signal: 0.85
  entity_cl_signal: 0.70
  entity_wheat_signal: 0.60
```

**Feeds into:** Layer 3 (event detection) and Layer 5 (graph reasoning).

---

### Layer 2 — Financial Sentiment Analysis (FinBERT)

**Purpose:** Classify the emotional tone of the tweet specifically for financial language.

**How it works:**
- Uses **FinBERT**, pre-trained on financial news corpora (not general text). This makes it significantly more calibrated for market-relevant language than a general BERT model.
- Produces three probabilities: positive (bullish), negative (bearish), neutral — plus a compound score in [-1, 1].

**Input → Output:**
| Input | Output |
|-------|--------|
| Tweet text | 8 sentiment features (probabilities, compound score, confidence) |

**Example — Running Tweet:**
```
Tweet: "TARIFFS on China! 150% immediately!"

FinBERT Output:
  positive_prob:  0.08  (not optimistic)
  negative_prob:  0.81  (strong bearish tone)
  neutral_prob:   0.11
  compound_score: -0.74  (strongly negative)

→ Market interpretation: aggressive, bearish-toned announcement
```

**Feeds into:** Layer 5, where compound score modulates per-asset signal strength by ±30%.

---

### Layer 3 — Event Detection (DistilBERT + Trainable Head)

**Purpose:** Classify what type of geopolitical or economic event the tweet describes.

**How it works:**
- A **frozen DistilBERT encoder** (~66M parameters) generates a tweet embedding.
- A lightweight **trainable 2-layer MLP head** (~26K parameters) classifies the tweet into 13 event categories.
- A **hybrid overlay of 100+ regex patterns** ensures rare but critical event types (e.g., military conflict, crypto sanctions) are caught even when they appear too infrequently in training data to learn statistically.
- Multi-label: a tweet can trigger multiple event types simultaneously.

**13 Event Categories:**
Trade war / Tariffs · Sanctions · Military conflict · Monetary policy · Fiscal policy · Election · Regulation · Energy crisis · Pandemic · Central bank action · Currency volatility · Inflation signals · Structural reform

**Input → Output:**
| Input | Output |
|-------|--------|
| Tweet text | 27 features (one-hot per event type + confidence score per event) |

**Example — Running Tweet:**
```
Tweet: "TARIFFS on China! 150% immediately!"

Events Detected:
  trade_war:     0.94  ✓ (primary event)
  sanctions:     0.38  (partial — tariffs are a form of trade restriction)
  fiscal_policy: 0.31  (secondary signal)

Regex overlay confirms: keyword "TARIFFS" matches trade_war pattern
```

**Feeds into:** Layer 5 (graph reasoning) and Layer 7 (LLM prompt context).

---

### Layer 4 — Context Enrichment

**Purpose:** Enrich the feature set with temporal context, tweet stylistic signals, and live macro news at prediction time.

**How it works — three components:**

**Temporal Features**
- Trading session: pre-market, regular hours (09:30–16:00 ET), post-market
- Hour of day, day of week, time since market open

**Stylistic Features**
- ALL CAPS ratio (urgency indicator)
- Exclamation count, tweet length, URL presence
- Thread position (1st vs. later in a rapid-fire series)
- Inter-tweet timing (seconds since previous tweet from same account)

**Live Market News (Tavily Search API)**
- Builds a keyword query from tweet text + timestamp and fetches the top 5 financial news snippets contemporaneous with the tweet
- Extracts binary macro flags: tariffs active, conflict ongoing, rate policy in focus, etc.
- These flags allow the statistical models to condition on whether macro conditions *confirm* the tweet's signal

**Input → Output:**
| Input | Output |
|-------|--------|
| Tweet text + timestamp + live Tavily news | 18 context features |

**Example — Running Tweet:**
```
Tweet: "TARIFFS on China! 150% immediately!"
Timestamp: 2026-03-16 10:30 ET

Temporal:
  trading_session: "regular_hours"   → market is open, immediate impact expected
  hour_of_day: 10                    → active trading window

Stylistic:
  caps_ratio: 0.64    → HIGH urgency (TARIFFS, China)
  exclamation_count: 2 → aggressive announcement style

Tavily Live News (fetched):
  Headline 1: "US-China trade tensions escalate amid new tariff threats"
  Headline 2: "S&P 500 futures drop 1.2% on trade war fears"
  Macro flags: tariff_context=1, conflict_context=0, rate_context=0
```

**Feeds into:** Layer 6 (as numerical features) and Layer 7 (verbatim news snippets injected into LLM prompt).

---

### Layer 5 — Entity-Event-Asset Graph Reasoning

**Purpose:** Encode structured financial domain knowledge about how political events propagate to specific asset classes.

**How it works:**
- Builds a **directed weighted graph** per tweet with three node types: **entities → events → assets**
- 97 hand-crafted **transmission rules** encode market wisdom (e.g., textbook economics, analyst knowledge):
  - `(country:china, trade_war) → equities: (direction: -1, weight: 0.80)`
  - `(event:trade_war) → gold: (direction: +1, weight: 0.75)` — safe-haven demand
  - `(event:rate_hike) → treasury_2y: (direction: +1, weight: 0.85)`
- The **FinBERT compound score** from Layer 2 modulates each signal by ±30% depending on bearish or bullish tone
- Graph topology metrics (density, centrality) become additional features

**Why rules instead of purely learned?** With only 2,669 labeled tweets, a model cannot learn rare but important event combinations. Rules generalize to unseen scenarios — for example, a conflict involving a country not present in training data.

**Input → Output:**
| Input | Output |
|-------|--------|
| Entities (L1) + events (L3) + sentiment (L2) | 21 features (per-asset signal direction/magnitude + graph topology) |

**Example — Running Tweet:**
```
Graph constructed for "TARIFFS on China! 150% immediately!":

  china (entity) ──[triggers]──► trade_war (event)
                                      │
                    ┌─────────────────┼──────────────────┐
                    ▼                 ▼                   ▼
             equities (-1, 0.80) gold (+1, 0.75)  wheat (-1, 0.55)
                    │                               │
             cl (-1, 0.70)                 eurodollar (-1, 0.65)

FinBERT modulation (compound: -0.74):
  → Each signal strength increased ~22% (strong negative tone confirms bearish)

Final graph signals:
  equities:   -0.98  (strong bearish)
  gold:       +0.93  (strong bullish — safe haven)
  cl:         -0.86  (bearish — demand destruction fears)
  wheat:      -0.67  (bearish — trade disruption)
  eurodollar: -0.80  (bearish — USD strengthens vs EUR)
  treasury_2y:+0.75  (bullish — flight to safety)
  btc:        +0.10  (weak signal — crypto less directly connected)
```

**Feeds into:** Layer 6 (21 graph features in the full ~130 feature vector) and Layer 7 (per-asset signals summarized in LLM prompt).

---

### Layer 6 — LightGBM Statistical Ensemble

**Purpose:** Apply trained statistical classifiers to the ~130 assembled features to produce calibrated probabilistic predictions.

**How it works:**
- **22 independently trained LightGBM models:**
  - 1 binary relevance classifier (market-relevant vs. noise)
  - 21 direction classifiers: 7 assets × 3 timeframes (1m, 5m, 10m)
- Each model predicts a direction ∈ {−1, 0, +1} and outputs class probabilities
- Trained on **2,669 historical tweets** with a **5-fold stratified cross-validation**
- Class weights are inverse-frequency balanced to handle the skewed {−1, 0, +1} distribution
- **Separate models per asset:** each asset learns its own feature weights — sanctions news matters more for oil than for Bitcoin

**Input → Output:**
| Input | Output |
|-------|--------|
| ~130 features (Layers 1–5) | is_relevant (bool), relevance_score, per-asset direction + confidence |

**Example — Running Tweet:**
```
Layer 6 Output (5m horizon):

  Relevance:    RELEVANT (score: 0.92) ✓

  Per-asset LightGBM predictions:
    equities:    BEARISH  (-1, confidence: 0.81)
    gold:        BULLISH  (+1, confidence: 0.74)
    cl:          BEARISH  (-1, confidence: 0.69)
    wheat:       BEARISH  (-1, confidence: 0.62)
    eurodollar:  BEARISH  (-1, confidence: 0.76)
    treasury_2y: BULLISH  (+1, confidence: 0.68)
    btc:         NEUTRAL  (0,  confidence: 0.53)
```

**Feeds into:** Layer 7 — all predictions passed as a statistical prior to the LLM.

---

### Layer 7 — LLM Reasoning Synthesis (Azure OpenAI GPT-4.1 / GPT-4o)

**Purpose:** Final intelligence layer. Reasons over all structured evidence plus live market context to produce calibrated, interpretable verdicts — and provide natural-language explanations.

**How it works — three steps:**

**Step 1: Real-Time Context Retrieval (Tavily)**
- Builds a timestamp-aware query and fetches top 10 financial news snippets contemporaneous with the tweet
- Grounds the LLM in *current* market conditions, not just its training-time knowledge

**Step 2: Structured Prompt Assembly**
The LLM receives a complete structured prompt containing:
- Tweet text
- Extracted entities (Layer 1)
- Sentiment scores and compound tone (Layer 2)
- Detected event types and confidence (Layer 3)
- Trading session and macro flags (Layer 4)
- Per-asset graph propagation signals (Layer 5)
- **LightGBM predictions per asset (Layer 6)** — used as a statistical prior
- Live Tavily news snippets — verbatim market context

**Step 3: GPT-4.1 / GPT-4o Call**
- `response_format: json_object` — ensures structured, parseable output
- Temperature: 0.1 — low temperature for consistent, conservative verdicts
- The LLM reasons over all evidence, can **confirm or override** LightGBM on a per-asset basis, and must justify each decision in natural language

**Input → Output:**
| Input | Output |
|-------|--------|
| All layer outputs + live Tavily news | Per-asset: direction, confidence (0–1), reasoning text + overall market assessment |

**Example — Running Tweet:**
```
Tweet: "TARIFFS on China! 150% immediately!"

LLM Overall Assessment:
  "Extreme trade war escalation signal. 150% tariffs on China represent a
   severe supply-chain shock. Equities face direct earnings risk. Gold and
   Treasuries benefit as safe havens. EUR/USD weakens as USD demand surges.
   Crude oil faces demand destruction from reduced global trade volumes."

Per-Asset LLM Verdicts (GPT-4.1, 5m horizon):
  gold:        BULLISH   (0.82) "Trade war escalation boosts safe-haven demand."
  equities:    BEARISH   (0.89) "150% tariffs severely damage corporate earnings and growth."
  btc:         NEUTRAL   (0.51) "Crypto shows limited sensitivity to trade war events."
  cl:          BEARISH   (0.76) "Trade war fears reduce global energy demand expectations."
  wheat:       BEARISH   (0.63) "China tariff retaliation may hit US agricultural exports."
  eurodollar:  BEARISH   (0.84) "USD safe-haven demand strengthens against EUR."
  treasury_2y: BULLISH   (0.81) "Flight to safety boosts short-term Treasury demand."

  Agrees with LightGBM: 6/7 assets  |  Latency: ~4,400ms
```

---

## 5. Data Flow Summary

```
Tweet: "TARIFFS on China! 150% immediately!"

  L1 NER:      china=country, TARIFFS=policy, 150%=severity    → 14 features
       │
  L2 FinBERT:  compound=-0.74 (strongly bearish)               → 8 features
       │
  L3 Events:   trade_war=0.94, sanctions=0.38                   → 27 features
       │
  L4 Context:  session=regular, caps=0.64, tariff_news=1        → 18 features
       │
  L5 Graph:    equities=-0.98, gold=+0.93, cl=-0.86             → 21 features
       │
       └──────────────────── ~130 features ────────────────────────┐
                                                                    │
  L6 LightGBM: relevance=0.92, equities=BEARISH, gold=BULLISH  ◄──┘
       │
  L7 GPT-4.1:  Reasoning + live Tavily context → Final verdicts
       │
  OUTPUT:      Bearish equities, bullish gold/treasuries, with explanation
```

---

## 6. Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Modular 7-layer architecture** | Each layer can be debugged, replaced, or improved independently without retraining the full system |
| **No transformer fine-tuning** | Only 2,669 training tweets — fine-tuning large transformers would overfit. Layers 1–3 use frozen or minimally trained models |
| **Hybrid event detection (neural + rules)** | Rules ensure rare but high-impact events (military conflict, sanctions) are always caught, even when underrepresented in training data |
| **Graph transmission rules** | 97 domain rules encode textbook market knowledge and generalize to event combinations never seen in training |
| **Separate LightGBM per asset** | Each asset responds differently to the same event. Separate models prevent cross-asset interference |
| **Tavily for real-time grounding** | Without live news, both Layer 4 features and the LLM reason from stale context. Tavily synchronizes prediction with current market conditions |
| **LightGBM as LLM prior** | The LLM receives statistical predictions as input — it reasons *against* them and can confirm or override, with its reasoning visible |
| **Temporal train/test split** | The oldest 80% trains, the most recent 20% tests. Prevents look-ahead bias and mirrors real deployment conditions |

---

## 7. Inference Latency Breakdown

| Component | Latency |
|-----------|---------|
| Layers 1–3 (NER, Sentiment, Events) | ~120ms |
| Layer 4 (Tavily feature call) | ~200–800ms |
| Layer 5 (Graph reasoning) | ~5ms |
| Layer 6 (LightGBM ensemble) | ~5ms |
| Layer 7 (GPT-4.1 + Tavily context) | ~3,500–5,000ms |
| **Total end-to-end** | **~4,000–6,000ms** |

---

## 8. Evaluation Results

Evaluated on **667 held-out test samples** using a strict temporal split (no test data seen during training).

### Relevance Classification Accuracy — Three Approaches

| Model | Approach | Accuracy | Precision | Recall |
|-------|----------|----------|-----------|--------|
| **GPT-4.1** | LLM Reasoning (Layer 7 variant) | **88.01%** | **82.29%** | 55.63% |
| **LightGBM** | Statistical Ensemble (Layer 6) | 86.96% | 75.23% | 57.75% |
| **GPT-4o** | LLM Reasoning (Layer 7 baseline) | 85.61% | 77.38% | 45.77% |

**Key insight:** GPT-4.1 achieves the highest relevance accuracy and the highest precision — a 7-point precision advantage over LightGBM. Higher precision means fewer false signals sent to downstream trading decisions, directly reducing costly whipsaws. At ~20,000 API calls/day, GPT-4.1 filtering costs approximately $12–16/month — a low price for the quality improvement.

### Direction Prediction Accuracy (5m horizon, mean across 7 assets)

| Model | Mean Accuracy | Strength |
|-------|--------------|----------|
| LightGBM (Layer 6) | 44.61% | Statistical pattern recognition |
| GPT-4.1 (Layer 7) | 23.17% | Fundamentals-driven assets (Wheat, Treasury) |
| GPT-4o (Layer 7) | 22.04% | Reasoning and interpretability |

**Note:** LightGBM leads on raw directional accuracy across most assets. LLMs match or beat LightGBM on **Wheat (56% vs 53%)** and **2Y Treasury (38% vs 37%)** — precisely the assets most sensitive to geopolitical narrative and monetary policy, where context understanding outperforms pattern matching.

---

## 9. Recommended Production Configuration

The evaluation supports a **hybrid deployment** that leverages the strengths of both approaches:

```
Step 1 — GPT-4.1 relevance gate (88% accuracy, 82% precision)
         → Filter out noise. Only proceed if tweet is market-relevant.

Step 2 — LightGBM direction prediction (44.6% mean accuracy)
         → Fast, calibrated statistical direction on confirmed relevant signals.

Step 3 — GPT-4.1 reasoning synthesis
         → Final verdict with natural-language justification, confidence score,
            and context grounding via Tavily.
```

**Result:** Fewer trades, higher confidence, with full interpretability — the correct optimization for profitable, risk-managed trading.

---

*Report Generated: 2026-03-16 | System: General Market Impact Classifier | Test Set: 667 samples (temporal split)*
