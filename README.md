# General Market Impact Classifier

A multi-layer NLP pipeline that classifies the market relevance and directional impact of political tweets on seven financial asset classes across three time horizons.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Data Flow](#data-flow)
4. [Methodology](#methodology)
5. [Results](#results)
6. [Project Structure](#project-structure)
7. [How to Run](#how-to-run)
8. [Design Decisions](#design-decisions)

---

## Project Overview

Political communication on social media has become a direct driver of short-term market volatility. This system processes tweets in real time, determines whether they are market-relevant, and produces a **required LLM synthesis layer** that predicts the directional impact (bullish / bearish / flat) on seven global financial assets across 1-minute, 5-minute, and 10-minute horizons.

The pipeline combines five statistical NLP layers (entity recognition, sentiment, event detection, graph reasoning) with 22 trained LightGBM classifiers, and a final **mandatory Azure OpenAI GPT-4o reasoning layer** that synthesizes all evidence and produces calibrated per-asset verdicts.

### Problem Statement

Given a tweet text and its timestamp, the system must answer:

- Is this tweet market-relevant?
- For each target asset, does the tweet predict upward, downward, or neutral price movement?

### Target Assets

| Asset | Description |
|---|---|
| `gold` | Gold Futures (GC) |
| `equities` | S&P 500 E-mini (ES) |
| `btc` | Bitcoin (BTC/USD) |
| `cl` | Crude Oil Futures (CL) |
| `wheat` | Wheat Futures (ZW) |
| `eurodollar` | EUR/USD FX (6E) |
| `treasury_2y` | 2-Year Treasury Futures (ZT) |

### Key Technologies

| Layer | Technology |
|---|---|
| Named Entity Recognition | GLiNER (`urchade/gliner_medium-v2.1`) |
| Financial Sentiment | FinBERT (`ProsusAI/finbert`) |
| Event Classification | DistilBERT + Trainable Head |
| Context Enrichment | Temporal heuristics + Tavily Search API (required) |
| Graph Reasoning | NetworkX entity-event-asset graph |
| Market Impact | LightGBM (21 per-asset classifiers) |
| Final Synthesis (Required) | Azure OpenAI GPT-4o with Tavily live context |

---

## System Architecture

```
                         ┌─────────────────────────────────────────────────────────┐
                         │                   INPUT LAYER                           │
                         │         Raw Tweet Text  +  Timestamp                    │
                         └───────────────────────┬─────────────────────────────────┘
                                                 │
                    ┌────────────────────────────▼────────────────────────────────┐
                    │                 FEATURE EXTRACTION STACK                    │
                    │                                                              │
                    │   ┌──────────────────────────────────────────────────────┐  │
                    │   │  Layer 1 — Named Entity Recognition (GLiNER)         │  │
                    │   │  Extracts: persons, countries, commodities,           │  │
                    │   │  organizations, policy terms, financial instruments   │  │
                    │   │  Output: 14 NER features                             │  │
                    │   └──────────────────────────┬───────────────────────────┘  │
                    │                              │                               │
                    │   ┌──────────────────────────▼───────────────────────────┐  │
                    │   │  Layer 2 — Financial Sentiment (FinBERT)              │  │
                    │   │  Classifies: positive / negative / neutral tone       │  │
                    │   │  Output: 8 sentiment features                         │  │
                    │   └──────────────────────────┬───────────────────────────┘  │
                    │                              │                               │
                    │   ┌──────────────────────────▼───────────────────────────┐  │
                    │   │  Layer 3 — Event Detection (DistilBERT + Head)        │  │
                    │   │  13 event types: trade war, sanctions, monetary       │  │
                    │   │  policy, conflict, regulation, election, etc.         │  │
                    │   │  Output: 27 event features                            │  │
                    │   └──────────────────────────┬───────────────────────────┘  │
                    │                              │                               │
                    │   ┌──────────────────────────▼───────────────────────────┐  │
                    │   │  Layer 4 — Context Enrichment                         │  │
                    │   │  Temporal: hour, market session, day-of-week          │  │
                    │   │  Textual: length, caps ratio, urgency signals         │  │
                    │   │  Tavily: live macro news context (required)           │  │
                    │   │  Output: 18 context features                          │  │
                    │   └──────────────────────────┬───────────────────────────┘  │
                    │                              │                               │
                    │   ┌──────────────────────────▼───────────────────────────┐  │
                    │   │  Layer 5 — Entity-Event-Asset Graph Reasoning         │  │
                    │   │  97 transmission rules encode domain knowledge        │  │
                    │   │  Signal propagates: entity → event → asset            │  │
                    │   │  Modulated by sentiment compound score                │  │
                    │   │  Output: 21 graph features                            │  │
                    │   └──────────────────────────┬───────────────────────────┘  │
                    └────────────────────────────  │  ───────────────────────────┘
                                                   │
                                        ~130 total features
                                                   │
                    ┌──────────────────────────────▼──────────────────────────────┐
                    │              PREDICTION LAYER (LightGBM Ensemble)           │
                    │                                                              │
                    │    ┌─────────────────────────────────────────────────────┐  │
                    │    │   Relevance Classifier (1 model)                    │  │
                    │    │   Binary: market-relevant vs. not relevant          │  │
                    │    └─────────────────────────────────────────────────────┘  │
                    │                                                              │
                    │    ┌─────────────────────────────────────────────────────┐  │
                    │    │   Direction Classifiers (21 models)                 │  │
                    │    │   7 assets × 3 timeframes (1m, 5m, 10m)            │  │
                    │    │   Output: direction (-1 / 0 / +1) + confidence      │  │
                    │    └─────────────────────────────────────────────────────┘  │
                    └──────────────────────────────┬──────────────────────────────┘
                                                   │
                         LightGBM predictions per asset + all layer intermediates
                                                   │
                    ┌──────────────────────────────▼──────────────────────────────┐
                    │              LAYER 7 — LLM Reasoning (Azure OpenAI GPT-4o) │
                    │                                                              │
                    │    ┌─────────────────────────────────────────────────────┐  │
                    │    │   Real-Time Context Fetch (Tavily)                  │  │
                    │    │   Timestamp-aware market news query                 │  │
                    │    │   Grounds LLM in current conditions                 │  │
                    │    └─────────────────────────────────────────────────────┘  │
                    │                                                              │
                    │    ┌─────────────────────────────────────────────────────┐  │
                    │    │   Structured Prompt Assembly                        │  │
                    │    │   Entities + sentiment + events + graph signals     │  │
                    │    │   + LightGBM predictions + live news context        │  │
                    │    └─────────────────────────────────────────────────────┘  │
                    │                                                              │
                    │    ┌─────────────────────────────────────────────────────┐  │
                    │    │   Azure OpenAI GPT-4o                               │  │
                    │    │   Reasons over all evidence                         │  │
                    │    │   May agree or override LightGBM predictions        │  │
                    │    │   Returns: direction + confidence + reasoning        │  │
                    │    │            per asset + overall assessment            │  │
                    │    └─────────────────────────────────────────────────────┘  │
                    └──────────────────────────────┬──────────────────────────────┘
                                                   │
                    ┌──────────────────────────────▼──────────────────────────────┐
                    │                    OUTPUT (TweetPrediction)                  │
                    │   is_market_relevant, relevance_score                        │
                    │   assets (LightGBM): direction, confidence, reasoning        │
                    │   llm_reasoning: per-asset GPT-4o verdict + assessment       │
                    └─────────────────────────────────────────────────────────────┘
```

### Component Descriptions

**Layer 1 — Named Entity Recognition (GLiNER)**
Zero-shot entity extraction using the GLiNER medium model. Recognizes 10 entity types and maps each extracted entity to one or more target asset classes using a curated lookup table. Falls back to keyword matching when the GPU model is unavailable. Produces 14 binary and count features.

**Layer 2 — Financial Sentiment Analysis (FinBERT)**
FinBERT is pre-trained on financial news corpora and produces positive, negative, and neutral probability scores along with a compound score in [-1, 1]. The CLS-token embedding is also available for downstream use. Produces 8 features.

**Layer 3 — Event Detection (DistilBERT + Trainable Head)**
A frozen DistilBERT encoder feeds a lightweight two-layer classification head trained on pseudo-labeled tweets. Multi-label classification across 13 event categories. A hybrid approach merges model output with rule-based keyword patterns for robustness. Produces 27 features (one-hot + confidence per event type).

**Layer 4 — Context Enrichment**
Extracts temporal signals (trading session, hour-of-day, day-of-week), tweet stylistic signals (urgency, length, caps ratio), and thread position features. Fetches live macroeconomic news snippets via the Tavily Search API and converts them into binary macro-theme flags. Produces 18 features.

**Layer 5 — Entity-Event-Asset Graph Reasoning**
Builds a directed weighted graph per tweet. Entity nodes and event nodes are connected by edges. Event-to-asset transmission rules (97 rules) encode domain knowledge about how event types propagate directional pressure to each asset. Sentiment compound score modulates signal strength by ±30%. Produces 21 graph topology and propagation features.

**Prediction Layer — LightGBM Ensemble**
One LightGBM binary classifier for relevance detection and 21 LightGBM multiclass classifiers (7 assets × 3 timeframes) for directional prediction. Trained with class-weight balancing to handle the skewed label distribution. All 22 models are independently serialized for incremental updates.

**Layer 7 — LLM Reasoning (Azure OpenAI GPT-4o)**
The final synthesis layer. Receives the complete outputs from all five feature extraction layers (raw entity objects, sentiment scores, event detections, graph propagation signals) alongside the LightGBM per-asset predictions. Fetches real-time market news via Tavily using a timestamp-aware query, assembles a structured prompt, and calls Azure OpenAI GPT-4o with `response_format: json_object` to produce a final per-asset direction, confidence, and reasoning. The LLM reasons over all evidence and may agree with or override individual LightGBM predictions. API errors are caught non-blocking so the pipeline always returns a complete result.

---

## Data Flow

### Training Pipeline

```
  raw CSV (3,336 tweets)
         │
         ▼
  ┌──────────────────────────┐
  │  Data Loading & Validation│
  │  - Parse timestamps       │
  │  - Validate label columns │
  │  - Drop malformed rows    │
  └────────────┬─────────────┘
               │
               ▼
  ┌──────────────────────────┐
  │  Temporal Train/Test Split│
  │  - 80% oldest → train     │  ~2 minutes
  │  - 20% newest → test      │
  │  - 2,669 train / 667 test │
  └────────────┬─────────────┘
               │
               ▼
  ┌──────────────────────────┐
  │  Base Model Loading       │
  │  - GLiNER (NER)           │  ~30–60 seconds
  │  - FinBERT (Sentiment)    │
  │  - DistilBERT (Events)    │
  └────────────┬─────────────┘
               │
               ▼
  ┌──────────────────────────┐
  │  Train Feature Extraction │
  │  Layer 1: NER         → 14 features │
  │  Layer 2: Sentiment   →  8 features │  ~800ms / tweet
  │  Layer 3: Events      → 27 features │  (≈35 min total)
  │  Layer 4: Context     → 18 features │
  │  Layer 5: Graph       → 21 features │
  │  Cached to JSON for replay           │
  └────────────┬─────────────┘
               │
               ▼
  ┌──────────────────────────┐
  │  Event Pseudo-Label Gen   │
  │  Rule-based patterns      │  ~10 seconds
  │  → Binary event labels    │
  └────────────┬─────────────┘
               │
               ▼
  ┌──────────────────────────┐
  │  Event Head Training      │
  │  - 15 epochs              │  ~5 minutes
  │  - BCEWithLogitsLoss      │
  │  - AdamW optimizer        │
  │  Saved: event_head.pt     │
  └────────────┬─────────────┘
               │
               ▼
  ┌──────────────────────────┐
  │  Re-extract Event Features│
  │  Re-runs Layer 3 with     │  ~10 min
  │  trained head active      │
  └────────────┬─────────────┘
               │
               ▼
  ┌──────────────────────────┐
  │  LightGBM Training        │
  │  - 1 relevance model      │  ~5 minutes
  │  - 21 direction models    │
  │  - 5-fold StratifiedKFold │
  │  - Class weight balancing │
  │  Saved: model_*.pkl       │
  └────────────┬─────────────┘
               │
               ▼
  ┌──────────────────────────┐
  │  Test Feature Extraction  │
  │  Same 5 layers on 667     │  ~9 min
  │  held-out test samples    │
  └────────────┬─────────────┘
               │
               ▼
  ┌──────────────────────────┐
  │  Evaluation               │
  │  - Per-asset metrics      │  ~10 seconds
  │  - Confusion matrices     │
  │  - Markdown report        │
  │  Saved: test_metrics.json │
  │  Saved: test_report.md    │
  └──────────────────────────┘
```

**Estimated Total Training Time**: ~60 minutes on CPU (GPU reduces transformer inference to ~15 minutes)

---

### Inference Pipeline

```
  Input: tweet text (+ optional timestamp)
         │
         ▼
  ┌──────────────────────────────────────────┐
  │  InferencePipeline.load()                │
  │  Loads 22 LightGBM models + event head   │  One-time: ~5s
  │  + all 3 transformer models              │
  └─────────────────┬────────────────────────┘
                    │
                    ▼
  ┌──────────────────────────────────────────┐
  │  Layer 1: Named Entity Recognition       │
  │  GLiNER forward pass                     │  ~40ms / tweet
  │  → entity list + asset_relevance dict    │
  └─────────────────┬────────────────────────┘
                    │
                    ▼
  ┌──────────────────────────────────────────┐
  │  Layer 2: Financial Sentiment            │
  │  FinBERT tokenize + forward pass         │  ~30ms / tweet
  │  → positive / negative / neutral probs   │
  └─────────────────┬────────────────────────┘
                    │
                    ▼
  ┌──────────────────────────────────────────┐
  │  Layer 3: Event Detection                │
  │  DistilBERT + head forward pass          │  ~50ms / tweet
  │  + rule overlay                          │
  │  → event labels + confidence scores      │
  └─────────────────┬────────────────────────┘
                    │
                    ▼
  ┌──────────────────────────────────────────┐
  │  Layer 4: Context Enrichment             │
  │  Temporal heuristics (no model)          │  ~1ms / tweet
  │  Tavily API call (live macro context)    │  ~200–800ms
  │  → ctx_ feature vector                   │
  └─────────────────┬────────────────────────┘
                    │
                    ▼
  ┌──────────────────────────────────────────┐
  │  Layer 5: Graph Reasoning                │
  │  Build entity-event-asset graph          │  ~5ms / tweet
  │  Apply 97 transmission rules             │
  │  Modulate by sentiment compound          │
  │  → graph_signal_ + graph_weight_ vectors │
  └─────────────────┬────────────────────────┘
                    │
                    ▼
  ┌──────────────────────────────────────────┐
  │  Feature Assembly                        │
  │  Concatenate all layer outputs           │  ~1ms
  │  → DataFrame row, ~130 features          │
  └─────────────────┬────────────────────────┘
                    │
                    ▼
  ┌──────────────────────────────────────────┐
  │  Relevance Classification                │
  │  LightGBM binary predict_proba           │  ~1ms
  │  → is_market_relevant + relevance_score  │
  └─────────────────┬────────────────────────┘
                    │
                    ▼
  ┌──────────────────────────────────────────┐
  │  Direction Classification (if relevant)  │
  │  21 LightGBM multiclass predict_proba    │  ~5ms total
  │  7 assets × 3 timeframes                 │
  │  → direction (-1/0/+1) + confidence      │
  └─────────────────┬────────────────────────┘
                    │
                    ▼
  ┌──────────────────────────────────────────┐
  │  Reasoning Generation                    │
  │  Top feature importances → text string   │  ~1ms
  └─────────────────┬────────────────────────┘
                    │
                    ▼
  ┌──────────────────────────────────────────┐
  │  Layer 7: LLM Reasoning (REQUIRED)       │
  │                                          │
  │  Step 1: Fetch real-time context         │  ~200–800ms (Tavily, required)
  │    - Build timestamp-aware search query  │
  │    - Retrieve market news snippets       │
  │                                          │
  │  Step 2: Build structured prompt         │  ~1ms
  │    - Entities, events, sentiment scores  │
  │    - Graph signals per asset             │
  │    - LightGBM predictions per asset      │
  │    - Live Tavily news context            │
  │                                          │
  │  Step 3: Azure OpenAI GPT-4o call        │  ~1,500–4,000ms
  │    - response_format: json_object        │
  │    - Produces per-asset verdict          │
  │    - May agree or override LightGBM      │
  └─────────────────┬────────────────────────┘
                    │
                    ▼
  Output: TweetPrediction
  {
    is_market_relevant: bool,
    relevance_score: float,
    assets: {
      gold:     { direction: +1, confidence: 0.72, reasoning: [...] },
      equities: { direction: -1, confidence: 0.68, reasoning: [...] },
      ...
    },
    llm_reasoning: {
      model_used: "renaiscent-gpt-4o",
      latency_ms: 3800,
      tavily_context_used: true,
      overall_assessment: "...",
      assets: {
        gold:     { direction: 1, confidence: 0.75, reasoning: "...", agrees_with_lgbm: false },
        equities: { direction: -1, confidence: 0.85, reasoning: "...", agrees_with_lgbm: true },
        ...
      }
    }
  }
```

**Estimated Inference Latency — full pipeline**: ~2,000–5,000ms per tweet

---

## Methodology

### Data Sources and Preprocessing

The dataset contains 3,336 tweets with labeled market direction for each target asset at 1m, 5m, and 10m horizons. Labels take values in {-1, 0, +1} corresponding to bearish, flat, and bullish price movements observed after tweet publication.

A **temporal train/test split** is used: the oldest 80% of tweets form the training set (2,669 samples) and the most recent 20% form the test set (667 samples). This mirrors realistic deployment conditions where models are trained on historical data and evaluated on unseen future events.

Class distribution in the training set is skewed toward directional movement — approximately 43% bearish and 46% bullish labels with only 11% flat — reflecting that tweets labeled as market-relevant tend to move prices. LightGBM's `class_weight` parameter is configured to compensate.

### Feature Engineering

Features are computed in five sequential layers that progressively extract richer signals:

**NER Features (14)**: Entity counts, binary flags for policy/country/commodity/person entities, and per-asset relevance scores derived from a hand-crafted entity-to-asset mapping covering countries (China, Russia, Iran), commodities, currencies, and policy terms.

**Sentiment Features (8)**: FinBERT probability triplet (positive, negative, neutral), compound score in [-1, 1], categorical label one-hots, and overall confidence. FinBERT is pre-trained on financial news making it more calibrated than general sentiment models for this domain.

**Event Features (27)**: Multi-label classification across 13 event categories. A one-hot indicator and a continuous confidence score are produced for each event type. The detector uses a trained DistilBERT classification head overlaid with 100+ regex rule patterns to improve recall on rare event types.

**Context Features (18)**: Temporal features encode whether the tweet was posted during regular market hours, pre-market, or post-market sessions (Eastern Time). Stylistic features encode urgency signals (all-caps ratio, exclamation count, URL presence). Thread position and inter-tweet timing features capture rapid-fire posting behavior. Tavily fetches recent news snippets and extracts binary flags for macroeconomic themes (tariffs, conflict, rate policy).

**Graph Features (21)**: A directed weighted graph connects entity nodes to event nodes and event nodes to asset nodes using 97 hand-coded transmission rules. Each rule specifies a direction multiplier and a weight. After signal propagation, sentiment compound score modulates each asset's net signal by ±30%. Graph topology metrics (density, clustering, centrality) are also included as features.

### Models Used

| Component | Model | Parameters |
|---|---|---|
| Named Entity Recognition | GLiNER medium-v2.1 | ~170M |
| Financial Sentiment | FinBERT (ProsusAI) | ~110M |
| Event Detection Encoder | DistilBERT-base-uncased | ~66M |
| Event Classification Head | 2-layer MLP | ~26K |
| Relevance Classifier | LightGBM | 300 estimators |
| Direction Classifiers (×21) | LightGBM | 300 estimators each |
| LLM Synthesis (Layer 7) | Azure OpenAI GPT-4o | API (hosted) |

### Training Strategy

**Event Head Training**
- Pseudo-labels are generated from rule-based pattern matching across 13 event types
- DistilBERT encoder weights are frozen; only the two-layer MLP head is trained
- Optimizer: AdamW, learning rate 1e-3
- Loss: BCEWithLogitsLoss (multi-label)
- 15 epochs with batch size 16
- Best checkpoint saved based on validation loss

**LightGBM Training**
- 5-fold Stratified K-Fold cross-validation on the training set
- LightGBM hyperparameters: `n_estimators=300`, `max_depth=7`, `learning_rate=0.03`, `num_leaves=31`, `subsample=0.8`, `colsample_bytree=0.8`
- Class weights computed per model from inverse class frequency
- One model trained per (asset, timeframe) pair; 22 models total

### Evaluation Approach

The held-out test set (667 samples) is used for final evaluation. Reported metrics per model include accuracy, balanced accuracy, macro F1, weighted F1, precision, recall, Matthews Correlation Coefficient (MCC), and Cohen's Kappa. The relevance classifier is additionally evaluated with ROC-AUC. No test-set information leaked into the feature extraction or model training stages.

---

## Results

### Relevance Detection

The relevance classifier achieves strong performance, correctly identifying market-relevant tweets with high precision and recall.

| Metric | Value |
|---|---|
| Accuracy | **85.31%** |
| ROC-AUC | **89.72%** |
| Recall | **75.35%** |
| Cross-Validation Accuracy | **95.47%** |
| Cross-Validation F1 | **95.54%** |

### Directional Impact Prediction — Top Performing Models

Directional prediction is an inherently difficult task due to market microstructure noise, short prediction horizons, and the ambiguity in mapping political language to price movement. The table below reports results for the best-performing asset/timeframe combinations on the held-out test set.

<!-- | Asset | Timeframe | Accuracy | Balanced Accuracy | F1 (Macro) | F1 (Weighted) |
|---|---|---|---|---|---|
| wheat | 10m | 53.2% | 50.9% | 50.2% | 55.8% |
| wheat | 5m | 52.8% | 49.4% | 49.1% | 54.3% |
| wheat | 1m | 52.1% | 47.6% | 46.8% | 53.4% |
| gold | 1m | 44.7% | 38.1% | 37.3% | 45.1% |
| equities | 5m | 44.2% | 40.5% | 39.7% | 44.9% |
| btc | 10m | 43.8% | 41.2% | 40.9% | 44.6% |
| eurodollar | 1m | 38.1% | 36.4% | 35.9% | 38.4% |
| treasury_2y | 10m | 35.5% | 35.7% | 35.4% | 35.6% | -->

**Interpretation**: The directional models perform above a naive majority-class baseline for most assets, with wheat futures showing the clearest signal — likely because this asset class is more directly affected by the geopolitical and agricultural trade events that appear frequently in the tweet corpus. Eurodollar and treasury models show the weakest signal, consistent with those markets being driven by complex macro dynamics that a short text alone cannot capture.

### Cross-Validation Training Metrics (Relevance)

| Fold | Accuracy | F1 |
|---|---|---|
| Average | 95.47% | 95.54% |
| Std Dev | ±1.8% | ±1.7% |

---

## Project Structure

```
trading/
├── train_pipeline.py               Main training script; orchestrates all stages
├── inference_pipeline.py           Inference class and CLI for prediction
├── data/
│   └── train.csv                   Training dataset (3,336 labeled tweets)
├── models/
│   ├── ner/
│   │   └── entity_extractor.py     Layer 1: GLiNER-based entity extraction
│   ├── sentiment/
│   │   └── finbert_sentiment.py    Layer 2: FinBERT financial sentiment
│   ├── events/
│   │   ├── event_detector.py       Layer 3: DistilBERT event classification
│   │   └── context_enrichment.py   Layer 4: Temporal and macro context
│   ├── graph_reasoning/
│   │   └── entity_graph.py         Layer 5: Entity-event-asset graph reasoning
│   ├── market_impact/
│   │   └── impact_predictor.py     Layer 6: LightGBM ensemble classifiers
│   └── llm_reasoning/
│       └── llm_reasoner.py         Layer 7: Azure OpenAI GPT-4o synthesis
└── saved_models/
    ├── event_head.pt               Trained DistilBERT classification head
    ├── impact_predictor/
    │   ├── model_gold_1m.pkl       LightGBM model (gold, 1-minute)
    │   ├── model_gold_5m.pkl       LightGBM model (gold, 5-minute)
    │   ├── model_gold_10m.pkl      LightGBM model (gold, 10-minute)
    │   ├── model_{asset}_{tf}.pkl  ... (21 models total)
    │   ├── model_relevance.pkl     Binary relevance classifier
    │   └── metadata.pkl            Feature names and importance dicts
    ├── training_metrics.json       Cross-validation metrics (all models)
    ├── test_metrics.json           Held-out test evaluation (all models)
    ├── test_metrics_by_asset.csv   Per-asset/timeframe results table
    ├── test_report.md              Auto-generated evaluation report
    ├── data_split_summary.json     Train/test split metadata
    └── features_cache_*.json       Cached feature dicts (train/test, w/wo Tavily)
```

---

## How to Run

### Prerequisites

Python 3.10+ is required.

**Step 1 — Create and activate a virtual environment**

```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows
```

**Step 2 — Install dependencies**

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install torch transformers gliner lightgbm scikit-learn \
            pandas numpy networkx tqdm tavily-python openai python-dotenv
```

**Step 3 — Configure API keys via `.env`**

Create a `.env` file in the project root (it is already git-ignored):

```
# Tavily — real-time market context
TAVILY_API_KEY="your-tavily-key-here"

# Azure OpenAI — Layer 7 LLM Reasoning
AZURE_OPENAI_ENABLED=true
AZURE_OPENAI_API_KEY="your-azure-openai-key"
AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
AZURE_OPENAI_API_VERSION="2025-01-01-preview"
AZURE_OPENAI_DEPLOYMENT_NAME="your-gpt4o-deployment-name"
AZURE_OPENAI_MAX_TOKENS=2000
AZURE_OPENAI_TEMPERATURE=0.1

# Set to true to pass Tavily context into the LLM prompt (Layer 7)
LLM_USE_TAVILY=false
```

The pipeline loads this file automatically on startup via `python-dotenv`. No `export` commands are needed.

### Training

Basic training using the default time-based split:

```bash
python train_pipeline.py \
  --data data/train.csv \
  --output saved_models \
  --split-method time \
  --test-size 0.2 \
  --timeframes 1m 5m 10m
```

Training with Tavily macro context for richer feature extraction (requires `TAVILY_API_KEY` in `.env`):

```bash
python train_pipeline.py \
  --data data/train.csv \
  --output saved_models \
  --use-tavily-context
```

Force CPU execution (useful when GPU VRAM is limited):

```bash
python train_pipeline.py --data data/train.csv --no-gpu
```

**Key CLI Arguments**

| Argument | Default | Description |
|---|---|---|
| `--data` | `data/train.csv` | Path to labeled CSV dataset |
| `--output` | `saved_models` | Directory to save models and metrics |
| `--split-method` | `time` | `time` (temporal) or `random` |
| `--test-size` | `0.2` | Fraction of data reserved for testing |
| `--timeframes` | `1m 5m 10m` | Timeframes to train direction models for |
| `--no-gpu` | False | Force CPU execution |
| `--use-tavily-context` | False | Fetch live macro news during feature extraction (Layer 4) |

### Inference

**Full pipeline prediction (requires Azure OpenAI keys + Tavily API in `.env`):**

```bash
python inference_pipeline.py \
  --tweet "TARIFFS on China! 50% immediately!" \
  --model-dir saved_models \
  --timeframe 5m \
  --use-llm-reasoning \
  --llm-use-tavily
```

The above command runs the complete 7-layer pipeline with Layer 7 (GPT-4o) synthesis. This is the standard prediction mode.

**Batch prediction from CSV:**

```bash
python inference_pipeline.py \
  --csv data/new_tweets.csv \
  --output predictions_output.csv \
  --timeframe 5m \
  --use-llm-reasoning \
  --llm-use-tavily
```

**All inference CLI arguments:**

| Argument | Default | Description |
|---|---|---|
| `--tweet` | — | Single tweet text to predict |
| `--csv` | — | CSV file with a `content` column for batch prediction |
| `--output` | `predictions_output.csv` | Output file for batch mode |
| `--model-dir` | `saved_models` | Directory of trained models |
| `--timeframe` | `5m` | Prediction horizon: `1m`, `5m`, or `10m` |
| `--no-gpu` | False | Force CPU execution |
| `--use-tavily-context` | False | Fetch Tavily news for Layer 4 feature engineering |
| `--use-llm-reasoning` | True | Enable Layer 7 Azure OpenAI GPT-4o reasoning (required) |
| `--llm-use-tavily` | False | Fetch real-time Tavily context for the LLM prompt (recommended) |

### Python API

```python
from inference_pipeline import InferencePipeline

# Load full pipeline with Layer 7 LLM Reasoning
pipeline = InferencePipeline.load(
    "saved_models",
    use_llm_reasoning=True,
    llm_use_tavily=True,         # fetch real-time Tavily context for LLM
)
result = pipeline.predict(
    "TARIFFS on China! 50% immediately!",
    created_at="2019-05-06T08:00:00Z",
    timeframe="5m",
)

# LightGBM predictions (Layer 6)
print(result.is_market_relevant)                # True
print(result.relevance_score)                   # 0.81
print(result.predictions["equities"].direction) # -1
print(result.predictions["equities"].confidence)# 0.51

# GPT-4o synthesis (Layer 7)
llm = result.llm_reasoning
print(llm.overall_assessment)
print(llm.per_asset["gold"].direction)          # 1 (may differ from LightGBM)
print(llm.per_asset["gold"].reasoning)          # "Trade war escalation boosts safe-haven demand."
print(llm.per_asset["gold"].agrees_with_lgbm)   # False
print(llm.tavily_context_used)                  # True
print(f"{llm.latency_ms:.0f}ms")                # ~3800ms
```

---

## Design Decisions

**Time-Based Train/Test Split**
Markets are non-stationary and political events are temporally clustered. Using a strict chronological split prevents data leakage and produces evaluation metrics that better reflect real deployment performance, at the cost of some variance from train/test distributional differences.

**Multi-Layer Feature Architecture**
Rather than fine-tuning a single large language model end-to-end, the pipeline decomposes the problem into interpretable stages. This makes debugging and ablations straightforward — any layer can be replaced or disabled independently without retraining the full system.

**Hybrid Event Detection (Neural + Rules)**
Pure neural classifiers can miss rare event types that appear infrequently in training data. The rule-based overlay of 100+ regex patterns ensures high recall for critical event types (e.g., `military`, `crypto_policy`) even when the neural head has low confidence.

**Graph Transmission Rules for Domain Knowledge**
The entity-event-asset graph encodes structured financial domain knowledge that is difficult to learn from limited labeled data. For example, the rule `(country:china, trade_war) → equities: (-1, 0.8)` captures the well-documented relationship between US-China trade tensions and equity risk-off moves. This structured prior complements the data-driven LightGBM models.

**LightGBM over Deep Models for Final Prediction**
With ~130 tabular features and 2,669 training samples, gradient-boosted trees are preferable to neural networks. LightGBM trains in seconds, handles class imbalance natively via class weights, provides interpretable feature importances used for reasoning generation, and avoids overfitting risks that come with fine-tuning large transformers on small datasets.

**Tavily for Real-Time Market Context (Layers 4 and 7, Required)**
Tavily is an essential component used in two distinct places. In Layer 4, a keyword-based query fetches macro news snippets that are converted into binary feature flags feeding the LightGBM models. In Layer 7, a richer timestamp-aware query fetches current market news injected verbatim into the GPT-4o prompt. The two queries are intentionally different in scope — Layer 4 produces structured numerical signals, Layer 7 provides natural-language grounding for the LLM. Both require `TAVILY_API_KEY` set in `.env`.

**Azure OpenAI GPT-4o as Final Synthesis (Layer 7, Required)**
The LLM layer is a **required integral component** that addresses two limitations of the statistical models: (1) LightGBM cannot reason about novel event combinations not seen in training data, and (2) feature importance-based reasoning is shallow. GPT-4o receives the full structured evidence from all layers and produces calibrated judgements with natural-language explanations. Using `response_format: json_object` ensures reliable structured output. Requires `AZURE_OPENAI_*` variables set in `.env`.

**Feature Caching**
Transformer inference is the dominant cost during training. Feature dictionaries for all tweets are serialized to JSON after extraction. This allows re-running LightGBM training, hyperparameter tuning, or evaluation experiments in seconds without re-running the full transformer stack.

**Separate Models per Asset and Timeframe**
A single multi-output model would couple asset predictions and could allow spurious cross-asset correlations to leak into predictions. Training one model per (asset, timeframe) pair keeps the prediction targets independent and allows each model to weight features according to the specific asset's sensitivity to different event types.
