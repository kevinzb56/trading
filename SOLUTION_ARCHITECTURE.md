# General Market Impact Classifier — Solution Architecture

## Executive Summary

The **General Market Impact Classifier** is a 7-layer machine learning pipeline that processes political tweets in real-time and predicts their directional impact on seven global financial asset classes across three prediction horizons (1m, 5m, 10m).

The system combines:
- **5 feature extraction layers** using state-of-the-art NLP models (GLiNER, FinBERT, DistilBERT)
- **1 statistical prediction layer** with 22 trained LightGBM classifiers
- **1 LLM synthesis layer** (Azure OpenAI GPT-4o) that reasons over all evidence and live market context from Tavily

Each layer is modular and can be debugged or improved independently. The output is a per-asset market direction prediction (+1 bullish, 0 neutral, -1 bearish) with confidence scores and interpretable reasoning.

---

## Architecture Overview

```
INPUT: Tweet Text + Timestamp
    │
    ├─────────────────────────────────────────────────────────────┐
    │                  FEATURE EXTRACTION (Layers 1-5)            │
    │                        ~130 features                        │
    └─────────────────────────────────────────────────────────────┤
    │
    ├─────────────────────────────────────────────────────────────┐
    │            STATISTICAL PREDICTION (Layer 6)                 │
    │            LightGBM ensemble (22 models)                    │
    └─────────────────────────────────────────────────────────────┤
    │
    ├─────────────────────────────────────────────────────────────┐
    │              LLM SYNTHESIS (Layer 7)                        │
    │        Azure OpenAI GPT-4o + Tavily real-time context      │
    └─────────────────────────────────────────────────────────────┤
    │
OUTPUT: TweetPrediction
    {
      is_market_relevant: bool
      relevance_score: float
      assets: { per-asset LightGBM predictions }
      llm_reasoning: { per-asset GPT-4o verdicts + explanations }
    }
```

---

## Layer 1 — Named Entity Recognition (GLiNER)

**Purpose**: Extract politically and financially relevant entities from the tweet text.

**Model**: GLiNER medium-v2.1 (zero-shot entity extractor, ~170M parameters)

**Entities Detected** (10 types):
- Persons (political figures, officials)
- Countries and geopolitical regions
- Commodities (oil, wheat, gold, etc.)
- Organizations (central banks, corporations, international bodies)
- Policy terms (tariffs, sanctions, regulations, etc.)
- Financial instruments and markets

**Output**: 14 binary and count features
- Per-entity relevance scores mapped to each of the 7 target asset classes via curated lookup table
- Fallback to keyword matching when GPU unavailable

**Example**:
- Tweet mentions "China" and "tariffs" → entities extracted and mapped to `equities`, `cl`, `wheat`

---

## Layer 2 — Financial Sentiment Analysis (FinBERT)

**Purpose**: Classify the emotional/sentiment tone of the tweet specific to financial context.

**Model**: FinBERT (pre-trained on financial news, ~110M parameters)

**Outputs**:
- Positive probability (bullish tone)
- Negative probability (bearish tone)
- Neutral probability (flat tone)
- Compound score in [-1, 1] (aggregated sentiment strength)
- CLS token embedding for downstream representations

**Output**: 8 features (3 probabilities, 1 compound, categorical one-hots, confidence)

**Why FinBERT?** General sentiment models (e.g., BERT base) are trained on product reviews, not financial news. FinBERT's financial pre-training makes it more calibrated for market-relevant language.

---

## Layer 3 — Event Detection (DistilBERT + Trainable Head)

**Purpose**: Classify what type of political/economic event the tweet describes.

**Architecture**:
- Frozen DistilBERT encoder (~66M parameters)
- Trainable 2-layer MLP classification head (~26K parameters)

**13 Event Categories**:
- Trade war / tariffs
- Sanctions
- Military conflict
- Monetary policy (interest rates, QE, etc.)
- Fiscal policy (stimulus, spending cuts)
- Election / political change
- Regulation / compliance
- Energy crisis / supply shock
- Pandemic / health crisis
- Central bank action
- Currency / forex volatility
- Inflation / deflation signals
- Structural reform

**Hybrid Approach**:
1. Neural model produces probability per event type
2. 100+ regex rule patterns applied as fallback/overlay
3. Multi-label classification (a tweet can trigger multiple events)

**Output**: 27 features (one-hot + confidence score per event type)

**Why hybrid?** Rare events (e.g., `military_conflict`) appear infrequently in training data; rules ensure high recall for critical categories.

---

## Layer 4 — Context Enrichment

**Purpose**: Enrich the feature set with temporal context, stylistic signals, and live macro news.

**Components**:

### Temporal Features
- Trading session (pre-market, regular hours, post-market in US Eastern Time)
- Hour of day
- Day of week
- Time since market open/close

### Stylistic Features
- Tweet length
- Caps ratio (ALL CAPS indicator of urgency)
- Exclamation count
- URL presence
- Thread position (1st, 2nd, 3rd... in a rapid-fire series)
- Inter-tweet timing (seconds since previous tweet)

### Live Market News (Tavily API, Required)
- Keyword-based query built from tweet text and timestamps
- Fetches top 5 financial news snippets contemporaneous with tweet
- Extracted signals: tariffs, conflict, rate policy, economic data, geopolitical
- Converted to binary macro flags

**Output**: 18 features

**Tavily Integration**: Essential for both Layer 4 and Layer 7. Lets the pipeline condition on what was actually happening in markets at prediction time, not just historical patterns.

---

## Layer 5 — Entity-Event-Asset Graph Reasoning

**Purpose**: Encode structured domain knowledge about how political events propagate to financial markets.

**Architecture**:
- Directed weighted graph with 3 node types: entities, events, assets
- Edge types:
  - Entity → Event (what event does this entity trigger?)
  - Event → Asset (which asset classes does this event affect?)

**Domain Rules** (97 transmission rules):
- Example: `(country:russia, military_conflict) → cl: (-1, 0.9)` (Russian conflict bearish for crude oil with 0.9 weight)
- Example: `(country:china, trade_war) → equities: (-1, 0.8)` (China trade war bearish for equities with 0.8 weight)
- Example: `(event:rate_hike) → treasury_2y: (+1, 0.85)` (Rate hike bullish for short-term Treasuries with 0.85 weight)
- Each rule specifies: (source entity/event, target asset, direction multiplier, weight)

**Signal Propagation**:
1. Extract entities and events from tweet (Layers 1, 3)
2. Apply transmission rules → per-asset directional signals
3. Modulate by FinBERT compound sentiment (±30% adjustment based on bullish/bearish tone)
4. Output: per-asset signal strength and topology metrics

**Output**: 21 features
- Per-asset signal direction and magnitude
- Graph density, clustering coefficient, centrality scores

**Why rules?** Domain knowledge is hard to learn from 2,669 labeled tweets alone. Transmission rules encode economics textbooks and market wisdom, improving generalization to unseen event combinations.

---

## Layer 6 — LightGBM Statistical Prediction Ensemble

**Purpose**: Train calibrated probabilistic classifiers for each asset/timeframe pair.

**Models**:
- **1 Relevance Classifier**: Binary (market-relevant vs. not)
- **21 Direction Classifiers**: Multiclass per asset–timeframe pair
  - 7 assets × 3 timeframes = 21 models
  - Each predicts direction ∈ {-1, 0, +1}

**Training Details**:
- Features: ~130 (concatenation of Layers 1–5)
- Training set: 2,669 tweets (80% of 3,336)
- Cross-validation: 5-fold Stratified K-Fold
- Hyperparameters (per model):
  - `n_estimators=300`, `max_depth=7`, `learning_rate=0.03`, `num_leaves=31`
  - `subsample=0.8`, `colsample_bytree=0.8`
- Class weighting: inverse frequency (handles skewed {-1, 0, +1} distribution)

**Output**: Per-asset direction + confidence scores (probabilities from `predict_proba`), plus feature importance for reasoning generation.

**Why separate models?** Each asset reacts differently to the same event. Sanctions news is more relevant to crude oil than Bitcoin. Separate models capture asset-specific feature weights without cross-asset interference.

---

## Layer 7 — LLM Reasoning Synthesis (Azure OpenAI GPT-4o, Required)

**Purpose**: Final intelligence layer that reasons over all evidence and current market conditions to produce calibrated verdicts.

**Three-Step Process**:

### Step 1: Real-Time Context Retrieval (Tavily, Required)
- Build timestamp-aware query from tweet text and timestamp
- Fetch top 10 financial news snippets from Tavily
- Retrieve news that was live/contemporaneous with the tweet
- Examples: "What was happening in markets on 2019-05-06?"

### Step 2: Structured Prompt Assembly
Assemble JSON-structured prompt containing:
- Tweet text
- Extracted entities (Layer 1)
- Sentiment scores (Layer 2)
- Detected events + confidences (Layer 3)
- Graph propagation signals per asset (Layer 5)
- **LightGBM predictions per asset (Layer 6)** — critical: GPT-4o uses these as a statistical prior
- Live Tavily news snippets
- Target prediction horizon (1m, 5m, 10m)

### Step 3: GPT-4o Call
- Model: Azure OpenAI GPT-4o
- `response_format: json_object` (ensures valid JSON output)
- Temperature: 0.1 (low—favor consistency over creativity)
- Max tokens: 2,000

**Output Per Asset**:
```json
{
  "direction": 1,            // +1 bullish, 0 neutral, -1 bearish
  "confidence": 0.75,        // 0.0–1.0 calibrated confidence
  "reasoning": "...",        // Natural-language explanation
  "agrees_with_lgbm": false  // Does LLM agree with Layer 6?
}
```

**Overall Assessment**:
```json
{
  "overall_assessment": "Trade war escalation primary driver...",
  "latency_ms": 3800,
  "tavily_context_used": true
}
```

**Why LLM?** LightGBM struggles with:
1. Novel event combinations not seen in training data
2. Shallow interpretability (feature importance is coarse)

GPT-4o receives all evidence + live context and produces:
- Reasoning that connects entities → events → assets
- Calibrated confidence adjusted for prediction horizon
- Ability to override LightGBM on a per-asset basis when context warrants

**Error Handling**: If LLM call fails, pipeline returns Layer 6 predictions as fallback (non-blocking).

---

## Data Flow

### Training Phase
1. Load 3,336 labeled tweets, split temporally (80% train, 20% test)
2. Extract features via Layers 1–5 (~35 min on CPU, cached to JSON)
3. Train event detection head (5 min, frozen DistilBERT encoder)
4. Re-extract event features with trained head (10 min)
5. Train 22 LightGBM models with 5-fold CV (5 min)
6. Evaluate on held-out 667-sample test set (10 sec)
7. Generate report with metrics and LLM evaluation

**Total Training Time**: ~60 minutes on CPU (15 min with GPU)

### Inference Phase
1. Load all models and transformers (~5 sec, one-time)
2. Layer 1 (NER): ~40ms
3. Layer 2 (Sentiment): ~30ms
4. Layer 3 (Events): ~50ms
5. Layer 4 (Context): ~1ms + ~200–800ms (Tavily API call, required)
6. Layer 5 (Graph): ~5ms
7. Layer 6 (LightGBM): ~5ms
8. Layer 7 (LLM): ~1,500–4,000ms (GPT-4o + Tavily context)

**Total Inference Latency**: ~2,000–5,000ms per tweet (LLM-inclusive pipeline required)

---

## Dependencies & Configuration

### API Keys (Required)
```
TAVILY_API_KEY                    # Real-time market news
AZURE_OPENAI_API_KEY              # GPT-4o LLM access
AZURE_OPENAI_ENDPOINT            # Azure OpenAI endpoint
AZURE_OPENAI_DEPLOYMENT_NAME     # Deployed GPT-4o model name
```

### Python Packages
- PyTorch (transformers inference)
- Hugging Face Transformers (GLiNER, FinBERT, DistilBERT)
- LightGBM (statistical models)
- NetworkX (graph reasoning)
- Tavily Python SDK (real-time news)
- Azure OpenAI SDK (LLM calls)
- scikit-learn (evaluation metrics)

---

## Results & Accuracy

### Layer 6 — Relevance Detection (LightGBM, all 667 test samples)

| Metric | Value |
|---|---|
| **Accuracy** | **85.31%** |
| **ROC-AUC** | **89.72%** |
| **Recall** | **75.35%** |
| **Precision** | **82.48%** |
| **F1 (Weighted)** | **85.14%** |
| **Cross-Validation Accuracy** | **95.47%** |
| **Cross-Validation F1** | **95.54%** |

**Interpretation**: The relevance gate is the highest-confidence component. It correctly identifies market-moving tweets with strong precision and recall.

---

### Layer 6 — Direction Prediction (LightGBM, 5m horizon, all 7 assets, 667 test samples)

| Asset | Accuracy | Balanced Accuracy | F1 (Macro) | F1 (Weighted) | MCC |
|---|---|---|---|---|---|
| **wheat** | 53.97% | 50.30% | 49.19% | 56.45% | 0.2925 |
| **btc** | 46.93% | 34.36% | 33.29% | 45.62% | 0.0178 |
| **equities** | 42.58% | 34.27% | 33.99% | 42.03% | 0.0057 |
| **gold** | 43.63% | 33.29% | 32.88% | 43.42% | -0.0154 |
| **eurodollar** | 43.93% | 41.74% | 40.78% | 44.29% | 0.0977 |
| **treasury_2y** | 35.23% | 34.58% | 34.55% | 35.33% | 0.0194 |
| **cl** | 40.03% | 37.06% | 36.57% | 40.30% | 0.0303 |

**Mean Across Assets (5m timeframe)**:
- Mean Accuracy: 44.14%
- Mean Balanced Accuracy: 38.08%
- Mean F1 (Macro): 37.50%
- Mean F1 (Weighted): 44.31%

**Interpretation**:
- **Wheat** shows the strongest signal (~54% accuracy) because geopolitical and agricultural trade events directly affect this commodity.
- **Equities, eurodollar, treasury** are harder due to complex macro drivers beyond tweet text.
- Directional prediction is inherently difficult due to market microstructure noise, short horizons, and ambiguous political language mapping to prices.
- Baseline (majority class): ~43–46% (LightGBM marginally above random in some assets).

---

### Layer 7 — LLM Reasoning (GPT-4o, 5m horizon, 667 test samples)

| Metric | GPT-4o Accuracy | LightGBM Accuracy | Agreement |
|---|---|---|---|
| **Mean Direction Accuracy** | 22.04% | 44.61% | 21.52% |
| **Mean F1 (Macro)** | 14.10% | 34.92% | — |

**Relevance Classification (LLM subsample, 667 samples)**:
| Metric | GPT-4o | LightGBM |
|---|---|---|
| Accuracy | 85.61% | 86.96% |
| Recall | 45.77% | 57.75% |
| F1 (Weighted) | 84.14% | 86.30% |

**Per-Asset Direction Comparison (5m)**:

| Asset | LightGBM Acc | LLM Acc | Agreement |
|---|---|---|---|
| wheat | 53.37% | 55.47% | 57.72% |
| treasury_2y | 37.33% | 37.48% | 41.38% |
| eurodollar | 45.88% | 13.79% | 15.89% |
| cl | 40.48% | 17.84% | 12.14% |
| equities | 44.68% | 13.94% | 10.19% |
| gold | 42.58% | 10.79% | 7.05% |
| btc | 47.98% | 4.95% | 6.30% |

**Interpretation**:
- GPT-4o struggles with **direction prediction** relative to LightGBM (22% vs. 44% mean accuracy)
- GPT-4o performs **comparably to LightGBM on relevance** (85.6% vs. 86.96%)
- High disagreement rates (mostly <50%) suggest LLM and LightGBM capture different signals
- **Wheat** shows highest LLM-LightGBM agreement (57.7%), suggesting consensus on this asset
- **Bitcoin** shows lowest (6.3%), possibly because LLM lacks training data or calibration for crypto

**Key Insight**: The LLM layer is primarily a **reasoning/synthesis layer**, not an accuracy booster. Its value lies in:
1. Producing interpretable natural-language explanations
2. Grounding predictions in live market context (Tavily)
3. Providing alternative verdicts when novel event combinations arise
4. Serving as a quality gate or escalation point for model uncertainty

---

## Design Rationale

**Modular Architecture**
Each layer can be swapped, improved, or disabled independently without retraining the full system.

**Statistical + Neural + LLM Ensemble**
- NLP layers extract rich contextual features
- LightGBM learns empirical patterns from labeled data
- LLM provides reasoning and live context integration

**Tavily Integration (Required)**
Real-time market context is essential. Without it, the LLM and Layer 4 features become stale, reducing calibration on market conditions at prediction time.

**Separate Models Per Asset**
Each asset has unique sensitivities. Separate classifiers prevent cross-asset spillover and allow asset-specific feature weighting.

**No Fine-Tuning**
Transformers are frozen or minimally trained (only event head trainable). This avoids overfitting on 2,669 tweets and leverages pre-trained financial knowledge.

---

## Quick Start

```bash
# Training
python train_pipeline.py \
  --data data/train.csv \
  --output saved_models \
  --use-tavily-context

# Inference (full pipeline with LLM + Tavily)
python inference_pipeline.py \
  --tweet "TARIFFS on China! 50% immediately!" \
  --model-dir saved_models \
  --timeframe 5m \
  --use-llm-reasoning \
  --llm-use-tavily
```

---

## Conclusion

The General Market Impact Classifier combines interpretable feature engineering with modern deep learning and LLM reasoning to predict how political tweets move financial markets. Its modular 7-layer architecture allows iterative improvement and detailed debugging at each stage. While directional accuracy is challenging due to market complexity, the system provides a solid statistical baseline (Layer 6) augmented by live context reasoning (Layer 7) for practitioner interpretation.
