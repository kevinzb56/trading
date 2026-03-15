# General Market Impact Classifier

## Solution Overview

The **General Market Impact Classifier** is a 7-layer machine learning pipeline that predicts how political tweets impact seven global financial asset classes across three prediction horizons (1m, 5m, 10m).

**Architecture**: 5 NLP feature extraction layers + 1 LightGBM statistical ensemble + 1 LLM reasoning synthesis layer. Each layer is modular and independently debuggable.

**Output**: Per-asset market direction predictions (+1 bullish, 0 neutral, -1 bearish) with confidence scores and natural-language reasoning.

---

## System Architecture

```
Tweet Text + Timestamp
         │
         ├─ Layer 1: NER (GLiNER)           → 14 features
         ├─ Layer 2: Sentiment (FinBERT)    → 8 features
         ├─ Layer 3: Events (DistilBERT)    → 27 features
         ├─ Layer 4: Context (Temporal+Tavily) → 18 features
         ├─ Layer 5: Graph Reasoning        → 21 features
         │                    (~130 total features)
         ├─ Layer 6: LightGBM Ensemble      → Direction + Confidence
         │           (22 models: 7 assets × 3 timeframes)
         │
         └─ Layer 7: LLM Synthesis (GPT-4o + Tavily) → Per-asset verdicts
                     with natural-language reasoning

         Output: TweetPrediction
         { is_relevant, assets: { direction, confidence, reasoning },
           llm_reasoning: { per-asset GPT-4o verdicts } }
```

---

## Layer Descriptions

**Layer 1 — NER (GLiNER)**: Extracts persons, countries, commodities, organizations, and policy terms. Maps entities to target asset classes via curated lookup table. 14 features.

**Layer 2 — Sentiment (FinBERT)**: Financial sentiment classifier (pre-trained on financial news). Produces positive/negative/neutral probabilities and compound sentiment score. 8 features.

**Layer 3 — Event Detection (DistilBERT + Head)**: Hybrid neural + rule-based classification across 13 event types (trade war, sanctions, conflict, monetary policy, election, regulation, energy crisis, etc.). 27 features.

**Layer 4 — Context Enrichment**: Temporal features (trading session, hour, day-of-week), stylistic signals (urgency, caps ratio, thread position), and **live market news from Tavily API** (required). 18 features.

**Layer 5 — Entity-Event-Asset Graph Reasoning**: Directed graph with 97 domain-knowledge transmission rules encoding how events propagate to asset classes (e.g., `russia military_conflict → crude oil: -1`). Signals modulated by sentiment. 21 features.

**Layer 6 — LightGBM Ensemble**:
- 1 binary relevance classifier (market-relevant or not)
- 21 direction classifiers (7 assets × 3 timeframes)
- 5-fold StratifiedKFold cross-validation, class-weight balanced
- Per-asset predictions: direction ∈ {-1, 0, +1} with confidence

**Layer 7 — LLM Reasoning (Azure OpenAI GPT-4o, Required)**:
- Step 1: Fetch real-time market news from Tavily (timestamp-aware query)
- Step 2: Assemble structured prompt with all layer outputs + LightGBM predictions + live news
- Step 3: Call GPT-4o (response_format: json_object, temperature: 0.1)
- Output: Per-asset direction + confidence + natural-language reasoning, plus overall assessment

LLM layer provides interpretable explanations and reasons over novel event combinations LightGBM hasn't seen.

---

## Data & Training

**Dataset**: 3,336 tweets with market direction labels for 7 assets at 3 timeframes.

**Split**: Temporal (80% train = 2,669, 20% test = 667) to prevent data leakage and reflect realistic deployment.

**Training Time**: ~60 minutes on CPU (15 min with GPU)

**Inference Latency**: ~2,000–5,000ms per tweet (full pipeline including LLM + Tavily)

---

## Results

### Relevance Detection (Layer 6, LightGBM, 667 test samples)

| Metric | Value |
|---|---|
| **Accuracy** | **85.31%** |
| **ROC-AUC** | **89.72%** |
| **Recall** | **75.35%** |
| **F1 (Weighted)** | **85.14%** |
| CV Accuracy | 95.47% |

The relevance gate is the highest-confidence component, correctly identifying market-moving tweets with strong precision and recall.

---

### Direction Prediction (Layer 6, LightGBM, 5m horizon, 667 test samples)

**Mean Across All Assets**:
- Accuracy: 44.14%
- Balanced Accuracy: 38.08%
- F1 (Macro): 37.50%

**Per-Asset Results** (5m):

| Asset | Accuracy | Balanced Accuracy | F1 (Macro) | MCC |
|---|---|---|---|---|
| wheat | 53.97% | 50.30% | 49.19% | 0.2925 |
| btc | 46.93% | 34.36% | 33.29% | 0.0178 |
| equities | 42.58% | 34.27% | 33.99% | 0.0057 |
| gold | 43.63% | 33.29% | 32.88% | -0.0154 |
| eurodollar | 43.93% | 41.74% | 40.78% | 0.0977 |
| cl | 40.03% | 37.06% | 36.57% | 0.0303 |
| treasury_2y | 35.23% | 34.58% | 34.55% | 0.0194 |

**Insight**: Wheat shows the strongest signal (~54% accuracy) due to direct geopolitical/agricultural sensitivity. Directional prediction remains challenging due to market microstructure noise, short horizons, and complex macro drivers.

---

### Layer 7 LLM Reasoning (GPT-4o, 5m horizon, 667 test samples)

| Component | GPT-4o | LightGBM |
|---|---|---|
| Mean Direction Accuracy | 22.04% | 44.61% |
| Mean F1 (Macro) | 14.10% | 34.92% |
| Relevance Accuracy | 85.61% | 86.96% |

**Per-Asset Agreement Rate** (LLM vs LightGBM):
- wheat: 57.7%
- treasury_2y: 41.4%
- eurodollar: 15.9%
- cl: 12.1%
- equities: 10.2%
- gold: 7.1%
- btc: 6.3%

**Insight**: GPT-4o underperforms on direction prediction vs. LightGBM, but serves as a **reasoning synthesis layer** that:
1. Produces interpretable natural-language explanations
2. Grounds predictions in live market context (Tavily)
3. Provides alternative verdicts for novel event combinations
4. Achieves parity on relevance classification (85.6%)

The LLM layer's value is complementary interpretability and context integration, not raw accuracy improvement.

---

## Key Design Choices

- **Modular 7-layer architecture**: Each layer independently debuggable and swappable
- **Tavily integration (required)**: Real-time market context grounds both statistical and LLM predictions
- **Separate models per asset**: Prevents cross-asset interference; captures asset-specific sensitivities
- **Graph transmission rules**: Encodes domain knowledge (e.g., trade war → equities down) difficult to learn from limited data
- **LightGBM + LLM ensemble**: Statistical baseline + LLM reasoning for interpretability and context awareness

---

## Deployment

**Required APIs**:
- Tavily (real-time market news)
- Azure OpenAI GPT-4o (LLM reasoning)

**Running Inference**:
```bash
python inference_pipeline.py \
  --tweet "TARIFFS on China! 50% immediately!" \
  --model-dir saved_models \
  --timeframe 5m \
  --use-llm-reasoning \
  --llm-use-tavily
```

**Output**: JSON with per-asset LightGBM predictions + per-asset GPT-4o verdicts + reasoning + live Tavily context used.
