# Full Pipeline Evaluation Report

Evaluation of the complete 7-layer pipeline on the held-out test set.

- **Layer 6 (LightGBM)** evaluated on: **667 samples** (full test set)
- **Layer 7 (GPT-4o LLM)** evaluated on: **667 samples** (full test set — same 667 as Layer 6)
- **Timeframe for LLM evaluation**: `5m`
- **Average LLM latency**: 3.7s / tweet

---

## Layer 6 — LightGBM Relevance Classifier (Full Test Set)

| Metric | Value |
|---|---|
| Accuracy | **85.01%** |
| ROC-AUC | **86.93%** |
| Recall | **57.04%** |
| F1 (Weighted) | **84.53%** |
| F1 (Macro) | **76.25%** |
| Samples | 667 |

## Layer 6 — LightGBM Direction Classifiers (Full Test Set)

Global averages across all 7 assets and 3 timeframes:

| Metric | Value |
|---|---|
| Mean Accuracy | 44.14% |
| Mean Balanced Accuracy | 38.08% |
| Mean F1 (Macro) | 37.50% |
| Mean F1 (Weighted) | 44.31% |
| Mean MCC | 0.0676 |

### Per-Asset Results (all timeframes)

| Asset | TF | Accuracy | Bal. Acc | F1 Macro | F1 Weighted | MCC |
|---|---|---|---|---|---|---|
| btc | 10m | 49.18% | 34.85% | 34.21% | 48.20% | 0.0468 |
| btc | 1m | 45.28% | 37.22% | 37.14% | 45.28% | 0.0289 |
| btc | 5m | 46.93% | 34.36% | 33.29% | 45.62% | 0.0178 |
| cl | 10m | 40.33% | 33.29% | 33.19% | 40.33% | -0.0085 |
| cl | 1m | 37.78% | 38.61% | 37.78% | 37.37% | 0.0717 |
| cl | 5m | 40.03% | 37.06% | 36.57% | 40.30% | 0.0303 |
| equities | 10m | 47.53% | 36.95% | 37.03% | 47.15% | 0.0633 |
| equities | 1m | 41.83% | 38.60% | 38.46% | 42.01% | 0.0711 |
| equities | 5m | 42.58% | 34.27% | 33.99% | 42.03% | 0.0057 |
| eurodollar | 10m | 42.88% | 35.87% | 35.02% | 41.95% | 0.0396 |
| eurodollar | 1m | 35.08% | 34.88% | 34.78% | 34.88% | 0.0221 |
| eurodollar | 5m | 43.93% | 41.74% | 40.78% | 44.29% | 0.0977 |
| gold | 10m | 45.88% | 37.97% | 36.73% | 46.04% | 0.0215 |
| gold | 1m | 43.18% | 37.57% | 36.58% | 43.60% | 0.0140 |
| gold | 5m | 43.63% | 33.29% | 32.88% | 43.42% | -0.0154 |
| treasury_2y | 10m | 36.88% | 37.12% | 36.81% | 37.00% | 0.0561 |
| treasury_2y | 1m | 44.83% | 35.77% | 35.67% | 44.09% | 0.0411 |
| treasury_2y | 5m | 35.23% | 34.58% | 34.55% | 35.33% | 0.0194 |
| wheat | 10m | 53.82% | 51.88% | 50.99% | 56.30% | 0.3141 |
| wheat | 1m | 56.07% | 43.42% | 41.84% | 58.95% | 0.1901 |
| wheat | 5m | 53.97% | 50.30% | 49.19% | 56.45% | 0.2925 |

---

## Layer 7 — GPT-4o LLM Reasoning vs LightGBM (5m horizon, 667 samples)

### Overall Comparison

| Metric | LightGBM (Layer 6) | GPT-4o (Layer 7) |
|---|---|---|
| Mean Accuracy | 44.61% | **22.04%** |
| Mean F1 (Macro) | 34.92% | **14.10%** |
| LightGBM–LLM Agreement | — | 21.52% |

### Relevance Classification (LLM sample)

| Metric | LightGBM | GPT-4o |
|---|---|---|
| Accuracy | 86.96% | **85.61%** |
| Recall | 57.75% | **45.77%** |
| F1 (Weighted) | 86.30% | **84.14%** |

### Per-Asset Direction Comparison

| Asset | LGBM Acc | LLM Acc | LGBM F1 | LLM F1 | Agreement |
|---|---|---|---|---|---|
| gold | 42.58% | **10.79%** | 27.63% | **10.46%** | 7.05% |
| equities | 44.68% | **13.94%** | 34.38% | **12.51%** | 10.19% |
| btc | 47.98% | **4.95%** | 33.74% | **3.84%** | 6.30% |
| cl | 40.48% | **17.84%** | 30.72% | **15.64%** | 12.14% |
| wheat | 53.37% | **55.47%** | 42.33% | **24.28%** | 57.72% |
| eurodollar | 45.88% | **13.79%** | 40.70% | **9.86%** | 15.89% |
| treasury_2y | 37.33% | **37.48%** | 34.96% | **22.10%** | 41.38% |

---

## Notes

- Layer 6 (LightGBM) evaluates all 21 classifiers (7 assets × 3 timeframes) on the full 667-sample test set.
- Layer 7 (GPT-4o) evaluates on 667 test tweets (same full set as Layer 6, enabling direct comparison).
- LLM accuracy reflects GPT-4o's final direction verdict compared to ground-truth labels.
- Agreement rate measures how often LLM and LightGBM predict the same direction (not necessarily correct).
- LLM results are for the specified timeframe only; LightGBM is evaluated across all three timeframes.
- Where the LLM call failed for a sample, LightGBM's prediction is used as fallback.