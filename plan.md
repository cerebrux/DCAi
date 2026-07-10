# DCAi — Πλάνο Υλοποίησης Branch `260710`

> Με τη λογική **Medallion Fund (RenTech)**: μέγιστο signal-to-noise ratio, αυστηρή στατιστική σημαντικότητα, μηδενικό repainting, systematic bottom detection για monthly DCA entries.

---

## Φάση 1: Feature Engineering — 6 νέα features + Weighted KNN

### 1.1 Νέα Features για KNN (γραμμές ~208-210)

| # | Feature | Συνάρτηση | Στόχος |
|---|---|---|---|
| **F4** | RSI(14) percentile rank | `ta.percentrank(ta.rsi(close, 14), 100)` | Ανεξάρτητη επιβεβαίωση oversold |
| **F5** | Bollinger %B | `(close - ta.sma(close, 20)) / (2 * ta.stdev(close, 20))` σε percentile | Πόσο "τεντωμένη" είναι η τιμή |
| **F6** | Απόσταση από 200-period MA | `(close - ta.sma(close, 200)) / ta.sma(close, 200)` σε percentile | Macro context για μεγάλα bottoms |
| **F7** | Volume Ratio | `volume / ta.sma(volume, 50)` σε percentile | Volume spike = capitulation |
| **F8** | Rolling 5-bar return std | `ta.stdev(close/close[1] - 1, 5)` σε percentile | Regime detection (ηρεμία/πανικός) |
| **F9** | Lagged versions (F4-F8, 1 bar lag) | Ίδια λογική με f1_lag..f3_lag | Momentum context |

**Αλλαγές:**
- `get_lorentzian_distance()`: επέκταση παραμέτρων για 9 features + 9 lagged = 18 distances αντί για 6
- `f1_history..f3_lag_history`: επέκταση σε `f1_history..f9_lag_history` (18 arrays)
- `train_labels`: ίδια λογική, αμετάβλητο

### 1.2 Weighted KNN Voting (γραμμές ~284-287)

**Αντί για** απλή πλειοψηφία:
```pine
prob = neighbors_found > 0 ? (prediction_sum / neighbors_found) * 100.0 : 0.0
```

**Θα γίνει** inverse-distance weighting:
```pine
float weight_sum = 0.0
float weighted_vote = 0.0
for k = 0 to neighbors_found - 1
    float w = 1.0 / (1.0 + array.get(distances, k))
    weight_sum += w
    if array.get(predictions, k) > 0.5
        weighted_vote += w
prob = weight_sum > 0 ? (weighted_vote / weight_sum) * 100.0 : 0.0
```

---

## Φάση 2: KNN Optimization

### 2.1 Adaptive Probability Threshold (γραμμή ~201-202)

Αντί για στατικό `knn_prob_thresh`:
- Νέο array `prediction_hits` (σωστό/λάθος prediction την τελευταία N περίοδο)
- `rolling_accuracy = ta.sma(array.avg(prediction_hits), 50)`
- `effective_thresh = 50 + rolling_accuracy * 20` (clamped 55-85)

### 2.2 Sigmoid Confidence Calibration (γραμμές ~406-408)

Αντί για `math.pow((prob - 50.0) / 50.0, 0.5)`:
```pine
ml_pot_factor = prob > 50 ? 1.0 / (1.0 + math.exp(-0.08 * (prob - 65))) : 0.0
```

### 2.3 Walk-Forward Validation Tracker (γραμμές ~579-591)

- Νέος πίνακας `ml_prediction_log` (predicted prob, actual outcome)
- Νέο dashboard metric: **"ML Accuracy"** — % των predictions που επαληθεύτηκαν στο prediction_window
- Χρώμα: πράσινο αν > 60%, πορτοκαλί 40-60%, κόκκινο < 40%

---

## Φάση 3: Decision Engine Refinements

### 3.1 Volume-Weighted Adaptive MFI Threshold (γραμμές ~327-336)

Αντί για στατικό `mfi_strong_lvl = 35`:
- Υπολογισμός median MFI τελευταίων 200 μπαρ
- `adaptive_mfi_strong = ta.percentile_linear_interpolation(mf, 50, 200) * 0.5 + 15`
- Χρήση μόνο όταν `volume` > ta.sma(volume, 50) * 1.2 (volume confirmation)

---

## Commit Log

| Commit | Περιεχόμενο |
|---|---|
| `f742e1f` | `docs: add implementation plan for branch 260710` |
| `4ca22c1` | `feat: add 5 new KNN features (F4-F9) and weighted KNN voting` |
| `231f9e6` | `feat: adaptive threshold, sigmoid calibration, walk-forward validation` |
| `be08a3b` | `feat: volume-weighted adaptive MFI threshold` |

---

## Υλοποιημένες Αλλαγές — Ανασκόπηση

### Phase 1: Feature Engineering + Weighted KNN ✅

| # | Αλλαγή | Τοποθεσία (γραμμές) |
|---|---|---|
| 1.1 | **F4: RSI(14) percentile** — oversold confirmation | ~211 |
| 1.2 | **F5: Bollinger %B percentile** — price stretch from mean | ~212-213 |
| 1.3 | **F6: MA200 distance percentile** — macro context for bottoms | ~214-215 |
| 1.4 | **F7: Volume ratio percentile** — capitulation volume spikes | ~216 |
| 1.5 | **F8: Return std percentile** — volatility regime detection | ~217 |
| 1.6 | **F4_lag..F8_lag** — lagged features for momentum context | ~219 |
| 1.7 | **Lorentzian distance expanded** — 8 current + 8 lagged = 16 log-distances | ~226-242 |
| 1.8 | **History arrays ×8** — f1..f8_history + f1..f8_lag_history | ~244-259 |
| 1.9 | **Training data push ×16** — all features + lagged stored per bar | ~264-280 |
| 1.10 | **Circular buffer shift ×16** — all arrays trimmed at knn_history | ~287-304 |
| 1.11 | **Weighted KNN** — `1/(1+distance)` weighting instead of simple majority | ~333-343 |

### Phase 2: KNN Optimization ✅

| # | Αλλαγή | Τοποθεσία (γραμμές) |
|---|---|---|
| 2.1 | **Walk-forward validation** — prediction_log_bar + prediction_log_prob + prediction_hits arrays | ~358-391 |
| 2.2 | **Rolling accuracy** — `array.avg(prediction_hits)` over last 100, min 10 samples | ~382-383 |
| 2.3 | **Adaptive threshold** — `50 + (accuracy - 0.5) * 70`, clamped 55-85 | ~384-386 |
| 2.4 | **Sigmoid calibration** — `1/(1+e^(-0.08*(prob-65)))` for ml_pot_factor | ~519-520 |
| 2.5 | **Dashboard: ML Accuracy row** — rolling accuracy % + sample count | ~705-713 |

### Phase 3: Decision Engine ✅

| # | Αλλαγή | Τοποθεσία (γραμμές) |
|---|---|---|
| 3.1 | **Volume-weighted MFI** — vol_ratio > 1.2 → +5 threshold relaxation | ~417-434 |

---

## No Repainting Guarantee (Verified)

| Αλλαγή | Status |
|---|---|
| Όλα τα νέα features (`ta.rsi`, `ta.sma`, `ta.stdev`) | ✅ Built-in συναρτήσεις χωρίς look-ahead |
| Weighted KNN over existing neighbors | ✅ Μόνο math, no repaint |
| Adaptive threshold από rolling_accuracy | ✅ Verification γίνεται μόνο αφού κλείσει το prediction_window |
| Sigmoid calibration | ✅ Καθαρό math πάνω στο prob |
| Walk-forward arrays | ✅ Store/verify με `barstate.isconfirmed` + `bar_index` check |
| Volume-weighted MFI | ✅ Standard ta.sma, no repaint |

---

## Μετρικές Επιτυχίας (Προς Επιβεβαίωση με Backtest)

- **ML Hit Rate > 65%** (ποσοστό predictions που επαληθεύονται στο prediction_window)
- **DCAi Avg Entry** < Blind DCA Avg Entry (βελτίωση entry price)
- **Sortino Ratio** DCAi > Blind DCA
- **Adaptive threshold range**: 55-85% (παρακολούθηση στο dashboard)
