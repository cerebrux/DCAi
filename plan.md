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

## Commit Plan

| Commit | Περιεχόμενο |
|---|---|
| `1` | `feat: add 5 new KNN features (F4-F9) and expand Lorentzian distance` |
| `2` | `feat: weighted KNN voting with inverse-distance weighting` |
| `3` | `feat: adaptive probability threshold based on rolling accuracy` |
| `4` | `feat: sigmoid confidence calibration for position sizing` |
| `5` | `feat: walk-forward validation tracker in dashboard` |
| `6` | `feat: volume-adaptive MFI threshold` |
| `7` | `docs: update plan.md with implementation log and results` |

---

## Μετρικές Επιτυχίας

- **ML Hit Rate > 65%** (ποσοστό predictions που επαληθεύονται στο prediction_window)
- **DCAi Avg Entry** < Blind DCA Avg Entry (βελτίωση entry price)
- **Sortino Ratio** DCAi > Blind DCA
- **Αριθμός False Signals** μειωμένος ≥ 15% σε σχέση με baseline
