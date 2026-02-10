# DCAi: Machine Learning Based DCA Strategy

### Technical White Paper & Documentation

## Abstract

**DCAi** is a high-fidelity algorithmic trading framework designed to optimize the traditional Dollar-Cost Averaging (DCA) methodology. By integrating **K-Nearest Neighbors (KNN)** classification with **Lorentzian Distance** and adaptive budget allocation, the strategy aims to lower the average entry price while managing downside risk more effectively than "blind" passive investing.

---

## 1. Core Architecture

### 1.1 Machine Learning Engine (KNN)

The system utilizes a non-parametric ML model to identify historical price "bottoms".

* 
**Feature Engineering**: The model normalizes three key market dimensions: Money Flow Index (MFI), Rate of Change (ROC), and Average True Range (ATR) into percentile ranks.

* 
**Distance Metric**: It employs **Lorentzian Distance**, a robust metric that uses log-based damping to ensure large feature variations do not distort pattern matching.

* 
**Classification**: The model finds the  closest historical neighbors to predict the probability of a bullish outcome over a 4-bar forward-looking window.

### 1.2 Adaptive Asset Sensitivity

DCAi dynamically adjusts its sensitivity (Rho) and momentum thresholds based on the selected asset class:
| Asset Class | MFI Target Min | MFI Target Max | Sensitivity (Rho) |
| :--- | :--- | :--- | :--- |
| **Crypto** | 0  | 35  | 1.7  |
| **Stocks** | 0  | 55  | 2.0  |
| **Indices** | 30  | 48  | 2.5  |

---

## 2. Decision Engine Logic

The strategy prioritizes trades into three distinct tiers based on signal conviction and liquidity availability:

1. **Tier 1: FEAR BUY (Extreme Panic)**
* Triggered when MFI is below the "Panic" threshold (default 20).

* Uses a **Max Multiplier** and aggressive pot allocation.

2. **Tier 2: OVERSOLD BUY (Standard Dip)**
* Triggered in oversold conditions (MFI < 35) with ML confirmation.

* Uses a **Strong Boost** multiplier (default 1.5x).

3. **Tier 3: PULLBACK BUY (Trend Following)**
* Occurs in healthy uptrends when the price is in the "discount zone" below the Ichimoku Kijun-sen but above the Kumo cloud.

---

## 3. Financial Engineering & Budgeting

### 3.1 Smart Budgeting (The Savings Pot)

If no buy signal is triggered within a calendar month, the monthly budget is automatically rolled over into a **Savings Pot**. This "carryover" mechanism allows for massive capital deployment during black swan events or extreme market lows.

### 3.2 Dynamic Position Sizing

The investment amount is calculated using an **Inverse-Price Weighting** formula:

* 
**Exponential Scaling**: As price falls relative to the historical average, the buy amount increases exponentially based on the **Rho** parameter.

* 
**Confidence Boost**: ML probability acts as a multiplier; a 90%+ confidence level triggers more aggressive pot usage.

---

## 4. Risk Mitigation & Metrics

* 
**Sortino Ratio Focus**: Unlike the Sharpe ratio, DCAi prioritizes the **Sortino Ratio**, which ignores "good" upside volatility and only penalizes "bad" downside swings.

* 
**Max Drawdown Tracking**: The script continuously calculates the peak-to-trough decline of both the smart strategy and a passive benchmark.

* 
**CVD Validation**: Cumulative Volume Delta is used to detect bullish divergences—where price makes a new low but volume remains stable—providing a secondary layer of "smart money" confirmation.

---

## How to Use

1. Copy the Pine Script code into the **TradingView** Pine Editor.
2. Select your **Asset Class** in the settings to auto-configure volatility parameters.
3. Monitor the **Live Dashboard** for real-time ML confidence and budget status.

---
## License
This project is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**. 

### What does this mean?
- **Commercial Use**: You can use this for commercial purposes.
- **Modification**: You can modify the code.
- **Source Disclosure**: If you use this code in a web application or service (SaaS), you **must** release your source code under the same license.
- **Attribution**: You must keep the original copyright and license notice in all copies.