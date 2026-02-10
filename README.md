# DCAi: Machine Learning Based DCA Strategy
![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)

## Abstract
**DCAi** is a high-fidelity algorithmic trading framework designed to optimize the traditional Dollar-Cost Averaging (DCA) methodology. By integrating **K-Nearest Neighbors (KNN)** classification with **Lorentzian Distance** and adaptive budget allocation, the strategy aims to lower the average entry price while managing downside risk more effectively than "blind" passive investing.

---
![DCAi](images/btcusdc-scrnsht-dcai.png)
---

## 1. The Inefficiency of Static DCA
Traditional Dollar-Cost Averaging (DCA) is a passive strategy that executes purchases at fixed intervals regardless of market valuations. While psychologically effective for retail investors, it suffers from several mathematical inefficiencies:

* **Opportunity Cost of Capital**: Static DCA fails to capitalize on deep market discounts (Black Swan events), allocating the same capital at "all-time highs" and "local bottoms."
* **Linear Exposure**: It leads to a suboptimal average entry price in highly volatile markets, as it does not scale exposure relative to historical standard deviations.
* **Fixed Exhaustion**: During prolonged bear markets, static DCA may exhaust capital too early, missing the ultimate "generational bottom."

---

## 2. Core Architecture

### 2.1 Machine Learning Engine (KNN)
The system utilizes a non-parametric ML model to identify historical price "bottoms".

* **Feature Engineering**: The model normalizes three key market dimensions: Money Flow Index (MFI), Rate of Change (ROC), and Average True Range (ATR) into percentile ranks.
* **Distance Metric**: It employs **Lorentzian Distance**, a robust metric that uses log-based damping to ensure large feature variations do not distort pattern matching.
* **Classification**: The model finds the $K$ closest historical neighbors to predict the probability of a bullish outcome over a 4-bar forward-looking window.

### 2.2 Adaptive Asset Sensitivity
DCAi dynamically adjusts its sensitivity ($\rho$) and momentum thresholds based on the selected asset class:

| Asset Class | MFI Target Min | MFI Target Max | Sensitivity ($\rho$) |
| :--- | :--- | :--- | :--- |
| **Crypto** | 0 | 35 | 1.7 |
| **Stocks** | 0 | 55 | 2.0 |
| **Indices** | 30 | 48 | 2.5 |

---

## 3. Decision Engine Logic
The strategy prioritizes trades into three distinct tiers based on signal conviction and liquidity availability:

1.  **Tier 1: FEAR BUY (Extreme Panic)**
    * Triggered when MFI is below the "Panic" threshold (default 20).
    * Uses a **Max Multiplier** and aggressive pot allocation.
2.  **Tier 2: OVERSOLD BUY (Standard Dip)**
    * Triggered in oversold conditions (MFI < 35) with ML confirmation.
    * Uses a **Strong Boost** multiplier (default 1.5x).
3.  **Tier 3: PULLBACK BUY (Trend Following)**
    * Occurs in healthy uptrends when the price is in the "discount zone" below the Ichimoku Kijun-sen but above the Kumo cloud.

---

## 4. Financial Engineering & Budgeting

### 4.1 Smart Budgeting (The Savings Pot)
If no buy signal is triggered within a calendar month, the monthly budget is automatically rolled over into a **Savings Pot**. This "carryover" mechanism allows for massive capital deployment during black swan events or extreme market lows.

### 4.2 Dynamic Position Sizing
The investment amount is calculated using an **Inverse-Price Weighting** formula:

* **Exponential Scaling**: As price falls relative to the historical average, the buy amount increases exponentially based on the **$\rho$** parameter.
* **Confidence Boost**: ML probability acts as a multiplier; a 90%+ confidence level triggers more aggressive pot usage.

---

## 5. Comparative Analysis: Pros & Cons

### 5.1 Advantages of DCAi
* **Dynamic Sensitivity**: Utilizes the $\rho$ parameter to exponentially increase buy size during extreme deviations.
* **ML-Driven Confirmation**: The KNN engine filters out "falling knives" by requiring historical pattern similarity.
* **Capital Preservation**: The **Savings Pot** ensures capital is preserved for high-probability reversal zones.

### 5.2 Limitations & Risks
* **Computational Overload**: KNN models in Pine Script are limited by the lookback window (max 3000 bars).
* **Overfitting Risk**: Risk that Lorentzian distance parameters may overfit to recent price action.
* **Execution Latency**: Relies on bar closes, which might result in slightly higher entries during fast V-shape recoveries.

---

## 6. Future Research & Roadmap
1.  **Sentiment Integration**: Incorporating on-chain data or Funding Rates as additional features.
2.  **Multi-Timeframe Validation**: Correlating 4-bar predictions across Daily and Weekly intervals.
3.  **Recursive Learning**: Implementing a self-correcting mechanism for the $\rho$ parameter based on realized drawdown.

---
## 7. Configuration & Parameters

DCAi offers a highly granular settings menu to align the algorithm with your specific risk profile and asset class.

![Settings Menu](images/dcai-settings.png)

### 7.1 Asset Selection & Auto-Optimization
* **Asset Class**: Choose between `Crypto`, `Stocks`, `Indices`, or `Commodities`. This selection automatically adjusts:
    * **MFI Thresholds**: Tailored to the typical volatility of each sector.
    * **Adaptive Sensitivity ($\rho$)**: Controls how aggressively the position size increases during dips.
* **Auto-Optimize Parameters**: When enabled, the script ignores manual overrides and uses pre-calculated optimal values for the selected asset.

### 7.2 Machine Learning Settings (KNN)
* **Lookback Window**: Number of historical bars (up to 3000) the ML model uses to find similar patterns.
* **K-Neighbors**: The number of "nearest neighbors" compared (default is 10).
* **Probability Threshold**: The minimum ML confidence required to trigger a buy signal (typically 55-60%).

### 7.3 Financial Parameters (Budgeting)
* **Monthly Budget**: Your total investable capital per month.
* **Max Multiplier Cap**: Limits the maximum investment size for a single signal to prevent over-exposure.
* **Pot Recovery Rate**: Defines how fast the "Savings Pot" refills after a major deployment.

### 7.4 Technical Confirmation (Filtering)
* **Ichimoku Cloud Filter**: When active, the script prioritizes buys that occur within or near the cloud to avoid buying in "no-man's land."
* **CVD Divergence Filter**: Toggle on/off the requirement for Volume Delta confirmation.
* **Cooldown Period**: Number of bars to wait between two major signals to avoid "signal clustering."
---
## How to Use
1.  Copy the Pine Script code into the **TradingView** Pine Editor.
2.  Select your **Asset Class** in the settings to auto-configure volatility parameters.
3.  Select the **Start Date** of your DCA 
4.  Monitor the **Live Dashboard** for real-time ML confidence and budget status.

---

## License
This project is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**. 

### What does this mean?
- **Commercial Use**: You can use this for commercial purposes.
- **Modification**: You can modify the code.
- **Source Disclosure**: If you use this code in a web application or service (SaaS), you **must** release your source code under the same license.
- **Attribution**: You must keep the original copyright and license notice in all copies.