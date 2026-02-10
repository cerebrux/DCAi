# DCAi: Machine Learning Based DCA Strategy
![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)

**Dadicated to all HODLers:**

> *We don’t panic sell, we DCA the dips so hard they file a class‑action restraining order.*


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
* **Confidence Use**: The ML probability functions as a confidence score. It gates entries (minimum threshold) and scales position sizing via a confidence multiplier and savings pot usage; higher confidence increases deployment, while low confidence reduces or blocks entries.

### 2.2 Adaptive Asset Sensitivity
DCAi dynamically adjusts its sensitivity ($\rho$) and momentum thresholds based on the selected asset class:

| Asset Class | MFI Target Min | MFI Target Max | Sensitivity ($\rho$) |
| :--- | :--- | :--- | :--- |
| **Crypto** | 0 | 35 | 1.7 |
| **Stocks** | 0 | 55 | 2.0 |
| **Indices** | 30 | 48 | 2.5 |
| **Commodities** | 35 | 50 | 2.5 |

---

## 3. Decision Engine Logic
The strategy prioritizes trades into three distinct tiers based on signal conviction and liquidity availability:

1.  **Tier 1: PULLBACK BUY (Trend Following)**
    * Occurs in healthy uptrends when price trades in the "discount zone" below the Ichimoku Kijun-sen, above Leading Span B, and the cloud is green.
2.  **Tier 2: OVERSOLD BUY (Standard Dip)**
    * Activated in oversold conditions (MFI < 35), with dynamic relaxation when CVD divergence is bullish, and ML confirmation.
    * Uses a **Strong Boost** multiplier (default 1.5x).
3.  **Tier 3: FEAR BUY (Extreme Panic)**
    * Activated when MFI falls below the "Panic" threshold (default 20), with dynamic relaxation when CVD divergence is bullish.
    * Uses a **Max Multiplier** and aggressive pot allocation.

---

## 4. Financial Engineering & Budgeting

### 4.1 Smart Budgeting (The Savings Pot)
If no buy signal is triggered within a calendar month, the monthly budget is automatically rolled over into a **Savings Pot**. This "carryover" mechanism allows for massive capital deployment during black swan events or extreme market lows.

### 4.2 Dynamic Position Sizing
The investment amount is calculated using an **Inverse-Price Weighting** formula:

* **Exponential Scaling**: As price falls relative to the historical average, the buy amount increases exponentially based on the **$\rho$** parameter.
* **Confidence Boost**: ML probability acts as a multiplier; higher confidence increases the multiplier and pot usage on a continuous scale (starting above 50% confidence).

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
* **Manual Sensitivity ($\rho$)**: Overrides auto-optimization when auto is disabled.

### 7.2 Machine Learning Settings (KNN)
* **Lookback Window**: Number of historical bars (up to 2800) the ML model uses to find similar patterns.
* **K-Neighbors**: The number of "nearest neighbors" compared (default is 5).
* **Probability Threshold**: The minimum ML confidence required to trigger an entry signal (defaults to 70% for strong signals and 50% for pullback entries).
* **ML Confidence Sensitivity**: Controls how strongly the distance-based confidence multiplier scales position sizing.
* **ROC Length (Feature)**: Length used for the ROC feature in the KNN input set.

### 7.3 Financial Parameters (Budgeting)
* **Monthly Budget**: Your total investable capital per month.
* **Start Date / End Date**: Limits signals and budgeting to a defined trading window.
* **Max Multiplier Cap**: Limits the maximum investment size for a single signal to prevent over-exposure.
* **Strong Buy Boost**: Multiplier applied to oversold signals.
* **Max Buy Boost**: Multiplier applied to fear signals.
* **Pot Reserve (%)**: Portion of the savings pot held back for extreme dips.
* **Show Savings Pot Usage**: Displays pot usage labels on the chart when enabled.

### 7.4 Technical Confirmation (Filtering)
* **Ichimoku Display Options**: Toggles for Tenkan/Kijun lines, Chikou span, and Kumo fill.
* **Cooldown Period**: Number of bars to wait between strong signals to avoid signal clustering.

---
## How to Use
1.  Copy the Pine Script code into the **TradingView** Pine Editor.
2.  Select your **Asset Class** in the settings to auto-configure volatility parameters.
3.  Select the **Start Date** of your DCA 
4.  Monitor the **Live Dashboard** for real-time ML confidence and budget status.

---
# References & Academic Foundation

This project integrates concepts from behavioral finance, machine learning, and quantitative risk management. Below is the list of foundational research papers and literature used to design the **QIA (Quant Investment Assistant)** logic.

## 1. Dollar-Cost Averaging (DCA) & Behavioral Finance
*   **Statman, M. (1995).** "A Behavioral Framework for Dollar-Cost Averaging." *The Journal of Portfolio Management*, 22(1), 70-78.
    *   *Explores why investors prefer DCA for psychological reasons (regret minimization) despite mathematical sub-optimality in bull markets.*
*   **Thorley, S. R. (1994).** "The Fallacy of Dollar-Cost Averaging." *Financial Practice and Education*, 4(2), 17-26.
    *   *Analyzes the mathematical properties of DCA compared to Lump Sum investing.*
*   **Leggio, K. B., & Lien, D. (2001).** "Does Dollar Cost Averaging Make Sense?" *Financial Services Review*, 10(1), 73-86.
    *   *A critical assessment of DCA performance using risk-adjusted return metrics (Sortino/Sharpe).*

## 2. Machine Learning & K-Nearest Neighbors
*   **Fix, E., & Hodges, J. L. (1951).** "Discriminatory Analysis: Nonparametric Discrimination: Consistency Properties." *USAF School of Aviation Medicine*, Randolph Field, Texas.
    *   *The seminal paper that introduced the K-Nearest Neighbors (KNN) algorithm.*
*   **De Prado, M. L. (2018).** *Advances in Financial Machine Learning.* Wiley.
    *   *The industry standard reference for applying ML techniques to financial time-series data.*
*   **Jansen, S. (2020).** *Machine Learning for Algorithmic Trading.* Packt Publishing.
    *   *Practical implementation of ML strategies in trading systems.*

## 3. Technical Analysis & Volatility Dynamics
*   **Wilder, R. S. (1978).** *New Concepts in Technical Trading Systems.* Trend Research.
    *   *The origin of the **Average True Range (ATR)**, which represents the volatility component (Feature f3) in the QIA machine learning model.*
*   **Quong, G., & Soudack, A. (1989).** "Volume-Weighted RSI: Money Flow Index." *Technical Analysis of Stocks & Commodities*, 7(3).
    *   *The original publication introducing the **Money Flow Index (MFI)**, used here as the primary momentum/volume oscillator for identifying capitulation.*
*   **Murphy, J. J. (1999).** *Technical Analysis of the Financial Markets.* Penguin York Institute of Finance.
    *   *The standard textbook for technical analysis, providing the definitive definition of **Rate of Change (ROC)** as the purest measure of price velocity and momentum used in Feature f2.*
*   **Harris, L. (2003).** *Trading and Exchanges: Market Microstructure for Practitioners.* Oxford University Press.
    *   *Provides the theoretical framework for **Cumulative Volume Delta (CVD)** by analyzing the bid-ask spread and aggressive order flow (Delta) to identify buyer/seller absorption.*
*   **Hosoda, G. (Ichimoku Sanjin). (1969).** *Ichimoku Kinko Hyo (一目均衡表).* Economic Statistics Research Institute.
    *   *The original source for the **Ichimoku Cloud** theory, used in this system to define the equilibrium zones for trend validation.*

## 4. Risk-Adjusted Performance Metrics
*   **Sortino, F. A., & Price, L. N. (1994).** "Performance Measurement in a Downside Risk Framework." *The Journal of Investing*, 3(3), 59-65.
    *   *Introduction of the Sortino Ratio, distinguishing between harmful volatility (downside) and general volatility.*
*   **Young, T. W. (1991).** "Calmar Ratio: A Smoother Tool." *Futures Magazine*, 20(10), 40.
    *   *Introduction of the Calmar Ratio (CAGR / Max Drawdown) for evaluating hedge fund performance.*

## 5. Quantitative Strategy Optimization
*   **Pardo, R. (2008).** *The Evaluation and Optimization of Trading Strategies.* Wiley Trading.
    *   *Methodologies for backtesting, walk-forward analysis, and avoiding overfitting.*
*   **Narang, R. K. (2013).** *Inside the Black Box: A Simple Guide to Quantitative and High-Frequency Trading.* Wiley.
    *   *Insights into the structure of professional quant systems and alpha generation.*
---

## License
This project is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**. 

### What does this mean?
- **Commercial Use**: You can use this for commercial purposes.
- **Modification**: You can modify the code.
- **Source Disclosure**: If you use this code in a web application or service (SaaS), you **must** release your source code under the same license.
- **Attribution**: You must keep the original copyright and license notice in all copies.