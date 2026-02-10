# DCAi: Machine Learning Based DCA Strategy

<div align="center">
  <img src="images/dcai-logo.png" width="150" alt="DCAi Logo">
</div>

---

**Dadicated to all HODLers:**

> *We don’t panic sell, we DCA the dips so hard they file a class‑action restraining order.*

[![Donate](https://img.shields.io/badge/Donate-Support%20DCAi-brightgreen)](https://donate.utappia.org/)
![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)
![Pine Script](https://img.shields.io/badge/Pine%20Script-v6.0-green.svg)
---
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
## 8. Frequently Asked Questions (FAQ)

### Q1: How does the "Savings Pot" work in practice?
**A:** If your monthly budget is, for example, $500 and the market is in a parabolic uptrend with no buy signals, that capital isn't lost. It accumulates in the **Savings Pot**. When the algorithm eventually detects a high-conviction opportunity (Tier 1 or Tier 2), it draws from this reserve to buy more units, significantly lowering your average entry price during "blood in the streets" scenarios.

### Q2: Why use Lorentzian Distance instead of standard Euclidean Distance?
**A:** Euclidean distance (straight-line) is highly sensitive to outliers. In financial markets, "flash crashes" or news-driven spikes are common. **Lorentzian Distance** uses logarithmic compression, which allows the algorithm to recognize the underlying "shape" of a price pattern even if the magnitude of the movement differs from historical examples.

### Q3: Is DCAi suitable for Day Trading or Scalping?
**A:** No. DCAi is an investment-grade framework designed for **Swing Traders** and **Long-term Investors**. It performs best on Daily (D) or Weekly (W) timeframes. Using it on low timeframes (e.g., 1-minute or 5-minute) may result in excessive signals caused by market noise, leading to premature capital exhaustion.

### Q4: What exactly does the Sensitivity ($\rho$) parameter control?
**A:** The $\rho$ (Rho) parameter determines how aggressively the algorithm scales its position size relative to price drops. 
* **High $\rho$**: The buy amount increases exponentially as the price falls further below the mean.
* **Low $\rho$**: The allocation is more linear, behaving closer to traditional, flat-rate DCA.

### Q5: How does the "Asset Selection" impact the strategy?
**A:** Different assets have different "volatility signatures." For example, the Money Flow Index (MFI) on Bitcoin can stay oversold much longer than on the S&P 500. By selecting the correct **Asset Class**, the algorithm automatically re-calibrates its "Panic" and "Oversold" thresholds to match that specific market's behavior.

### Q6: Can I use DCAi for an automated Trading Bot or Web Service?
**A:** Yes, you are permitted to do so under the **AGPL-3.0 License**. However, the "Network Interaction" clause of the AGPL states that if you run a modified version of this script on a server (SaaS), you **must** make your modified source code available to the users of that service.

### Q7: Does the ML engine "repaint"?
**A:** No. The KNN classification is calculated on bar closes. Once a bar is confirmed and the signal is printed, the historical pattern matching for that specific point in time remains fixed. This ensures that backtesting results are representative of real-world performance.

---
## 9. References & Academic Foundation

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

## 10. Disclaimer & Risk Warning

### 10.1 No Financial Advice
**THIS IS NOT FINANCIAL ADVICE.** This software and documentation are provided for **educational and research purposes only**. Nothing herein constitutes investment advice, a recommendation, or an endorsement of any security or investment strategy. The authors are not licensed financial advisors, investment advisors, or registered representatives. Do not rely on this software for investment decisions.

### 10.2 Assumption of Risk
**You use this software entirely at your own risk.** Trading and investing in ANY asset class (crypto, stocks, indices, commodities) involve substantial risk of loss, including the potential loss of your entire principal. No trading strategy, algorithm, or machine learning model—including DCAi—can guarantee profits or prevent losses. Market conditions are unpredictable and can change rapidly due to:
- Geopolitical events and news shocks
- Technical platform failures (TradingView, exchanges, brokers)
- Flash crashes, circuit breakers, and execution slippage
- Regulatory changes and sudden liquidity withdrawal
- Black Swan events and unforeseen systemic risks

### 10.3 No Warranties
Under the **AGPL-3.0 License**, this software is provided **"AS IS"** without any warranty whatsoever. The authors explicitly disclaim:
- Any express or implied warranty of merchantability
- Any warranty of fitness for a particular purpose
- Any warranty that the software will be error-free or uninterrupted
- Any warranty regarding accuracy, completeness, or usefulness of results
- Any warranty about the performance of the KNN algorithm or ML predictions

### 10.4 Limitation of Liability
**The authors shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages**, including but not limited to:
- Financial losses or costs incurred from using or relying on this software
- Lost profits, loss of investment capital, or opportunity costs
- Data loss or corruption
- Third-party claims or regulatory fines
- Any other damages arising out of or in connection with this software

This limitation applies regardless of whether damages were foreseeable or whether the authors were advised of the possibility of such damages.

### 10.5 Backtesting & Past Performance Limitations
- **Historical results do not guarantee future performance.** Backtesting uses historical data and cannot account for future market conditions, regime changes, or structural breaks.
- **Backtesting bias**: Model parameters may be overfit to past data. KNN models can have reduced predictive power in novel market conditions.
- **Execution reality**: Simulated results assume perfect order execution. Real-world trading incurs slippage, fees, commissions, and liquidity constraints.
- **No repainting disclaimer**: While the KNN model does not "repaint," TradingView's bar aggregation or time zone settings may cause signals to appear at different times than expected.

### 10.6 Technical & Platform-Specific Risks
- **Pine Script Limitations**: This is a Pine Script v6 indicator running on TradingView. It is dependent on:
  - TradingView's data feeds, which may contain gaps, errors, or delays
  - The stability and availability of the TradingView platform
  - Correct bar aggregation and time zone settings
- **Broker/Exchange Risks**: Execution of trades based on DCAi signals depends on your broker's systems, liquidity, and regulatory compliance. DCAi has no control over broker order execution.
- **Connectivity & Latency**: Internet outages, platform downtime, or network delays may prevent signal execution.

### 10.7 ML & Algorithm Limitations
- **KNN is non-parametric**: The algorithm relies on historical similarity. In truly novel market regimes, performance may degrade significantly.
- **Feature engineering limitations**: The three features (MFI, ROC, ATR) are normalized metrics. Extreme events (market halts, regulatory interventions) may break this model.
- **Lookback window constraint**: The model is limited to ~2,800 bars of history due to Pine Script computational constraints.
- **No guarantee of convergence**: The "4-bar forward prediction window" is arbitrary and may not capture the true reversal zone in all markets.

### 10.8 Asset Class Dependency
DCAi's behavior is highly asset-dependent. The pre-configured parameters (MFI thresholds, Rho sensitivity) may work well for the tested asset classes (Crypto, Stocks, Indices, Commodities) but:
- May not generalize to other assets (forex, futures, micro-caps)
- Require different tuning for different market conditions (bull vs. bear markets)
- May experience unexpected behavior on assets with low liquidity or unusual volatility profiles

### 10.9 User Responsibility
**You are solely and entirely responsible** for:
- Your own investment decisions and capital allocation
- Conducting thorough due diligence before using this software
- Understanding the risks and limitations outlined above
- Monitoring and validating signals before executing trades
- Complying with all applicable laws and regulations in your jurisdiction
- Consulting with a qualified, licensed financial advisor before deploying capital

### 10.10 No Endorsement
The inclusion of academic references in this README does not imply endorsement by those authors or institutions. Academic research is the foundation for concepts (DCA, KNN, Lorentzian Distance) but does not validate the specific implementation or results of DCAi.

---

**By using this software, you acknowledge that you have read, understood, and agree to assume all risks outlined above.**

---
## How to Use
1.  Copy the Pine Script code into the **TradingView** Pine Editor.
2.  Select your **Asset Class** in the settings to auto-configure volatility parameters.
3.  Select the **Start Date** of your DCA 
4.  Monitor the **Live Dashboard** for real-time ML confidence and budget status.


## License
This project is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**. 

### What does this mean?
- **Commercial Use**: You can use this for commercial purposes.
- **Modification**: You can modify the code.
- **Source Disclosure**: If you use this code in a web application or service (SaaS), you **must** release your source code under the same license.
- **Attribution**: You must keep the original copyright and license notice in all copies.