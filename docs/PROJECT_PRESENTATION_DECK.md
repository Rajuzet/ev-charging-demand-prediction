# ⚡ End-to-End EV Charging Demand Prediction & Usage Pattern Intelligence
## Capstone Project Master Presentation Deck

---

### Slide 1: Title Slide & Project Metadata
- **Project Title:** End-to-End Electric Vehicle (EV) Charging Demand Prediction, Behavioral Classification, and Grid Load Forecasting
- **Domain:** Clean Energy Systems, Smart Grid Infrastructure, Machine Learning, & Time-Series Analytics
- **Technologies:** Python, Scikit-Learn, XGBoost, LightGBM, Statsmodels, SHAP, LIME, Plotly, Streamlit
- **Project Repository:** [GitHub Repository: Rajuzet/ev-charging-demand-prediction](https://github.com/Rajuzet/ev-charging-demand-prediction.git)
- **Live Interactive System:** Streamlit Web Application (Headless Port `8501`)

---

### Slide 2: Executive Summary & Problem Statement
- **The Challenge:**
  The rapid global transition to Electric Vehicles creates severe stress on municipal electric distribution grids. Uncoordinated EV charging causes transformer thermal overloading, unpredictable local demand spikes, and inefficient capacity utilization.
- **Core Objectives:**
  1. **Demand Regression:** Predict exact energy delivery ($\text{Demand}$ in $\text{kWh}$) per individual vehicle charging session.
  2. **Demand Classification:** Classify high-load charging sessions ($\ge 14.20\text{ kWh}$) to allow proactive grid dispatch and smart throttling.
  3. **Time-Series Forecasting:** Model long-term macroscopic daily energy demand across the entire charging network.
  4. **Explainable AI (XAI):** Uncover global and local feature attributions using SHAP and LIME for transparent grid operator decisions.
  5. **Enterprise Application:** Deliver a 5-tab production dashboard for dynamic CSV ingestion, interactive session simulation, and automated batch scoring.

---

### Slide 3: EV Ecosystem Stakeholders & Utility
```
                  ┌────────────────────────────────────────┐
                  │       EV Charging Infrastructure       │
                  └──────────────────┬─────────────────────┘
                                     │
         ┌───────────────────────────┼───────────────────────────┐
         ▼                           ▼                           ▼
┌──────────────────┐       ┌──────────────────┐       ┌──────────────────┐
│  Grid Operators  │       │ Charging Station │       │   EV Drivers &   │
│   (Utilities)    │       │  Owners (CPOs)   │       │ Fleet Managers   │
├──────────────────┤       ├──────────────────┤       ├──────────────────┤
│• Peak shaving    │       │• Dynamic pricing │       │• Transparent fee │
│• Transformer     │       │• Hardware uptime │       │  estimations     │
│  health monitoring│      │  optimization    │       │• Session time    │
│• Renewable power │       │• Queue reduction │       │  planning        │
│  balancing       │       │  strategies      │       │• Charger choice  │
└──────────────────┘       └──────────────────┘       └──────────────────┘
```

---

### Slide 4: Raw Dataset Architecture & Properties
- **Data Source:** Real-world EV charging station transactions logged from 2021-09-30 to 2022-09-30.
- **Initial Volume:** $72,856$ raw transaction rows.
- **Initial Feature Set:**
  1. `StartDatetime`: Timestamp of plug-in arrival ($YYYY\text{-}MM\text{-}DD\ HH:MM:SS$).
  2. `EndDatetime`: Timestamp of unplug completion.
  3. `Duration`: Session duration in hours/minutes.
  4. `Location`: Categorical venue name ($14$ distinct sites: Mall, Hotel, Office, Apartment, etc.).
  5. `ChargerType`: Integer flag ($0 = \text{Level 2 AC Standard}$, $1 = \text{DC Fast Charger}$).
  6. `ChargerCompany`: Integer flag ($0 = \text{Operator Network A}$, $1 = \text{Operator Network B}$).
  7. `Demand`: Target variable — Total energy consumed ($y_i \in \mathbb{R}^+$ in $\text{kWh}$).

---

### Slide 5: Data Cleaning & Preprocessing Pipeline
- **Step 1: Duplicate Elimination**
  - Identified and purged $13$ redundant transaction rows:
    $$N_{\text{dedup}} = 72,856 - 13 = 72,843$$
- **Step 2: Timestamp Parsing & Chronological Ordering**
  - Converted strings to POSIX datetime objects and enforced condition:
    $$\Delta t = t_{\text{end}} - t_{\text{start}} > 0$$
- **Step 3: Extreme Outlier & Anomaly Filtering**
  - Duration anomalies filtered:
    $$\text{Valid Duration: } 2.0\text{ mins} \le \text{Duration} \le 1,440\text{ mins (24 hours)}$$
  - Demand anomalies filtered:
    $$\text{Valid Demand: } 0.1\text{ kWh} \le \text{Demand} \le 120.0\text{ kWh}$$
  - Negative and extreme records purged: $422$ records removed.
  - Final clean analytical base table ($N_{\text{clean}}$): **$72,421$ validated records**.

---

### Slide 6: Feature Engineering & Mathematical Formulations
To capture non-linear, diurnal, and temporal patterns, $28$ features were engineered:

1. **Cyclic Hour Encoding:**
   Hours $t \in \{0, 1, \dots, 23\}$ have a circular boundary (Hour 23 is 1 hour away from Hour 0). Linear representation creates an artificial discontinuity ($|23 - 0| = 23$).
   $$\text{Hour\_Sin} = \sin\left(\frac{2\pi t}{24}\right), \quad \text{Hour\_Cos} = \cos\left(\frac{2\pi t}{24}\right)$$
   *Mathematical Proof of Distance Invariance:*
   $$(\text{Hour\_Sin})^2 + (\text{Hour\_Cos})^2 = \sin^2\theta + \cos^2\theta = 1 \quad \forall t$$
   Euclidean distance between Hour 23 and Hour 0 on the unit circle:
   $$D(23, 0) = \sqrt{\left(\sin\frac{46\pi}{24} - \sin 0\right)^2 + \left(\cos\frac{46\pi}{24} - \cos 0\right)^2} = 2\sin\left(\frac{\pi}{24}\right) \approx 0.261$$

2. **Temporal Categorizations:**
   - $\text{IsWeekend} = \mathbb{I}(\text{DayOfWeek} \in \{5, 6\})$, where $\mathbb{I}$ is the indicator function.
   - $\text{TimeOfDay} = \begin{cases} \text{Morning}, & 06:00 \le t < 12:00 \\ \text{Afternoon}, & 12:00 \le t < 17:00 \\ \text{Evening}, & 17:00 \le t < 21:00 \\ \text{Night}, & \text{otherwise} \end{cases}$

3. **Target Encoding & Interaction Features:**
   - $\text{Loc\_Avg\_Demand} = \frac{1}{|S_L|}\sum_{j \in S_L} y_j$, where $S_L$ is the set of historical sessions at location $L$.
   - $\text{ChargerType\_Avg\_Demand} = \frac{1}{|S_C|}\sum_{j \in S_C} y_j$.
   - $\text{Avg\_Power\_kW} = \frac{\text{Demand}}{\text{Duration\_Mins} / 60}$.

---

### Slide 7: Exploratory Data Analysis (EDA) Discoveries
1. **Diurnal Inflow Waves:**
   - Primary arrival peak: $08:00\text{--}10:00$ (workplace plug-ins).
   - Secondary arrival peak: $17:00\text{--}19:00$ (residential/evening plug-ins).
   - Trough: $01:00\text{--}04:00$ ($< 2\%$ of total volume).
2. **Hardware Disparity:**
   - **Level 2 AC Chargers (Type 0):** $66.7\%$ of sessions, mean duration $\approx 220\text{ mins}$, mean delivery $\approx 12.8\text{ kWh}$.
   - **DC Fast Chargers (Type 1):** $33.3\%$ of sessions, mean duration $\approx 45\text{ mins}$, mean delivery $\approx 18.4\text{ kWh}$.
3. **Location Demand Distribution:**
   - Commercial centers, highway corridors, and workplaces drive $> 65\%$ of cumulative MWh load.
4. **Target Distribution:**
   - Median demand threshold: $\theta_{\text{demand}} = 14.20\text{ kWh}$.
   - High-demand split: $50.0\%$ baseline split across the full historical distribution.

---

### Slide 8: Track 1 — Regression Architecture (Demand in kWh)
- **Problem Formulation:**
  Given feature vector $\mathbf{x}_i \in \mathbb{R}^d$, estimate continuous target $\hat{y}_i \approx y_i \in \mathbb{R}^+$:
  $$f: \mathcal{X} \to \mathcal{Y}, \quad \hat{y} = f(\mathbf{x})$$
- **Algorithms Evaluated:**
  1. Linear Regression (Ordinary Least Squares baseline)
  2. Decision Tree Regressor (CART)
  3. Gradient Boosting Regressor (GBR)
  4. LightGBM Regressor (Light Gradient Boosting Machine)
  5. XGBoost Regressor (eXtreme Gradient Boosting)
  6. **Random Forest Regressor** (Ensemble Bagging)
- **Data Splitting Protocol:**
  - $80\%$ Training Set: $N_{\text{train}} = 57,936$ sessions.
  - $20\%$ Test Set: $N_{\text{test}} = 14,485$ sessions.
  - Random seed: $42$ (reproducible across all benchmarks).

---

### Slide 9: Regression Mathematical Formulations & Symbol Glossary
1. **Mean Absolute Error (MAE):**
   $$\text{MAE} = \frac{1}{n}\sum_{i=1}^n |y_i - \hat{y}_i|$$
   - $n$: Total number of test observations ($n = 14,485$).
   - $y_i$: Ground-truth energy delivered in the $i$-th session (kWh).
   - $\hat{y}_i$: Model-predicted energy delivered in the $i$-th session (kWh).
   - $|y_i - \hat{y}_i|$: Absolute residual. Measures linear average deviation without quadratic penalty.

2. **Mean Squared Error (MSE):**
   $$\text{MSE} = \frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2$$
   - $(y_i - \hat{y}_i)^2$: Squared residual. Heavily penalizes large prediction errors quadratically.

3. **Root Mean Squared Error (RMSE):**
   $$\text{RMSE} = \sqrt{\text{MSE}} = \sqrt{\frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2}$$
   - Restores the error metric to original physical units ($\text{kWh}$).

4. **Coefficient of Determination ($R^2$ Score):**
   $$R^2 = 1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}} = 1 - \frac{\sum_{i=1}^n (y_i - \hat{y}_i)^2}{\sum_{i=1}^n (y_i - \bar{y})^2}$$
   - $\bar{y} = \frac{1}{n}\sum_{i=1}^n y_i$: Arithmetic sample mean of the actual demand values.
   - $\text{SS}_{\text{res}}$: Residual Sum of Squares (variance unexplained by model).
   - $\text{SS}_{\text{tot}}$: Total Sum of Squares (inherent dispersion of the ground truth).
   - Interpretation: $R^2 = 0.7983$ means $79.83\%$ of variance in charging demand is captured by the model.

---

### Slide 10: Regression Benchmark Comparison & Results
| Model Rank | Regressor Algorithm | MAE (kWh) | MSE | RMSE (kWh) | $R^2$ Score | Selection Status |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| 🥇 | **Random Forest Regressor** | **3.3767** | **36.9744** | **6.0807** | **0.7983** | 🏆 **Production Champion** |
| 🥈 | XGBoost Regressor | 3.4298 | 37.4721 | 6.1214 | 0.7956 | Runner-Up |
| 🥉 | LightGBM Regressor | 3.4760 | 37.9712 | 6.1621 | 0.7929 | High-Efficiency Alternate |
| 4 | Gradient Boosting | 3.5092 | 38.8914 | 6.2363 | 0.7878 | Competitive |
| 5 | Decision Tree | 3.5231 | 45.4982 | 6.7452 | 0.7518 | High Variance |
| 6 | Linear Regression | 5.6104 | 75.5023 | 8.6892 | 0.5882 | Underfitting |

---

### Slide 11: Track 2 — Classification Architecture (High Demand)
- **Problem Formulation:**
  Binary classification to forecast whether a session will place high load on the feeder:
  $$y_i^{\text{class}} = \begin{cases} 1, & \text{Demand} \ge 14.20\text{ kWh (High Demand)} \\ 0, & \text{Demand} < 14.20\text{ kWh (Regular Demand)} \end{cases}$$
- **Why Predict High Demand?**
  Grid dispatchers need advance warning to dynamically throttle secondary loads, engage battery energy storage systems (BESS), or adjust real-time pricing tiers.
- **Algorithms Evaluated:**
  1. Logistic Regression
  2. Support Vector Machine (LinearSVC)
  3. K-Nearest Neighbors (KNN)
  4. Decision Tree Classifier
  5. Random Forest Classifier
  6. LightGBM Classifier
  7. **XGBoost Classifier**

---

### Slide 12: Classification Mathematical Formulations & Symbol Glossary
Confusion Matrix Partition:
$$\begin{array}{c|cc} & \hat{y} = 1 & \hat{y} = 0 \\ \hline y = 1 & TP & FN \\ y = 0 & FP & TN \end{array}$$

1. **True Positive ($TP$):** High-demand session correctly classified as High Demand ($7,046$ sessions).
2. **True Negative ($TN$):** Regular-demand session correctly classified as Regular Demand ($6,009$ sessions).
3. **False Positive ($FP$):** Type I error — Regular session falsely flagged as High Demand ($1,011$ sessions).
4. **False Negative ($FN$):** Type II error — High-demand session missed by the model ($419$ sessions).

5. **Classification Formulas:**
   - **Accuracy:** Overall proportion of correct decisions:
     $$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} = \frac{7,046 + 6,009}{14,485} = 90.13\%$$
   - **Precision:** Reliability of a positive alert (minimizes unnecessary grid reserve throttling):
     $$\text{Precision} = \frac{TP}{TP + FP} = \frac{7,046}{7,046 + 1,011} = 87.45\%$$
   - **Recall (Sensitivity):** Grid security protection (minimizes missed high-load spikes):
     $$\text{Recall} = \frac{TP}{TP + FN} = \frac{7,046}{7,046 + 419} = 93.64\%$$
   - **F1-Score:** Harmonic mean balancing Precision and Recall:
     $$F_1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2TP}{2TP + FP + FN} = 0.9044$$
   - **ROC-AUC:** Area under Receiver Operating Characteristic curve plotting $\text{TPR} = \frac{TP}{TP + FN}$ vs $\text{FPR} = \frac{FP}{FP + TN}$ across all decision thresholds $\tau \in [0, 1]$:
     $$\text{AUC} = \int_{0}^{1} \text{TPR}(\tau) \, d(\text{FPR}(\tau)) = 0.9615$$

---

### Slide 13: Handling Class Imbalance — Resampling Comparison
When sub-segmenting peak-hour high-demand sessions, class imbalance emerges. Four strategies were benchmarked:
1. **Baseline:** Standard unweighted class frequencies.
2. **Random Oversampling:** Duplicating minority instances.
3. **SMOTE (Synthetic Minority Over-sampling Technique):**
   Generates synthetic samples along line segments connecting $k$-nearest neighbors:
   $$\mathbf{x}_{\text{syn}} = \mathbf{x}_i + \lambda (\mathbf{x}_{zi} - \mathbf{x}_i), \quad \lambda \sim \mathcal{U}(0, 1), \quad \mathbf{x}_{zi} \in k\text{-NN}(\mathbf{x}_i)$$
4. **ADASYN (Adaptive Synthetic Sampling):**
   Weights minority samples by difficulty of learning:
   $$r_i = \frac{\Delta_i / k}{Z}, \quad g_i = r_i \times G$$
   - $\Delta_i$: Number of majority class examples among the $k$-nearest neighbors of minority sample $\mathbf{x}_i$.
   - $Z$: Normalization constant such that $\sum r_i = 1$.
   - $G$: Total synthetic samples to generate.

*Resampling Benchmark Results:*
- Baseline: F1 = $0.8608$, Recall = $0.8315$, ROC-AUC = $0.9566$
- Random Oversampling: F1 = $0.8829$, Recall = $0.9164$, ROC-AUC = $0.9587$
- SMOTE: F1 = $0.8852$, Recall = $0.9175$, ROC-AUC = $0.9588$
- **ADASYN:** F1 = **$0.8903$**, Recall = **$0.9374$**, ROC-AUC = **$0.9600$** *(Highest minority sensitivity)*

---

### Slide 14: Classification Benchmark Comparison & Results
| Model Rank | Classifier Algorithm | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Selection Status |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 | **XGBoost Classifier** | **0.9013** | **0.8745** | **0.9364** | **0.9044** | **0.9615** | 🏆 **Production Champion** |
| 🥈 | LightGBM Classifier | 0.9011 | 0.8732 | 0.9381 | 0.9041 | 0.9610 | High-Speed Alternate |
| 🥉 | Random Forest | 0.8984 | 0.8661 | 0.9412 | 0.9019 | 0.9562 | Robust Ensemble |
| 4 | Decision Tree | 0.8920 | 0.8738 | 0.9158 | 0.8941 | 0.9299 | Rule-based Baseline |
| 5 | Logistic Regression | 0.8431 | 0.8281 | 0.8656 | 0.8464 | 0.9146 | Linear Boundary |
| 6 | SVM (LinearSVC) | 0.8114 | 0.7989 | 0.8302 | 0.8142 | 0.8954 | Large-Margin Separator |
| 7 | KNN ($k=5$) | 0.7823 | 0.7768 | 0.7883 | 0.7825 | 0.8602 | Non-parametric Baseline |

---

### Slide 15: Track 3 — Time-Series Analysis & Grid Load Forecasting
- **Dataset Aggregation:**
  Individual session events aggregated into a continuous daily time series:
  $$Y_t = \sum_{j \in \mathcal{S}_t} \text{Demand}_j \quad \text{for day } t = 1, 2, \dots, 366$$
  - Horizon: $2021\text{-}09\text{-}30$ through $2022\text{-}09\text{-}30$.
  - Mean daily demand: $\mu_Y \approx 2,750\text{ kWh/day}$.
- **Decomposition Model (Additive):**
  $$Y_t = T_t + S_t + R_t$$
  - $T_t$: Upward secular trend reflecting growing EV adoption.
  - $S_t$: 7-day cyclical seasonality (weekday commuting vs weekend charging).
  - $R_t$: Stochastic irregular remainder / white noise.

---

### Slide 16: Time-Series Mathematical Formulations & Symbol Glossary
1. **Augmented Dickey-Fuller (ADF) Stationarity Test:**
   $$\Delta Y_t = \alpha + \beta t + \gamma Y_{t-1} + \sum_{p=1}^P \delta_p \Delta Y_{t-p} + \varepsilon_t$$
   - $\Delta Y_t = Y_t - Y_{t-1}$: First-order backward difference operator.
   - $\alpha$: Drift intercept.
   - $\beta t$: Deterministic linear time trend coefficient.
   - $\gamma = \rho - 1$: Test parameter.
     - $H_0: \gamma = 0$ (Unit root present $\implies$ Non-stationary series).
     - $H_1: \gamma < 0$ (No unit root $\implies$ Stationary series).
   - $\varepsilon_t \sim \mathcal{N}(0, \sigma^2)$: Uncorrelated error term.
   - **Empirical Test Result:**
     - ADF Statistic: $-1.880$
     - Critical Value ($5\%$ significance): $-2.869$
     - $p$-value: $0.3415 > 0.05 \implies$ Fail to reject $H_0$. Confirms non-stationarity driven by upward adoption trend!

2. **Forecasting Models:**
   - **7-Day Moving Average:**
     $$\hat{Y}_{t+1} = \frac{1}{7}\sum_{k=0}^{6} Y_{t-k}$$
   - **Simple Exponential Smoothing (SES):**
     $$\hat{Y}_{t+1} = \alpha Y_t + (1 - \alpha)\hat{Y}_t, \quad \alpha \in [0, 1]$$
   - **Holt-Winters Exponential Smoothing (Triple Additive):**
     $$\text{Level: } \ell_t = \alpha(Y_t - s_{t-m}) + (1-\alpha)(\ell_{t-1} + b_{t-1})$$
     $$\text{Trend: } b_t = \beta(\ell_t - \ell_{t-1}) + (1-\beta)b_{t-1}$$
     $$\text{Season: } s_t = \gamma(Y_t - \ell_{t-1} - b_{t-1}) + (1-\gamma)s_{t-m}$$
     $$\text{Forecast: } \hat{Y}_{t+h} = \ell_t + h b_t + s_{t-m + (h \bmod m)}$$
     - $m = 7$ (Weekly seasonality cycle).
     - $\alpha, \beta, \gamma$: Level, trend, and seasonal smoothing hyperparameters.
     - $h$: Multi-step ahead forecast horizon.

---

### Slide 17: Time-Series Experimental Results & Evaluation
| Time-Series Algorithm | MAE (Daily kWh) | RMSE (Daily kWh) | Forecasting Strengths |
| :--- | :---: | :---: | :--- |
| **7-Day Moving Average** | **581.46** | **892.01** | 🏆 Most robust against day-to-day noise; smooth baselines. |
| **Simple Exponential Smoothing (SES)** | 620.66 | 925.48 | Adapts rapidly to recent level shifts; low latency. |
| **Holt-Winters Additive** | 729.00 | 978.22 | Explicitly extrapolates weekly weekday/weekend seasonality. |

---

### Slide 18: Track 4 — Explainable AI (SHAP) Formulations & Findings
- **Shapley Additive Explanations (SHAP):**
  Based on cooperative game theory, computes the unique, fair contribution of each feature across all possible feature permutations:
  $$\phi_i(x) = \sum_{S \subseteq \mathcal{F} \setminus \{i\}} \frac{|S|!(|\mathcal{F}| - |S| - 1)!}{|\mathcal{F}|!} \left[ f(S \cup \{i\}) - f(S) \right]$$
  - $\phi_i(x)$: Shapley attribution value for feature $i$ on instance $x$.
  - $\mathcal{F}$: Complete set of all available model input features.
  - $S$: Subset of features excluding feature $i$.
  - $|S|$: Cardinality (number of features) in subset $S$.
  - $|\mathcal{F}|$: Total feature count ($28$ features).
  - $f(S)$: Model prediction evaluated only on feature subset $S$.
  - $f(S \cup \{i\}) - f(S)$: Marginal contribution of adding feature $i$ to coalition $S$.
  - $\frac{|S|!(|\mathcal{F}| - |S| - 1)!}{|\mathcal{F}|!}$: Probability weighting factor over all uniform permutations.

- **Global Feature Importance Ranking:**
  1. `Duration_Mins` ($\phi \approx +4.82$): The single dominant physical predictor of total kWh.
  2. `ChargerType_Avg_Demand` ($\phi \approx +2.15$): Fast DC chargers deliver $3\times\text{--}5\times$ power per unit time.
  3. `StartHour` ($\phi \approx +1.48$): High correlation with peak utility commute arrival windows.
  4. `Loc_Avg_Demand` ($\phi \approx +1.21$): Venue infrastructure capacity impact.

---

### Slide 19: Track 4 — Explainable AI (LIME) Formulations & Local Trust
- **Local Interpretable Model-agnostic Explanations (LIME):**
  Approximates any complex non-linear black-box model $f(\mathbf{x})$ locally around an individual customer session $\mathbf{x}$ using an interpretable sparse surrogate $g \in \mathcal{G}$:
  $$\xi(\mathbf{x}) = \arg\min_{g \in \mathcal{G}} \mathcal{L}(f, g, \pi_{\mathbf{x}}) + \Omega(g)$$
  - $\mathbf{x}$: The specific charging transaction being explained.
  - $f(\mathbf{x})$: Black-box model prediction (Random Forest / XGBoost).
  - $g$: Interpretable surrogate explanation model (sparse linear regressor).
  - $\mathcal{L}(f, g, \pi_{\mathbf{x}})$: Local fidelity loss measuring discrepancy between $f$ and $g$ in the perturbation neighborhood:
    $$\mathcal{L}(f, g, \pi_{\mathbf{x}}) = \sum_{\mathbf{z} \in \mathcal{Z}} \pi_{\mathbf{x}}(\mathbf{z}) \left( f(\mathbf{z}) - g(\mathbf{z}) \right)^2$$
  - $\mathbf{z}$: Perturbed synthetic instance drawn around $\mathbf{x}$.
  - $\pi_{\mathbf{x}}(\mathbf{z}) = \exp\left(-\frac{D(\mathbf{x}, \mathbf{z})^2}{\sigma^2}\right)$: Exponential proximity kernel weighting instances by Euclidean distance $D(\mathbf{x}, \mathbf{z})$.
  - $\Omega(g) = \lambda \|\mathbf{w}\|_0$: Complexity penalty enforcing interpretability (limits explanation to Top $K=5$ features).

- **Local Driver Output Example:**
  - Case: Fast DC Charger session at Commercial Center for $65\text{ mins}$.
  - Explanatory weights: $\text{Duration} \ge 60\text{ min } (+3.8\text{ kWh})$, $\text{ChargerType} = \text{DC Fast } (+4.2\text{ kWh})$.

---

### Slide 20: Software Architecture, Pipeline, & Serialization
```
                      ┌───────────────────────────────────────┐
                      │    Raw Data (72,856 Transactions)     │
                      └──────────────────┬────────────────────┘
                                         ▼
                      ┌───────────────────────────────────────┐
                      │     src/preprocessing.py Pipeline     │
                      │  • Deduplication & Outlier Clipping   │
                      │  • Cyclic Hour & Datetime Encoding    │
                      │  • StandardScaler & OneHotEncoder     │
                      └──────────────────┬────────────────────┘
                                         ▼
                      ┌───────────────────────────────────────┐
                      │       src/train.py ML Engine          │
                      │  • 6 Regressors & 7 Classifiers       │
                      │  • Time Series Moving Averages / HW   │
                      │  • SHAP Explainer Initialization      │
                      └──────────────────┬────────────────────┘
                                         ▼
                      ┌───────────────────────────────────────┐
                      │   Serialized Production Artifacts     │
                      │  • models/preprocessor.pkl            │
                      │  • models/regression_model.pkl        │
                      │  • models/classification_model.pkl    │
                      │  • models/timeseries_model.pkl        │
                      │  • models/metrics.json                │
                      └──────────────────┬────────────────────┘
                                         ▼
                      ┌───────────────────────────────────────┐
                      │    app/app.py Production Dashboard    │
                      │  • Executive Overview & EDA           │
                      │  • Demand Predictor & High-Load Alert │
                      │  • Time-Series Forecaster             │
                      │  • Local LIME Interpretability Engine │
                      │  • Dynamic Batch CSV Upload & Scoring │
                      └───────────────────────────────────────┘
```

---

### Slide 21: Streamlit Dashboard UI/UX & Interactive Capabilities
The unified Streamlit application consists of five dedicated operational modes:

1. **Tab 1: 📊 Executive Overview & EDA**
   - 5 KPI metric cards with hover elevation (Total Sessions, Delivered Energy, Avg Load, Avg Duration, High-Demand Rate).
   - Proportional responsive grid layout:
     - Row 1 ($62\% : 38\%$): Location Demand ($14$ venues) vs Charger Technology Donut.
     - Row 2 ($50\% : 50\%$): 24-Hour Diurnal Curve (Dual-Axis) vs Day-of-Week Cadence.
     - Row 3 ($62\% : 38\%$): $2,500$-point Duration-Demand Scatter vs Operator Network Split.
2. **Tab 2: ⚡ Demand Predictor (Regression)**
   - MAE, MSE, RMSE, $R^2$ metric strip ($4$ cards).
   - Interactive parameter sliders (Duration, Location, Charger Type, Start Hour, Day of Week).
   - Real-time gauge chart & 4 preset testing scenarios (e.g. Workplace Commute, Highway Fast Stop).
3. **Tab 3: 🎯 High-Demand Classifier**
   - Accuracy, Precision, Recall, F1, ROC-AUC metric strip ($5$ cards).
   - Dynamic threshold adjustment slider ($\tau \in [0.1, 0.9]$).
   - Probability progress bar and proactive grid overload alert banner.
4. **Tab 4: 📈 Grid Load Forecaster (Time Series)**
   - Historical 366-day energy delivery chart with rolling 7-day average.
   - Future horizon forecasting ($7\text{--}90$ days) with $80\%$ and $95\%$ confidence intervals.
   - ADF stationarity diagnostic report.
5. **Tab 5: 🔍 Explainable AI (LIME & SHAP)**
   - Instant local surrogate explanation for any simulated session.
   - Waterfall chart detailing positive/negative feature forces and global SHAP summary.
6. **Universal CSV Ingestion Engine (Sidebar):**
   - Drag-and-drop batch CSV processing.
   - Automatic schema validation, real-time batch regression & classification scoring, and instant CSV export.

---

### Slide 22: Business Impact, Grid Recommendations, & Future Scope
- **Quantifiable Business Value:**
  - **Grid Stability:** $93.64\%$ recall on high-demand sessions eliminates unexpected feeder overloads.
  - **Revenue Optimization:** CPOs can implement dynamic peak-hour pricing based on predicted duration-demand curves.
  - **Asset Longevity:** Smart charging throttling extends transformer lifecycle by avoiding high-temperature operational regimes.
- **Strategic Deployment Recommendations:**
  1. Integrate model API with Open Charge Point Protocol (OCPP 2.0.1) management systems.
  2. Implement automated demand-response triggers during evening peak commute hours ($17:00\text{--}19:00$).
  3. Expand fast-charging infrastructure at top-demand sites (Commercial centers and transit hubs).
- **Future Enhancements:**
  - Deep Learning architectures (LSTM / Temporal Fusion Transformers) for multi-year forecasting.
  - Integration of external weather, ambient temperature, and electricity wholesale market spot prices.

---

### Slide 23: Complete Mathematical & Notation Glossary
| Symbol | Physical / Mathematical Definition | Domain Context in Project |
| :--- | :--- | :--- |
| $y_i$ | Ground-truth observed continuous scalar value | Actual energy delivered in session $i$ ($\text{kWh}$) |
| $\hat{y}_i$ | Model-predicted continuous scalar value | Predicted energy delivered in session $i$ ($\text{kWh}$) |
| $\bar{y}$ | Arithmetic sample mean: $\frac{1}{n}\sum_{i=1}^n y_i$ | Average demand across all $14,485$ test sessions ($14.28\text{ kWh}$) |
| $n$ | Total count of sample observations | Sample size ($n_{\text{train}}=57,936$, $n_{\text{test}}=14,485$) |
| $\sum_{i=1}^n$ | Summation operator across index $i$ | Aggregating residuals and variances across all test points |
| $|x|$ | Absolute value operator | Linear magnitude of residual in MAE calculation |
| $R^2$ | Coefficient of Determination | Proportion of target variance explained by regression model |
| $\text{SS}_{\text{res}}$ | Residual Sum of Squares: $\sum (y_i - \hat{y}_i)^2$ | Unexplained error variance |
| $\text{SS}_{\text{tot}}$ | Total Sum of Squares: $\sum (y_i - \bar{y})^2$ | Inherent baseline variance of observed data |
| $TP$ | True Positive count | High-demand sessions correctly predicted as High Demand |
| $TN$ | True Negative count | Regular sessions correctly predicted as Regular Demand |
| $FP$ | False Positive count (Type I error) | Regular sessions falsely flagged as High Demand |
| $FN$ | False Negative count (Type II error) | High-demand sessions missed by the model |
| $F_1$ | Harmonic mean of Precision and Recall | Robust classification metric unaffected by class imbalance |
| $\text{AUC}$ | Area Under the ROC Curve | Discrimination threshold-independent classification power |
| $t$ | Discrete hour integer: $t \in \{0, 1, \dots, 23\}$ | Hour of arrival in diurnal cycle |
| $\pi$ | Archimedes' constant $\approx 3.14159265...$ | Trigonometric period conversion for cyclic hour encoding |
| $Y_t$ | Daily aggregated total energy demand | Time-series target on day $t$ ($\text{kWh/day}$) |
| $\Delta$ | First-order differencing operator: $Y_t - Y_{t-1}$ | Removing trend to induce stationarity in time series |
| $\gamma$ | ADF unit root coefficient: $\gamma = \rho - 1$ | Stationarity test parameter ($H_0: \gamma = 0$) |
| $p\text{-value}$ | Probability value under null hypothesis | Probability of observing data if $H_0$ is true ($p=0.3415$) |
| $\alpha, \beta, \gamma$ | Smoothing hyperparameters $\in [0, 1]$ | Holt-Winters level, trend, and seasonal decay constants |
| $\phi_i(x)$ | Shapley attribution value of feature $i$ | Fair marginal contribution of feature $i$ to prediction |
| $\mathcal{F}$ | Set of all input features | Full feature space ($|\mathcal{F}| = 28$) |
| $\xi(x)$ | LIME explanation optimization objective | Best local sparse linear surrogate model |
| $\pi_x(z)$ | Exponential distance kernel: $\exp(-D^2/\sigma^2)$ | Locality weight assigned to perturbed samples in LIME |
| $\Omega(g)$ | Regularization complexity penalty | Limits LIME explanation to Top $K=5$ features |

---

### Slide 24: Conclusion & Q&A
- **Key Milestones Achieved:**
  - $72,421$ validated real-world charging transactions processed.
  - $28$ rich engineered features including cyclic trigonometry and target encodings.
  - $\mathbf{0.7983\ R^2}$ achieved by Random Forest Regressor ($3.38\text{ kWh MAE}$).
  - $\mathbf{0.9615\ \text{ROC-AUC}}$ and $\mathbf{93.64\%\ \text{Recall}}$ achieved by XGBoost Classifier.
  - Transparent model interpretability established via SHAP global importance & LIME local inference.
  - Full production web application deployed and synchronized on GitHub.

**Thank You! Questions & Discussion.**
