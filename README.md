# Banking Customer Risk & Revenue Intelligence System
### 

---

## Project Overview

An advanced analytics platform designed for a retail bank's Risk & Analytics team.
The system ingests simulated banking data, engineers behavioural risk features,
trains interpretable ML models to predict loan defaults, estimates Customer Lifetime
Value (CLV) via time-series forecasting, segments customers into actionable cohorts,
and presents all insights through an interactive executive dashboard.

---

## Project Structure

```
hsbc_project/
├── data_generator.py       # Module 1 — Synthetic data creation (customers, repayments, txns)
├── feature_engineering.py  # Module 2 — Behavioural feature derivation
├── models_ml.py            # Module 3 — ML model training, evaluation, risk scoring
├── forecasting_clv.py      # Module 4 — ARIMA/Prophet CLV & revenue forecasting
├── segmentation.py         # Module 5 — K-Means customer clustering
├── sql_queries.sql         # Module 6 — Portfolio analytics SQL (BigQuery/PostgreSQL)
├── dashboard.py            # Module 7 — Flask + Plotly interactive dashboard
├── run_pipeline.py         # Module 8 — End-to-end master runner
├── requirements.txt        # Python dependencies
│
├── data/
│   ├── customers.csv           → 5,000 customer records
│   ├── repayment_history.csv   → 12-month EMI repayment logs
│   ├── transactions.csv        → 50,000 debit/credit transactions
│   └── feature_matrix.csv      → Engineered modelling features
│
└── outputs/
    ├── risk_scored_customers.csv   → Default probability + risk tier per customer
    ├── clv_estimates.csv           → CLV score + CLV segment per customer
    ├── customer_segments.csv       → K-Means cluster assignment
    ├── 01_roc_pr_curves.png        → ROC and Precision-Recall curves
    ├── 02_feature_importance.png   → Top risk predictor features
    ├── 03_confusion_matrix.png     → Model confusion matrix
    ├── 04_revenue_forecast.png     → ARIMA/Prophet revenue forecast
    ├── 05_clv_distribution.png     → CLV distribution and segment pie
    ├── 06_elbow_silhouette.png     → K-Means optimisation plots
    ├── 07_pca_clusters.png         → PCA 2D cluster visualisation
    ├── 08_segment_radar.png        → Radar chart — segment profiles
    └── 09_risk_revenue_scatter.png → Risk vs CLV quadrant analysis
```

---

## ⚡ Quick Start

```bash
# 1. Clone / download project
cd Banking_project

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run entire pipeline (generates data, trains models, creates all outputs)
python run_pipeline.py

# 4. Launch the interactive dashboard
python dashboard.py
# → Open http://localhost:5000
```

---

## 🔧 Tech Stack

| Layer               | Tools                                           |
|---------------------|-------------------------------------------------|
| Data Processing     | Python (Pandas, NumPy), SQL (PostgreSQL/BigQuery) |
| Machine Learning    | Scikit-learn, XGBoost, LightGBM, SHAP           |
| Forecasting         | Statsmodels (ARIMA), Prophet (optional)         |
| Clustering          | Scikit-learn KMeans, PCA                        |
| Visualisation       | Matplotlib, Seaborn, Plotly                     |
| Dashboard           | Flask, Plotly.js                                |
| Model Persistence   | Joblib (.pkl files)                             |

---

##  Module Descriptions

### Module 1 — Data Generator (`data_generator.py`)
Simulates a realistic retail banking dataset:
- **Customers**: Age, income, employment type, city, account type, tenure,
  credit score, loan details, credit card data, and a ground-truth default label
  engineered using weighted risk factors (Basel III aligned).
- **Repayment History**: 12 months of EMI payment records with On-Time / Late / Missed
  status, enabling repayment consistency analysis.
- **Transactions**: 50,000 debit/credit records across 9 spending categories,
  used to compute behavioural spending features.

### Module 2 — Feature Engineering (`feature_engineering.py`)
Derives the following behavioural risk indicators:
| Feature                      | Business Logic                                |
|------------------------------|-----------------------------------------------|
| `credit_utilisation_ratio`   | CC balance / credit limit                     |
| `income_to_loan_ratio`       | Debt serviceability indicator                 |
| `emi_to_income_ratio`        | Monthly EMI burden as % of income             |
| `repayment_consistency_score`| % of EMIs paid on time (last 12 months)       |
| `total_missed_payments`      | Absolute count of payment defaults            |
| `spending_volatility`        | Std-dev of monthly spend (instability signal) |
| `net_cash_flow_6m`           | Credit inflows minus debit outflows (6 months)|
| `avg_payment_coverage`       | Avg(paid / emi) → 1.0 = perfect               |

### Module 3 — ML Models (`models_ml.py`)
Trains three models on an 80/20 stratified split with 5-fold cross-validation:

| Model                | Description                              |
|----------------------|------------------------------------------|
| Logistic Regression  | Baseline interpretable model (L2, scaled)|
| Random Forest        | Ensemble, 200 trees, balanced class weight|
| Gradient Boosting    | GBM, 200 estimators, learning rate 0.05  |

**Metrics**: AUC-ROC, F1, Average Precision, Confusion Matrix  
**Threshold tuning**: 0.40 (adjusted for class imbalance)  
**Risk tiering**: Low (0–30%), Medium (30–60%), High (60–100%)

### Module 4 — Forecasting (`forecasting_clv.py`)
- **ARIMA(2,d,2)**: Stationarity-tested revenue forecasting, 12-month horizon
  with 95% confidence intervals.
- **Prophet** (optional): Multiplicative seasonality model with yearly cycles.
- **CLV Estimation**:
  ```
  CLV = (interest income + CC fees + transaction fees) × tenure_years × (1 − default_prob)
  ```
  Customers segmented into Bronze / Silver / Gold / Platinum tiers.

### Module 5 — Segmentation (`segmentation.py`)
K-Means clustering (K=4) with PCA-based visualisation:
| Segment                          | Profile                                    |
|----------------------------------|--------------------------------------------|
| 💎 Premium — High Value, Low Risk | High credit score, high CLV, low default   |
| ⚠️  Vulnerable — Low Income, High Risk | Low income, high missed payments         |
| 📈 Growth Potential               | Younger, credit-building, moderate spend   |
| 🔄 Churner Watch                  | Declining engagement, low transaction freq |

### Module 6 — SQL Queries (`sql_queries.sql`)
10 production-grade analytical queries covering:
- Portfolio default breakdown by employment type
- High-risk customer identification
- Monthly EMI collection efficiency
- Credit utilisation risk banding
- City-level NPA concentration
- Rolling 3-month default trend (window function)
- CLV distribution and revenue contribution by tier
- Cohort retention analysis by tenure

### Module 7 — Dashboard (`dashboard.py`)
Flask REST API + Plotly.js frontend with:
- 6 KPI cards (total customers, high-risk count, default rate, avg CLV, credit score, platinum count)
- Default probability histogram
- Risk tier donut chart
- CLV segment bar chart
- City NPA exposure chart
- Credit score box plot by employment type
- Top 20 high-risk customer table with action badges

---

## Sample Results (5,000 customer run)

| Metric                    | Value            |
|---------------------------|------------------|
| Dataset size              | 5,000 customers  |
| Default rate              | ~31%             |
| Best model AUC-ROC        | 0.88–0.92        |
| Best model F1-Score       | 0.84–0.87        |
| Silhouette score (K=4)    | 0.35–0.45        |
| High-risk customers (>60%)| ~18% of portfolio|
| Platinum CLV customers    | ~25% of portfolio|

---

##  Business Value Delivered

1. **Credit Approval Support**: Risk tier scores enable automated / semi-automated
   loan approval workflows, reducing manual review time.
2. **Portfolio Stress Testing**: Scenario analysis-ready model outputs align with
   Basel III ICAAP requirements.
3. **Targeted Campaigns**: CLV + segment labels enable personalised product
   recommendations (upsell / cross-sell) to Platinum/Gold customers.
4. **Proactive NPA Prevention**: High-risk customers identified 3–6 months before
   expected default, enabling early intervention.
5. **Revenue Forecasting**: 12-month portfolio revenue projections support FP&A
   planning cycles.

---

##  Assumptions & Limitations

- **Data**: All data is synthetic. Income, loan amounts, and repayment behaviour
  are simulated using log-normal and Bernoulli distributions calibrated to
  approximate real-world distributions. Not representative of any real HSBC portfolio.
- **CLV Model**: Simplified 3-component revenue model. A production system would
  use product-level margin data and discount rates.
- **ARIMA**: Assumes linearity and stationarity after differencing. Non-linear
  macroeconomic shocks (e.g., COVID-style events) are not captured.
- **Labelling**: Ground-truth default labels are synthetically engineered —
  not from actual credit bureau data.
- **BigQuery**: SQL queries are written in BigQuery/PostgreSQL dialect. Minor
  syntax adjustments may be required for other engines.

---

##  Future Enhancements

- **SHAP Waterfall Plots**: Individual loan-level explainability for compliance reporting
- **Real BigQuery Integration**: Replace CSV pipeline with GCP BigQuery connector
- **Tableau/Power BI Connector**: Export outputs as `.hyper` or `.pbix` data sources
- **MLflow Experiment Tracking**: Log model parameters, metrics, and artefacts
- **Streamlit Version**: Convert dashboard to Streamlit for faster prototyping
- **Feature Store**: Centralise feature computation with Great Expectations validation

---
