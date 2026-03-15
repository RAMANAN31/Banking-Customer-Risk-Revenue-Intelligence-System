
# Banking Customer Risk & Revenue Intelligence System

---

## Project Overview

An advanced analytics platform designed for a retail bank's Risk & Analytics team.
The system ingests simulated banking data, engineers behavioural risk features, trains machine learning models to predict loan default probability, estimates Customer Lifetime Value (CLV), segments customers into actionable cohorts, and presents insights through an interactive analytics dashboard for credit risk and portfolio decision-making.

---

## Project Structure

```
hsbc_project/
├── data_generator.py       # Module 1 — Synthetic banking data generation
├── feature_engineering.py  # Module 2 — Behavioural financial feature creation
├── models_ml.py            # Module 3 — ML model training and risk prediction
├── forecasting_clv.py      # Module 4 — Customer Lifetime Value estimation
├── segmentation.py         # Module 5 — Customer segmentation using K-Means
├── sql_queries.sql         # Module 6 — SQL portfolio analytics queries
├── dashboard.py            # Module 7 — Interactive analytics dashboard
├── run_pipeline.py         # Module 8 — End-to-end project pipeline
├── requirements.txt        # Python dependencies
│
├── data/
│   ├── customers.csv
│   ├── repayment_history.csv
│   ├── transactions.csv
│   └── feature_matrix.csv
│
└── outputs/
    ├── risk_scored_customers.csv
    ├── clv_estimates.csv
    ├── customer_segments.csv
    ├── roc_curve.png
    ├── feature_importance.png
    ├── confusion_matrix.png
    ├── clv_distribution.png
    ├── segmentation_clusters.png
    └── risk_vs_clv_analysis.png
```

---

## Quick Start

```bash
# Navigate to project
cd hsbc_project

# Install dependencies
pip install -r requirements.txt

# Run complete analytics pipeline
python run_pipeline.py

# Launch analytics dashboard
python dashboard.py
```

---

## Tech Stack

| Layer                 | Tools                                              |
| --------------------- | -------------------------------------------------- |
| Data Processing       | Python (Pandas, NumPy)                             |
| Data Querying         | SQL (BigQuery / PostgreSQL compatible)             |
| Machine Learning      | Scikit-learn (Logistic Regression, Random Forest)  |
| Forecasting           | Time Series Forecasting (ARIMA / Prophet optional) |
| Customer Segmentation | K-Means Clustering                                 |
| Data Visualization    | Matplotlib, Plotly                                 |
| Dashboard             | Flask + Plotly Interactive Dashboard               |
| Analytics Tools       | Python, SQL, Power BI / Tableau compatible outputs |

---

## Module Descriptions

### Module 1 — Data Generator (`data_generator.py`)

Simulates a retail banking dataset including:

Customers

* Age
* Income
* Employment type
* Credit score
* Loan amount
* Credit utilization
* Account tenure
* Default label

Repayment history

* 12-month EMI payment behavior
* On-time vs delayed payment records

Transaction records

* Debit and credit spending categories
* Monthly spending behaviour

---

### Module 2 — Feature Engineering (`feature_engineering.py`)

Key behavioural financial indicators generated:

| Feature                     | Description                              |
| --------------------------- | ---------------------------------------- |
| credit_utilisation_ratio    | Credit card balance / credit limit       |
| income_to_loan_ratio        | Customer ability to repay loan           |
| emi_to_income_ratio         | Monthly EMI burden                       |
| repayment_consistency_score | Percentage of EMIs paid on time          |
| spending_volatility         | Variability in monthly transaction spend |
| net_cash_flow               | Income inflow minus spending outflow     |

These features capture **customer financial stability and repayment behaviour**.

---

### Module 3 — Machine Learning Models (`models_ml.py`)

Two primary models used for risk prediction:

| Model               | Purpose                                           |
| ------------------- | ------------------------------------------------- |
| Logistic Regression | Interpretable baseline credit risk model          |
| Random Forest       | Non-linear ensemble model for improved prediction |

Model evaluation metrics:

* ROC-AUC
* Precision / Recall
* F1 Score
* Confusion Matrix

Customers are classified into **Low, Medium, and High Risk tiers** based on predicted default probability.

---

### Module 4 — CLV Forecasting (`forecasting_clv.py`)

Customer Lifetime Value estimation using transaction behaviour and repayment patterns.

Simplified CLV model:

```
CLV = expected_revenue × customer_tenure × (1 − default_probability)
```

Used to identify **high value customers for cross-selling opportunities**.

---

### Module 5 — Customer Segmentation (`segmentation.py`)

K-Means clustering used to segment customers into behavioural groups:

| Segment           | Description                                    |
| ----------------- | ---------------------------------------------- |
| Premium Customers | High income, low default risk                  |
| Growth Segment    | Moderate income with high transaction activity |
| Risk Segment      | High credit utilization and repayment issues   |
| Low Engagement    | Low transaction activity customers             |

This helps banks design **targeted marketing and retention strategies**.

---

### Module 6 — SQL Analytics (`sql_queries.sql`)

SQL queries used to perform portfolio-level analysis:

Examples include:

* High risk customer identification
* Loan exposure by employment type
* Credit utilization distribution
* Default probability analysis
* Customer revenue contribution analysis

Queries are compatible with **BigQuery / PostgreSQL environments**.

---

### Module 7 — Analytics Dashboard (`dashboard.py`)

Interactive analytics dashboard developed using **Flask and Plotly**.

Key dashboard components:

* Customer risk distribution chart
* Default probability histogram
* Risk tier segmentation
* CLV distribution visualization
* Portfolio risk vs revenue scatter analysis
* High risk customer monitoring table

The dashboard enables **senior management to quickly assess credit risk exposure and portfolio profitability**.

---

## Sample Results

| Metric              | Value |
| ------------------- | ----- |
| Total customers     | 5,000 |
| Default rate        | ~30%  |
| Best model ROC-AUC  | 0.87  |
| High-risk customers | ~18%  |
| Premium CLV segment | ~25%  |

---

## Business Value

**Credit Risk Monitoring**
Helps banks identify customers with high probability of loan default.

**Customer Segmentation**
Supports targeted marketing and cross-selling strategies.

**Portfolio Performance Analysis**
Provides insights into revenue contribution and risk exposure.

**Decision Support for Credit Approval**
Enables data-driven lending decisions using predictive analytics.

---

## Assumptions & Limitations

* Dataset used is **synthetic for demonstration purposes**.
* CLV model is simplified and does not include product margin calculations.
* Macroeconomic factors such as inflation or unemployment are not included.
* Default labels are generated using simulated financial risk patterns.

---

## Future Enhancements

* Integration with **real banking datasets**
* Deployment using **Streamlit or Dash for faster UI**
* Explainable AI using **SHAP for regulatory compliance**
* Integration with **cloud data warehouses like Google BigQuery**
* Automated model monitoring and retraining pipelines



