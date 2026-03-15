"""
============================================================
  Banking Intelligence Project — Banking Customer Risk & Revenue Intelligence
  Module 4: Time-Series Forecasting — CLV & Revenue
  Models  : ARIMA (statsmodels), Prophet (fallback to ARIMA)
  Purpose : Estimate Customer Lifetime Value and forecast
            portfolio-level revenue for next 12 months.
============================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings, os
warnings.filterwarnings("ignore")

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools   import adfuller

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("ℹ️  Prophet not installed — using ARIMA only.")

os.makedirs("outputs", exist_ok=True)

# ─────────────────────────────────────────────────────────
# 1. BUILD MONTHLY REVENUE SERIES
# ─────────────────────────────────────────────────────────

def build_monthly_revenue(transactions: pd.DataFrame,
                          customers: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate credit-side transactions as a proxy for bank revenue inflows
    (fees, EMI receipts, interest income are approximated here).
    Returns a monthly time-series DataFrame.
    """
    txn = transactions.copy()
    txn["transaction_date"] = pd.to_datetime(txn["transaction_date"])

    # Approximate revenue = inflows + a margin on debit transactions
    txn["revenue_proxy"] = np.where(
        txn["transaction_type"] == "Credit",
        txn["amount"] * 0.015,          # 1.5% interchange / fee
        txn["amount"] * 0.005           # 0.5% float margin on debits
    )

    monthly = (
        txn.groupby(txn["transaction_date"].dt.to_period("M"))["revenue_proxy"]
        .sum()
        .reset_index()
    )
    monthly.columns = ["month", "revenue"]
    monthly["month"] = monthly["month"].dt.to_timestamp()
    monthly = monthly.sort_values("month").reset_index(drop=True)
    return monthly


# ─────────────────────────────────────────────────────────
# 2. ARIMA FORECAST
# ─────────────────────────────────────────────────────────

def run_arima_forecast(series: pd.Series, steps: int = 12):
    """
    Fit ARIMA(2,1,2) on the revenue series and forecast `steps` months ahead.
    Returns forecast values and 95% confidence intervals.
    """
    # Stationarity check
    adf_result = adfuller(series.dropna())
    d = 0 if adf_result[1] < 0.05 else 1

    model  = ARIMA(series, order=(2, d, 2))
    fitted = model.fit()

    forecast_obj = fitted.get_forecast(steps=steps)
    forecast     = forecast_obj.predicted_mean
    conf_int     = forecast_obj.conf_int(alpha=0.05)

    return forecast, conf_int, fitted


# ─────────────────────────────────────────────────────────
# 3. PROPHET FORECAST (optional)
# ─────────────────────────────────────────────────────────

def run_prophet_forecast(monthly: pd.DataFrame, steps: int = 12):
    """
    Fit Facebook Prophet on the monthly revenue series.
    Returns forecasted DataFrame with yhat, yhat_lower, yhat_upper.
    """
    df_prophet = monthly.rename(columns={"month": "ds", "revenue": "y"})
    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
        changepoint_prior_scale=0.05,
    )
    m.fit(df_prophet)
    future   = m.make_future_dataframe(periods=steps, freq="MS")
    forecast = m.predict(future)
    return forecast, m


# ─────────────────────────────────────────────────────────
# 4. CUSTOMER LIFETIME VALUE ESTIMATION
# ─────────────────────────────────────────────────────────

def estimate_clv(customers: pd.DataFrame,
                 risk_scores: pd.DataFrame) -> pd.DataFrame:
    """
    Simplified CLV estimation:
        CLV = (avg_annual_revenue_per_customer × tenure_years) × (1 - default_prob)
    Annual revenue estimated from income, loan interest, and credit card fees.
    """
    df = customers.merge(
        risk_scores[["customer_id", "default_probability"]],
        on="customer_id", how="left"
    )

    # Revenue components
    df["interest_income"]  = df["loan_amount"]  * (df["interest_rate_pct"] / 100)
    df["cc_fee_income"]    = df["credit_limit"]  * 0.02     # 2% annual fee proxy
    df["transaction_fees"] = df["income"]        * 0.003    # 0.3% of income as fees

    df["annual_revenue_est"] = (
        df["interest_income"] + df["cc_fee_income"] + df["transaction_fees"]
    ).fillna(0)

    tenure_years = (df["tenure_months"] / 12).clip(lower=0.5)
    survival     = 1 - df["default_probability"].fillna(0.10)

    df["estimated_clv"] = (df["annual_revenue_est"] * tenure_years * survival).round(2)

    df["clv_segment"] = pd.qcut(
        df["estimated_clv"],
        q=4, labels=["Bronze", "Silver", "Gold", "Platinum"]
    )

    df[["customer_id", "estimated_clv", "clv_segment",
        "annual_revenue_est", "default_probability"]].to_csv(
        "outputs/clv_estimates.csv", index=False
    )
    print("✅  CLV estimates saved → outputs/clv_estimates.csv")
    return df


# ─────────────────────────────────────────────────────────
# 5. VISUALISATIONS
# ─────────────────────────────────────────────────────────

def plot_revenue_forecast(monthly: pd.DataFrame,
                          forecast_vals, conf_int,
                          method: str = "ARIMA"):
    fig, ax = plt.subplots(figsize=(13, 5))

    # Historical
    ax.plot(monthly["month"], monthly["revenue"],
            color="#003366", lw=2, label="Historical Revenue", marker="o", ms=4)

    # Forecast horizon
    last_date   = monthly["month"].max()
    freq_months = pd.date_range(last_date, periods=len(forecast_vals) + 1, freq="MS")[1:]

    if method == "ARIMA":
        ax.plot(freq_months, forecast_vals.values,
                color="#CC0000", lw=2, linestyle="--", label="ARIMA Forecast")
        ax.fill_between(freq_months,
                        conf_int.iloc[:, 0], conf_int.iloc[:, 1],
                        alpha=0.15, color="#CC0000", label="95% CI")
    else:
        future_rows = forecast_vals[forecast_vals["ds"] > last_date]
        ax.plot(future_rows["ds"], future_rows["yhat"],
                color="#CC0000", lw=2, linestyle="--", label="Prophet Forecast")
        ax.fill_between(future_rows["ds"],
                        future_rows["yhat_lower"], future_rows["yhat_upper"],
                        alpha=0.15, color="#CC0000", label="95% CI")

    ax.axvline(last_date, color="gray", linestyle=":", lw=1.5, label="Forecast Start")
    ax.set(title=f"Monthly Revenue Forecast — {method} (12-Month Horizon)",
           xlabel="Month", ylabel="Revenue (₹)")
    ax.legend(fontsize=9)
    ax.set_facecolor("#F9F9F9")
    plt.tight_layout()
    plt.savefig("outputs/04_revenue_forecast.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅  Saved → outputs/04_revenue_forecast.png")


def plot_clv_distribution(clv_df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # CLV histogram
    ax = axes[0]
    clv_df["estimated_clv"].clip(upper=clv_df["estimated_clv"].quantile(0.99)).plot.hist(
        bins=40, ax=ax, color="#003366", edgecolor="white", alpha=0.85
    )
    ax.set(title="Distribution of Estimated Customer Lifetime Value",
           xlabel="CLV (₹)", ylabel="Customer Count")
    ax.set_facecolor("#F9F9F9")

    # CLV segment donut
    ax = axes[1]
    seg_counts = clv_df["clv_segment"].value_counts()
    colors     = ["#CD7F32", "#C0C0C0", "#FFD700", "#4A90D9"]
    wedges, texts, autotexts = ax.pie(
        seg_counts.values, labels=seg_counts.index, colors=colors,
        autopct="%1.1f%%", startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 2}
    )
    ax.set_title("Customer CLV Segmentation")

    plt.tight_layout()
    plt.savefig("outputs/05_clv_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✅  Saved → outputs/05_clv_distribution.png")


# ─────────────────────────────────────────────────────────
# 6. MAIN
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    customers    = pd.read_csv("data/customers.csv")
    transactions = pd.read_csv("data/transactions.csv")
    risk_scores  = pd.read_csv("outputs/risk_scored_customers.csv")

    print("Building monthly revenue series …")
    monthly = build_monthly_revenue(transactions, customers)
    print(f"   {len(monthly)} monthly observations from "
          f"{monthly['month'].min().date()} to {monthly['month'].max().date()}")

    if PROPHET_AVAILABLE:
        print("Running Prophet forecast …")
        prophet_fc, prophet_model = run_prophet_forecast(monthly)
        plot_revenue_forecast(monthly, prophet_fc, None, method="Prophet")
    else:
        print("Running ARIMA forecast …")
        fc_vals, ci, fitted = run_arima_forecast(monthly["revenue"])
        plot_revenue_forecast(monthly, fc_vals, ci, method="ARIMA")

    print("Estimating Customer Lifetime Value …")
    clv_df = estimate_clv(customers, risk_scores)
    plot_clv_distribution(clv_df)

    print("\n✅  Forecasting complete.")
