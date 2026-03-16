"""
============================================================
  Banking Intelligence Project — Banking Customer Risk & Revenue Intelligence
  Module 2: Feature Engineering
  Purpose : Derive behavioural risk indicators from raw tables
            and produce a single modelling-ready feature matrix.
  Features: 18+ behavioral risk features engineered
============================================================
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────
# 1. REPAYMENT FEATURES
# ─────────────────────────────────────────────────────────

def build_repayment_features(repayments: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate 12-month repayment history into per-customer features.

    Features:
    ─────────
    repayment_consistency_score  : % of payments made on time
    total_missed_payments        : absolute count of missed payments
    avg_payment_coverage         : avg( paid / emi )  → 1 = perfect
    late_payment_rate            : proportion of late payments
    payment_delay_frequency      : proportion of payments NOT on-time (late or missed)
    max_consecutive_missed       : longest streak of missed payments
    """
    agg = repayments.copy()
    agg["payment_coverage"] = (agg["paid_amount"] / agg["emi_amount"]).clip(0, 1)
    agg["is_on_time"]       = (agg["payment_status"] == "On-Time").astype(int)
    agg["is_missed"]        = (agg["payment_status"] == "Missed").astype(int)
    agg["is_late"]          = (agg["payment_status"] == "Late").astype(int)
    agg["is_delayed"]       = ((agg["payment_status"] != "On-Time")).astype(int)

    # Max consecutive missed payments per customer
    def max_consecutive_missed(series):
        max_streak = 0
        streak = 0
        for v in series:
            if v == "Missed":
                streak += 1
                max_streak = max(max_streak, streak)
            else:
                streak = 0
        return max_streak

    consecutive = (
        agg.groupby("customer_id")["payment_status"]
        .apply(max_consecutive_missed)
        .rename("max_consecutive_missed")
        .reset_index()
    )

    feat = (
        agg.groupby("customer_id")
           .agg(
               repayment_consistency_score=(  "is_on_time",       "mean"),
               total_missed_payments       =(  "is_missed",        "sum"),
               avg_payment_coverage        =(  "payment_coverage", "mean"),
               late_payment_rate           =(  "is_late",          "mean"),
               payment_delay_frequency     =(  "is_delayed",       "mean"),
           )
           .reset_index()
    )
    feat = feat.merge(consecutive, on="customer_id", how="left")
    return feat


# ─────────────────────────────────────────────────────────
# 2. TRANSACTION FEATURES
# ─────────────────────────────────────────────────────────

def build_transaction_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """
    Derive spending behavioural signals from transaction history.

    Features:
    ─────────
    total_debit_6m        : total spend in last 6 months
    total_credit_6m       : total inflows in last 6 months
    spending_volatility   : std-dev of monthly spend (instability signal)
    avg_monthly_spend     : mean monthly debit amount
    net_cash_flow_6m      : credit – debit over last 6 months
    transaction_frequency : number of transactions per month
    debit_to_credit_ratio : spending vs income proxy ratio
    high_spend_month_flag : 1 if any month's spend > 2x avg spend
    """
    txn = transactions.copy()
    txn["transaction_date"] = pd.to_datetime(txn["transaction_date"])
    cutoff = txn["transaction_date"].max() - pd.DateOffset(months=6)
    txn_6m = txn[txn["transaction_date"] >= cutoff].copy()

    # Monthly spend per customer
    txn_6m["month"] = txn_6m["transaction_date"].dt.to_period("M")

    monthly_spend = (
        txn_6m[txn_6m["transaction_type"] == "Debit"]
        .groupby(["customer_id", "month"])["amount"]
        .sum()
        .reset_index()
    )

    volatility = (
        monthly_spend.groupby("customer_id")["amount"]
        .agg(spending_volatility="std", avg_monthly_spend="mean")
        .fillna(0)
        .reset_index()
    )

    # High spend month flag: any month > 2x the customer's own mean
    def high_spend_flag(amounts):
        if len(amounts) < 2:
            return 0
        avg = amounts.mean()
        return int((amounts > 2 * avg).any())

    high_spend = (
        monthly_spend.groupby("customer_id")["amount"]
        .apply(high_spend_flag)
        .rename("high_spend_month_flag")
        .reset_index()
    )

    total_debit = (
        txn_6m[txn_6m["transaction_type"] == "Debit"]
        .groupby("customer_id")["amount"].sum()
        .rename("total_debit_6m")
        .reset_index()
    )

    total_credit = (
        txn_6m[txn_6m["transaction_type"] == "Credit"]
        .groupby("customer_id")["amount"].sum()
        .rename("total_credit_6m")
        .reset_index()
    )

    txn_freq = (
        txn_6m.groupby("customer_id")
        .size()
        .rename("transaction_count_6m")
        .reset_index()
    )

    feat = (
        volatility
        .merge(total_debit,  on="customer_id", how="left")
        .merge(total_credit, on="customer_id", how="left")
        .merge(txn_freq,     on="customer_id", how="left")
        .merge(high_spend,   on="customer_id", how="left")
        .fillna(0)
    )
    feat["net_cash_flow_6m"]      = feat["total_credit_6m"] - feat["total_debit_6m"]
    feat["transaction_frequency"] = feat["transaction_count_6m"] / 6  # per month avg

    # Debit-to-credit ratio: spending intensity vs inflows
    feat["debit_to_credit_ratio"] = np.where(
        feat["total_credit_6m"] > 0,
        (feat["total_debit_6m"] / feat["total_credit_6m"]).clip(0, 5).round(4),
        0.0
    )
    return feat


# ─────────────────────────────────────────────────────────
# 3. MASTER FEATURE MATRIX
# ─────────────────────────────────────────────────────────

def build_feature_matrix(
    customers:    pd.DataFrame,
    repayments:   pd.DataFrame,
    transactions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Joins all feature tables and derives additional ratio-based features.
    Returns the final modelling DataFrame with 18+ behavioral features.

    Engineered Features:
    ────────────────────
    1.  credit_utilisation_ratio   — CC balance / credit limit
    2.  income_to_loan_ratio       — income / loan amount (repayment capacity)
    3.  emi_to_income_ratio        — monthly EMI / monthly income (DTI proxy)
    4.  interest_burden_score      — interest rate × loan amount
    5.  loan_to_value_ratio        — loan amount / credit limit (leverage)
    6.  credit_age_score           — normalized tenure score (log scale)
    7.  repayment_consistency_score
    8.  total_missed_payments
    9.  avg_payment_coverage
    10. late_payment_rate
    11. payment_delay_frequency    — NEW: proportion of non-on-time payments
    12. max_consecutive_missed     — NEW: longest missed streak
    13. spending_volatility
    14. avg_monthly_spend
    15. total_debit_6m
    16. total_credit_6m
    17. net_cash_flow_6m
    18. transaction_frequency
    19. debit_to_credit_ratio      — NEW: spending intensity vs inflows
    20. high_spend_month_flag      — NEW: 1 if any month had 2x avg spend
    """
    # Base customer features
    df = customers.copy()

    # ── Ratio-based features ──────────────────────────────
    df["credit_utilisation_ratio"] = np.where(
        df["credit_limit"] > 0,
        df["cc_balance"] / df["credit_limit"],
        0.0
    ).round(4)

    df["income_to_loan_ratio"] = np.where(
        df["loan_amount"] > 0,
        df["income"] / df["loan_amount"],
        0.0
    ).round(4)

    df["emi_to_income_ratio"] = np.where(
        df["has_loan"] == 1,
        (df["loan_amount"] / df["loan_tenure_months"]) / (df["income"] / 12),
        0.0
    ).round(4)

    df["interest_burden_score"] = np.where(
        df["has_loan"] == 1,
        (df["interest_rate_pct"] / 100) * df["loan_amount"],
        0.0
    ).round(2)

    # NEW: Loan-to-value ratio (loan exposure relative to credit limit — leverage)
    df["loan_to_value_ratio"] = np.where(
        (df["has_loan"] == 1) & (df["credit_limit"] > 0),
        (df["loan_amount"] / df["credit_limit"]).clip(0, 10).round(4),
        0.0
    )

    # NEW: Credit age score — logarithmic normalization of tenure (longer = better)
    df["credit_age_score"] = np.log1p(df["tenure_months"]).round(4)

    # ── Merge repayment features ──────────────────────────
    repay_feat = build_repayment_features(repayments)
    df = df.merge(repay_feat, on="customer_id", how="left")

    # ── Merge transaction features ────────────────────────
    txn_feat = build_transaction_features(transactions)
    df = df.merge(txn_feat, on="customer_id", how="left")

    # ── Fill NaNs ─────────────────────────────────────────
    fill_zero_cols = [
        "repayment_consistency_score", "total_missed_payments",
        "avg_payment_coverage", "late_payment_rate",
        "payment_delay_frequency", "max_consecutive_missed",
        "spending_volatility", "avg_monthly_spend",
        "total_debit_6m", "total_credit_6m",
        "net_cash_flow_6m", "transaction_frequency",
        "transaction_count_6m", "debit_to_credit_ratio",
        "high_spend_month_flag"
    ]
    df[fill_zero_cols] = df[fill_zero_cols].fillna(0)

    # ── One-hot encode categoricals ───────────────────────
    df = pd.get_dummies(df, columns=["employment_type", "account_type", "city"],
                        drop_first=False)

    print(f"✅  Feature matrix shape: {df.shape}")
    print(f"   Default rate: {df['defaulted'].mean():.2%}")

    behavioral_features = [
        "credit_utilisation_ratio", "income_to_loan_ratio", "emi_to_income_ratio",
        "interest_burden_score", "loan_to_value_ratio", "credit_age_score",
        "repayment_consistency_score", "total_missed_payments", "avg_payment_coverage",
        "late_payment_rate", "payment_delay_frequency", "max_consecutive_missed",
        "spending_volatility", "avg_monthly_spend", "total_debit_6m",
        "total_credit_6m", "net_cash_flow_6m", "transaction_frequency",
        "debit_to_credit_ratio", "high_spend_month_flag"
    ]
    available = [f for f in behavioral_features if f in df.columns]
    print(f"   Behavioral risk features engineered: {len(available)}")
    return df


# ─────────────────────────────────────────────────────────
# 4. MAIN
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading raw data …")
    customers    = pd.read_csv("data/customers.csv")
    repayments   = pd.read_csv("data/repayment_history.csv")
    transactions = pd.read_csv("data/transactions.csv")

    print("Building feature matrix …")
    features = build_feature_matrix(customers, repayments, transactions)
    features.to_csv("data/feature_matrix.csv", index=False)
    print("✅  Saved → data/feature_matrix.csv")
    print(features[["customer_id", "credit_score", "credit_utilisation_ratio",
                     "repayment_consistency_score", "payment_delay_frequency",
                     "loan_to_value_ratio", "credit_age_score",
                     "spending_volatility", "defaulted"]].head(5).to_string())
