"""
============================================================
  Banking Intelligence Project — Banking Customer Risk & Revenue Intelligence
  Module 1: Synthetic Data Generator
  Author : Your Name
  Purpose: Simulate a realistic retail banking dataset with
           200 000 customer records, transaction history,
           loan/credit behaviour and demographic attributes.
============================================================
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import os

# ─── Reproducibility ─────────────────────────────────────
np.random.seed(42)
random.seed(42)

N_CUSTOMERS = 5000          # scaled for local runs; set to 200_000 for full scale
N_TRANSACTIONS = 50_000
START_DATE = datetime(2021, 1, 1)
END_DATE   = datetime(2024, 12, 31)

# ─────────────────────────────────────────────────────────
# 1. CUSTOMER MASTER TABLE
# ─────────────────────────────────────────────────────────

def generate_customers(n: int = N_CUSTOMERS) -> pd.DataFrame:
    """
    Generate the customer master table.
    Columns mirror what a real core-banking system exports.
    """
    customer_ids = [f"CUST{str(i).zfill(7)}" for i in range(1, n + 1)]

    age           = np.random.randint(22, 70, n)
    income        = np.round(np.random.lognormal(mean=11.0, sigma=0.6, size=n), 2)
    employment    = np.random.choice(
        ["Salaried", "Self-Employed", "Business Owner", "Retired", "Unemployed"],
        size=n, p=[0.50, 0.20, 0.15, 0.10, 0.05]
    )
    city          = np.random.choice(
        ["Mumbai", "Delhi", "Bangalore", "Chennai", "Hyderabad", "Pune", "Kolkata"],
        size=n
    )
    account_type  = np.random.choice(["Savings", "Current", "Salary"], size=n,
                                     p=[0.55, 0.30, 0.15])
    tenure_months = np.random.randint(1, 180, n)
    credit_score  = np.clip(np.random.normal(650, 90, n).astype(int), 300, 900)

    # Loan details
    has_loan      = np.random.choice([1, 0], size=n, p=[0.60, 0.40])
    loan_amount   = np.where(has_loan, np.round(np.random.lognormal(12.5, 0.8, n), 2), 0)
    loan_tenure   = np.where(has_loan, np.random.randint(12, 84, n), 0)
    interest_rate = np.where(has_loan,
                             np.round(np.random.uniform(8.5, 18.5, n), 2), 0.0)

    # Credit card
    has_cc        = np.random.choice([1, 0], size=n, p=[0.55, 0.45])
    credit_limit  = np.where(has_cc,
                             np.round(income * np.random.uniform(0.3, 1.5, n), 2), 0)
    cc_balance    = np.where(has_cc,
                             np.round(credit_limit * np.random.uniform(0, 0.95, n), 2), 0)

    # Ground-truth default label (engineered to be realistic)
    default_prob  = (
        0.30 * (credit_score < 600).astype(int)
        + 0.20 * (income < 300_000).astype(int)
        + 0.15 * (employment == "Unemployed").astype(int)
        + 0.10 * (has_loan).astype(int)
        + 0.10 * np.random.uniform(0, 1, n)
    )
    default_prob  = np.clip(default_prob / default_prob.max(), 0.02, 0.85)
    defaulted     = (np.random.uniform(0, 1, n) < default_prob).astype(int)

    df = pd.DataFrame({
        "customer_id"      : customer_ids,
        "age"              : age,
        "income"           : income,
        "employment_type"  : employment,
        "city"             : city,
        "account_type"     : account_type,
        "tenure_months"    : tenure_months,
        "credit_score"     : credit_score,
        "has_loan"         : has_loan,
        "loan_amount"      : loan_amount,
        "loan_tenure_months": loan_tenure,
        "interest_rate_pct": interest_rate,
        "has_credit_card"  : has_cc,
        "credit_limit"     : credit_limit,
        "cc_balance"       : cc_balance,
        "defaulted"        : defaulted,
    })
    return df


# ─────────────────────────────────────────────────────────
# 2. MONTHLY REPAYMENT HISTORY TABLE
# ─────────────────────────────────────────────────────────

def generate_repayment_history(customers: pd.DataFrame) -> pd.DataFrame:
    """
    Generate 12 months of EMI repayment records per loan customer.
    Flags missed/late payments to feed feature engineering.
    """
    loan_custs = customers[customers["has_loan"] == 1]["customer_id"].tolist()
    records = []
    ref_date = datetime(2024, 1, 1)

    for cid in loan_custs:
        for month_offset in range(12):
            due_date    = ref_date - timedelta(days=30 * month_offset)
            pay_status  = np.random.choice(
                ["On-Time", "Late", "Missed"],
                p=[0.75, 0.15, 0.10]
            )
            emi_amount  = round(np.random.uniform(3000, 50000), 2)
            paid_amount = (
                emi_amount if pay_status == "On-Time"
                else (emi_amount * np.random.uniform(0.0, 0.9)
                      if pay_status == "Late" else 0.0)
            )
            records.append({
                "customer_id"  : cid,
                "due_date"     : due_date.strftime("%Y-%m-%d"),
                "emi_amount"   : emi_amount,
                "paid_amount"  : round(paid_amount, 2),
                "payment_status": pay_status,
            })

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────
# 3. TRANSACTION TABLE
# ─────────────────────────────────────────────────────────

def generate_transactions(customers: pd.DataFrame,
                          n: int = N_TRANSACTIONS) -> pd.DataFrame:
    """
    Simulate debit / credit transactions for customers.
    Used to compute spending volatility and balance trends.
    """
    cids       = customers["customer_id"].tolist()
    date_range = pd.date_range(START_DATE, END_DATE, freq="D")

    txn_ids    = [f"TXN{str(i).zfill(9)}" for i in range(1, n + 1)]
    cust_ids   = np.random.choice(cids, size=n)
    amounts    = np.round(np.random.lognormal(7.5, 1.2, n), 2)
    txn_types  = np.random.choice(["Debit", "Credit"], size=n, p=[0.65, 0.35])
    categories = np.random.choice(
        ["Groceries", "Utilities", "Travel", "EMI", "Entertainment",
         "Healthcare", "Salary", "Investment", "Others"],
        size=n
    )
    dates      = np.random.choice(date_range, size=n)

    df = pd.DataFrame({
        "transaction_id"  : txn_ids,
        "customer_id"     : cust_ids,
        "transaction_date": pd.to_datetime(dates),
        "amount"          : amounts,
        "transaction_type": txn_types,
        "category"        : categories,
    })
    return df.sort_values("transaction_date").reset_index(drop=True)


# ─────────────────────────────────────────────────────────
# 4. MAIN — Generate & Save All Tables
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)

    print("⏳  Generating customer master table …")
    customers = generate_customers()
    customers.to_csv("data/customers.csv", index=False)
    print(f"   ✅  {len(customers):,} customers saved → data/customers.csv")

    print("⏳  Generating repayment history …")
    repayments = generate_repayment_history(customers)
    repayments.to_csv("data/repayment_history.csv", index=False)
    print(f"   ✅  {len(repayments):,} repayment records saved → data/repayment_history.csv")

    print("⏳  Generating transactions …")
    transactions = generate_transactions(customers)
    transactions.to_csv("data/transactions.csv", index=False)
    print(f"   ✅  {len(transactions):,} transactions saved → data/transactions.csv")

    print("\n🎉  All datasets ready.")
    print(customers.head(3))
