"""
============================================================
  Banking Intelligence Project — Banking Customer Risk & Revenue Intelligence
  Module 8: Master Pipeline Runner
  Usage   : python run_pipeline.py
  Purpose : Executes all modules end-to-end in correct order.
============================================================
"""

import sys, time, os

def step(msg):
    print(f"\n{'='*60}")
    print(f"  ▶  {msg}")
    print(f"{'='*60}")

def run():
    start = time.time()
    print("\n" + "█"*60)
    print("  BANKING RISK & REVENUE INTELLIGENCE PIPELINE")
    print("  End-to-End Execution")
    print("█"*60)

    # ── Step 1: Data Generation ──────────────────────────
    step("STEP 1/5 — Synthetic Data Generation")
    from data_generator import generate_customers, generate_repayment_history, generate_transactions
    import pandas as pd, os
    os.makedirs("data", exist_ok=True)

    customers    = generate_customers()
    repayments   = generate_repayment_history(customers)
    transactions = generate_transactions(customers)

    customers.to_csv("data/customers.csv", index=False)
    repayments.to_csv("data/repayment_history.csv", index=False)
    transactions.to_csv("data/transactions.csv", index=False)
    print(f"   Customers:    {len(customers):,}")
    print(f"   Repayments:   {len(repayments):,}")
    print(f"   Transactions: {len(transactions):,}")

    # ── Step 2: Feature Engineering ──────────────────────
    step("STEP 2/5 — Feature Engineering")
    from feature_engineering import build_feature_matrix
    features = build_feature_matrix(customers, repayments, transactions)
    features.to_csv("data/feature_matrix.csv", index=False)

    # ── Step 3: ML Model Training ─────────────────────────
    step("STEP 3/5 — Machine Learning — Default Prediction")
    from models_ml import load_data, train_and_evaluate, \
        plot_roc_curves, plot_feature_importance, \
        plot_confusion_matrix, assign_risk_tiers

    X, y, feature_cols = load_data()
    results, X_train, X_test, y_train, y_test, best_name, feature_cols = \
        train_and_evaluate(X, y, feature_cols)

    plot_roc_curves(results, y_test)
    plot_feature_importance(results, best_name, feature_cols)
    plot_confusion_matrix(results, best_name, y_test)

    df_orig  = pd.read_csv("data/feature_matrix.csv")
    bool_c   = df_orig.select_dtypes("bool").columns
    df_orig[bool_c] = df_orig[bool_c].astype(int)
    assign_risk_tiers(df_orig, results[best_name]["model"], feature_cols)

    # ── Step 4: CLV Forecasting ───────────────────────────
    step("STEP 4/5 — CLV Estimation & Revenue Forecasting")
    from forecasting_clv import (build_monthly_revenue, run_arima_forecast,
                                  plot_revenue_forecast, estimate_clv,
                                  plot_clv_distribution, PROPHET_AVAILABLE)
    monthly    = build_monthly_revenue(transactions, customers)
    risk_scores = pd.read_csv("outputs/risk_scored_customers.csv")

    if PROPHET_AVAILABLE:
        from forecasting_clv import run_prophet_forecast
        pfc, _ = run_prophet_forecast(monthly)
        plot_revenue_forecast(monthly, pfc, None, method="Prophet")
    else:
        fc, ci, _ = run_arima_forecast(monthly["revenue"])
        plot_revenue_forecast(monthly, fc, ci, method="ARIMA")

    clv_df = estimate_clv(customers, risk_scores)
    plot_clv_distribution(clv_df)

    # ── Step 5: Clustering ────────────────────────────────
    step("STEP 5/5 — Customer Segmentation — K-Means")
    from segmentation import fit_kmeans, profile_segments, \
        plot_pca_clusters, plot_segment_radar, plot_risk_revenue_scatter

    feat_df   = pd.read_csv("data/feature_matrix.csv")
    bool_c    = feat_df.select_dtypes("bool").columns
    feat_df[bool_c] = feat_df[bool_c].astype(int)

    clv_out   = pd.read_csv("outputs/clv_estimates.csv")[["customer_id", "estimated_clv"]]
    risk_out  = pd.read_csv("outputs/risk_scored_customers.csv")[["customer_id", "default_probability"]]
    feat_df   = feat_df.merge(clv_out, on="customer_id", how="left")
    feat_df   = feat_df.merge(risk_out, on="customer_id", how="left")

    df_seg, _, _, X_scaled, seg_features = fit_kmeans(feat_df, n_clusters=4)
    profile_segments(df_seg, seg_features)
    plot_pca_clusters(X_scaled, df_seg["segment"].values)
    plot_segment_radar(df_seg, seg_features)
    plot_risk_revenue_scatter(df_seg)
    df_seg[["customer_id", "segment"]].to_csv("outputs/customer_segments.csv", index=False)

    # ── Summary ───────────────────────────────────────────
    elapsed = time.time() - start
    print(f"\n{'█'*60}")
    print(f"  ✅  PIPELINE COMPLETE  ({elapsed:.1f}s)")
    print(f"{'█'*60}")

    output_files = [f for f in os.listdir("outputs") if f.endswith((".csv", ".png"))]
    print(f"\n  📁  Output files ({len(output_files)}):")
    for f in sorted(output_files):
        size = os.path.getsize(f"outputs/{f}")
        print(f"     {f:<45} {size/1024:>6.1f} KB")

    print("\n  🚀  Launch dashboard:  python dashboard.py")
    print(f"{'─'*60}\n")


if __name__ == "__main__":
    run()
