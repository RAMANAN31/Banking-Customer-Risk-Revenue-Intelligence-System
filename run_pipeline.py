"""
============================================================
  Banking Intelligence Project: Banking Customer Risk & Revenue Intelligence
  Module 8: Master Pipeline Runner
  Usage   : python run_pipeline.py
  Purpose : Executes all modules end-to-end in correct order.
  Enhanced: SHAP computation, metrics JSON, forecast CSV, integrity check
============================================================
"""
import sys, time, os
# Force UTF-8 output for cross-platform compatibility
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

def step(msg):
    print(f"\n{'='*60}")
    print(f"  >>  {msg}")
    print(f"{'='*60}")

def run():
    start = time.time()
    print("\n" + "="*60)
    print("  BANKING RISK & REVENUE INTELLIGENCE PIPELINE")
    print("  End-to-End Execution")
    print("="*60)

    # ── Step 1: Data Generation ──────────────────────────
    step("STEP 1/6 - Synthetic Data Generation")
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
    step("STEP 2/6 - Feature Engineering (20 Behavioral Features)")
    from feature_engineering import build_feature_matrix
    features = build_feature_matrix(customers, repayments, transactions)
    features.to_csv("data/feature_matrix.csv", index=False)

    # ── Step 3: ML Model Training ─────────────────────────
    step("STEP 3/6 - Machine Learning: Default Prediction")
    from models_ml import load_data, train_and_evaluate, \
        plot_roc_curves, plot_feature_importance, \
        plot_confusion_matrix, assign_risk_tiers, compute_shap_values

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

    # ── Step 3b: SHAP Explainability ──────────────────────
    step("STEP 3b/6 - SHAP Explainability")
    compute_shap_values(results, best_name, X_train, X_test, feature_cols)

    # ── Step 4: CLV Forecasting ───────────────────────────
    step("STEP 4/6 - CLV Estimation & Revenue Forecasting (ARIMA 12-Month)")
    from forecasting_clv import (build_monthly_revenue, run_arima_forecast,
                                  plot_revenue_forecast, estimate_clv,
                                  plot_clv_distribution, save_forecast_to_csv,
                                  PROPHET_AVAILABLE)
    monthly     = build_monthly_revenue(transactions, customers)
    risk_scores = pd.read_csv("outputs/risk_scored_customers.csv")

    if PROPHET_AVAILABLE:
        from forecasting_clv import run_prophet_forecast
        pfc, _ = run_prophet_forecast(monthly)
        plot_revenue_forecast(monthly, pfc, None, method="Prophet")
        save_forecast_to_csv(monthly, pfc, None, method="Prophet")
    else:
        fc, ci, _ = run_arima_forecast(monthly["revenue"])
        plot_revenue_forecast(monthly, fc, ci, method="ARIMA")
        save_forecast_to_csv(monthly, fc, ci, method="ARIMA")

    clv_df = estimate_clv(customers, risk_scores)
    plot_clv_distribution(clv_df)

    # ── Step 5: Clustering ────────────────────────────────
    step("STEP 5/6 - Customer Segmentation: K-Means (4 Clusters)")
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

    # ── Step 6: System Integrity Check ───────────────────
    step("STEP 6/6 - System Integrity Check")
    required_files = [
        "data/customers.csv",
        "data/repayment_history.csv",
        "data/transactions.csv",
        "data/feature_matrix.csv",
        "outputs/risk_scored_customers.csv",
        "outputs/clv_estimates.csv",
        "outputs/customer_segments.csv",
        "outputs/monthly_revenue.csv",
        "outputs/revenue_forecast.csv",
        "outputs/model_metrics.json",
        "outputs/feature_importance.json",
    ]

    all_ok = True
    for fpath in required_files:
        exists = os.path.exists(fpath)
        status = "OK " if exists else "MISSING"
        if not exists:
            all_ok = False
        size = os.path.getsize(fpath) / 1024 if exists else 0
        print(f"   [{status}]  {fpath:<45} {size:>6.1f} KB")

    # Validate key metrics
    import json
    if os.path.exists("outputs/model_metrics.json"):
        with open("outputs/model_metrics.json") as f:
            metrics = json.load(f)
        best = metrics["best_model"]
        auc  = metrics["best_auc"]
        target_met = "TARGET MET (>0.88)" if auc > 0.88 else "Below 0.88 target"
        print(f"\n   Best Model : {best}")
        print(f"   AUC-ROC    : {auc:.4f}  [{target_met}]")

    cust_df = pd.read_csv("data/customers.csv")
    seg_df  = pd.read_csv("outputs/customer_segments.csv")
    clv_df2 = pd.read_csv("outputs/clv_estimates.csv")
    risk_df = pd.read_csv("outputs/risk_scored_customers.csv")
    print(f"\n   Customers         : {len(cust_df):,}")
    print(f"   With risk scores  : {len(risk_df):,}")
    print(f"   With CLV estimates: {len(clv_df2):,}")
    print(f"   With segments     : {len(seg_df):,}")

    # ── Summary ───────────────────────────────────────────
    elapsed = time.time() - start
    print(f"\n{'='*60}")
    if all_ok:
        print(f"  PIPELINE COMPLETE - ALL INTEGRITY CHECKS PASSED  ({elapsed:.1f}s)")
    else:
        print(f"  PIPELINE COMPLETE WITH WARNINGS  ({elapsed:.1f}s)")
    print(f"{'='*60}")

    output_files = [f for f in os.listdir("outputs") if f.endswith((".csv", ".png", ".json"))]
    print(f"\n  Output files ({len(output_files)}):")
    for f in sorted(output_files):
        size = os.path.getsize(f"outputs/{f}")
        print(f"     {f:<45} {size/1024:>6.1f} KB")

    print("\n  Launch Dashboard:  streamlit run streamlit_app.py")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    run()
