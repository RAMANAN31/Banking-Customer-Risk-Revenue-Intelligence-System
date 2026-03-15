import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import os

st.set_page_config(page_title="Customer Risk & Revenue Intelligence Dashboard", layout="wide", page_icon="🏦")

st.title("🏦 Banking Intelligence | Customer Risk & Revenue Dashboard")

# --- Helper Functions ---

@st.cache_resource
def load_models():
    models = {}
    if os.path.exists("models/Logistic_Regression.pkl"):
        models["Logistic Regression"] = joblib.load("models/Logistic_Regression.pkl")
    if os.path.exists("models/Random_Forest.pkl"):
        models["Random Forest"] = joblib.load("models/Random_Forest.pkl")
    return models

models = load_models()

@st.cache_data
def load_data(file_buffer=None):
    if file_buffer:
        df = pd.read_csv(file_buffer)
    else:
        if os.path.exists("data/feature_matrix.csv"):
            df = pd.read_csv("data/feature_matrix.csv")
        else:
            return None
            
    # Merge outputs if available (for the default dataset)
    if not file_buffer:
        if os.path.exists("outputs/risk_scored_customers.csv"):
            risk = pd.read_csv("outputs/risk_scored_customers.csv")
            if "risk_tier" in risk.columns:
                df = df.merge(risk[["customer_id", "default_probability", "risk_tier"]], on="customer_id", how="left")
        
        if os.path.exists("outputs/clv_estimates.csv"):
            clv = pd.read_csv("outputs/clv_estimates.csv")
            if "estimated_clv" in clv.columns:
                df = df.merge(clv[["customer_id", "estimated_clv", "clv_segment"]], on="customer_id", how="left")
                
        if os.path.exists("outputs/customer_segments.csv"):
            seg = pd.read_csv("outputs/customer_segments.csv")
            if "segment" in seg.columns:
                df = df.merge(seg[["customer_id", "segment"]], on="customer_id", how="left")
                
    return df

def get_feature_cols(df):
    exclude = ["customer_id", "defaulted", "default_probability", "risk_tier", "estimated_clv", "clv_segment", "segment"]
    feature_cols = [c for c in df.columns if c not in exclude and df[c].dtype in [np.float64, np.int64, np.uint8, int, float, bool]]
    return feature_cols

def predict_profile(profile_df, model, feature_cols):
    X = profile_df[feature_cols].fillna(0).astype(float)
    probs = model.predict_proba(X)[:, 1]
    return probs[0]


# --- Sidebar ---
with st.sidebar:
    st.header("Data Connection")
    uploaded_file = st.file_uploader("Upload Customer Dataset (CSV)", type="csv")
    df = load_data(uploaded_file)
    
if df is None:
    st.warning("No data available. Please run the pipeline or upload a dataset.")
    st.stop()
    
feature_cols = get_feature_cols(df)

# --- Tabs ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Portfolio Risk Overview", 
    "Risk Segmentation", 
    "Customer Risk Prediction", 
    "What-If Analysis", 
    "Executive Insights"
])

# --- TAB 1: Portfolio Risk Overview ---
with tab1:
    st.header("Portfolio Risk Overview")
    
    total_loan = df["loan_amount"].sum() if "loan_amount" in df.columns else 0
    total_customers = len(df)
    
    if "risk_tier" in df.columns:
        high_risk_df = df[df["risk_tier"] == "High Risk"]
    elif "default_probability" in df.columns:
        high_risk_df = df[df["default_probability"] >= 0.60]
    else:
        high_risk_df = pd.DataFrame()
        
    high_risk_pct = (len(high_risk_df) / total_customers * 100) if total_customers > 0 else 0
    expected_loss = high_risk_df["loan_amount"].sum() if not high_risk_df.empty and "loan_amount" in df.columns else 0
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Loan Exposure", f"₹ {total_loan:,.0f}")
    col2.metric("High-Risk Portfolio %", f"{high_risk_pct:.1f}%")
    col3.metric("Expected Default Losses", f"₹ {expected_loss:,.0f}")
    col4.metric("Total Customers", f"{total_customers:,}")

    if "clv_segment" in df.columns and "estimated_clv" in df.columns:
        st.subheader("Revenue Potential by Customer Segment")
        rev_segment = df.groupby("clv_segment")["estimated_clv"].sum().reset_index()
        fig = px.bar(rev_segment, x="clv_segment", y="estimated_clv", 
                     title="Total Estimated CLV by Segment",
                     color="clv_segment", labels={"clv_segment": "Segment", "estimated_clv": "Total Estimated CLV (₹)"})
        st.plotly_chart(fig, use_container_width=True)

# --- TAB 2: Risk Segmentation ---
with tab2:
    st.header("Risk Segmentation Visualization")
    c1, c2 = st.columns(2)
    
    with c1:
        if "risk_tier" in df.columns:
            st.subheader("Customers Across Risk Levels")
            fig1 = px.pie(df, names="risk_tier", title="Risk Category Distribution", hole=0.4,
                          color="risk_tier", color_discrete_map={"Low Risk": "#28A745", "Medium Risk": "#FF8C00", "High Risk": "#CC0000"})
            st.plotly_chart(fig1, use_container_width=True)
            
    with c2:
        if "segment" in df.columns:
            st.subheader("K-Means Customer Segments")
            # map segment indices to labels as per original segmentation.py
            seg_labels = {0: "💎 Premium", 1: "⚠️ Vulnerable", 2: "📈 Growth", 3: "🔄 Churner"}
            df_plot = df.copy()
            df_plot["Segment_Label"] = df_plot["segment"].map(seg_labels)
            fig2 = px.pie(df_plot, names="Segment_Label", title="Customer Segments (K=4)", hole=0.4)
            st.plotly_chart(fig2, use_container_width=True)
            
    if "credit_utilisation_ratio" in df.columns and "default_probability" in df.columns:
        st.subheader("Credit Utilization vs Default Probability")
        
        color_col = "risk_tier" if "risk_tier" in df.columns else None
        
        # sample to avoid lag
        plot_df = df.sample(min(1000, len(df)))
        fig3 = px.scatter(plot_df, x="credit_utilisation_ratio", y="default_probability", 
                          color=color_col, hover_data=["customer_id", "income"],
                          title="Utilization vs Risk Profile",
                          labels={"credit_utilisation_ratio": "Credit Utilisation", "default_probability": "Default Probability"},
                          color_discrete_map={"Low Risk": "#28A745", "Medium Risk": "#FF8C00", "High Risk": "#CC0000"} if color_col else None)
        st.plotly_chart(fig3, use_container_width=True)

# --- TAB 3: Customer Risk Prediction Panel ---
with tab3:
    st.header("Customer Risk Prediction Panel")
    if not models:
        st.error("No Models Found. Please train models by running the pipeline first.")
    else:
        model_choice = st.selectbox("Select Model", list(models.keys()))
        selected_model = models[model_choice]
        
        cust_id = st.selectbox("Select Customer to Pre-fill", df["customer_id"].head(500).tolist())
        cust_data = df[df["customer_id"] == cust_id].iloc[0].to_dict()
        
        st.subheader("Customer Profile Details")
        
        # We allow adjusting key features here or showing them
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Age", int(cust_data.get("age", 0)))
        c2.metric("Income", f"₹{cust_data.get('income', 0):,.0f}")
        c3.metric("Credit Score", int(cust_data.get("credit_score", 0)))
        c4.metric("Loan Amount", f"₹{cust_data.get('loan_amount', 0):,.0f}")
        
        if st.button("Predict Default Risk & CLV"):
            profile_df = pd.DataFrame([cust_data])
            prob = predict_profile(profile_df, selected_model, feature_cols)
            
            # Recalculate CLV roughly using the formula
            interest_income = profile_df["loan_amount"].iloc[0] * (profile_df["interest_rate_pct"].iloc[0] / 100) if "interest_rate_pct" in profile_df else 0
            cc_fee = profile_df.get("credit_limit", pd.Series([0])).iloc[0] * 0.02
            txn_fee = profile_df.get("income", pd.Series([0])).iloc[0] * 0.003
            annual_rev = interest_income + cc_fee + txn_fee
            tenure = max(0.5, profile_df.get("tenure_months", pd.Series([6])).iloc[0] / 12)
            clv = annual_rev * tenure * (1 - prob)
            
            risk_tier = "High Risk" if prob >= 0.6 else ("Medium Risk" if prob >= 0.3 else "Low Risk")
            
            gc1, gc2, gc3 = st.columns(3)
            gc1.metric("Predicted Default Prob", f"{prob:.1%}")
            gc2.metric("Assigned Risk Tier", risk_tier)
            gc3.metric("Est. Customer Lifetime Value (CLV)", f"₹{clv:,.0f}")


# --- TAB 4: What-If Analysis ---
with tab4:
    st.header("What-If Analysis Tool")
    st.write("Adjust parameters to see how it affects the default probability and CLV dynamically.")
    
    if models:
        wi_model_choice = st.selectbox("Select Model for What-If", list(models.keys()), key="wi_mod")
        wi_model = models[wi_model_choice]
        
        wi_cust_id = st.selectbox("Select Baseline Customer", df["customer_id"].head(50).tolist(), key="wi_cust")
        base_data = df[df["customer_id"] == wi_cust_id].iloc[0].to_dict()
        
        st.subheader("Adjustable Parameters")
        
        c1, c2 = st.columns(2)
        with c1:
            adj_loan = st.slider("Loan Amount (₹)", min_value=0.0, max_value=float(df["loan_amount"].max() * 1.5 if "loan_amount" in df.columns else 1000000), 
                                 value=float(base_data.get("loan_amount", 0)), step=1000.0)
            adj_util = st.slider("Credit Utilization Ratio", min_value=0.0, max_value=1.5, 
                                 value=float(base_data.get("credit_utilisation_ratio", 0.0)), step=0.05)
        with c2:
            adj_income = st.slider("Monthly/Annual Income (₹)", min_value=10000.0, max_value=float((df["income"].max() * 1.5) if "income" in df.columns else 5000000),
                                   value=float(base_data.get("income", 50000)), step=5000.0)
            adj_score = st.slider("Credit Score", min_value=300, max_value=900, 
                                  value=int(base_data.get("credit_score", 650)), step=10)
                                  
        # Create updated profile
        updated_data = base_data.copy()
        updated_data["loan_amount"] = adj_loan
        updated_data["credit_utilisation_ratio"] = adj_util
        updated_data["income"] = adj_income
        updated_data["credit_score"] = adj_score
        
        if updated_data["loan_amount"] > 0 and updated_data["income"] > 0:
            updated_data["income_to_loan_ratio"] = updated_data["income"] / updated_data["loan_amount"]
        else:
            updated_data["income_to_loan_ratio"] = 0
            
        profile_df = pd.DataFrame([updated_data])
        
        prob = predict_profile(profile_df, wi_model, feature_cols)
        
        interest_income = adj_loan * (updated_data.get("interest_rate_pct", 0) / 100)
        cc_fee = updated_data.get("credit_limit", 0) * 0.02
        txn_fee = adj_income * 0.003
        annual_rev = interest_income + cc_fee + txn_fee
        tenure = max(0.5, updated_data.get("tenure_months", 6) / 12)
        clv = annual_rev * tenure * (1 - prob)
        
        risk_tier = "High Risk" if prob >= 0.6 else ("Medium Risk" if prob >= 0.3 else "Low Risk")
        
        st.subheader("Simulated Outcome")
        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("Predicted Default Probability", f"{prob:.1%}", delta=f"{(prob - base_data.get('default_probability', 0)):.1%} from baseline", delta_color="inverse")
        sc2.metric("Risk Tier", risk_tier)
        sc3.metric("Estimated CLV", f"₹{clv:,.0f}", delta=f"{(clv - base_data.get('estimated_clv', clv)):,.0f} from baseline")
                                  

# --- TAB 5: Executive Insights ---
with tab5:
    st.header("Executive Insights & Recommendations")
    
    if "risk_tier" in df.columns and "estimated_clv" in df.columns:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Top High-Risk Customers")
            st.write("These customers present an immediate default threat due to predicted probabilities > 80% with high balance exposures.")
            top_risk = df[df["risk_tier"] == "High Risk"].sort_values("default_probability", ascending=False).head(10)
            disp_cols = ["customer_id", "default_probability", "loan_amount", "credit_score", "credit_utilisation_ratio"]
            disp_cols = [c for c in disp_cols if c in top_risk.columns]
            
            st.dataframe(top_risk[disp_cols].style.format({"default_probability": "{:.1%}", "loan_amount": "₹{:,.0f}"}))
            
        with c2:
            st.subheader("Most Profitable Segments")
            st.write("Focus cross-selling efforts on these cluster segments driven by high consistency and transaction volumes.")
            if "clv_segment" in df.columns:
                clv_summary = df.groupby("clv_segment")["estimated_clv"].agg(["mean", "count", "sum"]).sort_values("mean", ascending=False).reset_index()
                st.dataframe(clv_summary.style.format({"mean": "₹{:,.0f}", "sum": "₹{:,.0f}"}))
                
        st.subheader("Automated Recommendations")
        
        st.info("**Cross-Selling Opportunity**: Target the **Platinum CLV Segment** and **Premium Cluster** with tailored wealth management products. They show low utilization and high transaction volumes.")
        st.warning("**Risk Mitigation Action**: Customers with `credit_utilisation_ratio` > 0.85 and `credit_score` < 600 should be restricted from further credit line increases immediately.")

    else:
        st.write("Please ensure CLV and Risk data are computed to generate executive insights.")
