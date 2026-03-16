"""
============================================================
  Banking Intelligence Project — Credit Risk & Revenue Intelligence
  Streamlit Executive Dashboard — McKinsey-Style Design
  Tabs: Executive Overview | Risk Analytics | Revenue & CLV
        SQL Insights | Customer Intelligence | What-If Analysis
============================================================
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib, os, json, warnings
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
warnings.filterwarnings("ignore")

# ── Page Config ───────────────────────────────────────────
st.set_page_config(
    page_title="Credit Risk & Revenue Intelligence",
    layout="wide",
    page_icon="🏦",
    initial_sidebar_state="expanded"
)

# ── Global Dark-Navy Theme ────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }

  .stApp { background: #0A1628; color: #E8EDF5; }

  section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D1F3C 0%, #0A1628 100%) !important;
    border-right: 1px solid #1E3A5A;
  }

  .block-container { padding: 1.5rem 2rem; }

  /* KPI Card */
  .kpi-card {
    background: linear-gradient(135deg, #0D1F3C 0%, #122B4D 100%);
    border: 1px solid #1E3A5A;
    border-radius: 12px;
    padding: 18px 20px;
    margin-bottom: 4px;
    position: relative;
    overflow: hidden;
  }
  .kpi-card::before {
    content: '';
    position: absolute; top: 0; left: 0;
    width: 4px; height: 100%;
    background: var(--accent, #1565C0);
  }
  .kpi-card.red  { --accent: #CC0000; }
  .kpi-card.orange { --accent: #E65100; }
  .kpi-card.green  { --accent: #1B5E20; }
  .kpi-card.gold   { --accent: #F9A825; }
  .kpi-label { font-size: 11px; font-weight: 500; color: #90A4AE;
               text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px; }
  .kpi-value { font-size: 28px; font-weight: 700; color: #E3F2FD; margin-bottom: 2px; }
  .kpi-sub   { font-size: 11px; color: #546E7A; }

  /* Section Header */
  .section-title {
    font-size: 15px; font-weight: 600; color: #90CAF9;
    text-transform: uppercase; letter-spacing: 1.5px;
    padding-bottom: 8px; margin-bottom: 16px;
    border-bottom: 1px solid #1E3A5A;
  }

  /* Chart card */
  .chart-card {
    background: #0D1F3C;
    border: 1px solid #1E3A5A;
    border-radius: 12px;
    padding: 16px;
  }

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] { background: #0D1F3C; border-radius: 8px; padding: 4px; }
  .stTabs [data-baseweb="tab"] { color: #90A4AE !important; font-weight: 500; }
  .stTabs [aria-selected="true"] { background: #1565C0 !important; color: white !important; border-radius: 6px; }

  /* Metrics override */
  [data-testid="stMetricValue"] { color: #E3F2FD !important; font-size: 22px !important; }
  [data-testid="stMetricLabel"] { color: #90A4AE !important; }

  /* Dataframe */
  .dataframe { background: #0D1F3C !important; color: #E8EDF5 !important; }

  /* Header banner */
  .dash-header {
    background: linear-gradient(135deg, #0D2137 0%, #1a237e 50%, #0D1F3C 100%);
    border: 1px solid #1E3A5A;
    border-radius: 14px;
    padding: 24px 32px;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
  .dash-title { font-size: 20px; font-weight: 700; color: #E3F2FD; }
  .dash-subtitle { font-size: 12px; color: #78909C; margin-top: 4px; }
  .badge-live {
    background: rgba(21,101,192,0.2);
    border: 1px solid #1565C0;
    color: #64B5F6;
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 11px;
    font-weight: 600;
  }
  div[data-testid="stHorizontalBlock"] > div { gap: 12px; }
</style>
""", unsafe_allow_html=True)

# ── Layout Constants ──────────────────────────────────────
PLOT_BG  = "#0D1F3C"
PAPER_BG = "#0D1F3C"
FONT_CLR = "#E8EDF5"
GRID_CLR = "#1E3A5A"
BANK_BLUE = "#1565C0"
BANK_RED  = "#CC0000"
PALETTE   = ["#1565C0","#CC0000","#E65100","#1B5E20","#F9A825","#6A1B9A","#00695C"]
SEG_COLORS = {0:"#1565C0",1:"#CC0000",2:"#E65100",3:"#2E7D32"}

def dark_layout(title="", height=320, **kw):
    return dict(
        title=dict(text=title, font=dict(color=FONT_CLR, size=13), x=0.01),
        paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
        font=dict(color=FONT_CLR, size=11),
        height=height,
        xaxis=dict(gridcolor=GRID_CLR, linecolor=GRID_CLR, tickfont=dict(color=FONT_CLR)),
        yaxis=dict(gridcolor=GRID_CLR, linecolor=GRID_CLR, tickfont=dict(color=FONT_CLR)),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=FONT_CLR, size=10)),
        margin=dict(l=40, r=20, t=40, b=40),
        **kw
    )

# ── Data Loaders ─────────────────────────────────────────
@st.cache_resource
def load_models():
    models = {}
    for name, fname in [("Logistic Regression","Logistic_Regression.pkl"),
                        ("Random Forest","Random_Forest.pkl"),
                        ("Gradient Boosting","Gradient_Boosting.pkl")]:
        path = f"models/{fname}"
        if os.path.exists(path):
            models[name] = joblib.load(path)
    return models

@st.cache_data
def load_all_data():
    def sr(p, **kw):
        return pd.read_csv(p, **kw) if os.path.exists(p) else pd.DataFrame()

    customers  = sr("data/customers.csv")
    repayments = sr("data/repayment_history.csv")
    transactions = sr("data/transactions.csv")
    feature_mx = sr("data/feature_matrix.csv")
    risk       = sr("outputs/risk_scored_customers.csv")
    clv        = sr("outputs/clv_estimates.csv")
    segments   = sr("outputs/customer_segments.csv")
    rev_hist   = sr("outputs/monthly_revenue.csv", parse_dates=["month"])
    rev_fc     = sr("outputs/revenue_forecast.csv", parse_dates=["month"])

    df = feature_mx.copy()
    if not risk.empty and "risk_tier" in risk.columns:
        df = df.merge(risk[["customer_id","default_probability","risk_tier"]], on="customer_id", how="left")
    if not clv.empty and "estimated_clv" in clv.columns:
        df = df.merge(clv[["customer_id","estimated_clv","clv_segment"]], on="customer_id", how="left")
    if not segments.empty and "segment" in segments.columns:
        df = df.merge(segments[["customer_id","segment"]], on="customer_id", how="left")
    if not customers.empty:
        for col in ["city","employment_type"]:
            if col in customers.columns and col not in df.columns:
                df = df.merge(customers[["customer_id",col]], on="customer_id", how="left")

    return df, repayments, transactions, customers, rev_hist, rev_fc

@st.cache_data
def load_json(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}

def feature_cols(df):
    excl = ["customer_id","defaulted","default_probability","risk_tier",
            "estimated_clv","clv_segment","segment","city","employment_type","account_type"]
    return [c for c in df.columns if c not in excl and df[c].dtype in [np.float64,np.int64,np.uint8,int,float,bool]]

def kpi_card(label, value, sub="", color=""):
    return f"""<div class="kpi-card {color}">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub">{sub}</div>
    </div>"""

# ── Load Everything ───────────────────────────────────────
models = load_models()
df, repayments, transactions, customers, rev_hist, rev_fc = load_all_data()
metrics_json = load_json("outputs/model_metrics.json")
shap_json    = load_json("outputs/shap_feature_importance.json")
feat_imp_json= load_json("outputs/feature_importance.json")

# ── Header ────────────────────────────────────────────────
st.markdown("""
<div class="dash-header">
  <div>
    <div class="dash-title">🏦 Credit Risk & Revenue Intelligence Platform</div>
    <div class="dash-subtitle">Retail Banking Portfolio Analytics · 5,000 Customers · Python | ML | ARIMA | K-Means | SHAP</div>
  </div>
  <div>
    <span class="badge-live">● LIVE ANALYTICS</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── No Data Guard ─────────────────────────────────────────
if df is None or df.empty:
    st.error("⚠️ No data found. Please run: `python run_pipeline.py`")
    st.stop()

# ── Sidebar Filters ───────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎛️ Global Filters")
    cities = ["All"] + sorted(df["city"].dropna().unique().tolist()) if "city" in df.columns else ["All"]
    sel_city = st.selectbox("City", cities)

    risk_tiers = ["All","Low Risk","Medium Risk","High Risk"]
    sel_risk = st.selectbox("Risk Tier", risk_tiers)

    seg_map = {0:"💎 Premium",1:"⚠️ Vulnerable",2:"📈 Growth",3:"🔄 Churner"}
    seg_opts = ["All"] + [seg_map[i] for i in range(4)]
    sel_seg = st.selectbox("Customer Segment", seg_opts)

    st.markdown("---")
    uploaded = st.file_uploader("Upload Custom CSV", type="csv")
    if uploaded:
        df = pd.read_csv(uploaded)

    st.markdown("---")
    st.markdown("**Pipeline Status**")
    for f, label in [("outputs/risk_scored_customers.csv","Risk Scores"),
                     ("outputs/clv_estimates.csv","CLV Estimates"),
                     ("outputs/customer_segments.csv","Segments"),
                     ("outputs/revenue_forecast.csv","Forecast")]:
        icon = "✅" if os.path.exists(f) else "❌"
        st.markdown(f"{icon} {label}")

    if models:
        st.markdown(f"**Models Loaded:** {len(models)}")
    if metrics_json.get("best_auc"):
        auc = metrics_json["best_auc"]
        color = "🟢" if auc > 0.88 else "🟡"
        st.markdown(f"{color} Best AUC-ROC: **{auc:.4f}**")

# ── Apply Filters ─────────────────────────────────────────
fdf = df.copy()
if sel_city != "All" and "city" in fdf.columns:
    fdf = fdf[fdf["city"] == sel_city]
if sel_risk != "All" and "risk_tier" in fdf.columns:
    fdf = fdf[fdf["risk_tier"] == sel_risk]
if sel_seg != "All" and "segment" in fdf.columns:
    seg_rev = {v:k for k,v in seg_map.items()}
    fdf = fdf[fdf["segment"] == seg_rev.get(sel_seg, -1)]

# ── Tabs ──────────────────────────────────────────────────
tabs = st.tabs(["📊 Executive Overview","⚠️ Risk Analytics",
                "💰 Revenue & CLV","🗃️ SQL Insights",
                "👥 Customer Intelligence","🔬 What-If Analysis"])

# ════════════════════════════════════════════════════════
# TAB 1 — EXECUTIVE OVERVIEW
# ════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown('<div class="section-title">Portfolio Executive Summary</div>', unsafe_allow_html=True)

    total    = len(fdf)
    hi_risk  = int((fdf.get("risk_tier","") == "High Risk").sum()) if "risk_tier" in fdf else 0
    def_rate = f"{fdf['defaulted'].mean():.1%}" if "defaulted" in fdf else "N/A"
    avg_clv  = f"₹{fdf['estimated_clv'].mean():,.0f}" if "estimated_clv" in fdf else "N/A"
    exposure = f"₹{fdf['loan_amount'].sum():,.0f}" if "loan_amount" in fdf else "N/A"
    avg_score= f"{fdf['credit_score'].mean():.0f}" if "credit_score" in fdf else "N/A"

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.markdown(kpi_card("Total Customers", f"{total:,}", "Active retail portfolio"), unsafe_allow_html=True)
    c2.markdown(kpi_card("High-Risk Customers", f"{hi_risk:,}", "Default prob > 60%","red"), unsafe_allow_html=True)
    c3.markdown(kpi_card("Portfolio Default Rate", def_rate, "Historical ground truth","orange"), unsafe_allow_html=True)
    c4.markdown(kpi_card("Avg Customer CLV", avg_clv, "Lifetime value estimate","green"), unsafe_allow_html=True)
    c5.markdown(kpi_card("Total Loan Exposure", exposure, "Active loan book"), unsafe_allow_html=True)
    c6.markdown(kpi_card("Avg Credit Score", avg_score, "CIBIL equivalent","gold"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Risk vs CLV Quadrant Scatter
    col_l, col_r = st.columns([3,2])
    with col_l:
        st.markdown('<div class="section-title">Risk vs CLV — Customer Quadrant</div>', unsafe_allow_html=True)
        if "default_probability" in fdf.columns and "estimated_clv" in fdf.columns:
            plot_df = fdf.sample(min(1200, len(fdf)), random_state=42)
            plot_df["seg_label"] = plot_df["segment"].map(seg_map).fillna("Unknown") if "segment" in plot_df.columns else "All"
            fig = px.scatter(
                plot_df, x="default_probability", y="estimated_clv",
                color="seg_label" if "segment" in plot_df.columns else None,
                color_discrete_sequence=list(SEG_COLORS.values()),
                hover_data=["customer_id","credit_score"] if "customer_id" in plot_df.columns else None,
                labels={"default_probability":"Default Probability →","estimated_clv":"Estimated CLV (₹) →"}
            )
            clv_mid = fdf["estimated_clv"].median()
            fig.add_vline(x=0.40, line_dash="dash", line_color="#546E7A", line_width=1)
            fig.add_hline(y=clv_mid, line_dash="dash", line_color="#546E7A", line_width=1)
            # Quadrant annotations
            for txt, xx, yy in [("⭐ Stars\n(Low Risk, High CLV)",0.1, clv_mid*1.5),
                                  ("⚠️ Watch\n(High Risk, High CLV)",0.7, clv_mid*1.5),
                                  ("✅ Safe\n(Low Risk, Low CLV)",0.1, clv_mid*0.3),
                                  ("🔴 Default\n(High Risk, Low CLV)",0.7, clv_mid*0.3)]:
                fig.add_annotation(x=xx, y=yy, text=txt, showarrow=False,
                                   font=dict(size=9, color="#546E7A"))
            fig.update_layout(**dark_layout(height=380))
            fig.update_traces(marker=dict(size=5, opacity=0.65))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run pipeline to generate risk scores and CLV estimates.")

    with col_r:
        st.markdown('<div class="section-title">Segment Distribution</div>', unsafe_allow_html=True)
        if "segment" in fdf.columns:
            seg_counts = fdf["segment"].map(seg_map).value_counts().reset_index()
            seg_counts.columns = ["Segment","Count"]
            fig2 = px.pie(seg_counts, names="Segment", values="Count", hole=0.55,
                          color_discrete_sequence=list(SEG_COLORS.values()))
            fig2.update_layout(**dark_layout(height=220))
            fig2.update_traces(textfont=dict(color=FONT_CLR), textinfo="percent+label")
            st.plotly_chart(fig2, use_container_width=True)

        if "risk_tier" in fdf.columns:
            tier_cnt = fdf["risk_tier"].value_counts().reset_index()
            tier_cnt.columns = ["Risk Tier","Count"]
            clr_map = {"Low Risk":"#1B5E20","Medium Risk":"#E65100","High Risk":"#CC0000"}
            fig3 = px.bar(tier_cnt, x="Risk Tier", y="Count",
                          color="Risk Tier", color_discrete_map=clr_map)
            fig3.update_layout(**dark_layout("Risk Tier Distribution", height=180))
            fig3.update_traces(showlegend=False)
            st.plotly_chart(fig3, use_container_width=True)

# ════════════════════════════════════════════════════════
# TAB 2 — RISK ANALYTICS
# ════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown('<div class="section-title">Credit Risk Deep Dive</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        # Default Probability Distribution
        if "default_probability" in fdf.columns:
            hist_vals = fdf["default_probability"].dropna()
            counts, bins = np.histogram(hist_vals, bins=25)
            fig = go.Figure(go.Bar(
                x=[(bins[i]+bins[i+1])/2 for i in range(len(counts))],
                y=counts, marker_color=BANK_BLUE, opacity=0.85
            ))
            fig.update_layout(**dark_layout("Default Probability Distribution", height=300))
            fig.update_xaxes(title="Default Probability")
            fig.update_yaxes(title="Customer Count")
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        # City NPA Exposure
        if "city" in fdf.columns and "defaulted" in fdf.columns and "loan_amount" in fdf.columns:
            city_df = fdf[fdf.get("has_loan", pd.Series(1, index=fdf.index)) == 1] if "has_loan" in fdf.columns else fdf
            city_agg = city_df.groupby("city").apply(
                lambda x: pd.Series({
                    "npa_exposure": x[x["defaulted"]==1]["loan_amount"].sum(),
                    "npa_rate": round(x[x["defaulted"]==1]["loan_amount"].sum() / max(x["loan_amount"].sum(),1)*100, 1)
                })
            ).reset_index().sort_values("npa_exposure", ascending=False)
            fig = px.bar(city_agg, x="city", y="npa_rate",
                         color="npa_rate", color_continuous_scale=["#1B5E20","#F9A825","#CC0000"])
            fig.update_layout(**dark_layout("NPA Rate (%) by City", height=300))
            fig.update_xaxes(title="City"); fig.update_yaxes(title="NPA Rate %")
            st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        # Rolling 3-Month Default Trend
        if not repayments.empty and "due_date" in repayments.columns:
            rep = repayments.copy()
            rep["due_date"] = pd.to_datetime(rep["due_date"])
            rep["month"] = rep["due_date"].dt.to_period("M").dt.to_timestamp()
            monthly_miss = (
                rep[rep["payment_status"] == "Missed"]
                .groupby("month")["customer_id"].nunique()
                .reset_index(name="defaulting_customers")
                .sort_values("month")
            )
            monthly_miss["rolling_3m"] = monthly_miss["defaulting_customers"].rolling(3, min_periods=1).mean()
            fig = go.Figure()
            fig.add_trace(go.Bar(x=monthly_miss["month"], y=monthly_miss["defaulting_customers"],
                                 name="Monthly Missed", marker_color=BANK_RED, opacity=0.5))
            fig.add_trace(go.Scatter(x=monthly_miss["month"], y=monthly_miss["rolling_3m"],
                                     name="3M Rolling Avg", line=dict(color="#64B5F6",width=2)))
            fig.update_layout(**dark_layout("Rolling 3-Month Default Trend", height=300))
            st.plotly_chart(fig, use_container_width=True)

    with col4:
        # SHAP Feature Importance
        shap_data = shap_json if shap_json else feat_imp_json
        if shap_data and "features" in shap_data:
            key = "mean_abs_shap" if "mean_abs_shap" in shap_data else "importances"
            top12 = list(zip(shap_data["features"][:12], shap_data[key][:12]))
            feats, vals = zip(*top12)
            colors = [BANK_BLUE if v > np.median(vals) else "#4FC3F7" for v in vals]
            fig = go.Figure(go.Bar(
                x=list(vals)[::-1], y=list(feats)[::-1],
                orientation="h", marker_color=colors[::-1]
            ))
            title = "SHAP Feature Importance" if shap_json else "Feature Importance"
            if shap_data.get("model"):
                title += f" ({shap_data['model']})"
            fig.update_layout(**dark_layout(title, height=300))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run pipeline to generate SHAP values.")

    # Model Performance Cards
    if metrics_json.get("models"):
        st.markdown('<div class="section-title">Model Performance Comparison</div>', unsafe_allow_html=True)
        mc = st.columns(len(metrics_json["models"]))
        for i, (mname, mvals) in enumerate(metrics_json["models"].items()):
            auc = mvals.get("auc_roc", 0)
            is_best = mname == metrics_json.get("best_model","")
            badge = " 🏆 Best" if is_best else ""
            color = "green" if is_best else ""
            mc[i].markdown(kpi_card(
                f"{mname}{badge}",
                f"AUC {auc:.4f}",
                f"F1: {mvals.get('f1_score',0):.3f} | AP: {mvals.get('avg_precision',0):.3f}",
                color
            ), unsafe_allow_html=True)

# ════════════════════════════════════════════════════════
# TAB 3 — REVENUE & CLV
# ════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown('<div class="section-title">Revenue Forecasting & Customer Lifetime Value</div>', unsafe_allow_html=True)

    # ARIMA 12-Month Revenue Forecast
    if not rev_fc.empty:
        hist = rev_fc[rev_fc["type"] == "historical"]
        fc   = rev_fc[rev_fc["type"] == "forecast"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist["month"], y=hist["revenue"],
                                 name="Historical", line=dict(color=BANK_BLUE,width=2),
                                 mode="lines+markers", marker=dict(size=4)))
        fig.add_trace(go.Scatter(x=fc["month"], y=fc["revenue"],
                                 name="ARIMA Forecast", line=dict(color=BANK_RED,width=2,dash="dash")))
        if "upper_95" in fc.columns and fc["upper_95"].notna().any():
            fig.add_trace(go.Scatter(
                x=pd.concat([fc["month"], fc["month"][::-1]]),
                y=pd.concat([fc["upper_95"], fc["lower_95"][::-1]]),
                fill="toself", fillcolor="rgba(204,0,0,0.10)",
                line=dict(color="rgba(0,0,0,0)"), name="95% CI"
            ))
        if not hist.empty:
            # Split into vline + separate annotation to avoid Plotly datetime string bug
            fig.add_vline(x=str(hist["month"].max()), line_dash="dot", line_color="#546E7A")
            fig.add_annotation(
                x=str(hist["month"].max()), y=1, yref="paper",
                text="◀ Historical  |  Forecast ▶", showarrow=False,
                font=dict(color="#90A4AE", size=10), xanchor="left", yanchor="top",
                bgcolor="rgba(13,31,60,0.8)", borderpad=4
            )
        fig.update_layout(**dark_layout("ARIMA 12-Month Portfolio Revenue Forecast", height=350))
        fig.update_yaxes(title="Revenue (₹)")
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        # CLV Distribution
        if "estimated_clv" in fdf.columns:
            clip_val = fdf["estimated_clv"].quantile(0.99)
            hist_data = fdf["estimated_clv"].clip(upper=clip_val)
            fig = go.Figure(go.Histogram(x=hist_data, nbinsx=40,
                                         marker_color=BANK_BLUE, opacity=0.8))
            fig.update_layout(**dark_layout("CLV Distribution", height=280))
            fig.update_xaxes(title="Estimated CLV (₹)")
            fig.update_yaxes(title="Customers")
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        # CLV by Segment Bar
        if "clv_segment" in fdf.columns and "estimated_clv" in fdf.columns:
            seg_order = ["Bronze","Silver","Gold","Platinum"]
            clv_agg = fdf.groupby("clv_segment")["estimated_clv"].mean().reindex(seg_order).dropna().reset_index()
            clv_agg.columns = ["Segment","Avg CLV"]
            colors_clv = ["#CD7F32","#9E9E9E","#FFD600","#1565C0"]
            fig = px.bar(clv_agg, x="Segment", y="Avg CLV",
                         color="Segment",
                         color_discrete_sequence=colors_clv,
                         text="Avg CLV")
            fig.update_traces(texttemplate="₹%{text:,.0f}", textposition="outside")
            fig.update_layout(**dark_layout("Average CLV by Segment", height=280))
            st.plotly_chart(fig, use_container_width=True)

    # CLV Summary Table
    if "clv_segment" in fdf.columns and "estimated_clv" in fdf.columns:
        st.markdown('<div class="section-title">CLV Segment Summary</div>', unsafe_allow_html=True)
        summ = fdf.groupby("clv_segment")["estimated_clv"].agg(
            Customers="count", Avg_CLV="mean", Total_CLV="sum"
        ).reset_index()
        summ.columns = ["Segment","Customers","Avg CLV (₹)","Total CLV (₹)"]
        summ["% Revenue"] = (summ["Total CLV (₹)"] / summ["Total CLV (₹)"].sum() * 100).round(1)
        st.dataframe(
            summ.style.format({"Avg CLV (₹)":"₹{:,.0f}","Total CLV (₹)":"₹{:,.0f}","% Revenue":"{:.1f}%"}),
            use_container_width=True
        )

# ════════════════════════════════════════════════════════
# TAB 4 — SQL INSIGHTS
# ════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown('<div class="section-title">10 Production-Grade SQL Analytical Insights</div>', unsafe_allow_html=True)

    SQL_TEXTS = {
        "Q1 — Default Rate by Employment Type": """SELECT
    employment_type,
    COUNT(*)                                         AS total_customers,
    SUM(defaulted)                                   AS total_defaults,
    ROUND(AVG(defaulted) * 100, 2)                   AS default_rate_pct,
    ROUND(AVG(credit_score), 0)                      AS avg_credit_score,
    ROUND(AVG(income), 2)                            AS avg_income,
    ROUND(AVG(loan_amount), 2)                       AS avg_loan_amount
FROM customers
GROUP BY employment_type
ORDER BY default_rate_pct DESC;""",

        "Q2 — High-Risk Customers (Default Prob > 60%)": """SELECT
    c.customer_id, c.city, c.employment_type, c.credit_score,
    c.income, c.loan_amount, rs.default_probability, rs.risk_tier, cs.segment
FROM customers c
JOIN risk_scored rs ON rs.customer_id = c.customer_id
JOIN customer_segments cs ON cs.customer_id = c.customer_id
WHERE rs.risk_tier = 'High Risk' AND c.has_loan = 1
ORDER BY rs.default_probability DESC
LIMIT 100;""",

        "Q3 — Monthly EMI Collection Efficiency": """SELECT
    DATE_TRUNC('month', due_date::DATE)              AS repayment_month,
    COUNT(*)                                         AS total_emi_dues,
    SUM(emi_amount)                                  AS total_emi_billed,
    SUM(paid_amount)                                 AS total_collected,
    ROUND(SUM(paid_amount)/NULLIF(SUM(emi_amount),0)*100, 2) AS collection_rate_pct,
    SUM(CASE WHEN payment_status = 'Missed' THEN 1 ELSE 0 END) AS missed_payments,
    SUM(CASE WHEN payment_status = 'Late' THEN 1 ELSE 0 END)   AS late_payments
FROM repayment_history
GROUP BY 1
ORDER BY 1 DESC;""",

        "Q4 — Credit Utilisation Risk Bands": """SELECT
    CASE
        WHEN cc_balance/NULLIF(credit_limit,0) < 0.30 THEN 'Low (< 30%)'
        WHEN cc_balance/NULLIF(credit_limit,0) < 0.60 THEN 'Medium (30–60%)'
        WHEN cc_balance/NULLIF(credit_limit,0) < 0.90 THEN 'High (60–90%)'
        ELSE 'Very High (> 90%)'
    END                                              AS utilisation_band,
    COUNT(*)                                         AS customer_count,
    ROUND(AVG(defaulted) * 100, 2)                   AS default_rate_pct,
    ROUND(AVG(credit_score), 0)                      AS avg_credit_score
FROM customers
WHERE has_credit_card = 1
GROUP BY 1
ORDER BY default_rate_pct DESC;""",

        "Q5 — City-Level NPA Exposure": """SELECT
    city,
    COUNT(*)                                         AS total_customers,
    SUM(loan_amount)                                 AS total_loan_exposure,
    SUM(CASE WHEN defaulted = 1 THEN loan_amount ELSE 0 END) AS npa_exposure,
    ROUND(SUM(CASE WHEN defaulted = 1 THEN loan_amount ELSE 0 END) / 
          NULLIF(SUM(loan_amount), 0) * 100, 2)      AS npa_rate_pct,
    ROUND(AVG(credit_score), 0)                      AS avg_credit_score
FROM customers
WHERE has_loan = 1
GROUP BY city
ORDER BY npa_exposure DESC;""",

        "Q6 — Transaction Category Analysis": """SELECT
    category,
    transaction_type,
    COUNT(*)                                         AS txn_count,
    ROUND(SUM(amount), 2)                            AS total_volume,
    ROUND(AVG(amount), 2)                            AS avg_txn_amount,
    ROUND(STDDEV(amount), 2)                         AS txn_amount_stddev
FROM transactions
GROUP BY category, transaction_type
ORDER BY total_volume DESC;""",

        "Q7 — CLV Distribution by Segment": """SELECT
    clv_segment,
    COUNT(*)                                         AS customer_count,
    ROUND(AVG(estimated_clv), 2)                     AS avg_clv,
    ROUND(SUM(estimated_clv), 2)                     AS total_clv,
    ROUND(SUM(estimated_clv)/SUM(SUM(estimated_clv)) OVER () * 100, 2) AS pct_of_total_clv,
    ROUND(AVG(default_probability) * 100, 2)         AS avg_default_prob_pct
FROM clv_estimates
GROUP BY clv_segment
ORDER BY avg_clv DESC;""",

        "Q8 — Rolling 3-Month Default Trend": """WITH monthly_defaults AS (
    SELECT DATE_TRUNC('month', due_date::DATE)       AS month,
           COUNT(DISTINCT r.customer_id)             AS defaulting_customers
    FROM repayment_history r
    WHERE payment_status = 'Missed'
    GROUP BY 1
)
SELECT month, defaulting_customers,
       ROUND(AVG(defaulting_customers) OVER (
           ORDER BY month ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
       ), 1)                                         AS rolling_3m_avg
FROM monthly_defaults
ORDER BY month;""",

        "Q9 — Repayment by Customer Tenure Band": """SELECT
    CASE
        WHEN c.tenure_months < 12 THEN '0–12 months'
        WHEN c.tenure_months < 36 THEN '1–3 years'
        WHEN c.tenure_months < 60 THEN '3–5 years'
        ELSE '5+ years'
    END                                              AS tenure_band,
    COUNT(DISTINCT c.customer_id)                    AS customers,
    ROUND(AVG(CASE WHEN r.payment_status = 'On-Time' THEN 1.0 ELSE 0.0 END) * 100, 2) AS on_time_rate_pct,
    ROUND(AVG(c.credit_score), 0)                    AS avg_credit_score
FROM customers c
JOIN repayment_history r ON r.customer_id = c.customer_id
GROUP BY 1
ORDER BY on_time_rate_pct DESC;""",

        "Q10 — Top 20 Cross-Sell Targets (Low Risk, High CLV)": """SELECT
    c.customer_id, c.city, c.employment_type, c.credit_score,
    cv.estimated_clv, cv.clv_segment, rs.default_probability, rs.risk_tier
FROM customers c
JOIN clv_estimates cv ON cv.customer_id = c.customer_id
JOIN risk_scored rs ON rs.customer_id = c.customer_id
WHERE rs.risk_tier = 'Low Risk' AND cv.clv_segment IN ('Gold', 'Platinum')
ORDER BY cv.estimated_clv DESC
LIMIT 20;""",

        "Q11 — Risk Concentration by Income Band": """SELECT
    CASE
        WHEN income < 200000  THEN 'Low Income (< ₹2L)'
        WHEN income < 500000  THEN 'Lower Middle (₹2L–5L)'
        WHEN income < 1000000 THEN 'Middle (₹5L–10L)'
        WHEN income < 2000000 THEN 'Upper Middle (₹10L–20L)'
        ELSE 'High Income (> ₹20L)'
    END                                              AS income_band,
    COUNT(*)                                         AS total_customers,
    SUM(defaulted)                                   AS total_defaults,
    ROUND(AVG(defaulted)*100, 2)                     AS default_rate_pct,
    ROUND(AVG(credit_score), 0)                      AS avg_credit_score,
    ROUND(AVG(loan_amount), 2)                       AS avg_loan_amount,
    ROUND(SUM(CASE WHEN defaulted=1 THEN loan_amount ELSE 0 END), 2) AS npa_exposure,
    ROUND(SUM(CASE WHEN defaulted=1 THEN loan_amount ELSE 0 END)/NULLIF(SUM(loan_amount),0)*100, 2) AS npa_rate_pct
FROM customers
WHERE has_loan = 1
GROUP BY 1
ORDER BY default_rate_pct DESC;""",

        "Q12 — Churner Segment CLV at Risk by City": """WITH churner_base AS (
    SELECT c.customer_id, c.city, rs.default_probability, cv.estimated_clv, c.credit_score
    FROM customers c
    JOIN risk_scored rs ON rs.customer_id = c.customer_id
    JOIN clv_estimates cv ON cv.customer_id = c.customer_id
    JOIN customer_segments cs ON cs.customer_id = c.customer_id
    WHERE cs.segment = 3  -- Churner Watch cluster
)
SELECT city,
    COUNT(*)                                         AS churner_count,
    ROUND(AVG(default_probability)*100, 2)           AS avg_default_prob_pct,
    ROUND(AVG(estimated_clv), 2)                     AS avg_clv,
    ROUND(SUM(estimated_clv), 2)                     AS total_clv_at_risk,
    ROUND(AVG(credit_score), 0)                      AS avg_credit_score
FROM churner_base
GROUP BY city
ORDER BY total_clv_at_risk DESC;"""
    }

    SQL_CAPTIONS = {
        "Q1 — Default Rate by Employment Type": "💡 **What this does:** Aggregates the portfolio cross-sectioning by employment type. It highlights which employment sectors generate the highest delinquency rates to inform credit policy constraints.",
        "Q2 — High-Risk Customers (Default Prob > 60%)": "💡 **What this does:** Joins the core customer table with the ML-generated risk scores and segmentation data to extract a targeted, actionable list of the top 100 customers with critical default probabilities. Designed for immediate collections outreach.",
        "Q3 — Monthly EMI Collection Efficiency": "💡 **What this does:** Evaluates operational health by comparing total EMI billed vs. total cash collected per month. It isolates missed and late payments to calculate a precise collection rate percentage over the trailing 12 months.",
        "Q4 — Credit Utilisation Risk Bands": "💡 **What this does:** Segments revolving credit users into four distinct utilisation bands (Low to Very High) by dividing current CC Balance by Credit Limit. It proves the correlation between high card utilisation and ultimate loan default.",
        "Q5 — City-Level NPA Exposure": "💡 **What this does:** Performs a geographic risk concentration analysis. It calculates the raw financial exposure (Non-Performing Assets in ₹) and NPA rate for every operating city to identify localized economic stress.",
        "Q6 — Transaction Category Analysis": "💡 **What this does:** Summarizes thousands of raw debit/credit transactions into top-level categories. It calculates volumes and standard deviations to uncover high-frequency spending behaviors and product usage.",
        "Q7 — CLV Distribution by Segment": "💡 **What this does:** Distributes the modelled Customer Lifetime Value across Bronze, Silver, Gold, and Platinum tiers. The window function `SUM(...) OVER ()` cleanly calculates the percentage of total portfolio revenue captured by each tier.",
        "Q8 — Rolling 3-Month Default Trend": "💡 **What this does:** Uses an advanced `OVER (ORDER BY ... ROWS BETWEEN ...)` window function to smooth out month-over-month volatility in missed payments, presenting a cleaner macroscopic view of portfolio degrading or improving trends.",
        "Q9 — Repayment by Customer Tenure Band": "💡 **What this does:** Groups customers into tenure bands based on their account age. It joins against raw repayment streams to prove that 'On-Time Rate' generally improves as the institutional relationship matures.",
        "Q10 — Top 20 Cross-Sell Targets (Low Risk, High CLV)": "💡 **What this does:** The optimal target list for the marketing department. It filters for `Low Risk` customers strictly in the `Gold` and `Platinum` CLV segments, ensuring upselling efforts are completely risk-averse.",
        "Q11 — Risk Concentration by Income Band": "💡 **What this does:** Automatically buckets raw continuous income into readable macroeconomic brackets. It calculates both human count defaults and raw financial NPA exposure to validate pricing bands.",
        "Q12 — Churner Segment CLV at Risk by City": "💡 **What this does:** A powerful cross-module query. It isolates customers explicitly placed into the 'Churner Watch' cluster by the K-Means algorithm (Segment 3) and aggregates the total estimated CLV sitting at risk across different geographies."
    }

    SQL_QUERIES = {
        "Q1 — Default Rate by Employment Type": lambda: (
            df.groupby("employment_type").agg(
                Total_Customers=("customer_id","count"),
                Total_Defaults=("defaulted","sum"),
                Default_Rate_Pct=("defaulted",lambda x: round(x.mean()*100,2)),
                Avg_Credit_Score=("credit_score","mean"),
                Avg_Income=("income","mean")
            ).reset_index().sort_values("Default_Rate_Pct",ascending=False)
            if "employment_type" in df.columns else pd.DataFrame()
        ),
        "Q2 — High-Risk Customers (Default Prob > 60%)": lambda: (
            df[df["risk_tier"]=="High Risk"][
                [c for c in ["customer_id","city","employment_type","credit_score","income","loan_amount","default_probability","risk_tier"] if c in df.columns]
            ].sort_values("default_probability",ascending=False).head(20)
            if "risk_tier" in df.columns else pd.DataFrame()
        ),
        "Q3 — Monthly EMI Collection Efficiency": lambda: (
            (lambda r: (
                r.assign(month=pd.to_datetime(r["due_date"]).dt.to_period("M").dt.to_timestamp())
                .groupby("month").agg(
                    Total_EMI_Billed=("emi_amount","sum"),
                    Total_Collected=("paid_amount","sum"),
                    Missed=("payment_status",lambda x:(x=="Missed").sum()),
                    Late=("payment_status",lambda x:(x=="Late").sum()),
                ).assign(Collection_Rate_Pct=lambda x: (x["Total_Collected"]/x["Total_EMI_Billed"]*100).round(2))
                .reset_index().sort_values("month",ascending=False)
            ))(repayments) if not repayments.empty else pd.DataFrame()
        ),
        "Q4 — Credit Utilisation Risk Bands": lambda: (
            df.assign(util=df["cc_balance"]/df["credit_limit"].replace(0,np.nan))
            .assign(Utilisation_Band=lambda x: pd.cut(x["util"],
                bins=[0,0.3,0.6,0.9,999],
                labels=["Low (<30%)","Medium (30-60%)","High (60-90%)","Very High (>90%)"]
            ))
            .groupby("Utilisation_Band").agg(
                Customer_Count=("customer_id","count"),
                Default_Rate_Pct=("defaulted",lambda x:round(x.mean()*100,2)),
                Avg_Credit_Score=("credit_score","mean")
            ).reset_index().sort_values("Default_Rate_Pct",ascending=False)
            if "cc_balance" in df.columns and "credit_limit" in df.columns else pd.DataFrame()
        ),
        "Q5 — City-Level NPA Exposure": lambda: (
            df[df.get("has_loan",pd.Series(1,index=df.index))==1].groupby("city").apply(
                lambda x: pd.Series({
                    "Total_Customers": len(x),
                    "Total_Loan_Exposure": x["loan_amount"].sum(),
                    "NPA_Exposure": x[x["defaulted"]==1]["loan_amount"].sum(),
                    "NPA_Rate_Pct": round(x[x["defaulted"]==1]["loan_amount"].sum()/max(x["loan_amount"].sum(),1)*100,2),
                    "Avg_Credit_Score": round(x["credit_score"].mean(),0)
                })
            ).reset_index().sort_values("NPA_Exposure",ascending=False)
            if "city" in df.columns else pd.DataFrame()
        ),
        "Q6 — Transaction Category Analysis": lambda: (
            transactions.groupby(["category","transaction_type"]).agg(
                Txn_Count=("amount","count"),
                Total_Volume=("amount","sum"),
                Avg_Txn_Amount=("amount","mean")
            ).reset_index().sort_values("Total_Volume",ascending=False).head(20)
            if not transactions.empty else pd.DataFrame()
        ),
        "Q7 — CLV Distribution by Segment": lambda: (
            df.groupby("clv_segment")["estimated_clv"].agg(
                Customer_Count="count", Avg_CLV="mean", Total_CLV="sum"
            ).assign(Pct_of_Total=lambda x: (x["Total_CLV"]/x["Total_CLV"].sum()*100).round(2))
            .reset_index().sort_values("Avg_CLV",ascending=False)
            if "clv_segment" in df.columns else pd.DataFrame()
        ),
        "Q8 — Rolling 3-Month Default Trend": lambda: (
            (lambda r: (
                r[r["payment_status"]=="Missed"]
                .assign(month=pd.to_datetime(r["due_date"]).dt.to_period("M").dt.to_timestamp())
                .groupby("month")["customer_id"].nunique().reset_index(name="Defaulting_Customers")
                .sort_values("month")
                .assign(Rolling_3M_Avg=lambda x: x["Defaulting_Customers"].rolling(3,min_periods=1).mean().round(1))
            ))(repayments) if not repayments.empty else pd.DataFrame()
        ),
        "Q9 — Repayment by Customer Tenure Band": lambda: (
            (lambda merged: merged.groupby(
                pd.cut(merged["tenure_months"],bins=[0,12,36,60,999],
                       labels=["0-12m","1-3yr","3-5yr","5+yr"])
            ).agg(
                Customers=("customer_id","nunique"),
                On_Time_Rate_Pct=("payment_status",lambda x:round((x=="On-Time").mean()*100,2)),
                Avg_Credit_Score=("credit_score","mean")
            ).reset_index())(
                df.merge(repayments,on="customer_id",how="inner")
            ) if not repayments.empty and "tenure_months" in df.columns else pd.DataFrame()
        ),
        "Q10 — Top 20 Cross-Sell Targets (Low Risk, High CLV)": lambda: (
            df[(df.get("risk_tier","")=="Low Risk") & (df.get("clv_segment","").isin(["Gold","Platinum"]))][
                [c for c in ["customer_id","city","employment_type","credit_score","estimated_clv","clv_segment","default_probability"] if c in df.columns]
            ].sort_values("estimated_clv",ascending=False).head(20)
            if "risk_tier" in df.columns and "clv_segment" in df.columns else pd.DataFrame()
        ),
        "Q11 — Risk Concentration by Income Band": lambda: (
            df[df.get("has_loan",pd.Series(1,index=df.index))==1].assign(
                Income_Band=pd.cut(df["income"],
                    bins=[0,200000,500000,1000000,2000000,999999999],
                    labels=["<2L","2-5L","5-10L","10-20L",">20L"])
            ).groupby("Income_Band").agg(
                Total_Customers=("customer_id","count"),
                Total_Defaults=("defaulted","sum"),
                Default_Rate_Pct=("defaulted",lambda x:round(x.mean()*100,2)),
                Avg_Credit_Score=("credit_score","mean"),
                NPA_Exposure=("loan_amount",lambda x: x[df.loc[x.index,"defaulted"]==1].sum())
            ).reset_index().sort_values("Default_Rate_Pct",ascending=False)
            if "income" in df.columns else pd.DataFrame()
        ),
        "Q12 — Churner Segment CLV at Risk by City": lambda: (
            df[df.get("segment",-1)==3].groupby("city").agg(
                Churner_Count=("customer_id","count"),
                Avg_Default_Prob_Pct=("default_probability",lambda x:round(x.mean()*100,2)),
                Avg_CLV=("estimated_clv","mean"),
                Total_CLV_At_Risk=("estimated_clv","sum"),
                Avg_Credit_Score=("credit_score","mean")
            ).reset_index().sort_values("Total_CLV_At_Risk",ascending=False)
            if "segment" in df.columns and "estimated_clv" in df.columns and "city" in df.columns else pd.DataFrame()
        ),
    }

    selected_q = st.selectbox("Select SQL Query to Execute", list(SQL_QUERIES.keys()))
    if st.button("▶ Run Query", type="primary"):
        st.info(SQL_CAPTIONS.get(selected_q, ""))
        with st.expander("View Target SQL Source"):
            st.code(SQL_TEXTS.get(selected_q, "-- No SQL available"), language="sql")
        
        try:
            result = SQL_QUERIES[selected_q]()
            if result is not None and not result.empty:
                st.success(f"Returned {len(result):,} rows")
                numeric_cols = result.select_dtypes(include=[np.number]).columns
                fmt = {c: "{:,.2f}" for c in numeric_cols}
                if any("clv" in c.lower() or "exposure" in c.lower() or "income" in c.lower() for c in numeric_cols):
                    for c in numeric_cols:
                        if "clv" in c.lower() or "exposure" in c.lower() or "income" in c.lower():
                            fmt[c] = "₹{:,.0f}"
                st.dataframe(result.style.format(fmt, na_rep="—"), use_container_width=True)

                # Auto chart for numeric results
                if len(result) > 1 and len(numeric_cols) >= 1:
                    x_col = result.columns[0]
                    y_col = numeric_cols[0]
                    fig = px.bar(result.head(15), x=x_col, y=y_col,
                                 color_discrete_sequence=[BANK_BLUE])
                    fig.update_layout(**dark_layout(f"{selected_q}: {y_col}", height=280))
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No results. Run pipeline first.")
        except Exception as e:
            st.error(f"Query error: {e}")
    else:
        st.info("👆 Click **Run Query** to execute and visualize the selected analytical query.")

# ════════════════════════════════════════════════════════
# TAB 5 — CUSTOMER INTELLIGENCE
# ════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown('<div class="section-title">Customer Intelligence & Segment Drill-Down</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([3,2])
    with col1:
        # Drill-through customer table
        show_cols = [c for c in ["customer_id","city","employment_type","credit_score","income",
                                  "loan_amount","default_probability","risk_tier","estimated_clv","clv_segment","segment"]
                     if c in fdf.columns]
        page_df = fdf[show_cols].copy()
        if "segment" in page_df.columns:
            page_df["segment"] = page_df["segment"].map(seg_map)
        if "default_probability" in page_df.columns:
            page_df["default_probability"] = page_df["default_probability"].map("{:.1%}".format)
        if "estimated_clv" in page_df.columns:
            page_df["estimated_clv"] = page_df["estimated_clv"].map("₹{:,.0f}".format)

        st.markdown(f"**Showing {min(200,len(page_df)):,} of {len(fdf):,} customers** (apply sidebar filters to narrow down)")
        st.dataframe(page_df.head(200), use_container_width=True, height=400)

    with col2:
        # Segment Radar
        if "segment" in df.columns:
            radar_features = [f for f in ["credit_score","repayment_consistency_score",
                               "spending_volatility","default_probability","credit_utilisation_ratio"]
                              if f in df.columns]
            if radar_features:
                seg_means = df.groupby("segment")[radar_features].mean()
                seg_norm  = (seg_means - seg_means.min()) / (seg_means.max() - seg_means.min() + 1e-9)

                fig = go.Figure()
                colors_r = ["#1565C0","#CC0000","#E65100","#2E7D32"]
                for i, seg_id in enumerate(seg_norm.index):
                    vals = seg_norm.loc[seg_id].tolist() + [seg_norm.loc[seg_id].tolist()[0]]
                    angles = [n/len(radar_features)*360 for n in range(len(radar_features))] + [0]
                    fig.add_trace(go.Scatterpolar(
                        r=vals, theta=radar_features+[radar_features[0]],
                        fill="toself", name=seg_map.get(seg_id,f"Seg {seg_id}"),
                        line=dict(color=colors_r[i%4]), opacity=0.7
                    ))
                fig.update_layout(
                    polar=dict(bgcolor=PLOT_BG,
                               radialaxis=dict(visible=True,gridcolor=GRID_CLR,color=FONT_CLR),
                               angularaxis=dict(gridcolor=GRID_CLR,color=FONT_CLR)),
                    paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
                    font=dict(color=FONT_CLR), height=380,
                    legend=dict(bgcolor="rgba(0,0,0,0)",font=dict(color=FONT_CLR)),
                    title=dict(text="Segment Radar Profile", font=dict(color=FONT_CLR)),
                    margin=dict(l=60,r=60,t=50,b=40)
                )
                st.plotly_chart(fig, use_container_width=True)

    # High-Risk Action Table
    if "risk_tier" in fdf.columns and "default_probability" in fdf.columns:
        st.markdown('<div class="section-title">🔴 Immediate Action Required — Top 15 High-Risk</div>', unsafe_allow_html=True)
        hr_cols = [c for c in ["customer_id","city","employment_type","credit_score","loan_amount","default_probability","risk_tier"] if c in fdf.columns]
        top_hr = fdf[fdf["risk_tier"]=="High Risk"].sort_values("default_probability",ascending=False).head(15)[hr_cols]
        if "default_probability" in top_hr.columns:
            st.dataframe(
                top_hr.style.format({"default_probability":"{:.1%}","loan_amount":"₹{:,.0f}","income":"₹{:,.0f}"})
                      .background_gradient(subset=["default_probability"],cmap="Reds"),
                use_container_width=True
            )

# ════════════════════════════════════════════════════════
# TAB 6 — WHAT-IF ANALYSIS
# ════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown('<div class="section-title">Credit Decision Simulator — What-If Analysis</div>', unsafe_allow_html=True)
    st.caption("Adjust customer parameters to simulate how risk and CLV respond dynamically.")

    if not models:
        st.error("No models loaded. Run `python run_pipeline.py` first.")
    else:
        feat_list = feature_cols(df)
        wi_model_name = st.selectbox("Select Model", list(models.keys()), key="wi_mod")
        wi_model = models[wi_model_name]

        cust_ids = df["customer_id"].head(100).tolist()
        wi_cust  = st.selectbox("Select Baseline Customer", cust_ids, key="wi_cust")
        base     = df[df["customer_id"] == wi_cust].iloc[0].to_dict()

        col1, col2, col3 = st.columns(3)
        with col1:
            adj_loan  = st.slider("Loan Amount (₹)", 0.0,
                                   float(df["loan_amount"].max()*1.5 if "loan_amount" in df.columns else 1e6),
                                   float(base.get("loan_amount",0)), step=5000.0)
            adj_util  = st.slider("Credit Utilization", 0.0, 1.5,
                                   float(base.get("credit_utilisation_ratio",0.0)), step=0.05)
        with col2:
            adj_income= st.slider("Annual Income (₹)", 50000.0,
                                   float(df["income"].max()*1.5 if "income" in df.columns else 5e6),
                                   float(base.get("income",100000)), step=10000.0)
            adj_score = st.slider("Credit Score", 300, 900,
                                   int(base.get("credit_score",650)), step=10)
        with col3:
            adj_rcs   = st.slider("Repayment Consistency", 0.0, 1.0,
                                   float(base.get("repayment_consistency_score",0.75)), step=0.05)
            adj_miss  = st.slider("Total Missed Payments", 0, 12,
                                   int(base.get("total_missed_payments",0)), step=1)

        updated = base.copy()
        updated.update({
            "loan_amount": adj_loan, "credit_utilisation_ratio": adj_util,
            "income": adj_income, "credit_score": adj_score,
            "repayment_consistency_score": adj_rcs, "total_missed_payments": adj_miss,
            "income_to_loan_ratio": adj_income / max(adj_loan, 1),
        })

        try:
            X_wi = pd.DataFrame([updated])[feat_list].fillna(0).astype(float)
            prob = wi_model.predict_proba(X_wi)[0, 1]
        except Exception:
            prob = 0.5

        interest_income = adj_loan * (updated.get("interest_rate_pct",10)/100)
        cc_fee = updated.get("credit_limit",0) * 0.02
        txn_fee = adj_income * 0.003
        annual_rev = interest_income + cc_fee + txn_fee
        tenure = max(0.5, updated.get("tenure_months",12)/12)
        clv_sim = annual_rev * tenure * (1 - prob)
        risk_tier_sim = "High Risk" if prob >= 0.6 else ("Medium Risk" if prob >= 0.3 else "Low Risk")
        base_prob = base.get("default_probability", prob)

        st.markdown("<br>", unsafe_allow_html=True)
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Simulated Default Prob", f"{prob:.1%}",
                  delta=f"{(prob-base_prob):.1%}", delta_color="inverse")
        r2.metric("Risk Tier", risk_tier_sim)
        r3.metric("Simulated CLV", f"₹{clv_sim:,.0f}",
                  delta=f"₹{(clv_sim - base.get('estimated_clv', clv_sim)):,.0f}")
        r4.metric("Annual Revenue Est.", f"₹{annual_rev:,.0f}")

        # Gauge
        gauge_color = "#CC0000" if prob >= 0.6 else ("#E65100" if prob >= 0.3 else "#2E7D32")
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=prob * 100,
            delta={"reference": base_prob * 100, "valueformat":".1f",
                   "increasing":{"color":"#CC0000"},"decreasing":{"color":"#2E7D32"}},
            number={"suffix":"%","font":{"size":28,"color":FONT_CLR}},
            gauge={
                "axis":{"range":[0,100],"tickcolor":FONT_CLR},
                "bar":{"color":gauge_color},
                "bgcolor":PLOT_BG,
                "steps":[{"range":[0,30],"color":"#1B5E20"},
                         {"range":[30,60],"color":"#E65100"},
                         {"range":[60,100],"color":"#CC0000"}],
                "threshold":{"line":{"color":"white","width":2},"thickness":0.75,"value":40}
            },
            title={"text":"Default Probability Gauge","font":{"color":FONT_CLR,"size":13}}
        ))
        fig_g.update_layout(paper_bgcolor=PAPER_BG, font=dict(color=FONT_CLR), height=280, margin=dict(l=30,r=30,t=50,b=20))
        st.plotly_chart(fig_g, use_container_width=True)

        # Recommendation
        if prob >= 0.6:
            st.error("🚨 **DECLINE**: High default risk. Recommend credit restriction and risk mitigation review.")
        elif prob >= 0.3:
            st.warning("⚠️ **CONDITIONAL APPROVAL**: Medium risk. Apply enhanced monitoring, covenants, or collateral requirements.")
        else:
            st.success("✅ **APPROVE**: Low default risk. Eligible for cross-sell of premium products and credit line increase.")
