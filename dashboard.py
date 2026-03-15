"""
============================================================
  Banking Intelligence Project — Banking Customer Risk & Revenue Intelligence
  Module 7: Interactive Dashboard — Flask + Plotly
  Purpose : Executive-facing web dashboard displaying all KPIs,
            model outputs, and segment insights.
  Run     : python dashboard.py   →  http://localhost:5000
============================================================
"""

from flask import Flask, render_template_string, jsonify
import pandas as pd
import numpy as np
import json, os

app = Flask(__name__)

# ─── Load all output files (graceful fallback to empty) ──

def safe_read(path, **kwargs):
    try:
        return pd.read_csv(path, **kwargs)
    except FileNotFoundError:
        return pd.DataFrame()


def load_data():
    customers  = safe_read("data/customers.csv")
    risk       = safe_read("outputs/risk_scored_customers.csv")
    clv        = safe_read("outputs/clv_estimates.csv")
    segments   = safe_read("outputs/customer_segments.csv")
    repayments = safe_read("data/repayment_history.csv")
    txn        = safe_read("data/transactions.csv")

    df = customers.copy()
    if not risk.empty:
        df = df.merge(risk[["customer_id", "default_probability", "risk_tier"]],
                      on="customer_id", how="left")
    if not clv.empty:
        df = df.merge(clv[["customer_id", "estimated_clv", "clv_segment"]],
                      on="customer_id", how="left")
    if not segments.empty:
        df = df.merge(segments, on="customer_id", how="left")

    return df, repayments, txn


# ─────────────────────────────────────────────────────────
# DASHBOARD HTML TEMPLATE
# ─────────────────────────────────────────────────────────

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Banking Intelligence | Credit Risk & Revenue Intelligence Dashboard</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.26.0/plotly.min.js"></script>
<style>
  * { margin:0; padding:0; box-sizing:border-box; }
  body { font-family: 'Segoe UI', Arial, sans-serif; background:#F4F6F9; color:#1a1a2e; }

  /* ── Header ── */
  .header {
    background: linear-gradient(135deg, #003366 0%, #CC0000 100%);
    color: white; padding: 20px 32px;
    display: flex; align-items: center; justify-content: space-between;
  }
  .header h1 { font-size: 22px; font-weight: 700; letter-spacing: 0.5px; }
  .header .subtitle { font-size: 13px; opacity: 0.85; margin-top: 4px; }
  .header .timestamp { font-size: 12px; opacity: 0.75; text-align:right; }
  .bank-hex { width: 40px; height: 40px; background: white;
              clip-path: polygon(50% 0%, 100% 25%, 100% 75%, 50% 100%, 0% 75%, 0% 25%);
              display:flex; align-items:center; justify-content:center;
              color: #CC0000; font-weight: 900; font-size: 14px; margin-right: 16px; }

  /* ── KPI Cards ── */
  .kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
              gap: 16px; padding: 24px 32px 0; }
  .kpi-card { background: white; border-radius: 10px; padding: 18px 20px;
              box-shadow: 0 2px 8px rgba(0,0,0,0.07); border-left: 4px solid #003366; }
  .kpi-card.red    { border-left-color: #CC0000; }
  .kpi-card.orange { border-left-color: #FF8C00; }
  .kpi-card.green  { border-left-color: #28A745; }
  .kpi-label { font-size: 11px; color: #666; text-transform: uppercase;
               letter-spacing: 0.5px; margin-bottom: 6px; }
  .kpi-value { font-size: 26px; font-weight: 700; color: #003366; }
  .kpi-card.red    .kpi-value { color: #CC0000; }
  .kpi-card.orange .kpi-value { color: #FF8C00; }
  .kpi-card.green  .kpi-value { color: #28A745; }
  .kpi-sub   { font-size: 11px; color: #999; margin-top: 4px; }

  /* ── Charts Grid ── */
  .charts-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px;
                 padding: 24px 32px; }
  .chart-card  { background: white; border-radius: 10px; padding: 20px;
                 box-shadow: 0 2px 8px rgba(0,0,0,0.07); }
  .chart-card.wide { grid-column: span 2; }
  .chart-title { font-size: 14px; font-weight: 600; color: #003366;
                 margin-bottom: 14px; padding-bottom: 8px;
                 border-bottom: 2px solid #F0F2F5; }

  /* ── Table ── */
  .table-wrapper { overflow-x: auto; }
  table { width: 100%; border-collapse: collapse; font-size: 12px; }
  th { background: #003366; color: white; padding: 8px 12px; text-align:left; }
  td { padding: 7px 12px; border-bottom: 1px solid #F0F2F5; }
  tr:hover td { background: #F7F8FA; }
  .badge { padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: 600; }
  .badge.high   { background: #FFE4E4; color: #CC0000; }
  .badge.medium { background: #FFF3E0; color: #FF8C00; }
  .badge.low    { background: #E8F5E9; color: #28A745; }

  /* ── Footer ── */
  .footer { text-align: center; padding: 20px; font-size: 11px; color: #999;
            border-top: 1px solid #E0E0E0; margin-top: 8px; }
</style>
</head>
<body>

<!-- ── Header ── -->
<div class="header">
  <div style="display:flex;align-items:center;">
    <div class="bank-hex">B</div>
    <div>
      <h1>Credit Risk & Revenue Intelligence Dashboard</h1>
      <div class="subtitle">Retail Banking Portfolio Analytics · Banking Intelligence Project</div>
    </div>
  </div>
  <div class="timestamp">
    Last refreshed: <span id="ts"></span><br>
    Data: Synthetic · 5,000 Customers
  </div>
</div>

<!-- ── KPI Cards ── -->
<div class="kpi-grid">
  <div class="kpi-card">
    <div class="kpi-label">Total Customers</div>
    <div class="kpi-value" id="kpi-total">—</div>
    <div class="kpi-sub">Active retail portfolio</div>
  </div>
  <div class="kpi-card red">
    <div class="kpi-label">High Risk Customers</div>
    <div class="kpi-value" id="kpi-highrisk">—</div>
    <div class="kpi-sub">Default prob > 60%</div>
  </div>
  <div class="kpi-card orange">
    <div class="kpi-label">Portfolio Default Rate</div>
    <div class="kpi-value" id="kpi-defrate">—</div>
    <div class="kpi-sub">Historical ground truth</div>
  </div>
  <div class="kpi-card green">
    <div class="kpi-label">Avg Customer CLV</div>
    <div class="kpi-value" id="kpi-clv">—</div>
    <div class="kpi-sub">Lifetime value estimate</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">Avg Credit Score</div>
    <div class="kpi-value" id="kpi-score">—</div>
    <div class="kpi-sub">CIBIL equivalent</div>
  </div>
  <div class="kpi-card green">
    <div class="kpi-label">Platinum CLV Customers</div>
    <div class="kpi-value" id="kpi-platinum">—</div>
    <div class="kpi-sub">Top revenue contributors</div>
  </div>
</div>

<!-- ── Charts ── -->
<div class="charts-grid">

  <div class="chart-card">
    <div class="chart-title">📊 Default Probability Distribution</div>
    <div id="chart-defprob" style="height:280px;"></div>
  </div>

  <div class="chart-card">
    <div class="chart-title">🎯 Risk Tier Breakdown</div>
    <div id="chart-risktier" style="height:280px;"></div>
  </div>

  <div class="chart-card">
    <div class="chart-title">📈 CLV by Segment</div>
    <div id="chart-clvseg" style="height:280px;"></div>
  </div>

  <div class="chart-card">
    <div class="chart-title">🏙️ NPA Exposure by City</div>
    <div id="chart-city" style="height:280px;"></div>
  </div>

  <div class="chart-card wide">
    <div class="chart-title">💳 Credit Score Distribution by Employment Type</div>
    <div id="chart-creditbox" style="height:300px;"></div>
  </div>

  <div class="chart-card wide">
    <div class="chart-title">🔴 Top 20 High-Risk Customers Requiring Immediate Action</div>
    <div class="table-wrapper">
      <table>
        <thead>
          <tr>
            <th>Customer ID</th><th>City</th><th>Credit Score</th>
            <th>Income (₹)</th><th>Loan Amount (₹)</th>
            <th>Default Prob</th><th>Risk Tier</th>
          </tr>
        </thead>
        <tbody id="high-risk-table">
          <tr><td colspan="7" style="text-align:center;color:#999;">Loading…</td></tr>
        </tbody>
      </table>
    </div>
  </div>

</div>

<div class="footer">
  Banking Intelligence Project · Banking Customer Risk & Revenue Intelligence System ·
  Python | SQL | Machine Learning | Flask | Plotly
</div>

<script>
document.getElementById("ts").textContent = new Date().toLocaleString();

async function loadDashboard() {
  const res  = await fetch("/api/dashboard-data");
  const data = await res.json();

  // KPIs
  document.getElementById("kpi-total").textContent     = data.kpis.total_customers.toLocaleString();
  document.getElementById("kpi-highrisk").textContent  = data.kpis.high_risk.toLocaleString();
  document.getElementById("kpi-defrate").textContent   = data.kpis.default_rate;
  document.getElementById("kpi-clv").textContent       = "₹" + data.kpis.avg_clv;
  document.getElementById("kpi-score").textContent     = data.kpis.avg_credit_score;
  document.getElementById("kpi-platinum").textContent  = data.kpis.platinum_count.toLocaleString();

  const BANK_BLUE = "#003366"; const BANK_RED = "#CC0000";
  const layout_base = { margin:{t:10,b:40,l:50,r:20}, paper_bgcolor:"white",
                         plot_bgcolor:"#F9FAFB", font:{size:11} };

  // Default probability histogram
  Plotly.newPlot("chart-defprob", [{
    x: data.def_prob_hist.x, y: data.def_prob_hist.y,
    type:"bar", marker:{color:BANK_RED, opacity:0.8},
    name:"Customers"
  }], {...layout_base, xaxis:{title:"Default Probability"},
       yaxis:{title:"Count"}}, {responsive:true});

  // Risk tier donut
  Plotly.newPlot("chart-risktier", [{
    labels: data.risk_tier.labels, values: data.risk_tier.values,
    type:"pie", hole:0.5,
    marker:{colors:["#28A745","#FF8C00","#CC0000"]},
    textinfo:"label+percent"
  }], {margin:{t:10,b:10,l:10,r:10}, paper_bgcolor:"white",
       showlegend:true, legend:{orientation:"h",y:-0.15}},
      {responsive:true});

  // CLV by segment bar
  Plotly.newPlot("chart-clvseg", [{
    x: data.clv_seg.labels, y: data.clv_seg.avg_clv,
    type:"bar",
    marker:{color:["#CD7F32","#C0C0C0","#FFD700","#4A90D9"]},
    text: data.clv_seg.avg_clv.map(v => "₹" + v.toLocaleString()),
    textposition:"outside"
  }], {...layout_base, yaxis:{title:"Avg CLV (₹)"},
       xaxis:{title:"CLV Segment"}}, {responsive:true});

  // City NPA bar
  Plotly.newPlot("chart-city", [{
    x: data.city_npa.cities, y: data.city_npa.npa_rate,
    type:"bar", marker:{color:BANK_BLUE, opacity:0.85},
    text: data.city_npa.npa_rate.map(v => v + "%"), textposition:"outside"
  }], {...layout_base, yaxis:{title:"NPA Rate %"},
       xaxis:{title:"City"}}, {responsive:true});

  // Credit score box plot
  const boxTraces = data.credit_box.employment_types.map((emp, i) => ({
    y: data.credit_box.scores[i], name: emp, type:"box",
    boxpoints: false, marker:{size:3}
  }));
  Plotly.newPlot("chart-creditbox", boxTraces,
    {...layout_base, yaxis:{title:"Credit Score"},
     xaxis:{title:"Employment Type"}}, {responsive:true});

  // High-risk table
  const tbody = document.getElementById("high-risk-table");
  tbody.innerHTML = data.high_risk_table.map(r => `
    <tr>
      <td><strong>${r.customer_id}</strong></td>
      <td>${r.city}</td>
      <td>${r.credit_score}</td>
      <td>₹${Number(r.income).toLocaleString()}</td>
      <td>₹${Number(r.loan_amount).toLocaleString()}</td>
      <td><strong style="color:#CC0000;">${(r.default_probability*100).toFixed(1)}%</strong></td>
      <td><span class="badge high">${r.risk_tier}</span></td>
    </tr>
  `).join("");
}

loadDashboard();
</script>
</body>
</html>
"""


# ─────────────────────────────────────────────────────────
# API ENDPOINT — Return all chart data as JSON
# ─────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template_string(DASHBOARD_HTML)


@app.route("/api/dashboard-data")
def dashboard_data():
    df, repayments, txn = load_data()

    if df.empty:
        return jsonify({"error": "No data found. Run pipeline first."}), 500

    # ── KPIs ──
    kpis = {
        "total_customers": int(len(df)),
        "high_risk":       int((df.get("risk_tier", pd.Series()) == "High Risk").sum()),
        "default_rate":    f"{df['defaulted'].mean():.1%}" if "defaulted" in df.columns else "N/A",
        "avg_clv":         f"{df['estimated_clv'].mean():,.0f}" if "estimated_clv" in df.columns else "N/A",
        "avg_credit_score": int(df["credit_score"].mean()),
        "platinum_count":  int((df.get("clv_segment", pd.Series()) == "Platinum").sum()),
    }

    # ── Default probability histogram ──
    if "default_probability" in df.columns:
        hist_vals = df["default_probability"].dropna()
        counts, bins = np.histogram(hist_vals, bins=20)
        def_hist = {
            "x": [round((bins[i] + bins[i+1]) / 2, 3) for i in range(len(counts))],
            "y": counts.tolist()
        }
    else:
        def_hist = {"x": [], "y": []}

    # ── Risk tier donut ──
    if "risk_tier" in df.columns:
        vc = df["risk_tier"].value_counts()
        risk_tier = {"labels": vc.index.tolist(), "values": vc.values.tolist()}
    else:
        risk_tier = {"labels": [], "values": []}

    # ── CLV by segment ──
    if "clv_segment" in df.columns and "estimated_clv" in df.columns:
        seg_order = ["Bronze", "Silver", "Gold", "Platinum"]
        clv_agg   = df.groupby("clv_segment")["estimated_clv"].mean().reindex(seg_order).dropna()
        clv_seg   = {
            "labels":  clv_agg.index.tolist(),
            "avg_clv": [round(v, 0) for v in clv_agg.values.tolist()]
        }
    else:
        clv_seg = {"labels": [], "avg_clv": []}

    # ── City NPA ──
    if "defaulted" in df.columns and "loan_amount" in df.columns:
        city_df  = df[df["has_loan"] == 1].copy()
        city_agg = (
            city_df.groupby("city")
            .apply(lambda x: round(
                x[x["defaulted"] == 1]["loan_amount"].sum() /
                max(x["loan_amount"].sum(), 1) * 100, 2
            ))
            .reset_index()
        )
        city_agg.columns = ["city", "npa_rate"]
        city_agg = city_agg.sort_values("npa_rate", ascending=False).head(7)
        city_npa = {"cities": city_agg["city"].tolist(),
                    "npa_rate": city_agg["npa_rate"].tolist()}
    else:
        city_npa = {"cities": [], "npa_rate": []}

    # ── Credit score box by employment ──
    emp_types = df["employment_type"].dropna().unique().tolist() if "employment_type" in df.columns else []
    credit_box = {
        "employment_types": emp_types,
        "scores": [
            df[df["employment_type"] == e]["credit_score"].dropna().tolist()
            for e in emp_types
        ]
    }

    # ── High-risk table ──
    if "default_probability" in df.columns:
        hr_cols = ["customer_id", "city", "credit_score", "income",
                   "loan_amount", "default_probability", "risk_tier"]
        hr_cols = [c for c in hr_cols if c in df.columns]
        hr = (
            df[df.get("risk_tier", pd.Series()) == "High Risk"]
            .sort_values("default_probability", ascending=False)
            .head(20)[hr_cols]
            .fillna(0)
        )
        hr_table = hr.to_dict(orient="records")
    else:
        hr_table = []

    return jsonify({
        "kpis":           kpis,
        "def_prob_hist":  def_hist,
        "risk_tier":      risk_tier,
        "clv_seg":        clv_seg,
        "city_npa":       city_npa,
        "credit_box":     credit_box,
        "high_risk_table": hr_table,
    })


if __name__ == "__main__":
    print("🚀  Starting Dashboard → http://localhost:5000")
    # Keep debugger enabled, but avoid watchdog reload loops from external package changes.
    use_reloader = os.getenv("BANK_DASHBOARD_RELOADER", "0") == "1"
    app.run(debug=True, use_reloader=use_reloader, port=5000)
