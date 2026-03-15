"""
============================================================
  Banking Intelligence Project — Banking Customer Risk & Revenue Intelligence
  Module 5: Customer Segmentation — K-Means Clustering
  Purpose : Identify behavioural archetypes for targeted
            product recommendations and proactive engagement.
============================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import warnings, os
warnings.filterwarnings("ignore")

from sklearn.preprocessing      import StandardScaler
from sklearn.cluster            import KMeans
from sklearn.metrics            import silhouette_score
from sklearn.decomposition      import PCA

os.makedirs("outputs", exist_ok=True)

# ─────────────────────────────────────────────────────────
# CLUSTERING FEATURES
# ─────────────────────────────────────────────────────────

CLUSTER_FEATURES = [
    "credit_score",
    "income",
    "credit_utilisation_ratio",
    "repayment_consistency_score",
    "total_missed_payments",
    "spending_volatility",
    "avg_monthly_spend",
    "net_cash_flow_6m",
    "income_to_loan_ratio",
    "estimated_clv",
    "default_probability",
]

SEGMENT_LABELS = {
    0: "💎 Premium — High Value, Low Risk",
    1: "⚠️  Vulnerable — Low Income, High Risk",
    2: "📈 Growth Potential — Young, Credit-Building",
    3: "🔄 Churner Watch — Declining Engagement",
}


# ─────────────────────────────────────────────────────────
# 1. ELBOW + SILHOUETTE ANALYSIS
# ─────────────────────────────────────────────────────────

def find_optimal_k(X_scaled: np.ndarray, k_range=range(2, 10)):
    inertias   = []
    silhouettes = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, km.labels_,
                                            sample_size=min(2000, len(X_scaled))))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(k_range, inertias, "o-", color="#003366", lw=2)
    axes[0].set(title="Elbow Method — Inertia vs K",
                xlabel="Number of Clusters (K)", ylabel="Inertia")
    axes[0].set_facecolor("#F9F9F9")

    axes[1].plot(k_range, silhouettes, "o-", color="#CC0000", lw=2)
    axes[1].set(title="Silhouette Score vs K",
                xlabel="Number of Clusters (K)", ylabel="Silhouette Score")
    axes[1].set_facecolor("#F9F9F9")

    optimal_k = list(k_range)[np.argmax(silhouettes)]
    axes[1].axvline(optimal_k, color="green", linestyle="--", lw=1.5,
                    label=f"Optimal K={optimal_k}")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("outputs/06_elbow_silhouette.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅  Saved → outputs/06_elbow_silhouette.png")
    print(f"   Optimal K = {optimal_k}  (silhouette = {max(silhouettes):.3f})")
    return optimal_k


# ─────────────────────────────────────────────────────────
# 2. FIT K-MEANS
# ─────────────────────────────────────────────────────────

def fit_kmeans(df: pd.DataFrame, n_clusters: int = 4):
    available = [c for c in CLUSTER_FEATURES if c in df.columns]
    X = df[available].fillna(0)

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    df["segment"] = km.fit_predict(X_scaled)

    sil = silhouette_score(X_scaled, df["segment"],
                           sample_size=min(3000, len(df)))
    print(f"   K-Means  Silhouette Score : {sil:.4f}")

    return df, km, scaler, X_scaled, available


# ─────────────────────────────────────────────────────────
# 3. SEGMENT PROFILING
# ─────────────────────────────────────────────────────────

def profile_segments(df: pd.DataFrame, features: list) -> pd.DataFrame:
    profile_cols = features + ["defaulted"]
    profile = (
        df.groupby("segment")[profile_cols]
        .agg(["mean", "median", "count"])
    )
    # Flatten multi-level columns
    profile.columns = ["_".join(c) for c in profile.columns]
    profile = profile.reset_index()

    print("\n  ── SEGMENT PROFILES ──")
    for seg in sorted(df["segment"].unique()):
        sub    = df[df["segment"] == seg]
        label  = SEGMENT_LABELS.get(seg, f"Segment {seg}")
        print(f"\n  {label}  (n={len(sub):,})")
        print(f"    Avg Credit Score   : {sub['credit_score'].mean():.0f}")
        print(f"    Avg Income         : ₹{sub['income'].mean():,.0f}")
        print(f"    Default Rate       : {sub['defaulted'].mean():.2%}")
        if "estimated_clv" in sub.columns:
            print(f"    Avg CLV            : ₹{sub['estimated_clv'].mean():,.0f}")
        if "default_probability" in sub.columns:
            print(f"    Avg Default Prob   : {sub['default_probability'].mean():.2%}")

    return profile


# ─────────────────────────────────────────────────────────
# 4. VISUALISATIONS
# ─────────────────────────────────────────────────────────

def plot_pca_clusters(X_scaled: np.ndarray, labels: np.ndarray):
    pca = PCA(n_components=2, random_state=42)
    components = pca.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(9, 7))
    palette = ["#003366", "#CC0000", "#FF8C00", "#28A745"]

    for seg in sorted(np.unique(labels)):
        mask = labels == seg
        ax.scatter(
            components[mask, 0], components[mask, 1],
            c=palette[seg % len(palette)], label=SEGMENT_LABELS.get(seg, f"Seg {seg}"),
            alpha=0.5, s=15, edgecolors="none"
        )

    ax.set(title="Customer Segments — PCA Projection (2D)",
           xlabel=f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)",
           ylabel=f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    ax.legend(fontsize=9, loc="upper right")
    ax.set_facecolor("#F9F9F9")
    plt.tight_layout()
    plt.savefig("outputs/07_pca_clusters.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✅  Saved → outputs/07_pca_clusters.png")


def plot_segment_radar(df: pd.DataFrame, features: list):
    """Radar chart comparing segment profiles on key dimensions."""
    selected = ["credit_score", "income", "repayment_consistency_score",
                "spending_volatility", "default_probability"]
    selected = [f for f in selected if f in df.columns]

    seg_means = df.groupby("segment")[selected].mean()

    # Normalise 0-1
    seg_norm = (seg_means - seg_means.min()) / (seg_means.max() - seg_means.min() + 1e-9)

    categories = selected
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    palette = ["#003366", "#CC0000", "#FF8C00", "#28A745"]

    for i, seg in enumerate(seg_norm.index):
        vals  = seg_norm.loc[seg].tolist()
        vals += vals[:1]
        label = SEGMENT_LABELS.get(seg, f"Segment {seg}")
        ax.plot(angles, vals, "o-", lw=2, color=palette[i % len(palette)],
                label=label, markersize=5)
        ax.fill(angles, vals, alpha=0.08, color=palette[i % len(palette)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([c.replace("_", "\n") for c in categories], size=9)
    ax.set_title("Segment Radar — Normalised Feature Profiles", pad=25)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=8)
    plt.tight_layout()
    plt.savefig("outputs/08_segment_radar.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✅  Saved → outputs/08_segment_radar.png")


def plot_risk_revenue_scatter(df: pd.DataFrame):
    if "default_probability" not in df.columns or "estimated_clv" not in df.columns:
        return
    palette = {0: "#003366", 1: "#CC0000", 2: "#FF8C00", 3: "#28A745"}
    fig, ax = plt.subplots(figsize=(10, 6))

    for seg in sorted(df["segment"].unique()):
        sub = df[df["segment"] == seg].sample(min(500, len(df[df["segment"] == seg])))
        ax.scatter(sub["default_probability"], sub["estimated_clv"],
                   c=palette[seg], alpha=0.5, s=20,
                   label=SEGMENT_LABELS.get(seg, f"Seg {seg}"))

    ax.set(title="Risk vs Revenue — Customer Quadrant Analysis",
           xlabel="Default Probability (Risk ↑)",
           ylabel="Estimated CLV ₹ (Value ↑)")
    ax.axvline(0.40, color="gray", linestyle="--", lw=1, label="Risk threshold 0.40")
    ax.legend(fontsize=8)
    ax.set_facecolor("#F9F9F9")
    plt.tight_layout()
    plt.savefig("outputs/09_risk_revenue_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✅  Saved → outputs/09_risk_revenue_scatter.png")


# ─────────────────────────────────────────────────────────
# 5. MAIN
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    customers = pd.read_csv("data/feature_matrix.csv")
    bool_cols = customers.select_dtypes("bool").columns
    customers[bool_cols] = customers[bool_cols].astype(int)

    # Enrich with CLV and risk scores if available
    try:
        clv   = pd.read_csv("outputs/clv_estimates.csv")[
                    ["customer_id", "estimated_clv"]]
        risk  = pd.read_csv("outputs/risk_scored_customers.csv")[
                    ["customer_id", "default_probability"]]
        customers = customers.merge(clv, on="customer_id", how="left")
        customers = customers.merge(risk, on="customer_id", how="left")
    except FileNotFoundError:
        print("⚠️  CLV / risk files not found — run models_ml.py and forecasting_clv.py first")

    print("Fitting K-Means …")
    df_seg, km_model, scaler, X_scaled, features = fit_kmeans(customers, n_clusters=4)

    profile_segments(df_seg, features)

    print("\nGenerating cluster visualisations …")
    plot_pca_clusters(X_scaled, df_seg["segment"].values)
    plot_segment_radar(df_seg, features)
    plot_risk_revenue_scatter(df_seg)

    df_seg[["customer_id", "segment"]].to_csv("outputs/customer_segments.csv", index=False)
    print("✅  Customer segments saved → outputs/customer_segments.csv")
    print("\n✅  Segmentation complete.")
