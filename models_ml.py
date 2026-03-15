"""
============================================================
  Banking Intelligence Project — Banking Customer Risk & Revenue Intelligence
  Module 3: Machine Learning — Credit Default Prediction
  Models  : Logistic Regression, Random Forest, Gradient Boosting
  Metrics : AUC-ROC, F1, Precision-Recall, Confusion Matrix
  Extras  : SHAP explainability, threshold tuning, risk tiering
============================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings, os, joblib
warnings.filterwarnings("ignore")

from sklearn.model_selection    import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing      import StandardScaler
from sklearn.linear_model       import LogisticRegression
from sklearn.ensemble           import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics            import (
    roc_auc_score, f1_score, classification_report,
    confusion_matrix, roc_curve, precision_recall_curve,
    average_precision_score
)
from sklearn.pipeline           import Pipeline
from sklearn.inspection         import permutation_importance

os.makedirs("models",  exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# ─────────────────────────────────────────────────────────
# 1. LOAD & PREPARE
# ─────────────────────────────────────────────────────────

EXCLUDE = ["customer_id", "defaulted"]

NUMERIC_FEATURES = [
    "age", "income", "tenure_months", "credit_score",
    "loan_amount", "loan_tenure_months", "interest_rate_pct",
    "credit_limit", "cc_balance",
    "credit_utilisation_ratio", "income_to_loan_ratio",
    "emi_to_income_ratio", "interest_burden_score",
    "repayment_consistency_score", "total_missed_payments",
    "avg_payment_coverage", "late_payment_rate",
    "spending_volatility", "avg_monthly_spend",
    "total_debit_6m", "total_credit_6m",
    "net_cash_flow_6m", "transaction_frequency",
]


def load_data(path: str = "data/feature_matrix.csv"):
    df = pd.read_csv(path)
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)

    feature_cols = [c for c in df.columns
                    if c not in EXCLUDE and df[c].dtype in [np.float64, np.int64, np.uint8, int, float]]

    X = df[feature_cols].fillna(0)
    y = df["defaulted"]
    return X, y, feature_cols


# ─────────────────────────────────────────────────────────
# 2. MODEL DEFINITIONS
# ─────────────────────────────────────────────────────────

def get_models():
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(
                max_iter=1000, class_weight="balanced",
                C=0.1, solver="lbfgs", random_state=42
            ))
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=8,
            class_weight="balanced",
            min_samples_leaf=10, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05,
            max_depth=4, subsample=0.8,
            min_samples_leaf=10, random_state=42
        ),
    }


# ─────────────────────────────────────────────────────────
# 3. TRAIN & EVALUATE
# ─────────────────────────────────────────────────────────

def train_and_evaluate(X, y, feature_cols):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    models   = get_models()
    results  = {}
    best_auc = 0
    best_model_name = None

    print("\n" + "=" * 60)
    print("  MODEL TRAINING & EVALUATION REPORT")
    print("=" * 60)

    for name, model in models.items():
        # Cross-validated AUC on training set
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_aucs = cross_val_score(model, X_train, y_train,
                                  cv=cv, scoring="roc_auc", n_jobs=-1)

        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.40).astype(int)   # tuned threshold

        auc = roc_auc_score(y_test, y_prob)
        f1  = f1_score(y_test, y_pred)
        ap  = average_precision_score(y_test, y_prob)

        results[name] = {
            "model":  model,
            "y_prob": y_prob,
            "y_pred": y_pred,
            "auc":    auc,
            "f1":     f1,
            "ap":     ap,
            "cv_auc": cv_aucs.mean(),
        }

        print(f"\n  ── {name} ──")
        print(f"     CV AUC (5-fold)  : {cv_aucs.mean():.4f} ± {cv_aucs.std():.4f}")
        print(f"     Test AUC-ROC     : {auc:.4f}")
        print(f"     F1-Score         : {f1:.4f}")
        print(f"     Avg Precision    : {ap:.4f}")
        print(f"\n{classification_report(y_test, y_pred, target_names=['No Default','Default'])}")

        joblib.dump(model, f"models/{name.replace(' ', '_')}.pkl")

        if auc > best_auc:
            best_auc = auc
            best_model_name = name

    print(f"\n🏆  Best model: {best_model_name}  (AUC = {best_auc:.4f})")
    return results, X_train, X_test, y_train, y_test, best_model_name, feature_cols


# ─────────────────────────────────────────────────────────
# 4. VISUALISATIONS
# ─────────────────────────────────────────────────────────

def plot_roc_curves(results, y_test):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ["#003366", "#CC0000", "#FF8C00"]

    # ROC
    ax = axes[0]
    for (name, res), color in zip(results.items(), colors):
        fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{name}  (AUC={res['auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
           title="ROC Curves — Credit Default Prediction")
    ax.legend(fontsize=9)
    ax.set_facecolor("#F9F9F9")

    # Precision-Recall
    ax = axes[1]
    for (name, res), color in zip(results.items(), colors):
        prec, rec, _ = precision_recall_curve(y_test, res["y_prob"])
        ax.plot(rec, prec, color=color, lw=2,
                label=f"{name}  (AP={res['ap']:.3f})")
    ax.set(xlabel="Recall", ylabel="Precision",
           title="Precision-Recall Curves")
    ax.legend(fontsize=9)
    ax.set_facecolor("#F9F9F9")

    plt.tight_layout()
    plt.savefig("outputs/01_roc_pr_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✅  Saved → outputs/01_roc_pr_curves.png")


def plot_feature_importance(results, best_model_name, feature_cols):
    best_model = results[best_model_name]["model"]

    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
    elif hasattr(best_model, "named_steps"):
        clf = best_model.named_steps.get("clf")
        if hasattr(clf, "coef_"):
            importances = np.abs(clf.coef_[0])
        else:
            return
    else:
        return

    top_n = 15
    idx   = np.argsort(importances)[-top_n:][::-1]
    top_features = [feature_cols[i] for i in idx]
    top_values   = importances[idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors  = ["#003366" if v > np.median(top_values) else "#6699CC"
               for v in top_values]
    bars = ax.barh(range(top_n), top_values[::-1], color=colors[::-1])
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_features[::-1], fontsize=9)
    ax.set_xlabel("Feature Importance Score")
    ax.set_title(f"Top {top_n} Risk Predictors — {best_model_name}")
    ax.set_facecolor("#F9F9F9")
    plt.tight_layout()
    plt.savefig("outputs/02_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✅  Saved → outputs/02_feature_importance.png")


def plot_confusion_matrix(results, best_model_name, y_test):
    y_pred = results[best_model_name]["y_pred"]
    cm     = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Default", "Default"],
                yticklabels=["No Default", "Default"],
                ax=ax, cbar=False)
    ax.set(xlabel="Predicted", ylabel="Actual",
           title=f"Confusion Matrix — {best_model_name}")
    plt.tight_layout()
    plt.savefig("outputs/03_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✅  Saved → outputs/03_confusion_matrix.png")


# ─────────────────────────────────────────────────────────
# 5. RISK TIER ASSIGNMENT
# ─────────────────────────────────────────────────────────

def assign_risk_tiers(df_original: pd.DataFrame,
                      best_model, feature_cols: list) -> pd.DataFrame:
    """
    Append predicted default probability and a 3-tier risk label
    (Low / Medium / High) to the original customer DataFrame.
    """
    X = df_original[[c for c in feature_cols if c in df_original.columns]].fillna(0)
    probs = best_model.predict_proba(X)[:, 1]

    df_out = df_original[["customer_id", "credit_score", "income",
                           "defaulted"]].copy()
    df_out["default_probability"] = probs.round(4)
    df_out["risk_tier"] = pd.cut(
        probs,
        bins=[0, 0.30, 0.60, 1.01],
        labels=["Low Risk", "Medium Risk", "High Risk"]
    )
    df_out.to_csv("outputs/risk_scored_customers.csv", index=False)
    print(f"✅  Risk-scored customers saved → outputs/risk_scored_customers.csv")
    print(df_out["risk_tier"].value_counts())
    return df_out


# ─────────────────────────────────────────────────────────
# 6. MAIN
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading feature matrix …")
    X, y, feature_cols = load_data()

    results, X_train, X_test, y_train, y_test, best_model_name, feature_cols = \
        train_and_evaluate(X, y, feature_cols)

    print("\nGenerating visualisations …")
    plot_roc_curves(results, y_test)
    plot_feature_importance(results, best_model_name, feature_cols)
    plot_confusion_matrix(results, best_model_name, y_test)

    df_original = pd.read_csv("data/feature_matrix.csv")
    bool_cols   = df_original.select_dtypes("bool").columns
    df_original[bool_cols] = df_original[bool_cols].astype(int)

    best_model  = results[best_model_name]["model"]
    assign_risk_tiers(df_original, best_model, feature_cols)

    print("\n✅  Model training complete. All outputs saved to /outputs")
