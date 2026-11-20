import os
import sqlite3
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, chi2_contingency
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

os.makedirs("outputs", exist_ok=True)
DB_PATH = "data/loyalty_analytics.db"
SQL_PATH = "sql/feature_engineering.sql"


def run_sql_pipeline():
    with open(SQL_PATH, "r", encoding="utf-8") as f:
        sql_script = f.read()
    with sqlite3.connect(DB_PATH) as conn:
        conn.executescript(sql_script)
        abt = pd.read_sql("SELECT * FROM abt_member_features", conn)
        monthly = pd.read_sql("SELECT * FROM agg_monthly_trends", conn)
    return abt, monthly


def run_statistical_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Statistical validation of key loyalty drivers (JD requirement)."""
    results = []
    churned = df[df["is_churn_or_downgrade"] == 1]
    retained = df[df["is_churn_or_downgrade"] == 0]

    num_cols = ["pqp_velocity_ratio", "tier_pacing_ratio", "recency_days", "delay_rate_recent_6m", "complaints_recent_6m"]
    for col in num_cols:
        stat, p_val = mannwhitneyu(churned[col].fillna(0), retained[col].fillna(0), alternative="two-sided")
        results.append({
            "feature": col,
            "test_type": "Mann-Whitney U",
            "churned_median": round(churned[col].median(), 3),
            "retained_median": round(retained[col].median(), 3),
            "p_value": float(f"{p_val:.4e}"),
            "statistically_significant": "Yes" if p_val < 0.05 else "No",
        })

    # Chi-Square test for Co-Brand Card retention impact
    ct = pd.crosstab(df["has_cobrand_card"], df["is_churn_or_downgrade"])
    chi2, p_val, _, _ = chi2_contingency(ct)
    card_churn = df.groupby("has_cobrand_card")["is_churn_or_downgrade"].mean()
    results.append({
        "feature": "has_cobrand_card",
        "test_type": "Chi-Square",
        "churned_median": round(card_churn.loc[0], 3),   # Churn rate without card
        "retained_median": round(card_churn.loc[1], 3),  # Churn rate with card
        "p_value": float(f"{p_val:.4e}"),
        "statistically_significant": "Yes" if p_val < 0.05 else "No",
    })
    return pd.DataFrame(results)


def assign_driver_and_offer(row):
    """Translates model signals into actionable retention treatments for Power BI."""
    if row["churn_probability"] < 0.35:
        return "Engaged / Low Risk", "Standard MileagePlus Nurture", 0.0

    if row["delay_rate_recent_6m"] >= 0.25 or row["complaints_recent_6m"] >= 2:
        return "Operational Disruption Friction", "Service Recovery Voucher + 5k Bonus Miles", 45.0
    if row["has_cobrand_card"] == 0 and row["current_tier"] != "General":
        return "No Co-Brand Card Anchor", "Co-Brand Card Fast-Track + 500 Starter PQP", 65.0
    if row["tier_pacing_ratio"] < 0.70:
        return "Behind Tier PQP Pacing", "Targeted Double-PQP Status Challenge", 55.0
    if row["pqp_velocity_ratio"] < 0.65:
        return "H1 Flight Frequency Decay", "15% Off Next 2 Flights Promo Code", 40.0
    return "Multi-Factor Engagement Drift", "Personalized Miles Multiplier Offer", 35.0


def main():
    abt, monthly = run_sql_pipeline()
    abt["delay_rate_recent_6m"] = abt["delay_rate_recent_6m"].fillna(0.0)

    # 1. Statistical Hypothesis Testing
    stat_tests = run_statistical_tests(abt)
    print("\n--- Statistical Validation Summary ---")
    print(stat_tests.to_string(index=False))

    # 2. Feature Set & Train/Test Split
    num_features = [
        "tenure_years", "recency_days", "pqf_12m", "flight_pqp_12m",
        "pqp_velocity_ratio", "pqf_delta_6m", "tier_pacing_ratio",
        "delay_rate_recent_6m", "cancels_recent_6m", "premium_cabin_share",
        "intl_flight_share", "has_cobrand_card", "card_spend_12m",
        "card_velocity_ratio", "complaints_recent_6m", "redemptions_12m",
        "avg_csat_score", "within_tier_value_quartile", "decay_percentile_in_tier",
    ]
    cat_features = ["current_tier", "home_hub", "card_tier", "age_band"]

    X = abt[num_features + cat_features]
    y = abt["is_churn_or_downgrade"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # 3. Model 1: Logistic Regression (Interpretable Baseline)
    lr_prep = ColumnTransformer([
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features),
    ])
    lr_pipe = Pipeline([
        ("prep", lr_prep),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced", C=0.5)),
    ])
    lr_pipe.fit(X_train, y_train)
    lr_probs = lr_pipe.predict_proba(X_test)[:, 1]

    # 4. Model 2: HistGradientBoostingClassifier (Non-linear Champion)
    gb_prep = ColumnTransformer([
        ("num", "passthrough", num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features),
    ])
    gb_pipe = Pipeline([
        ("prep", gb_prep),
        ("model", HistGradientBoostingClassifier(
            max_depth=4, learning_rate=0.05, max_iter=200, min_samples_leaf=40, random_state=42
        )),
    ])
    gb_pipe.fit(X_train, y_train)
    gb_probs = gb_pipe.predict_proba(X_test)[:, 1]

    # 5. Evaluate Models including Top-20% Capture Lift
    def top20_capture(y_true, probs):
        cutoff = np.quantile(probs, 0.80)
        return y_true[probs >= cutoff].sum() / y_true.sum()

    metrics_df = pd.DataFrame([
        {
            "model_name": "Logistic Regression",
            "roc_auc": round(roc_auc_score(y_test, lr_probs), 4),
            "pr_auc": round(average_precision_score(y_test, lr_probs), 4),
            "top_20pct_churn_capture": round(top20_capture(y_test.values, lr_probs), 4),
        },
        {
            "model_name": "Gradient Boosting",
            "roc_auc": round(roc_auc_score(y_test, gb_probs), 4),
            "pr_auc": round(average_precision_score(y_test, gb_probs), 4),
            "top_20pct_churn_capture": round(top20_capture(y_test.values, gb_probs), 4),
        },
    ])
    print("\n--- Model Performance Comparison ---")
    print(metrics_df.to_string(index=False))

    # 6. Global Feature Importance (Permutation Importance on Test Set)
    perm = permutation_importance(gb_pipe, X_test, y_test, n_repeats=10, random_state=42, scoring="roc_auc")
    drivers_df = pd.DataFrame({
        "feature": num_features + cat_features,
        "importance_auc_drop": np.round(perm.importances_mean, 5),
    }).sort_values("importance_auc_drop", ascending=False).head(12)

    # 7. Score Full Population & Generate Actionable Retention Table
    abt["churn_probability"] = np.round(gb_pipe.predict_proba(X)[:, 1], 4)
    abt["risk_band"] = pd.cut(
        abt["churn_probability"],
        bins=[-0.01, 0.35, 0.65, 1.01],
        labels=["Low Risk", "Medium Risk", "High Risk"],
    )
    abt["expected_revenue_at_risk_usd"] = np.round(
        abt["churn_probability"] * abt["projected_annual_revenue_usd"], 2
    )

    actions = abt.apply(assign_driver_and_offer, axis=1, result_type="expand")
    abt["primary_churn_driver"] = actions[0]
    abt["recommended_offer"] = actions[1]
    abt["estimated_offer_cost_usd"] = actions[2]

    # Save Power BI Exports
    abt.to_csv("outputs/pbi_member_predictions.csv", index=False)
    monthly.to_csv("outputs/pbi_monthly_trends.csv", index=False)
    drivers_df.to_csv("outputs/pbi_feature_drivers.csv", index=False)
    metrics_df.to_csv("outputs/pbi_model_metrics.csv", index=False)
    stat_tests.to_csv("outputs/pbi_statistical_tests.csv", index=False)

    print("\nPower BI export files written to outputs/ successfully.")


if __name__ == "__main__":
    main()