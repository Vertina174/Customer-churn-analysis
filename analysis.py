"""
analysis.py
-----------
Reusable data loading, cleaning, and analysis functions for
the Customer Churn Analysis dashboard.

All functions return plain pandas DataFrames / dicts so they
remain framework-agnostic and easy to unit-test.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_FILE = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

CATEGORICAL_COLS = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

SERVICE_COLS = [
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies",
]

TENURE_BINS   = [0, 12, 24, 48, 72]
TENURE_LABELS = ["0-12 months", "13-24 months", "25-48 months", "49-72 months"]

# ---------------------------------------------------------------------------
# Loading & Cleaning
# ---------------------------------------------------------------------------

def load_data(filepath: str = DATA_FILE) -> pd.DataFrame:
    """
    Load the Telco Customer Churn CSV, fix known data-quality issues,
    and return a clean DataFrame ready for analysis.

    Raises
    ------
    FileNotFoundError
        If the CSV is not found at *filepath*.
    ValueError
        If required columns are missing.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: '{filepath}'. "
            "Please place 'WA_Fn-UseC_-Telco-Customer-Churn.csv' "
            "in the project folder and restart."
        )

    df = pd.read_csv(filepath)

    required = {"customerID", "Churn", "tenure", "MonthlyCharges", "TotalCharges"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(f"Dataset is missing expected columns: {missing_cols}")

    df = _clean_data(df)
    return df


def _clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all cleaning steps and return the sanitised DataFrame."""

    df = df.copy()

    # --- Fix TotalCharges: stored as object with blank strings for new customers
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Fill the 11 blank TotalCharges with 0 (new customers, tenure==0)
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # --- SeniorCitizen: convert 0/1 int to Yes/No string for consistency
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})

    # --- Churn: keep as Yes/No string, also add binary column for maths
    df["ChurnFlag"] = (df["Churn"] == "Yes").astype(int)

    # --- Tenure cohorts
    df["TenureGroup"] = pd.cut(
        df["tenure"],
        bins=TENURE_BINS,
        labels=TENURE_LABELS,
        right=True,
        include_lowest=True,
    )

    # --- Monthly charge tier
    df["ChargeTier"] = pd.cut(
        df["MonthlyCharges"],
        bins=[0, 35, 65, 100, df["MonthlyCharges"].max() + 1],
        labels=["Low (<$35)", "Medium ($35-$65)", "High ($65-$100)", "Very High (>$100)"],
        right=False,
    )

    return df

# ---------------------------------------------------------------------------
# KPI Helpers
# ---------------------------------------------------------------------------

def get_kpis(df: pd.DataFrame) -> dict:
    """Return a dict of headline KPI values."""
    total        = len(df)
    churned      = df["ChurnFlag"].sum()
    churn_rate   = churned / total * 100
    avg_monthly  = df["MonthlyCharges"].mean()
    avg_tenure   = df["tenure"].mean()
    total_rev    = df["TotalCharges"].sum()
    rev_at_risk  = df.loc[df["Churn"] == "Yes", "MonthlyCharges"].sum()

    return {
        "total_customers": int(total),
        "churned_customers": int(churned),
        "retained_customers": int(total - churned),
        "churn_rate": round(churn_rate, 2),
        "avg_monthly_charges": round(avg_monthly, 2),
        "avg_tenure_months": round(avg_tenure, 1),
        "total_revenue": round(total_rev, 2),
        "monthly_revenue_at_risk": round(rev_at_risk, 2),
    }

# ---------------------------------------------------------------------------
# Churn Breakdown Helpers
# ---------------------------------------------------------------------------

def churn_by_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Return a DataFrame with churn counts and rates for each
    unique value in *col*.
    """
    grp = (
        df.groupby(col)["ChurnFlag"]
        .agg(Total="count", Churned="sum")
        .reset_index()
    )
    grp["Retained"] = grp["Total"] - grp["Churned"]
    grp["ChurnRate"] = (grp["Churned"] / grp["Total"] * 100).round(2)
    return grp.sort_values("ChurnRate", ascending=False)


def churn_by_services(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each add-on service, compute the churn rate among customers
    who HAVE that service vs those who DON'T.

    Returns a long-form DataFrame: [Service, HasService, ChurnRate]
    """
    rows = []
    for svc in SERVICE_COLS:
        for val in df[svc].dropna().unique():
            mask  = df[svc] == val
            rate  = df.loc[mask, "ChurnFlag"].mean() * 100
            rows.append({"Service": svc, "Value": val, "ChurnRate": round(rate, 2),
                         "Count": mask.sum()})
    return pd.DataFrame(rows)


def monthly_charges_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Return histogram-ready data for MonthlyCharges split by Churn."""
    return df[["MonthlyCharges", "Churn"]].copy()


def tenure_churn_trend(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return average churn rate per tenure month (1–72) for a line-chart
    showing when customers are most likely to leave.
    """
    trend = (
        df.groupby("tenure")["ChurnFlag"]
        .mean()
        .reset_index()
        .rename(columns={"ChurnFlag": "ChurnRate"})
    )
    trend["ChurnRate"] = (trend["ChurnRate"] * 100).round(2)
    return trend


def revenue_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return per-contract-type revenue summary.
    """
    return (
        df.groupby("Contract")
        .agg(
            Customers=("customerID", "count"),
            AvgMonthly=("MonthlyCharges", "mean"),
            TotalRevenue=("TotalCharges", "sum"),
            ChurnRate=("ChurnFlag", lambda x: round(x.mean() * 100, 2)),
        )
        .reset_index()
    )

# ---------------------------------------------------------------------------
# Machine Learning
# ---------------------------------------------------------------------------

def build_churn_model(df: pd.DataFrame):
    """
    Train a Logistic Regression model to predict churn.

    Returns
    -------
    model : fitted sklearn Pipeline
    X_test : pd.DataFrame
    y_test : pd.Series
    feature_names : list[str]
    metrics : dict  (accuracy, precision, recall, f1, roc_auc)
    """
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score,
    )

    # --- Feature selection (exclude IDs, target, derived cols)
    drop_cols = ["customerID", "Churn", "ChurnFlag", "TenureGroup", "ChargeTier"]
    feature_df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # One-hot encode categoricals
    feature_df = pd.get_dummies(feature_df, drop_first=True)
    feature_names = feature_df.columns.tolist()

    X = feature_df.values
    y = df["ChurnFlag"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    pipe.fit(X_train, y_train)

    y_pred  = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy" : round(accuracy_score(y_test, y_pred)  * 100, 2),
        "precision": round(precision_score(y_test, y_pred) * 100, 2),
        "recall"   : round(recall_score(y_test, y_pred)    * 100, 2),
        "f1"       : round(f1_score(y_test, y_pred)        * 100, 2),
        "roc_auc"  : round(roc_auc_score(y_test, y_proba)  * 100, 2),
    }

    # Return test split as DataFrames for the confusion matrix plot
    X_test_df = pd.DataFrame(X_test, columns=feature_names)
    return pipe, X_test_df, pd.Series(y_test), feature_names, metrics


def get_feature_importance(model, feature_names: list) -> pd.DataFrame:
    """
    Extract feature coefficients from the Logistic Regression inside
    the Pipeline and return a tidy DataFrame.
    """
    clf  = model.named_steps["clf"]
    coef = clf.coef_[0]
    imp  = pd.DataFrame({"Feature": feature_names, "Coefficient": coef})
    imp["AbsCoef"] = imp["Coefficient"].abs()
    return imp.sort_values("AbsCoef", ascending=False).head(15)
