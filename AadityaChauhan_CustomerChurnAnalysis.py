"""
AdityaChauhan_CustomerChurnAnalysis.py
=======================================
Customer Churn Analysis — Complete Streamlit Dashboard
Author  : Aditya Chauhan
Project : Customer Churn Analysis
Run     : streamlit run AdityaChauhan_CustomerChurnAnalysis.py

All data loading, cleaning, analysis, ML, and UI code is in this single file.
"""

import warnings
warnings.filterwarnings("ignore")

# ============================================================
# IMPORTS
# ============================================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

# ============================================================
# SECTION 1 — CONSTANTS
# ============================================================

DATA_FILE = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

CATEGORICAL_COLS = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]

SERVICE_COLS = [
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies",
]

TENURE_BINS   = [0, 12, 24, 48, 72]
TENURE_LABELS = ["0-12 months", "13-24 months", "25-48 months", "49-72 months"]

COLORS = {
    "primary"   : "#4F46E5",
    "churn"     : "#EF4444",
    "retain"    : "#10B981",
    "neutral"   : "#6B7280",
    "text_light": "#F8FAFC",
}

# ============================================================
# SECTION 2 — DATA LOADING & CLEANING
# ============================================================

def load_data(filepath: str = DATA_FILE) -> pd.DataFrame:
    """
    Load the Telco Customer Churn CSV, fix known data-quality issues,
    and return a clean DataFrame ready for analysis.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: '{filepath}'. "
            "Please place 'WA_Fn-UseC_-Telco-Customer-Churn.csv' "
            "in the same folder and restart."
        )

    df = pd.read_csv(filepath)

    required = {"customerID", "Churn", "tenure", "MonthlyCharges", "TotalCharges"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(f"Dataset is missing expected columns: {missing_cols}")

    return _clean_data(df)


def _clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all cleaning steps and return the sanitised DataFrame."""
    df = df.copy()

    # TotalCharges: stored as object with blank strings for new customers (tenure=0)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # SeniorCitizen: convert 0/1 int to Yes/No string for consistency
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})

    # Binary churn flag for calculations
    df["ChurnFlag"] = (df["Churn"] == "Yes").astype(int)

    # Tenure cohort labels
    df["TenureGroup"] = pd.cut(
        df["tenure"],
        bins=TENURE_BINS,
        labels=TENURE_LABELS,
        right=True,
        include_lowest=True,
    )

    # Monthly charge tier
    df["ChargeTier"] = pd.cut(
        df["MonthlyCharges"],
        bins=[0, 35, 65, 100, df["MonthlyCharges"].max() + 1],
        labels=["Low (<$35)", "Medium ($35-$65)", "High ($65-$100)", "Very High (>$100)"],
        right=False,
    )

    return df

# ============================================================
# SECTION 3 — ANALYSIS FUNCTIONS
# ============================================================

def get_kpis(df: pd.DataFrame) -> dict:
    """Return a dict of headline KPI values."""
    total   = len(df)
    churned = df["ChurnFlag"].sum()
    return {
        "total_customers"        : int(total),
        "churned_customers"      : int(churned),
        "retained_customers"     : int(total - churned),
        "churn_rate"             : round(churned / total * 100, 2),
        "avg_monthly_charges"    : round(df["MonthlyCharges"].mean(), 2),
        "avg_tenure_months"      : round(df["tenure"].mean(), 1),
        "total_revenue"          : round(df["TotalCharges"].sum(), 2),
        "monthly_revenue_at_risk": round(
            df.loc[df["Churn"] == "Yes", "MonthlyCharges"].sum(), 2
        ),
    }


def churn_by_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Churn counts and rates for each unique value in col."""
    grp = (
        df.groupby(col)["ChurnFlag"]
        .agg(Total="count", Churned="sum")
        .reset_index()
    )
    grp["Retained"]  = grp["Total"] - grp["Churned"]
    grp["ChurnRate"] = (grp["Churned"] / grp["Total"] * 100).round(2)
    return grp.sort_values("ChurnRate", ascending=False)


def churn_by_services(df: pd.DataFrame) -> pd.DataFrame:
    """Churn rate per add-on service value."""
    rows = []
    for svc in SERVICE_COLS:
        for val in df[svc].dropna().unique():
            mask = df[svc] == val
            rows.append({
                "Service"  : svc,
                "Value"    : val,
                "ChurnRate": round(df.loc[mask, "ChurnFlag"].mean() * 100, 2),
                "Count"    : mask.sum(),
            })
    return pd.DataFrame(rows)


def tenure_churn_trend(df: pd.DataFrame) -> pd.DataFrame:
    """Average churn rate per tenure month for trend chart."""
    trend = (
        df.groupby("tenure")["ChurnFlag"]
        .mean()
        .reset_index()
        .rename(columns={"ChurnFlag": "ChurnRate"})
    )
    trend["ChurnRate"] = (trend["ChurnRate"] * 100).round(2)
    return trend


def revenue_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Revenue summary by contract type."""
    return (
        df.groupby("Contract")
        .agg(
            Customers   =("customerID", "count"),
            AvgMonthly  =("MonthlyCharges", "mean"),
            TotalRevenue=("TotalCharges", "sum"),
            ChurnRate   =("ChurnFlag", lambda x: round(x.mean() * 100, 2)),
        )
        .reset_index()
    )


def build_churn_model(df: pd.DataFrame):
    """
    Train a Logistic Regression pipeline to predict churn.
    Returns model, X_test, y_test, feature_names, metrics dict.
    """
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    )

    drop_cols  = ["customerID", "Churn", "ChurnFlag", "TenureGroup", "ChargeTier"]
    feature_df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    feature_df = pd.get_dummies(feature_df, drop_first=True)
    feature_names = feature_df.columns.tolist()

    X = feature_df.values
    y = df["ChurnFlag"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(max_iter=1000, random_state=42)),
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

    X_test_df = pd.DataFrame(X_test, columns=feature_names)
    return pipe, X_test_df, pd.Series(y_test), feature_names, metrics


def get_feature_importance(model, feature_names: list) -> pd.DataFrame:
    """Top 15 features by absolute logistic regression coefficient."""
    coef = model.named_steps["clf"].coef_[0]
    imp  = pd.DataFrame({"Feature": feature_names, "Coefficient": coef})
    imp["AbsCoef"] = imp["Coefficient"].abs()
    return imp.sort_values("AbsCoef", ascending=False).head(15)

# ============================================================
# SECTION 4 — STREAMLIT UI HELPERS
# ============================================================

def kpi_card(label: str, value: str, sub: str = "", border_color: str = COLORS["primary"]):
    st.markdown(
        f"""
        <div class="kpi-card" style="border-left-color:{border_color}">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            {"<div class='kpi-sub'>"+sub+"</div>" if sub else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(title: str):
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)


def insight_box(text: str):
    st.markdown(f'<div class="insight-box">💡 {text}</div>', unsafe_allow_html=True)


def plotly_defaults(fig, height: int = 420):
    """Apply consistent dark theme to every chart."""
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#CBD5E1", family="Inter, Segoe UI, sans-serif"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(gridcolor="#2d2d44", showline=False),
        yaxis=dict(gridcolor="#2d2d44", showline=False),
    )
    return fig


@st.cache_data(show_spinner="Loading dataset…")
def cached_load() -> pd.DataFrame:
    return load_data()


@st.cache_data(show_spinner="Training predictive model…")
def cached_model(_df_hash):
    df = cached_load()
    return build_churn_model(df)

# ============================================================
# SECTION 5 — PAGE CONFIGURATION & GLOBAL CSS
# ============================================================

st.set_page_config(
    page_title="Customer Churn Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    f"""
    <style>
    html, body, [class*="css"] {{ font-family: 'Inter', 'Segoe UI', sans-serif; }}
    .kpi-card {{
        background: linear-gradient(135deg, #1e1e2e 0%, #2d2d44 100%);
        border-radius: 12px; padding: 20px 24px;
        border-left: 4px solid {COLORS["primary"]}; margin-bottom: 8px;
    }}
    .kpi-label {{
        font-size: 0.78rem; color: #9CA3AF;
        text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 4px;
    }}
    .kpi-value  {{ font-size: 2rem; font-weight: 700; color: {COLORS["text_light"]}; line-height: 1; }}
    .kpi-sub    {{ font-size: 0.78rem; color: #9CA3AF; margin-top: 4px; }}
    .section-header {{
        font-size: 1.3rem; font-weight: 700; color: {COLORS["primary"]};
        border-bottom: 2px solid {COLORS["primary"]}; padding-bottom: 6px; margin: 24px 0 16px 0;
    }}
    .insight-box {{
        background: #1e293b; border-left: 4px solid {COLORS["primary"]};
        border-radius: 8px; padding: 12px 16px; color: #CBD5E1;
        font-size: 0.88rem; margin-top: 8px;
    }}
    section[data-testid="stSidebar"] {{ background: #12121e; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# SECTION 6 — SIDEBAR NAVIGATION & FILTERS
# ============================================================

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/combo-chart.png", width=64)
    st.title("📊 Customer Churn")
    st.caption("Telco Analytics Dashboard")
    st.divider()

    page = st.radio(
        "Navigation",
        ["🏠 Overview", "👥 Demographics", "📡 Services",
         "💰 Financials", "🤖 Predictive Model", "🗂️ Data Explorer"],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown("#### Filters")

# --- Load data
try:
    df_raw = cached_load()
except FileNotFoundError as e:
    st.error(str(e)); st.stop()
except ValueError as e:
    st.error(f"Data validation error: {e}"); st.stop()

# --- Sidebar filters
with st.sidebar:
    contracts    = ["All"] + sorted(df_raw["Contract"].unique().tolist())
    sel_contract = st.selectbox("Contract Type", contracts)

    genders      = ["All"] + sorted(df_raw["gender"].unique().tolist())
    sel_gender   = st.selectbox("Gender", genders)

    sel_senior   = st.selectbox("Senior Citizen", ["All", "Yes", "No"])

    internet_opts = ["All"] + sorted(df_raw["InternetService"].unique().tolist())
    sel_internet  = st.selectbox("Internet Service", internet_opts)

    t_min, t_max  = int(df_raw["tenure"].min()), int(df_raw["tenure"].max())
    sel_tenure    = st.slider("Tenure (months)", t_min, t_max, (t_min, t_max))

    mc_min = float(df_raw["MonthlyCharges"].min())
    mc_max = float(df_raw["MonthlyCharges"].max())
    sel_charges = st.slider("Monthly Charges ($)", mc_min, mc_max, (mc_min, mc_max), step=0.5)

    st.divider()
    st.caption(f"v1.0 · {len(df_raw):,} total records")

# --- Apply filters
df = df_raw.copy()
if sel_contract != "All": df = df[df["Contract"]        == sel_contract]
if sel_gender   != "All": df = df[df["gender"]          == sel_gender]
if sel_senior   != "All": df = df[df["SeniorCitizen"]   == sel_senior]
if sel_internet != "All": df = df[df["InternetService"] == sel_internet]
df = df[df["tenure"].between(*sel_tenure)]
df = df[df["MonthlyCharges"].between(*sel_charges)]

if df.empty:
    st.warning("⚠️ No records match the current filters. Please adjust the sidebar."); st.stop()

kpis = get_kpis(df)

# ============================================================
# SECTION 7 — PAGE: OVERVIEW
# ============================================================

if page == "🏠 Overview":
    st.title("📊 Customer Churn Analysis")
    st.markdown("A comprehensive analysis of customer churn for a telecom company. Use the sidebar filters to drill down.")

    section_header("Key Performance Indicators")
    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi_card("Total Customers",     f"{kpis['total_customers']:,}",       border_color=COLORS["primary"])
    with c2: kpi_card("Churned Customers",   f"{kpis['churned_customers']:,}",      f"Churn rate: {kpis['churn_rate']}%", COLORS["churn"])
    with c3: kpi_card("Avg Monthly Charges", f"${kpis['avg_monthly_charges']}",     border_color=COLORS["retain"])
    with c4: kpi_card("Avg Tenure",          f"{kpis['avg_tenure_months']} mo",     border_color=COLORS["neutral"])

    c5, c6, c7, c8 = st.columns(4)
    with c5: kpi_card("Retained Customers",      f"{kpis['retained_customers']:,}",                    border_color=COLORS["retain"])
    with c6: kpi_card("Monthly Revenue at Risk", f"${kpis['monthly_revenue_at_risk']:,.0f}", "from churned customers", COLORS["churn"])
    with c7: kpi_card("Total Revenue",           f"${kpis['total_revenue']:,.0f}",                     border_color=COLORS["primary"])
    with c8: kpi_card("Retention Rate",          f"{100 - kpis['churn_rate']:.1f}%",                   border_color=COLORS["retain"])

    section_header("Churn Overview")
    col_a, col_b = st.columns([1, 2])
    with col_a:
        fig_pie = px.pie(
            values=[kpis["churned_customers"], kpis["retained_customers"]],
            names=["Churned", "Retained"],
            color_discrete_sequence=[COLORS["churn"], COLORS["retain"]],
            hole=0.55, title="Churn Distribution",
        )
        fig_pie.update_traces(textposition="outside", textinfo="percent+label")
        st.plotly_chart(plotly_defaults(fig_pie, 380), use_container_width=True)
    with col_b:
        contract_data = churn_by_column(df, "Contract")
        fig_contract  = px.bar(
            contract_data, x="Contract", y=["Retained", "Churned"], barmode="stack",
            color_discrete_sequence=[COLORS["retain"], COLORS["churn"]],
            title="Customer Distribution by Contract Type", text_auto=True,
        )
        st.plotly_chart(plotly_defaults(fig_contract, 380), use_container_width=True)

    insight_box(
        f"**{kpis['churn_rate']}%** of customers have churned, representing "
        f"**${kpis['monthly_revenue_at_risk']:,.0f}** in monthly revenue at risk. "
        "Month-to-month contract customers churn at significantly higher rates."
    )

    section_header("Churn Rate by Tenure")
    trend = tenure_churn_trend(df)
    fig_trend = px.area(
        trend, x="tenure", y="ChurnRate",
        title="Churn Rate (%) Across Customer Lifetime (months)",
        labels={"tenure": "Tenure (months)", "ChurnRate": "Churn Rate (%)"},
        color_discrete_sequence=[COLORS["primary"]],
    )
    fig_trend.update_traces(line_width=2, fillcolor="rgba(79,70,229,0.15)")
    st.plotly_chart(plotly_defaults(fig_trend, 350), use_container_width=True)
    insight_box("New customers (tenure < 12 months) are the highest-risk segment. Churn drops sharply after the first year.")

# ============================================================
# SECTION 8 — PAGE: DEMOGRAPHICS
# ============================================================

elif page == "👥 Demographics":
    st.title("👥 Demographics Analysis")
    st.markdown("Explore how customer demographics relate to churn behavior.")

    section_header("Churn Rate by Demographic Group")
    demo_cols = {"gender": "Gender", "SeniorCitizen": "Senior Citizen",
                 "Partner": "Partner", "Dependents": "Dependents"}
    cols = st.columns(2)
    for idx, (col, label) in enumerate(demo_cols.items()):
        data = churn_by_column(df, col)
        fig  = px.bar(
            data, x=col, y="ChurnRate", color="ChurnRate",
            color_continuous_scale=["#10B981", "#F59E0B", "#EF4444"],
            title=f"Churn Rate by {label}",
            labels={col: label, "ChurnRate": "Churn Rate (%)"}, text="ChurnRate",
        )
        fig.update_traces(texttemplate="%{text}%", textposition="outside")
        fig.update_coloraxes(showscale=False)
        cols[idx % 2].plotly_chart(plotly_defaults(fig, 360), use_container_width=True)

    section_header("Demographic Interaction Heatmap")
    heat_data = (
        df.groupby(["SeniorCitizen", "Partner"])["ChurnFlag"].mean().reset_index()
        .pivot(index="SeniorCitizen", columns="Partner", values="ChurnFlag") * 100
    )
    fig_heat = px.imshow(heat_data, text_auto=".1f", color_continuous_scale="RdYlGn_r",
                         title="Churn Rate (%) — Senior Citizen × Partner Status",
                         labels={"color": "Churn Rate (%)"}, zmin=0, zmax=60)
    st.plotly_chart(plotly_defaults(fig_heat, 320), use_container_width=True)
    insight_box("Senior citizens churn at ~42% vs ~26% for non-seniors. Customers without a partner are more likely to churn.")

    section_header("Tenure Distribution by Churn Status")
    fig_box = px.box(df, x="Churn", y="tenure", color="Churn",
                     color_discrete_map={"Yes": COLORS["churn"], "No": COLORS["retain"]},
                     title="Tenure Distribution: Churned vs Retained",
                     labels={"tenure": "Tenure (months)", "Churn": "Churn Status"}, points="outliers")
    st.plotly_chart(plotly_defaults(fig_box, 380), use_container_width=True)

# ============================================================
# SECTION 9 — PAGE: SERVICES
# ============================================================

elif page == "📡 Services":
    st.title("📡 Services Analysis")
    st.markdown("Understand which services are associated with higher or lower churn.")

    section_header("Churn Rate by Internet Service Type")
    inet_data = churn_by_column(df, "InternetService")
    fig_inet  = px.bar(inet_data, x="InternetService", y="ChurnRate", color="ChurnRate",
                       color_continuous_scale=["#10B981", "#F59E0B", "#EF4444"],
                       title="Churn Rate by Internet Service", text="ChurnRate",
                       labels={"InternetService": "Internet Service", "ChurnRate": "Churn Rate (%)"})
    fig_inet.update_traces(texttemplate="%{text}%", textposition="outside")
    fig_inet.update_coloraxes(showscale=False)
    st.plotly_chart(plotly_defaults(fig_inet, 360), use_container_width=True)

    section_header("Add-On Services — Churn Rate Comparison")
    svc_data = churn_by_services(df)
    svc_yes  = svc_data[svc_data["Value"] == "Yes"]
    svc_no   = svc_data[svc_data["Value"] == "No"]
    fig_svc  = go.Figure()
    fig_svc.add_trace(go.Bar(name="Has Service", x=svc_yes["Service"], y=svc_yes["ChurnRate"],
                             marker_color=COLORS["primary"],
                             text=svc_yes["ChurnRate"].apply(lambda x: f"{x}%"), textposition="outside"))
    fig_svc.add_trace(go.Bar(name="No Service",  x=svc_no["Service"],  y=svc_no["ChurnRate"],
                             marker_color=COLORS["neutral"],
                             text=svc_no["ChurnRate"].apply(lambda x: f"{x}%"), textposition="outside"))
    fig_svc.update_layout(barmode="group", title="Churn Rate: With vs Without Each Add-On Service")
    st.plotly_chart(plotly_defaults(fig_svc, 420), use_container_width=True)
    insight_box("Customers without OnlineSecurity or TechSupport churn at nearly twice the rate of those who have them.")

    section_header("Churn Rate by Payment Method")
    pay_data = churn_by_column(df, "PaymentMethod")
    fig_pay  = px.bar(pay_data.sort_values("ChurnRate", ascending=True),
                      x="ChurnRate", y="PaymentMethod", orientation="h", color="ChurnRate",
                      color_continuous_scale=["#10B981", "#F59E0B", "#EF4444"],
                      title="Churn Rate by Payment Method", text="ChurnRate")
    fig_pay.update_traces(texttemplate="%{text}%", textposition="outside")
    fig_pay.update_coloraxes(showscale=False)
    st.plotly_chart(plotly_defaults(fig_pay, 360), use_container_width=True)
    insight_box("Electronic check users have the highest churn rate (~45%). Auto-payment customers are far more likely to stay.")

    section_header("Churn by Paperless Billing")
    pb_data = churn_by_column(df, "PaperlessBilling")
    fig_pb  = px.bar(pb_data, x="PaperlessBilling", y=["Retained", "Churned"], barmode="group",
                     color_discrete_sequence=[COLORS["retain"], COLORS["churn"]],
                     title="Paperless Billing vs Churn", text_auto=True)
    st.plotly_chart(plotly_defaults(fig_pb, 360), use_container_width=True)

# ============================================================
# SECTION 10 — PAGE: FINANCIALS
# ============================================================

elif page == "💰 Financials":
    st.title("💰 Financial Analysis")
    st.markdown("Examine revenue patterns and identify financial risk from churn.")

    section_header("Revenue by Contract Type")
    rev_df = revenue_analysis(df).copy()
    rev_df["AvgMonthly"]   = rev_df["AvgMonthly"].apply(lambda x: f"${x:,.2f}")
    rev_df["TotalRevenue"] = rev_df["TotalRevenue"].apply(lambda x: f"${x:,.0f}")
    rev_df["ChurnRate"]    = rev_df["ChurnRate"].apply(lambda x: f"{x}%")
    st.dataframe(rev_df, use_container_width=True, hide_index=True)

    section_header("Monthly Charges Distribution by Churn Status")
    fig_hist = px.histogram(df, x="MonthlyCharges", color="Churn", barmode="overlay", nbins=40,
                            color_discrete_map={"Yes": COLORS["churn"], "No": COLORS["retain"]},
                            opacity=0.75, title="Monthly Charges: Churned vs Retained",
                            labels={"MonthlyCharges": "Monthly Charges ($)"})
    st.plotly_chart(plotly_defaults(fig_hist, 420), use_container_width=True)
    insight_box("Churned customers have higher average monthly charges (~$74) vs retained (~$61).")

    section_header("Average Monthly Charges by Tenure Group")
    tenure_charges = df.groupby(["TenureGroup", "Churn"])["MonthlyCharges"].mean().reset_index()
    fig_tc = px.bar(tenure_charges, x="TenureGroup", y="MonthlyCharges", color="Churn", barmode="group",
                    color_discrete_map={"Yes": COLORS["churn"], "No": COLORS["retain"]},
                    title="Avg Monthly Charges by Tenure Group and Churn Status",
                    labels={"MonthlyCharges": "Avg Monthly Charges ($)"}, text_auto=".1f")
    st.plotly_chart(plotly_defaults(fig_tc, 400), use_container_width=True)

    section_header("Tenure vs Monthly Charges")
    fig_scatter = px.scatter(df.sample(min(len(df), 2000), random_state=42),
                             x="tenure", y="MonthlyCharges", color="Churn",
                             color_discrete_map={"Yes": COLORS["churn"], "No": COLORS["retain"]},
                             opacity=0.6, title="Tenure vs Monthly Charges (sample up to 2,000 customers)",
                             hover_data=["Contract", "InternetService", "PaymentMethod"])
    st.plotly_chart(plotly_defaults(fig_scatter, 450), use_container_width=True)

# ============================================================
# SECTION 11 — PAGE: PREDICTIVE MODEL
# ============================================================

elif page == "🤖 Predictive Model":
    st.title("🤖 Predictive Churn Model")
    st.markdown("A **Logistic Regression** model trained on all features to predict churn (evaluated on 20% held-out test set).")

    with st.spinner("Training model…"):
        try:
            df_hash = hash(tuple(df_raw.columns.tolist()) + (len(df_raw),))
            model, X_test, y_test, feature_names, metrics = cached_model(df_hash)
        except Exception as e:
            st.error(f"Model training failed: {e}"); st.stop()

    section_header("Model Performance Metrics")
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1: kpi_card("Accuracy",  f"{metrics['accuracy']}%",  border_color=COLORS["primary"])
    with m2: kpi_card("Precision", f"{metrics['precision']}%", border_color=COLORS["retain"])
    with m3: kpi_card("Recall",    f"{metrics['recall']}%",    border_color=COLORS["churn"])
    with m4: kpi_card("F1 Score",  f"{metrics['f1']}%",        border_color="#F59E0B")
    with m5: kpi_card("ROC-AUC",   f"{metrics['roc_auc']}%",  border_color=COLORS["neutral"])
    insight_box(f"ROC-AUC of **{metrics['roc_auc']}%** — the model distinguishes churners from non-churners correctly {metrics['roc_auc']}% of the time.")

    section_header("Confusion Matrix & Feature Importance")
    from sklearn.metrics import confusion_matrix
    y_pred = model.predict(X_test.values)
    cm     = confusion_matrix(y_test, y_pred)
    col_cm, col_fi = st.columns(2)
    fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                       x=["Predicted: No", "Predicted: Yes"], y=["Actual: No", "Actual: Yes"],
                       title="Confusion Matrix (Test Set)")
    col_cm.plotly_chart(plotly_defaults(fig_cm, 380), use_container_width=True)

    fi = get_feature_importance(model, feature_names)
    fig_fi = px.bar(fi.sort_values("AbsCoef", ascending=True), x="AbsCoef", y="Feature",
                    orientation="h", color="Coefficient", color_continuous_scale="RdBu",
                    title="Top 15 Most Influential Features")
    col_fi.plotly_chart(plotly_defaults(fig_fi, 380), use_container_width=True)

    section_header("ROC Curve")
    from sklearn.metrics import roc_curve
    y_proba     = model.predict_proba(X_test.values)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                 name=f"Logistic Regression (AUC={metrics['roc_auc']}%)",
                                 line=dict(color=COLORS["primary"], width=2),
                                 fill="tozeroy", fillcolor="rgba(79,70,229,0.15)"))
    fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Random Chance",
                                 line=dict(color=COLORS["neutral"], dash="dash")))
    fig_roc.update_layout(title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    st.plotly_chart(plotly_defaults(fig_roc, 420), use_container_width=True)

    section_header("🔮 Live Churn Predictor")
    st.markdown("Fill in a customer profile and get an instant churn probability.")
    with st.form("predict_form"):
        p1, p2, p3 = st.columns(3)
        with p1:
            p_gender     = st.selectbox("Gender",           ["Female", "Male"])
            p_senior     = st.selectbox("Senior Citizen",   ["No", "Yes"])
            p_partner    = st.selectbox("Partner",          ["No", "Yes"])
            p_dependents = st.selectbox("Dependents",       ["No", "Yes"])
            p_tenure     = st.slider("Tenure (months)", 0, 72, 12)
        with p2:
            p_phone      = st.selectbox("Phone Service",    ["No", "Yes"])
            p_multiline  = st.selectbox("Multiple Lines",   ["No", "Yes", "No phone service"])
            p_internet   = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            p_security   = st.selectbox("Online Security",  ["No", "Yes", "No internet service"])
            p_backup     = st.selectbox("Online Backup",    ["No", "Yes", "No internet service"])
        with p3:
            p_device      = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
            p_techsupport = st.selectbox("Tech Support",       ["No", "Yes", "No internet service"])
            p_streamingtv = st.selectbox("Streaming TV",       ["No", "Yes", "No internet service"])
            p_streamingmv = st.selectbox("Streaming Movies",   ["No", "Yes", "No internet service"])
            p_contract    = st.selectbox("Contract",           ["Month-to-month", "One year", "Two year"])
            p_paperless   = st.selectbox("Paperless Billing",  ["No", "Yes"])
            p_payment     = st.selectbox("Payment Method",     [
                "Electronic check", "Mailed check",
                "Bank transfer (automatic)", "Credit card (automatic)"])
            p_monthly = st.number_input("Monthly Charges ($)", 18.0, 120.0, 65.0, step=0.5)
            p_total   = st.number_input("Total Charges ($)", 0.0, 9000.0,
                                        float(p_monthly * p_tenure), step=1.0)
        submitted = st.form_submit_button("Predict Churn Probability", use_container_width=True)

    if submitted:
        try:
            input_row = pd.DataFrame([{
                "gender": p_gender, "SeniorCitizen": p_senior, "Partner": p_partner,
                "Dependents": p_dependents, "tenure": p_tenure, "PhoneService": p_phone,
                "MultipleLines": p_multiline, "InternetService": p_internet,
                "OnlineSecurity": p_security, "OnlineBackup": p_backup,
                "DeviceProtection": p_device, "TechSupport": p_techsupport,
                "StreamingTV": p_streamingtv, "StreamingMovies": p_streamingmv,
                "Contract": p_contract, "PaperlessBilling": p_paperless,
                "PaymentMethod": p_payment, "MonthlyCharges": p_monthly, "TotalCharges": p_total,
            }])
            input_encoded = pd.get_dummies(input_row)
            input_aligned = input_encoded.reindex(columns=feature_names, fill_value=0)
            prob  = model.predict_proba(input_aligned.values)[0][1] * 100
            color = COLORS["churn"] if prob >= 50 else COLORS["retain"]
            label = "HIGH RISK — Likely to Churn" if prob >= 50 else "LOW RISK — Likely to Stay"
            _, r2, _ = st.columns([1, 2, 1])
            r2.markdown(
                f"""<div style="background:#1e1e2e;border-radius:12px;padding:28px;text-align:center;
                border:2px solid {color};margin-top:16px;">
                <div style="font-size:3rem;font-weight:800;color:{color};">{prob:.1f}%</div>
                <div style="font-size:1rem;color:{color};margin-top:8px;font-weight:600;">{label}</div>
                <div style="color:#9CA3AF;font-size:0.82rem;margin-top:8px;">
                Predicted churn probability — Logistic Regression model</div></div>""",
                unsafe_allow_html=True,
            )
        except Exception as e:
            st.error(f"Prediction error: {e}")

# ============================================================
# SECTION 12 — PAGE: DATA EXPLORER
# ============================================================

elif page == "🗂️ Data Explorer":
    st.title("🗂️ Data Explorer")
    st.markdown("Browse, filter, and export the customer dataset.")

    section_header("Dataset Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows (filtered)", f"{len(df):,}")
    c2.metric("Columns",         len(df_raw.columns))
    c3.metric("Missing Values",  df_raw.isnull().sum().sum())
    c4.metric("Duplicate Rows",  df_raw.duplicated().sum())

    section_header("Data Quality Report")
    quality = pd.DataFrame({
        "Column"    : df_raw.columns,
        "Type"      : df_raw.dtypes.values.astype(str),
        "Non-Null"  : df_raw.notnull().sum().values,
        "Null Count": df_raw.isnull().sum().values,
        "Null %"    : (df_raw.isnull().sum().values / len(df_raw) * 100).round(2),
        "Unique"    : df_raw.nunique().values,
    })
    st.dataframe(quality, use_container_width=True, hide_index=True)

    section_header(f"Filtered Customer Records ({len(df):,} rows)")
    cols_to_show = st.multiselect(
        "Select columns to display",
        options=df_raw.columns.tolist(),
        default=["customerID", "gender", "SeniorCitizen", "tenure", "Contract",
                 "MonthlyCharges", "TotalCharges", "InternetService", "PaymentMethod", "Churn"],
    )
    if cols_to_show:
        display_df = df[cols_to_show].reset_index(drop=True)
        st.dataframe(display_df, use_container_width=True, height=400)
        st.download_button("⬇️ Download Filtered Data as CSV",
                           display_df.to_csv(index=False).encode("utf-8"),
                           "customer_churn_filtered.csv", "text/csv", use_container_width=True)
    else:
        st.info("Select at least one column to display the data table.")

    section_header("Descriptive Statistics")
    st.dataframe(df[["tenure", "MonthlyCharges", "TotalCharges"]].describe().T.round(2),
                 use_container_width=True)

    section_header("Correlation Matrix")
    corr = df[["tenure", "MonthlyCharges", "TotalCharges", "ChurnFlag"]].corr().round(3)
    fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu",
                         title="Pearson Correlation — Numeric Features", zmin=-1, zmax=1)
    st.plotly_chart(plotly_defaults(fig_corr, 380), use_container_width=True)
