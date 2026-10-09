# 📊 Customer Churn Analysis

A professional, interactive data analytics dashboard built with **Streamlit** and **Python** to explore and predict customer churn for a telecommunications company.

---

## 🎯 Problem Statement

Customer churn — customers cancelling their service — is a critical metric for subscription businesses. This project analyses the **IBM Telco Customer Churn dataset** (7,043 customers, 21 features) to:

- Understand the demographic, service, and financial drivers of churn.
- Identify high-risk customer segments.
- Predict individual churn probability using Machine Learning.
- Enable data-driven retention decisions.

---

## ✨ Features

| Page | What you'll find |
|---|---|
| 🏠 **Overview** | KPI cards, churn distribution pie chart, churn by contract, tenure trend |
| 👥 **Demographics** | Churn rates by gender, senior status, partner, dependents; interaction heatmap |
| 📡 **Services** | Churn by internet service, add-on services comparison, payment method, paperless billing |
| 💰 **Financials** | Revenue by contract, monthly charges distribution, tenure vs charges scatter |
| 🤖 **Predictive Model** | Logistic Regression metrics, confusion matrix, ROC curve, feature importance, **live predictor** |
| 🗂️ **Data Explorer** | Data quality report, filterable table, column selector, CSV export, correlation matrix |

### Sidebar Filters (global)
- Contract Type
- Gender
- Senior Citizen
- Internet Service
- Tenure range
- Monthly Charges range

---

## 🛠️ Technology Stack

| Layer | Technology |
|---|---|
| Frontend / Dashboard | Streamlit |
| Data Processing | Pandas, NumPy |
| Visualisation | Plotly Express, Plotly Graph Objects |
| Machine Learning | Scikit-learn (Logistic Regression) |
| Language | Python 3.10+ |

---

## 📁 Project Structure

```
customer_churn_analysis/
├── app.py                                  # Streamlit entry point
├── analysis.py                             # Data loading, cleaning & analysis functions
├── requirements.txt                        # Python dependencies
├── README.md                               # This file
└── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Dataset (IBM Telco)
```

---

## ⚡ Quick Start

### 1. Clone / download the repository

```bash
git clone <your-repo-url>
cd customer_churn_analysis
```

### 2. Create and activate a virtual environment (recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Ensure the dataset is present

Make sure `WA_Fn-UseC_-Telco-Customer-Churn.csv` is in the project root.  
Download from [IBM Sample Data Sets](https://www.ibm.com/communities/analytics/watson-analytics-blog/guide-to-sample-datasets/).

### 5. Run the application

```bash
streamlit run app.py
```

The dashboard will open automatically at **http://localhost:8501**.

---

## 📊 Dataset Details

| Property | Value |
|---|---|
| Source | IBM Watson Analytics — Telco Customer Churn |
| Rows | 7,043 customers |
| Columns | 21 |
| Target variable | `Churn` (Yes / No) |
| Known data issue | `TotalCharges` stored as string with blanks for new customers (tenure = 0) — handled automatically |
| Duplicates | 0 |

### Column Categories

- **Demographics:** `gender`, `SeniorCitizen`, `Partner`, `Dependents`
- **Account info:** `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod`
- **Services:** `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`
- **Financials:** `MonthlyCharges`, `TotalCharges`

---

## 🤖 Machine Learning

A **Logistic Regression** model is trained with:
- One-hot encoded categorical features
- StandardScaler normalisation
- 80/20 stratified train/test split

**Typical performance on the full dataset:**

| Metric | Score |
|---|---|
| Accuracy | ~81% |
| Precision | ~68% |
| Recall | ~57% |
| F1 Score | ~62% |
| ROC-AUC | ~85% |

> The live predictor on the Predictive Model page lets you fill in any customer profile and get an instant churn probability.

---

## 🔑 Key Insights

1. **Overall churn rate: ~26.5%** — significantly higher than industry average.
2. **Month-to-month contracts** churn at ~42% vs 11% (one year) and 3% (two year).
3. **New customers (0–12 months)** are the highest-risk cohort.
4. **Fiber optic** internet users churn at ~42% vs ~19% for DSL users.
5. **Electronic check** payment users churn at ~45% — highest of any payment method.
6. Customers **without** OnlineSecurity / TechSupport churn at nearly **2× the rate** of those with these add-ons.
7. **Senior citizens** churn at ~42% vs ~26% for non-seniors.

---

## 📝 License

This project is for educational and demonstration purposes.  
Dataset: © IBM Corporation — publicly available sample data.
