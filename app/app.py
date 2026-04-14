import streamlit as st
import pandas as pd
import joblib
import shap
import numpy as np
import matplotlib.pyplot as plt

# =========================
# ⚙️ CONFIG
# =========================

st.set_page_config(page_title="Customer Churn Intelligence", layout="wide")

# =========================
# LOAD MODEL
# =========================

model = joblib.load("models/best_model.pkl")
preprocessor = model.named_steps["preprocessor"]
classifier = model.named_steps["classifier"]
explainer = shap.TreeExplainer(classifier)

# =========================
# 🧠 HERO SECTION
# =========================

st.title("📊 Customer Churn Intelligence")

st.markdown("""
### 🧠 Predict • Understand • Act

This system identifies customers likely to leave within the next 30 days.

It combines machine learning and explainability to not only predict churn, 
but to **understand why it happens**.

👉 Use this dashboard to:
- Predict churn for individual customers  
- Analyze risk across customer segments  
- Discover the key drivers behind churn  
""")

st.divider()

# =========================
# 📂 DATASET VIEWER
# =========================

st.header("📂 Explore Your Data")
st.caption("Upload your dataset to explore customer information.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head())

st.divider()

# =========================
# 🎯 SINGLE PREDICTION
# =========================

st.header("🎯 Predict Individual Customer Risk")
st.caption("Simulate a customer and estimate churn probability.")

with st.form("prediction_form"):

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
        Partner = st.selectbox("Partner", ["Yes", "No"])
        Dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)

    with col2:
        MonthlyCharges = st.slider("Monthly Charges", 0.0, 150.0, 70.0)
        Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        PaymentMethod = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
        )

    submit = st.form_submit_button("Predict")

if submit:

    input_data = {
        "gender": gender,
        "SeniorCitizen": SeniorCitizen,
        "Partner": Partner,
        "Dependents": Dependents,
        "tenure": tenure,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Contract": Contract,
        "PaperlessBilling": "Yes",
        "PaymentMethod": PaymentMethod,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": MonthlyCharges * max(tenure, 1)
    }

    input_df = pd.DataFrame([input_data])
    input_df = input_df[model.feature_names_in_]

    proba = model.predict_proba(input_df)[0][1]

    st.subheader("📊 Prediction Result")
    st.metric("Churn Probability", f"{proba:.2%}")

    if proba > 0.6:
        st.error("🔴 High Risk – Immediate action recommended")
    elif proba > 0.3:
        st.warning("🟡 Medium Risk – Monitor closely")
    else:
        st.success("🟢 Low Risk – Likely to stay")

st.divider()

# =========================
# 📊 RISK DASHBOARD
# =========================

st.header("📊 Customer Risk Intelligence Dashboard")
st.caption("Analyze churn risk across multiple customers.")

uploaded_file_bulk = st.file_uploader(
    "Upload dataset for batch analysis",
    type=["csv"],
    key="bulk"
)

if uploaded_file_bulk:
    df_bulk = pd.read_csv(uploaded_file_bulk)
    st.dataframe(df_bulk.head())

    df_bulk = df_bulk[model.feature_names_in_]

    probs = model.predict_proba(df_bulk)[:, 1]
    df_bulk["churn_probability"] = probs

    # Risk classification
    def risk_level(p):
        if p < 0.3:
            return "Low"
        elif p < 0.6:
            return "Medium"
        else:
            return "High"

    df_bulk["risk_level"] = df_bulk["churn_probability"].apply(risk_level)

    # =========================
    # 📈 DISTRIBUTION
    # =========================

    st.subheader("📈 Risk Distribution")
    st.bar_chart(df_bulk["risk_level"].value_counts())

    # =========================
    # 📋 TABLE
    # =========================

    def color_risk(val):
        if val == "High":
            return "background-color: #ff4d4d"
        elif val == "Medium":
            return "background-color: #ffd966"
        else:
            return "background-color: #66ff66"

    st.subheader("📋 Customer Risk Table")
    st.dataframe(df_bulk.style.applymap(color_risk, subset=["risk_level"]))

    # =========================
    # 🔥 HIGH RISK
    # =========================

    st.subheader("🔥 High Risk Customers")
    st.dataframe(df_bulk[df_bulk["risk_level"] == "High"].sort_values(by="churn_probability", ascending=False))

    st.divider()

    # =========================
    # 🔍 SHAP EXPLANATION
    # =========================

    st.header("🔍 Explain Customer Risk")
    st.caption("Understand why a customer is predicted to churn.")

    selected_index = st.selectbox("Select customer", df_bulk.index)
    selected_customer = df_bulk.loc[[selected_index]]

    X_processed = preprocessor.transform(selected_customer)

    if hasattr(X_processed, "toarray"):
        X_processed = X_processed.toarray()

    shap_values = explainer.shap_values(X_processed)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    values = shap_values[0]
    feature_names = preprocessor.get_feature_names_out()

    # =========================
    # 💡 TEXT INSIGHT
    # =========================

    st.subheader("💡 Key Drivers")

    top_idx = np.argsort(np.abs(values))[-5:]

    for i in reversed(top_idx):
        feature = feature_names[i]
        impact = values[i]

        direction = "increases" if impact > 0 else "reduces"
        color = "🔴" if impact > 0 else "🟢"

        st.write(f"{color} **{feature}** {direction} churn risk")

    # =========================
    # 📈 SHAP PLOT
    # =========================

    st.subheader("📈 SHAP Explanation")

    base_value = explainer.expected_value
    if isinstance(base_value, list):
        base_value = base_value[1]

    fig, ax = plt.subplots()

    shap.plots.waterfall(
        shap.Explanation(
            values=values,
            base_values=base_value,
            data=X_processed[0],
            feature_names=feature_names
        ),
        show=False
    )

    st.pyplot(fig)

# =========================
# 🌙 FINAL INSIGHT
# =========================

st.divider()

st.markdown("""
### 💡 Insight

Churn is rarely random.

It is often driven by:
- Pricing pressure  
- Contract flexibility  
- Lack of engagement  

This system transforms raw data into **actionable retention strategy**.
""")