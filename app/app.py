import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("models/best_model.pkl")

st.set_page_config(page_title="Customer Churn Predictor", layout="wide")

st.title("📊 Customer Churn Dashboard")

# =========================
# 📂 DATASET VIEWER
# =========================

st.header("📂 Dataset Viewer")

uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Preview of dataset:")
    st.dataframe(df.head())

# =========================
# 🎯 PREDICTION FORM
# =========================

st.header("🎯 Predict Customer Churn")

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

# =========================
# 🧠 PREDICTION LOGIC
# =========================

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
        "TotalCharges": MonthlyCharges * tenure
    }

    input_df = pd.DataFrame([input_data])

    proba = model.predict_proba(input_df)[0][1]

    st.subheader("📊 Prediction Result")

    st.write(f"Churn Probability: **{proba:.2f}**")

    if proba > 0.3:
        st.error("⚠️ High risk of churn")
    else:
        st.success("✅ Low risk of churn")

st.header("📊 Customer Risk Dashboard")

uploaded_file_bulk = st.file_uploader(
    "Upload customer dataset for risk analysis",
    type=["csv"],
    key="bulk"
)

if uploaded_file_bulk:
    df_bulk = pd.read_csv(uploaded_file_bulk)

    st.write("Preview:")
    st.dataframe(df_bulk.head())

    # Ensure correct columns
    df_bulk = df_bulk[model.feature_names_in_]

    # Predict probabilities
    probs = model.predict_proba(df_bulk)[:, 1]
    df_bulk["churn_probability"] = probs

    # Risk levels
    def risk_level(p):
        if p < 0.3:
            return "Low"
        elif p < 0.6:
            return "Medium"
        else:
            return "High"

    df_bulk["risk_level"] = df_bulk["churn_probability"].apply(risk_level)

    # Color styling
    def color_risk(val):
        if val == "High":
            return "background-color: #ff4d4d"
        elif val == "Medium":
            return "background-color: #ffd966"
        else:
            return "background-color: #66ff66"

    st.subheader("📋 Customer Risk Table")

    styled_df = df_bulk.style.applymap(color_risk, subset=["risk_level"])
    st.dataframe(styled_df)

    # High risk customers
    st.subheader("🔥 High Risk Customers")

    high_risk = df_bulk[df_bulk["risk_level"] == "High"]
    st.dataframe(high_risk.sort_values(by="churn_probability", ascending=False))