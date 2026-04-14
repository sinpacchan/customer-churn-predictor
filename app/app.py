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