import joblib
import pandas as pd

# Load model once
model = joblib.load("models/best_model.pkl")

def predict_churn(input_data):
    """
    Predict churn probability for new customer data.

    input_data: dict or pandas DataFrame
    """

    # Convert dict to DataFrame
    if isinstance(input_data, dict):
        input_data = pd.DataFrame([input_data])

    # Ensure all expected columns exist
    expected_cols = model.feature_names_in_

    for col in expected_cols:
        if col not in input_data.columns:
            input_data[col] = 0  # default fallback

    # Ensure correct column order
    input_data = input_data[expected_cols]

    # Predict probability
    proba = model.predict_proba(input_data)[:, 1]

    return proba

# 🔥 Example usage
if __name__ == "__main__":

    sample = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 80.0,
        "TotalCharges": 80.0
    }

    result = predict_churn(sample)[0]

    print("===== CHURN PREDICTION =====")
    print(f"Probability of churn: {result:.3f}")

    if result > 0.3:
        print("⚠️ High risk of churn")
    else:
        print("✅ Low risk of churn")

# ===== CHURN PREDICTION =====
# Probability of churn: 0.822
# ⚠️ High risk of churn