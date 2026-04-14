# 📊 Customer Churn Predictor

This project builds an end-to-end machine learning system to predict customer churn using telecom customer data.

---

# 🎯 Goal

Identify customers who are likely to leave so companies can take proactive retention actions.

The project focuses on building a **production-like ML workflow**, from data exploration to deployment-ready prediction and interactive dashboards.

---

# 🧠 Key Features

- 📊 Exploratory Data Analysis (EDA)
- 🧹 Data preprocessing pipeline (scikit-learn)
- 🤖 Multiple models:
  - Logistic Regression (baseline)
  - Random Forest
  - XGBoost (best performing)
- 🔧 Hyperparameter tuning (GridSearchCV)
- 📈 Model evaluation using:
  - F1-score (primary metric)
  - ROC-AUC
- 🔍 Model explainability with SHAP
- ⚙️ Model packaging with reusable prediction function
- 🌐 Interactive Streamlit dashboard:
  - Single customer prediction
  - Dataset upload & preview
  - Risk dashboard with color-coded segmentation

---

# 🏆 Final Model Performance

| Model           | F1 Score | ROC-AUC |
|----------------|--------|--------|
| Random Forest  | 0.579  | 0.826  |
| XGBoost        | **0.585**  | **0.847**  |

👉 XGBoost was selected as the final model.

---

# 🔍 Model Insights (SHAP)

Key drivers of churn:

- Contract type (month-to-month)
- Tenure (shorter → higher churn)
- Monthly charges (higher → higher churn)
- Lack of tech support / online security
- Payment method (electronic check)

These insights align with real-world customer behavior.

---

# 📊 Streamlit Dashboard

The project includes an interactive dashboard for:

### 🎯 Single Prediction
- Input customer data manually
- Get churn probability instantly

### 📂 Dataset Viewer
- Upload CSV files
- Preview customer data

### 📊 Risk Dashboard
- Batch prediction for multiple customers
- Color-coded segmentation:
  - 🟢 Low risk
  - 🟡 Medium risk
  - 🔴 High risk
- Identify high-risk customers instantly

---

# 🧰 Tech Stack

- Python
- pandas
- scikit-learn
- XGBoost
- SHAP
- Streamlit

---

# 📊 Dataset

Telco Customer Churn dataset (Kaggle)

---

# 📁 Project Structure

- customer-churn-predictor/
- │
- ├── app/ # Streamlit dashboard
- ├── src/ # prediction logic
- ├── models/ # trained models
- ├── data/ # dataset
- ├── notebooks/ # EDA, training, experiments
- └── README.md


---

# 🚀 How to Run

### 1. Install dependencies
### 2. Run Streamlit app (streamlit run app/app.py)

---

# 💡 Key Learnings

- Building end-to-end ML pipelines
- Handling real-world data issues
- Model evaluation beyond accuracy
- Interpreting models with SHAP
- Turning ML models into interactive applications

---

# 🎯 Future Improvements

- Deploy app online (Streamlit Cloud)
- Add SHAP visualizations to dashboard
- Improve feature engineering
- Add API (FastAPI / Flask)

---

# 📌 Summary

This project demonstrates how to go from raw data to a **deployable, explainable machine learning application**, combining modeling, interpretation, and user-facing tools.