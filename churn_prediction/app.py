# ===================== PATH SETUP =====================
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# ===================== STREAMLIT CONFIG (FIRST) =====================
import streamlit as st

st.set_page_config(
    page_title="Customer Churn Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

# ===================== IMPORTS =====================
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# ===================== LOAD MODELS =====================
log_model = pickle.load(
    open(os.path.join(BASE_DIR, "models", "Logistic_model.pkl"), "rb")
)

rf_model = pickle.load(
    open(os.path.join(BASE_DIR, "models", "RandomForest_model.pkl"), "rb")
)

xgb_model = pickle.load(
    open(os.path.join(BASE_DIR, "models", "XGBoost_model.pkl"), "rb")
)

scaler = pickle.load(
    open(os.path.join(BASE_DIR, "models", "scaler.pkl"), "rb")
)

# ===================== LOAD DATA =====================
from src.preprocess import preprocess_data

X_train, X_test, y_train, y_test, _ = preprocess_data(
    os.path.join(BASE_DIR, "data", "churn.csv")
)

# ===================== MODEL SELECTION =====================
model_dict = {
    "Logistic Regression": log_model,
    "Random Forest": rf_model,
    "XGBoost": xgb_model
}

roc_scores = {}
for name, m in model_dict.items():
    probs = m.predict_proba(X_test)[:, 1]
    roc_scores[name] = roc_auc_score(y_test, probs)

best_model_name = max(roc_scores, key=roc_scores.get)
model = model_dict[best_model_name]

# ===================== HEADER =====================
st.title("📉 Customer Churn Analytics Dashboard")
st.write("End-to-end ML system with prediction, analytics, and explainability")

st.success(
    f"🏆 Auto-selected Best Model: **{best_model_name}** "
    f"(ROC-AUC = {roc_scores[best_model_name]:.2f})"
)

# ===================== TABS =====================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔮 Single Prediction",
    "📈 Feature Importance",
    "📥 Bulk Prediction",
    "🧪 Model Comparison",
    "🔎 Explainability (SHAP)"
])

# ===================== TAB 1: SINGLE PREDICTION =====================
with tab1:
    st.subheader("🔍 Predict Churn for a Single Customer")

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        phone = st.selectbox("Phone Service", ["Yes", "No"])

    with col2:
        tenure = st.slider("Tenure (Months)", 1, 72, 12)
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        monthly = st.number_input("Monthly Charges", 20.0, 120.0, 50.0)
        total = st.number_input("Total Charges", 20.0, 8000.0, 500.0)

    input_data = pd.DataFrame({
        "SeniorCitizen": [senior],
        "tenure": [tenure],
        "MonthlyCharges": [monthly],
        "TotalCharges": [total],
        "gender_Male": [1 if gender == "Male" else 0],
        "Partner_Yes": [1 if partner == "Yes" else 0],
        "Dependents_Yes": [1 if dependents == "Yes" else 0],
        "PhoneService_Yes": [1 if phone == "Yes" else 0],
        "InternetService_Fiber optic": [1 if internet == "Fiber optic" else 0],
        "InternetService_No": [1 if internet == "No" else 0]
    })

    if st.button("🚀 Predict Churn"):
        scaled = scaler.transform(input_data)
        prob = model.predict_proba(scaled)[0][1]

        if prob >= 0.5:
            st.error(f"⚠️ High Risk of Churn ({prob*100:.2f}%)")
        else:
            st.success(f"✅ Low Risk of Churn ({prob*100:.2f}%)")

# ===================== TAB 2: FEATURE IMPORTANCE =====================
with tab2:
    st.subheader("📈 Feature Importance (Random Forest)")

    imp_df = pd.DataFrame({
        "Feature": input_data.columns,
        "Importance": rf_model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    st.dataframe(imp_df)

    fig, ax = plt.subplots()
    ax.barh(imp_df["Feature"], imp_df["Importance"])
    ax.invert_yaxis()
    st.pyplot(fig)

# ===================== TAB 3: BULK PREDICTION =====================
with tab3:
    st.subheader("📥 Bulk Prediction")

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        df = pd.read_csv(file)
        df_encoded = pd.get_dummies(df, drop_first=True)
        df_encoded = df_encoded.reindex(columns=input_data.columns, fill_value=0)

        df_scaled = scaler.transform(df_encoded)
        df["Churn_Prediction"] = model.predict(df_scaled)
        df["Churn_Probability"] = model.predict_proba(df_scaled)[:, 1]

        st.dataframe(df.head())

# ===================== TAB 4: MODEL COMPARISON =====================
with tab4:
    results = []

    for name, m in model_dict.items():
        y_pred = m.predict(X_test)
        y_prob = m.predict_proba(X_test)[:, 1]

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "F1-Score": f1_score(y_test, y_pred),
            "ROC-AUC": roc_auc_score(y_test, y_prob)
        })

    results_df = pd.DataFrame(results)
    st.dataframe(results_df)
    st.bar_chart(results_df.set_index("Model")["ROC-AUC"])

# ===================== TAB 5: SHAP =====================
with tab5:
    st.subheader("🔎 Explainability (SHAP)")

    if best_model_name in ["Random Forest", "XGBoost"]:
        import shap

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        shap_df = pd.DataFrame(
            np.abs(shap_values[1]).mean(axis=0),
            index=X_test.columns,
            columns=["Mean |SHAP Value|"]
        ).sort_values(by="Mean |SHAP Value|", ascending=False)

        st.dataframe(shap_df)
        st.bar_chart(shap_df)

    else:
        st.info("SHAP works best with tree-based models.")

# ===================== FOOTER =====================
st.write("---")
st.markdown(
    "**Developed by:** Vicky Kumar Singh  \n"
    "**Tech Stack:** Python · Scikit-learn · XGBoost · SHAP · Streamlit"
)
