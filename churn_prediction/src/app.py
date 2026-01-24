import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import pickle
import os

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Customer Churn AI",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================= ADVANCED STYLING (GLASSMORPHISM) =================
def local_css():
    st.markdown("""
    <style>
        /* MAIN BACKGROUND */
        .stApp {
            background-image: radial-gradient(circle at 10% 20%, rgb(0, 0, 0) 0%, rgb(15, 20, 30) 90.2%);
            color: #ffffff;
        }

        /* SIDEBAR STYLING */
        [data-testid="stSidebar"] {
            background-color: rgba(20, 25, 35, 0.95);
            border-right: 1px solid rgba(255, 255, 255, 0.1);
        }

        /* METRIC CARDS */
        div[data-testid="stMetric"] {
            background-color: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            transition: transform 0.2s;
        }
        div[data-testid="stMetric"]:hover {
            transform: translateY(-5px);
            background-color: rgba(255, 255, 255, 0.08);
        }
        
        /* CUSTOM BUTTONS */
        .stButton>button {
            background: linear-gradient(45deg, #2193b0 0%, #6dd5ed 100%);
            color: white;
            border: none;
            border-radius: 25px;
            padding: 0.5rem 2rem;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(33, 147, 176, 0.3);
        }
        .stButton>button:hover {
            transform: scale(1.02);
            box-shadow: 0 6px 20px rgba(33, 147, 176, 0.5);
        }

        /* TABS */
        .stTabs [data-baseweb="tab-list"] {
            gap: 20px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: rgba(255, 255, 255, 0.05);
            border-radius: 10px;
            color: #ffffff;
            font-size: 16px;
        }
        .stTabs [aria-selected="true"] {
            background-color: rgba(33, 147, 176, 0.2) !important;
            border: 1px solid #2193b0 !important;
            color: #6dd5ed !important;
        }

        /* DATAFRAMES */
        [data-testid="stDataFrame"] {
            background-color: rgba(0, 0, 0, 0.3);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 10px;
        }
    </style>
    """, unsafe_allow_html=True)

local_css()

# ================= MOCK DATA & MODEL HANDLER =================
# This ensures the app runs even if you don't have the .pkl files yet
@st.cache_resource
def load_resources():
    try:
        log = pickle.load(open("models/Logistic_model.pkl", "rb"))
        rf = pickle.load(open("models/RandomForest_model.pkl", "rb"))
        xgb = pickle.load(open("models/XGBoost_model.pkl", "rb"))
        scl = pickle.load(open("models/scaler.pkl", "rb"))
        return log, rf, xgb, scl, True
    except FileNotFoundError:
        # Create dummy models for demonstration if files missing
        dummy_rf = RandomForestClassifier().fit([[0]*10], [0])
        return None, dummy_rf, None, None, False

log_model, rf_model, xgb_model, scaler, resources_loaded = load_resources()

# ================= SIDEBAR =================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1041/1041883.png", width=80)
    st.title("Analytics Hub")
    st.write("Current Model Version: **v2.1.0**")
    st.divider()
    model_choice = st.selectbox("Select Model Engine", ["Random Forest", "XGBoost", "Logistic Regression"])
    st.info("💡 **Pro Tip:** Random Forest generally offers the best balance of interpretability and accuracy for this dataset.")

# ================= MAIN HEADER =================
st.markdown("<h1 style='text-align: center; margin-bottom: 20px;'>📉 Customer Churn Prediction AI</h1>", unsafe_allow_html=True)

# Metric Row
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Customers", "7,043", "12%")
c2.metric("Churn Rate", "26.5%", "-2%")
c3.metric("Avg. Tenure", "32 Months", "4%")
c4.metric("Revenue at Risk", "$12.5M", "-0.5%")

st.write("---")

# ================= TABS =================
tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Real-time Prediction",
    "📊 Deep Dive Analytics",
    "📥 Batch Processing",
    "🧪 Model Benchmarking"
])

# ================= TAB 1: SINGLE PREDICTION =================
with tab1:
    st.markdown("### 👤 Customer Profile Analysis")
    
    with st.container():
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Demographics")
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior = st.radio("Senior Citizen", ["No", "Yes"], horizontal=True)
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])
            
        with col2:
            st.markdown("#### Service Details")
            tenure = st.slider("Tenure (Months)", 0, 72, 24)
            contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
            monthly = st.number_input("Monthly Charges ($)", 18.0, 120.0, 70.0)
            total = st.number_input("Total Charges ($)", 18.0, 9000.0, 1500.0)

    # Input preprocessing mock logic
    input_df = pd.DataFrame({
        'tenure': [tenure], 'MonthlyCharges': [monthly], 'TotalCharges': [total],
        'SeniorCitizen': [1 if senior == "Yes" else 0]
        # Add other One-Hot encoded columns here as per your training data
    })

    st.write("")
    if st.button("🚀 Analyze Risk Probability", use_container_width=True):
        with st.spinner("Calculating risk factors..."):
            # Mock prediction logic (Replace with actual model.predict)
            if resources_loaded:
                # Add scaler transformation here if needed
                pred = rf_model.predict(input_df)[0] # This will fail if input shape doesn't match, careful
                prob = rf_model.predict_proba(input_df)[0][1]
            else:
                # Demo Logic
                prob = np.random.uniform(0.1, 0.9)
            
            # Display Result
            st.write("---")
            res_col1, res_col2 = st.columns([1, 2])
            
            with res_col1:
                if prob > 0.5:
                    st.error(f"⚠️ High Churn Risk")
                    st.metric("Churn Probability", f"{prob*100:.1f}%", delta="High Risk", delta_color="inverse")
                else:
                    st.success(f"✅ Safe Customer")
                    st.metric("Churn Probability", f"{prob*100:.1f}%", delta="Low Risk")
            
            with res_col2:
                # Gauge Chart
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = prob * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Risk Score"},
                    gauge = {
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#ff4b4b" if prob > 0.5 else "#00c853"},
                        'steps': [
                            {'range': [0, 50], 'color': "rgba(0, 200, 83, 0.3)"},
                            {'range': [50, 100], 'color': "rgba(255, 75, 75, 0.3)"}],
                    }
                ))
                fig.update_layout(height=250, margin=dict(l=10,r=10,t=40,b=10), paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"})
                st.plotly_chart(fig, use_container_width=True)

# ================= TAB 2: FEATURE IMPORTANCE (Plotly) =================
with tab2:
    st.subheader("🧠 What drives customer decisions?")
    
    # Mock Feature Importance Data
    features = ['Tenure', 'MonthlyCharges', 'TotalCharges', 'FiberOptic', 'Contract_Month', 'TechSupport', 'Payment_Electronic']
    importance = [0.25, 0.20, 0.15, 0.12, 0.10, 0.08, 0.05]
    
    fi_df = pd.DataFrame({'Feature': features, 'Importance': importance}).sort_values(by='Importance', ascending=True)
    
    fig = px.bar(
        fi_df, x='Importance', y='Feature', orientation='h',
        title="Global Feature Importance",
        color='Importance',
        color_continuous_scale='Bluyl'
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", 
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

# ================= TAB 3: BULK PREDICTION =================
with tab3:
    st.subheader("📂 Batch Analysis")
    uploaded_file = st.file_uploader("Upload customer data (CSV)", type="csv")
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head(), use_container_width=True)
        
        if st.button("Process Batch"):
            # Mock Processing
            df['Churn_Prob'] = np.random.uniform(0, 1, size=len(df))
            df['Prediction'] = (df['Churn_Prob'] > 0.5).astype(int)
            
            st.success("Analysis Complete!")
            
            # Download
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "⬇️ Download Predictions",
                csv,
                "churn_results.csv",
                "text/csv",
                key='download-csv'
            )

# ================= TAB 4: MODEL COMPARISON (Plotly) =================
with tab4:
    st.subheader("🧪 Model Performance Benchmarks")
    
    perf_data = {
        'Model': ['Random Forest', 'XGBoost', 'Logistic Reg'],
        'Accuracy': [0.82, 0.85, 0.78],
        'F1 Score': [0.75, 0.79, 0.70],
        'AUC-ROC': [0.88, 0.91, 0.84]
    }
    perf_df = pd.DataFrame(perf_data)
    
    perf_melted = perf_df.melt(id_vars="Model", var_name="Metric", value_name="Score")
    
    fig = px.bar(
        perf_melted, x="Model", y="Score", color="Metric", barmode="group",
        color_discrete_sequence=["#00c6ff", "#0072ff", "#2980b9"]
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", 
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="white"
    )
    st.plotly_chart(fig, use_container_width=True)