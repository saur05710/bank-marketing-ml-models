# File: project-folder/app.py
import sys
import subprocess

# Automatically check and install missing dependencies from requirements.txt
def install_requirements():
    try:
        import streamlit
        import pandas
        import sklearn
        import matplotlib
        import seaborn
        import joblib
    except ImportError:
        print("Installing missing dependencies from requirements.txt...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

install_requirements()

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

st.set_page_config(
    page_title="Bank Marketing Term Deposit Predictor",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 Bank Marketing Prediction Dashboard")
st.markdown("Evaluate term deposit subscription models trained on the UCI Bank Marketing Dataset (20,000 instances).")

# Sidebar
st.sidebar.header("1. Model Selection")
model_options = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "kNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest (Ensemble)": "random_forest.pkl"
}

selected_model_name = st.sidebar.selectbox("Choose Permitted ML Model", list(model_options.keys()))

st.sidebar.header("2. Dataset Input")
uploaded_file = st.sidebar.file_uploader("Upload Test Data (CSV)", type=["csv"])

@st.cache_resource
def load_artifacts():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        current_dir = os.getcwd()

    base_path = os.path.join(current_dir, "model")
    preprocessor_path = os.path.join(base_path, "preprocessor.pkl")
    
    if not os.path.exists(preprocessor_path):
        st.error(f"Preprocessor artifact missing at {preprocessor_path}. Run train_models.py first.")
        st.stop()
        
    preprocessor = joblib.load(preprocessor_path)
    loaded_models = {}
    for display_name, file_name in model_options.items():
        m_path = os.path.join(base_path, file_name)
        if os.path.exists(m_path):
            loaded_models[display_name] = joblib.load(m_path)
    return preprocessor, loaded_models

preprocessor, models_dict = load_artifacts()

# Environment-safe path resolution for test_data.csv
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    current_dir = os.getcwd()

default_path = os.path.join(current_dir, "test_data.csv")

if uploaded_file is not None:
    sep = ";" if ";" in uploaded_file.getvalue().decode("utf-8", errors="ignore")[:500] else ","
    uploaded_file.seek(0)
    df_test = pd.read_csv(uploaded_file, sep=sep)
    st.sidebar.success("Custom CSV file uploaded successfully!")
elif os.path.exists(default_path):
    df_test = pd.read_csv(default_path)
    st.sidebar.info("Using default test_data.csv")
else:
    st.error("No dataset uploaded and default test_data.csv is missing.")
    st.stop()

st.subheader("Data Preview")
st.dataframe(df_test.head(10), use_container_width=True)

has_target = 'y' in df_test.columns
if has_target:
    X_raw = df_test.drop(columns=['y'])
    y_true = (df_test['y'].astype(str).str.strip().str.lower() == 'yes').astype(int)
else:
    X_raw = df_test.copy()
    y_true = None

try:
    X_transformed = preprocessor.transform(X_raw)
    current_model = models_dict[selected_model_name]
    y_pred = current_model.predict(X_transformed)
    y_prob = current_model.predict_proba(X_transformed)[:, 1] if hasattr(current_model, "predict_proba") else y_pred
except Exception as e:
    st.error(f"Error executing feature preprocessing or model prediction: {str(e)}")
    st.stop()

if has_target and y_true is not None:
    st.markdown("---")
    st.subheader(f"📊 Performance Metrics: {selected_model_name}")

    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Accuracy", f"{acc:.4f}")
    c2.metric("AUC Score", f"{auc:.4f}")
    c3.metric("Precision", f"{prec:.4f}")
    c4.metric("Recall", f"{rec:.4f}")
    c5.metric("F1 Score", f"{f1:.4f}")
    c6.metric("MCC Score", f"{mcc:.4f}")

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### Confusion Matrix")
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['No Subscription (0)', 'Subscribed (1)'],
                    yticklabels=['No Subscription (0)', 'Subscribed (1)'])
        ax.set_ylabel('Actual')
        ax.set_xlabel('Predicted')
        st.pyplot(fig)

    with col_right:
        st.markdown("### Classification Report")
        report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        report_df = pd.DataFrame(report_dict).transpose()
        st.dataframe(report_df.style.format("{:.4f}"), use_container_width=True)

    st.markdown("---")
    st.subheader("📋 Comparative Evaluation of All 5 Permitted Models")
    
    comp_results = []
    for m_name, m_obj in models_dict.items():
        m_pred = m_obj.predict(X_transformed)
        m_prob = m_obj.predict_proba(X_transformed)[:, 1] if hasattr(m_obj, "predict_proba") else m_pred
        
        comp_results.append({
            'ML Model Name': m_name,
            'Accuracy': round(accuracy_score(y_true, m_pred), 4),
            'AUC': round(roc_auc_score(y_true, m_prob), 4),
            'Precision': round(precision_score(y_true, m_pred, zero_division=0), 4),
            'Recall': round(recall_score(y_true, m_pred, zero_division=0), 4),
            'F1': round(f1_score(y_true, m_pred, zero_division=0), 4),
            'MCC': round(matthews_corrcoef(y_true, m_pred), 4)
        })
    
    comp_df = pd.DataFrame(comp_results)
    st.dataframe(comp_df, use_container_width=True)

else:
    st.subheader("Inference Results (No Ground Truth Target Column Found)")
    df_results = df_test.copy()
    df_results['Predicted_Subscription'] = y_pred
    df_results['Subscription_Probability'] = np.round(y_prob, 4)
    st.dataframe(df_results, use_container_width=True)