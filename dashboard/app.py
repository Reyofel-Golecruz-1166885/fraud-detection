import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# Load model and scaler
model = joblib.load('../models/fraud_model_tuned.joblib')
scaler = joblib.load('../models/scaler.joblib')

# Page config
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Credit Card Fraud Detection")
st.markdown("Enter transaction details to check if it is fraudulent.")

# ── Sidebar inputs ──────────────────────────────────────────
st.sidebar.header("Transaction Details")

Amount = st.sidebar.number_input("Amount ($)", min_value=0.0, value=149.62)
Time = st.sidebar.number_input("Time (seconds)", min_value=0.0, value=406.0)

st.sidebar.markdown("---")
st.sidebar.markdown("**V Features (PCA Components)**")

v_values = {}
for i in range(1, 29):
    v_values[f'V{i}'] = st.sidebar.number_input(
        f"V{i}", value=0.0, format="%.4f"
    )

# ── Predict button ───────────────────────────────────────────
if st.button("🔎 Analyse Transaction", use_container_width=True):

    # Build input dataframe
    input_data = {'Time': Time, 'Amount': Amount}
    input_data.update(v_values)
    df = pd.DataFrame([input_data])

    # Scale Amount and Time
    df[['Amount', 'Time']] = scaler.transform(df[['Amount', 'Time']])

    # Reorder columns
    feature_cols = ['Time','V1','V2','V3','V4','V5','V6','V7','V8','V9',
                    'V10','V11','V12','V13','V14','V15','V16','V17','V18',
                    'V19','V20','V21','V22','V23','V24','V25','V26','V27',
                    'V28','Amount']
    df = df[feature_cols]

    # Predict
    probability = model.predict_proba(df)[0][1]
    prediction = int(probability >= 0.5)

    # ── Results ──────────────────────────────────────────────
    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        if prediction == 1:
            st.error("🚨 FRAUD DETECTED")
        else:
            st.success("✅ LEGITIMATE")

    with col2:
        st.metric("Fraud Probability", f"{probability:.2%}")

    with col3:
        confidence = "HIGH" if probability > 0.8 or probability < 0.2 else "MEDIUM"
        st.metric("Confidence", confidence)

    # ── SHAP explanation ─────────────────────────────────────
    st.markdown("---")
    st.subheader("Why did the model make this prediction?")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df)

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            data=df.iloc[0],
            feature_names=feature_cols
        ),
        show=False
    )
    st.pyplot(fig)
    plt.close()