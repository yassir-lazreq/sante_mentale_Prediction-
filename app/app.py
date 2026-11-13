import streamlit as st
import pandas as pd
import joblib
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

st.set_page_config(page_title="Student Mental Health Risk", layout="wide")

st.title("🧠 Student Mental Health Risk Prediction")
st.markdown("""
**Pedagogical Prototype:** This tool predicts the mental health risk level (Low / Medium / High). 
*Disclaimer: This educational prototype must not be used for clinical diagnosis,
clinical screening, disciplinary decisions, student selection, exclusion, or
official individual assessment. It does not replace qualified professional support.*
""")

# Load Model and Metrics
model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'model.joblib')
metrics_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'metrics.json')
cm_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'confusion_matrix.json')

@st.cache_resource
def load_model():
    return joblib.load(model_path)

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Create Tabs
tab1, tab2 = st.tabs(["📊 Model Performance Dashboard", "🎯 Student Risk Prediction"])

# ==========================================
# TAB 1: PERFORMANCE DASHBOARD
# ==========================================
with tab1:
    st.header("Model Evaluation on Test Set (20%)")
    
    if os.path.exists(metrics_path) and os.path.exists(cm_path):
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        with open(cm_path, 'r') as f:
            cm = json.load(f)
            
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{metrics['accuracy']*100:.2f}%")
        col2.metric("Macro Precision", f"{metrics['precision_macro']*100:.2f}%")
        col3.metric("Macro Recall", f"{metrics['recall_macro']*100:.2f}%")
        col4.metric("Macro F1-Score", f"{metrics['f1_macro']*100:.2f}%")
        
        st.divider()
        st.subheader("Confusion Matrix")
        st.markdown("This matrix shows how the model performed across the 3 classes. The diagonal represents correct predictions.")
        
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Low', 'Medium', 'High'], 
                    yticklabels=['Low', 'Medium', 'High'], ax=ax)
        ax.set_ylabel('Actual Risk')
        ax.set_xlabel('Predicted Risk')
        st.pyplot(fig)
    else:
        st.warning("Metrics files not found. Please run `python src/evaluate.py` first.")

# ==========================================
# TAB 2: PREDICTION FORM
# ==========================================
with tab2:
    st.header("Student Details")
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("Age", min_value=17, max_value=30, value=20)
            gender = st.selectbox("Gender", ["Male", "Female", "Non-binary"])
            year = st.number_input("Year of Study", min_value=1, max_value=5, value=2)
            gpa = st.number_input("GPA (0.0 - 4.0)", min_value=0.0, max_value=4.0, value=3.0, step=0.1)
            
        with col2:
            sleep_hours = st.number_input("Sleep Hours (daily)", min_value=0.0, max_value=24.0, value=7.0, step=0.5)
            study_hours = st.number_input("Study Hours (weekly)", min_value=0.0, max_value=60.0, value=15.0)
            attendance = st.number_input("Attendance Rate (%)", min_value=0.0, max_value=100.0, value=85.0)
            social_media = st.number_input("Social Media Hours (daily)", min_value=0.0, max_value=24.0, value=3.0, step=0.5)
            
        with col3:
            screen_time = st.number_input("Total Screen Time (daily)", min_value=0.0, max_value=24.0, value=6.0, step=0.5)
            financial_stress = st.slider("Financial Stress (1-10)", min_value=1.0, max_value=10.0, value=5.0)
            family_support = st.slider("Family Support (1-10)", min_value=1.0, max_value=10.0, value=5.0)
            physical_activity = st.selectbox("Physical Activity", ["Low", "Moderate", "High"])
            diet_quality = st.selectbox("Diet Quality", ["Poor", "Average", "Good"])
            part_time_job = st.selectbox("Part-Time Job", ["Yes", "No"])
            history = st.selectbox("Previous Mental Health History", ["Yes", "No"])
            
        submit_button = st.form_submit_button(label="Predict Risk Level")

    if submit_button:
        input_data = pd.DataFrame([{
            'Student_ID': 'STU9999',
            'Age': age,
            'Gender': gender,
            'Year_of_Study': year,
            'Sleep_Hours': sleep_hours,
            'Study_Hours_Per_Week': study_hours,
            'GPA': gpa,
            'Social_Media_Hours': social_media,
            'Screen_Time_Hours': screen_time,
            'Physical_Activity': physical_activity,
            'Financial_Stress': financial_stress,
            'Family_Support': family_support,
            'Attendance_Rate': attendance,
            'Part_Time_Job': part_time_job,
            'Diet_Quality': diet_quality,
            'Previous_Mental_Health_History': history
        }])
        
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]
        classes = model.classes_
        
        st.divider()
        st.subheader("Prediction Result")
        
        if prediction == "High":
            st.error(f"**Predicted Risk: {prediction}** 🚨")
        elif prediction == "Medium":
            st.warning(f"**Predicted Risk: {prediction}** ⚠️")
        else:
            st.success(f"**Predicted Risk: {prediction}** ✅")
            
        prob_df = pd.DataFrame({'Risk Level': classes, 'Probability': probabilities})
        st.write("**Confidence Levels:**")
        st.dataframe(prob_df.set_index('Risk Level').T, use_container_width=True)