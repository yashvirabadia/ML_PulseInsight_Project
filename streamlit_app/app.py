import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import os

# Page configuration
st.set_page_config(
    page_title="PulseInsight",
    page_icon="❤️",
    layout="wide"
)

# Load model and scaler
@st.cache_resource
def load_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, "best_model.pkl")
    scaler_path = os.path.join(BASE_DIR, "scaler.pkl")

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    return model, scaler
model, scaler = load_model()

# Title and description
st.title("❤️ Cardiovascular Disease Prediction System")
st.markdown("### Enter patient information to predict cardiovascular disease risk")
st.markdown("---")

# Create two columns for input
col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 Basic Information")
    age = st.number_input("Age (years)", min_value=1, max_value=120, value=50)
    gender = st.selectbox("Gender", options=[("Female", 0), ("Male", 1)], format_func=lambda x: x[0])
    height = st.number_input("Height (cm)", min_value=100, max_value=250, value=170)
    weight = st.number_input("Weight (kg)", min_value=20, max_value=300, value=70)
    
with col2:
    st.subheader("🩺 Medical Information")
    ap_hi = st.number_input("Systolic Blood Pressure", min_value=80, max_value=200, value=120)
    ap_lo = st.number_input("Diastolic Blood Pressure", min_value=50, max_value=140, value=80)
    cholesterol = st.selectbox("Cholesterol Level", 
                               options=[(1, "Normal"), (2, "Above Normal"), (3, "Well Above Normal")],
                               format_func=lambda x: x[1])
    gluc = st.selectbox("Glucose Level",
                       options=[(1, "Normal"), (2, "Above Normal"), (3, "Well Above Normal")],
                       format_func=lambda x: x[1])

st.markdown("---")

# Lifestyle factors
st.subheader("🏃 Lifestyle Factors")
col3, col4, col5 = st.columns(3)

with col3:
    smoke = st.selectbox("Smoking", options=[(0, "No"), (1, "Yes")], format_func=lambda x: x[1])
    
with col4:
    alco = st.selectbox("Alcohol Consumption", options=[(0, "No"), (1, "Yes")], format_func=lambda x: x[1])
    
with col5:
    active = st.selectbox("Physical Activity", options=[(0, "No"), (1, "Yes")], format_func=lambda x: x[1])

st.markdown("---")

# Predict button
if st.button("🔍 Predict Risk", type="primary", use_container_width=True):
    # Convert age to days
    age_days = age * 365
    
    # Calculate derived features
    bmi = weight / ((height / 100) ** 2)
    pulse_pressure = ap_hi - ap_lo
    map_value = ap_lo + (pulse_pressure / 3)
    
    # Create feature array
    features = np.array([[age_days, gender[1], height, weight, ap_hi, ap_lo,
                     cholesterol[0], gluc[0], smoke[0], alco[0], active[0],
                     bmi, pulse_pressure, map_value]])

    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Make prediction
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0]
    
    # Display results
    st.markdown("---")
    st.subheader("📊 Prediction Results")
    
    col_result1, col_result2 = st.columns(2)
    
    with col_result1:
        if prediction == 1:
            st.error("### ⚠️ HIGH RISK")
            st.markdown("The patient is at **high risk** for cardiovascular disease.")
        else:
            st.success("### ✅ LOW RISK")
            st.markdown("The patient is at **low risk** for cardiovascular disease.")
    
    with col_result2:
        # Create gauge chart for probability
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability[1] * 100,
            title={'text': "Disease Risk (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkred" if prediction == 1 else "green"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "lightcoral"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    # Additional information
    st.markdown("---")
    st.subheader("📈 Patient Metrics")
    
    col_metric1, col_metric2, col_metric3, col_metric4 = st.columns(4)
    
    with col_metric1:
        st.metric("BMI", f"{bmi:.2f}")
    
    with col_metric2:
        st.metric("Pulse Pressure", f"{pulse_pressure:.0f}")
    
    with col_metric3:
        st.metric("MAP", f"{map_value:.0f}")
    
    with col_metric4:
        st.metric("Risk Score", f"{probability[1]*100:.1f}%")
    
    # Recommendations
    st.markdown("---")
    st.subheader("💡 Recommendations")
    
    if prediction == 1:
        st.warning("""
        **Please consult a healthcare professional immediately. Consider the following:**
        - Schedule a comprehensive cardiovascular examination
        - Monitor blood pressure regularly
        - Follow a heart-healthy diet
        - Engage in regular physical activity
        - Manage stress levels
        - Avoid smoking and limit alcohol consumption
        """)
    else:
        st.info("""
        **Maintain a healthy lifestyle to keep your low risk status:**
        - Continue regular exercise
        - Maintain a balanced diet
        - Keep blood pressure in check
        - Stay active and manage stress
        - Regular health checkups
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>⚠️ Disclaimer: This is a prediction model for educational purposes. 
    Always consult healthcare professionals for medical advice.</p>
</div>
""", unsafe_allow_html=True)
