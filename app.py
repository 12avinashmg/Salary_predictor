
# app.py
import streamlit as st
import pandas as pd
import joblib
import os

# Load model and encoders
model = joblib.load("best_salary_model.pkl")
encoders = joblib.load("encoders.pkl")

# Streamlit page setup
st.set_page_config(page_title="Salary Predictor", page_icon="💰", layout="centered")

# Header
st.markdown("""
    <h1 style='text-align: center; color: #2c3e50;'>💰 Employee Salary Class Predictor</h1>
    <p style='text-align: center;'>Predict whether an employee earns more than 50K using key features.</p>
    <hr style='border: 1px solid #ddd;'>
""", unsafe_allow_html=True)

# Sidebar Inputs
st.sidebar.header("📋 Input Employee Data")
age = st.sidebar.slider("Age", 18, 65, 30)
educational_num = st.sidebar.slider("Education Level (numeric)", 1, 16, 10)
occupation = st.sidebar.selectbox("Occupation", list(encoders['occupation'].classes_))
hours_per_week = st.sidebar.slider("Hours per Week", 1, 80, 40)
company_type = st.sidebar.selectbox("Company Type", list(encoders['company_type'].classes_))
location_tier = st.sidebar.selectbox("Location Tier", list(encoders['location_tier'].classes_))
workclass = st.sidebar.selectbox("Workclass", list(encoders['workclass'].classes_))
experience = st.sidebar.slider("Years of Experience", 0, 40, 5)

# Input DataFrame
input_df = pd.DataFrame({
    'age': [age],
    'educational-num': [educational_num],
    'occupation': [occupation],
    'hours-per-week': [hours_per_week],
    'company_type': [company_type],
    'location_tier': [location_tier],
    'workclass': [workclass],
    'experience': [experience]
})

st.subheader("🔍 Input Summary")
st.write(input_df)

# Encode categorical fields
for col in ['occupation', 'company_type', 'location_tier', 'workclass']:
    if col in input_df.columns:
        input_df[col] = encoders[col].transform(input_df[col])

# Predict
if st.button("🔮 Predict Salary Class"):
    prediction = model.predict(input_df)[0]
    st.success(f"💼 Predicted Salary Class: **{prediction}**")

    # Log input and prediction
    input_df['PredictedSalaryClass'] = prediction
    log_file = "prediction_log.csv"
    if os.path.exists(log_file):
        prev = pd.read_csv(log_file)
        prev = pd.concat([prev, input_df], ignore_index=True)
        prev.to_csv(log_file, index=False)
    else:
        input_df.to_csv(log_file, index=False)
    st.info("📂 Prediction saved successfully.")

# Show previous predictions
if os.path.exists("prediction_log.csv"):
    with st.expander("📊 View Previous User Predictions"):
        st.dataframe(pd.read_csv("prediction_log.csv"))

# Batch Prediction
st.markdown("---")
st.markdown("### 📁 Batch Prediction (CSV Upload)")
uploaded_file = st.file_uploader("Upload CSV file with the same columns as input", type=["csv"])

if uploaded_file is not None:
    batch_data = pd.read_csv(uploaded_file)
    st.write("📄 Uploaded Data Preview:")
    st.dataframe(batch_data.head())

    # Encode categorical columns
    for col in ['occupation', 'company_type', 'location_tier', 'workclass']:
        if col in batch_data.columns:
            batch_data[col] = encoders[col].transform(batch_data[col])

    # Predict
    batch_data['PredictedSalaryClass'] = model.predict(batch_data)

    st.success("✅ Predictions completed!")
    st.write("### 💡 Predicted Output:")
    st.dataframe(batch_data.head())

    csv = batch_data.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Predicted CSV", csv, "predicted_salaries.csv", "text/csv")
