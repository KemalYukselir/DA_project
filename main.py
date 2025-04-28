import streamlit as st
import pandas as pd
import numpy as np
import pickle
import statsmodels.api as sm
from model import Linear_Regression_Model
# Load your trained model
model = Linear_Regression_Model().build_model()

# Title
st.title("🎓 EdTrack - Predict Course Certification Rates")

# Sidebar for navigation
mode = st.sidebar.selectbox("Select Input Mode:", ["Upload CSV", "Manual Input"])

# Upload CSV mode
if mode == "Upload CSV":
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        
        # Preprocess if needed (e.g., add constant, encode subject)
        if 'const' not in data.columns:
            data = sm.add_constant(data)

        # Make predictions
        predictions = model.predict(data)
        
        st.subheader("Predicted % Certified for Uploaded Courses")
        data["Predicted % Certified"] = predictions
        st.dataframe(data)

# Manual input mode
elif mode == "Manual Input":
    st.subheader("Enter Course Details Manually")

    year = st.number_input("Year", min_value=2012, max_value=2030, value=2025)
    percent_audited = st.slider("% Audited", 0.0, 100.0, 50.0)
    percent_certified_50plus = st.slider("% Certified of > 50% Course Content Accessed", 0.0, 100.0, 50.0)
    percent_grade_higher = st.slider("% Grade Higher Than Zero", 0.0, 100.0, 70.0)
    median_age = st.slider("Median Age", 10.0, 80.0, 30.0)
    percent_male = st.slider("% Male", 0.0, 100.0, 50.0)
    
    course_subject = st.selectbox("Course Subject", [
        "Government, Health, and Social Science",
        "Humanities, History, Design, Religion, and Education",
        "Science, Technology, Engineering, and Mathematics"
    ])
    
    # Create the input dataframe
    input_data = {
        "const": 1,
        "Year": year,
        "% Audited": percent_audited,
        "% Certified of > 50% Course Content Accessed": percent_certified_50plus,
        "% Grade Higher Than Zero": percent_grade_higher,
        "Median Age": median_age,
        "% Male": percent_male,
        "Course Subject_Government, Health, and Social Science": 1 if course_subject == "Government, Health, and Social Science" else 0,
        "Course Subject_Humanities, History, Design, Religion, and Education": 1 if course_subject == "Humanities, History, Design, Religion, and Education" else 0,
        "Course Subject_Science, Technology, Engineering, and Mathematics": 1 if course_subject == "Science, Technology, Engineering, and Mathematics" else 0,
    }
    
    input_df = pd.DataFrame([input_data])

    # Predict
    if st.button("Predict % Certified"):
        prediction = model.predict(input_df)
        st.success(f"Predicted % Certified: {prediction.iloc[0]:.2f}%")

