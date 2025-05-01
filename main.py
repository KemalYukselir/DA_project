import streamlit as st
import pandas as pd
from model import LinearRegressionModel

# Load the trained model once
@st.cache_resource
def load_model():
    return LinearRegressionModel()

model = load_model()

# Title
st.title("🎓 EdTrack - Predict Course Certification Rates")

# Sidebar for navigation
mode = st.sidebar.selectbox("Select Input Mode:", ["Manual Input"])

# Manual input mode
if mode == "Manual Input":
    st.subheader("Enter Course Details Manually")

    course_subject = st.selectbox("Course Subject", [
        "Government, Health, and Social Science",
        "Humanities, History, Design, Religion, and Education",
        "Science, Technology, Engineering, and Mathematics",
        'Computer Science'
    ])

    percent_male = st.slider("% Male Participants In Course", 0.0, 100.0, 88.28)
    total_course_hours = st.slider("Total Course Hours", 0.0, 1000.0, 418.94)
    median_age = st.slider("Median Age of Participants", 0.0, 100.0, 26.0)
    percent_audited = st.slider("% Audited", 0.0, 100.0, 15.04)
    percent_certified_50plus = st.slider("% Certified of > 50% Course Content Accessed", 0.0, 100.0, 54.98)
    percent_grade_higher = st.slider("% Grade Higher Than Zero", 0.0, 100.0, 28.97)

    # Build input data dictionary
    input_data = {
        "const": 1,
        "% Audited": percent_audited,
        "% Certified of > 50% Course Content Accessed": percent_certified_50plus,
        "% Grade Higher Than Zero": percent_grade_higher,
        "Total Course Hours (Thousands)": total_course_hours / 1000,
        "Median Age": median_age,
        "% Male": percent_male,
        "Course Subject": course_subject,
    }

    if st.button("Predict % Certified"):
        prediction = model.predict_from_model(input_data)
        st.success(f"Predicted Certification Rate: {prediction.iloc[0]:.2f}%")
