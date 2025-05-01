import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
from model import Linear_Regression_Model

# Load your trained model
model = Linear_Regression_Model()

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
        "Science, Technology, Engineering, and Mathematics"
    ])

    percent_male = st.slider("% Male Participants In Course", 0.0, 100.0, 88.28)
    percent_bachelor_degree = st.slider("% Participants with Bachelor Degree or Higher", 0.0, 100.0, 60.68)
    total_course_hours = st.slider("Total Course Hours (Thousands)", 0.0, 1000.0, 418.94)
    median_age = st.slider("Median Age of Participants", 0.0, 100.0, 26.0)

    percent_audited = st.slider("% Audited", 0.0, 100.0, 15.04)
    percent_certified_50plus = st.slider("% Certified of > 50% Course Content Accessed", 0.0, 100.0, 54.98)
    percent_played_video = st.slider("% Played Video", 0.0, 100.0, 83.2)
    percent_posted_in_forum = st.slider("% Posted in Forum", 0.0, 100.0, 8.17)
    percent_grade_higher = st.slider("% Grade Higher Than Zero", 0.0, 100.0, 28.97)
    median_hours_certification = st.slider("Median Hours for Certification", 0.0, 1000.0, 64.45)

    # Create the input dataframe
    input_data = {
        "const": 1,
        "% Audited": percent_audited,
        "% Certified of > 50% Course Content Accessed": percent_certified_50plus,
        "% Played Video": percent_played_video,
        "% Posted in Forum": percent_posted_in_forum,
        "% Grade Higher Than Zero": percent_grade_higher,
        "Total Course Hours (Thousands)": total_course_hours / 1000,
        "Median Hours for Certification": median_hours_certification,
        "Median Age": median_age,
        "% Male": percent_male,
        "% Bachelor's Degree or Higher": percent_bachelor_degree,
        "Course Subject_Government, Health, and Social Science": 1 if course_subject == "Government, Health, and Social Science" else 0,
        "Course Subject_Humanities, History, Design, Religion, and Education": 1 if course_subject == "Humanities, History, Design, Religion, and Education" else 0,
        "Course Subject_Science, Technology, Engineering, and Mathematics": 1 if course_subject == "Science, Technology, Engineering, and Mathematics" else 0,
    }

    # Predict
    if st.button("Predict % Certified"):
        prediction = model.predict_from_model(input_data)
        st.success(f"Percentage of students who will certify: {prediction.iloc[0]:.2f}%")