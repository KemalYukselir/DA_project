import streamlit as st
import pandas as pd
from model import LinearRegressionModel

# Load the trained model once
@st.cache_resource
def load_model():
    return LinearRegressionModel()

model = load_model()


# Sidebar for navigation
page = st.sidebar.selectbox("📂 Select a Page", ["Project Overview", "Predictor"])
mode = st.sidebar.selectbox("Select Input Mode:", ["Manual Input"])

if page == "README":
    st.title("📘 Project README 📘")

    st.markdown("""
    ## 🎓 Course Certification Rates Predictor 🎓
    ### By Kemal Yukselir

    **Objective:**  
    Predict the percentage of students who will complete and earn a certificate in an online course, using historical course performance data.
                
    # References
    - [Harvard / MIT](https://www.kaggle.com/datasets/edx/course-study?resource=download)

    **Key Features:**
    - Linear regression model
    - Feature engineering 
    - Robust scaling  
    - Feature combination
    - Cross-validation
    - Target encoding for categorical variables
    - Streamlit dashboard with live predictions

    **Modules:**
    - Pandas
    - NumPy
    - Scikit-learn
    - Statsmodels
    - Category Encoders
    - Streamlit
    - Matplotlib
    - Seaborn

    **Project Highlights:**
    - R² = 0.823
    - Cond. No. = 9.04
    - AIC, BIC = 1166, 1197
    - F-statistic = 128.7        
    - (Train) Average CV RMSE: 3.077
    - (Test) Average CV RMSE: 3.100
    """)
else:
    # Title
    st.title("🎓 Student Certification Rate Predictor")
    # Manual input mode
    if mode == "Manual Input":
        st.subheader("Enter Course Details")

        course_subject = st.selectbox("Course Subject", [
            "Government, Health, and Social Science",
            "Humanities, History, Design, Religion, and Education",
            "Science, Technology, Engineering, and Mathematics",
            'Computer Science'
        ])

        particapants = st.slider("Number of Participants", 0, 10000, 5000)
        total_course_hours = st.slider("Total Course Hours", 0.0, 1000.0, 418.94)
        percent_male = st.slider("% Male Participants In Course", 0.0, 100.0, 88.28)
        median_age = st.slider("Median Age of Participants", 0.0, 100.0, 26.0)
        percent_bachelor_degree = st.slider("% Participants With Bachelor's Degree or Higher", 0.0, 100.0, 50.0)
        percent_grade_higher = st.slider("% Participants With Grade Higher Than Zero From Quizes", 0.0, 100.0, 28.97)
        audited_50plus = st.slider("Number of Audited Participants (> 50% Course Content Accessed)", 0, 10000, 5000)
        percent_certified_50plus = st.slider("% Certified of > 50% Course Content Accessed", 0.0, 100.0, 54.98)


        # Build input data dictionary
        input_data = {
            "const": 1,
            "% Certified of > 50% Course Content Accessed": percent_certified_50plus,
            "% Grade Higher Than Zero": percent_grade_higher,
            "Total Course Hours (Thousands)": total_course_hours,
            "Median Age": median_age,
            "% Male": percent_male,
            "Course Subject": course_subject,
            "% Bachelor's Degree or Higher": percent_bachelor_degree,
            "% Deep learners": (audited_50plus / particapants) * 100
        }

        if st.button("Predict % Certified"):
            prediction = model.predict_from_model(input_data)
            st.success(f"Predicted Certification Rate: {prediction.iloc[0]:.2f}%")
