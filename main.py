import streamlit as st
import pandas as pd
from model import LinearRegressionModel

# Load the trained model once
@st.cache_resource
def load_model():
    return LinearRegressionModel()

model = load_model()

# Sidebar for navigation
page = st.sidebar.radio("📂 Select a Page", ["Project Overview","Ethical Standards", "Predictor"])
mode = st.sidebar.selectbox("Select Input Mode:", ["Manual Input"])

if page == "Project Overview":
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
elif page == "Ethical Standards":
    st.title("📄 Project Ethical Standards 📄")

    st.markdown("""
    ## Responsible Use of Machine Learning in Education 

    **Overview**  
    - This project uses historical data from Harvard and MIT to predict the percentage of students likely to complete a course and earn a certificate.  
    - While such models can help institutions improve course design, they also carry ethical risks that must be addressed.

    ### ⚖️ Key Ethical Considerations

    - **Bias in Data**  
      The dataset reflects historical learner behavior, which may be biased toward certain age groups or educational backgrounds.  
      Predictions should not be used to make high-stakes decisions for individuals.

    - **Data Privacy**  
      This analysis uses anonymous and aggregate course data. No personal identifiers are included.

    - **Fairness in Features**  
      Categorical variables like `Course Subject` are encoded using **target encoding** to avoid arbitrary weight assignments.  
      Gender-based columns like `% Male` and `% Female` are handled with caution and transparency.

    - **Transparency & Interpretability**  
      Head over to Project Overview for a detailed rundown of how the model is created.
                
    - **Intended Use**  
      This tool is designed for **educational insights only** — such as identifying which course features may lead to higher certification rates.  
      This tool may be used to help twoards improving current ongoing courses institutions may have.
                
      It is **not** intended to create bias towards any group of learners such as ❌**age**❌.

    ### 📚 Further Reading

    - [Harvard / MIT MOOC Dataset on Kaggle](https://www.kaggle.com/datasets/edx/course-study?resource=download)
    - [The Ethics of Learning Analytics (Jisc Report)](https://www.jisc.ac.uk/guides/code-of-practice-for-learning-analytics)
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
