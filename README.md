# 🎯 Course Certification Rates Predictor

## Live on -> https://huggingface.co/spaces/Kemzo/Student-certify-rate

### ✍️ By Kemal Yukselir

## How to run on test
- streamlit run main.py

## Updating venv modules
- pip freeze > requirements.txt
- Make sure to keep numpy at version 1.26.4 for streamlit compatibility
- Make sure to keep scipy at version 1.11.4 for streamlit compatibility

## Description
Predict the percentage of students who will complete and earn a certificate in an online course, using historical course performance data.
Explore insights on some key factors that influence certification rates.
Dive into ethical standards when dealing with data and using machine learning responsibly

### 🔗 Reference
- [Harvard / MIT Dataset](https://www.kaggle.com/datasets/edx/course-study?resource=download)

## 🔑 Key Features
- Linear regression model  
- Feature engineering  
- Robust scaling  
- Feature combination  
- Cross-validation  
- Target encoding for categorical variables  
- Streamlit dashboard with live predictions

## 📦 Modules Used
- Pandas  
- NumPy  
- Scikit-learn  
- Statsmodels  
- Category Encoders  
- Streamlit  
- Matplotlib  
- Seaborn
- itertools

## 📊 Project Highlights
With all ethical practise considered, this is the best model I can get with many reruns.
- **R²** = 0.688
- **Cond. No.** = 4.46
- **AIC, BIC** = 1293, 1317
- **F-statistic** = 82.39
- **(Train) Average** CV RMSE: 3.966
- **(Test) Average** CV RMSE: 4.677
