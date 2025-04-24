from eda import get_clean_df
import pandas as pd
import matplotlib.pyplot as plt    # for data visualisation
import seaborn as sns  # for data visualisation

from sklearn.model_selection import train_test_split    # for performing train-test split on the data
from sklearn.preprocessing import RobustScaler, StandardScaler   # for scaling the data

# Use statsmodels for both the model and its evaluation
import statsmodels.api as sm    # we'll get the model from
import statsmodels.tools        # we'll get the evaluation metrics from

##################
## Get Dataframe
##################

df_model = get_clean_df()

df_model.info()

##################
## Train Test Split
##################

features = list(df_model.columns)
features.remove('% Certified')

x = df_model[features]
y = df_model['% Certified'] # Price target

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

##################
## Feature engineering
##################

def feature_eng(df):

    # Creating a copy of the DataFrame
    df = df.copy()

    # Can't be OHE so we drop it.
    # df.drop(columns=['Address'], inplace=True)

    # adding a constant
    df = sm.add_constant(df)

    return df  # returning the DataFrame

# Transform the data
X_train_fe = feature_eng(X_train)
X_test_fe = feature_eng(X_test)

# Check index
print(all(X_train_fe.index == X_train.index))
print(all(X_test_fe.index == X_test.index))

##################
## Scaling
##################

pd.set_option('display.max_columns', None)
print(df_model.describe())

# Notes
"""
Numerical value data ranges:
No scaling needed:
- Year 
- Honor Code Certificates 

Scaling needed:
- Participants (Course Content Accessed) 
- Audited (> 50% Course Content Accessed) 
- Certified
- Audited
- % Certified of > 50% Course Content Accessed
- % Played Video  
- % Posted in Forum  
- % Grade Higher Than Zero
- Total Course Hours (Thousands)  
- Median Hours for Certification
- Median Age      
- % Male    
- % Female  
- % Bachelor's Degree or Higher
"""

# Creating a list of columns to scale using Robust Scaler
# Area of income and Area population has a large range
columns_to_scale = [
    "Participants (Course Content Accessed)",
    "Audited (> 50% Course Content Accessed)",
    "Certified",
    "Audited",
    "% Certified of > 50% Course Content Accessed",
    "% Played Video",
    "% Posted in Forum",
    "% Grade Higher Than Zero",
    "Total Course Hours (Thousands)",
    "Median Hours for Certification",
    "Median Age",
    "% Male",    
    "% Female",
    "% Bachelor's Degree or Higher"
    ]

# Initialize scaler
scaler = RobustScaler()

# Fit on train and transform both sets
X_train_fe[columns_to_scale] = scaler.fit_transform(X_train_fe[columns_to_scale])
X_test_fe[columns_to_scale] = scaler.transform(X_test_fe[columns_to_scale])

##################
## Model
##################