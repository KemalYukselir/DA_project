from eda import get_clean_df
import pandas as pd
import matplotlib.pyplot as plt    # for data visualisation
import seaborn as sns  # for data visualisation

from sklearn.model_selection import train_test_split    # for performing train-test split on the data
from sklearn.preprocessing import RobustScaler, StandardScaler   # for scaling the data

# Use statsmodels for both the model and its evaluation
import statsmodels.api as sm    # we'll get the model from
import statsmodels.tools        # we'll get the evaluation metrics from
from statsmodels.stats.outliers_influence import variance_inflation_factor

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
    df = df.copy()

    # OHE for categorical variables
    df = pd.get_dummies(df, columns=['Course Subject'], drop_first=True, dtype=int)

    # Drop irrelevant or highly collinear columns
    df.drop(columns=[
        "Honor Code Certificates",
        "Year",
        "Institution",
        "Course Number",
        "Launch Date",
        "Course Title",
        "Instructors",
        "Participants (Course Content Accessed)",  # DROP
        "Audited (> 50% Course Content Accessed)",  # DROP
        "% Female"  # DROP because % Male exists
    ], inplace=True)

    # Add constant
    df = sm.add_constant(df)

    return df


# Transform the data
pd.set_option('display.max_columns', None)

X_train_fe = feature_eng(X_train)
X_test_fe = feature_eng(X_test)

# Check index
print(all(X_train_fe.index == X_train.index))
print(all(X_test_fe.index == X_test.index))

# Create a DataFrame to hold VIF values
vif_data = pd.DataFrame()
vif_data["Feature"] = X_train_fe.columns
vif_data["VIF"] = [variance_inflation_factor(X_train_fe.values, i) for i in range(X_train_fe.shape[1])]
print(vif_data)

##################
## Scaling
##################

print(X_train_fe.describe())

# Creating a list of columns to scale using Robust Scaler
# Area of income and Area population has a large range
num_cols = X_train_fe.select_dtypes(include=['float64', 'int64']).columns.tolist()
num_cols.remove('const')  # Year is not a feature


# Initialize scaler
scaler = RobustScaler()

# Fit on train and transform both sets
X_train_fe[num_cols] = scaler.fit_transform(X_train_fe[num_cols])
X_test_fe[num_cols] = scaler.transform(X_test_fe[num_cols])

##################
## Model
##################

linreg = sm.OLS(y_train, X_train_fe).fit()

##################
## Summary and metrics
##################

print(linreg.summary())