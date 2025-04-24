from eda import get_clean_df
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
## Train Test Split
##################

