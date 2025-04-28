from eda import get_clean_df
import pandas as pd

from sklearn.model_selection import train_test_split    # for performing train-test split on the data
from sklearn.preprocessing import RobustScaler    # for scaling the data

# Use statsmodels for both the model and its evaluation
import statsmodels.api as sm    # we'll get the model from
import statsmodels.tools        # we'll get the evaluation metrics from
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Set display options for pandas
pd.set_option('display.max_columns', None)

class Linear_Regression_Model:
    DEBUG = False
    def __init__(self):
        self.df_model = self.load_dataframe()
        self.X_train, self.X_test, self.y_train, self.y_test = self.split_train_test()
        self.X_train_fe, self.X_test_fe = self.prepare_features()
        self.calculate_vif()
        self.apply_scaling()

    def __repr__(self):
        return f"Linear Regression Model: {self.linreg}"

    def load_dataframe(self):
        ##################
        ## Get Dataframe
        ##################

        df_model = get_clean_df()
        if self.DEBUG:
            df_model.info()

        return df_model

    def split_train_test(self):
        ##################
        ## Train Test Split
        ##################

        features = list(self.df_model.columns)
        features.remove('% Certified')

        x = self.df_model[features]
        y = self.df_model['% Certified'] # Price target

        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test

    def apply_feature_engineering(self, df):
        ##################
        ## Feature engineering
        ##################

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

    def prepare_features(self):
        ##################
        ## Feature engineering
        ##################

        # Feature engineering for train and test sets
        X_train_fe = self.apply_feature_engineering(self.X_train)
        X_test_fe = self.apply_feature_engineering(self.X_test)

        # Check index
        if self.DEBUG:
            print(all(X_train_fe.index == self.X_train.index))
            print(all(X_test_fe.index == self.X_test.index))

        return X_train_fe, X_test_fe

    def calculate_vif(self):
        ##################
        ## VIF
        ##################

        # Create a DataFrame to hold VIF values
        vif_data = pd.DataFrame()
        vif_data["Feature"] = self.X_train_fe.columns
        vif_data["VIF"] = [variance_inflation_factor(self.X_train_fe.values, i) for i in range(self.X_train_fe.shape[1])]
        if self.DEBUG:
            print(vif_data)

    def apply_scaling(self):
        ##################
        ## Scaling
        ##################

        if self.DEBUG:
            print(self.X_train_fe.describe())

        # Creating a list of columns to scale using Robust Scaler
        # Area of income and Area population has a large range
        num_cols = self.X_train_fe.select_dtypes(include=['float64', 'int64']).columns.tolist()
        num_cols.remove('const')  # Year is not a feature

        # Initialize scaler
        scaler = RobustScaler()

        # Fit on train and transform both sets
        self.X_train_fe[num_cols] = scaler.fit_transform(self.X_train_fe[num_cols])
        self.X_test_fe[num_cols] = scaler.transform(self.X_test_fe[num_cols])

    def build_model(self):
        ##################
        ## Get Model
        ##################

        # Create a linear regression model
        linreg = sm.OLS(self.y_train, self.X_train_fe).fit()

        return linreg

    def summarise_model(self):
        ##################
        ## Summary and metrics
        ##################
        linreg = self.build_model()

        # Print the summary of the model
        print(linreg.summary())

        y_pred = linreg.predict(self.X_train_fe)
        y_test_pred = linreg.predict(self.X_test_fe)

        print("\nTrain Metrics\n")
        print("RMSE: ", statsmodels.tools.eval_measures.rmse(self.y_train, y_pred))
        print("MAE: ", statsmodels.tools.eval_measures.meanabs(self.y_train, y_pred))
        print("MSE: ", statsmodels.tools.eval_measures.mse(self.y_train, y_pred))

        print("\nTest Metrics\n")
        print("RMSE: ", statsmodels.tools.eval_measures.rmse(self.y_test, y_test_pred))
        print("MAE: ", statsmodels.tools.eval_measures.meanabs(self.y_test, y_test_pred))
        print("MSE: ", statsmodels.tools.eval_measures.mse(self.y_test, y_test_pred))

    def predict(self):
        ##################
        ## Predict
        ##################
        # Will be done later
        pass


if __name__ == "__main__":
    result = Linear_Regression_Model()
    result.summarise_model()
