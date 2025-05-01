from eda import get_clean_df
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split    # for performing train-test split on the data
from sklearn.preprocessing import RobustScaler    # for scaling the data

# Import target encoder
from category_encoders import TargetEncoder  as ce   # for target encoding categorical features

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold

# Use statsmodels for both the model and its evaluation
import statsmodels.api as sm    # we'll get the model from
import statsmodels.tools        # we'll get the evaluation metrics from
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Set display options for pandas
pd.set_option('display.max_columns', None)

class Linear_Regression_Model:
    DEBUG = True
    def __init__(self):
        # Cleaned dataframe
        self.df_model = self.load_dataframe()
        self.apply_target_encoding()
        self.X_train, self.X_test, self.y_train, self.y_test = self.split_train_test()
        self.X_train_fe, self.X_test_fe = self.prepare_features()
        self.calculate_vif()
        self.scaler = ""
        self.apply_scaling()
        self.linreg = self.build_model()

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
    
    def apply_target_encoding(self):
        encoder = ce(cols=["Course Subject"])
        self.df_model["Course Subject"] = encoder.fit_transform(self.df_model["Course Subject"], self.df_model['% Certified'])

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

        # Drop irrelevant or highly collinear columns
        df.drop(columns=[
            # "Course Subject", # Temp
            "% Bachelor's Degree or Higher", # Drop from log
            "% Played Video",  # DROP because high p value
            "% Posted in Forum",  # DROP because high p value
            "Median Hours for Certification",  # DROP because high p value
            "Certified", # Very correlated with target
            "Honor Code Certificates",
            "Year", # Useless
            "Institution",
            "Course Number", # Useless
            "Launch Date", # Useless
            "Course Title",
            "Instructors", # Useless
            "Participants (Course Content Accessed)",  # DROP High VIF
            "Audited (> 50% Course Content Accessed)",  # DROP High VIF
            "% Female"  # DROP because % Male exists
        ], inplace=True)
    
        # Target encode
        # def apply_target_encoding(X_train, X_test, y_train, col_name, target_name='% Certified'):
        #         temp_df = X_train.copy()
        #         temp_df[target_name] = y_train
        #         target_mean_map = temp_df.groupby(col_name)[target_name].mean()
        #         X_train[f'{col_name}_encoded'] = X_train[col_name].map(target_mean_map)
        #         X_test[f'{col_name}_encoded'] = X_test[col_name].map(target_mean_map)
        #         X_train.drop(columns=[col_name], inplace=True)
        #         X_test.drop(columns=[col_name], inplace=True)
        #         return X_train, X_test

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
        self.scaler = RobustScaler()

        # Fit on train and transform both sets
        self.X_train_fe[num_cols] = self.scaler.fit_transform(self.X_train_fe[num_cols])
        self.X_test_fe[num_cols] = self.scaler.transform(self.X_test_fe[num_cols])

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

        # Print the summary of the model
        print(self.linreg.summary())

        y_pred = self.linreg.predict(self.X_train_fe)
        y_test_pred = self.linreg.predict(self.X_test_fe)

        print("\nTrain Metrics\n")
        print("RMSE: ", statsmodels.tools.eval_measures.rmse(self.y_train, y_pred))
        print("MAE: ", statsmodels.tools.eval_measures.meanabs(self.y_train, y_pred))
        print("MSE: ", statsmodels.tools.eval_measures.mse(self.y_train, y_pred))

        print("\nTest Metrics\n")
        print("RMSE: ", statsmodels.tools.eval_measures.rmse(self.y_test, y_test_pred))
        print("MAE: ", statsmodels.tools.eval_measures.meanabs(self.y_test, y_test_pred))
        print("MSE: ", statsmodels.tools.eval_measures.mse(self.y_test, y_test_pred))

        print("")
        self.manual_check(y_test_pred)
        print("")
        # Cross-validation
        self.cross_validation_score(cv=10)

    def cross_validation_score(self, cv=5):
        # Use the scaled + encoded training data
        X = self.X_train_fe
        y = self.y_train

        # Use scikit-learn's LinearRegression (same algorithm, just different lib)
        model = LinearRegression()

        # Use K-Fold cross-validation
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)

        # Get scores (neg mean squared error, we take the root to get RMSE)
        neg_mse_scores = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=kf)

        rmse_scores = np.sqrt(-neg_mse_scores)
        print(f"Cross-Validation RMSE scores: {rmse_scores}")
        print(f"Average CV RMSE: {rmse_scores.mean():.3f}")


    def manual_check(self,y_test_pred):
        ##################
        ## Check predictions manually
        ##################
        df_manual = pd.DataFrame({'Actual': self.y_test, 'Predicted': y_test_pred})
        print(df_manual.head(10))

    def predict_from_model(self,test_dict):
        # test_dict = {
        #     "const": 1,
        #     "% Audited": 15.04,
        #     "% Certified of > 50% Course Content Accessed": 54.98,
        #     "% Played Video": 83.2,
        #     "% Posted in Forum": 8.17,
        #     "% Grade Higher Than Zero": 28.97,
        #     "Total Course Hours (Thousands)": 418.94,
        #     "Median Hours for Certification": 64.45,
        #     "Median Age": 26.0,
        #     "% Male": 88.28,
        #     "% Bachelor's Degree or Higher": 60.68,
        #     "Course Subject_Government, Health, and Social Science": 0,
        #     "Course Subject_Humanities, History, Design, Religion, and Education": 0,
        #     "Course Subject_Science, Technology, Engineering, and Mathematics": 1,
        # }

        test_df = pd.DataFrame([test_dict])

        # Ensure the same column order as training data
        test_df = test_df[self.X_train_fe.columns]

        # Scale the numeric columns
        num_cols = self.X_train_fe.select_dtypes(include=['float64', 'int64']).columns.tolist()
        num_cols.remove('const')
        test_df[num_cols] = self.scaler.transform(test_df[num_cols])

        # Predict
        prediction = self.linreg.predict(test_df)
        return prediction


        
if __name__ == "__main__":
    result = Linear_Regression_Model()
    result.summarise_model()
