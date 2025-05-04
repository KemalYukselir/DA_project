from eda import get_clean_df
import pandas as pd
import numpy as np
import itertools

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

class LinearRegressionModel:
    DEBUG = True
    def __init__(self):
        # Cleaned dataframe
        self.df_model = self.load_dataframe()
        self.X_train, self.X_test, self.y_train, self.y_test = self.split_train_test()
        self.encoder = ""
        self.apply_target_encoding()
        self.X_train_fe, self.X_test_fe = self.prepare_features()
        self.scaler = ""
        self.apply_scaling()
        self.calculate_vif()
        self.selected_features = self.feature_selection_aic()
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
        # After splitting:
        self.encoder = ce()
        self.X_train['Course Subject'] = self.encoder.fit_transform(self.X_train['Course Subject'], self.y_train)
        self.X_test['Course Subject'] = self.encoder.transform(self.X_test['Course Subject'])


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
        # Current R measures
        # 0.695
        # 0.686

        df = df.copy()

        df['log_MedianHours'] = np.log(df['Median Hours for Certification'] + 1)
        df["log_Bachelor's"] = np.log(df["% Bachelor's Degree or Higher"] + 1)
        df['log_Total_Course_Hours'] = np.log(df['Total Course Hours (Thousands)'] + 1)
        df['interaction_deep_learn'] = df['% Audited'] * df['% Played Video']

        # Drop irrelevant or highly collinear columns
        df.drop(columns=[
            "Institution",  # Useless
            "Course Number", # Useless
            "Launch Date", # Useless
            "Course Title",  # Useless
            "Instructors", # Useless
            "Year", # Useless
            "Honor Code Certificates",  # Useless
            "Certified", # Same as target
            "Median Age", # DROP due to ethical reasons
            "% Female",  # DROP due to ethical reasons
            "% Male", # DROP due to ethical reasons
            # "Course Subject", # Target encoded
            "Participants (Course Content Accessed)",  # DROP High VIF
            "Audited (> 50% Course Content Accessed)",  # DROP High VIF
            "% Certified of > 50% Course Content Accessed", # Don't include it dosnt make sense

            "% Audited",  # Highly correlated with Deep learners
            "% Played Video",  # Drop - High p value
            "% Posted in Forum",  # Drop - High p value
            # # "% Grade Higher Than Zero",
            "Total Course Hours (Thousands)",
            "Median Hours for Certification",  # DROP because high p value
            "% Bachelor's Degree or Higher", # Drop from log
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
            X_train_fe.info()

        return X_train_fe, X_test_fe

    def calculate_vif(self):
        ##################
        ## VIF
        ##################

        # Create a DataFrame to hold VIF values
        vif_data = pd.DataFrame()
        vif_data["Feature"] = self.X_train_fe.columns
        vif_data["VIF"] = [variance_inflation_factor(self.X_train_fe.values, i) for i in range(self.X_train_fe.shape[1])]
        vif_data = vif_data[vif_data["Feature"] != "const"]
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
        print(f"(Train) Cross-Validation RMSE scores: {rmse_scores}")
        print(f"(Train )Average CV RMSE: {rmse_scores.mean():.3f}")
        print("\n")

        X = self.X_test_fe
        y = self.y_test

        # Use K-Fold cross-validation
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)

        # Get scores (neg mean squared error, we take the root to get RMSE)
        neg_mse_scores = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=kf)

        rmse_scores = np.sqrt(-neg_mse_scores)

        print(f"(Test) Cross-Validation RMSE scores: {rmse_scores}")
        print(f"(Test )Average CV RMSE: {rmse_scores.mean():.3f}")


    def manual_check(self,y_test_pred):
        ##################
        ## Check predictions manually
        ##################
        print("Manual Check\n")
        df_manual = pd.DataFrame({'Actual': self.y_test, 'Predicted': y_test_pred})
        print(df_manual.tail(10))

    def predict_from_model(self,test_dict):
        ##################
        ## Predict from model
        ##################
        # Separate out 'Course Subject' before forming DataFrame
        course_subject = test_dict.pop("Course Subject")
        test_df = pd.DataFrame([test_dict])

        # Create a 1-row DataFrame with same column name
        subject_df = pd.DataFrame({"Course Subject": [course_subject]})
        encoded_subject = self.encoder.transform(subject_df)

        # Assign the encoded value back into the test row
        test_df["Course Subject"] = encoded_subject["Course Subject"].values


        # Ensure the same column order as training data
        test_df = test_df[self.X_train_fe.columns]

        # Scale the numeric columns
        num_cols = self.X_train_fe.select_dtypes(include=['float64', 'int64']).columns.tolist()
        num_cols.remove('const')
        test_df[num_cols] = self.scaler.transform(test_df[num_cols])

        print("Test df:")
        print(test_df.head(10))

        # Predict
        prediction = self.linreg.predict(test_df)
        return prediction
    
    def feature_selection_aic(self):
        X = self.X_train_fe.copy()
        y = self.y_train
        best_aic = np.inf
        best_combo = X.columns.tolist()
        for k in range(1, len(X.columns) + 1):
            for combo in itertools.combinations(X.columns, k):
                try:
                    model = sm.OLS(y, X[list(combo)]).fit()
                    if model.aic < best_aic:
                        best_aic = model.aic
                        best_combo = list(combo)
                except:
                    continue
        if self.DEBUG:
            print("Selected features by AIC:", best_combo)
        return best_combo


        
if __name__ == "__main__":
    result = LinearRegressionModel()
    result.summarise_model()
