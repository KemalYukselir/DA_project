from import_df import *
import pandas as pd

# Notes
"""
Nulls:
- Instructors has one null - Row can be dropped

Objects:
- Institution seems fine
- Course Number seems fine
- Launch date needs to be converted to a format
- Course Title seems fine
- Instructors seems fine
- Course Subject seems fine
- Played video has --- ?? - can aslo be converted to a float
"""

def check_uniques_in_objects(df):
    """Print unique values for object-type columns."""
    for col in df.select_dtypes(include=['object']).columns:
        print(f"\n\n{col}: {df[col].unique()}\n\n")

def clean_dataframe(df):
    """Clean and format the DataFrame."""
    df_clean = df.copy()

    # Drop rows with null values
    df_clean.dropna(inplace=True)

    # Convert 'Launch Date' to datetime format
    df_clean['Launch Date'] = pd.to_datetime(df_clean['Launch Date'], format='%m/%d/%Y')

    # Replace '---' with NaN and convert '% Played Video' to numeric
    df_clean['% Played Video'] = pd.to_numeric(
        df_clean['% Played Video'].replace('---', pd.NA), errors='coerce'
    )

    # Fill NaN values in '% Played Video' with the column mean
    mean_played_video = df_clean['% Played Video'].mean()
    df_clean['% Played Video'].fillna(mean_played_video, inplace=True)

    return df_clean

def main():
    df = get_dataframe()

    print("\nDataFrame Info:")
    print(df.info())
    print(df.shape)

    print("\nUnique values in object columns:")
    check_uniques_in_objects(df)

    df_clean = clean_dataframe(df)

    print("\nNew unique values in '% Played Video':")
    print(df_clean['% Played Video'].unique())

    print("\nCleaned DataFrame Head:")
    print(df_clean.head())

    print("\nCleaned DataFrame Info:")
    print(df_clean.info())

def get_clean_df():
    """Return the cleaned DataFrame."""
    return clean_dataframe(get_dataframe())

if __name__ == "__main__":
    main()