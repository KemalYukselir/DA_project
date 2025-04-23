from import_df import *


df = get_dataframe()

##################
## Check Data types, Nulls and Coloumns
##################

print(df.info())
# Seems to be no nulls 
# We will look at object data types

##################
## Check unique values in columns 
##################

# Quick way to check values in all columns
def check_uniques_in_objects(df):
  for col in df.columns:
    if df[col].dtype == 'object':
      print("\n\n")
      print(f'{col}: {df[col].unique()}')
      print("\n\n")

check_uniques_in_objects(df)


# Notes
"""
Nulls:
- Instructors has one null - Can be dropped

Objects:
- Institution seems fine
- Course Number seems fine
- Launch date needs to be converted to a format
- Course Title seems fine
- Instructors seems fine
- Course Subject seems fine
- Played video has --- ?? - can aslo be converted to a float
"""

##################
## Data formatting
##################

df_clean = df.copy()

# Drop nulls
df_clean.dropna(inplace=True)

# Convert to date format
df_clean['Launch Date'] = pd.to_datetime(df_clean['Launch Date'], format='%m/%d/%Y')

# Replace '---' with NaN
df_clean['% Played Video'] = df_clean['% Played Video'].replace('---', pd.NA)

# Convert column to numeric
df_clean['% Played Video'] = pd.to_numeric(df_clean['% Played Video'], errors='coerce')

# Calculate the mean and fill NaN values
mean_played_video = df_clean['% Played Video'].mean()
df_clean['% Played Video'].fillna(mean_played_video, inplace=True)

print("New unique values")
print(df_clean['% Played Video'].unique())

print(df_clean.head())

df_clean.info()