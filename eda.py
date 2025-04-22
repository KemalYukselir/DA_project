from import_df import get_dataframe


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
- Institution seems fine
- Course Number seems fine
- Launch date needs to be converted to a format
- Course Title seems fine
- Instructors seems fine
- Course Subject seems fine
- Played video has --- ?? - can aslo be converted to a float
"""