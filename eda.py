from import_df import get_dataframe


df = get_dataframe()

##################
## Check Data types, Nulls and Coloumns
##################

print(df.info())
# Seems to be no nulls 
# We will look at object data types
