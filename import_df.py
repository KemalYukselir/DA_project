import pandas as pd
import numpy as np



def get_dataframe():
    """
    Import appendix CSV file into a pandas DataFrame.
    Returns:
        pd.DataFrame: The imported DataFrame.
    """

    df = pd.read_csv('appendix.csv')

    # Ensure all columns are displayed
    # pd.set_option('display.max_columns', None)

    return df