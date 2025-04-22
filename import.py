import pandas as pd
import numpy as np
import os

# Specify the folder path
folder_path = '/Users/kemalyukselir/Desktop/DA_project/open+university+learning+analytics+dataset'

for file in os.listdir(folder_path):
    if not file.endswith('.csv'):
        continue
    csv_path = os.path.join(folder_path, file)
    df = pd.read_csv(csv_path)
    print(file)
    print(df.columns)

# Update the path to the CSV file
# csv_path = os.path.join(folder_path, 'studentVle.csv')
# df = pd.read_csv(csv_path)

# print(df.columns)