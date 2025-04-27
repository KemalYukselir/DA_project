import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from eda import get_clean_df

df_visuals = get_clean_df()

# Filter the DataFrame to include only float and integer columns
numeric_df = df_visuals.select_dtypes(include=['float', 'int'])

# Correlation heatmap
plt.figure(figsize=(12, 8))
corr = numeric_df.corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Correlation Heatmap')
plt.show()