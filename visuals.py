import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from eda import get_clean_df

df_visuals = get_clean_df()

# Filter the DataFrame to include only float and integer columns
numeric_df = df_visuals.select_dtypes(include=['float', 'int'])
numeric_df = numeric_df.drop(columns=['% Certified'])  # Drop the target variable

# Correlation heatmap
def heatmap():
    """Generate a heatmap to visualize correlations between numeric features."""
    plt.figure(figsize=(12, 8))  # Bigger but still manageable
    corr_matrix = numeric_df.corr()

    # Set up the matplotlib figure
    sns.heatmap(
        corr_matrix,
        cmap='coolwarm',
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        square=True,
        cbar_kws={"shrink": 0.8},
        annot_kws={"size":8}
    )
    plt.title('Correlation Heatmap', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


# Show linear relationships
def scatter_plot():
    """Generate scatter plots to visualize linear relationships between features."""
    sns.pairplot(numeric_df)
    plt.show()

heatmap()