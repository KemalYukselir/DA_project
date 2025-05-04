import seaborn as sns
import matplotlib.pyplot as plt
from eda import get_clean_df
import pandas as pd
import numpy as np

# Helper tools
from collections import Counter # Counting things
import string # Contains String stuffs

# NLTK suite
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.snowball import SnowballStemmer

# Download list of words and characters
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('omw-1.4')

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

print(df_visuals.info())

def check_relationship():
    """
    Check and visualize the relationship between a single feature and the target.

    Parameters:
    - df: pandas DataFrame
    - feature_col: str, the feature column to check
    - target_col: str, default is '% Certified'
    """

    features = ['Median Hours for Certification', '% Played Video', '% Bachelor\'s Degree or Higher','% Posted in Forum']
    target = '% Certified'

    fig, axes = plt.subplots(nrows=1, ncols=len(features), figsize=(5 * len(features), 5))

    for i, feature in enumerate(features):
        sns.regplot(x=feature, y=target, data=df_visuals, ax=axes[i])
        axes[i].set_title(f'{feature} vs {target}')
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel(target)

    plt.tight_layout()
    plt.show()




# Bachelors degree vs Certification rate
def get_course_title_common_words():
    """Generate a bar plot to visualize the most common words in course titles."""
    # Extract the first word from each course title in dataset
    word_bank = ""
    for title in df_visuals['Course Title']:
        # Higher than 30% certified at least
        if df_visuals[df_visuals['Course Title'] == title]['% Certified'].values[0] >= 20:
            word_bank += title + " "

    word_bank = word_bank.lower().split(" ")
    p_stemmer = PorterStemmer() # Stemmer tool
    s_stemmer = SnowballStemmer(language='english') # Stemmer tool
    lemmatizer = WordNetLemmatizer() # Lemmatizer tool

    stpwrd = nltk.corpus.stopwords.words('english') # Stop words
    stpwrd.extend(string.punctuation)

    # Add custom stop words
    manual_stpwrd = ['–']
    stpwrd.extend(manual_stpwrd)
    
    lemma = [lemmatizer.lemmatize(x) for x in word_bank]
    # porter = [p_stemmer.stem(x) for x in word_bank]
    # snowball = [s_stemmer.stem(x) for x in word_bank]
    # print(list(zip(word_bank, lemma)))
    
    lemma = [x for x in lemma if x not in stpwrd]
    # Count top words
    word_count = Counter(lemma)

    # Visual with barplot
    word_count_df = pd.DataFrame(word_count.most_common(10), columns=['Word', 'Count'])
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Word', y='Count', data=word_count_df, palette='viridis', ax=ax)
    ax.set_title('Top 10 Common Words in Course Titles (Certified > 20%)', fontsize=16)
    ax.set_xlabel('Word', fontsize=14)
    ax.set_ylabel('Count', fontsize=14)

    # Show figure
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Course subject vs Certification rate

def get_course_subject_certification_rate():
    """Generate a bar plot to visualize the relationship between course subject and certification rate."""
    # Group by Course Subject and calculate mean % Certified
    grouped_df = df_visuals.groupby('Course Subject')['% Certified'].mean().reset_index()

    # Visual with barplot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Course Subject', y='% Certified', data=grouped_df, palette='viridis', ax=ax)
    ax.set_title('Certification Rate by Course Subject', fontsize=16)
    ax.set_xlabel('Course Subject', fontsize=14)
    ax.set_ylabel('% Certified', fontsize=14)

    # Show figure
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()

# Posted in forum vs certification rate lineplot
def get_posted_forum_certification_rate():
    """Generate a line plot to visualize the relationship between forum participation and certification rate."""
    x = df_visuals["% Posted in Forum"]
    df_visuals['Forum Bin'] = pd.cut(x, bins=[0, 5, 10, 15, 20, 25, 30, 35], right=False)

    # Calculate the mean certification rate for each bin
    bin_means = df_visuals.groupby('Forum Bin')['% Certified'].mean().reset_index()

    # Convert bins to string for better visualization
    bin_means['Forum Bin'] = bin_means['Forum Bin'].astype(str)

    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Forum Bin', y='% Certified', data=bin_means, marker='o')
    plt.title("Average Certification Rate by Forum Participation Bin")
    plt.xlabel("Forum Participation (%) Bin")
    plt.ylabel("% Certified")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    check_relationship()

