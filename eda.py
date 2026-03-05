import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def perform_eda(file_path):
    os.makedirs('eda_plots', exist_ok=True)
    df = pd.read_csv(file_path)
    
    # 1. Basic Statistics
    stats = df.describe()
    stats.to_csv('results/data_statistics.csv')
    
    # 2. Correlation Matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig('eda_plots/correlation_matrix.png')
    plt.close()
    
    # 3. Distribution of Target Variable
    plt.figure(figsize=(10, 6))
    sns.histplot(df['MinedTonnes'], kde=True)
    plt.title('Distribution of MinedTonnes (Target)')
    plt.tight_layout()
    plt.savefig('eda_plots/target_distribution.png')
    plt.close()
    
    # 4. Pairplot to see relationships
    plt.figure(figsize=(15, 15))
    sns.pairplot(df)
    plt.tight_layout()
    plt.savefig('eda_plots/pairplot.png')
    plt.close()
    
    # 5. Boxplots for Outlier Detection
    plt.figure(figsize=(12, 8))
    df.boxplot()
    plt.title('Boxplot of Features')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('eda_plots/boxplot.png')
    plt.close()
    
    # 6. Check for Missing Values and Data Types
    with open('results/data_info.txt', 'w') as f:
        f.write("Data Info:\n")
        df.info(buf=f)
        f.write("\nMissing Values:\n")
        f.write(df.isnull().sum().to_string())
        f.write("\n\nSkewness:\n")
        f.write(df.skew().to_string())

    print("EDA complete. Plots saved in eda_plots/ and stats in results/.")

if __name__ == "__main__":
    perform_eda('cleaned_data.csv')
