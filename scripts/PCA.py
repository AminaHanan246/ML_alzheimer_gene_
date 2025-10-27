import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from scripts.utilis import load_config,model_tuning_CV

if __name__ == "__main__":

    '''
    PCA vs Non-PCA
    Dimensionality reduction performed to check if it would produce models with better results.
    '''
    cfg = load_config("config.yml")

    data = joblib.load('data_clean.pkl')
    X = data['X']
    y = data['y']

    #======================================
    #With PCA
    #======================================
    #Standaridisation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)  # ← use transform(), not fit()

    # Apply PCA to retain 95% variance
    pca = PCA(n_components=0.95, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    # Explained variance
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)

    # Scree Plot
    plt.figure(figsize=(8, 5))
    plt.bar(range(1, len(explained_variance) + 1), explained_variance, color='skyblue', edgecolor='black')
    plt.title("Scree Plot")
    plt.xlabel("Principal Component")
    plt.ylabel("Variance Explained")
    plt.xticks(range(1, len(explained_variance) + 1))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

    # Cumulative Explained Variance Plot
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 's-', color='g')
    plt.title("Cumulative Explained Variance")
    plt.xlabel("Principal Component")
    plt.ylabel("Cumulative Variance Explained")
    plt.grid(True)
    plt.show()

    # Scatter Plot of PC1 vs PC2
    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='coolwarm', edgecolor='k')
    plt.title("PCA: PC1 vs PC2")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.colorbar(label='Diagnosis (0 = Control, 1 = Condition)')
    plt.show()

    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

    tuning_results,skf_results = [],[]

    for run_ml in cfg["models"]:
        tuning_results, skf_results,best_model = model_tuning_CV("with_pca",X,y,X_train,y_train,X_test,y_test,run_ml,cfg["metrics"], tuning_results,skf_results)
    #Save to CSV
    pca_csv = pd.DataFrame(skf_results)
    pca_csv.to_csv("results/pca_results.csv",index=False)


    # Load CSV without PCA
    no_pca_csv = pd.read_csv("results/skf_results.csv")

    #Merging data for comparison
    merge_csv = pd.concat([pca_csv, no_pca_csv], ignore_index=True)
    merge_csv['PCA'] = ['Yes'] * len(pca_csv) + ['No'] * len(no_pca_csv)

    merge_csv.to_csv("results/pca_exp_results.csv", index=False)

    #Bar plot for PCA results
    plt.figure(figsize=(8, 6))
    sns.barplot(x='Model',y='accuracy', data=merge_csv, edgecolor='black', hue='PCA')
    #Labels
    plt.xlabel("Model")
    plt.ylabel("Accuracy Score")
    plt.title('Model Accuracy: PCA vs Without PCA')
    plt.legend(title="PCA Applied")

    plt.tight_layout()
    plt.savefig("plots/pca_exp.png")
    plt.show()

