import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from utilis import (
ml_model,
save_param,
strat_k_fold
)

if __name__ == "__main__":

    '''
    PCA vs Non-PCA
    Dimensionality reduction performed to check if it would produce models with better results.
    '''
    # Load data
    path = r"D:\BI_prj\ML_biomarker\alzheimers_gene"
    os.chdir(path.replace("\\", "/"))
    os.makedirs('plots', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    alz_gene = pd.read_csv("alzheimer_disease_vs_control.csv")
    data = alz_gene


    # EDA
    alz_gene.drop('batch', axis=1, inplace=True)
    print(alz_gene['label'].unique())

    print("\nDataset distribution:", alz_gene['label'].value_counts())

    # Model Training
    alz_gene['label'] = alz_gene['label'].map({'condition': 1, 'control': 0})

    X = alz_gene.drop('label', axis=1)
    y = alz_gene['label']
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

    #List of models to run
    run_model = ['logistic regression','random forest','xgboost']
    # List of metrics to be measured
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

    pca_results = []

    for run_ml in run_model:
        result = {'Model': run_ml}
        tuning_scores, best_model, best_params, best_score, fpr, tpr = ml_model(X_train, y_train, X_test, y_test,
                                                                                run_ml, tuning=True)
        joblib.dump(best_model, f"{run_ml} PCA_tuned_model.pkl")
        print(f"\nBest Hyperparameters from RandomizedSearchCV {run_ml.capitalize()}:")
        print(best_params)
        save_param(run_ml, best_params, "results/alz_pca_params.yml")

        # 10-fold Cross validation
        skf_scores = strat_k_fold(X_pca, y, best_model)
        print(f"\n10-Fold Stratified Cross Validation Results {run_ml} with PCA:")
        # CV evaluation
        for metric in metrics:
            print(f"\n {metric.capitalize()} Score       : {skf_scores[metric][0]:.4f} ± {skf_scores[metric][1]:.4f}")
            result[metric] = skf_scores[metric][0]
            result[f'{metric}_std'] = skf_scores[metric][1]

        joblib.dump(best_model, f"{run_ml}_CV_model.pkl")
        # Storing skf scores
        pca_results.append(result)

    #Save to CSV
    pca_csv = pd.DataFrame(pca_results)
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

