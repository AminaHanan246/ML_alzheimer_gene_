import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_predict
from utilis import (
basic_eda,
ml_model,
strat_k_fold,
save_param,
)
#-----------------------------------------------
#Defining Functions
#-----------------------------------------------
#EDA

if __name__=='__main__':
    # Load data
    path = r"D:\BI_prj\ML_biomarker\alzheimers_gene"
    os.chdir(path.replace("\\","/"))
    os.makedirs('plots', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    alz_gene = pd.read_csv("alzheimer_disease_vs_control.csv")
    data = alz_gene

    # Data Overview
    summary = basic_eda(data)
    print(summary['head'])
    #print(summary['columns'])
    print(summary['missing_values'])
    print(summary['shape'])

    #EDA
    alz_gene.drop('batch',axis=1,inplace=True)
    print(alz_gene['label'].unique())

    # Check class imbalance
    plt.figure(figsize=(5, 5))
    sns.countplot(x='label', data=alz_gene, edgecolor='black', hue='label')
    plt.xlabel("Label")
    plt.ylabel("Frequency")
    plt.title('Class Distribution')
    plt.tight_layout()
    plt.savefig("plots/Label freq.png")
    plt.show()
    print("\nDataset distribution:", alz_gene['label'].value_counts())

    #Model Training
    alz_gene['label'] = alz_gene['label'].map({'condition': 1, 'control': 0})

    X = alz_gene.drop('label',axis=1)
    y = alz_gene['label']
    joblib.dump({'X': X, 'y': y}, 'data_clean.pkl')

    features = X.columns

    #============================
    # Training with raw data
    #============================
    '''
    PCA reduced model performance indicating loosing of important info with discriminative power. Continuing with raw data
    '''
    #Train - Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Train size:", X_train.shape, y_train.shape)
    print("Test size:", X_test.shape, y_test.shape)

    print("\ny_train distribution", y_train.value_counts())
    print("\ny_test distribution", y_test.value_counts())

    # List of models to be run
    run_model = ['logistic regression','random forest','xgboost']

    #List of metrics to be measured
    metrics = ['accuracy','precision','recall','f1','roc_auc']

    tuning_results = []
    skf_results = []

    for run_ml in run_model:
        print(f"Tuning {run_ml}")
        result = {'Model': run_ml}

        # Hyperparameter tuning
        tuning_scores, best_model, best_params, best_score, fpr, tpr = ml_model(X_train, y_train, X_test, y_test,
                                                                                    run_ml, tuning=True)
        joblib.dump(best_model, f"{run_ml}_tuned_model.pkl")
        print(f"\nBest Hyperparameters from RandomizedSearchCV {run_ml.capitalize()}:")
        print(best_params)
        save_param(run_ml, best_params,"results/alz_params.yml")

        # Hyperparameter Evaluation
        print(f"\nBest Parameters Results {run_ml} without PCA:")
        for metric in metrics:
            if metric in tuning_scores:
                print(f"\n {metric.capitalize()} Score       : {tuning_scores[metric]:.4f}")
                result[metric] = tuning_scores[metric]
        print(f"Confusion Matrix     :\n{tuning_scores['confusion']}")
        print(f"AUC Score     :\n{tuning_scores['auc score']:.4f}")

        #Storing tuning scores
        tuning_results.append(result)

        #Plot confusion matrix
        plt.figure(figsize=(4, 3))  # Adjust figure size as needed
        sns.heatmap(
            tuning_scores['confusion'],
            annot=True,
            annot_kws={"size": 20},
            fmt='d',
            cmap='Blues',
            xticklabels=(['Positive', 'Negative']),
            yticklabels=['Positive', 'Negative'],
            cbar=False

        )
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.xlabel('Actual Values', fontsize=16)
        plt.ylabel('Predicted Values',fontsize=16)
        plt.title('Confusion Matrix', fontsize=16)
        plt.tight_layout()
        plt.tight_layout()
        plt.savefig(f"plots/Confusion Matrix_{run_ml}_tuning.png")

        # 10-fold Cross validation
        skf_scores = strat_k_fold(X, y, best_model)
        print(f"\n10-Fold Stratified Cross Validation Results {run_ml} without PCA:")
        # CV evaluation
        for metric in metrics:
            print(f"\n {metric.capitalize()} Score       : {skf_scores[metric][0]:.4f} ± {skf_scores[metric][1]:.4f}")
            result[metric] = skf_scores[metric][0]
            result["Std. Dev."] = skf_scores[metric][1]
        joblib.dump(best_model, f"{run_ml}_CV_model.pkl")
        # Storing skf scores
        skf_results.append(result)

    #Saving all results to CSV
    tuning_csv = pd.DataFrame(tuning_results)
    tuning_csv.to_csv("results/tuning_results.csv")
    skf_csv = pd.DataFrame(skf_results)
    skf_csv.to_csv("results/skf_results.csv")




