import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_predict

from scripts.utilis import load_csv, model_tuning_CV, basic_eda, plot_class_dist
#-----------------------------------------------
#Defining Functions
#-----------------------------------------------

if __name__=='__main__':
    # Load data
    path = r"D:\BI_prj\ML_biomarker\alzheimers_gene"
    os.chdir(path.replace("\\","/"))
    os.makedirs('plots', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    alz_gene = load_csv("alzheimer_disease_vs_control.csv")

    # Data Overview
    summary = basic_eda(alz_gene)
    #print(summary['columns'])
    print(summary['missing_values'])
    print(summary['shape'])

    #Class distribution
    plot_class_dist(alz_gene,'label','label')

    #EDA
    alz_gene.drop('batch',axis=1,inplace=True)
    print(alz_gene['label'].unique())

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
        tuning_results, skf_results,best_model = model_tuning_CV("",X,y,X_train,y_train,X_test,y_test,run_ml,metrics, tuning_results,skf_results)

    #Saving all results to CSV
    tuning_csv = pd.DataFrame(tuning_results)
    tuning_csv.to_csv("results/tuning_results.csv")
    skf_csv = pd.DataFrame(skf_results)
    skf_csv.to_csv("results/skf_results.csv")


