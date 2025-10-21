import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, f1_score, recall_score, classification_report,roc_curve,roc_auc_score

def strat_k_fold(X, y, model, n_splits=10, shuffle= True, random_state=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    metrics = ['accuracy','recall','precision','f1','roc_auc']
    skf_scores = {}
    for metric in metrics:
        score = cross_val_score(model, X, y, cv=skf, scoring=metric)
        skf_scores[metric] = [score.mean(), score.std()]
    return skf_scores

def hyper_tuning(X,y,input_model,model,method = 'random search'):
    if input_model.lower() == 'logistic regression':
        pass
        param_dist = {
            "penalty": ["l1", "l2"],  # Regularisation
            "C": [1, 0.1, 0.01],  #
            "solver": ["lbfgs", "liblinear", "newton-cg", "sag", "saga"],
            # Minimum number of samples required to split a node
            "max_iter": [100, 1000, 2500, 5000],  # Minimum number of samples required at a leaf node
        }

    elif input_model == 'decision tree':
        param_dist = {
            "max_depth": [None, 5, 10, 20, 30],
            # Maximum depth of the tree (None = nodes expanded until all leaves are pure)
            "min_samples_split": [2, 5, 10],  # Minimum number of samples required to split a node
            "min_samples_leaf": [1, 2, 4],  # Minimum number of samples required at a leaf node
            "max_features": ["sqrt", "log2", None],  # Number of features to consider when looking for best split
        }

    elif input_model == 'random forest':
        param_dist = {
            "n_estimators": [100, 200, 300, 400, 500],  # Number of trees in the forest
            "max_depth": [None, 5, 10, 20, 30],
            # Maximum depth of the tree (None = nodes expanded until all leaves are pure)
            "min_samples_split": [2, 5, 10],  # Minimum number of samples required to split a node
            "min_samples_leaf": [1, 2, 4],  # Minimum number of samples required at a leaf node
            "max_features": ["sqrt", "log2", None],  # Number of features to consider when looking for best split
            "bootstrap": [True]  # Whether bootstrap samples are used when building trees
        }

    elif input_model == 'gradient boosting':
        param_dist = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 0.9, 1.0],
            'max_features': ['sqrt', 'log2']
        }

    elif input_model == 'xgboost':
        param_dist = {
            "n_estimators": [50, 100, 200, 300], # Number of boosting rounds (trees)
            "max_depth": [3, 4, 5, 6, 8, 10], # Maximum depth of each decision tree
            # Acts like min_samples_split, higher = fewer splits (regularization)
            "min_child_weight": [1, 3, 5, 7, 10], # Minimum sum of instance weights ("cover") in a child node
            "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3], # Step size shrinkage (learning rate)
            "subsample": [0.6, 0.8, 1.0], # Fraction of training samples used per tree
            "colsample_bytree": [0.6, 0.8, 1.0], # Fraction of features used per tree
            "gamma": [0, 0.1, 0.2, 0.3, 1]  # Minimum loss reduction required for a split (higher = more conservative)
        }

    # elif input_model == 'svm':
    #     param_dist = {}

    else:
        raise ValueError(f"Unsupported model type: {input_model}")

    if method == 'random search':
        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_dist,
            n_iter=20,
            cv=5,
            scoring="accuracy",
            verbose=2,
            random_state=42
        )
        random_search.fit(X, y)
        best_model = random_search.best_estimator_
        best_params = random_search.best_params_
        best_score = random_search.best_score_

    elif method == 'grid search':
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_dist,
            cv=5,
            scoring="accuracy",
            n_jobs=-1,
            verbose=2,
        )
        grid_search.fit(X, y)
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_

    return best_model, best_params, best_score

def ml_model(X_train,y_train, X_test, y_test,  input_model, tuning = False, scale = 'MinMax'):
    ml_chain = {
        'logistic regression': LogisticRegression(),
        'random forest': RandomForestClassifier(),
        'decision tree': DecisionTreeClassifier(),
        'gradient boosting': GradientBoostingClassifier(),
        'xgboost': xgb.XGBClassifier(),
        # 'svm':
    }
    if input_model.lower() not in ml_chain:
        raise ValueError(f"Unsupported model type: {input_model}")

    input_model = input_model.lower()
    model = ml_chain[input_model]

    if input_model == 'logistic regression':
        if scale == 'MinMax':
            pass
        #     scaler = MinMaxScaler()
        #     scaler.fit(X_train)
        #     X_train = scaler.transform(X_train)
        #     X_test = scaler.transform(X_test)
        # elif scale == 'Standard':
        #     scaler = StandardScaler()
        #     scaler.fit(X_train)
        #     X_train = scaler.transform(X_train)
        #     X_test = scaler.transform(X_test)

    if tuning:
        best_model, best_params, best_score = hyper_tuning(X_train,y_train, input_model,model)
        y_pred_best = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc_score = roc_auc_score(y_test, y_proba)

        tuning_scores = {
        'accuracy' : accuracy_score(y_test, y_pred_best),
        'confusion' : confusion_matrix(y_test, y_pred_best),
        'recall' : recall_score(y_test, y_pred_best),
        'precision' : precision_score(y_test, y_pred_best),
        'f1' : f1_score(y_test, y_pred_best),
        'auc score' : auc_score,
        }

        # return acc_score, rec_score, prec_score, f_score, confusion_mat, best_model, best_params
        return tuning_scores, best_model, best_params, best_score, fpr, tpr

    else:
        model.fit(X_train, y_train)
        skf_scores = strat_k_fold(X_train, y_train,model)

        return skf_scores
#YAML file for parameter
def save_param(model, params, path_file):
    import yaml
    # If file exists
    if os.path.exists(path_file):
        with open(path_file, 'r') as f:
            all_params = yaml.load(f, Loader=yaml.FullLoader)
    else:
        all_params = {}
    #update with new params
    all_params[model] = params
    #save it to file
    with open(path_file, 'w') as f:
        yaml.dump(all_params, f)


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

