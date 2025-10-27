import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, f1_score, recall_score, classification_report,roc_curve,roc_auc_score
from scipy.stats import randint, uniform, zscore
import shap

#Load CSV
def load_csv(filepath):
    try:
        df = pd.read_csv(filepath)
        print(df.head())  # Display the first few rows
    except FileNotFoundError:
        print("Error: The specified CSV file was not found.")
    except pd.errors.EmptyDataError:
        print("Error: The CSV file is empty.")
    except pd.errors.ParserError:
        print("Error: There was an issue parsing the CSV file. Check for malformed lines or incorrect delimiters.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return df

#Basic EDA
def basic_eda(data):
    summary = {
        "head": data.head(5),
        "shape": data.shape,
        "info": data.info(),
        "describe": data.describe(),
        "columns": data.columns.tolist(),
        "missing_values": data.isnull().sum()
    }
    return summary

#class distubution
def plot_class_dist(data,x,hue):
    # Check class imbalance
    plt.figure(figsize=(5, 5))
    sns.countplot(x=x, data=data, edgecolor='black', hue=hue)
    plt.xlabel(f"{x.capitalize()}")
    plt.ylabel("Frequency")
    plt.title('Class Distribution')
    plt.tight_layout()
    plt.savefig("plots/Label freq.png")
    plt.show()
    print("\nDataset distribution:", data[x].value_counts())

#Outlier removal
def outlier_remove(method, data):
    if method.lower() == 'iqr':
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1

        iqr = ~((data > (Q3 + 1.5 * IQR)) | (data < (Q1 - 1.5 * IQR))).any(axis=1)
        outliers = len(data) - len(data[iqr])

        return iqr, outliers
    elif method.lower() == 'z score':
        pass
        # z = np.abs(zscore(data['age']))
        # print(z)
        #
# Class imbalance fix
def fix_imbalance(method,data, target_column):
    if method.lower() == 'resampling':
        majority_class = data[data[target_column] == 0]
        minority_class = data[data[target_column] == 1]

        # Downsample majority class to match minority class size
        majority_downsampled = resample(
            majority_class,
            replace=False,  # sample without replacement
            n_samples=len(minority_class),  # match minority class
            random_state=42  # reproducibility
        )

        # Combine minority class with downsampled majority class
        df_balanced = pd.concat([minority_class, majority_downsampled])

        # Shuffle rows
        df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

        # Check class distribution
        return df_balanced

# K-fold cross validation
def strat_k_fold(X, y, model, n_splits=10, shuffle= True, random_state=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    metrics = ['accuracy','recall','precision','f1','roc_auc']
    skf_scores = {}
    for metric in metrics:
        score = cross_val_score(model, X, y, cv=skf, scoring=metric)
        skf_scores[metric] = [score.mean(), score.std()]

    return skf_scores

# Hyperparameter tuning
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

#Evaluation metrics
def report_metric(scores, metrics, result):
    for metric in metrics:
        if metric in scores:
            value = scores[metric]
            # Handle CV metrics (mean ± std) stored as list/tuple
            if isinstance(value, (list, tuple)) and len(value) == 2:
                print(f"\n {metric.capitalize()} Score       : {value[0]:.4f} ± {value[1]:.4f}")
                result[metric] = value[0]
                result[f"{metric}_std"] = value[1]
            else:
                print(f"\n {metric.capitalize()} Score       : {value:.4f}")
                result[metric] = value

    if 'confusion' in scores:
        print(f"Confusion Matrix     :\n{scores['confusion']}")

    if 'auc score' in scores:
        print(f"AUC Score            : {scores['auc score']:.4f}")

    return result

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

# Model training
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
            scaler = MinMaxScaler()
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)
        elif scale == 'Standard':
            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)

    if tuning:
        best_model, best_params, best_score = hyper_tuning(X_train,y_train, input_model,model)
        y_pred_best = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)

        tuning_scores = {
        'accuracy' : accuracy_score(y_test, y_pred_best),
        'confusion' : confusion_matrix(y_test, y_pred_best),
        'recall' : recall_score(y_test, y_pred_best),
        'precision' : precision_score(y_test, y_pred_best),
        'f1' : f1_score(y_test, y_pred_best),
        'auc score' : roc_auc_score(y_test, y_proba),
        'fpr':fpr,
        'tpr':tpr,
        }

        # return acc_score, rec_score, prec_score, f_score, confusion_mat, best_model, best_params
        return tuning_scores, best_model, best_params, best_score

    else:
        model.fit(X_train, y_train)
        skf_scores = strat_k_fold(X_train, y_train,model)

        return skf_scores


#Plot confusion matrix
def conf_mat(metric, model):
    plt.figure(figsize=(4, 3))  # Adjust figure size as needed
    sns.heatmap(
        metric,
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
    plt.ylabel('Predicted Values', fontsize=16)
    plt.title('Confusion Matrix', fontsize=16)
    plt.tight_layout()
    plt.tight_layout()
    plt.savefig(f"plots/Confusion Matrix_{model}_tuning.png")

# Model is tuned using training data and cross-validated on entire set
def model_tuning_CV(exp,X,y,X_train, y_train, X_test, y_test, input_model, metrics, tuning_results,skf_results):

    print(f"Tuning {input_model}")
    result = {'Model': input_model}

    # Hyperparameter tuning
    tuning_scores, best_model, best_params, best_score, fpr, tpr = ml_model(X_train, y_train, X_test, y_test,
                                                                            input_model, tuning=True)
    joblib.dump(best_model, f"{input_model}_{exp}_tuned_model.pkl")
    save_param(input_model, best_params, f"results/{input_model}_{exp}_bestparams.yml")
    print(f"\nBest Hyperparameters from RandomizedSearchCV {exp.capitalize()} {input_model.capitalize()}:")
    print(best_params)

    # Hyperparameter Evaluation
    print(f"\nBest Parameters Results {input_model} {exp.capitalize()}:")
    result = report_metric(tuning_scores, metrics, result)

    # Storing tuning scores
    tuning_results.append(result)

    # Plot confusion matrix
    conf_mat(tuning_scores['confusion'], input_model)

    # 10-fold Cross validation
    skf_scores = strat_k_fold(X, y, best_model)
    print(f"\n10-Fold Stratified Cross Validation Results {input_model} {exp.capitalize()}:")
    # CV evaluation
    result = report_metric(skf_scores, metrics, result)
    # Storing skf scores
    skf_results.append(result)

    return tuning_results, skf_results,best_model


# Shap plots - based on sample-wise predictions
def shap_plot(model,X_train,X_test,X):
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)
    print(f"\nSHAP Summary Plot ({model}):")
    shap.summary_plot(shap_values, X_test, feature_names=X.columns, show=False)
    plt.savefig(f"plots/shap_summmary.png")
    plt.close()

# Top features for model prediction
def plot_top_features(model, feature_names, title, top_n):
    if hasattr(model, 'coef_'):
        coefs = model.coef_[0]
        abs_coefs = np.abs(coefs)
        top_idx = np.argsort(abs_coefs)[::-1][:top_n]
        top_features = np.array(feature_names)[top_idx]
        top_values = coefs[top_idx]
        x_labels = 'Coefficient Values'
    elif hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        top_idx = np.argsort(importances)[::-1][:top_n]
        top_features = np.array(feature_names)[top_idx]
        top_values = importances[top_idx]
        x_labels = "Feature Importances"

    plt.figure(figsize=(8, 5))
    sns.barplot(x=top_values, y=top_features, palette="viridis")
    plt.title(f"Top {top_n} Features — {title}")
    plt.xlabel(x_labels)
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(f"plots/top_features.png")
    plt.show()

