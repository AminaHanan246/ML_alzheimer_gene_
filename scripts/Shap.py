import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_predict
import shap

# Shap plots - based on sample-wise predictions
def shap_plot(model,X_train,X_test):
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)
    print(f"\nSHAP Summary Plot ({model}):")
    shap.summary_plot(shap_values, X_test, feature_names=X.columns)
    plt.tight_layout()
    plt.savefig(f"plots/shap_summmary_{model}.png")
    plt.show()

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
    plt.savefig(f"plots/top_features_{model}.png")
    plt.show()

if __name__ == '__main__':
    '''
    Using the cross validated model , the best model (based on accuracy), XGBoost is used for SHAP plot and identifying top features
    '''
    #Load Data
    path = r"D:\BI_prj\ML_biomarker\alzheimers_gene"
    os.chdir(path.replace("\\", "/"))
    os.makedirs('plots', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    data = joblib.load('data_clean.pkl')
    X = data['X']
    y = data['y']
    features = X.columns

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #Models to run - best
    run_model = ['xgboost']

    for run_ml in run_model:
        #Loading model
        model_path = f'{run_ml}_CV_model.pkl'
        if not os.path.exists(model_path):
            print(f"⚠️ Skipping {run_ml} — model file not found: {model_path}")
            continue

        best_model = joblib.load(f'{run_ml}_CV_model.pkl')


        # SHAP plot
        shap_plot(best_model,X_train, X_test)

        #Top features
        plot_top_features(best_model, features,run_ml.capitalize(), 10)
