import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, f1_score, recall_score, classification_report,roc_curve,roc_auc_score

if __name__=='__main__':
    '''
    Using the cross validated model, Roc-Auc curve is predicted for 3 models - worst, mid, best
    '''
    # Load data
    path = r"D:\BI_prj\ML_biomarker\alzheimers_gene"
    os.chdir(path.replace("\\", "/"))
    os.makedirs('plots', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    data = joblib.load('data_clean.pkl')
    X = data['X']
    y = data['y']

    #Train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #List of models to be run - worst, mid, best
    run_model = ['random forest','logistic regression','xgboost']

    plt.figure(figsize=(8, 6))

    #ROC plotting
    for run_ml in run_model:
        #Load stored model
        model_path = f'{run_ml}_CV_model.pkl'
        if not os.path.exists(model_path):
            print(f"⚠️ Skipping {run_ml} — model file not found: {best_model}")
            continue #Skips to next iteration

        best_model = joblib.load(f'{run_ml}_CV_model.pkl')

        #ROC estimation
        y_pred_best = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc_score = roc_auc_score(y_test, y_proba)
        print(f"{run_ml} AUC score: {auc_score:.4f}")
        #ploting ROC
        plt.plot(fpr, tpr, label=f"{run_ml} without PCA (AUC = {auc_score:.4f})", linewidth=2)

    # Plot chance line
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.title("ROC Curves — Alzheimer's Disease Classification", fontsize=13)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/ROC_alzgene.png")
    plt.show()
