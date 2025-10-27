import os
import joblib
from sklearn.model_selection import train_test_split, cross_val_predict
from utilis import (
shap_plot,
plot_top_features
)

if __name__ == '__main__':
    '''
    Using the cross validated model , the best model (based on accuracy), XGBoost is used for SHAP plot and identifying top features
    '''
    #Load Data
    data = joblib.load('data_clean.pkl')
    X = data['X']
    y = data['y']
    features = X.columns

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #Models to run - best
    model = ['xgboost']


    #Loading model
    model_path = f'{model}_CV_model.pkl'
    if not os.path.exists(model_path):
        print(f"⚠️ Skipping {model} — model file not found: {model_path}")

    best_model = joblib.load(f'{model}_CV_model.pkl')


    # SHAP plot
    shap_plot(best_model,X_train, X_test)

    #Top features
    plot_top_features(best_model, features,model.capitalize(), 10)
