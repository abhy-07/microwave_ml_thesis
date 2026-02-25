from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

def get_rf_classifier(random_state=42):
    """
    Returns a Random Forest Classifier.
    Use this when your target variable is categorical (e.g., 'Tumor' vs 'Reference').
    """
    # n_jobs=-1 tells the model to use all available CPU cores for faster training
    return RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)

def get_rf_regressor(random_state=42):
    """
    Returns a Random Forest Regressor.
    Use this later when predicting continuous tissue properties or thickness (for MAE/RMSE).
    """
    return RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1)