from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb

def get_rf_classifier(random_state=42):
    """Returns a Random Forest Classifier."""
    return RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)

def get_xgb_classifier(random_state=42):
    """Returns an XGBoost Classifier."""
    return xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        eval_metric='logloss',
        random_state=random_state,
        n_jobs=-1
    )

def get_fcnn_classifier(random_state=42):
    """
    Returns a Fully Connected Neural Network (MLP) Classifier.
    Configured with two hidden layers (128 and 64 neurons) and early stopping.
    """
    return MLPClassifier(
        hidden_layer_sizes=(128, 64),
        max_iter=1000,
        early_stopping=True,
        random_state=random_state
    )