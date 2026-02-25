import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from data.loader import create_master_dataset
from models.baselines import get_rf_classifier, get_xgb_classifier, get_fcnn_classifier
from config import OUTPUT_DIR


def prepare_binary_data(dataset):
    print("\nRe-labeling data for Binary Classification (Tumor vs. Reference)...")
    dataset['label'] = dataset['label'].apply(lambda x: 'Tumor' if 'Tumor' in x else 'Reference')

    le = LabelEncoder()
    dataset['encoded_label'] = le.fit_transform(dataset['label'])

    X = dataset.drop(columns=['filename', 'label', 'encoded_label'])
    y = dataset['encoded_label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Neural Networks require feature scaling to converge properly
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, le


def train_and_evaluate(model_name, model, X_train, X_test, y_train, y_test, target_names):
    print(f"\nTraining {model_name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=target_names)

    results_text = f"{'=' * 40}\nModel: {model_name}\nOverall Accuracy: {acc * 100:.2f}%\n\n"
    results_text += f"Confusion Matrix:\n{cm}\n\nDetailed Classification Report:\n{report}\n\n"

    print(results_text)
    return results_text, cm


def save_proof(results_dict, target_names):
    """Saves text reports and generates 1x3 side-by-side heatmaps."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. Save Text Log
    log_path = OUTPUT_DIR / f"baseline_metrics_{timestamp}.txt"
    with open(log_path, "w") as file:
        file.write("Microwave Body Composition Assessment - Deterministic Baselines\n")
        file.write(f"Run Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        for name, data in results_dict.items():
            file.write(data['text'])
    print(f"SUCCESS: Metrics log saved to -> {log_path}")

    # 2. Save Visual Heatmaps
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    models = list(results_dict.keys())
    cmaps = ['Blues', 'Oranges', 'Greens']

    for i, model_name in enumerate(models):
        sns.heatmap(results_dict[model_name]['cm'], annot=True, fmt='d', cmap=cmaps[i],
                    ax=axes[i], xticklabels=target_names, yticklabels=target_names, cbar=False)
        axes[i].set_title(f'{model_name}\nConfusion Matrix')
        axes[i].set_xlabel('Predicted Label')
        axes[i].set_ylabel('True Label')

    plt.tight_layout()
    plot_path = OUTPUT_DIR / f"baseline_heatmaps_{timestamp}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"SUCCESS: Heatmaps saved to -> {plot_path}")
    plt.show()


def main():
    print("Starting Microwave Body Composition Analysis Pipeline...")

    dataset = create_master_dataset()
    X_train, X_test, y_train, y_test, le = prepare_binary_data(dataset)
    target_names = le.classes_

    # Dictionary to store all our results for the save function
    results = {}

    # 1. Random Forest
    rf_text, rf_cm = train_and_evaluate("Random Forest", get_rf_classifier(), X_train, X_test, y_train, y_test,
                                        target_names)
    results["Random Forest"] = {'text': rf_text, 'cm': rf_cm}

    # 2. XGBoost
    xgb_text, xgb_cm = train_and_evaluate("XGBoost", get_xgb_classifier(), X_train, X_test, y_train, y_test,
                                          target_names)
    results["XGBoost"] = {'text': xgb_text, 'cm': xgb_cm}

    # 3. Fully Connected Neural Network
    fcnn_text, fcnn_cm = train_and_evaluate("Fully Connected NN", get_fcnn_classifier(), X_train, X_test, y_train,
                                            y_test, target_names)
    results["Fully Connected NN"] = {'text': fcnn_text, 'cm': fcnn_cm}

    # Generate Proof
    save_proof(results, target_names)


if __name__ == "__main__":
    main()