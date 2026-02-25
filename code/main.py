import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from data.loader import create_master_dataset
from models.baselines import get_rf_classifier
from config import OUTPUT_DIR


def prepare_binary_data(dataset):
    print("\nRe-labeling data for Binary Classification (Tumor vs. Reference)...")

    # Map 'Left Tumor' and 'Right Tumor' to simply 'Tumor'
    dataset['label'] = dataset['label'].apply(lambda x: 'Tumor' if 'Tumor' in x else 'Reference')

    print("\nNew Class Distribution:")
    print(dataset['label'].value_counts())

    # Separate features and target
    X = dataset.drop(columns=['filename', 'label'])
    y = dataset['label']

    # Split the data, keeping the 2:1 class ratio consistent
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"\nTraining set shape: X={X_train.shape}, y={y_train.shape}")
    print(f"Testing set shape:  X={X_test.shape}, y={y_test.shape}")

    return X_train, X_test, y_train, y_test


def train_and_evaluate(X_train, X_test, y_train, y_test):
    print("\nInitializing Random Forest Classifier...")
    model = get_rf_classifier()

    print("Training binary model... (this might take a few seconds)")
    model.fit(X_train, y_train)

    print("Generating predictions on the unseen test set...")
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n--- BINARY MODEL EVALUATION ---")
    print(f"Overall Accuracy: {accuracy * 100:.2f}%\n")

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred))

    return model


def plot_feature_importances(model, dataset):
    print("\nExtracting binary feature importances...")
    importances = model.feature_importances_

    feature_cols = [col for col in dataset.columns if col.startswith('feature_')]
    num_freq_points = len(feature_cols) // 2

    s11_importance = importances[:num_freq_points]
    s21_importance = importances[num_freq_points:]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].plot(range(num_freq_points), s11_importance, color='purple', linewidth=2)
    axes[0].set_title('Binary $S_{11}$ Feature Importance (Reflection)')
    axes[0].set_xlabel('Frequency Point Index')
    axes[0].set_ylabel('Importance Score')
    axes[0].grid(True, linestyle='--', alpha=0.7)

    axes[1].plot(range(num_freq_points), s21_importance, color='orange', linewidth=2)
    axes[1].set_title('Binary $S_{21}$ Feature Importance (Transmission)')
    axes[1].set_xlabel('Frequency Point Index')
    axes[1].set_ylabel('Importance Score')
    axes[1].grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = OUTPUT_DIR / f"binary_feature_importance_{timestamp}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"SUCCESS: Binary feature importance plot saved to: {save_path}")

    plt.show()


def main():
    print("Starting Microwave Body Composition Analysis Pipeline...")

    # 1. Load the master dataset
    dataset = create_master_dataset()

    # 2. Re-label to binary and split
    X_train, X_test, y_train, y_test = prepare_binary_data(dataset)

    # 3. Train and evaluate
    trained_model = train_and_evaluate(X_train, X_test, y_train, y_test)

    # 4. Plot the new importances
    plot_feature_importances(trained_model, dataset)


if __name__ == "__main__":
    main()