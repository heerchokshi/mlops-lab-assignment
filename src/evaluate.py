import pandas as pd
import yaml
import argparse
import os
import json
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

def evaluate_model(config_path):
    """
    Evaluates the trained model, saves metrics and plots.
    """
    # Load parameters from YAML
    with open(config_path) as f:
        params = yaml.safe_load(f)

    test_data_path = params['data']['test_path']
    target_col = params['preprocess']['target_col']
    model_path = params['evaluate']['model_path']
    metrics_path = params['evaluate']['metrics_path']
    plots_path = params['evaluate']['plots_path']

    print("Loading test data and model...")
    test_df = pd.read_csv(test_data_path)
    model = joblib.load(model_path)

    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    print("Making predictions...")
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Create metrics directory if it doesn't exist
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

    # Save metrics
    metrics = {"accuracy": accuracy, "f1_score": f1}
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {metrics_path}")

    # Create plots directory if it doesn't exist
    os.makedirs(plots_path, exist_ok=True)

    # Plot and save confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    cm_path = os.path.join(plots_path, 'confusion_matrix.png')
    plt.savefig(cm_path)
    print(f"Confusion matrix saved to {cm_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml", help="Path to the parameters file")
    args = parser.parse_args()
    evaluate_model(config_path=args.config)