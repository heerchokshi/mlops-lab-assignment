import pandas as pd
import yaml
import argparse
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

def train_model(config_path):
    """
    Trains a model based on the specified parameters and saves it.
    """
    # Load parameters from YAML
    with open(config_path) as f:
        params = yaml.safe_load(f)

    train_data_path = params['data']['train_path']
    target_col = params['preprocess']['target_col']
    model_name = params['model']['name']
    model_params = params['model'][model_name]
    model_dir = os.path.dirname(params['evaluate']['model_path'])

    print("Loading training data...")
    train_df = pd.read_csv(train_data_path)

    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]

    print(f"Training model: {model_name}...")
    
    if model_name == 'random_forest':
        model = RandomForestClassifier(**model_params)
    elif model_name == 'logistic_regression':
        model = LogisticRegression(**model_params, max_iter=500)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    model.fit(X_train, y_train)

    # Create models directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Save the model
    model_path = params['evaluate']['model_path']
    print(f"Saving model to {model_path}")
    joblib.dump(model, model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml", help="Path to the parameters file")
    args = parser.parse_args()
    train_model(config_path=args.config)