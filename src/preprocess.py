import pandas as pd
import yaml
import argparse
import os
from sklearn.model_selection import train_test_split

def preprocess_data(config_path):
    """
    Loads data, splits it into train/test sets, and saves them.
    """
    # Load parameters from YAML
    with open(config_path) as f:
        params = yaml.safe_load(f)

    raw_data_path = params['data']['raw_path']
    processed_data_path = params['data']['processed_path']
    train_path = params['data']['train_path']
    test_path = params['data']['test_path']
    target_col = params['preprocess']['target_col']
    test_size = params['training']['test_size']
    random_state = params['training']['random_state']

    print("Loading raw data...")
    df = pd.read_csv(raw_data_path, sep=';')

    # Splitting data
    print("Splitting data into train and test sets...")
    train, test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[target_col] # Stratify to maintain target distribution
    )

    # Create processed directory if it doesn't exist
    os.makedirs(processed_data_path, exist_ok=True)

    # Save processed data
    print(f"Saving train data to {train_path}")
    train.to_csv(train_path, index=False)
    
    print(f"Saving test data to {test_path}")
    test.to_csv(test_path, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml", help="Path to the parameters file")
    args = parser.parse_args()
    preprocess_data(config_path=args.config)