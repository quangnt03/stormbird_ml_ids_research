#!/usr/bin/env python3
"""
train_model.py
Train XGBoost or LightGBM model using preprocessed data and best hyperparameters
"""

import os
import json
import yaml
import argparse
import joblib
import pandas as pd
import numpy as np
import time
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import f1_score

# Import resource monitoring utilities
import sys
sys.path.append(os.path.dirname(__file__))
from utils.resource_monitor import ResourceMonitor


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_best_params(params_path):
    """Load best hyperparameters from JSON file"""
    with open(params_path, 'r') as f:
        return json.load(f)


def load_preprocessed_data(processed_dir):
    """Load preprocessed training data"""
    processed_dir = Path(processed_dir)
    train_path = processed_dir / 'train.parquet'

    if not train_path.exists():
        raise FileNotFoundError(f"Preprocessed training data not found: {train_path}")

    print(f"Loading training data from: {train_path}")
    train_df = pd.read_parquet(train_path)

    # Split features and labels
    X_train = train_df.drop(columns=['label'])
    y_train = train_df['label']

    print(f"Training data loaded: {X_train.shape}")
    print(f"Class distribution:")
    print(y_train.value_counts())

    return X_train, y_train


def train_model(config_path):
    """Main training pipeline"""

    # Load configuration
    config = load_config(config_path)
    print(f"Loaded config from: {config_path}")

    # Extract config parameters
    model_type = config['model']['type']
    params_file = config['model']['params_file']
    processed_dir = config['dataset']['processed_dir']
    model_output_path = Path(config['output']['model_path'])

    print(f"\nModel type: {model_type}")
    print(f"Hyperparameters file: {params_file}")

    # Load best hyperparameters
    best_params = load_best_params(params_file)
    print(f"\nLoaded hyperparameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")

    # Load preprocessed data
    print(f"\n=== Loading preprocessed data ===")
    X_train_full, y_train_full = load_preprocessed_data(processed_dir)

    # Split into train and validation sets for monitoring
    print(f"\n=== Creating validation split ===")
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=0.2,
        random_state=42,
        stratify=y_train_full
    )
    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")

    # Initialize model and prepare for training with history
    print(f"\n=== Initializing {model_type} model ===")
    training_history = []

    if model_type.lower() == 'xgboost':
        import xgboost as xgb

        # Create eval result dictionary to capture history
        eval_results = {}

        model = xgb.XGBClassifier(**best_params)

        # Train model with evaluation and resource monitoring
        print(f"\n=== Training model with validation monitoring ===")
        with ResourceMonitor() as monitor:
            model.fit(
                X_train, y_train,
                eval_set=[(X_train, y_train), (X_val, y_val)],
                verbose=False
            )

        # Get resource metrics
        train_metrics = monitor.get_metrics()
        training_time = train_metrics['elapsed_time_sec']

        # Extract training history from model
        eval_results = model.evals_result()

        # Convert to standard format
        for i in range(len(eval_results['validation_0']['logloss'])):
            training_history.append({
                'iteration': i + 1,
                'train_logloss': float(eval_results['validation_0']['logloss'][i]),
                'val_logloss': float(eval_results['validation_1']['logloss'][i])
            })

    elif model_type.lower() == 'lightgbm':
        import lightgbm as lgb

        # Create callback to capture history
        eval_results = {}

        model = lgb.LGBMClassifier(**best_params)

        # Train model with evaluation and resource monitoring
        print(f"\n=== Training model with validation monitoring ===")
        with ResourceMonitor() as monitor:
            model.fit(
                X_train, y_train,
                eval_set=[(X_train, y_train), (X_val, y_val)],
                eval_names=['train', 'valid'],
                callbacks=[lgb.record_evaluation(eval_results)]
            )

        # Get resource metrics
        train_metrics = monitor.get_metrics()
        training_time = train_metrics['elapsed_time_sec']

        # Convert to standard format
        metric_name = list(eval_results['train'].keys())[0]  # Usually 'binary_logloss' or 'multi_logloss'
        for i in range(len(eval_results['train'][metric_name])):
            training_history.append({
                'iteration': i + 1,
                'train_logloss': float(eval_results['train'][metric_name][i]),
                'val_logloss': float(eval_results['valid'][metric_name][i])
            })
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    print(f"Training completed in {training_time:.2f} seconds")
    print(f"Total iterations: {len(training_history)}")

    # Calculate F1 scores
    print(f"\n=== Calculating F1 scores ===")
    y_train_pred = model.predict(X_train)
    f1_train = f1_score(y_train, y_train_pred, average='weighted')
    print(f"F1 Score (train): {f1_train:.6f}")

    y_val_pred = model.predict(X_val)
    f1_val = f1_score(y_val, y_val_pred, average='weighted')
    print(f"F1 Score (validation): {f1_val:.6f}")

    # Perform cross-validation (on full training set)
    print(f"\n=== Running 5-fold cross-validation ===")
    cv_scores = cross_val_score(model, X_train_full, y_train_full, cv=5, scoring='f1_weighted', n_jobs=-1)
    f1_cv_mean = np.mean(cv_scores)
    f1_cv_std = np.std(cv_scores)
    print(f"F1 CV Mean: {f1_cv_mean:.6f} (+/- {f1_cv_std:.6f})")

    # Save model
    print(f"\n=== Saving model ===")
    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_output_path)
    print(f"Model saved to: {model_output_path}")

    # Save training history
    print(f"\n=== Saving training history ===")
    history_path = config['output'].get('training_history_path')
    if history_path:
        history_path = Path(history_path)
        history_path.parent.mkdir(parents=True, exist_ok=True)
        with open(history_path, 'w') as f:
            json.dump(training_history, f, indent=2)
        print(f"Training history saved to: {history_path}")
    else:
        # Fallback path if not specified in config
        history_path = model_output_path.parent / f"{model_output_path.stem}_training_history.json"
        history_path.parent.mkdir(parents=True, exist_ok=True)
        with open(history_path, 'w') as f:
            json.dump(training_history, f, indent=2)
        print(f"Training history saved to: {history_path}")

    # Save training metadata with comprehensive metrics
    metadata = {
        'model_type': model_type,
        'training_samples': len(X_train),
        'validation_samples': len(X_val),
        'num_features': X_train.shape[1],
        'total_iterations': len(training_history),
        'hyperparameters': best_params,
        # Performance metrics
        'f1_train': float(f1_train),
        'f1_val': float(f1_val),
        'f1_cv_mean': float(f1_cv_mean),
        'f1_cv_std': float(f1_cv_std),
        # Training resource metrics
        'train_time_sec': float(training_time),
        'cpu_usage_percent_train': float(train_metrics['cpu_usage_percent']) if train_metrics['cpu_usage_percent'] else None,
        'peak_memory_mb_train': float(train_metrics['peak_memory_mb']) if train_metrics['peak_memory_mb'] else None,
        'gpu_util_percent_train': float(train_metrics['gpu_util_percent']) if train_metrics['gpu_util_percent'] else None,
        'gpu_mem_mb_train': float(train_metrics['gpu_mem_mb']) if train_metrics['gpu_mem_mb'] else None
    }

    metadata_path = model_output_path.parent / f"{model_output_path.stem}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Training metadata saved to: {metadata_path}")

    print("\n[SUCCESS] Training completed successfully!")

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train model using preprocessed data and best hyperparameters'
    )
    parser.add_argument(
        '--config',
        required=True,
        help='Path to configuration YAML file'
    )

    args = parser.parse_args()
    train_model(args.config)
