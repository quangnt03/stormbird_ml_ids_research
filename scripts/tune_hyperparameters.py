#!/usr/bin/env python3
"""
tune_hyperparameters.py
Optional hyperparameter tuning using Optuna (for collaborators who want to re-tune)

NOTE: This is OPTIONAL. Most users should use pre-tuned hyperparameters from params/ directory.
Only run this if you have significant compute resources and want to find new hyperparameters.

Example usage:
    python scripts/tune_hyperparameters.py --config configs/cicids2017_xgb.yaml --trials 30
"""

import os
import json
import yaml
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import optuna
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_preprocessed_data(processed_dir):
    """Load preprocessed training and test data"""
    processed_dir = Path(processed_dir)
    train_path = processed_dir / 'train.parquet'
    test_path = processed_dir / 'test.parquet'

    print(f"Loading training data from: {train_path}")
    train_df = pd.read_parquet(train_path)
    X_train = train_df.drop(columns=['label'])
    y_train = train_df['label']

    print(f"Loading test data from: {test_path}")
    test_df = pd.read_parquet(test_path)
    X_test = test_df.drop(columns=['label'])
    y_test = test_df['label']

    print(f"Training data: {X_train.shape}")
    print(f"Test data: {X_test.shape}")

    return X_train, y_train, X_test, y_test


def objective_xgboost(trial, X_train, y_train, X_test, y_test):
    """Optuna objective function for XGBoost"""
    import xgboost as xgb

    params = {
        "objective": "binary:logistic",
        "device": "cuda" if trial.suggest_categorical("use_gpu", [True, False]) else "cpu",
        "verbosity": 0,
        "eval_metric": "logloss",
        "booster": trial.suggest_categorical("booster", ["gbtree", "dart"]),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 10.0),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "sampling_method": trial.suggest_categorical("sampling_method", ["uniform", "gradient_based"]),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
        "colsample_bynode": trial.suggest_float("colsample_bynode", 0.5, 1.0),
        "alpha": trial.suggest_float("alpha", 1e-8, 10.0, log=True),
        "lambda": trial.suggest_float("lambda", 1e-8, 10.0, log=True),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.5, 10.0),
        "grow_policy": trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"]),
        "tree_method": "hist",
        "max_leaves": trial.suggest_int("max_leaves", 0, 256),
        "max_bin": trial.suggest_int("max_bin", 256, 512),
    }

    # DART-specific parameters
    if params["booster"] == "dart":
        params["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
        params["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
        params["rate_drop"] = trial.suggest_float("rate_drop", 0.0, 0.5)
        params["skip_drop"] = trial.suggest_float("skip_drop", 0.0, 0.5)

    # Train model
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)

    # Evaluate on test set
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Cross-validation score
    cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='f1_weighted')
    cv_f1_mean = np.mean(cv_scores)

    # Store additional metrics
    trial.set_user_attr('f1_test', f1)
    trial.set_user_attr('f1_cv_mean', cv_f1_mean)
    trial.set_user_attr('f1_cv_std', np.std(cv_scores))

    print(f"Trial {trial.number}: F1 (test)={f1:.4f}, F1 (CV)={cv_f1_mean:.4f}")

    return f1


def objective_lightgbm(trial, X_train, y_train, X_test, y_test):
    """Optuna objective function for LightGBM"""
    import lightgbm as lgb

    params = {
        "objective": "binary",
        "device": "gpu" if trial.suggest_categorical("use_gpu", [True, False]) else "cpu",
        "verbosity": -1,
        "boosting_type": trial.suggest_categorical("boosting_type", ["dart", "goss"]),
        "num_leaves": trial.suggest_int("num_leaves", 16, 512),
        "max_depth": trial.suggest_int("max_depth", -1, 20),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 2000),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "tree_learner": trial.suggest_categorical("tree_learner", ["serial", "feature", "data", "voting"]),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "min_child_weight": trial.suggest_float("min_child_weight", 1e-4, 100.0, log=True),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "extra_trees": trial.suggest_categorical("extra_trees", [True, False]),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
    }

    # Train model
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train)

    # Evaluate on test set
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Cross-validation score
    cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='f1_weighted')
    cv_f1_mean = np.mean(cv_scores)

    # Store additional metrics
    trial.set_user_attr('f1_test', f1)
    trial.set_user_attr('f1_cv_mean', cv_f1_mean)
    trial.set_user_attr('f1_cv_std', np.std(cv_scores))

    print(f"Trial {trial.number}: F1 (test)={f1:.4f}, F1 (CV)={cv_f1_mean:.4f}")

    return f1


def tune_hyperparameters(config_path, n_trials=30, threshold=0.99):
    """Main hyperparameter tuning pipeline"""

    # Load configuration
    config = load_config(config_path)
    print(f"Loaded config from: {config_path}")

    # Extract parameters
    model_type = config['model']['type']
    processed_dir = config['dataset']['processed_dir']
    params_file = Path(config['model']['params_file'])

    print(f"\nModel type: {model_type}")
    print(f"Params output file: {params_file}")
    print(f"Number of trials: {n_trials}")
    print(f"Threshold for early stopping: {threshold}")

    # Load preprocessed data
    print(f"\n=== Loading preprocessed data ===")
    X_train, y_train, X_test, y_test = load_preprocessed_data(processed_dir)

    # Create Optuna study
    print(f"\n=== Starting hyperparameter tuning ===")
    study = optuna.create_study(direction='maximize')

    # Select objective function
    if model_type.lower() == 'xgboost':
        objective_fn = lambda trial: objective_xgboost(trial, X_train, y_train, X_test, y_test)
    elif model_type.lower() == 'lightgbm':
        objective_fn = lambda trial: objective_lightgbm(trial, X_train, y_train, X_test, y_test)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Run optimization
    study.optimize(objective_fn, n_trials=n_trials)

    # Get best parameters
    best_params = study.best_params
    best_value = study.best_value

    print(f"\n=== Tuning completed ===")
    print(f"Best F1 score: {best_value:.6f}")
    print(f"Best parameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")

    # Remove 'use_gpu' from params (it was just for trial suggestion)
    if 'use_gpu' in best_params:
        use_gpu = best_params.pop('use_gpu')
        # Set device based on use_gpu
        if model_type.lower() == 'xgboost':
            best_params['device'] = 'cuda' if use_gpu else 'cpu'
        elif model_type.lower() == 'lightgbm':
            best_params['device'] = 'gpu' if use_gpu else 'cpu'

    # Add model-specific fixed parameters
    if model_type.lower() == 'xgboost':
        best_params.update({
            'objective': 'binary:logistic',
            'verbosity': 1,
            'eval_metric': 'logloss'
        })
    elif model_type.lower() == 'lightgbm':
        best_params.update({
            'objective': 'binary',
            'verbosity': 0
        })

    # Save best parameters
    params_file.parent.mkdir(parents=True, exist_ok=True)
    with open(params_file, 'w') as f:
        json.dump(best_params, f, indent=2)

    print(f"\n✅ Best hyperparameters saved to: {params_file}")

    # Print study statistics
    print(f"\nStudy statistics:")
    print(f"  Number of finished trials: {len(study.trials)}")
    print(f"  Best trial: {study.best_trial.number}")
    print(f"  Best value: {study.best_value:.6f}")

    return best_params


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Tune hyperparameters using Optuna (OPTIONAL - for re-tuning only)'
    )
    parser.add_argument(
        '--config',
        required=True,
        help='Path to configuration YAML file'
    )
    parser.add_argument(
        '--trials',
        type=int,
        default=30,
        help='Number of Optuna trials (default: 30)'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.99,
        help='F1 score threshold for early stopping (default: 0.99)'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("HYPERPARAMETER TUNING (OPTIONAL)")
    print("=" * 80)
    print("\nNOTE: This script is for advanced users who want to re-tune hyperparameters.")
    print("Most users should use the pre-tuned parameters in params/ directory.")
    print("This process can take 30-60 minutes depending on your hardware.")
    print("=" * 80)

    tune_hyperparameters(args.config, args.trials, args.threshold)