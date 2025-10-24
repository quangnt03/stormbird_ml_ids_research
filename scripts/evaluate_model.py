#!/usr/bin/env python3
"""
evaluate_model.py
Evaluate trained model on test data and generate metrics and plots
"""

import os
import json
import yaml
import argparse
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from pathlib import Path
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve
)

# Import resource monitoring utilities
import sys
sys.path.append(os.path.dirname(__file__))
from utils.resource_monitor import ResourceMonitor

# Use non-GUI backend for headless systems
plt.switch_backend('Agg')


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_test_data(processed_dir):
    """Load preprocessed test data"""
    processed_dir = Path(processed_dir)
    test_path = processed_dir / 'test.parquet'

    if not test_path.exists():
        raise FileNotFoundError(f"Preprocessed test data not found: {test_path}")

    print(f"Loading test data from: {test_path}")
    test_df = pd.read_parquet(test_path)

    # Split features and labels
    X_test = test_df.drop(columns=['label'])
    y_test = test_df['label']

    print(f"Test data loaded: {X_test.shape}")
    print(f"Class distribution:")
    print(y_test.value_counts())

    return X_test, y_test


def evaluate_model(config_path):
    """Main evaluation pipeline"""

    # Load configuration
    config = load_config(config_path)
    print(f"Loaded config from: {config_path}")

    # Extract config parameters
    model_path = Path(config['output']['model_path'])
    processed_dir = config['dataset']['processed_dir']
    metrics_path = Path(config['output']['metrics_path'])
    confusion_matrix_path = Path(config['output']['confusion_matrix_path'])
    classification_report_path = Path(config['output']['classification_report_path'])

    # Load model
    print(f"\n=== Loading trained model ===")
    if not model_path.exists():
        raise FileNotFoundError(f"Trained model not found: {model_path}")

    model = joblib.load(model_path)
    print(f"Model loaded from: {model_path}")

    # Load test data
    print(f"\n=== Loading test data ===")
    X_test, y_test = load_test_data(processed_dir)

    # Make predictions with resource monitoring
    print(f"\n=== Making predictions ===")
    with ResourceMonitor() as monitor:
        y_pred = model.predict(X_test)

    test_metrics = monitor.get_metrics()
    inference_time = test_metrics['elapsed_time_sec']
    mean_infer_time_sec = inference_time / len(X_test)

    print(f"Predictions completed in {inference_time:.4f} seconds")
    print(f"Average inference time per sample: {mean_infer_time_sec:.6f} seconds")

    # Try to get prediction probabilities for ROC AUC
    try:
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        has_proba = True
    except:
        has_proba = False
        y_pred_proba = None

    # Calculate metrics
    print(f"\n=== Calculating metrics ===")

    # Basic metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Binary metrics (if applicable)
    if len(np.unique(y_test)) == 2:
        precision_binary = precision_score(y_test, y_pred)
        recall_binary = recall_score(y_test, y_pred)
        f1_binary = f1_score(y_test, y_pred)
        if has_proba:
            roc_auc = roc_auc_score(y_test, y_pred_proba)
        else:
            roc_auc = None
    else:
        precision_binary = None
        recall_binary = None
        f1_binary = None
        roc_auc = None

    print(f"Accuracy: {accuracy:.6f}")
    print(f"Precision (weighted): {precision:.6f}")
    print(f"Recall (weighted): {recall:.6f}")
    print(f"F1 Score (weighted): {f1:.6f}")
    if f1_binary is not None:
        print(f"F1 Score (binary): {f1_binary:.6f}")
    if roc_auc is not None:
        print(f"ROC AUC: {roc_auc:.6f}")

    # Generate classification report
    print(f"\n=== Classification Report ===")
    class_report = classification_report(y_test, y_pred)
    print(class_report)

    # Save classification report
    classification_report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(classification_report_path, 'w') as f:
        f.write(class_report)
    print(f"Classification report saved to: {classification_report_path}")

    # Generate and save confusion matrix
    print(f"\n=== Generating confusion matrix ===")
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap='Blues')
    plt.title(f'Confusion Matrix - {config["model"]["type"].upper()}')
    plt.tight_layout()

    confusion_matrix_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(confusion_matrix_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to: {confusion_matrix_path}")

    # Generate ROC curve if possible
    if has_proba and roc_auc is not None:
        print(f"\n=== Generating ROC curve ===")
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {config["model"]["type"].upper()}')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()

        roc_path = confusion_matrix_path.parent / f"{confusion_matrix_path.stem.replace('confusion_matrix', 'roc_curve')}.png"
        plt.savefig(roc_path, dpi=150)
        plt.close()
        print(f"ROC curve saved to: {roc_path}")

    # Load training metadata if available to include training metrics
    metadata_path = model_path.parent / f"{model_path.stem}_metadata.json"
    training_metrics = {}
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            training_metrics = json.load(f)

    # Save comprehensive metrics to JSON
    print(f"\n=== Saving metrics ===")
    metrics = {
        # Test performance metrics
        'accuracy': float(accuracy),
        'precision_weighted': float(precision),
        'recall_weighted': float(recall),
        'f1_weighted': float(f1),
        'f1_test': float(f1),  # Alias for consistency with training metrics
        'test_samples': int(len(X_test)),
        'num_features': int(X_test.shape[1]),

        # Test inference metrics
        'mean_infer_time_sec': float(mean_infer_time_sec),
        'total_inference_time_sec': float(inference_time),

        # Test resource metrics
        'cpu_usage_percent_test': float(test_metrics['cpu_usage_percent']) if test_metrics['cpu_usage_percent'] else None,
        'peak_memory_mb_test': float(test_metrics['peak_memory_mb']) if test_metrics['peak_memory_mb'] else None,
        'gpu_util_percent_test': float(test_metrics['gpu_util_percent']) if test_metrics['gpu_util_percent'] else None,
        'gpu_mem_mb_test': float(test_metrics['gpu_mem_mb']) if test_metrics['gpu_mem_mb'] else None,
    }

    # Add binary classification metrics if applicable
    if f1_binary is not None:
        metrics['precision_binary'] = float(precision_binary)
        metrics['recall_binary'] = float(recall_binary)
        metrics['f1_binary'] = float(f1_binary)

    if roc_auc is not None:
        metrics['roc_auc'] = float(roc_auc)

    # Include training metrics from metadata
    if training_metrics:
        metrics['f1_train'] = training_metrics.get('f1_train')
        metrics['f1_val'] = training_metrics.get('f1_val')
        metrics['f1_cv_mean'] = training_metrics.get('f1_cv_mean')
        metrics['f1_cv_std'] = training_metrics.get('f1_cv_std')
        metrics['train_time_sec'] = training_metrics.get('train_time_sec')
        metrics['cpu_usage_percent_train'] = training_metrics.get('cpu_usage_percent_train')
        metrics['peak_memory_mb_train'] = training_metrics.get('peak_memory_mb_train')
        metrics['gpu_util_percent_train'] = training_metrics.get('gpu_util_percent_train')
        metrics['gpu_mem_mb_train'] = training_metrics.get('gpu_mem_mb_train')

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")

    print("\n[SUCCESS] Evaluation completed successfully!")
    print(f"\nSummary:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  F1 Score (weighted): {f1:.4f}")
    if f1_binary is not None:
        print(f"  F1 Score (binary): {f1_binary:.4f}")
    if roc_auc is not None:
        print(f"  ROC AUC: {roc_auc:.4f}")

    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate trained model on test data'
    )
    parser.add_argument(
        '--config',
        required=True,
        help='Path to configuration YAML file'
    )

    args = parser.parse_args()
    evaluate_model(args.config)