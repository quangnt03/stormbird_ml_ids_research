#!/usr/bin/env python3
"""
preprocess_cicids2017.py
Preprocessing pipeline for CICIDS2017 dataset that matches the notebook workflow
Includes: data loading, train/test split, scaling, undersampling, oversampling (SMOTE)
"""

import os
import argparse
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_raw_data(raw_path):
    """Load all CSV/Parquet files from raw data directory"""
    print(f"\n=== Loading raw data from {raw_path} ===")

    data_files = []
    raw_path = Path(raw_path)

    # Find all CSV and parquet files
    for ext in ['*.csv', '*.parquet']:
        data_files.extend(list(raw_path.glob(ext)))

    if not data_files:
        raise FileNotFoundError(f"No data files found in {raw_path}")

    print(f"Found {len(data_files)} data files:")
    for f in data_files:
        print(f"  - {f.name}")

    # Load and concatenate all files
    dfs = []
    for file_path in data_files:
        if file_path.suffix == '.csv':
            df = pd.read_csv(file_path)
        else:
            df = pd.read_parquet(file_path)

        # Strip whitespace from column names
        df.columns = df.columns.str.strip()
        dfs.append(df)
        print(f"  Loaded {file_path.name}: {df.shape}")

    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"\nCombined dataset shape: {combined_df.shape}")

    return combined_df


def clean_data(df):
    """Improved data cleaning with smart handling of infinite values"""
    print("\n=== Cleaning data ===")

    initial_rows = len(df)
    initial_cols = len(df.columns)

    # Separate label column if it exists
    label_col = None
    for col in ['Label', 'label', ' Label']:
        if col in df.columns:
            label_col = col
            break

    # Store labels separately
    labels = df[label_col].copy() if label_col else None
    feature_cols = [col for col in df.columns if col != label_col]

    # Step 1: Check for columns with too many infinite values
    inf_cols = []
    for col in feature_cols:
        if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
            inf_count = np.isinf(df[col]).sum()
            if inf_count > 0:
                inf_pct = inf_count / len(df) * 100
                if inf_pct > 50:  # Drop columns with >50% infinite values
                    inf_cols.append(col)
                    print(f"  Dropping column '{col}': {inf_pct:.1f}% infinite values")

    if inf_cols:
        df = df.drop(columns=inf_cols)
        feature_cols = [col for col in df.columns if col != label_col]
        print(f"Dropped {len(inf_cols)} columns with excessive infinite values")

    # Step 2: Replace remaining infinite values with column-specific values
    print("Replacing infinite values with column max/min...")
    for col in feature_cols:
        if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
            # Get finite values for this column
            finite_mask = np.isfinite(df[col])
            if finite_mask.any():
                col_max = df.loc[finite_mask, col].max()
                col_min = df.loc[finite_mask, col].min()

                # Replace +inf with max, -inf with min
                df[col] = df[col].replace([np.inf], col_max)
                df[col] = df[col].replace([-np.inf], col_min)
            else:
                # If all values are infinite, replace with 0
                df[col] = 0

    # Step 3: Handle any remaining NaN values
    # First check how many rows have NaN
    nan_rows_before = df[feature_cols].isna().any(axis=1).sum()
    if nan_rows_before > 0:
        print(f"Found {nan_rows_before} rows with NaN values")

        # Drop rows where ALL feature values are NaN
        df = df[~df[feature_cols].isna().all(axis=1)]

        # For remaining NaN, fill with column median
        for col in feature_cols:
            if df[col].isna().any():
                median_val = df[col].median()
                if pd.isna(median_val):
                    median_val = 0
                df[col] = df[col].fillna(median_val)

    # Step 4: Drop rows with missing labels (if labels exist)
    if label_col and df[label_col].isna().any():
        before = len(df)
        df = df[df[label_col].notna()]
        print(f"Dropped {before - len(df)} rows with missing labels")

    # Final statistics
    final_rows = len(df)
    final_cols = len(df.columns)
    dropped_rows = initial_rows - final_rows
    dropped_cols = initial_cols - final_cols

    print(f"\nCleaning summary:")
    print(f"  Rows: {initial_rows} -> {final_rows} (dropped {dropped_rows}, {dropped_rows/initial_rows*100:.2f}%)")
    print(f"  Columns: {initial_cols} -> {final_cols} (dropped {dropped_cols})")
    print(f"  Final dataset shape: {df.shape}")

    return df


def split_features_labels(df, label_col='Label', label_mapping=None):
    """Split features and labels, apply label mapping if provided"""
    print("\n=== Splitting features and labels ===")

    if label_col not in df.columns:
        # Try common variations
        for col in df.columns:
            if col.lower() == 'label':
                label_col = col
                break

    # Strip whitespace from labels
    if df[label_col].dtype == 'object':
        df[label_col] = df[label_col].str.strip()

    print(f"Label column: {label_col}")
    print(f"Class distribution before mapping:")
    try:
        print(df[label_col].value_counts())
    except UnicodeEncodeError:
        print("[Encoding error - skipping detailed label distribution]")
        print(f"Number of unique labels: {df[label_col].nunique()}")

    X = df.drop(columns=[label_col])
    y = df[label_col]

    # Apply label mapping if provided
    if label_mapping:
        print(f"\nApplying label mapping...")
        # Check for unmapped labels
        unique_labels = set(y.unique())
        mapped_labels = set(label_mapping.keys())
        unmapped = unique_labels - mapped_labels
        if unmapped:
            print(f"WARNING: {len(unmapped)} unmapped labels found")
            print(f"Dropping {len(y[y.isin(unmapped)])} rows with unmapped labels...")
            # Print unmapped labels one by one with error handling
            for label in unmapped:
                try:
                    print(f"  - {label}")
                except UnicodeEncodeError:
                    print(f"  - [Unicode label - cannot display]")
            # Remove rows with unmapped labels
            mask = y.isin(mapped_labels)
            X = X[mask]
            y = y[mask]

        y = y.map(label_mapping)
        print(f"Class distribution after mapping:")
        try:
            print(y.value_counts())
        except UnicodeEncodeError:
            print("[Encoding error - skipping detailed distribution]")
            print(f"Number of unique values: {y.nunique()}")

    return X, y


def preprocess_cicids2017(config_path):
    """Main preprocessing pipeline"""

    # Load configuration
    config = load_config(config_path)
    print(f"Loaded config from: {config_path}")

    # Extract config parameters
    raw_path = config['dataset']['raw_path']
    processed_dir = Path(config['dataset']['processed_dir'])
    test_size = config['preprocessing']['test_size']
    random_state = config['preprocessing']['random_state']
    label_mapping = config['preprocessing'].get('label_mapping')

    # Create output directory
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load raw data
    df = load_raw_data(raw_path)

    # Step 2: Clean data
    df = clean_data(df)

    # Step 3: Split features and labels
    X, y = split_features_labels(df, label_mapping=label_mapping)

    # Step 4: Train/test split
    print(f"\n=== Splitting train/test (test_size={test_size}) ===")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    print(f"Train set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")

    # Step 5: Feature scaling
    scaler_type = config['preprocessing'].get('scaler', 'RobustScaler')
    print(f"\n=== Scaling features with {scaler_type} ===")

    scaler = RobustScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )

    # Step 6: Undersampling (on training data only)
    undersample_config = config['preprocessing'].get('undersampling', {})
    if undersample_config.get('enabled', False):
        print("\n=== Applying undersampling ===")
        strategy = undersample_config['strategy']
        print(f"Undersampling strategy: {strategy}")

        undersampler = RandomUnderSampler(
            sampling_strategy=strategy,
            random_state=undersample_config.get('random_state', 42)
        )
        X_train_scaled, y_train = undersampler.fit_resample(X_train_scaled, y_train)
        print(f"Training set after undersampling: {X_train_scaled.shape}")
        print(f"Class distribution after undersampling:")
        try:
            print(y_train.value_counts())
        except UnicodeEncodeError:
            print(f"[Encoding error] Classes: {y_train.nunique()}")

    # Step 7: Oversampling with SMOTE (on training data only)
    oversample_config = config['preprocessing'].get('oversampling', {})
    if oversample_config.get('enabled', False):
        print("\n=== Applying SMOTE oversampling ===")
        strategy = oversample_config['strategy']
        print(f"Oversampling strategy: {strategy}")

        smote = SMOTE(
            sampling_strategy=strategy,
            random_state=oversample_config.get('random_state', 42)
        )
        X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)
        print(f"Training set after SMOTE: {X_train_scaled.shape}")
        print(f"Class distribution after SMOTE:")
        try:
            print(y_train.value_counts())
        except UnicodeEncodeError:
            print(f"[Encoding error] Classes: {y_train.nunique()}")

    # Step 8: Save processed data
    print(f"\n=== Saving processed data to {processed_dir} ===")

    # Combine features and labels for saving
    train_df = X_train_scaled.copy()
    train_df['label'] = y_train.values

    test_df = X_test_scaled.copy()
    test_df['label'] = y_test.values

    # Save as parquet (more efficient for large datasets)
    train_path = processed_dir / 'train.parquet'
    test_path = processed_dir / 'test.parquet'

    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)

    print(f"Saved training data: {train_path} ({train_df.shape})")
    print(f"Saved test data: {test_path} ({test_df.shape})")

    print("\n[SUCCESS] Preprocessing completed successfully!")

    return {
        'train_path': str(train_path),
        'test_path': str(test_path),
        'train_shape': train_df.shape,
        'test_shape': test_df.shape
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Preprocess CICIDS2017 dataset with config-driven pipeline'
    )
    parser.add_argument(
        '--config',
        required=True,
        help='Path to configuration YAML file'
    )

    args = parser.parse_args()
    preprocess_cicids2017(args.config)
