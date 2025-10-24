#!/usr/bin/env python3
"""
preprocess.py
Unified comprehensive preprocessing pipeline for IDS datasets
Combines best features from preprocess_cicids2017.py and preprocess_ids_dataset.py

Supports:
- YAML configuration for DVC integration
- Smart infinite value handling
- Feature selection (variance, correlation, Kruskal-Wallis)
- Outlier removal (IQR)
- Undersampling/Oversampling (SMOTE)
- Train/test splitting
- Multiple scaling methods
"""

import os
import json
import yaml
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from scipy.stats import kruskal

# Use non-GUI backend for headless systems
plt.switch_backend('Agg')


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_raw_data(raw_path):
    """Load all CSV files from the raw data directory"""
    print(f"\n=== Loading raw data from {raw_path} ===")

    raw_path = Path(raw_path)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data path not found: {raw_path}")

    # Find all CSV files
    csv_files = list(raw_path.glob('*.csv'))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {raw_path}")

    print(f"Found {len(csv_files)} data files:")
    for f in csv_files:
        print(f"  - {f.name}")

    # Load and combine all files
    dfs = []
    for csv_file in csv_files:
        print(f"  Loaded {csv_file.name}: ", end='')
        try:
            df = pd.read_csv(csv_file, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(csv_file, encoding='latin-1')

        # Strip whitespace from column names
        df.columns = df.columns.str.strip()
        print(f"{df.shape}")
        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"\nCombined dataset shape: {combined_df.shape}")

    return combined_df


def clean_data(df, config):
    """Improved data cleaning with smart handling of infinite values"""
    print("\n=== Cleaning data ===")

    initial_rows = len(df)
    initial_cols = len(df.columns)

    # Get cleaning config
    cleaning_config = config.get('preprocessing', {}).get('cleaning', {})
    drop_inf_threshold = cleaning_config.get('drop_inf_columns_threshold', 0.5)
    drop_nan_threshold = cleaning_config.get('drop_nan_columns_threshold', 0.5)

    # Separate label column if it exists
    label_col = None
    for col in ['Label', 'label', ' Label']:
        if col in df.columns:
            label_col = col
            break

    feature_cols = [col for col in df.columns if col != label_col]

    # Step 1: Check for columns with too many infinite values
    print(f"Checking for columns with >{drop_inf_threshold*100}% infinite values...")
    inf_cols = []
    for col in feature_cols:
        if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
            inf_count = np.isinf(df[col]).sum()
            if inf_count > 0:
                inf_pct = inf_count / len(df)
                if inf_pct > drop_inf_threshold:
                    inf_cols.append(col)
                    print(f"  Dropping column '{col}': {inf_pct*100:.1f}% infinite values")

    if inf_cols:
        df = df.drop(columns=inf_cols)
        feature_cols = [col for col in df.columns if col != label_col]
        print(f"Dropped {len(inf_cols)} columns with excessive infinite values")

    # Step 2: Replace remaining infinite values with column-specific values
    print("Replacing remaining infinite values...")
    inf_replaced = 0
    for col in feature_cols:
        if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
            # Get finite values for this column
            finite_mask = np.isfinite(df[col])
            if finite_mask.any():
                col_max = df.loc[finite_mask, col].max()
                col_min = df.loc[finite_mask, col].min()

                # Count and replace
                inf_count = np.isinf(df[col]).sum()
                if inf_count > 0:
                    inf_replaced += inf_count
                    df[col] = df[col].replace([np.inf], col_max)
                    df[col] = df[col].replace([-np.inf], col_min)
            else:
                # If all values are infinite, replace with 0
                df[col] = 0

    if inf_replaced > 0:
        print(f"  Replaced {inf_replaced} infinite values")

    # Step 3: Handle NaN - drop columns with too many missing values
    print(f"Checking for columns with >{drop_nan_threshold*100}% missing values...")
    missing_ratio = df[feature_cols].isna().mean()
    nan_cols = missing_ratio[missing_ratio > drop_nan_threshold].index.tolist()

    if nan_cols:
        print(f"  Dropping {len(nan_cols)} columns with excessive missing values:")
        for col in nan_cols:
            print(f"    - {col}: {missing_ratio[col]*100:.1f}% missing")
        df = df.drop(columns=nan_cols)
        feature_cols = [col for col in df.columns if col != label_col]

    # Step 4: Impute remaining NaN values with median
    for col in feature_cols:
        if df[col].isna().any():
            if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                median_val = df[col].median()
                if pd.isna(median_val):
                    median_val = 0
                df[col] = df[col].fillna(median_val)

    # Step 5: Drop rows with missing labels
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


def remove_low_variance_features(df, label_col, threshold=0.0):
    """Remove features with low variance"""
    print(f"\n=== Removing low variance features (threshold={threshold}) ===")

    feature_cols = [col for col in df.columns if col != label_col]
    num_df = df[feature_cols].select_dtypes(include=[np.number])

    selector = VarianceThreshold(threshold=threshold)
    selector.fit(num_df)

    kept_features = num_df.columns[selector.get_support()].tolist()
    removed_features = list(set(num_df.columns) - set(kept_features))

    if removed_features:
        print(f"Removed {len(removed_features)} low-variance features:")
        for feat in removed_features[:10]:  # Show first 10
            print(f"  - {feat} (variance={num_df[feat].var():.6f})")
        if len(removed_features) > 10:
            print(f"  ... and {len(removed_features) - 10} more")

        # Keep numeric kept features + label
        df = df[kept_features + [label_col]]

    return df


def remove_correlated_features(df, label_col, threshold=0.95, output_dir=None):
    """Remove highly correlated features"""
    print(f"\n=== Removing highly correlated features (threshold={threshold}) ===")

    feature_cols = [col for col in df.columns if col != label_col]
    num_df = df[feature_cols].select_dtypes(include=[np.number])

    # Calculate correlation matrix
    corr = num_df.corr()

    # Save correlation heatmap if output_dir specified
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(12, 10))
        sns.heatmap(corr, cmap='coolwarm', center=0, square=True)
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        corr_path = output_dir / 'correlation_heatmap.png'
        plt.savefig(corr_path, dpi=150)
        plt.close()
        print(f"  Correlation heatmap saved to: {corr_path}")

    # Find redundant features
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    redundant = [column for column in upper.columns if any(upper[column].abs() > threshold)]

    if redundant:
        print(f"Removed {len(redundant)} highly correlated features:")
        for feat in redundant[:10]:
            print(f"  - {feat}")
        if len(redundant) > 10:
            print(f"  ... and {len(redundant) - 10} more")

        df = df.drop(columns=redundant)

    return df


def remove_outliers_iqr(df, label_col):
    """Remove outliers using IQR method"""
    print("\n=== Removing outliers (IQR method) ===")

    feature_cols = [col for col in df.columns if col != label_col]
    num_df = df[feature_cols].select_dtypes(include=[np.number])

    Q1 = num_df.quantile(0.25)
    Q3 = num_df.quantile(0.75)
    IQR = Q3 - Q1

    # Find outliers
    outlier_mask = ~((num_df < (Q1 - 1.5 * IQR)) | (num_df > (Q3 + 1.5 * IQR))).any(axis=1)

    before = len(df)
    df = df.loc[outlier_mask]
    removed = before - len(df)

    print(f"Removed {removed} outliers ({removed/before*100:.2f}%)")
    print(f"Dataset shape after outlier removal: {df.shape}")

    return df


def rank_features_kruskal(df, label_col, top_n=20, output_dir=None):
    """Rank features using Kruskal-Wallis test"""
    print(f"\n=== Ranking features using Kruskal-Wallis test ===")

    feature_cols = [col for col in df.columns if col != label_col]
    num_cols = df[feature_cols].select_dtypes(include=[np.number]).columns

    importance = []
    for col in num_cols:
        try:
            groups = [group[col].values for name, group in df.groupby(label_col)]
            if len(groups) > 1:
                stat, p = kruskal(*groups)
                importance.append((col, stat, p))
        except Exception:
            continue

    imp_df = pd.DataFrame(
        importance, columns=['Feature', 'Statistic', 'p-value']
    ).sort_values('Statistic', ascending=False)

    print(f"Top {top_n} features by Kruskal-Wallis statistic:")
    for idx, row in imp_df.head(top_n).iterrows():
        print(f"  {row['Feature']}: {row['Statistic']:.2f} (p={row['p-value']:.4e})")

    # Save plot if output_dir specified
    if output_dir and len(imp_df) > 0:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(12, 6))
        sns.barplot(x='Statistic', y='Feature', data=imp_df.head(top_n), palette='viridis')
        plt.title(f'Top {top_n} Features by Kruskal-Wallis Statistic')
        plt.tight_layout()
        kruskal_path = output_dir / 'kruskal_importance.png'
        plt.savefig(kruskal_path, dpi=150)
        plt.close()
        print(f"  Kruskal-Wallis chart saved to: {kruskal_path}")

    return imp_df


def split_features_labels(df, label_col='Label', label_mapping=None):
    """Split features and labels, apply label mapping if provided"""
    print("\n=== Splitting features and labels ===")

    # Find label column
    if label_col not in df.columns:
        for col in df.columns:
            if col.lower() == 'label' or col.strip().lower() == 'label':
                label_col = col
                break

    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in dataframe")

    print(f"Label column: {label_col}")

    # Get class distribution before mapping
    print("Class distribution before mapping:")
    value_counts = df[label_col].value_counts()
    try:
        for label, count in value_counts.items():
            print(f"  {label}: {count}")
    except UnicodeEncodeError:
        print("[Unicode encoding error - skipping detailed label distribution]")
    print(f"Number of unique labels: {len(value_counts)}")

    # Apply label mapping if provided
    if label_mapping:
        print("\nApplying label mapping...")
        original_labels = set(df[label_col].unique())
        mapped_labels = set(label_mapping.keys())

        unmapped = original_labels - mapped_labels
        if unmapped:
            print(f"WARNING: {len(unmapped)} unmapped labels found")
            print(f"Dropping {len(df[~df[label_col].isin(mapped_labels)])} rows with unmapped labels...")
            try:
                for label in list(unmapped)[:5]:
                    print(f"  - {label}")
            except UnicodeEncodeError:
                print("[Unicode encoding error - cannot display unmapped labels]")

        # Filter to only mapped labels
        df = df[df[label_col].isin(mapped_labels)]

        # Apply mapping
        df[label_col] = df[label_col].map(label_mapping)

        print("Class distribution after mapping:")
        try:
            print(df[label_col].value_counts())
        except UnicodeEncodeError:
            print("[Unicode encoding error - skipping detailed distribution]")
            print(f"Number of unique mapped labels: {df[label_col].nunique()}")

    # Rename to standard 'label' column
    df = df.rename(columns={label_col: 'label'})

    return df


def preprocess_dataset(config_path):
    """Main preprocessing pipeline"""

    # Load configuration
    config = load_config(config_path)
    print(f"Loaded config from: {config_path}")

    # Extract config
    dataset_name = config['dataset']['name']
    raw_path = config['dataset']['raw_path']
    processed_dir = Path(config['dataset']['processed_dir'])
    preprocessing_config = config['preprocessing']

    print(f"\n=== Preprocessing {dataset_name} ===")

    # Create output directory
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load raw data
    df = load_raw_data(raw_path)

    # Step 2: Clean data
    df = clean_data(df, config)

    # Step 3: Feature engineering (if enabled)
    feature_selection = preprocessing_config.get('feature_selection', {})

    if feature_selection.get('remove_low_variance', False):
        variance_threshold = feature_selection.get('variance_threshold', 0.0)
        # Temporarily determine label column
        label_col = None
        for col in ['Label', 'label', ' Label']:
            if col in df.columns:
                label_col = col
                break
        if label_col:
            df = remove_low_variance_features(df, label_col, variance_threshold)

    if feature_selection.get('remove_correlated', False):
        correlation_threshold = feature_selection.get('correlation_threshold', 0.95)
        label_col = None
        for col in ['Label', 'label', ' Label']:
            if col in df.columns:
                label_col = col
                break
        if label_col:
            df = remove_correlated_features(df, label_col, correlation_threshold, processed_dir)

    if feature_selection.get('remove_outliers', False):
        label_col = None
        for col in ['Label', 'label', ' Label']:
            if col in df.columns:
                label_col = col
                break
        if label_col:
            df = remove_outliers_iqr(df, label_col)

    if feature_selection.get('rank_features', False):
        label_col = None
        for col in ['Label', 'label', ' Label']:
            if col in df.columns:
                label_col = col
                break
        if label_col:
            rank_features_kruskal(df, label_col, top_n=20, output_dir=processed_dir)

    # Step 4: Split features and labels
    label_mapping = preprocessing_config.get('label_mapping')
    df = split_features_labels(df, label_mapping=label_mapping)

    # Step 5: Train/test split
    print(f"\n=== Splitting train/test (test_size={preprocessing_config['test_size']}) ===")
    X = df.drop(columns=['label'])
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=preprocessing_config['test_size'],
        random_state=preprocessing_config['random_state'],
        stratify=y
    )

    print(f"Train set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")

    # Step 6: Feature scaling
    scaler_type = preprocessing_config.get('scaler', 'RobustScaler')
    print(f"\n=== Scaling features with {scaler_type} ===")

    if scaler_type == 'RobustScaler':
        scaler = RobustScaler()
    elif scaler_type == 'MinMaxScaler':
        scaler = MinMaxScaler()
    elif scaler_type == 'StandardScaler':
        scaler = StandardScaler()
    else:
        raise ValueError(f"Unknown scaler: {scaler_type}")

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

    # Step 7: Undersampling (if enabled)
    undersampling = preprocessing_config.get('undersampling', {})
    if undersampling.get('enabled', False):
        print(f"\n=== Applying undersampling ===")
        strategy = {int(k): v for k, v in undersampling['strategy'].items()}
        print(f"Undersampling strategy: {strategy}")

        undersampler = RandomUnderSampler(
            sampling_strategy=strategy,
            random_state=undersampling.get('random_state', 42)
        )
        X_train_scaled, y_train = undersampler.fit_resample(X_train_scaled, y_train)

        print(f"Training set after undersampling: {X_train_scaled.shape}")
        print(f"Class distribution after undersampling:")
        print(y_train.value_counts())

    # Step 8: Oversampling (if enabled)
    oversampling = preprocessing_config.get('oversampling', {})
    if oversampling.get('enabled', False):
        print(f"\n=== Applying SMOTE oversampling ===")
        strategy = {int(k): v for k, v in oversampling['strategy'].items()}
        print(f"Oversampling strategy: {strategy}")

        smote = SMOTE(
            sampling_strategy=strategy,
            random_state=oversampling.get('random_state', 42)
        )
        X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)

        print(f"Training set after SMOTE: {X_train_scaled.shape}")
        print(f"Class distribution after SMOTE:")
        print(y_train.value_counts())

    # Step 9: Save processed data
    print(f"\n=== Saving processed data to {processed_dir} ===")

    # Combine features and labels
    train_df = X_train_scaled.copy()
    train_df['label'] = y_train

    test_df = X_test_scaled.copy()
    test_df['label'] = y_test

    # Save as parquet
    train_path = processed_dir / 'train.parquet'
    test_path = processed_dir / 'test.parquet'

    train_df.to_parquet(train_path)
    test_df.to_parquet(test_path)

    print(f"Saved training data: {train_path} ({train_df.shape})")
    print(f"Saved test data: {test_path} ({test_df.shape})")

    print("\n[SUCCESS] Preprocessing completed successfully!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Unified preprocessing pipeline for IDS datasets'
    )
    parser.add_argument(
        '--config',
        required=True,
        help='Path to configuration YAML file'
    )

    args = parser.parse_args()
    preprocess_dataset(args.config)
