#!/usr/bin/env python3
"""
preprocess_ids_dataset.py
Generalized preprocessing pipeline for flow-based IDS datasets
(e.g., CIC-IDS 2017/2018, CIC-DDoS 2019, UNSW-NB15)

Automatically reads all .csv or .parquet files in a folder.
Strips whitespace from feature names before processing.
"""

import os
import argparse
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import kruskal
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
from datetime import datetime

plt.switch_backend("Agg")  # disable GUI backend for headless systems


# ---------------- Utility Functions ---------------- #


def log_message(logs, message):
    """Print and store log message"""
    print(message)
    logs.append(message)


def save_logs(logs, output_dir):
    """Save logs to a text file"""
    log_path = os.path.join(output_dir, "preprocessing_log.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(logs))
    print(f"\n[INFO] Logs saved to {log_path}")


# ---------------- Main Preprocessing Function ---------------- #


def preprocess_dataset(input_dir, target_col, output_dir):
    logs = []
    os.makedirs(output_dir, exist_ok=True)

    # STEP 1 — Load Dataset(s)
    log_message(logs, "=== STEP 1: Loading Dataset Files ===")

    file_paths = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.lower().endswith((".csv", ".parquet"))
    ]

    if not file_paths:
        raise FileNotFoundError(f"No CSV or Parquet files found in {input_dir}")

    dfs = []
    for path in file_paths:
        ext = os.path.splitext(path)[1].lower()
        log_message(logs, f"Loading file: {path}")
        if ext == ".csv":
            df = pd.read_csv(path)
        else:
            df = pd.read_parquet(path)
        # Strip whitespace from column names
        df.columns = df.columns.str.strip()
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    log_message(logs, f"Combined dataset shape: {df.shape}")
    log_message(logs, f"Column names normalized (whitespace stripped).")

    if target_col not in df.columns:
        raise KeyError(
            f"Target column '{target_col}' not found after name normalization. Available columns:\n{df.columns.tolist()}"
        )

    # STEP 2 — Basic Cleaning
    log_message(logs, "\n=== STEP 2: Basic Cleaning ===")
    df = df.replace([np.inf, -np.inf], np.nan)

    missing_ratio = df.isna().mean()
    removed_features = missing_ratio[missing_ratio > 0.5].index.tolist()
    df.drop(columns=removed_features, inplace=True, errors="ignore")
    log_message(
        logs, f"Removed {len(removed_features)} features with >50% missing values:"
    )
    for feat in removed_features:
        log_message(logs, f"  - {feat}: missing_ratio={missing_ratio[feat]:.2f}")

    # Imputation
    imputed_features = df.columns[df.isna().any()].tolist()
    for col in imputed_features:
        if df[col].dtype.kind in "biufc":
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            log_message(
                logs, f"Imputed numeric feature '{col}' with median={median_val:.4f}"
            )
        else:
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)
            log_message(
                logs, f"Imputed categorical feature '{col}' with mode='{mode_val}'"
            )

    # STEP 3 — Low Variance Filter
    log_message(logs, "\n=== STEP 3: Low Variance Filtering ===")
    num_df = df.select_dtypes(include=[np.number])
    selector = VarianceThreshold(threshold=0.0)
    selector.fit(num_df)
    kept = num_df.columns[selector.get_support()]
    removed = list(set(num_df.columns) - set(kept))
    df = df[kept.tolist() + [target_col]]
    for feat in removed:
        log_message(
            logs,
            f"Removed low-variance feature '{feat}' (variance={num_df[feat].var():.6f})",
        )

    # STEP 4 — Correlation Analysis
    log_message(logs, "\n=== STEP 4: Correlation Analysis ===")
    corr = df.corr(numeric_only=True)
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Feature Correlation Heatmap")
    corr_path = os.path.join(output_dir, "correlation_heatmap.png")
    plt.savefig(corr_path)
    plt.close()
    log_message(logs, f"Correlation heatmap saved to {corr_path}")

    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    redundant = [column for column in upper.columns if any(upper[column].abs() > 0.95)]
    df.drop(columns=redundant, inplace=True, errors="ignore")
    for feat in redundant:
        log_message(logs, f"Removed highly correlated feature '{feat}' (|corr| > 0.95)")

    # STEP 5 — Outlier Removal (IQR)
    log_message(logs, "\n=== STEP 5: Outlier Removal (IQR) ===")
    num_df = df.select_dtypes(include=[np.number])
    Q1 = num_df.quantile(0.25)
    Q3 = num_df.quantile(0.75)
    IQR = Q3 - Q1
    outlier_mask = ~((num_df < (Q1 - 1.5 * IQR)) | (num_df > (Q3 + 1.5 * IQR))).any(
        axis=1
    )
    before = len(df)
    df = df.loc[outlier_mask]
    log_message(logs, f"Removed {before - len(df)} outliers using IQR threshold.")

    # STEP 6 — Kruskal-Wallis Feature Ranking
    log_message(logs, "\n=== STEP 6: Kruskal-Wallis Feature Ranking ===")
    importance = []
    for col in df.select_dtypes(include=[np.number]).columns:
        if col == target_col:
            continue
        try:
            groups = [group[col].values for name, group in df.groupby(target_col)]
            stat, p = kruskal(*groups)
            importance.append((col, stat, p))
        except Exception:
            continue
    imp_df = pd.DataFrame(
        importance, columns=["Feature", "Statistic", "p-value"]
    ).sort_values("Statistic", ascending=False)

    plt.figure(figsize=(12, 6))
    sns.barplot(x="Statistic", y="Feature", data=imp_df.head(20), palette="viridis")
    plt.title("Top 20 Features by Kruskal-Wallis Statistic")
    plt.tight_layout()
    kruskal_path = os.path.join(output_dir, "kruskal_importance.png")
    plt.savefig(kruskal_path)
    plt.close()
    log_message(logs, f"Kruskal-Wallis chart saved to {kruskal_path}")

    # STEP 7 — Feature Scaling
    log_message(logs, "\n=== STEP 7: Feature Scaling ===")
    log_message(logs, f"Feature list before scaling: {df.columns.tolist()}")
    features = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in features:
        features = features.drop([target_col])
    if len(features) == 0:
        log_message(
            logs, "No numeric features available for scaling. Skipping this step."
        )
    else:
        scaler = MinMaxScaler()
        df[features] = scaler.fit_transform(df[features])
        log_message(
            logs,
            f"Applied MinMax scaling on {len(features)} features. Dataset imbalance retained.",
        )

    # STEP 8 — Save Outputs
    log_message(logs, "\n=== STEP 8: Saving Outputs ===")
    out_csv = os.path.join(output_dir, "processed_dataset.csv")
    out_parquet = os.path.join(output_dir, "processed_dataset.parquet")
    df.to_csv(out_csv, index=False)
    df.to_parquet(out_parquet, index=False)
    log_message(logs, f"Processed dataset saved to:\n  - {out_csv}\n  - {out_parquet}")
    log_message(logs, f"Final dataset shape: {df.shape}")

    # Save logs
    save_logs(logs, output_dir)


# ---------------- CLI Entry ---------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generalized Preprocessing for Flow-based IDS Datasets"
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Directory containing dataset files (CSV or Parquet)",
    )
    parser.add_argument(
        "--target_col",
        required=True,
        help="Name of the target label column (whitespace-insensitive)",
    )
    parser.add_argument(
        "--output_dir",
        default="./processed",
        help="Directory to save processed data and plots",
    )
    args = parser.parse_args()

    preprocess_dataset(args.input_dir, args.target_col.strip(), args.output_dir)
