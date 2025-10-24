#!/usr/bin/env python3
"""
test_pipeline.py
Quick test script to verify pipeline setup without running full experiments

This script checks:
- All config files exist and are valid
- All param files exist and are valid
- All scripts are accessible
- Dependencies are installed
- DVC configuration is correct

Run this BEFORE attempting full `dvc repro` to catch configuration issues early.
"""

import sys
import yaml
import json
from pathlib import Path
import importlib.util


def check_file_exists(file_path, description):
    """Check if a file exists"""
    path = Path(file_path)
    if path.exists():
        print(f"[OK] {description}: {file_path}")
        return True
    else:
        print(f"[FAIL] {description} NOT FOUND: {file_path}")
        return False


def check_yaml_valid(file_path):
    """Check if YAML file is valid"""
    try:
        with open(file_path, 'r') as f:
            yaml.safe_load(f)
        return True
    except Exception as e:
        print(f"  ERROR: Invalid YAML - {e}")
        return False


def check_json_valid(file_path):
    """Check if JSON file is valid"""
    try:
        with open(file_path, 'r') as f:
            json.load(f)
        return True
    except Exception as e:
        print(f"  ERROR: Invalid JSON - {e}")
        return False


def check_python_dependencies():
    """Check if required Python packages are installed"""
    print("\n" + "="*80)
    print("CHECKING PYTHON DEPENDENCIES")
    print("="*80)

    required_packages = [
        'pandas',
        'numpy',
        'scikit-learn',
        'xgboost',
        'lightgbm',
        'optuna',
        'matplotlib',
        'seaborn',
        'imblearn',
        'yaml',
        'dvc'
    ]

    all_installed = True
    for package in required_packages:
        try:
            if package == 'sklearn':
                __import__('sklearn')
            elif package == 'yaml':
                __import__('yaml')
            else:
                __import__(package)
            print(f"[OK] {package} installed")
        except ImportError:
            print(f"[FAIL] {package} NOT installed")
            all_installed = False

    return all_installed


def check_configs():
    """Check all configuration files"""
    print("\n" + "="*80)
    print("CHECKING CONFIGURATION FILES")
    print("="*80)

    configs = [
        'configs/cicids2017_xgb.yaml',
        'configs/cicids2017_lgbm.yaml',
        'configs/cicids2018_xgb.yaml',
        'configs/cicids2018_lgbm.yaml',
    ]

    all_valid = True
    for config in configs:
        if check_file_exists(config, "Config"):
            if not check_yaml_valid(config):
                all_valid = False
        else:
            all_valid = False

    return all_valid


def check_params():
    """Check all hyperparameter files"""
    print("\n" + "="*80)
    print("CHECKING HYPERPARAMETER FILES")
    print("="*80)

    params = [
        'params/cicids2017_xgb_params.json',
        'params/cicids2017_lgbm_params.json',
        'params/cicids2018_xgb_params.json',
        'params/cicids2018_lgbm_params.json',
    ]

    all_valid = True
    for param in params:
        if check_file_exists(param, "Params"):
            if not check_json_valid(param):
                all_valid = False
        else:
            all_valid = False

    return all_valid


def check_scripts():
    """Check all pipeline scripts"""
    print("\n" + "="*80)
    print("CHECKING PIPELINE SCRIPTS")
    print("="*80)

    scripts = [
        'scripts/preprocess_cicids2017.py',
        'scripts/train_model.py',
        'scripts/evaluate_model.py',
        'scripts/tune_hyperparameters.py',
    ]

    all_exist = True
    for script in scripts:
        if not check_file_exists(script, "Script"):
            all_exist = False

    return all_exist


def check_dvc_setup():
    """Check DVC setup"""
    print("\n" + "="*80)
    print("CHECKING DVC SETUP")
    print("="*80)

    all_valid = True

    # Check dvc.yaml
    if check_file_exists('dvc.yaml', "DVC pipeline"):
        if not check_yaml_valid('dvc.yaml'):
            all_valid = False
    else:
        all_valid = False

    # Check .dvc directory
    if Path('.dvc').exists():
        print("[OK] .dvc directory exists")
    else:
        print("[FAIL] .dvc directory NOT FOUND (run 'dvc init')")
        all_valid = False

    # Check .dvc/config
    if check_file_exists('.dvc/config', "DVC config"):
        # Check for remote
        with open('.dvc/config', 'r') as f:
            config_content = f.read()
            if 'remote' in config_content:
                print("[OK] DVC remote configured")
            else:
                print("[WARNING] WARNING: No DVC remote configured")
    else:
        all_valid = False

    return all_valid


def check_directory_structure():
    """Check expected directory structure"""
    print("\n" + "="*80)
    print("CHECKING DIRECTORY STRUCTURE")
    print("="*80)

    required_dirs = [
        'configs',
        'params',
        'scripts',
        'data',
        'models',
        'results',
    ]

    all_exist = True
    for dir_name in required_dirs:
        path = Path(dir_name)
        if path.exists():
            print(f"[OK] {dir_name}/ exists")
        else:
            print(f"[FAIL] {dir_name}/ NOT FOUND (will be created during pipeline run)")
            # Don't mark as failure - these can be created automatically

    return True


def check_data_availability():
    """Check if raw data is available"""
    print("\n" + "="*80)
    print("CHECKING DATA AVAILABILITY")
    print("="*80)

    datasets = [
        ('data/raw/cic_ids_2017', 'CICIDS2017'),
        ('data/raw/cic_ids_2018', 'CICIDS2018'),
    ]

    data_available = False
    for path, name in datasets:
        if Path(path).exists():
            print(f"[OK] {name} raw data exists at {path}")
            data_available = True
        else:
            print(f"[FAIL] {name} raw data NOT FOUND at {path}")
            print(f"  Run 'dvc pull' to download data")

    if not data_available:
        print("\n[WARNING] WARNING: No raw data found. Run 'dvc pull' before 'dvc repro'")

    return data_available


def main():
    """Run all checks"""
    print("="*80)
    print("PIPELINE SETUP TEST")
    print("="*80)
    print("\nThis script verifies that your pipeline is properly configured.")
    print("It does NOT run the actual pipeline (use 'dvc repro' for that).\n")

    results = {
        'Dependencies': check_python_dependencies(),
        'Configs': check_configs(),
        'Params': check_params(),
        'Scripts': check_scripts(),
        'DVC Setup': check_dvc_setup(),
        'Directory Structure': check_directory_structure(),
        'Data Available': check_data_availability(),
    }

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    all_passed = True
    for check, passed in results.items():
        status = "[OK] PASS" if passed else "[FAIL] FAIL"
        print(f"{status}: {check}")
        if not passed:
            all_passed = False

    print("\n" + "="*80)
    if all_passed:
        print("[SUCCESS] ALL CHECKS PASSED!")
        print("\nYou can now run the pipeline:")
        print("  dvc repro evaluate_cicids2017_xgb")
        print("\nOr run individual stages:")
        print("  python scripts/preprocess_cicids2017.py --config configs/cicids2017_xgb.yaml")
    else:
        print("[ERROR] SOME CHECKS FAILED")
        print("\nPlease fix the issues above before running the pipeline.")
        print("See README.md for setup instructions.")
        sys.exit(1)
    print("="*80)


if __name__ == '__main__':
    main()