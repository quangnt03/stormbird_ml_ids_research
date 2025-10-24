# Intrusion Detection System (IDS) using Machine Learning on CIC-IDS Datasets

This project implements a machine learning-based Intrusion Detection System (IDS) using the CIC-IDS2017 and CIC-IDS2018 datasets. It focuses on binary/multiclass classification of network intrusions using gradient boosting models like LightGBM and XGBoost. The pipeline includes data preprocessing, model training, evaluation, and visualization of results (e.g., confusion matrices, training curves).

Key features:
- **DVC-powered pipeline** for versioning datasets, models, metrics, and plots
- **Unified preprocessing** with smart infinite value handling, feature selection, and SMOTE
- **Training history tracking** with time-series plots of training/validation loss
- **Comprehensive metrics** including F1-scores, resource usage (CPU, GPU, memory), and timing
- **Interactive visualizations** using DVC plots with custom Vega-Lite templates
- **Jupyter notebooks** for exploratory research and hyperparameter tuning
- **Reproducible pipeline** using config-driven scripts for production use

## Project Structure
```
project-root/
├── .dvc/                   # DVC configuration and cache
│   ├── config              # DVC remote storage configuration
│   └── plots/              # Custom Vega-Lite plot templates
│       ├── training_curves.json       # Template for training/validation loss curves
│       ├── smooth_training.json       # Smooth line plots
│       └── metrics_comparison.json    # Bar charts for metrics comparison
├── configs/                # YAML configuration files for each experiment
│   ├── cicids2017_xgb.yaml           # XGBoost on CICIDS2017
│   ├── cicids2017_lgbm.yaml          # LightGBM on CICIDS2017
│   ├── cicids2018_xgb.yaml           # XGBoost on CICIDS2018
│   └── cicids2018_lgbm.yaml          # LightGBM on CICIDS2018
├── data/                   # Dataset files (tracked by DVC)
│   ├── raw/                # Raw CIC-IDS CSV files
│   │   ├── cic_ids_2017/   # CICIDS2017 dataset
│   │   └── cic_ids_2018/   # CICIDS2018 dataset
│   └── processed/          # Preprocessed data (train/test splits, scaled)
│       ├── cicids2017/     # Processed CICIDS2017 (parquet files)
│       └── cicids2018/     # Processed CICIDS2018 (parquet files)
├── models/                 # Trained models (tracked by DVC)
│   ├── cicids2017_xgb.pkl           # XGBoost model for CICIDS2017
│   ├── cicids2017_lgbm.pkl          # LightGBM model for CICIDS2017
│   ├── cicids2018_xgb.pkl           # XGBoost model for CICIDS2018
│   └── cicids2018_lgbm.pkl          # LightGBM model for CICIDS2018
├── notebooks/              # Jupyter notebooks for exploratory analysis
│   └── benchmarks/         # Benchmark experiments
│       ├── CICIDS2017/     # CICIDS2017 experiments and tuning
│       └── CICIDS2018/     # CICIDS2018 experiments and tuning
├── params/                 # Best hyperparameters (JSON files, tracked by Git)
│   ├── cicids2017_xgb_params.json   # Best params from Optuna tuning
│   ├── cicids2017_lgbm_params.json
│   ├── cicids2018_xgb_params.json
│   └── cicids2018_lgbm_params.json
├── results/                # Evaluation outputs (tracked by DVC)
│   ├── cicids2017_xgb_metrics.json           # Performance metrics
│   ├── cicids2017_xgb_training_history.json  # Training curves data
│   ├── cicids2017_xgb_confusion_matrix.png   # Confusion matrix plot
│   ├── cicids2017_xgb_classification_report.txt
│   └── ... (similar files for other experiments)
├── scripts/                # Python scripts for reproducible pipeline
│   ├── preprocess.py       # Unified preprocessing script
│   ├── train_model.py      # Model training with best hyperparameters
│   ├── evaluate_model.py   # Model evaluation and metrics generation
│   └── utils/              # Utility modules
│       ├── __init__.py
│       └── resource_monitor.py  # CPU/GPU/memory monitoring
├── dvc.yaml                # DVC pipeline definition
├── dvc.lock                # DVC pipeline lock file (auto-generated)
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

**Key Directories:**
- **configs/**: YAML files control preprocessing, model type, and output paths
- **params/**: JSON files with best hyperparameters from Optuna tuning (Git-tracked)
- **scripts/**: Production-ready scripts for reproducible pipeline execution
- **results/**: All evaluation outputs including metrics, plots, and training history
- **.dvc/plots/**: Custom Vega-Lite templates for beautiful visualizations

## Setup

1. **Clone the Repository**:
```
git clone <repo-url>
cd <project-name>
text2. **Create Virtual Environment**:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
text3. **Install Dependencies**:
pip install -r requirements.txt
```
Includes: `pandas, numpy, scikit-learn, lightgbm, xgboost, optuna, matplotlib, seaborn, jupyter, dvc`

4. **Install DVC** (for data/model versioning):
```
pip install dvc
dvc init
text5. **Pull Data and Models**:
dvc pull  # Downloads tracked data/models from remote (e.g., S3, Git LFS)
```
If no remote is set, run `dvc add data/raw/` (after downloading datasets manually) and `git add data/raw.dvc` + `dvc push`.

6. **Launch Jupyter** (optional, for notebooks): 
```bash
jupyter notebook
```

## Usage

This project supports **two workflows**:
1. **Exploratory workflow** (Jupyter notebooks in Colab) - for research and experimentation
2. **Reproducible workflow** (DVC pipelines) - for production and collaboration

---

## Workflow 1: Exploratory Research (Jupyter Notebooks)

**Use this workflow for:** Experimentation, hyperparameter tuning, and research

### 1. Data Preprocessing & Exploration
- Run notebooks in `notebooks/benchmarks/CICIDS2017/experiments/` or `notebooks/benchmarks/CICIDS2018/experiments/`
- These notebooks include full pipeline: data loading → preprocessing → hyperparameter tuning (Optuna) → training → evaluation

### 2. Hyperparameter Tuning
- Notebooks use **Optuna** for automated hyperparameter search
- Best parameters are found through multi-trial optimization
- Example: `[30-9]_CIC_IDS2017_Benchmark.ipynb` runs 30 trials with early stopping

### 3. Export Best Parameters
After finding optimal hyperparameters in your notebook:

```python
import json

# Export XGBoost best params
json.dump(xgb_study.best_params,
          open('params/cicids2017_xgb_params.json', 'w'),
          indent=2)

# Export LightGBM best params
json.dump(lgbm_study.best_params,
          open('params/cicids2017_lgbm_params.json', 'w'),
          indent=2)
```

Then commit to git:
```bash
git add params/cicids2017_*_params.json
git commit -m "Add best hyperparameters for CICIDS2017"
git push
```

---

## Workflow 2: Reproducible Pipeline (DVC)

**Use this workflow for:** Reproducing results without re-running expensive hyperparameter tuning

### Prerequisites
```bash
# Pull data from DVC remote
dvc pull -r gcs

# Authenticate with Google Cloud (if using GCS remote)
gcloud auth activate-service-account --key-file=.credentials/stormbird-vn-svac-2e541f116f50.json
```

### Running the Complete Pipeline

#### Option 1: Run Everything
```bash
# Reproduce all CICIDS2017 experiments (XGBoost + LightGBM)
dvc repro evaluate_cicids2017_xgb
dvc repro evaluate_cicids2017_lgbm
```

#### Option 2: Run Specific Stages
```bash
# Run only preprocessing
dvc repro preprocess_cicids2017_xgb

# Run training only (requires preprocessing first)
dvc repro train_cicids2017_xgb

# Run evaluation only (requires training first)
dvc repro evaluate_cicids2017_xgb
```

#### Option 3: Run Individual Scripts
```bash
# Preprocess data
python scripts/preprocess_cicids2017.py --config configs/cicids2017_xgb.yaml

# Train model (uses pre-tuned hyperparameters from params/)
python scripts/train_model.py --config configs/cicids2017_xgb.yaml

# Evaluate model
python scripts/evaluate_model.py --config configs/cicids2017_xgb.yaml
```

### Pipeline Stages Explained

**Stage 1: Preprocessing** (`preprocess_cicids2017_xgb`)
- Loads raw data from `data/raw/cic_ids_2017/`
- Applies data cleaning, train/test split
- Performs feature scaling (RobustScaler)
- Applies undersampling and SMOTE oversampling
- Outputs: `data/processed/cicids2017/train.parquet` and `test.parquet`

**Stage 2: Training** (`train_cicids2017_xgb`)
- Loads preprocessed training data
- Loads best hyperparameters from `params/cicids2017_xgb_params.json`
- Trains XGBoost/LightGBM model (NO hyperparameter tuning)
- Saves trained model to `models/cicids2017_xgb.pkl`

**Stage 3: Evaluation** (`evaluate_cicids2017_xgb`)
- Loads trained model and test data
- Generates predictions
- Computes metrics (accuracy, precision, recall, F1, ROC AUC)
- Saves metrics to `results/cicids2017_xgb_metrics.json`
- Generates confusion matrix plot

### Viewing Results

**Metrics:**
```bash
# View metrics for XGBoost
cat results/cicids2017_xgb_metrics.json

# View metrics for LightGBM
cat results/cicids2017_lgbm_metrics.json

# Compare metrics across experiments
dvc metrics show
```

**Plots:**
- Confusion matrices: `results/cicids2017_xgb_confusion_matrix.png`
- ROC curves: `results/cicids2017_xgb_roc_curve.png`
- Classification reports: `results/cicids2017_xgb_classification_report.txt`

---

## Configuration Files

Each experiment is controlled by a YAML config file in `configs/`:

**Example: `configs/cicids2017_xgb.yaml`**
```yaml
dataset:
  name: cicids2017
  raw_path: data/raw/cic_ids_2017
  processed_dir: data/processed/cicids2017

preprocessing:
  undersampling:
    enabled: true
    strategy:
      Normal Traffic: 500000

  oversampling:
    enabled: true
    method: SMOTE
    strategy:
      Bots: 2000
      Web Attacks: 2000
      ...

model:
  type: xgboost
  params_file: params/cicids2017_xgb_params.json

output:
  model_path: models/cicids2017_xgb.pkl
  metrics_path: results/cicids2017_xgb_metrics.json
```

**To create a new experiment:**
1. Copy an existing config file
2. Modify dataset paths, preprocessing parameters, or output paths
3. Create corresponding params file (or run hyperparameter tuning in notebook)
4. Add new stages to `dvc.yaml`

---

## Hyperparameter Files

Best hyperparameters are stored as JSON in `params/`:

**Example: `params/cicids2017_xgb_params.json`**
```json
{
  "objective": "binary:logistic",
  "device": "cuda",
  "booster": "dart",
  "max_depth": 6,
  "learning_rate": 0.1467,
  "n_estimators": 439,
  ...
}
```

These files bridge the gap between:
- **Exploration** (Jupyter notebooks with Optuna tuning) → Find best params
- **Production** (DVC scripts) → Use best params for reproducible training

---

## Workflow Comparison

| Aspect | Exploratory (Notebooks) | Reproducible (DVC) |
|--------|------------------------|-------------------|
| **Purpose** | Research, tuning, visualization | Production, collaboration |
| **Environment** | Google Colab, local Jupyter | Any environment with Python |
| **Hyperparameter Tuning** | Yes (Optuna, 30+ trials) | No (uses pre-tuned params) |
| **Runtime** | ~30-60 min (with tuning) | ~5-10 min (no tuning) |
| **Output** | Best params + trained models | Trained models + metrics |
| **Tracked by Git** | Params JSON files | Configs, scripts, params |
| **Tracked by DVC** | N/A | Datasets, models, metrics |

---

## Collaboration Guide

### For Researchers (Adding New Experiments)

1. **Run experiments in Jupyter notebooks** (Colab recommended)
2. **Export best hyperparameters** to `params/` directory
3. **Commit params to git:**
   ```bash
   git add params/my_new_experiment_params.json
   git commit -m "Add hyperparameters for new experiment"
   ```
4. **Optional: Create config file** for reproducibility
5. **Optional: Update `dvc.yaml`** with new pipeline stages

### For Collaborators (Reproducing Results)

1. **Clone repository:**
   ```bash
   git clone <repo-url>
   cd ensemble
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Pull data with DVC:**
   ```bash
   # Authenticate (if using GCS)
   gcloud auth activate-service-account --key-file=.credentials/<your-key>.json

   # Pull datasets
   dvc pull -r gcs
   ```

4. **Reproduce experiments:**
   ```bash
   # Run complete pipeline
   dvc repro evaluate_cicids2017_xgb

   # View results
   cat results/cicids2017_xgb_metrics.json
   ```

**No GPU? No problem!** Scripts automatically fall back to CPU if CUDA is unavailable.

---

## Advanced Usage

### Re-running Hyperparameter Tuning

If you have sufficient compute and want to re-tune hyperparameters:

1. Run the Jupyter notebook with Optuna
2. Export new best params
3. Re-run training with `dvc repro --force train_cicids2017_xgb`

### Adding a New Dataset (e.g., CICIDS2018)

1. Track dataset with DVC:
   ```bash
   dvc add data/raw/cic_ids_2018
   git add data/raw/cic_ids_2018.dvc
   ```

2. Create config file: `configs/cicids2018_xgb.yaml`

3. Run notebook to find best params → export to `params/cicids2018_xgb_params.json`

4. Add stages to `dvc.yaml`:
   ```yaml
   preprocess_cicids2018_xgb:
     cmd: python scripts/preprocess_cicids2017.py --config configs/cicids2018_xgb.yaml
     ...
   ```

5. Run pipeline: `dvc repro evaluate_cicids2018_xgb`

---

## Testing Your Setup

Before running the full pipeline, test your setup:

```bash
python scripts/test_pipeline.py
```

This quick test script verifies:
- All configuration files are valid
- All hyperparameter files exist
- Required Python packages are installed
- DVC is properly configured
- Directory structure is correct

**Run this test first to catch configuration issues early!**

---

## Troubleshooting

### Problem: `dvc pull` fails with authentication error

**Symptoms:**
```
ERROR: failed to connect to gs (...) - Anonymous caller does not have storage.objects.get access
```

**Solution:**
```bash
# Authenticate with Google Cloud
gcloud auth activate-service-account --key-file=".credentials/stormbird-vn-svac-2e541f116f50.json"

# Then try again
dvc pull -r gcs
```

---

### Problem: `ModuleNotFoundError` when running scripts

**Symptoms:**
```
ModuleNotFoundError: No module named 'xgboost'
```

**Solution:**
```bash
# Install all dependencies
pip install -r requirements.txt

# Verify installation
python scripts/test_pipeline.py
```

---

### Problem: `dvc repro` shows "Stage is up to date"

**Symptoms:**
```
Stage 'train_cicids2017_xgb' didn't change, skipping
```

**Solution:**
```bash
# Force re-run of a specific stage
dvc repro --force train_cicids2017_xgb

# Or force re-run entire pipeline
dvc repro --force evaluate_cicids2017_xgb
```

---

### Problem: Out of memory during training

**Symptoms:**
```
MemoryError: Unable to allocate array
```

**Solution 1: Use CPU instead of GPU (in params JSON)**
```json
{
  "device": "cpu",  // Change from "cuda"
  ...
}
```

**Solution 2: Reduce batch size or use subsampling**
Edit preprocessing config to reduce data size.

---

### Problem: CUDA/GPU not detected

**Symptoms:**
```
GPU available: False
```

**Solution:**
```bash
# Check if CUDA is available
python -c "import torch; print(torch.cuda.is_available())"

# If False, models will automatically fall back to CPU
# Performance will be slower but results will be identical
```

---

### Problem: Config file not found

**Symptoms:**
```
FileNotFoundError: configs/cicids2017_xgb.yaml
```

**Solution:**
Make sure you're running scripts from the project root directory:
```bash
# Wrong (from scripts/ directory)
cd scripts
python preprocess_cicids2017.py --config ../configs/cicids2017_xgb.yaml

# Correct (from project root)
cd /path/to/ensemble
python scripts/preprocess_cicids2017.py --config configs/cicids2017_xgb.yaml
```

---

### Problem: DVC stages not found

**Symptoms:**
```
ERROR: failed to reproduce 'evaluate_cicids2017_xgb': stage 'evaluate_cicids2017_xgb' not found
```

**Solution:**
```bash
# Check available stages
dvc dag

# Verify dvc.yaml is valid
python scripts/test_pipeline.py
```

---

### Problem: Data preprocessing fails with label mismatch

**Symptoms:**
```
KeyError: 'Label'
```

**Solution:**
- Check that your raw data has the correct column names
- CICIDS2017: should have "Label" column
- CICIDS2018: should have "Label" column
- Column names are case-sensitive!

---

### Problem: Want to add custom preprocessing

**Solution:**
1. Create new config file (copy existing one)
2. Modify preprocessing parameters
3. Add new stage to `dvc.yaml`
4. Run: `dvc repro <your_new_stage>`

Example:
```yaml
# configs/cicids2017_xgb_custom.yaml
preprocessing:
  undersampling:
    strategy:
      Normal Traffic: 100000  # Reduce further
```

---

### Problem: Different results from notebook

**Possible causes:**
1. **Random seed difference** - Check `random_state` in config
2. **Data preprocessing difference** - Verify undersampling/SMOTE settings
3. **Hyperparameter difference** - Compare params JSON with notebook
4. **Library version difference** - Check `pip list` vs notebook environment

**Debug steps:**
```bash
# Compare hyperparameters
cat params/cicids2017_xgb_params.json

# Check preprocessing settings
cat configs/cicids2017_xgb.yaml

# Verify library versions
pip list | grep -E "(xgboost|lightgbm|scikit-learn)"
```

---

## Datasets

- CIC-IDS2017: ~2.8M samples, 80 features, 8 attack types (DoS, DDoS, etc.).
- CIC-IDS2018: ~16M samples, similar features, advanced attacks.

Source: Canadian Institute for Cybersecurity and IDS-2018.
Preprocessing: Handle imbalances with scale_pos_weight, categorical encoding.

## Results

Best Model: LightGBM on 2017 data (F1: 0.97).
See figures/ for confusion matrices.

## Contributing

Fork, branch, PR.
Add new notebooks/models with DVC tracking.
Run black . for code formatting.

## License
MIT License. See LICENSE (add if needed).
Contact
Built for educational/research purposes. Questions? Open an issue.
Last Updated: October 03, 2025