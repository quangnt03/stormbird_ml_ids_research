# Intrusion Detection System (IDS) using Machine Learning on CIC-IDS Datasets

This project implements a machine learning-based Intrusion Detection System (IDS) using the CIC-IDS2017 and CIC-IDS2018 datasets. It focuses on binary/multiclass classification of network intrusions using gradient boosting models like LightGBM and XGBoost. The pipeline includes data preprocessing, model training, evaluation, and visualization of results (e.g., confusion matrices).

Key features:
- Data handling with DVC for versioning large datasets and models.
- Jupyter notebooks for reproducible experiments.
- Trained models saved as pickled files (versioned via DVC).
- Metrics: F1-score, confusion matrices, and ROC curves.

## Project Structure
```
project-root/
├── .idea/                  # IntelliJ/PyCharm IDE files (gitignored)
├── data/                   # Dataset files (gitignored; use DVC to pull)
│   ├── raw/                # Raw CIC-IDS CSV files
│   └── cleaned/            # Preprocessed features (e.g., scaled, encoded)
├── figures/                # Generated plots (e.g., confusion matrices; gitignored)
│   └── xgb_cicids2017_confusion...  # Example: XGBoost confusion matrix for 2017 dataset
├── models/                 # Trained models (gitignored; use DVC to pull)
│   ├── lgbm_cicids2017.pkl # LightGBM model for CIC-IDS2017
│   ├── lgbm_cicids2018.pkl # LightGBM model for CIC-IDS2018
│   ├── xgb_cicids2017.pkl  # XGBoost model for CIC-IDS2017
│   └── xgb_cicids2018.pkl  # XGBoost model for CIC-IDS2018
├── notebooks/              # Jupyter notebooks for analysis and training
│   ├── [10-01] CICIDS2018_preprocessing.ipynb  # Data loading and cleaning for 2018
│   ├── [10-02] CICIDS2018_XGB_Training.ipynb   # XGBoost training on 2018 data
│   ├── [15-01] CIC-IDS 2017_preprocessing.ipynb # Data loading and cleaning for 2017
│   └── [30-01] cic2017-benchmark.ipynb         # Benchmarking models on 2017 data
├── pkl/                    # Temporary pickle files (gitignored)
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

- **data/**: Large CSV files (~GBs); track with `dvc add data/` and commit `.dvc` files.
- **models/**: Serialized ML models; add via `dvc add models/` for versioning.
- **notebooks/**: Run in order (e.g., preprocessing first, then training).
- **figures/**: Outputs from notebooks (e.g., Matplotlib plots).

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

### 1. Data Preprocessing
- Run `[10-01] CICIDS2018_preprocessing.ipynb` and `[15-01] CIC-IDS 2017_preprocessing.ipynb`.
- Outputs: Cleaned data in `data/cleaned/` (features like flow duration, packet sizes; labels: benign/attack).

### 2. Model Training
- Run `[10-02] CICIDS2018_XGB_Training.ipynb` for XGBoost on 2018 data.
- Run `[30-01] cic2017-benchmark.ipynb` for LightGBM/XGBoost benchmarking on 2017 data (uses Optuna for hyperparameter tuning).
- Models are saved to `models/` and tracked with DVC.

### 3. Evaluation
- Confusion matrices and metrics are generated in `figures/`.
- Test F1-scores: ~0.95+ for binary classification (benign vs. attack).

### Example Training Command (Script Equivalent)
If notebooks are too heavy, convert to scripts:
```python
import lightgbm as lgb
from sklearn.model_selection import train_test_split

# Load data (assume preprocessed)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

params = {...}  # From Optuna or defaults
model = lgb.LGBMClassifier(**params)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
model.save_model('models/lgbm_cicids2017.txt')  # Or .pkl via joblib
```

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