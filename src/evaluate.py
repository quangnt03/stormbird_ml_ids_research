from pathlib import Path
import pandas as pd, joblib
from sklearn.metrics import precision_score, recall_score, f1_score
from dvclive import Live

test = pd.read_csv("data/splits/test.csv")
X, y = test.drop(columns=["label"]), test["label"]
clf = joblib.load("models/model.pkl")
y_pred = clf.predict(X)

precision = float(precision_score(y, y_pred))
recall    = float(recall_score(y, y_pred))
f1        = float(f1_score(y, y_pred))

Path("results/live").mkdir(parents=True, exist_ok=True)
with Live(dir="results/live", save_dvc_exp=True) as live:
    live.log_metric("precision", precision)
    live.log_metric("recall",    recall)
    live.log_metric("f1",        f1)

print(f"✅ metrics logged: precision={precision:.4f} recall={recall:.4f} f1={f1:.4f}")
