from pathlib import Path
import pandas as pd, yaml, joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

params = yaml.safe_load(open("configs/default.yaml"))
test_size = params["dataset"]["test_size"]
rs        = params["dataset"]["random_state"]
C         = params["model"]["C"]
max_iter  = params["model"]["max_iter"]

df = pd.read_csv("data/processed/processed.csv")

Path("data/splits").mkdir(parents=True, exist_ok=True)
train_df, test_df = train_test_split(df, test_size=test_size, random_state=rs, stratify=df["label"])
train_df.to_csv("data/splits/train.csv", index=False)
test_df.to_csv("data/splits/test.csv", index=False)

X_train, y_train = train_df.drop(columns=["label"]), train_df["label"]
clf = LogisticRegression(C=C, max_iter=max_iter)
clf.fit(X_train, y_train)

Path("models").mkdir(exist_ok=True)
joblib.dump(clf, "models/model.pkl")
print("✅ saved splits + models/model.pkl")
