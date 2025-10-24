from pathlib import Path
import pandas as pd, numpy as np, yaml

params = yaml.safe_load(open("configs/default.yaml"))
n  = params["dataset"]["n_samples"]
rs = params["dataset"]["random_state"]

Path("data/processed").mkdir(parents=True, exist_ok=True)

# demo data; replace with your real preprocessing later (keep the same output path)
rng = np.random.default_rng(rs)
X = rng.normal(size=(n, 4))
y = (0.8*X[:,0] - 0.4*X[:,1] + 0.6*X[:,2] + rng.normal(0,0.5,n) > 0).astype(int)

df = pd.DataFrame(X, columns=["f1","f2","f3","f4"])
df["label"] = y
df.to_csv("data/processed/processed.csv", index=False)
print("✅ wrote data/processed/processed.csv", df.shape)
