from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import cross_val_score
from utils.measurements import measure_resources
import numpy as np


def evaluate_model(model, X_train, y_train, X_test, y_test, cv=5):
    """
    Generic evaluation function for ML models.

    Parameters
    ----------
    model : sklearn-like model (fit, predict)
    X_train, y_train : training data
    X_test, y_test   : test data
    cv : int, default=5
        Number of folds for cross-validation.
    n_infer_runs : int, default=5
        Number of runs to average inference time and resource usage.

    Returns
    -------
    results : dict
        Dictionary with training time, inference time, CPU/mem/GPU usage,
        F1 scores for train, CV, and test.
    """
    results_train = measure_resources(model.fit, X_train, y_train)

    # Training metrics
    y_pred_train = model.predict(X_train)
    f1_train = f1_score(y_train, y_pred_train, average="weighted")

    # Cross-validation metrics
    scorer = make_scorer(f1_score, average="weighted")
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scorer)

    # ---------- Measure inference ----------

    results_test = measure_resources(model.predict, X_test)

    f1_test = f1_score(y_test, results_test["result"], average="weighted")

    # ---------- Store results ----------
    results = dict()
    results["train_time_sec"] = results_train["elapsed_time_sec"]
    results["test_time_sec"] = results_test["elapsed_time_sec"]

    results["cpu_usage_percent_train"] = abs(results_train["cpu_usage"])
    results["memory_usage_mb_train"] = abs(results_train["python_peak_memory_mb"])
    results["gpu_util_percent_train"] = abs(results_train["gpu_util_percent"])
    results["gpu_memory_used_mb_train"] = abs(results_train["gpu_memory_used_mb"])

    results["cpu_usage_percent_test"] = abs(results_test["cpu_usage"])
    results["memory_usage_mb_test"] = abs(results_test["python_peak_memory_mb"])
    results["gpu_util_percent_test"] = abs(results_test["gpu_util_percent"])
    results["gpu_memory_used_mb_test"] = abs(results_test["gpu_memory_used_mb"])

    results["f1_train"] = f1_train
    results["f1_cv_mean"] = np.mean(cv_scores)
    results["f1_cv_std"] = np.std(cv_scores)
    results["f1_test"] = f1_test

    return results
