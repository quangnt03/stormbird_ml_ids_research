from utils.evaluations import evaluate_model


def objective_xgb(trial, model_class, X_train, y_train, X_test, y_test):
    """
    Optuna objective function.
    - model_class : constructor (e.g., XGBClassifier, LGBMClassifier, RandomForestClassifier)
    - trial : Optuna trial
    """
    # Example: suggest some hyperparameters (you can customize per model)

    params = {
        "objective": "binary:logistic",
        "device": "gpu",
        "verbosity": 0,
        "eval_metric": "logloss",
        # Tree booster parameters
        "booster": trial.suggest_categorical("booster", ["gbtree", "dart"]),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "sampling_method": trial.suggest_categorical(
            "sampling_method", ["uniform", "gradient_based"]
        ),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
        "colsample_bynode": trial.suggest_float("colsample_bynode", 0.5, 1.0),
        "alpha": trial.suggest_float("alpha", 1e-8, 10.0, log=True),
        "lambda": trial.suggest_float("lambda", 1e-8, 10.0, log=True),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.5, 10.0),
        "grow_policy": trial.suggest_categorical(
            "grow_policy", ["depthwise", "lossguide"]
        ),
        "tree_method": "hist",
        "max_leaves": trial.suggest_int("max_leaves", 0, 256),
        "max_bin": trial.suggest_int("max_bin", 256, 512),
        # DART booster specific
        "sample_type": trial.suggest_categorical("sample_type", ["uniform", "weighted"])
        if trial.params.get("booster") == "dart"
        else "uniform",
        "normalize_type": trial.suggest_categorical(
            "normalize_type", ["tree", "forest"]
        )
        if trial.params.get("booster") == "dart"
        else "tree",
        "rate_drop": trial.suggest_float("rate_drop", 0.0, 0.5)
        if trial.params.get("booster") == "dart"
        else 0.0,
        "skip_drop": trial.suggest_float("skip_drop", 0.0, 0.5)
        if trial.params.get("booster") == "dart"
        else 0.0,
    }

    # Instantiate model
    model = model_class(**params)

    # Call our evaluation function
    results = evaluate_model(model, X_train, y_train, X_test, y_test)

    # Report intermediate metrics to Optuna
    for k, v in results.items():
        trial.set_user_attr(k, v)

    # Optuna needs a scalar to optimize → return CV F1 (or test F1)
    return results["f1_test"]
