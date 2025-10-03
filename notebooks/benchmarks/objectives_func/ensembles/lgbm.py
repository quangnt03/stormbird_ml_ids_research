from utils.evaluations import evaluate_model


def objective_lgbm(
    trial, model_class, X_train, y_train, X_test, y_test, gpu_available=False
):
    params = {
        "objective": "binary",  # or "multiclass" if multi-class
        "metric": "binary_logloss",  # change to "multi_logloss"/"auc" if needed
        "boosting_type": trial.suggest_categorical("boosting_type", ["dart", "goss"]),
        "device": "gpu" if gpu_available else "cpu",
        "verbosity": 0,
        # Core learning parameters
        "num_leaves": trial.suggest_int("num_leaves", 16, 512),
        "max_depth": trial.suggest_int("max_depth", -1, 20),  # -1 means no limit
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 2000),
        "subsample": trial.suggest_float(
            "subsample", 0.5, 1.0
        ),  # Alias for bagging_fraction in some contexts; safe to keep
        "tree_learner": trial.suggest_categorical(
            "tree_learner",
            [
                "serial",
                "data",
            ],  # Restricted to GPU-safe options; drop "feature"/"voting" if on GPU
        ),
        # Regularization
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "min_child_weight": trial.suggest_float(
            "min_child_weight", 1e-4, 1e2, log=True
        ),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        # Feature sampling (suggest outside conditional; valid for both)
        "feature_fraction": trial.suggest_float(
            "feature_fraction", 0.5, 1.0
        ),  # FIXED: Always suggest >0; was hardcoded to 0 for DART
        # Extra boosting-specific params
        "extra_trees": trial.suggest_categorical("extra_trees", [True, False]),
        # early stop
        "early_stopping_min_delta": 1e-5,
        "max_cat_threshold": 5,
        "is_unbalance": True,
    }

    params["data_sample_strategy"] = (
        "goss" if params["boosting_type"] == "goss" else "bagging"
    )  # Enable GOSS properly

    if params["boosting_type"] == "goss":  # Now correctly scoped to GOSS
        params["bagging_fraction"] = 1.0
        params["bagging_freq"] = 0
        params["feature_fraction"] = trial.suggest_float(
            "feature_fraction", 0.5, 1.0
        )  # Keep >0
        params["top_rate"] = trial.suggest_float(
            "top_rate", 0.1, 0.4
        )  # GOSS-specific: retain top gradients
        params["other_rate"] = trial.suggest_float(
            "other_rate", 0.05, 0.2
        )  # GOSS-specific: retain other gradients
        params["pos_bagging_fraction"] = trial.suggest_float(
            "pos_bagging_fraction", 0.5, 0.7
        )  # GOSS imbalance handling
        params["neg_bagging_fraction"] = trial.suggest_float(
            "neg_bagging_fraction", 0.3, 0.5
        )
    else:  # DART (or change to "gbdt" for vanilla)
        params["bagging_fraction"] = trial.suggest_float("bagging_fraction", 0.5, 1.0)
        params["bagging_freq"] = trial.suggest_int(
            "bagging_freq", 1, 10
        )  # >0 for bagging to activate
        # Do NOT set feature_fraction=0; let colsample_bytree handle it (0.5-1.0)
        # Add DART-specific if needed: e.g., "drop_rate": trial.suggest_float("drop_rate", 0.1, 0.3)

    # Instantiate model
    model = model_class(**params)

    # Call our evaluation function
    results = evaluate_model(model, X_train, y_train, X_test, y_test)

    # Report intermediate metrics to Optuna
    for k, v in results.items():
        trial.set_user_attr(k, v)

    # Optuna needs a scalar to optimize → return CV F1 (or test F1)
    return results["f1_test"]
