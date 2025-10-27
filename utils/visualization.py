import matplotlib as plt
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
)


def plot_optuna_attributes(study):
    trial_numbers = [t.number for t in study.trials if t.state.name == "COMPLETE"]

    # Metric groups
    metrics = {
        "f1": ["f1_train", "f1_cv_mean", "f1_test"],
        "memory": [
            "memory_usage_mb_train",
            "memory_usage_mb_test",
        ],
        "cpu": ["cpu_usage_percent_train", "cpu_usage_percent_test"],
        "gpu": [
            "gpu_util_percent_train",
            "gpu_util_percent_test",
            "gpu_memory_used_mb_train",
            "gpu_memory_used_mb_test",
        ],
        "time": ["train_time_sec", "test_time_sec", "trial_duration_min"],
    }

    # Initialize storage
    data = {group: {a: [] for a in attrs} for group, attrs in metrics.items()}

    for t in study.trials:
        if t.state.name == "COMPLETE":
            # compute trial duration in minutes (absolute value to avoid negatives)
            if t.datetime_start and t.datetime_complete:
                duration = (t.datetime_complete - t.datetime_start).total_seconds() / 60
                duration = abs(duration)
            else:
                duration = None

            for group, attrs in metrics.items():
                for a in attrs:
                    if a == "trial_duration_min":
                        data[group][a].append(duration)
                    else:
                        data[group][a].append(t.user_attrs.get(a, None))

    # Plot each group
    for group, attrs in metrics.items():
        plt.figure(figsize=(8, 5))
        for a in attrs:
            plt.plot(trial_numbers, data[group][a], marker="o", label=a)
        plt.xlabel("Trial")
        if group == "f1":
            plt.ylabel("F1 Score")
        elif group == "memory":
            plt.ylabel("Memory (MB)")
        elif group == "cpu":
            plt.ylabel("CPU Usage (%)")
        elif group == "gpu":
            plt.ylabel("GPU Usage (%)")
        elif group == "time":
            plt.ylabel("Time (sec or min)")
        plt.title(f"{group.upper()} Metrics per Trial")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.show()


def plot_confusion_matrix(
    y_true, y_pred, labels, title="Confusion Matrix", *args, **kwargs
):
    cm = confusion_matrix(y_true, y_pred, labels=labels, *args, **kwargs)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=labels, *args, **kwargs
    )
    disp.plot(cmap=plt.cm.Blues, values_format="d")
    plt.title(title)
    plt.show()
