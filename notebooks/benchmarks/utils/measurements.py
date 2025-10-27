import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


# Thresholding Function
def apply_threshold(scores, threshold):
    """Convert continuous anomaly scores to binary 0/1 labels."""
    return (scores > threshold).astype(int)

# ------------------------------------------------------------

# Core Evaluation Metrics
def compute_core_metrics(y_true, y_pred, scores):
    """Compute Precision, Recall, F1, and AUC-PR."""
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc_pr = average_precision_score(y_true, scores)
    return {
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "AUC-PR": auc_pr,
    }

# ------------------------------------------------------------
# Behavioral / Real-time Metrics
def compute_behavioral_metrics(y_true, y_pred, timestamps=None):
    """
    Compute temporal metrics such as TTD, Detection Latency,
    ARL2FA, and Event Coverage.
    """
    # Find anomaly event windows
    anomaly_events = []
    in_anomaly = False
    start = None
    for i, label in enumerate(y_true):
        if label == 1 and not in_anomaly:
            start = i
            in_anomaly = True
        elif label == 0 and in_anomaly:
            anomaly_events.append((start, i - 1))
            in_anomaly = False
    if in_anomaly:
        anomaly_events.append((start, len(y_true) - 1))

    if not anomaly_events:
        return {"TTD": None, "Latency": None, "ARL2FA": None, "Coverage": None}

    ttds, latencies, coverage = [], [], []
    false_alarm_durations = []

    last_false_alarm = 0
    for start, end in anomaly_events:
        detection_points = [i for i in range(start, end + 1) if y_pred[i] == 1]
        # Time To Detect (TTD)
        if any(y_pred[:start]):
            ttd = np.argmax(y_pred[:start][::-1] == 1)
        else:
            ttd = None
        # Detection Latency
        latency = (detection_points[0] - start) if detection_points else None
        # Coverage
        covered = 1 if detection_points else 0

        ttds.append(ttd)
        latencies.append(latency)
        coverage.append(covered)

    # ARL2FA (average run length to false alarm)
    false_alarm_idxs = [i for i in range(len(y_true)) if y_true[i] == 0 and y_pred[i] == 1]
    if len(false_alarm_idxs) > 1:
        gaps = np.diff(false_alarm_idxs)
        arl2fa = np.mean(gaps)
    else:
        arl2fa = None

    return {
        "TTD": np.nanmean([v for v in ttds if v is not None]) if ttds else None,
        "Latency": np.nanmean([v for v in latencies if v is not None]) if latencies else None,
        "ARL2FA": arl2fa,
        "Coverage": np.mean(coverage),
    }

# ------------------------------------------------------------
# Model Stability Test
def evaluate_stability(metric_list):
    """
    Evaluate stability by computing standard deviation of metric values
    across multiple runs.
    """
    df = pd.DataFrame(metric_list)
    stability = df.std().to_dict()
    return {"Metric Variability": stability}

# ------------------------------------------------------------
# Visualization Tools
def plot_pr_curve(y_true, scores):
    """Plot Precision-Recall curve."""
    from sklearn.metrics import precision_recall_curve
    precision, recall, _ = precision_recall_curve(y_true, scores)
    plt.plot(recall, precision, label="PR Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.show()


def plot_confusion(y_true, y_pred):
    """Plot Confusion Matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
    plt.show()

# ------------------------------------------------------------
# Main Evaluation Pipeline
def evaluate_model(y_true, scores, threshold):
    """
    Evaluate a model fully — includes:
    Core metrics + Behavioral metrics
    """
    y_pred = apply_threshold(scores, threshold)

    # Core metrics
    core = compute_core_metrics(y_true, y_pred, scores)

    # Behavioral metrics
    behavior = compute_behavioral_metrics(y_true, y_pred)

    # Merge results
    metrics = {**core, **behavior}
    return metrics


# Example
if __name__ == "__main__":
    # Example data
    np.random.seed(42)
    y_true = np.random.choice([0, 1], size=1000, p=[0.98, 0.02])
    scores = np.random.rand(1000)

    # Choose threshold
    threshold = 0.9

    # Evaluate
    results = evaluate_model(y_true, scores, threshold)
    print("Evaluation Results:\n", results)

    # Visualizations
    plot_pr_curve(y_true, scores)
    plot_confusion(y_true, (scores > threshold).astype(int))
