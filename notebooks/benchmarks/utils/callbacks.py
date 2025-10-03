from optuna import Study, Trial


class EarlyStoppingCallback:
    def __init__(self, patience=5, threshold=0.05, direction="maximize"):
        self.patience = patience
        self.threshold = threshold  # relative improvement (5% = 0.05)
        self.direction = direction
        self.best_value = None
        self.counter = 0

    def __call__(self, study, trial):
        value = trial.value
        if value is None:
            return

        # First trial
        if self.best_value is None:
            self.best_value = value
            self.counter = 0
            return

        # Check improvement
        if self.direction == "maximize":
            improvement = (value - self.best_value) / abs(self.best_value)
            if improvement > self.threshold:
                self.best_value = value
                self.counter = 0
            else:
                self.counter += 1
        else:  # minimize
            improvement = (self.best_value - value) / abs(self.best_value)
            if improvement > self.threshold:
                self.best_value = value
                self.counter = 0
            else:
                self.counter += 1

        # Stop if no improvement for patience trials
        if self.counter >= self.patience:
            print(
                f"⏹️ Early stopping: No {self.threshold * 100:.1f}% improvement in {self.patience} consecutive trials."
            )
            study.stop()


class ThresholdStopper:
    def __init__(self, threshold: float):
        self.threshold = threshold

    def __call__(self, study: Study, trial: Trial):
        if study.best_value is not None and study.best_value >= self.threshold:
            print(
                f"Stopping study! Best value {study.best_value} <= threshold {self.threshold}"
            )
            study.stop()
