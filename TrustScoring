import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay

np.random.seed(42)

#  Behavioral Trust Engine

class BehavioralTrustEngine:
    def __init__(self, n_users=60):
        self.n_users = n_users
        self.user_profiles = self._generate_users()

    def _generate_users(self):
        profiles = {}
        for i in range(self.n_users):
            user = f"user_{i}"

            user_type = random.choices(
                ["human", "bot", "adversary"],
                weights=[0.7, 0.15, 0.15]
            )[0]

            if user_type == "human":
                history = np.clip(np.random.normal(0.75, 0.1, 50), 0, 1)

            elif user_type == "bot":
                history = np.clip(np.random.normal(0.45, 0.15, 50), 0, 1)

            else:
                history = np.clip(np.random.normal(0.25, 0.2, 50), 0, 1)

            profiles[user] = {
                "user_type": user_type,
                "history": history.tolist(),
                "trust_score": round(np.mean(history), 2)
            }

        return profiles

    def generate_dataset(self):
        now = datetime.now()
        records = []

        for user, data in self.user_profiles.items():
            for i, score in enumerate(data["history"]):
                records.append({
                    "user_id": user,
                    "trust_score": score,
                    "timestamp": now - timedelta(minutes=5 * i),
                    "user_type": data["user_type"]
                })

        df = pd.DataFrame(records)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df


#  Analyzer

class TrustScoreAnalyzer:
    def __init__(self, dataset):
        self.dataset = dataset.copy()

    def assign_true_labels(self):
        self.dataset["true_label"] = self.dataset["user_type"].apply(
            lambda x: 1 if x in ["bot", "adversary"] else 0
        )

        #  collapse to user-level truth BEFORE grouping
        self.user_truth = (
            self.dataset.groupby("user_id")["true_label"]
            .max()
            .reset_index()
        )

    def compute_behavioral_features(self):
        features = (
            self.dataset.groupby("user_id")["trust_score"]
            .agg(["mean", "std", "min", "max"])
            .fillna(0)
            .reset_index()
        )

        features["range"] = features["max"] - features["min"]
        features["cv"] = features["std"] / (features["mean"] + 1e-6)

        return features

    def train_anomaly_detector(self, features):
        X = features[["mean", "std", "range", "cv"]]

        model = IsolationForest(
            contamination=0.1,
            random_state=42
        )
        model.fit(X)

        anomaly_scores = -model.decision_function(X)

        return model, anomaly_scores

    def compute_adaptive_thresholds(self):
        mu = self.dataset["trust_score"].mean()
        sigma = self.dataset["trust_score"].std()

        return {
            "block": max(0.1, mu - 2 * sigma),
            "challenge": max(0.3, mu - sigma),
            "allow": max(0.6, mu - 0.5 * sigma),
        }

#  Visualization

class TrustResearchVisualizer:
    def __init__(self):
        try:
            plt.style.use("seaborn-v0_8")
        except:
            plt.style.use("ggplot")

    def plot_trust_distribution(self, dataset):
        plt.figure(figsize=(8, 5))
        sns.histplot(dataset["trust_score"], bins=20, kde=True)
        plt.title("Trust Score Distribution")
        plt.show()

    def plot_temporal_patterns(self, dataset):
        df = dataset.copy()
        df["hour"] = df["timestamp"].dt.hour

        plt.figure(figsize=(10, 5))
        sns.lineplot(x="hour", y="trust_score", data=df)
        plt.title("Temporal Trust Patterns")
        plt.show()

    def plot_roc(self, y_true, y_scores):
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.2f}")
        plt.plot([0, 1], [0, 1], "--", color="gray")
        plt.legend()
        plt.title("Anomaly Detection ROC")
        plt.show()

    def plot_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=["Normal", "Anomaly"])
        disp.plot(cmap="Blues")
        plt.title("Confusion Matrix (Threshold = 0.5)")
        plt.show()

#  Runner

def run_research_pipeline():
    print("\n=== Dynamic Trust Scoring Research Pipeline ===\n")

    engine = BehavioralTrustEngine()
    dataset = engine.generate_dataset()

    analyzer = TrustScoreAnalyzer(dataset)
    visualizer = TrustResearchVisualizer()

    analyzer.assign_true_labels()

    features = analyzer.compute_behavioral_features()

    model, anomaly_scores = analyzer.train_anomaly_detector(features)

    features["anomaly_score"] = anomaly_scores
    features["predicted_anomaly"] = (anomaly_scores > anomaly_scores.mean()).astype(int)

    # ALIGN truth + predictions
    merged = features.merge(analyzer.user_truth, on="user_id")

    y_true = merged["true_label"].values
    y_pred = features["predicted_anomaly"].values
    y_scores = features["anomaly_score"].values

    thresholds = analyzer.compute_adaptive_thresholds()
    print("Adaptive thresholds:", thresholds, "\n")

    visualizer.plot_trust_distribution(dataset)
    visualizer.plot_temporal_patterns(dataset)
    visualizer.plot_roc(y_true, y_scores)
    visualizer.plot_confusion_matrix(y_true, y_pred)

if __name__ == "__main__":
    run_research_pipeline()
