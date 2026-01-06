
# zero_trust_analysis.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Seed for reproducibility
np.random.seed(42)

# 1. SIMULATE ZERO TRUST BEHAVIORAL DATA
entities = [f"Device_{i}" for i in range(1, 11)]  # 10 devices/users
data = {
    "Entity": entities,
    "Access Requests": np.random.randint(50, 200, size=10),
    "Auth Success Rate (%)": np.random.randint(70, 100, size=10),
    "Anomalous Behavior": np.random.randint(0, 20, size=10),
    "Device Compliance (%)": np.random.randint(50, 100, size=10)
}

df = pd.DataFrame(data)

print("\n=== Zero Trust Behavior Summary ===\n")
print(df)

# 2. TRAIN ISOLATION FOREST FOR ANOMALY DETECTION

# Select behavioural features
features = df[
    ["Access Requests",
     "Auth Success Rate (%)",
     "Anomalous Behavior",
     "Device Compliance (%)"]
]

# Standardize for ML
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# Train Isolation Forest
iso = IsolationForest(
    contamination=0.10,   # assume 10% anomalies
    random_state=42
)
iso.fit(X_scaled)

# Predict anomaly flags
df["Anomaly Flag"] = iso.predict(X_scaled)   # -1 = anomaly, 1 = normal
df["Anomaly Score"] = -iso.decision_function(X_scaled)

print("\n=== Isolation Forest Results ===\n")
print(df[["Entity","Anomaly Flag","Anomaly Score"]])

# 3. VISUAL ANALYTICS

# Melted dataframe for grouped metric bar chart
plt.figure(figsize=(12,6))
df_melted = df.melt(
    id_vars="Entity",
    value_vars=["Access Requests","Auth Success Rate (%)","Anomalous Behavior","Device Compliance (%)"],
    var_name="Metric", value_name="Value"
)
sns.barplot(data=df_melted, x="Entity", y="Value", hue="Metric")
plt.title("Zero Trust Metrics per Entity")
plt.xticks(rotation=45)
plt.legend(bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.show()

# Scatter: Access vs Anomalies
plt.figure(figsize=(10,6))
sns.scatterplot(
    data=df,
    x="Access Requests",
    y="Anomalous Behavior",
    hue="Device Compliance (%)",
    size="Auth Success Rate (%)",
    palette="coolwarm",
    sizes=(50, 300)
)
plt.title("Anomalous Behavior vs Access Requests (ZTA Analysis)")
plt.xlabel("Access Requests")
plt.ylabel("Anomalous Behavior")
plt.legend(bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.show()

# Correlation heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df.drop(columns="Entity").corr(), annot=True, cmap="YlGnBu")
plt.title("Correlation Between Zero Trust Metrics")
plt.tight_layout()
plt.show()

# Compliance vs Anomalies
plt.figure(figsize=(10,6))
sns.scatterplot(
    data=df,
    x="Device Compliance (%)",
    y="Anomalous Behavior",
    hue="Auth Success Rate (%)",
    size="Access Requests",
    palette="viridis",
    sizes=(50,300)
)
plt.title("Device Compliance vs Anomalous Behavior")
plt.xlabel("Device Compliance (%)")
plt.ylabel("Anomalous Behavior")
plt.tight_layout()
plt.show()

# 4. ANOMALY VISUALIZATION

plt.figure(figsize=(10,6))
sns.barplot(data=df, x="Entity", y="Anomaly Score", hue="Anomaly Flag", palette="Set2")
plt.title("Isolation Forest Anomaly Scores per Entity")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
