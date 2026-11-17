
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score
)

# Create output folder
os.makedirs("outputs", exist_ok=True)

print("=" * 50)
print("CREDIT CARD FRAUD DETECTION")
print("Using Isolation Forest")
print("=" * 50)

# Load dataset
data = pd.read_csv("data/creditcard.csv")

print("\nDataset Loaded Successfully!")
print("Total Transactions:", len(data))
print("Total Features:", data.shape[1])

print("\nFirst 5 Records:")
print(data.head())

print("\nClass Distribution:")
print(data["Class"].value_counts())

# Class distribution visualization
plt.figure(figsize=(7, 5))
sns.countplot(x="Class", data=data)
plt.title("Normal vs Fraudulent Transactions")
plt.xlabel("Class (0 = Normal, 1 = Fraud)")
plt.ylabel("Number of Transactions")
plt.savefig("outputs/class_distribution.png")
plt.close()

# Transaction amount visualization
plt.figure(figsize=(8, 5))
sns.histplot(data["Amount"], bins=50, kde=True)
plt.title("Transaction Amount Distribution")
plt.xlabel("Transaction Amount")
plt.ylabel("Frequency")
plt.savefig("outputs/transaction_amount.png")
plt.close()

# Separate features and actual labels
X = data.drop(columns=["Class"])
y = data["Class"]

# Train Isolation Forest
print("\nTraining Isolation Forest Model...")

model = IsolationForest(
    n_estimators=100,
    contamination=0.00173,
    random_state=42,
    n_jobs=-1
)

model.fit(X)

# Predict anomalies
predictions = model.predict(X)

# Convert predictions: -1 = anomaly, 1 = normal
y_pred = np.where(predictions == -1, 1, 0)

print("\nModel Training Completed!")

# Model evaluation
print("\nClassification Report:")
print(classification_report(
    y,
    y_pred,
    labels=[0, 1],
    target_names=["Normal", "Fraud"],
    zero_division=0
))

precision = precision_score(y, y_pred, zero_division=0)
recall = recall_score(y, y_pred, zero_division=0)
f1 = f1_score(y, y_pred, zero_division=0)

anomaly_scores = -model.decision_function(X)
pr_auc = average_precision_score(y, anomaly_scores)

print("\nModel Evaluation:")
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1 Score:", round(f1, 4))
print("Precision-Recall AUC:", round(pr_auc, 4))

# Confusion matrix
cm = confusion_matrix(y, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Normal", "Fraud"],
    yticklabels=["Normal", "Fraud"]
)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("outputs/confusion_matrix.png")
plt.close()

# Save prediction results
results = pd.DataFrame({
    "Actual": y,
    "Predicted": y_pred
})

results.to_csv("outputs/prediction_results.csv", index=False)

# Predicted class distribution
plt.figure(figsize=(7, 5))
sns.countplot(x="Predicted", data=results)
plt.title("Predicted Normal vs Fraudulent Transactions")
plt.xlabel("Predicted Class (0 = Normal, 1 = Fraud)")
plt.ylabel("Number of Transactions")
plt.savefig("outputs/fraud_detection.png")
plt.close()

print("\nAll output files saved successfully!")
print("\nPROJECT EXECUTED SUCCESSFULLY!")
