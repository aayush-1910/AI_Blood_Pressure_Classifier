import pandas as pd
import numpy as np
import os
import joblib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score, confusion_matrix,
                              classification_report)

os.makedirs("models", exist_ok=True)
os.makedirs("static/plots", exist_ok=True)

# ── 1. Load UCI Cleveland Heart Disease dataset ──────────────────────────────
url = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)
cols = ["age","sex","cp","trestbps","chol","fbs","restecg",
        "thalach","exang","oldpeak","slope","ca","thal","target"]

print("Downloading UCI Cleveland Heart Disease dataset...")
df = pd.read_csv(url, names=cols, na_values="?")
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)
print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# ── 2. Create BP state labels from systolic BP (trestbps) ────────────────────
def classify_bp(sbp):
    if sbp < 120:
        return "Normal"
    elif sbp < 130:
        return "Elevated"
    elif sbp < 140:
        return "Hypertension Stage 1"
    elif sbp < 180:
        return "Hypertension Stage 2"
    else:
        return "Hypertensive Crisis"

df["bp_state"] = df["trestbps"].apply(classify_bp)
print("\nClass distribution:")
print(df["bp_state"].value_counts())

# ── 3. Feature selection ──────────────────────────────────────────────────────
FEATURES = ["age", "sex", "trestbps", "chol", "fbs",
            "thalach", "exang", "oldpeak"]
TARGET = "bp_state"

X = df[FEATURES].values
y = df[TARGET].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(FEATURES, "models/features.pkl")

# ── 4. Train all four models ──────────────────────────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree":        DecisionTreeClassifier(max_depth=6, random_state=42),
    "Random Forest":        RandomForestClassifier(n_estimators=200, random_state=42),
    "Support Vector Machine": SVC(kernel="rbf", probability=True, random_state=42),
}

results = {}
best_model_name, best_f1 = None, 0.0

for name, clf in models.items():
    clf.fit(X_train_sc, y_train)
    preds = clf.predict(X_test_sc)

    acc  = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average="weighted", zero_division=0)
    rec  = recall_score(y_test, preds, average="weighted", zero_division=0)
    f1   = f1_score(y_test, preds, average="weighted", zero_division=0)

    results[name] = {
        "accuracy":  round(acc * 100, 2),
        "precision": round(prec * 100, 2),
        "recall":    round(rec * 100, 2),
        "f1_score":  round(f1 * 100, 2),
    }

    safe_name = name.replace(" ", "_")
    joblib.dump(clf, f"models/{safe_name}.pkl")

    # Confusion matrix plot
    cm = confusion_matrix(y_test, preds, labels=clf.classes_)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=clf.classes_, yticklabels=clf.classes_, ax=ax)
    ax.set_title(f"Confusion Matrix — {name}")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"static/plots/cm_{safe_name}.png", dpi=100)
    plt.close()

    print(f"\n{name}  →  Acc: {acc*100:.1f}%  F1: {f1*100:.1f}%")
    if f1 > best_f1:
        best_f1, best_model_name = f1, name

# ── 5. Save metadata ──────────────────────────────────────────────────────────
metadata = {
    "best_model":    best_model_name,
    "best_model_key": best_model_name.replace(" ", "_"),
    "classes":       list(df["bp_state"].unique()),
    "results":       results,
}
with open("models/metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

# Comparison bar chart
fig, ax = plt.subplots(figsize=(10, 5))
metrics_df = pd.DataFrame(results).T.reset_index().rename(columns={"index": "Model"})
x = np.arange(len(metrics_df))
width = 0.2
for i, metric in enumerate(["accuracy", "precision", "recall", "f1_score"]):
    ax.bar(x + i * width, metrics_df[metric], width, label=metric.capitalize())
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(metrics_df["Model"], rotation=15, ha="right")
ax.set_ylim(0, 115)
ax.set_ylabel("Score (%)")
ax.set_title("Model Performance Comparison")
ax.legend()
plt.tight_layout()
plt.savefig("static/plots/model_comparison.png", dpi=100)
plt.close()

print(f"\n✅  Best model: {best_model_name}  (F1 = {best_f1*100:.1f}%)")
print("All models and plots saved successfully.")