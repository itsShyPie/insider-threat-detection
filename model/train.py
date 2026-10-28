"""
Training and evaluation script for the AI-Powered Insider Threat Detection
System.

Trains four supervised classifiers on the prescribed Kaggle dataset and
produces all artefacts required by the project submission:

  - models/rf_model.pkl       - primary Random Forest model
  - models/preprocessor.pkl   - fitted preprocessor (encoders + feature spec)
  - models/lr_model.pkl       - Logistic Regression baseline
  - models/dt_model.pkl       - Decision Tree baseline
  - models/svm_model.pkl      - SVM comparison model
  - reports/metrics.json      - full evaluation metrics for all models
  - reports/confusion_matrix.png  - RF confusion matrix
  - reports/feature_importance.png - RF feature importance plot
  - reports/model_comparison.png - bar chart of all 4 models across 4 metrics

Run from the project root:
    python -m model.train

All hyperparameters are intentionally modest so training completes quickly on
a laptop while still producing competitive results.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (works in Docker, no display)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

import pickle

from .preprocessor import Preprocessor, TARGET_COL

# Configuration

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "insider_threat_clean_dataset.csv"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20


# Helpers

def evaluate(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute all required metrics for a single model."""
    return {
        "model": name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_malicious": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_malicious": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_malicious": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def save_pickle(obj, path: Path) -> None:
    """Persist a model with pickle."""
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def plot_confusion_matrix(cm: np.ndarray, path: Path) -> None:
    """Render confusion matrix for the Random Forest model."""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title("Confusion Matrix - Random Forest", fontsize=13)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Normal", "Malicious"])
    ax.set_yticklabels(["Normal", "Malicious"])

    # Annotate cells
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color=color, fontsize=12, fontweight="bold")

    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()


def plot_feature_importance(importances: np.ndarray, feature_names: list[str], path: Path) -> None:
    """Horizontal bar chart of top 15 features."""
    order = np.argsort(importances)[::-1][:15]
    names = [feature_names[i] for i in order]
    values = importances[order]

    fig, ax = plt.subplots(figsize=(9, 7))
    bars = ax.barh(range(len(names)), values[::-1], color="#2E75B6")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names[::-1])
    ax.set_xlabel("Importance (Gini decrease)")
    ax.set_title("Top 15 Features - Random Forest", fontsize=13)
    for bar, v in zip(bars, values[::-1]):
        ax.text(v + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{v:.3f}", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()


def plot_model_comparison(all_metrics: list[dict], path: Path) -> None:
    """Side-by-side bar chart comparing all 4 models on 4 metrics."""
    models = [m["model"] for m in all_metrics]
    metric_keys = ["accuracy", "precision_malicious", "recall_malicious", "f1_malicious"]
    metric_labels = ["Accuracy", "Precision (mal.)", "Recall (mal.)", "F1 (mal.)"]

    n_models = len(models)
    n_metrics = len(metric_keys)
    x = np.arange(n_metrics)
    width = 0.18

    colors = ["#1F3864", "#2E75B6", "#7F6000", "#C00000"]
    fig, ax = plt.subplots(figsize=(11, 6))
    for i, m in enumerate(all_metrics):
        vals = [m[k] for k in metric_keys]
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=m["model"], color=colors[i % len(colors)])
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01,
                    f"{v:.2f}", ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison - All Metrics", fontsize=13)
    ax.legend(loc="lower right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()


# Main training routine

def main() -> None:
    print(f"[1/6] Loading dataset from {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH)
    print(f"      Loaded {len(df):,} records, {df.shape[1]} columns")
    print(f"      Class distribution: {dict(df[TARGET_COL].value_counts())}")
    print(f"      Class imbalance ratio: {df[TARGET_COL].mean():.4f} malicious")

    print("\n[2/6] Fitting preprocessor and engineering features ...")
    preprocessor = Preprocessor()
    X, y = preprocessor.fit_transform(df)
    feature_names = preprocessor.feature_names()
    print(f"      Feature matrix: {X.shape}")
    print(f"      Features: {len(feature_names)} columns including 2 derived")

    print("\n[3/6] Stratified train/test split (80/20) ...")
    # stratify=y preserves the same 5.4% malicious rate in BOTH train and
    # test splits. Without stratification a random split could put few or
    # no malicious cases in the test set, making evaluation unreliable.
    # random_state=42 makes the split reproducible across runs.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"      Train: {X_train.shape[0]:,}   Test: {X_test.shape[0]:,}")
    print(f"      Test set malicious count: {int(y_test.sum()):,}")

    # Scaler used only by Logistic Regression and SVM (tree models don't need it)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    all_metrics = []

    # Random Forest (primary)
    print("\n[4/6] Training Random Forest (primary model) ...")
    t0 = time.time()
    # Hyperparameters:
    # - n_estimators=200: aggregate over 200 trees (more = better but slower)
    # - max_depth=20: cap depth to avoid overfitting on this 23-feature space
    # - min_samples_split=10, min_samples_leaf=2: regularisation to keep
    #   small noisy subsets from creating unstable splits
    # - class_weight='balanced': counters the 94.6%/5.4% imbalance by
    #   weighting the minority (malicious) class higher during fitting.
    #   This addresses the base-rate fallacy from Stallings & Brown Ch. 8.
    # - n_jobs=-1: parallel training across all CPU cores
    # - random_state=42: deterministic so results are reproducible
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    rf.fit(X_train, y_train)
    rf_train_time = time.time() - t0
    rf_pred = rf.predict(X_test)
    rf_metrics = evaluate("Random Forest", y_test.values, rf_pred)
    rf_metrics["train_time_seconds"] = round(rf_train_time, 2)
    all_metrics.append(rf_metrics)
    print(f"      Trained in {rf_train_time:.1f}s")
    print(f"      Accuracy: {rf_metrics['accuracy']:.4f}")
    print(f"      Recall (malicious): {rf_metrics['recall_malicious']:.4f}")
    print(f"      F1 (malicious): {rf_metrics['f1_malicious']:.4f}")

    save_pickle(rf, MODELS_DIR / "rf_model.pkl")
    preprocessor.save(str(MODELS_DIR / "preprocessor.pkl"))

    # Logistic Regression (baseline)
    print("\n[5/6] Training comparison models ...")
    print("      Logistic Regression ...")
    t0 = time.time()
    lr = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=RANDOM_STATE,
    )
    lr.fit(X_train_scaled, y_train)
    lr_metrics = evaluate("Logistic Regression", y_test.values, lr.predict(X_test_scaled))
    lr_metrics["train_time_seconds"] = round(time.time() - t0, 2)
    all_metrics.append(lr_metrics)
    save_pickle(lr, MODELS_DIR / "lr_model.pkl")

    # Decision Tree (baseline)
    print("      Decision Tree ...")
    t0 = time.time()
    dt = DecisionTreeClassifier(
        max_depth=20,
        min_samples_split=10,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    dt.fit(X_train, y_train)
    dt_metrics = evaluate("Decision Tree", y_test.values, dt.predict(X_test))
    dt_metrics["train_time_seconds"] = round(time.time() - t0, 2)
    all_metrics.append(dt_metrics)
    save_pickle(dt, MODELS_DIR / "dt_model.pkl")

    # Linear SVM (comparison)
    print("      Linear SVM (LinearSVC) ...")
    t0 = time.time()
    # LinearSVC is much faster than SVC on 118k records and gives a fair
    # comparison point. SVC with RBF kernel would take ~hours on this volume.
    svm = LinearSVC(
        class_weight="balanced",
        max_iter=5000,
        random_state=RANDOM_STATE,
        dual="auto",
    )
    svm.fit(X_train_scaled, y_train)
    svm_metrics = evaluate("SVM (Linear)", y_test.values, svm.predict(X_test_scaled))
    svm_metrics["train_time_seconds"] = round(time.time() - t0, 2)
    all_metrics.append(svm_metrics)
    save_pickle(svm, MODELS_DIR / "svm_model.pkl")

    # Reports
    print("\n[6/6] Writing reports and plots ...")

    # JSON metrics
    summary = {
        "dataset": {
            "total_records": len(df),
            "test_records": len(y_test),
            "test_malicious_count": int(y_test.sum()),
            "test_normal_count": int(len(y_test) - y_test.sum()),
            "class_imbalance_ratio": float(df[TARGET_COL].mean()),
        },
        "models": all_metrics,
        "primary_model": "Random Forest",
        "feature_names": feature_names,
        "feature_importances": rf.feature_importances_.tolist(),
    }
    with open(REPORTS_DIR / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Plots
    rf_cm = np.array(rf_metrics["confusion_matrix"])
    plot_confusion_matrix(rf_cm, REPORTS_DIR / "confusion_matrix.png")
    plot_feature_importance(rf.feature_importances_, feature_names,
                            REPORTS_DIR / "feature_importance.png")
    plot_model_comparison(all_metrics, REPORTS_DIR / "model_comparison.png")

    # Console summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"\nSaved to {MODELS_DIR}:")
    for f in sorted(MODELS_DIR.iterdir()):
        print(f"  {f.name}  ({f.stat().st_size / 1024:.1f} KB)")
    print(f"\nReports saved to {REPORTS_DIR}:")
    for f in sorted(REPORTS_DIR.iterdir()):
        print(f"  {f.name}")

    # Comparative table
    print("\n" + "-" * 86)
    print(f"{'Model':<22} {'Accuracy':>10} {'Precision':>12} {'Recall':>10} {'F1':>10} {'Time(s)':>10}")
    print("-" * 86)
    for m in all_metrics:
        print(f"{m['model']:<22} "
              f"{m['accuracy']:>10.4f} "
              f"{m['precision_malicious']:>12.4f} "
              f"{m['recall_malicious']:>10.4f} "
              f"{m['f1_malicious']:>10.4f} "
              f"{m['train_time_seconds']:>10.2f}")
    print("-" * 86)

    # FP/FN headline for primary model
    tn, fp, fn, tp = rf_cm.ravel()
    print(f"\nRandom Forest Confusion Matrix:")
    print(f"  TN: {tn:,}   FP: {fp:,}")
    print(f"  FN: {fn:,}   TP: {tp:,}")
    print(f"\nTop 5 features by importance:")
    order = np.argsort(rf.feature_importances_)[::-1][:5]
    for i in order:
        print(f"  {feature_names[i]:<35} {rf.feature_importances_[i]:.4f}")


if __name__ == "__main__":
    main()
