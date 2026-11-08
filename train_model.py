"""
Vetiver CatBoost Classifier Training Script.
Trained on 100% empirical scientific peer-reviewed evidence:
  - 124 empirical success/growth cases (global + regional trials)
  - 60 empirical failure/exclusion boundary cases (global + East Africa cohort)
Features 5-fold stratified cross-validation, SHAP explainability, and model serialization.
"""
import os
import sys

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
import shap

import config


def load_datasets() -> pd.DataFrame:
    """
    Load clean canonical machine-ready empirical dataset.
    Contains exactly 184 cases (124 empirical growth trials + 60 scientific failure cases from Sheet 1).
    """
    if os.path.exists(config.CANONICAL_DATASET_PATH):
        df = pd.read_csv(config.CANONICAL_DATASET_PATH)
    else:
        combined_path = os.path.join(config.DATA_DIR, "vetiver_combined_dataset.csv")
        df = pd.read_csv(combined_path)

    for col in config.CAT_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    for col in config.NUM_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.5)

    return df


def run_training():
    print("=" * 70)
    print("VETIVER CATBOOST CLASSIFIER - EMPIRICAL TRAINING & VALIDATION")
    print("=" * 70)

    # 1. Load empirical dataset
    df_all = load_datasets()
    pos_count = (df_all["Suitability"] == 1).sum()
    neg_count = (df_all["Suitability"] == 0).sum()

    print(f"\n[1/5] Loaded empirical dataset: {len(df_all)} total cases.")
    print(f"      - Empirical Success Cases (Suitability = 1): {pos_count}")
    print(f"      - Scientific Failure Cases (Suitability = 0): {neg_count}")
    print(f"      - Class Ratio: {pos_count / len(df_all) * 100:.1f}% Positive / {neg_count / len(df_all) * 100:.1f}% Negative")

    # 2. Features & Target
    features = config.MODEL_FEATURES
    cat_features = config.CAT_FEATURES
    
    # Ensure numeric columns are proper float type
    for num_col in config.NUM_FEATURES:
        df_all[num_col] = pd.to_numeric(df_all[num_col], errors="coerce").fillna(0.5)

    X = df_all[features]
    y = df_all["Suitability"]

    # 3. Stratified 5-Fold Cross-Validation
    print("\n[2/5] Running Stratified 5-Fold Cross-Validation on Empirical Data (with Monotonic Physics)...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_accuracies = []
    fold_f1s = []
    fold_recalls = []
    fold_precisions = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        fold_model = CatBoostClassifier(
            iterations=150,
            learning_rate=0.07,
            depth=5,
            cat_features=cat_features,
            monotone_constraints=config.MONOTONE_CONSTRAINTS,
            verbose=0,
            random_seed=42 + fold,
        )
        fold_model.fit(X_train, y_train)
        y_pred = fold_model.predict(X_val)

        acc = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred, zero_division=0)
        rec = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)

        fold_accuracies.append(acc)
        fold_f1s.append(f1)
        fold_recalls.append(rec)
        fold_precisions.append(prec)

        print(f"      Fold {fold} -> Accuracy: {acc*100:.1f}% | Precision: {prec*100:.1f}% | Recall: {rec*100:.1f}% | F1: {f1*100:.1f}%")

    print("\n[3/5] Cross-Validation Summary (5 Folds):")
    print(f"      * Mean Accuracy  : {np.mean(fold_accuracies)*100:.2f}% (+/- {np.std(fold_accuracies)*100:.2f}%)")
    print(f"      * Mean Precision : {np.mean(fold_precisions)*100:.2f}% (+/- {np.std(fold_precisions)*100:.2f}%)")
    print(f"      * Mean Recall    : {np.mean(fold_recalls)*100:.2f}% (+/- {np.std(fold_recalls)*100:.2f}%)")
    print(f"      * Mean F1-Score  : {np.mean(fold_f1s)*100:.2f}% (+/- {np.std(fold_f1s)*100:.2f}%)")

    # 4. Train Final Model on All Data
    print("\n[4/5] Training final production model on all 184 empirical cases (Physics-Monotonic)...")
    final_model = CatBoostClassifier(
        iterations=160,
        learning_rate=0.07,
        depth=5,
        cat_features=cat_features,
        monotone_constraints=config.MONOTONE_CONSTRAINTS,
        verbose=0,
        random_seed=42,
    )
    final_model.fit(X, y)

    # 5. SHAP Summary Plot
    pool = Pool(X, cat_features=cat_features)
    explainer = shap.TreeExplainer(final_model)
    shap_values = explainer.shap_values(pool)

    os.makedirs(config.DATA_DIR, exist_ok=True)
    shap_plot_path = os.path.join(config.DATA_DIR, "shap_summary.png")

    plt.figure(figsize=(9, 5))
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.title("CatBoost SHAP Global Feature Importance (Empirical Evidence Base)", fontsize=13, pad=12)
    plt.tight_layout()
    plt.savefig(shap_plot_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"      SHAP summary plot saved to: {shap_plot_path}")

    # 6. Save model
    print(f"\n[5/5] Saving serialized model to: {config.MODEL_PATH}")
    final_model.save_model(config.MODEL_PATH)
    print("      Model saved successfully.")
    print("=" * 70)
    print("Empirical CatBoost training and validation completed successfully!\n")


if __name__ == "__main__":
    run_training()
