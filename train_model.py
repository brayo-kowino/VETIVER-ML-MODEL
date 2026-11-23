"""
Vetiver CatBoost Classifier Training Script.
Trained on 100% empirical scientific peer-reviewed evidence:
  - 124 empirical success/growth cases (global + regional trials)
  - 60 empirical failure/exclusion boundary cases (global + East Africa cohort)
Features 5-fold cross-validation grouped by source reference, SHAP explainability, and model serialization.
"""
import argparse
import os
import sys

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    brier_score_loss,
)

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


def validate_by_reference(df: pd.DataFrame) -> None:
    """Hold out whole references to evaluate transfer to unseen studies."""
    if "raw_reference" not in df:
        raise ValueError("Grouped validation requires the canonical dataset with raw_reference.")

    groups = (
        df["raw_reference"].fillna("").str.casefold().str.replace(r"\s+", " ", regex=True).str.strip()
        .replace("", "unknown reference")
    )
    X = df[config.MODEL_FEATURES]
    y = df["Suitability"]
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    predictions = np.empty(len(df), dtype=int)
    scores = np.empty(len(df))
    print(f"Five-fold validation on {len(df)} cases grouped by {groups.nunique()} source references.")
    for fold, (train_idx, val_idx) in enumerate(splitter.split(X, y, groups), 1):
        if y.iloc[train_idx].nunique() < 2:
            raise ValueError(f"Training fold {fold} needs both suitability classes.")
        model = CatBoostClassifier(
            iterations=160,
            learning_rate=0.07,
            depth=5,
            cat_features=config.CAT_FEATURES,
            monotone_constraints=config.MONOTONE_CONSTRAINTS,
            verbose=0,
            random_seed=42,
            thread_count=4,
            allow_writing_files=False,
        )
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        predictions[val_idx] = model.predict(X.iloc[val_idx]).astype(int)
        scores[val_idx] = model.predict_proba(X.iloc[val_idx])[:, 1]
        accuracy = accuracy_score(y.iloc[val_idx], predictions[val_idx])
        print(f"Fold {fold} · {len(val_idx)} cases, {groups.iloc[val_idx].nunique()} held-out references · {accuracy:.1%} accuracy")

    print(f"Held-out accuracy          {accuracy_score(y, predictions):.2%}")
    print(f"Held-out balanced accuracy {balanced_accuracy_score(y, predictions):.2%}")
    print(f"Held-out precision         {precision_score(y, predictions, zero_division=0):.2%}")
    print(f"Held-out recall (suitable)   {recall_score(y, predictions, zero_division=0):.2%}")
    print(f"Held-out recall (unsuitable) {recall_score(y, predictions, pos_label=0, zero_division=0):.2%}")
    print(f"Held-out F1                {f1_score(y, predictions, zero_division=0):.2%}")
    print(f"Held-out Brier score       {brier_score_loss(y, scores):.4f}")


def run_training(*, validate_only: bool = False):
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

    validate_by_reference(df_all)
    if validate_only:
        return

    import matplotlib.pyplot as plt
    import shap

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
    parser = argparse.ArgumentParser(description="Validate and train the Vetiver CatBoost model.")
    parser.add_argument("--validate-only", action="store_true", help="Run grouped validation without writing model files.")
    run_training(validate_only=parser.parse_args().validate_only)
