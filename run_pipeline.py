"""
End-to-end ML Pipeline Runner.

Runs the full ML portfolio project:
  1. Load & validate data
  2. Feature engineering & preprocessing
  3. Train supervised models
  4. Evaluate & compare models
  5. Unsupervised analysis (KMeans, PCA, t-SNE)
  6. Save all results

Usage:
    uv run python run_pipeline.py
"""

import sys
import os
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Setup logging (use utf-8 for file handler to avoid Windows encoding issues)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
    ],
)
# Add file handler with utf-8 encoding
file_handler = logging.FileHandler("reports/pipeline_run.log", mode="w", encoding="utf-8")
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logging.getLogger().addHandler(file_handler)

logger = logging.getLogger("ml_portfolio")

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = PROJECT_ROOT / "figures"

# Create directories
for d in [DATA_PROCESSED, MODELS_DIR, REPORTS_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def main():
    """Run the complete ML pipeline."""
    print("=" * 70)
    print("  ML PORTFOLIO PROJECT -- End-to-End Pipeline")
    print("=" * 70)

    # --- PHASE 1: DATA LOADING ---
    print("\n[Phase 1] Loading Data...")
    from src.data_load import load_data, split_data

    df = load_data(DATA_RAW)
    print(f"   Loaded: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"   Target: SalePrice (mean=${df['SalePrice'].mean():,.0f})")

    # --- PHASE 2: FEATURE ENGINEERING ---
    print("\n[Phase 2] Feature Engineering & Preprocessing...")
    from src.preprocessing import (
        engineer_features,
        build_preprocessing_pipeline,
        get_feature_lists,
    )

    df_eng = engineer_features(df.copy(), dataset_type="house-prices")
    print(f"   Engineered features: {df_eng.shape[1]} columns (was {df.shape[1]})")

    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        df_eng, "SalePrice", test_size=0.2, val_size=0.1, random_state=42
    )
    print(f"   Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Identify feature types
    numerical_features, categorical_features = get_feature_lists(
        X_train, target_column="SalePrice", drop_columns=["Id"]
    )
    print(f"   Numerical: {len(numerical_features)}, Categorical: {len(categorical_features)}")

    # Build preprocessing pipeline
    preprocessor = build_preprocessing_pipeline(
        numerical_features, categorical_features
    )

    # Fit & transform
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)

    # Convert sparse to dense if needed
    if hasattr(X_train_processed, "toarray"):
        X_train_processed = X_train_processed.toarray()
        X_val_processed = X_val_processed.toarray()
        X_test_processed = X_test_processed.toarray()

    print(f"   Processed feature matrix: {X_train_processed.shape[1]} features")

    # Save preprocessor
    joblib.dump(preprocessor, DATA_PROCESSED / "preprocessor.joblib")
    print("   [OK] Preprocessor saved")

    # --- PHASE 3: SUPERVISED LEARNING ---
    print("\n[Phase 3] Training Supervised Models...")
    from src.train_supervised import train_all_models, save_model

    all_results = train_all_models(
        X_train_processed,
        y_train.values,
        task_type="regression",
        cv=5,
        random_state=42,
    )

    trained_models = {}
    for name, (model, info) in all_results.items():
        if model is not None:
            trained_models[name] = model
            cv_score = info.get("cv_mean_score", "N/A")
            if isinstance(cv_score, (int, float)):
                print(f"   {name}: CV Score = {cv_score:.4f}")
            else:
                print(f"   {name}: trained")

            # Save model
            save_model(model, MODELS_DIR / f"{name}.joblib", metadata=info)

    print(f"   [OK] {len(trained_models)} models trained and saved")

    # --- PHASE 4: EVALUATION ---
    print("\n[Phase 4] Model Evaluation & Comparison...")
    from src.evaluate import compare_models, generate_evaluation_report, get_best_model

    comparison_df = compare_models(
        trained_models, X_test_processed, y_test.values, task_type="regression"
    )

    print("\n   Model Comparison (Test Set):")
    print("   " + "-" * 60)
    for _, row in comparison_df.iterrows():
        model_name = row.get("model", row.name if hasattr(row, 'name') else "?")
        rmse = row.get("rmse", row.get("RMSE", "N/A"))
        r2 = row.get("r2", row.get("R2", "N/A"))
        if isinstance(rmse, (int, float)) and isinstance(r2, (int, float)):
            print(f"   {model_name:25s} RMSE={rmse:>12,.0f}  R2={r2:.4f}")
        else:
            print(f"   {model_name:25s} RMSE={rmse}  R2={r2}")

    # Generate report
    generate_evaluation_report(comparison_df, REPORTS_DIR / "model_comparison", task_type="regression")
    print("\n   [OK] Evaluation report saved to reports/")

    # Best model
    best_name, best_model = get_best_model(
        trained_models, comparison_df, task_type="regression"
    )
    print(f"   [BEST] Best model: {best_name}")

    # --- PHASE 5: UNSUPERVISED LEARNING ---
    print("\n[Phase 5] Unsupervised Learning...")
    from src.train_unsupervised import (
        perform_kmeans_clustering,
        perform_pca,
        perform_tsne,
    )

    # PCA
    pca_model, X_pca = perform_pca(X_train_processed, variance_threshold=0.95)
    print(f"   PCA: {X_train_processed.shape[1]} features -> {X_pca.shape[1]} components (95% variance)")

    # KMeans
    kmeans_results = perform_kmeans_clustering(
        X_train_processed, k_range=list(range(2, 8)), random_state=42
    )
    best_k = kmeans_results.get("best_k", "?")
    print(f"   KMeans: Best K={best_k}")

    # t-SNE (on a subset for speed)
    sample_size = min(500, len(X_train_processed))
    X_sample = X_train_processed[:sample_size]
    X_tsne = perform_tsne(X_sample, n_components=2, random_state=42)
    print(f"   t-SNE: {sample_size} samples -> 2D embedding")

    # Save unsupervised results
    np.save(DATA_PROCESSED / "pca_components.npy", X_pca)
    np.save(DATA_PROCESSED / "tsne_embedding.npy", X_tsne)
    joblib.dump(kmeans_results, DATA_PROCESSED / "kmeans_results.joblib")
    print("   [OK] Unsupervised results saved")

    # --- PHASE 6: DRIFT REPORT ---
    print("\n[Phase 6] Data Drift Report...")
    try:
        from src.monitoring import monitor_drift
        # Use train as reference, test as current (feature drift)
        ref_df = X_train.drop(columns=["SalePrice"], errors="ignore")
        cur_df = X_test.drop(columns=["SalePrice"], errors="ignore")
        if ref_df.shape[1] > 0 and cur_df.shape[1] > 0:
            drift_path = REPORTS_DIR / "data_drift_report.html"
            monitor_drift(ref_df, cur_df, drift_path)
            print(f"   [OK] Drift report saved to {drift_path}")
        else:
            print("   [SKIP] No features for drift (target-only data)")
    except Exception as e:
        logger.warning(f"Drift report skipped: {e}")
        print(f"   [SKIP] Drift report: {e}")

    # --- SUMMARY ---
    print("\n" + "=" * 70)
    print("  [DONE] PIPELINE COMPLETE")
    print("=" * 70)
    print(f"\n  Models saved to:      {MODELS_DIR}")
    print(f"  Reports saved to:     {REPORTS_DIR}")
    print(f"  Processed data at:    {DATA_PROCESSED}")
    print(f"  Best model:           {best_name}")
    print()

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        print(f"\n[FAILED] Pipeline failed: {e}")
        sys.exit(1)
