"""
MLOps Pipeline Runner.

Runs the full ML portfolio project with MLOps features:
  1. Experiment Tracking (MLflow)
  2. Data Drift Detection (Evidently)
  3. Model Serving Instructions (FastAPI)

Usage:
    uv run python run_mlflow_pipeline.py
"""

import sys
import os
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import mlflow

# Suppress warnings
warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("mlops_pipeline")

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

def main():
    print("=" * 70)
    print("  MLOPS PIPELINE -- Experiment Tracking & Monitoring")
    print("=" * 70)

    # 1. Data Loading
    print("\n[Phase 1] Loading Data...")
    from src.data_load import load_data, split_data
    df = load_data(DATA_RAW)
    print(f"   Loaded: {df.shape[0]} rows")

    # 2. Feature Engineering
    print("\n[Phase 2] Feature Engineering...")
    from src.preprocessing import engineer_features, build_preprocessing_pipeline, get_feature_lists
    df_eng = engineer_features(df.copy())
    
    # Split
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        df_eng, "SalePrice", test_size=0.2, val_size=0.1, random_state=42
    )
    
    # Identify features
    numerical_features, categorical_features = get_feature_lists(
        X_train, target_column="SalePrice", drop_columns=["Id"]
    )
    
    # Build & Fit Preprocessor
    preprocessor = build_preprocessing_pipeline(numerical_features, categorical_features)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Save preprocessor
    joblib.dump(preprocessor, DATA_PROCESSED / "preprocessor.joblib")

    # 3. Training with MLflow
    print("\n[Phase 3] Training with MLflow Tracking...")
    # Set experiment name
    mlflow.set_experiment("housing_prices_prod")
    
    from src.train_supervised import train_all_models, save_model
    
    # Train models (this function now uses MLflow internally)
    # We pass the processed data
    # Note: sparse matrix handling
    if hasattr(X_train_processed, "toarray"):
        X_train_processed = X_train_processed.toarray()
        X_test_processed = X_test_processed.toarray()
        
    results = train_all_models(
        X_train_processed, 
        y_train.values, 
        task_type="regression",
        experiment_name="housing_prices_prod"
    )
    
    # Save best model
    best_score = -float("inf")
    best_model = None
    best_name = ""
    
    for name, (model, info) in results.items():
        if model:
            score = info.get("best_score", info.get("cv_mean_score", -float("inf")))
            if score > best_score:
                best_score = score
                best_model = model
                best_name = name
        # Save locally too
        if model:
            save_model(model, MODELS_DIR / f"{name}.joblib", metadata=info)

    print(f"   [BEST] Best model: {best_name} (Score: {best_score:.4f})")

    # 4. Drift Detection (Monitoring)
    print("\n[Phase 4] Monitoring & Drift Detection...")
    from src.monitoring import monitor_drift
    
    # Calculate drift on RAW features (before encoding) for interpretability
    # subset columns to numerical for simplicity in this demo, or allow evidently to handle all
    # Evidently handles categorical too!
    
    # We compare Train vs Test (proxy for production batch)
    monitor_drift(
        reference_data=X_train,
        current_data=X_test,
        output_path=REPORTS_DIR / "data_drift_report.html"
    )
    print("   [OK] Drift report generated: reports/data_drift_report.html")

    # 5. Serving Instructions
    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE")
    print("=" * 70)
    print("\nTo view MLflow UI:")
    print("    uv run mlflow ui")
    print("\nTo start Model Serving API:")
    print("    uv run uvicorn src.app:app --reload")
    print("\nTo view Drift Report:")
    print(f"    Open {REPORTS_DIR / 'data_drift_report.html'}")
    print()

if __name__ == "__main__":
    main()
