"""
FastAPI Model Serving Application.
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.preprocessing import engineer_features

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ml_serving")

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

# Global artifacts
artifacts = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifacts on startup."""
    try:
        logger.info("Loading artifacts...")
        
        # Load Preprocessor
        preprocessor_path = PROCESSED_DATA_DIR / "preprocessor.joblib"
        if not preprocessor_path.exists():
            raise FileNotFoundError(f"Preprocessor not found at {preprocessor_path}")
        artifacts["preprocessor"] = joblib.load(preprocessor_path)
        logger.info("Preprocessor loaded.")

        # Load Model (XGBoost by default, fallback to RandomForest)
        model_name = "XGBoost"
        model_path = MODELS_DIR / f"{model_name}.joblib"
        
        if not model_path.exists():
            logger.warning(f"{model_name} not found, trying RandomForest...")
            model_name = "RandomForest"
            model_path = MODELS_DIR / f"{model_name}.joblib"
        
        if not model_path.exists():
             raise FileNotFoundError(f"No trained models found in {MODELS_DIR}")

        model_data = joblib.load(model_path)
        artifacts["model"] = model_data["model"]
        artifacts["model_info"] = model_data.get("metadata", {})
        logger.info(f"Model {model_name} loaded.")
        
    except Exception as e:
        logger.error(f"Failed to load artifacts: {e}")
        raise e
    
    yield
    
    # Clean up
    artifacts.clear()

app = FastAPI(title="House Price Prediction API", lifespan=lifespan)

class HouseFeatures(BaseModel):
    """
    Flexible input schema for house features.
    In a strict production env, we would define all 80 fields.
    """
    features: Dict[str, Any]

class PredictionResponse(BaseModel):
    prediction: float
    model_info: Dict[str, Any]

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": "model" in artifacts}

@app.post("/predict", response_model=PredictionResponse)
def predict(input_data: HouseFeatures):
    """
    Predict house price from features.
    """
    if "model" not in artifacts or "preprocessor" not in artifacts:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # 1. Convert to DataFrame
        df = pd.DataFrame([input_data.features])
        
        # 2. Feature Engineering
        # Note: dataset_type="house-prices" is default
        df_eng = engineer_features(df)
        
        # 3. Preprocessing (Transform)
        # We need to handle cases where input might miss columns expected by pipelines
        # The preprocessor (ColumnTransformer) usually handles this if columns match
        # But if we are missing numeric columns, SimpleImputer might need them present to impute
        # Let's trust the preprocessor handles it or fails gracefully
        
        X_processed = artifacts["preprocessor"].transform(df_eng)
        
        # 4. Predict
        prediction = artifacts["model"].predict(X_processed)[0]
        
        return {
            "prediction": float(prediction),
            "model_info": {
                "type": type(artifacts["model"]).__name__,
                "score": artifacts["model_info"].get("best_score", "N/A")
            }
        }
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
