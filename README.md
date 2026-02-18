# ML Portfolio Project 🚀

## 🎯 Project Overview

An end-to-end Machine Learning portfolio project demonstrating supervised and unsupervised learning on the **Kaggle House Prices** dataset. This project showcases production-quality ML practices including:

- **Supervised Learning:** Baseline → Ridge → RandomForest → XGBoost → LightGBM → SVM
- **Unsupervised Learning:** KMeans (with elbow/silhouette), DBSCAN, PCA, t-SNE
- **Model Interpretability:** SHAP (global + local), Partial Dependence, Coefficient Analysis
- **Reproducibility:** `random_state=42` everywhere, locked dependencies via UV

---

## 📁 Project Structure

```
ml-portfolio-project/
├── pyproject.toml            # UV project config & dependencies
├── README.md                 # This file
├── Dockerfile                # Container for reproducible execution
├── .gitignore
├── data/
│   ├── raw/                  # Original Kaggle data
│   └── processed/            # Preprocessed arrays & pipeline
├── notebooks/
│   ├── 01_EDA_and_FeatureEngineering.ipynb
│   └── 02_Modeling_and_Evaluation.ipynb
├── src/
│   ├── __init__.py
│   ├── utils.py              # Logging, figure saving
│   ├── data_load.py          # Kaggle download, load, split
│   ├── preprocessing.py      # Feature engineering, pipelines
│   ├── train_supervised.py   # Model training with tuning
│   ├── train_unsupervised.py # Clustering & dimensionality reduction
│   └── evaluate.py           # Metrics, comparison, reports
├── models/                   # Saved model artifacts
├── reports/                  # Generated reports
├── figures/                  # Generated visualizations
└── tests/
    ├── __init__.py
    └── test_data_load.py     # Unit tests
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- [UV Package Manager](https://github.com/astral-sh/uv)
- Kaggle API credentials (optional — can place CSV manually)

### Setup

```bash
# Clone repository
git clone <repository-url>
cd ml-portfolio-project

# Install UV (if not installed)
pip install uv

# Create environment and install dependencies
uv sync

# Set up Kaggle credentials (optional)
# Place kaggle.json in ~/.kaggle/

# Download dataset (or manually place train.csv in data/raw/)
uv run python -c "from src.data_load import download_dataset; download_dataset()"

# Launch notebooks
uv run jupyter lab
```

### Running Notebooks
1. Open `notebooks/01_EDA_and_FeatureEngineering.ipynb` — Run all cells for EDA & preprocessing
2. Open `notebooks/02_Modeling_and_Evaluation.ipynb` — Run all cells for training & evaluation

---

## 📊 Results Summary

| Model | RMSE | MAE | R² |
|-------|------|-----|-----|
| Baseline | — | — | — |
| Ridge | — | — | — |
| RandomForest | — | — | — |
| XGBoost | — | — | — |
| LightGBM | — | — | — |
| SVM | — | — | — |

> **Note:** Run the notebooks to populate these results.

---

## 📈 Key Techniques

| Category | Methods |
|----------|---------|
| Feature Engineering | TotalSF, TotalBath, HouseAge, QualityArea |
| Preprocessing | StandardScaler, OneHotEncoder, ColumnTransformer |
| Supervised | DummyRegressor, Ridge, RandomForest, XGBoost, LightGBM, SVR |
| Unsupervised | KMeans, DBSCAN, PCA, t-SNE |
| Interpretability | SHAP (TreeExplainer), Partial Dependence, Coefficient Analysis |
| Tuning | GridSearchCV, RandomizedSearchCV (cv=5) |

---

## 🐳 Docker

```bash
# Build image
docker build -t ml-portfolio .

# Run training
docker run -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models ml-portfolio

# Run with Jupyter
docker run -p 8888:8888 -v $(pwd):/app ml-portfolio \
    uv run jupyter lab --ip=0.0.0.0 --allow-root --no-browser
```

---

## 🧪 Testing

```bash
# Run all tests
uv run pytest tests/ -v

# Run with coverage
uv run pytest --cov=src --cov-report=html
```

---

## 📝 Reports

- [Model Comparison](reports/model_comparison.md) — Side-by-side metrics for all models
- [Final Report](reports/final_report.md) — Detailed methodology and findings

---

## 📄 License

MIT License
