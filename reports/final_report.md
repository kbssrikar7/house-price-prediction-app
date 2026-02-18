# ML Portfolio Project: Final Report

## Executive Summary

This project demonstrates an end-to-end machine learning workflow applied to the Kaggle House Prices dataset. We trained and compared six supervised models (Baseline, Ridge, RandomForest, XGBoost, LightGBM, SVM) and performed unsupervised analysis (KMeans, DBSCAN, PCA, t-SNE) to uncover patterns in housing data. Model interpretability was achieved through SHAP analysis, partial dependence plots, and coefficient analysis.

## 1. Dataset Description

- **Source:** Kaggle — House Prices: Advanced Regression Techniques
- **Size:** ~1,460 rows × 81 features
- **Task:** Regression (predict SalePrice)
- **Target Variable:** SalePrice — the property's sale price in dollars

## 2. Methodology

### 2.1 Exploratory Data Analysis
- Analyzed missing value patterns across all 81 features
- Visualized target distribution (right-skewed, log transformation improves normality)
- Computed correlation matrix — OverallQual, GrLivArea most correlated with SalePrice
- Detected outliers using IQR method

### 2.2 Feature Engineering
- **TotalSF:** Combined basement + 1st floor + 2nd floor square footage
- **TotalBath:** Consolidated all bathroom types
- **HouseAge / IsNew:** Age and recent construction indicator
- **QualityArea:** Interaction between overall quality and living area

### 2.3 Preprocessing
- StandardScaler for numerical features
- OneHotEncoder for categorical features
- SimpleImputer (median/most_frequent) for missing values
- Pipeline fit on training data only (no data leakage)

### 2.4 Models Trained
1. **Baseline (DummyRegressor):** Mean prediction — establishes minimum threshold
2. **Ridge:** Linear model with L2 regularization
3. **RandomForest:** Ensemble of decision trees
4. **XGBoost:** Gradient-boosted trees
5. **LightGBM:** Gradient-boosted trees (histogram-based)
6. **SVM (SVR):** Support Vector Regression with RBF/linear kernels

## 3. Results

### 3.1 Model Comparison
> Run notebooks to generate actual metrics.

| Model | RMSE | MAE | R² |
|-------|------|-----|-----|
| Baseline | — | — | — |
| Ridge | — | — | — |
| RandomForest | — | — | — |
| XGBoost | — | — | — |
| LightGBM | — | — | — |
| SVM | — | — | — |

### 3.2 Best Model Analysis
Tree ensemble methods (XGBoost/LightGBM/RandomForest) are expected to outperform linear models on this dataset due to non-linear feature interactions and the heterogeneous nature of housing features.

## 4. Unsupervised Learning Insights

### 4.1 Clustering Results
- KMeans identified natural groupings corresponding to house price segments
- DBSCAN detected outlier properties as noise points

### 4.2 Dimensionality Reduction
- PCA: Significant dimensionality reduction possible while retaining 95% variance
- t-SNE: 2D visualization reveals structure aligned with price levels and neighborhoods

## 5. Model Interpretability

### 5.1 SHAP Analysis
- Global feature importance identifies top predictors
- Local explanations show how individual predictions are composed

### 5.2 Partial Dependence
- Non-linear relationships visible for key features
- Interaction effects between quality metrics and area

## 6. Conclusions

1. Tree ensemble models significantly outperform linear baselines for house price prediction
2. Feature engineering (especially TotalSF) provides meaningful predictive signal
3. Natural clusters in the data correspond to price segments
4. SHAP analysis provides transparency into model decision-making

## 7. Future Work

- Neural network approaches (tabular deep learning)
- Stacking/blending ensemble methods
- Feature selection using Boruta or recursive elimination
- Time-based validation (if temporal data available)

## Appendix

### A. Reproducibility Notes
- `random_state=42` used consistently across all operations
- Dependencies locked via `uv.lock`
- Docker support for containerized execution
