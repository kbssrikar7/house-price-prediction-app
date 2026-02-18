"""Streamlit Dashboard for ML Portfolio Project - House Price Predictor."""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
import joblib
import sys

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CONFIG

# Page Config
st.set_page_config(
    page_title="House Price Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paths
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RAW_DIR = PROJECT_ROOT / "data" / "raw"

# Dataset source (Kaggle House Prices - Ames Housing)
DATASET_URL = "https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques"
DATASET_NAME = "House Prices - Advanced Regression Techniques (Ames Housing)"


@st.cache_resource
def load_model_artifacts():
    """Load model and preprocessor for predictions."""
    artifacts = {}
    
    preprocessor_path = PROCESSED_DIR / "preprocessor.joblib"
    if preprocessor_path.exists():
        artifacts["preprocessor"] = joblib.load(preprocessor_path)
    
    for model_name in ["XGBoost", "RandomForest", "LightGBM", "Linear"]:
        model_path = MODELS_DIR / f"{model_name}.joblib"
        if model_path.exists():
            model_data = joblib.load(model_path)
            artifacts["model"] = model_data["model"]
            artifacts["model_name"] = model_name
            artifacts["model_info"] = model_data.get("metadata", {})
            break
    
    return artifacts


@st.cache_data
def load_raw_data():
    """Load raw dataset for defaults and exploration."""
    csv_path = RAW_DIR / "AmesHousing.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return None


@st.cache_data
def load_model_comparison():
    """Load model comparison report."""
    for path in [
        REPORTS_DIR / "model_comparison" / "model_comparison.csv",
        REPORTS_DIR / "model_comparison.csv"
    ]:
        if path.exists():
            return pd.read_csv(path, index_col=0)
    return None


# Light-theme CSS override for Evidently report (matches Streamlit light mode)
_DRIFT_REPORT_LIGHT_THEME_CSS = """
<style id="streamlit-drift-light-theme">
body, body *, [id^="root_"], [id^="metric_"], .evidently-ui-iframe,
.MuiPaper-root, .MuiBox-root, .MuiTypography-root, .MuiTable-root,
[class*="MuiPaper"], [class*="MuiBox"], [class*="MuiTypography"] {
  background-color: #FFFFFF !important;
  color: #262730 !important;
  border-color: #F0F2F6 !important;
}
.MuiTableCell-root, th, td { background-color: #FFFFFF !important; color: #262730 !important; }
</style>
"""


@st.cache_data
def load_drift_report():
    """Load drift report HTML and inject light-theme override to match Streamlit."""
    html_path = REPORTS_DIR / "data_drift_report.html"
    if html_path.exists():
        with open(html_path, 'r', encoding='utf-8') as f:
            html = f.read()
        # Inject light theme after <body> so embedded report matches Streamlit
        if "<body>" in html:
            html = html.replace("<body>", "<body>" + _DRIFT_REPORT_LIGHT_THEME_CSS, 1)
        return html
    return None


def engineer_features_for_prediction(df: pd.DataFrame) -> pd.DataFrame:
    """Apply feature engineering for prediction."""
    df = df.copy()
    
    # Total SF
    if all(c in df.columns for c in ['TotalBsmtSF', '1stFlrSF', '2ndFlrSF']):
        df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    
    # Total Bath
    if all(c in df.columns for c in ['FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath']):
        df['TotalBath'] = df['FullBath'] + 0.5*df['HalfBath'] + df['BsmtFullBath'] + 0.5*df['BsmtHalfBath']
    
    # House Age
    if 'YearBuilt' in df.columns:
        df['HouseAge'] = 2026 - df['YearBuilt']
        df['IsNew'] = (df['HouseAge'] <= 5).astype(int)
    
    # Remodel
    if 'YearRemod/Add' in df.columns and 'YearBuilt' in df.columns:
        df['RemodAge'] = 2026 - df['YearRemod/Add']
        df['WasRemodeled'] = (df['YearRemod/Add'] != df['YearBuilt']).astype(int)
    
    # Quality interaction
    if 'OverallQual' in df.columns and 'GrLivArea' in df.columns:
        df['QualityArea'] = df['OverallQual'] * df['GrLivArea']
    
    return df


def predict_price(input_data: dict) -> dict:
    """Make house price prediction."""
    artifacts = load_model_artifacts()
    
    if "model" not in artifacts:
        return {"error": "No model found. Run the training pipeline first."}
    if "preprocessor" not in artifacts:
        return {"error": "No preprocessor found. Run the training pipeline first."}
    
    try:
        df = pd.DataFrame([input_data])
        df = engineer_features_for_prediction(df)
        X = artifacts["preprocessor"].transform(df)
        pred = artifacts["model"].predict(X)[0]
        
        return {
            "prediction": float(pred),
            "model": artifacts.get("model_name", "Unknown")
        }
    except Exception as e:
        return {"error": str(e)}


# Sidebar
with st.sidebar:
    st.title("House Price Predictor")
    st.divider()
    
    artifacts = load_model_artifacts()
    
    if "model" in artifacts:
        st.success(f"Model: {artifacts['model_name']}")
    else:
        st.error("No model loaded")
    
    if "preprocessor" in artifacts:
        st.success("Preprocessor: Ready")
    else:
        st.error("Preprocessor: Missing")
    
    st.divider()
    
    if st.button("Clear Cache"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

    st.divider()
    st.caption("Dataset")
    st.markdown(f"[{DATASET_NAME}]({DATASET_URL})")
    st.caption("Learning: **Supervised** (price prediction) + **Unsupervised** (clustering, PCA, t-SNE)")


# Main
st.title("House Price Prediction")

with st.expander("About this project", expanded=False):
    st.markdown(f"""
    - **Dataset:** This project uses the **Ames Housing** data from Kaggle:  
      [{DATASET_NAME}]({DATASET_URL}).  
      You can download the data and full description from that link.
    - **Learning types:** The project uses **both**:
      - **Supervised learning** — Regression models (e.g. XGBoost, Random Forest, Linear) to **predict sale price** from house features. The **Models** tab compares them using RMSE, R2, MAE, MAPE.
      - **Unsupervised learning** — KMeans clustering, PCA, and t-SNE run in the pipeline for exploration and dimensionality reduction (results are saved; the dashboard focuses on supervised prediction and drift).
    """)

tab1, tab2, tab3, tab4 = st.tabs(["Predict", "Models", "Data", "Drift"])

# TAB 1: Predict
with tab1:
    st.header("Enter House Details")

    with st.expander("What does this do?", expanded=True):
        st.markdown(f"""
        This tab lets you **predict a house sale price** (in dollars) based on the features you enter.
        - The model was trained on the **Ames Housing** dataset ([Kaggle: House Prices]({DATASET_URL})) and uses size, quality, rooms, location, and other attributes.
        - Default values in the form are set to **median values** from the dataset so you can get a quick prediction without filling every field.
        - Click **Predict Price** to see the estimated sale price. Lower prediction error (see **Models** tab) means the model is more accurate on average.
        """)

    if "model" not in artifacts or "preprocessor" not in artifacts:
        st.error("Model not ready. Run the pipeline first:")
        st.code("uv run python run_pipeline.py")
        st.stop()
    
    raw_df = load_raw_data()
    
    # Get median values for defaults
    def get_median(col, default=0):
        if raw_df is not None and col in raw_df.columns:
            val = raw_df[col].median()
            return int(val) if not pd.isna(val) else default
        return default
    
    st.markdown("Fill in the house details below. Values default to dataset medians.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Size")
        gr_liv_area = st.number_input("Living Area (sq ft)", 300, 6000, get_median('GrLivArea', 1500), 100)
        total_bsmt = st.number_input("Basement (sq ft)", 0, 4000, get_median('TotalBsmtSF', 1000), 100)
        first_flr = st.number_input("1st Floor (sq ft)", 300, 4000, get_median('1stFlrSF', 1100), 100)
        second_flr = st.number_input("2nd Floor (sq ft)", 0, 3000, get_median('2ndFlrSF', 0), 100)
        lot_area = st.number_input("Lot Area (sq ft)", 1000, 100000, get_median('LotArea', 10000), 500)
        garage_area = st.number_input("Garage (sq ft)", 0, 1500, get_median('GarageArea', 450), 50)
    
    with col2:
        st.subheader("Quality")
        overall_qual = st.slider("Overall Quality (1-10)", 1, 10, get_median('OverallQual', 6))
        overall_cond = st.slider("Overall Condition (1-10)", 1, 10, get_median('OverallCond', 5))
        year_built = st.number_input("Year Built", 1872, 2026, get_median('YearBuilt', 1973))
        year_remod = st.number_input("Year Remodeled", 1950, 2026, get_median('YearRemod/Add', 1994))
        garage_cars = st.slider("Garage Cars", 0, 5, get_median('GarageCars', 2))
        ms_subclass = st.selectbox("Building Class", [20, 30, 40, 45, 50, 60, 70, 75, 80, 85, 90, 120, 150, 160, 180, 190], index=0, help="Type of dwelling")
    
    with col3:
        st.subheader("Rooms")
        full_bath = st.slider("Full Baths", 0, 4, get_median('FullBath', 2))
        half_bath = st.slider("Half Baths", 0, 3, get_median('HalfBath', 0))
        bedrooms = st.slider("Bedrooms", 0, 8, get_median('BedroomAbvGr', 3))
        tot_rooms = st.slider("Total Rooms", 2, 15, get_median('TotRmsAbvGrd', 6))
        fireplaces = st.slider("Fireplaces", 0, 4, get_median('Fireplaces', 1))
    
    with st.expander("Additional Options"):
        col_a, col_b = st.columns(2)
        with col_a:
            neighborhood = st.selectbox("Neighborhood", 
                ['NAmes', 'CollgCr', 'OldTown', 'Edwards', 'Somerst', 'Gilbert', 
                 'NridgHt', 'Sawyer', 'NWAmes', 'SawyerW', 'BrkSide', 'Crawfor',
                 'Mitchel', 'NoRidge', 'Timber', 'IDOTRR', 'ClearCr', 'StoneBr',
                 'SWISU', 'MeadowV', 'Blmngtn', 'BrDale', 'Veenker', 'NPkVill',
                 'Blueste', 'Greens', 'GreenHills', 'Landmrk'])
            bldg_type = st.selectbox("Building Type", ['1Fam', '2fmCon', 'Duplex', 'TwnhsE', 'Twnhs'])
            house_style = st.selectbox("Style", ['1Story', '2Story', '1.5Fin', 'SLvl', 'SFoyer'])
        with col_b:
            central_air = st.selectbox("Central Air", ['Y', 'N'])
            kitchen_qual = st.selectbox("Kitchen Quality", ['Ex', 'Gd', 'TA', 'Fa'], index=2)
            garage_type = st.selectbox("Garage Type", ['Attchd', 'Detchd', 'BuiltIn', 'NA'])
    
    st.divider()
    
    if st.button("Predict Price", type="primary", use_container_width=True):
        input_data = {
            'GrLivArea': gr_liv_area,
            'TotalBsmtSF': total_bsmt,
            '1stFlrSF': first_flr,
            '2ndFlrSF': second_flr,
            'LotArea': lot_area,
            'GarageArea': garage_area,
            'OverallQual': overall_qual,
            'OverallCond': overall_cond,
            'YearBuilt': year_built,
            'YearRemod/Add': year_remod,
            'GarageCars': garage_cars,
            'MSSubClass': ms_subclass,
            'FullBath': full_bath,
            'HalfBath': half_bath,
            'BedroomAbvGr': bedrooms,
            'TotRmsAbvGrd': tot_rooms,
            'Fireplaces': fireplaces,
            'Neighborhood': neighborhood,
            'BldgType': bldg_type,
            'HouseStyle': house_style,
            'CentralAir': central_air,
            'KitchenQual': kitchen_qual,
            'GarageType': garage_type,
            # Defaults for remaining features
            'BsmtFullBath': 0,
            'BsmtHalfBath': 0,
            'LotFrontage': 70.0,
            'MasVnrArea': 0.0,
            'BsmtFinSF1': 0.0,
            'BsmtFinSF2': 0.0,
            'BsmtUnfSF': float(total_bsmt),
            'LowQualFinSF': 0,
            'WoodDeckSF': 0,
            'OpenPorchSF': 0,
            'EnclosedPorch': 0,
            '3SsnPorch': 0,
            'ScreenPorch': 0,
            'PoolArea': 0,
            'MiscVal': 0,
            'MoSold': 6,
            'YrSold': 2026,
            'GarageYrBlt': float(year_built),
            'KitchenAbvGr': 1,
            'MSZoning': 'RL',
            'Street': 'Pave',
            'LotShape': 'Reg',
            'LandContour': 'Lvl',
            'Utilities': 'AllPub',
            'LotConfig': 'Inside',
            'LandSlope': 'Gtl',
            'Condition1': 'Norm',
            'Condition2': 'Norm',
            'RoofStyle': 'Gable',
            'RoofMatl': 'CompShg',
            'Exterior1st': 'VinylSd',
            'Exterior2nd': 'VinylSd',
            'MasVnrType': 'None',
            'ExterQual': 'TA',
            'ExterCond': 'TA',
            'Foundation': 'PConc',
            'BsmtQual': 'TA',
            'BsmtCond': 'TA',
            'BsmtExposure': 'No',
            'BsmtFinType1': 'Unf',
            'BsmtFinType2': 'Unf',
            'Heating': 'GasA',
            'HeatingQC': 'Ex',
            'Electrical': 'SBrkr',
            'Functional': 'Typ',
            'FireplaceQu': 'TA' if fireplaces > 0 else 'NA',
            'GarageFinish': 'Unf',
            'GarageQual': 'TA',
            'GarageCond': 'TA',
            'PavedDrive': 'Y',
            'PoolQC': 'NA',
            'Fence': 'NA',
            'MiscFeature': 'NA',
            'SaleType': 'WD',
            'SaleCondition': 'Normal',
            'Alley': 'NA'
        }
        
        with st.spinner("Calculating..."):
            result = predict_price(input_data)
        
        if "error" in result:
            st.error(f"Error: {result['error']}")
        else:
            st.success("Prediction Complete")
            col_r1, col_r2 = st.columns([2, 1])
            with col_r1:
                st.metric("Predicted Price", f"${result['prediction']:,.0f}")
            with col_r2:
                st.caption(f"Model: {result['model']}")
                st.caption(f"Area: {gr_liv_area} sq ft")
                st.caption(f"Quality: {overall_qual}/10")


# TAB 2: Models
with tab2:
    st.header("Model Comparison")

    with st.expander("What do these metrics mean?", expanded=True):
        st.markdown("""
        These metrics measure how well each model predicts **sale price** on held-out test data. For regression (predicting a number):

        | Metric | What it measures | Better when |
        |--------|------------------|-------------|
        | **RMSE** (Root Mean Squared Error) | Typical prediction error in **dollars** (same units as price). | **Lower** is better (less error). |
        | **MAE** (Mean Absolute Error) | Average absolute error in **dollars**. | **Lower** is better. |
        | **R2** (R-squared) | Fraction of variation in prices explained by the model (0 to 1). | **Higher** is better (1 = perfect fit). |
        | **MAPE** (Mean Absolute % Error) | Average error as a **percentage** of the actual price. | **Lower** is better (e.g. 8% is better than 15%). |

        In short: **lower RMSE, MAE, and MAPE** mean more accurate predictions; **higher R2** means the model explains more of the variation in prices. The "Best Model" is the one with the lowest RMSE.
        """)

    df_metrics = load_model_comparison()

    if df_metrics is not None:
        st.dataframe(df_metrics, width='stretch')
        
        metric = st.selectbox("Metric", df_metrics.columns.tolist())
        
        fig = px.bar(df_metrics, y=metric, title=f"Comparison: {metric}", color=metric)
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, width='stretch')
        
        if 'RMSE' in df_metrics.columns:
            best = df_metrics['RMSE'].idxmin()
            st.success(f"Best Model: {best} (RMSE: {df_metrics.loc[best, 'RMSE']:.2f})")
    else:
        st.warning("No comparison data. Run the pipeline first.")


# TAB 3: Data
with tab3:
    st.header("Dataset Explorer")

    with st.expander("What am I looking at?", expanded=True):
        st.markdown(f"""
        This tab explores the **Ames Housing** dataset used to train the models.  
        **Dataset source:** [House Prices - Advanced Regression Techniques (Kaggle)]({DATASET_URL})

        - **Records** — Number of houses (rows) in the dataset.
        - **Features** — Number of attributes (columns) per house (e.g. size, quality, neighborhood).
        - **Sample Data** — First few rows so you can see the raw values.
        - **Price Distribution** — How sale prices are spread (e.g. most houses in a certain range).
        - **Top Correlations** — Features that are most related to **SalePrice**; higher correlation (closer to 1) means that feature tends to go up when price goes up. This helps explain what drives price in the data.
        """)

    raw_df = load_raw_data()

    if raw_df is not None:
        c1, c2, c3 = st.columns(3)
        c1.metric("Records", f"{len(raw_df):,}")
        c2.metric("Features", len(raw_df.columns))
        if 'SalePrice' in raw_df.columns:
            c3.metric("Avg Price", f"${raw_df['SalePrice'].mean():,.0f}")
        
        st.subheader("Sample Data")
        st.dataframe(raw_df.head(10), width='stretch')
        
        if 'SalePrice' in raw_df.columns:
            st.subheader("Price Distribution")
            fig = px.histogram(raw_df, x='SalePrice', nbins=50)
            st.plotly_chart(fig, width='stretch')
            
            st.subheader("Top Correlations")
            num_cols = raw_df.select_dtypes(include=[np.number]).columns
            corr = raw_df[num_cols].corr()['SalePrice'].drop('SalePrice').sort_values(ascending=False).head(10)
            fig = px.bar(x=corr.values, y=corr.index, orientation='h', title="Top 10 Correlations")
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, width='stretch')
    else:
        st.warning("No data found.")


# TAB 4: Drift
with tab4:
    st.header("Data Drift Monitoring")

    with st.expander("What is data drift?", expanded=True):
        st.markdown("""
        **Data drift** means the distribution of your data has changed between two datasets:
        - **Reference** = training data (baseline)
        - **Current** = e.g. test data or new production data

        The report below compares these two. You may see:
        - **"Dataset Drift is NOT detected"** — The *overall* share of drifted columns is below the threshold (default 0.5).
          So as a whole, the dataset is considered stable for the model.
        - **"Drift is detected for X% of columns"** — Some *individual* columns show a statistically significant
          distribution change (e.g. 10 out of 84). That is normal; the **dataset** is still "no drift" when the
          share of such columns stays under the threshold.
        """)

    report = load_drift_report()

    if report:
        import streamlit.components.v1 as components
        st.caption("Report: train vs test data. Scroll within the frame to see all columns.")
        components.html(report, height=800, scrolling=True)
    else:
        st.info(
            "No drift report found. Run the pipeline first to generate it: "
            "`uv run python run_pipeline.py`"
        )
