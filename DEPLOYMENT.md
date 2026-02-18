# Deploy on Streamlit Community Cloud

## Push to GitHub

1. Create a **new repository** on GitHub (e.g. `ml-portfolio-project`). Do **not** add a README or .gitignore (we already have them).
2. In your project folder, add the remote and push (replace `YOUR_USERNAME` and `YOUR_REPO` with your GitHub username and repo name):

   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
   git branch -M main
   git push -u origin main
   ```

   If you use SSH: `git remote add origin git@github.com:YOUR_USERNAME/YOUR_REPO.git`

## One-time setup (if you haven’t run the pipeline)

1. **Run the pipeline once** (so the app has models and data):
   ```bash
   uv run python run_pipeline.py
   ```
2. **Push this repo to GitHub** (including `data/raw/`, `data/processed/`, `models/`, `reports/` if present).

## Deploy steps

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. Click **"New app"**.
3. Choose your **repository**, **branch** (e.g. `main`), and set:
   - **Main file path:** `src/dashboard.py`
4. Click **Deploy**.  
   Dependencies are read from the repo’s **`requirements.txt`** at the root.

## Settings (optional)

- **Python version:** e.g. 3.10 or 3.11 (Advanced settings).
- **requirements.txt** is at the project root; no need to change paths.

## If the app shows "Run the pipeline first"

The app needs trained models and data. Either:

- Run `uv run python run_pipeline.py` locally, then commit and push the new/updated files under `data/`, `models/`, and `reports/`, or  
- Keep using the app for the **Data** tab and the **About** / **Drift** explanations; **Predict** and **Models** will ask to run the pipeline.
