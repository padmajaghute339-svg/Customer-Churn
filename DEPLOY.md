# Deploying the Churn Predictor (Streamlit Community Cloud - free)

1. Test locally:  `pip install -r requirements.txt`  then  `streamlit run app.py`
2. Create a GitHub account and a new **public** repository (e.g. `churn-predictor`).
3. Upload the whole `churn_project` folder contents to the repo (keep `app.py`, `requirements.txt`,
   `data/`, `outputs/models/` at the repo root level).
4. Go to https://share.streamlit.io -> sign in with GitHub -> **Create app**.
5. Pick the repo, branch `main`, main file `app.py`. Under *Advanced settings* choose Python 3.11 or 3.12.
6. Click **Deploy**. You get a public URL like `https://your-app.streamlit.app`.
