"""Shared helpers for the Telco Customer Churn project."""
from pathlib import Path
import warnings
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
RAW_CSV = ROOT / "data" / "Telco_Customer_Churn_Dataset.csv"
PROCESSED = ROOT / "data"
FIG = ROOT / "outputs" / "figures"
TAB = ROOT / "outputs" / "tables"
MODELS = ROOT / "outputs" / "models"
for p in (FIG, TAB, MODELS):
    p.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
sns.set_theme(style="whitegrid", palette="deep")
PALETTE = {"No": "#4C78A8", "Yes": "#E4572E"}


def load_clean() -> pd.DataFrame:
    """Raw CSV with TotalCharges fixed (blank strings -> 0 for tenure-0 customers)."""
    df = pd.read_csv(RAW_CSV)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    # The only blanks belong to customers with tenure == 0 (no bill issued yet)
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)
    return df


def save_fig(name: str):
    plt.tight_layout()
    plt.savefig(FIG / name, dpi=150, bbox_inches="tight")
    plt.close()
