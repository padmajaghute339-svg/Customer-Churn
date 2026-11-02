"""
Task 1: Data Preparation
 - load + explore, handle missing values, encode categoricals, train/test split.
Outputs: data/train.csv, data/test.csv (encoded, include customerID + Churn)
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from common import *

# ---------------------------------------------------------------- 1. Load
raw = pd.read_csv(RAW_CSV)
print("Shape:", raw.shape)
print("\nColumn types:\n", raw.dtypes)
print("\nFirst rows:\n", raw.head())
print("\nDuplicate customerIDs:", raw["customerID"].duplicated().sum())

# ---------------------------------------------------------------- 2. Explore
print("\nNumeric summary:\n", raw.describe().T)
print("\nChurn distribution:\n", raw["Churn"].value_counts())
print("\nNull values per column (pandas view):\n", raw.isna().sum()[lambda s: s > 0])

# ---------------------------------------------------------------- 3. Missing values
# TotalCharges is read as text because 11 rows contain a blank string " ".
blank = raw["TotalCharges"].str.strip() == ""
print(f"\nBlank TotalCharges rows: {blank.sum()}  | tenure of those rows: {raw.loc[blank, 'tenure'].unique()}")
df = load_clean()   # to numeric; blanks -> 0 (these customers have tenure 0, nothing billed yet)
assert df.isna().sum().sum() == 0
print("Missing values after cleaning:", int(df.isna().sum().sum()))

# ---------------------------------------------------------------- 4. Encoding
ids = df["customerID"]
data = df.drop(columns="customerID").copy()

# 'No internet service' / 'No phone service' carry the same information as 'No'
# (the dependency is already captured by InternetService / PhoneService).
addons = ["MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
          "TechSupport", "StreamingTV", "StreamingMovies"]
for c in addons:
    data[c] = data[c].replace({"No internet service": "No", "No phone service": "No"})

# Binary columns -> 0/1
binary_map = {"Yes": 1, "No": 0}
for c in ["Partner", "Dependents", "PhoneService", "PaperlessBilling", "Churn"] + addons:
    data[c] = data[c].map(binary_map)
data["gender"] = data["gender"].map({"Male": 1, "Female": 0})

# Multi-class columns -> one-hot (drop_first avoids the dummy-variable trap)
data = pd.get_dummies(data, columns=["InternetService", "Contract", "PaymentMethod"],
                      drop_first=True, dtype=int)
print("\nEncoded shape:", data.shape)
print("Encoded columns:", list(data.columns))

# ---------------------------------------------------------------- 5. Split
X = data.drop(columns="Churn")
y = data["Churn"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)  # stratify keeps ~26.5% churn in both

train = X_train.assign(Churn=y_train); train.insert(0, "customerID", ids.loc[train.index])
test = X_test.assign(Churn=y_test);   test.insert(0, "customerID", ids.loc[test.index])
train.to_csv(PROCESSED / "train.csv", index=False)
test.to_csv(PROCESSED / "test.csv", index=False)

print(f"\nTrain: {train.shape}  churn rate {y_train.mean():.3%}")
print(f"Test : {test.shape}  churn rate {y_test.mean():.3%}")
print("\nSaved data/train.csv and data/test.csv")
print("Note: scaling is done inside the Task 4 model pipelines (fit on train only) to avoid leakage.")
