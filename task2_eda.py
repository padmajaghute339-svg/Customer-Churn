"""
Task 2: Exploratory Data Analysis
 - overall churn rate, demographics, tenure vs churn, contract & payment method vs churn.
"""
import numpy as np
import pandas as pd
from scipy import stats
from common import *

df = load_clean()
df["ChurnFlag"] = (df["Churn"] == "Yes").astype(int)
df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
results = []   # statistical tests

# ------------------------------------------------------------ 1. Overall churn rate
rate = df["ChurnFlag"].mean()
n_churn = int(df["ChurnFlag"].sum())
print(f"Overall churn rate: {rate:.2%}  ({n_churn} of {len(df)} customers)")

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
counts = df["Churn"].value_counts().reindex(["No", "Yes"])
ax[0].pie(counts, labels=["Retained", "Churned"], autopct="%1.1f%%", startangle=90,
          colors=[PALETTE["No"], PALETTE["Yes"]], wedgeprops={"edgecolor": "white"})
ax[0].set_title("Overall Churn Rate")
sns.barplot(x=["Retained", "Churned"], y=counts.values, palette=[PALETTE["No"], PALETTE["Yes"]], ax=ax[1])
for i, v in enumerate(counts.values):
    ax[1].text(i, v + 40, f"{v:,}", ha="center")
ax[1].set_ylabel("Customers"); ax[1].set_title("Customer Counts")
save_fig("01_overall_churn_rate.png")


def churn_by(col, title, fname, order=None, rotate=0):
    """Customer distribution + churn rate for a categorical column; chi-square test."""
    g = df.groupby(col).agg(customers=("ChurnFlag", "size"), churned=("ChurnFlag", "sum"),
                            churn_rate=("ChurnFlag", "mean")).reset_index()
    if order: g = g.set_index(col).loc[order].reset_index()
    chi2, p, _, _ = stats.chi2_contingency(pd.crosstab(df[col], df["ChurnFlag"]))
    results.append({"variable": col, "test": "chi-square", "statistic": round(chi2, 2), "p_value": p,
                    "significant_at_0.05": p < 0.05})
    g.to_csv(TAB / f"churn_by_{col}.csv", index=False)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    sns.barplot(data=g, x=col, y="customers", color="#4C78A8", ax=ax[0])
    ax[0].set_title(f"Customers by {title}"); ax[0].tick_params(axis="x", rotation=rotate)
    sns.barplot(data=g, x=col, y="churn_rate", color="#E4572E", ax=ax[1])
    ax[1].axhline(rate, ls="--", c="grey", label=f"Overall {rate:.1%}")
    for i, v in enumerate(g["churn_rate"]):
        ax[1].text(i, v + 0.008, f"{v:.1%}", ha="center")
    ax[1].set_title(f"Churn Rate by {title}  (p={p:.3g})"); ax[1].legend()
    ax[1].tick_params(axis="x", rotation=rotate); ax[1].set_ylim(0, g["churn_rate"].max() * 1.2)
    save_fig(fname)
    return g

# ------------------------------------------------------------ 2. Demographics
print("\n--- Demographics ---")
for col, title, f in [("gender", "Gender", "02_gender.png"),
                      ("Partner", "Partner Status", "03_partner.png"),
                      ("Dependents", "Dependent Status", "04_dependents.png"),
                      ("SeniorCitizen", "Senior Citizen", "05_senior.png")]:
    print(churn_by(col, title, f).round(3).to_string(index=False), "\n")

# ------------------------------------------------------------ 3. Tenure
print("--- Tenure ---")
t_churn, t_stay = df.loc[df.ChurnFlag == 1, "tenure"], df.loc[df.ChurnFlag == 0, "tenure"]
print(f"Median tenure: churned={t_churn.median():.0f} months | retained={t_stay.median():.0f} months")
u, p = stats.mannwhitneyu(t_churn, t_stay)
results.append({"variable": "tenure", "test": "Mann-Whitney U", "statistic": round(u, 1), "p_value": p,
                "significant_at_0.05": p < 0.05})

fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
sns.histplot(data=df, x="tenure", hue="Churn", bins=36, multiple="stack", palette=PALETTE, ax=ax[0])
ax[0].set_title("Tenure Distribution")
sns.boxplot(data=df, x="Churn", y="tenure", palette=PALETTE, ax=ax[1]); ax[1].set_title("Tenure by Churn")
df["TenureBand"] = pd.cut(df["tenure"], [-1, 6, 12, 24, 36, 48, 60, 72],
                          labels=["0-6", "7-12", "13-24", "25-36", "37-48", "49-60", "61-72"])
tb = df.groupby("TenureBand", observed=True)["ChurnFlag"].mean()
sns.lineplot(x=tb.index.astype(str), y=tb.values, marker="o", color="#E4572E", ax=ax[2])
for i, v in enumerate(tb.values): ax[2].text(i, v + 0.015, f"{v:.0%}", ha="center")
ax[2].set_title("Churn Rate by Tenure (months)"); ax[2].set_ylim(0, tb.max() * 1.2)
save_fig("06_tenure_vs_churn.png")
tb.rename("churn_rate").to_csv(TAB / "churn_by_tenure_band.csv")
print(tb.round(3).to_string(), "\n")

# ------------------------------------------------------------ 4. Contract & payment method
print("--- Contract & payment ---")
print(churn_by("Contract", "Contract Type", "07_contract.png",
               order=["Month-to-month", "One year", "Two year"]).round(3).to_string(index=False), "\n")
print(churn_by("PaymentMethod", "Payment Method", "08_payment_method.png", rotate=15).round(3)
      .to_string(index=False), "\n")

# Contract x payment method heat-map
ct = df.pivot_table(index="Contract", columns="PaymentMethod", values="ChurnFlag", aggfunc="mean")
plt.figure(figsize=(9, 3.6))
sns.heatmap(ct, annot=True, fmt=".1%", cmap="Reds", cbar_kws={"label": "Churn rate"})
plt.title("Churn Rate: Contract x Payment Method"); plt.ylabel(""); plt.xlabel("")
plt.xticks(rotation=15)
save_fig("09_contract_x_payment_heatmap.png")

# Monthly charges (supporting analysis)
u, p = stats.mannwhitneyu(df.loc[df.ChurnFlag == 1, "MonthlyCharges"], df.loc[df.ChurnFlag == 0, "MonthlyCharges"])
results.append({"variable": "MonthlyCharges", "test": "Mann-Whitney U", "statistic": round(u, 1), "p_value": p,
                "significant_at_0.05": p < 0.05})
churn_by("InternetService", "Internet Service", "10_internet_service.png")

pd.DataFrame(results).to_csv(TAB / "eda_statistical_tests.csv", index=False)
print("Statistical tests:\n", pd.DataFrame(results).to_string(index=False))
