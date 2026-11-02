"""
Task 3: Customer Segmentation
 - rule-based segments (tenure x monthly charges x contract) + K-Means clustering
 - churn rate within segments, high-value customers at risk
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from common import *

df = load_clean()
df["ChurnFlag"] = (df["Churn"] == "Yes").astype(int)
overall = df["ChurnFlag"].mean()

# ------------------------------------------------------------ 1. Rule-based segments
df["TenureSegment"] = pd.cut(df["tenure"], [-1, 12, 24, 48, 72],
                             labels=["New (0-12m)", "Developing (13-24m)", "Established (25-48m)", "Loyal (49-72m)"])
q = df["MonthlyCharges"].quantile([1 / 3, 2 / 3]).values
df["ChargeSegment"] = pd.cut(df["MonthlyCharges"], [-np.inf, q[0], q[1], np.inf],
                             labels=[f"Low (<${q[0]:.0f})", f"Medium (${q[0]:.0f}-{q[1]:.0f})", f"High (>${q[1]:.0f})"])
print(f"Monthly-charge tercile cut-offs: ${q[0]:.2f} / ${q[1]:.2f}")

def seg_table(cols):
    t = (df.groupby(cols, observed=True)
           .agg(customers=("ChurnFlag", "size"), churned=("ChurnFlag", "sum"),
                churn_rate=("ChurnFlag", "mean"), avg_monthly=("MonthlyCharges", "mean"))
           .reset_index())
    t["vs_overall"] = t["churn_rate"] / overall
    return t

for cols, name in [(["TenureSegment"], "tenure"), (["ChargeSegment"], "charges"), (["Contract"], "contract")]:
    t = seg_table(cols); t.to_csv(TAB / f"segment_{name}.csv", index=False)
    print(f"\nChurn by {name} segment:\n", t.round(3).to_string(index=False))

combo = seg_table(["TenureSegment", "ChargeSegment", "Contract"])
combo = combo.sort_values("churn_rate", ascending=False)
combo.to_csv(TAB / "segment_tenure_charges_contract.csv", index=False)
print("\nHighest-churn segments (n >= 50):\n", combo[combo.customers >= 50].head(8).round(3).to_string(index=False))

# Heat-maps
fig, ax = plt.subplots(1, 2, figsize=(15, 4.8))
for a, ccol, ttl in [(ax[0], "ChargeSegment", "Tenure x Monthly Charges"), (ax[1], "Contract", "Tenure x Contract")]:
    p = df.pivot_table(index="TenureSegment", columns=ccol, values="ChurnFlag", aggfunc="mean", observed=True)
    sns.heatmap(p, annot=True, fmt=".0%", cmap="Reds", vmin=0, vmax=0.7, ax=a, cbar_kws={"label": "Churn rate"})
    a.set_title(f"Churn Rate: {ttl}"); a.set_xlabel(""); a.set_ylabel("")
save_fig("11_segment_heatmaps.png")

fig, ax = plt.subplots(figsize=(11, 5))
pm = df.pivot_table(index="Contract", columns="ChargeSegment", values="ChurnFlag", aggfunc="mean", observed=True)
pm.plot(kind="bar", ax=ax, colormap="OrRd", edgecolor="black"); ax.axhline(overall, ls="--", c="grey")
ax.set_ylabel("Churn rate"); ax.set_title("Churn Rate by Contract and Monthly-Charge Tier"); plt.xticks(rotation=0)
save_fig("12_contract_charge_churn.png")

# ------------------------------------------------------------ 2. K-Means clustering
feat = pd.DataFrame({"tenure": df["tenure"], "MonthlyCharges": df["MonthlyCharges"],
                     "ContractLength": df["Contract"].map({"Month-to-month": 0, "One year": 1, "Two year": 2})})
Xs = StandardScaler().fit_transform(feat)
sil = {k: silhouette_score(Xs, KMeans(k, n_init=10, random_state=RANDOM_STATE).fit_predict(Xs),
                           sample_size=4000, random_state=RANDOM_STATE) for k in range(2, 7)}
print("\nSilhouette scores:", {k: round(v, 3) for k, v in sil.items()})
K = 4   # chosen for interpretability (see silhouette scores above)
df["Cluster"] = KMeans(K, n_init=10, random_state=RANDOM_STATE).fit_predict(Xs)
prof = (df.groupby("Cluster").agg(customers=("ChurnFlag", "size"), avg_tenure=("tenure", "mean"),
                                  avg_monthly=("MonthlyCharges", "mean"),
                                  pct_month_to_month=("Contract", lambda s: (s == "Month-to-month").mean()),
                                  churn_rate=("ChurnFlag", "mean")).round(3))
# Readable names derived from each cluster's profile
def name_cluster(r):
    t = "New" if r.avg_tenure < 20 else ("Loyal" if r.avg_tenure > 45 else "Mid-tenure")
    c = "premium" if r.avg_monthly > 80 else ("budget" if r.avg_monthly < 45 else "mid-price")
    k = "month-to-month" if r.pct_month_to_month > 0.6 else "contracted"
    return f"{t}, {c}, {k}"
prof["name"] = prof.apply(name_cluster, axis=1)
prof.to_csv(TAB / "kmeans_cluster_profiles.csv")
print("\nK-Means cluster profiles:\n", prof.to_string())
df["ClusterName"] = df["Cluster"].map(prof["name"])

fig, ax = plt.subplots(1, 2, figsize=(14, 5))
sns.scatterplot(data=df, x="tenure", y="MonthlyCharges", hue="ClusterName", alpha=0.6, s=18, ax=ax[0])
ax[0].set_title("K-Means Customer Segments"); ax[0].legend(fontsize=8)
pr = prof.reset_index()
sns.barplot(data=pr, x="name", y="churn_rate", color="#E4572E", ax=ax[1])
ax[1].axhline(overall, ls="--", c="grey", label=f"Overall {overall:.1%}")
for i, v in enumerate(pr["churn_rate"]): ax[1].text(i, v + 0.01, f"{v:.1%}", ha="center")
ax[1].set_title("Churn Rate by Cluster"); ax[1].set_xlabel(""); ax[1].legend()
plt.setp(ax[1].get_xticklabels(), rotation=15, ha="right")
save_fig("13_kmeans_segments.png")

# ------------------------------------------------------------ 3. High-value customers at risk
# High value = top quartile of monthly charges AND at least 6 months tenure (established revenue).
# At risk (rule-based, before modelling) = month-to-month contract.
hv_cut = df["MonthlyCharges"].quantile(0.75)
df["HighValue"] = (df["MonthlyCharges"] >= hv_cut) & (df["tenure"] >= 6)
df["AtRisk"] = df["Contract"] == "Month-to-month"
hv = df[df.HighValue]
hv_risk = df[df.HighValue & df.AtRisk]
print(f"\nHigh-value cut-off: monthly charges >= ${hv_cut:.2f} and tenure >= 6 months")
print(f"High-value customers: {len(hv)}  | churn rate {hv.ChurnFlag.mean():.1%}")
print(f"High-value + month-to-month: {len(hv_risk)} | churn rate {hv_risk.ChurnFlag.mean():.1%}")
still = hv_risk[hv_risk.ChurnFlag == 0]
print(f"  of which still active: {len(still)} -> monthly revenue exposed ${still.MonthlyCharges.sum():,.0f} "
      f"(${still.MonthlyCharges.sum()*12:,.0f}/year)")
lost = hv[hv.ChurnFlag == 1]
print(f"High-value customers already lost: {len(lost)} -> ${lost.MonthlyCharges.sum():,.0f} monthly revenue lost")

cols = ["customerID", "tenure", "MonthlyCharges", "Contract", "InternetService", "PaymentMethod",
        "TechSupport", "OnlineSecurity", "ClusterName"]
watch = still[cols].sort_values("MonthlyCharges", ascending=False)
watch.to_csv(TAB / "high_value_at_risk_watchlist.csv", index=False)
print(f"Saved watch-list ({len(watch)} active customers) -> outputs/tables/high_value_at_risk_watchlist.csv")

summary = pd.DataFrame({
    "group": ["All customers", "High-value", "High-value + month-to-month", "Other month-to-month", "Contracted (1/2yr)"],
    "customers": [len(df), len(hv), len(hv_risk), int(((df.Contract == "Month-to-month") & ~df.HighValue).sum()),
                  int((df.Contract != "Month-to-month").sum())],
    "churn_rate": [overall, hv.ChurnFlag.mean(), hv_risk.ChurnFlag.mean(),
                   df[(df.Contract == "Month-to-month") & ~df.HighValue].ChurnFlag.mean(),
                   df[df.Contract != "Month-to-month"].ChurnFlag.mean()]})
summary.to_csv(TAB / "high_value_summary.csv", index=False)
plt.figure(figsize=(9, 4.5))
sns.barplot(data=summary, y="group", x="churn_rate", color="#E4572E")
for i, v in enumerate(summary.churn_rate): plt.text(v + 0.005, i, f"{v:.1%}", va="center")
plt.title("Churn Rate: High-Value Customers at Risk"); plt.ylabel(""); plt.xlabel("Churn rate")
plt.xlim(0, summary.churn_rate.max() * 1.15)
save_fig("14_high_value_at_risk.png")
print("\n", summary.round(3).to_string(index=False))
