"""
Task 6: Business Recommendations
 - quantify the retention opportunity using the model's test-set predictions
 - sensitivity analysis on the key business assumptions
 - writes outputs/Business_Recommendations.md with all numbers filled in from the data
"""
import json
import numpy as np
import pandas as pd
from common import *

raw = load_clean()
raw["ChurnFlag"] = (raw.Churn == "Yes").astype(int)
pred = pd.read_csv(TAB / "test_predictions.csv").merge(
    raw[["customerID", "MonthlyCharges", "Contract", "tenure", "InternetService", "PaymentMethod"]], on="customerID")
info = json.load(open(MODELS / "best_model_info.json"))
N_ALL, N_TEST = len(raw), len(pred)
SCALE = N_ALL / N_TEST                    # test set = 20% of customers -> scale results to full base
base_churners = raw.ChurnFlag.sum()
base_churn_rate = raw.ChurnFlag.mean()
annual_rev_churned = (raw.loc[raw.ChurnFlag == 1, "MonthlyCharges"] * 12).sum()
print(f"Customers {N_ALL:,} | churn rate {base_churn_rate:.2%} | annualised revenue of churned customers ${annual_rev_churned:,.0f}")

# ----------------------------------------------------------------- Retention programme economics
# Programme: customers flagged by the model get a retention offer = `discount` off the monthly bill for
# 12 months in return for a 12-month commitment.  `uptake` = share of flagged customers who accept.
# Accepting churners are retained (we keep (1-discount) of their revenue); accepting non-churners just
# cost us the discount.  Realised churn behaviour in the test set is used (not raw model probabilities,
# which are inflated by class weighting).
def programme(threshold, uptake, discount):
    f = pred[pred.churn_probability >= threshold]
    annual = f.MonthlyCharges * 12
    churners = f.actual_churn == 1
    saved_rev = uptake * (annual[churners] * (1 - discount)).sum()
    cost = uptake * (annual[~churners] * discount).sum()
    saved_cust = uptake * churners.sum()
    return {"threshold": threshold, "uptake": uptake, "discount": discount, "flagged": len(f) * SCALE,
            "precision": churners.mean(), "customers_saved": saved_cust * SCALE,
            "revenue_retained": saved_rev * SCALE, "discount_cost_non_churners": cost * SCALE,
            "net_annual_impact": (saved_rev - cost) * SCALE,
            "new_churn_rate": (base_churners - saved_cust * SCALE) / N_ALL}

# 1) pick the targeting threshold with the best net impact (base assumptions)
BASE_UPTAKE, BASE_DISC = 0.25, 0.10
thr_tbl = pd.DataFrame([programme(t, BASE_UPTAKE, BASE_DISC) for t in np.arange(0.3, 0.86, 0.05)])
thr_tbl.to_csv(TAB / "task6_threshold_economics.csv", index=False)
# Net impact (excluding contact costs) keeps rising as the threshold falls, because it ignores the cost of
# contacting many people.  We therefore use the model's standard 0.50 operating point (precision ~53%,
# recall ~78%) - a realistic campaign size - and show other thresholds in the table for reference.
best_thr = 0.50
print("\nNet impact by targeting threshold (uptake 25%, 10% discount):\n",
      thr_tbl[["threshold", "flagged", "precision", "net_annual_impact", "new_churn_rate"]].round(3).to_string(index=False))
print(f"\nOperating threshold used: {best_thr:.2f}")

# 2) scenarios
scen = pd.DataFrame([{"scenario": n, **programme(best_thr, u, d)} for n, u, d in
                     [("Conservative", 0.15, 0.10), ("Base", 0.25, 0.10), ("Optimistic", 0.40, 0.10)]])
scen.to_csv(TAB / "task6_scenarios.csv", index=False)
print("\nScenarios:\n", scen[["scenario", "uptake", "customers_saved", "revenue_retained",
                              "discount_cost_non_churners", "net_annual_impact", "new_churn_rate"]].round(3).to_string(index=False))

# 3) sensitivity grid (uptake x discount)
grid = pd.DataFrame([programme(best_thr, u, d) for u in (0.10, 0.20, 0.30, 0.40, 0.50) for d in (0.05, 0.10, 0.15, 0.20)])
piv = grid.pivot(index="uptake", columns="discount", values="net_annual_impact")
piv.to_csv(TAB / "task6_sensitivity_net_impact.csv")
plt.figure(figsize=(7.5, 4.5))
sns.heatmap(piv / 1000, annot=True, fmt=".0f", cmap="RdYlGn", center=0, cbar_kws={"label": "Net annual impact ($K)"})
plt.xlabel("Discount offered"); plt.ylabel("Offer uptake"); plt.title("Retention Programme: Net Annual Revenue Impact ($K)")
plt.gca().set_xticklabels([f"{c:.0%}" for c in piv.columns]); plt.gca().set_yticklabels([f"{i:.0%}" for i in piv.index], rotation=0)
save_fig("21_sensitivity_net_impact.png")

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
sns.barplot(data=scen, x="scenario", y="net_annual_impact", color="#4C78A8", ax=ax[0])
for i, v in enumerate(scen.net_annual_impact): ax[0].text(i, v, f"${v/1000:,.0f}K", ha="center", va="bottom")
ax[0].set_title("Net Annual Revenue Impact"); ax[0].set_xlabel(""); ax[0].set_ylabel("$")
rates = [base_churn_rate] + list(scen.new_churn_rate)
sns.barplot(x=["Today"] + list(scen.scenario), y=rates, color="#E4572E", ax=ax[1])
for i, v in enumerate(rates): ax[1].text(i, v, f"{v:.1%}", ha="center", va="bottom")
ax[1].set_title("Projected Churn Rate"); ax[1].set_ylim(0, base_churn_rate * 1.2)
save_fig("22_scenario_impact.png")

# ----------------------------------------------------------------- Facts for the narrative (all computed)
def rate(mask): return raw.loc[mask, "ChurnFlag"].mean()
inet = raw.InternetService != "No"
mtm = raw.Contract == "Month-to-month"
facts = {
    "mtm": rate(mtm), "one": rate(raw.Contract == "One year"), "two": rate(raw.Contract == "Two year"),
    "mtm_share": mtm.mean(),
    "echeck": rate(raw.PaymentMethod == "Electronic check"), "echeck_share": (raw.PaymentMethod == "Electronic check").mean(),
    "auto": rate(raw.PaymentMethod.str.contains("automatic")),
    "t0_6": rate(raw.tenure <= 6), "t0_12": rate(raw.tenure <= 12), "t49": rate(raw.tenure >= 49),
    "fiber": rate(raw.InternetService == "Fiber optic"), "dsl": rate(raw.InternetService == "DSL"),
    "nots": rate(inet & (raw.TechSupport == "No")), "ts": rate(inet & (raw.TechSupport == "Yes")),
    "nosec": rate(inet & (raw.OnlineSecurity == "No")), "sec": rate(inet & (raw.OnlineSecurity == "Yes")),
    "senior": rate(raw.SeniorCitizen == 1), "nonsenior": rate(raw.SeniorCitizen == 0),
    "single": rate((raw.Partner == "No") & (raw.Dependents == "No")),
    "family": rate((raw.Partner == "Yes") | (raw.Dependents == "Yes")),
    "new_hi_mtm": rate((raw.tenure <= 12) & (raw.MonthlyCharges > raw.MonthlyCharges.quantile(2 / 3)) & mtm),
}
hv = pd.read_csv(TAB / "high_value_at_risk_watchlist.csv")
hv_rev = hv.MonthlyCharges.sum() * 12
decile = pd.read_csv(TAB / "task5_risk_deciles.csv")
top2 = decile.head(2).churn_rate.mean()
cmp_ = pd.read_csv(TAB / "task4_model_comparison.csv")
b = scen.set_index("scenario")
ts_gap = facts["nots"] - facts["ts"]

md = f"""# Task 6 - Business Recommendations
*Customer Churn Analysis & Prediction | Telco dataset ({N_ALL:,} customers)*

## 1. Where we stand
- Overall churn is **{base_churn_rate:.1%}** ({int(base_churners):,} customers). The customers who left represent roughly
  **${annual_rev_churned/1e6:.2f}M of annualised revenue**.
- The best model ({info['best_model']}, tuned, test ROC-AUC {cmp_.loc[cmp_.model=='Random Forest (tuned)','roc_auc'].iloc[0]:.3f}) ranks customers well:
  the top 20% highest-risk customers churn at **{top2:.0%}**, versus {pred.actual_churn.mean():.0%} on average (test set).

## 2. What drives churn (evidence from EDA, segmentation and the model)
| Driver | Finding |
|---|---|
| Contract type | Month-to-month churn **{facts['mtm']:.1%}** vs one-year {facts['one']:.1%} vs two-year {facts['two']:.1%}. {facts['mtm_share']:.0%} of customers are month-to-month. |
| Tenure | Customers in their first 6 months churn at **{facts['t0_6']:.0%}**; those with 49+ months at {facts['t49']:.0%}. |
| Payment method | Electronic-check customers churn at **{facts['echeck']:.1%}** ({facts['echeck_share']:.0%} of base) vs {facts['auto']:.1%} for automatic payments. |
| Internet service | Fiber-optic customers churn at **{facts['fiber']:.1%}** vs DSL {facts['dsl']:.1%}. |
| Support add-ons | Internet customers without Tech Support churn at {facts['nots']:.1%} vs {facts['ts']:.1%} with it; without Online Security {facts['nosec']:.1%} vs {facts['sec']:.1%} with it. |
| Demographics | Seniors churn at {facts['senior']:.1%} (non-seniors {facts['nonsenior']:.1%}); single customers with no dependents {facts['single']:.1%} vs {facts['family']:.1%} for customers with a partner/dependents. Gender has no significant effect. |
| Highest-risk segment | New (<=12m), high-charge, month-to-month customers churn at **{facts['new_hi_mtm']:.0%}**. |

*Model view:* tenure, contract type, fiber-optic internet and total/monthly charges are the most important predictors
(see `outputs/tables/task5_permutation_importance.csv`). Note these are associations, not proven causes.

## 3. Recommendations
1. **Move month-to-month customers onto 12-month plans.** Offer a time-limited discount or a free add-on
   (e.g. Tech Support or Online Security) for committing to a 1-year contract. Contract length is the single biggest lever:
   the churn gap between month-to-month and contracted customers is {(facts['mtm']-facts['one'])*100:.0f} percentage points or more.
2. **Build a first-90-days onboarding programme.** Churn is steepest early ({facts['t0_12']:.0%} within the first year).
   Use welcome calls, a day-30 and day-90 check-in, and proactive billing/usage explanations.
3. **Target by model score, not by hunch.** Score the base monthly and send the retention offer to customers above the
   **{best_thr:.2f}** predicted-risk threshold (flags about {thr_tbl.loc[(thr_tbl.threshold-best_thr).abs().idxmin(),'flagged']:,.0f} customers, ~{thr_tbl.loc[(thr_tbl.threshold-best_thr).abs().idxmin(),'flagged']/N_ALL:.0%} of the base). Lower thresholds reach more churners but contact many more people - see `outputs/tables/task6_threshold_economics.csv`. Review the
   high-value watch-list: **{len(hv)} active high-value month-to-month customers** worth about **${hv_rev:,.0f}/year**
   (`outputs/tables/high_value_at_risk_watchlist.csv`) - give these a named account contact / priority support.
4. **Shift electronic-check payers to automatic payment.** Offer a small one-off bill credit for switching to card or
   bank transfer. Automatic payers churn far less; this may also be a proxy for engagement, so test it with an A/B pilot.
5. **Review fiber-optic pricing and service quality.** Fiber customers churn at {facts['fiber']:.0%}; investigate price
   competitiveness, outages and support experience (survey, exit interviews) before cutting prices blindly.
6. **Bundle security/support services** into fiber and month-to-month plans as a trial: customers using them churn less
   (a {ts_gap*100:.0f}-percentage-point gap for Tech Support), though part of this may reflect customer engagement rather than the service itself.
7. **Senior and single-person households:** simplified plans, assisted billing, and loyalty perks aimed at these higher-churn groups.

## 4. Estimated impact of the retention offer (recommendation 3)
**Programme:** customers scored >= {best_thr:.2f} receive {BASE_DISC:.0%} off for 12 months in exchange for a 12-month commitment.
Results use realised churn in the held-out test set, scaled x{SCALE:.1f} to the full customer base.

| Scenario | Offer uptake | Customers flagged | Customers saved | Revenue retained / yr | Discount given to non-churners | **Net revenue / yr** | Churn rate after |
|---|---|---|---|---|---|---|---|
""" + "\n".join(
    f"| {r.scenario} | {r.uptake:.0%} | {r.flagged:,.0f} | {r.customers_saved:,.0f} | ${r.revenue_retained:,.0f} | ${r.discount_cost_non_churners:,.0f} | **${r.net_annual_impact:,.0f}** | {r.new_churn_rate:.1%} |"
    for r in scen.itertuples()) + f"""

Baseline churn rate is {base_churn_rate:.1%}. Under the base case the churn rate falls to **{b.loc['Base','new_churn_rate']:.1%}**
and net annual revenue improves by about **${b.loc['Base','net_annual_impact']:,.0f}**.

### Assumptions & caveats (please read)
- **Uptake (15-40%) and the 10% discount are assumptions, not data.** The dataset has no information on how customers respond to offers.
  The sensitivity heat-map (`outputs/figures/21_sensitivity_net_impact.png`) shows the result for other values.
- Customers who accept are assumed to be retained for 12 months and to pay (1 - discount) of their current bill. Non-churners who
  accept the offer cost us the discount - this is included.
- Costs of running campaigns (staff, messaging, contacting ~{thr_tbl.loc[(thr_tbl.threshold-best_thr).abs().idxmin(),'flagged']:,.0f} customers) are **not** included. Revenue = monthly charges x 12; margin is not modelled.
- The estimate uses a single 80/20 split; with ~1,400 test customers it carries meaningful uncertainty.
- Recommended next step: run an **A/B pilot** on a random half of the flagged customers to measure real uptake and
  retention before a full roll-out.
"""
(ROOT / "outputs" / "Business_Recommendations.md").write_text(md)
print("\nWrote outputs/Business_Recommendations.md")
