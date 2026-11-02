# Task 6 - Business Recommendations
*Customer Churn Analysis & Prediction | Telco dataset (7,043 customers)*

## 1. Where we stand
- Overall churn is **26.5%** (1,869 customers). The customers who left represent roughly
  **$1.67M of annualised revenue**.
- The best model (Random Forest, tuned, test ROC-AUC 0.847) ranks customers well:
  the top 20% highest-risk customers churn at **68%**, versus 27% on average (test set).

## 2. What drives churn (evidence from EDA, segmentation and the model)
| Driver | Finding |
|---|---|
| Contract type | Month-to-month churn **42.7%** vs one-year 11.3% vs two-year 2.8%. 55% of customers are month-to-month. |
| Tenure | Customers in their first 6 months churn at **53%**; those with 49+ months at 10%. |
| Payment method | Electronic-check customers churn at **45.3%** (34% of base) vs 16.0% for automatic payments. |
| Internet service | Fiber-optic customers churn at **41.9%** vs DSL 19.0%. |
| Support add-ons | Internet customers without Tech Support churn at 41.6% vs 15.2% with it; without Online Security 41.8% vs 14.6% with it. |
| Demographics | Seniors churn at 41.7% (non-seniors 23.6%); single customers with no dependents 34.2% vs 19.8% for customers with a partner/dependents. Gender has no significant effect. |
| Highest-risk segment | New (<=12m), high-charge, month-to-month customers churn at **75%**. |

*Model view:* tenure, contract type, fiber-optic internet and total/monthly charges are the most important predictors
(see `outputs/tables/task5_permutation_importance.csv`). Note these are associations, not proven causes.

## 3. Recommendations
1. **Move month-to-month customers onto 12-month plans.** Offer a time-limited discount or a free add-on
   (e.g. Tech Support or Online Security) for committing to a 1-year contract. Contract length is the single biggest lever:
   the churn gap between month-to-month and contracted customers is 31 percentage points or more.
2. **Build a first-90-days onboarding programme.** Churn is steepest early (47% within the first year).
   Use welcome calls, a day-30 and day-90 check-in, and proactive billing/usage explanations.
3. **Target by model score, not by hunch.** Score the base monthly and send the retention offer to customers above the
   **0.50** predicted-risk threshold (flags about 2,739 customers, ~39% of the base). Lower thresholds reach more churners but contact many more people - see `outputs/tables/task6_threshold_economics.csv`. Review the
   high-value watch-list: **394 active high-value month-to-month customers** worth about **$466,319/year**
   (`outputs/tables/high_value_at_risk_watchlist.csv`) - give these a named account contact / priority support.
4. **Shift electronic-check payers to automatic payment.** Offer a small one-off bill credit for switching to card or
   bank transfer. Automatic payers churn far less; this may also be a proxy for engagement, so test it with an A/B pilot.
5. **Review fiber-optic pricing and service quality.** Fiber customers churn at 42%; investigate price
   competitiveness, outages and support experience (survey, exit interviews) before cutting prices blindly.
6. **Bundle security/support services** into fiber and month-to-month plans as a trial: customers using them churn less
   (a 26-percentage-point gap for Tech Support), though part of this may reflect customer engagement rather than the service itself.
7. **Senior and single-person households:** simplified plans, assisted billing, and loyalty perks aimed at these higher-churn groups.

## 4. Estimated impact of the retention offer (recommendation 3)
**Programme:** customers scored >= 0.50 receive 10% off for 12 months in exchange for a 12-month commitment.
Results use realised churn in the held-out test set, scaled x5.0 to the full customer base.

| Scenario | Offer uptake | Customers flagged | Customers saved | Revenue retained / yr | Discount given to non-churners | **Net revenue / yr** | Churn rate after |
|---|---|---|---|---|---|---|---|
| Conservative | 15% | 2,739 | 219 | $175,086 | $17,119 | **$157,967** | 23.4% |
| Base | 25% | 2,739 | 365 | $291,810 | $28,531 | **$263,279** | 21.4% |
| Optimistic | 40% | 2,739 | 584 | $466,896 | $45,650 | **$421,246** | 18.2% |

Baseline churn rate is 26.5%. Under the base case the churn rate falls to **21.4%**
and net annual revenue improves by about **$263,279**.

### Assumptions & caveats (please read)
- **Uptake (15-40%) and the 10% discount are assumptions, not data.** The dataset has no information on how customers respond to offers.
  The sensitivity heat-map (`outputs/figures/21_sensitivity_net_impact.png`) shows the result for other values.
- Customers who accept are assumed to be retained for 12 months and to pay (1 - discount) of their current bill. Non-churners who
  accept the offer cost us the discount - this is included.
- Costs of running campaigns (staff, messaging, contacting ~2,739 customers) are **not** included. Revenue = monthly charges x 12; margin is not modelled.
- The estimate uses a single 80/20 split; with ~1,400 test customers it carries meaningful uncertainty.
- Recommended next step: run an **A/B pilot** on a random half of the flagged customers to measure real uptake and
  retention before a full roll-out.
