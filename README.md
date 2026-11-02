# Customer Churn Analysis and Prediction
**SaiKet Systems - Data Science Internship** | Telco Customer Churn dataset (7,043 customers, 21 columns)

All **6 tasks** are completed (the brief requires 4 of 6), each in its own file.

| Task | File | What it does |
|---|---|---|
| 1 | `task1_data_preparation.py` | Load/explore, fix 11 blank `TotalCharges`, encode categoricals, stratified 80/20 split |
| 2 | `task2_eda.py` | Churn rate, demographics, tenure, contract, payment method + significance tests |
| 3 | `task3_segmentation.py` | Tenure x charges x contract segments, K-Means clusters, high-value at-risk watch-list |
| 4 | `task4_churn_model.py` | LR / Decision Tree / Random Forest / Gradient Boosting, feature selection, GridSearchCV |
| 5 | `task5_model_evaluation.py` | Test-set evaluation, ROC/AUC, importances, threshold & decile analysis |
| 6 | `task6_business_recommendations.py` | Recommendations + revenue/churn impact scenarios -> `outputs/Business_Recommendations.md` |

## How to run
```bash
pip install -r requirements.txt
python run_all.py            # or run the task files one by one, in order (each depends on the previous)
```
Results are written to `outputs/figures` (22 charts), `outputs/tables` (CSV), `outputs/models` (saved models).
The data is in `data/`. Task 4 grid search takes ~3-4 minutes.

## Key results
- **Overall churn: 26.5%** (1,869 of 7,043).
- **Strongest drivers:** month-to-month contract (42.7% churn vs 2.8% for two-year), short tenure (53% churn in first 6 months),
  electronic-check payment (45%), fiber-optic internet (42%), no Tech Support / Online Security. Gender is not significant.
- **Highest-risk segment:** new, high-charge, month-to-month customers - 75% churn.
- **High-value at-risk watch-list:** 394 active customers (~$466K annual revenue) - `outputs/tables/high_value_at_risk_watchlist.csv`.
- **Best model:** Random Forest (selected on 5-fold CV F1, not on test data): test ROC-AUC **0.847**, recall 0.78, precision 0.53, F1 0.63, accuracy 0.76.
  Class-weighting trades some accuracy for recall - a deliberate choice, since missing a churner costs more than a wasted offer.
  (A "never churn" baseline scores 73% accuracy but 0 recall.)
- **Business case (assumption-based):** a targeted 10%-discount retention offer could cut churn from 26.5% to roughly 21% and add about
  $263K net annual revenue at 25% uptake (range $158K-$421K). Uptake and discount are *assumptions* - validate with an A/B pilot.

## Methodology notes
- Scaling and feature selection happen inside the CV pipeline (fit on training folds only) to prevent data leakage.
- The test set is used only for final reporting. Tuning selected all 23 features for the winning model; the selector was part of the search.
- Stratified split keeps the churn ratio (~26.5%) identical in train and test.
- Feature importances show association, not causation.

## Reminder (from the internship brief)
Post a LinkedIn video of your work tagging SaiKet Systems with #saiketsystems #saiket #saiketsys, and make sure all code is your own understanding - review and be able to explain every step before submitting.

## Web app / deployment
`app.py` is a Streamlit app (single-customer prediction + batch CSV upload). Run `streamlit run app.py`.
See `DEPLOY.md` for free deployment on Streamlit Community Cloud.
