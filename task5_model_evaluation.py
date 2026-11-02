"""
Task 5: Model Evaluation and Interpretation
 - evaluate the best model on the held-out test set
 - confusion matrix, classification report, ROC curves + AUC (all tuned models)
 - feature importances / coefficients + permutation importance (test set)
 - threshold analysis; saves per-customer churn probabilities for Task 6
"""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (ConfusionMatrixDisplay, auc, classification_report, confusion_matrix,
                             precision_recall_curve, roc_auc_score, roc_curve, average_precision_score)
from common import *

test = pd.read_csv(PROCESSED / "test.csv")
X_test, y_test = test.drop(columns=["customerID", "Churn"]), test["Churn"]
info = json.load(open(MODELS / "best_model_info.json"))
models = joblib.load(MODELS / "all_tuned_models.joblib")
best_name, best = info["best_model"], models[info["best_model"]]
print(f"Best model: {best_name}\nParams: {info['best_params']}")

proba = best.predict_proba(X_test)[:, 1]
pred = (proba >= 0.5).astype(int)

# ---------------------------------------------------------------- 1. Test-set performance
print("\nClassification report (threshold 0.5):\n",
      classification_report(y_test, pred, target_names=["Retained", "Churned"], digits=3))
tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
print(f"Confusion matrix: TN={tn} FP={fp} FN={fn} TP={tp}")
print(f"ROC-AUC = {roc_auc_score(y_test, proba):.4f} | PR-AUC = {average_precision_score(y_test, proba):.4f}")
ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=["Retained", "Churned"], cmap="Blues")
plt.title(f"Confusion Matrix - {best_name}"); save_fig("16_confusion_matrix.png")

# ---------------------------------------------------------------- 2. ROC curves, all models
plt.figure(figsize=(7, 6))
auc_rows = []
for name, m in models.items():
    p = m.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, p); a = auc(fpr, tpr)
    auc_rows.append({"model": name, "test_auc": a})
    plt.plot(fpr, tpr, lw=3 if name == best_name else 1.5, label=f"{name} (AUC={a:.3f})")
plt.plot([0, 1], [0, 1], "k--", label="Random (AUC=0.500)")
plt.xlabel("False positive rate"); plt.ylabel("True positive rate (recall)")
plt.title("ROC Curves - Test Set"); plt.legend(loc="lower right")
save_fig("17_roc_curves.png")
pd.DataFrame(auc_rows).to_csv(TAB / "task5_auc_by_model.csv", index=False)

# ---------------------------------------------------------------- 3. Interpretation
sel = np.array(info["all_features"])[best.named_steps["select"].get_support()]
clf = best.named_steps["clf"]
if hasattr(clf, "feature_importances_"):
    imp = pd.Series(clf.feature_importances_, index=sel).sort_values(ascending=False)
    imp.rename("importance").to_csv(TAB / "task5_feature_importance.csv")
    top = imp.head(15)[::-1]
    plt.figure(figsize=(8, 6)); top.plot(kind="barh", color="#4C78A8")
    plt.title(f"{best_name}: Top 15 Feature Importances (impurity)"); save_fig("18_feature_importance.png")
    print("\nTop feature importances:\n", imp.head(10).round(4).to_string())
if hasattr(clf, "coef_"):
    coef = pd.Series(clf.coef_[0], index=sel).sort_values()
    odds = np.exp(coef).rename("odds_ratio_per_1sd")
    pd.concat([coef.rename("coefficient"), odds], axis=1).to_csv(TAB / "task5_coefficients.csv")
    plt.figure(figsize=(8, 7)); coef.plot(kind="barh", color=np.where(coef > 0, "#E4572E", "#4C78A8"))
    plt.title("Logistic Regression Coefficients (standardised features)"); save_fig("18b_coefficients.png")
    print("\nCoefficients:\n", coef.round(3).to_string())

# Permutation importance on the TEST set (model-agnostic, uses all pipeline steps)
pi = permutation_importance(best, X_test, y_test, scoring="roc_auc", n_repeats=15,
                            random_state=RANDOM_STATE, n_jobs=-1)
perm = pd.DataFrame({"feature": X_test.columns, "mean_auc_drop": pi.importances_mean,
                     "std": pi.importances_std}).sort_values("mean_auc_drop", ascending=False)
perm.to_csv(TAB / "task5_permutation_importance.csv", index=False)
t = perm.head(12)[::-1]
plt.figure(figsize=(8, 5.5))
plt.barh(t.feature, t.mean_auc_drop, xerr=t["std"], color="#E4572E")
plt.xlabel("Drop in ROC-AUC when feature is shuffled"); plt.title("Permutation Importance (test set)")
save_fig("19_permutation_importance.png")
print("\nTop permutation importances (AUC drop):\n", perm.head(10).round(4).to_string(index=False))

# Direction of effect for the top features (works for any model)
train = pd.read_csv(PROCESSED / "train.csv")
dir_rows = []
for f in perm.head(8).feature:
    if train[f].nunique() <= 2:
        r1 = train.loc[train[f] == 1, "Churn"].mean(); r0 = train.loc[train[f] == 0, "Churn"].mean()
        dir_rows.append({"feature": f, "churn_rate_when_1": r1, "churn_rate_when_0": r0})
    else:
        hi = train[f] >= train[f].median()
        dir_rows.append({"feature": f, "churn_rate_when_above_median": train.loc[hi, "Churn"].mean(),
                         "churn_rate_when_below_median": train.loc[~hi, "Churn"].mean()})
dirs = pd.DataFrame(dir_rows); dirs.to_csv(TAB / "task5_direction_of_effect.csv", index=False)
print("\nDirection of effect (train data):\n", dirs.round(3).to_string(index=False))

# ---------------------------------------------------------------- 4. Threshold analysis
prec, rec, thr = precision_recall_curve(y_test, proba)
f1s = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
plt.figure(figsize=(7, 4.5))
plt.plot(thr, prec[:-1], label="Precision"); plt.plot(thr, rec[:-1], label="Recall"); plt.plot(thr, f1s[:-1], label="F1")
plt.axvline(0.5, ls=":", c="grey"); plt.xlabel("Decision threshold"); plt.legend(); plt.title("Precision / Recall vs Threshold")
save_fig("20_threshold_analysis.png")
rows = []
for t_ in [0.3, 0.4, 0.5, 0.6, 0.7]:
    p_ = (proba >= t_).astype(int)
    tp_, fp_ = int(((p_ == 1) & (y_test == 1)).sum()), int(((p_ == 1) & (y_test == 0)).sum())
    rows.append({"threshold": t_, "flagged": int(p_.sum()), "precision": tp_ / max(p_.sum(), 1),
                 "recall": tp_ / y_test.sum()})
print("\nThreshold trade-off:\n", pd.DataFrame(rows).round(3).to_string(index=False))
pd.DataFrame(rows).to_csv(TAB / "task5_threshold_tradeoff.csv", index=False)

# Risk deciles (lift) - how concentrated is churn among the top-scored customers?
d = pd.DataFrame({"p": proba, "y": y_test.values})
d["decile"] = pd.qcut(d.p.rank(method="first", ascending=False), 10, labels=range(1, 11))
lift = d.groupby("decile", observed=True).agg(customers=("y", "size"), churn_rate=("y", "mean"), avg_prob=("p", "mean"))
lift["lift"] = lift.churn_rate / y_test.mean()
lift.to_csv(TAB / "task5_risk_deciles.csv")
print("\nRisk deciles (1 = highest predicted risk):\n", lift.round(3).to_string())

# Save predictions for Task 6
pd.DataFrame({"customerID": test["customerID"], "churn_probability": proba, "actual_churn": y_test.values}) \
  .to_csv(TAB / "test_predictions.csv", index=False)
