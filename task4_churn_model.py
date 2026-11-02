"""
Task 4: Churn Prediction Model
 - Logistic Regression, Decision Tree, Random Forest, Gradient Boosting
 - feature selection (SelectKBest inside the pipeline) + hyper-parameter tuning (GridSearchCV, 5-fold, F1)
 - the best model is chosen on CROSS-VALIDATION F1 (train data only); the test set is used only for reporting.
"""
import json, time
import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from common import *

train, test = pd.read_csv(PROCESSED / "train.csv"), pd.read_csv(PROCESSED / "test.csv")
X_train, y_train = train.drop(columns=["customerID", "Churn"]), train["Churn"]
X_test, y_test = test.drop(columns=["customerID", "Churn"]), test["Churn"]
print(f"Train {X_train.shape} | Test {X_test.shape} | features: {X_train.shape[1]}")


def pipe(clf):
    # scaler + selector are fit inside each CV fold -> no leakage
    return Pipeline([("scale", StandardScaler()), ("select", SelectKBest(f_classif)), ("clf", clf)])

K_OPTIONS = [10, 15, 20, "all"]
candidates = {
    "Logistic Regression": (pipe(LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        {"select__k": K_OPTIONS, "clf__C": [0.01, 0.1, 1, 10], "clf__class_weight": [None, "balanced"]}),
    "Decision Tree": (pipe(DecisionTreeClassifier(random_state=RANDOM_STATE)),
        {"select__k": K_OPTIONS, "clf__max_depth": [3, 5, 7, 10], "clf__min_samples_leaf": [1, 10, 30],
         "clf__class_weight": [None, "balanced"]}),
    "Random Forest": (pipe(RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)),
        {"select__k": [15, "all"], "clf__max_depth": [6, 10], "clf__min_samples_leaf": [3, 10],
         "clf__class_weight": [None, "balanced"]}),
    "Gradient Boosting": (pipe(GradientBoostingClassifier(random_state=RANDOM_STATE)),
        {"select__k": [15, "all"], "clf__n_estimators": [100, 200], "clf__learning_rate": [0.05, 0.1],
         "clf__max_depth": [2, 3]}),
}
cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)


def metrics(model, X, y, thr=0.5):
    proba = model.predict_proba(X)[:, 1]; pred = (proba >= thr).astype(int)
    return {"accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred), "f1": f1_score(y, pred), "roc_auc": roc_auc_score(y, proba)}

rows, tuned, best_params = [], {}, {}
# Baseline: always predict the majority class (shows why accuracy alone is misleading)
base = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
rows.append({"model": "Baseline (majority class)", "cv_f1": np.nan, **metrics(base, X_test, y_test)})

for name, (pl, grid) in candidates.items():
    t0 = time.time()
    # (a) default settings for reference
    default = pl.fit(X_train, y_train)
    d = metrics(default, X_test, y_test)
    rows.append({"model": f"{name} (default, all features)", "cv_f1": np.nan, **d})
    # (b) tuned
    gs = GridSearchCV(pl, grid, scoring="f1", cv=cv, n_jobs=-1, refit=True).fit(X_train, y_train)
    tuned[name], best_params[name] = gs.best_estimator_, gs.best_params_
    m = metrics(gs.best_estimator_, X_test, y_test)
    rows.append({"model": f"{name} (tuned)", "cv_f1": gs.best_score_, **m})
    print(f"{name:20s} best CV F1={gs.best_score_:.4f} | test F1={m['f1']:.4f} AUC={m['roc_auc']:.4f} "
          f"| {time.time()-t0:.0f}s\n   params: {gs.best_params_}")

res = pd.DataFrame(rows)
res.to_csv(TAB / "task4_model_comparison.csv", index=False)
print("\n", res.round(4).to_string(index=False))

# ---- choose best by cross-validated F1 (never by test score)
cv_scores = res[res.model.str.endswith("(tuned)")].set_index("model")["cv_f1"]
best_name = cv_scores.idxmax().replace(" (tuned)", "")
best = tuned[best_name]
print(f"\nBest model by CV F1: {best_name}")

# which features were kept by the selector?
mask = best.named_steps["select"].get_support()
selected = list(X_train.columns[mask])
print(f"Selected {len(selected)}/{X_train.shape[1]} features: {selected}")

joblib.dump(tuned, MODELS / "all_tuned_models.joblib")
joblib.dump(best, MODELS / "best_model.joblib")
json.dump({"best_model": best_name, "best_params": {k: str(v) for k, v in best_params[best_name].items()},
           "selected_features": selected, "all_features": list(X_train.columns)},
          open(MODELS / "best_model_info.json", "w"), indent=2)

# ---- chart
plot = res[res.model.str.endswith("(tuned)") | res.model.str.startswith("Baseline")].melt(
    id_vars="model", value_vars=["accuracy", "precision", "recall", "f1", "roc_auc"])
plot["model"] = plot["model"].str.replace(" (tuned)", "", regex=False).str.replace(" (majority class)", "", regex=False)
plt.figure(figsize=(12, 5))
sns.barplot(data=plot, x="variable", y="value", hue="model")
plt.ylim(0, 1); plt.xlabel(""); plt.ylabel("Score (test set)"); plt.title("Tuned Model Comparison")
plt.legend(fontsize=8, loc="lower right")
save_fig("15_model_comparison.png")
