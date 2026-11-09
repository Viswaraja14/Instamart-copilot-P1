import os

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Reuse the data, the split and the column lists from lapse_data.py
from lapse_data import X_train, X_test, y_train, y_test, cat_cols, num_cols, data


def make_pre():
    """A fresh, unfitted preprocessor (it is refitted inside every cross-validation fold)."""
    return ColumnTransformer([
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols),
        ("num", StandardScaler(), num_cols),
    ])


# ---------- Task 6: tune logistic regression ----------
lr_grid = GridSearchCV(
    Pipeline([("pre", make_pre()), ("model", LogisticRegression(max_iter=1000))]),
    param_grid={"model__C": [0.01, 0.1, 1, 10, 100]},
    cv=5, scoring="f1",
).fit(X_train, y_train)
print("LR best:", lr_grid.best_params_, round(lr_grid.best_score_, 3))

# ---------- Task 6: tune random forest ----------
rf_grid = GridSearchCV(
    Pipeline([("pre", make_pre()),
              ("model", RandomForestClassifier(n_estimators=100, random_state=42))]),
    param_grid={"model__max_depth": [3, 5, 8], "model__min_samples_leaf": [5, 10, 20]},
    cv=5, scoring="f1",
).fit(X_train, y_train)
print("RF best:", rf_grid.best_params_, round(rf_grid.best_score_, 3))

# ---------- Task 6: evaluate on the untouched test set (once, after tuning) ----------
for name, grid in [("Logistic regression", lr_grid), ("Random forest", rf_grid)]:
    pred = grid.predict(X_test)
    print(name)
    print(confusion_matrix(y_test, pred))
    print("precision", round(precision_score(y_test, pred, zero_division=0), 3),
          "recall", round(recall_score(y_test, pred, zero_division=0), 3),
          "F1", round(f1_score(y_test, pred, zero_division=0), 3))

# ---------- Task 7: class-balance experiment (training set only) ----------
minority = X_train[y_train == 1]
n_extra = (y_train == 0).sum() - (y_train == 1).sum()
extra = minority.sample(n=n_extra, replace=True, random_state=42)

X_bal = pd.concat([X_train, extra])
y_bal = pd.concat([y_train, pd.Series(1, index=extra.index)])
print("balanced training set:", len(X_bal), y_bal.value_counts().to_dict())

balanced_lr = clone(lr_grid.best_estimator_).fit(X_bal, y_bal)


def scores(model):
    pred = model.predict(X_test)
    return (round(precision_score(y_test, pred, zero_division=0), 3),
            round(recall_score(y_test, pred, zero_division=0), 3),
            round(f1_score(y_test, pred, zero_division=0), 3))


print("original  precision, recall, F1:", scores(lr_grid))
print("balanced  precision, recall, F1:", scores(balanced_lr))
print(confusion_matrix(y_test, balanced_lr.predict(X_test)))

# ---------- Task 8a: interpretation ----------
rf_best = rf_grid.best_estimator_
importance = pd.Series(
    rf_best.named_steps["model"].feature_importances_,
    index=rf_best.named_steps["pre"].get_feature_names_out(),
).sort_values(ascending=False)
print(importance.head(5).round(3))

lr_best = lr_grid.best_estimator_
coefs = pd.Series(
    lr_best.named_steps["model"].coef_[0],
    index=lr_best.named_steps["pre"].get_feature_names_out(),
)
print(coefs[["num__frequency", "cat__fav_category_Electronics", "num__cancel_rate"]].round(3))

# ---------- Task 8b: save the selected model and prove it reloads exactly ----------
os.makedirs("outputs", exist_ok=True)
selected = lr_grid.best_estimator_            # selected model: logistic regression
joblib.dump(selected, "outputs/lapse_model.joblib")
reloaded = joblib.load("outputs/lapse_model.joblib")
print("reloaded predictions identical:", (reloaded.predict(X_test) == selected.predict(X_test)).all())

# ---------- Task 8c: retention list for Premium customers in the test set ----------
segments = pd.read_csv("outputs/customer_segments.csv")

test = data.loc[X_test.index, ["customer_id", "lapsed"]].copy()
test["lapse_probability"] = reloaded.predict_proba(X_test)[:, 1]
test = test.merge(segments[["customer_id", "is_premium"]], on="customer_id", how="left")

premium_test = test[test["is_premium"] == True]
top = premium_test.sort_values("lapse_probability", ascending=False).head(10)

retention = top.rename(columns={"lapsed": "actually_lapsed"})[
    ["customer_id", "lapse_probability", "actually_lapsed"]]
retention["lapse_probability"] = retention["lapse_probability"].round(4)
retention.to_csv("outputs/retention_list.csv", index=False)

print("premium customers in test set, truly lapsed in top 10:",
      len(premium_test), retention["actually_lapsed"].sum())
print(retention)