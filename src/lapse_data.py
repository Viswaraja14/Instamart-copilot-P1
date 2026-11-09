import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------- Load the clean data from Part A ----------
orders = pd.read_csv("data/clean/orders.csv", parse_dates=["order_ts"])
customers = pd.read_csv("data/clean/customers.csv", parse_dates=["signup_date"])
products = pd.read_csv("data/clean/products.csv")

# ---------- Task 4: cohort and label ----------
cutoff = pd.Timestamp("2025-09-01")
window_end = pd.Timestamp("2025-10-01")

completed = orders[orders["status"] == "completed"]
history = completed[completed["order_ts"] < cutoff]
window = completed[(completed["order_ts"] >= cutoff) & (completed["order_ts"] < window_end)]

cohort = pd.DataFrame({"customer_id": history["customer_id"].unique()})
cohort["lapsed"] = (~cohort["customer_id"].isin(window["customer_id"])).astype(int)
print("cohort, lapsed, lapsed %:", len(cohort), cohort["lapsed"].sum(),
      round(cohort["lapsed"].mean() * 100, 2))

# ---------- Task 4: features (only orders before the cutoff) ----------
all_before = orders[orders["order_ts"] < cutoff]
hist = history.merge(products[["product_id", "category"]], on="product_id")

feat = hist.groupby("customer_id").agg(
    last_order_ts=("order_ts", "max"),
    frequency=("order_id", "nunique"),
    monetary=("order_amount", "sum"),
    avg_discount=("discount_pct", "mean"),
    avg_delivery=("delivery_minutes", "mean"),
    fav_category=("category", lambda s: s.mode().iloc[0]),
)
feat["recency"] = (cutoff - feat["last_order_ts"]).dt.days
feat = feat.drop(columns="last_order_ts")

cancel_rate = (all_before.groupby("customer_id")["status"]
               .apply(lambda s: (s == "cancelled").mean())
               .rename("cancel_rate"))

data = (cohort.merge(feat.reset_index(), on="customer_id")
              .merge(cancel_rate.reset_index(), on="customer_id")
              .merge(customers, on="customer_id"))
data["tenure_days"] = (cutoff - data["signup_date"]).dt.days
data = data.drop(columns="signup_date")
print("data shape, missing values:", data.shape, data.isna().sum().sum())

# ---------- Task 4: split and baseline ----------
y = data["lapsed"]
X_feat = data.drop(columns=["customer_id", "lapsed"])
X_train, X_test, y_train, y_test = train_test_split(
    X_feat, y, test_size=0.2, stratify=y, random_state=42)
print("train, test:", len(X_train), len(X_test))

baseline = [0] * len(y_test)
print("baseline accuracy, F1:", round(accuracy_score(y_test, baseline), 3),
      f1_score(y_test, baseline, zero_division=0))

# ---------- Task 5: preprocessing without leakage ----------
cat_cols = ["city", "gender", "fav_category"]
num_cols = [c for c in X_feat.columns if c not in cat_cols]

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols),
    ("num", StandardScaler(), num_cols),
])
preprocessor.fit(X_train)                      # learn from the training rows only
X_train_t = preprocessor.transform(X_train)
X_test_t = preprocessor.transform(X_test)
print("encoded shapes:", X_train_t.shape, X_test_t.shape)

scaler = preprocessor.named_transformers_["num"]
i = num_cols.index("frequency")
print("frequency mean (train, cohort):", round(scaler.mean_[i], 3), round(data["frequency"].mean(), 3))

assert X_feat.isna().sum().sum() == 0, "missing values found in the features"