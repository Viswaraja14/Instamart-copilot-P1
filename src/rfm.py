import pandas as pd


orders = pd.read_csv("data/clean/orders.csv", parse_dates=["order_ts"])
completed = orders[orders["status"] == "completed"]
snapshot = pd.Timestamp("2025-10-01 00:00:00")

g = completed.groupby("customer_id").agg(
    last_order_ts=("order_ts", "max"),
    frequency=("order_id", "nunique"),
    monetary=("order_amount", "sum"),
)
g["recency"] = (snapshot - g["last_order_ts"]).dt.days
rfm = g[["recency", "frequency", "monetary"]].reset_index()

print(len(rfm))
print(rfm["recency"].median(), rfm["recency"].min(), rfm["recency"].max())
print(round(rfm["frequency"].mean(), 2), rfm["frequency"].max())
print(round(rfm["monetary"].mean(), 2), round(rfm["monetary"].max(), 2))
print(rfm.head())

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

X = StandardScaler().fit_transform(rfm[["recency", "frequency", "monetary"]])
print(X.shape)
print(X.mean(axis=0).round(3), X.std(axis=0).round(3))

import pandas as pd

rows = []
for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    rows.append((k, round(km.inertia_, 1), round(silhouette_score(X, km.labels_), 3)))

table = pd.DataFrame(rows, columns=["K", "inertia", "silhouette"])
print(table.to_string(index=False))

import os

kmeans = KMeans(n_clusters=4, n_init=10, random_state=42).fit(X)
rfm["segment_id"] = kmeans.labels_

profile = rfm.groupby("segment_id").agg(
    size=("customer_id", "count"),
    mean_recency=("recency", "mean"),
    mean_frequency=("frequency", "mean"),
    mean_monetary=("monetary", "mean"),
).round(1)
print(profile)

names = {0: "Casual Buyers", 1: "Premium Big Spenders",
         2: "Loyal Regulars", 3: "Lapsing Customers"}
rfm["segment_name"] = rfm["segment_id"].map(names)

premium_id = profile["mean_monetary"].idxmax()
rfm["is_premium"] = rfm["segment_id"] == premium_id

os.makedirs("outputs", exist_ok=True)
rfm[["customer_id", "recency", "frequency", "monetary",
     "segment_id", "segment_name", "is_premium"]].to_csv("outputs/customer_segments.csv", index=False)
print(rfm["is_premium"].sum(), rfm.loc[rfm["is_premium"], "segment_name"].unique())

from sklearn.decomposition import PCA

pca = PCA(n_components=2).fit(X)
print(pca.explained_variance_ratio_.round(3))
print(round(pca.explained_variance_ratio_.sum(), 3))