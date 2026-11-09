import pandas as pd

orders = pd.read_csv("data/clean/orders.csv")
customers = pd.read_csv("data/clean/customers.csv")
products = pd.read_csv("data/clean/products.csv")

completed = orders[orders["status"] == "completed"]
df = (completed
      .merge(customers[["customer_id", "city"]], on="customer_id")
      .merge(products[["product_id", "category"]], on="product_id"))
print(df.shape)

pivot = pd.pivot_table(df, index="city", columns="category",
                       values="order_amount", aggfunc="sum", margins=True)
print(pivot.round(2))
print(round(pivot.loc["All", "All"], 2))
city_summary = df.groupby("city")["order_amount"].agg(
    total="sum",
    mean_amount="mean",
    n_orders="count",
).round(2)
print(city_summary)
print(round(city_summary["total"].sum(), 2) == 2998668.8)

s = df["order_amount"]
print(round(s.mean(), 2))
print(s.median())
print(round(s.skew(), 2))

q1 = s.quantile(0.25)
q3 = s.quantile(0.75)
iqr = q3 - q1
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr
print(q1, q3, iqr)
print(round(lower, 2), round(upper, 2))
print((s < lower).sum(), (s > upper).sum())

cols = ["order_amount", "discount_pct", "quantity", "delivery_minutes"]
corr = df[cols].corr().round(3)
print(corr)