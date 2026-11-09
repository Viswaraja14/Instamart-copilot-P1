import pandas as pd
import sqlite3

customers = pd.read_csv("data/clean/customers.csv")
products = pd.read_csv("data/clean/products.csv")
orders = pd.read_csv("data/clean/orders.csv")
print(customers.shape, products.shape, orders.shape)
conn = sqlite3.connect("data/instamart.db")
customers.to_sql("customers", conn, if_exists="replace", index=False)
products.to_sql("products", conn, if_exists="replace", index=False)
orders.to_sql("orders", conn, if_exists="replace", index=False)
print(pd.read_sql_query("SELECT COUNT(*) AS n FROM orders", conn))
print(pd.read_sql_query("SELECT COUNT(*) AS n FROM customers", conn))
print(pd.read_sql_query("SELECT COUNT(*) AS n FROM products", conn))
conn.close()