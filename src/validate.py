import pandas as pd
import sqlite3

conn = sqlite3.connect("data/instamart.db")
orders = pd.read_sql_query("SELECT * FROM orders", conn)
customers = pd.read_sql_query("SELECT * FROM customers", conn)
conn.close()

assert len(orders) == 12309, "orders row count wrong"
assert len(customers) == 1200, "customers row count wrong"
assert (orders["status"] == "completed").sum() == 11110, "completed count wrong"
assert (orders["status"] == "cancelled").sum() == 1199, "cancelled count wrong"
assert (orders["status"] == "pending").sum() == 0, "pending count wrong"
assert orders["order_amount"].isna().sum() == 0, "missing order_amount found"
completed_total = round(orders.loc[orders["status"] == "completed", "order_amount"].sum(), 2)
assert completed_total == 2998668.8, "completed total wrong"
cancelled_blank = (orders["delivery_minutes"].isna() & (orders["status"] == "cancelled")).sum()
assert cancelled_blank == 1199, "cancelled delivery_minutes should stay blank"


print("ALL CHECKPOINTS PASSED") #passed