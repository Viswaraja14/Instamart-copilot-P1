import pandas as pd
import sqlite3

conn = sqlite3.connect("data/instamart.db")
text = open("sql/queries.sql").read()
queries = [q.strip() for q in text.split(";") if q.strip()]

for i, q in enumerate(queries, start=1):
    result = pd.read_sql_query(q, conn)
    print(f"Q{i}: {len(result)} rows")
    print(result.head())


q2 = pd.read_sql_query(queries[1], conn)
print(round(q2["revenue"].sum(), 2))

print(pd.read_sql_query("SELECT AVG(price_inr) AS avg_price FROM products", conn))
q6 = pd.read_sql_query(queries[5], conn)
print(q6[q6["row_num"] == 1])
print((q6["rnk"] != q6["row_num"]).sum())

q7 = pd.read_sql_query(queries[6], conn)
print(q7.tail(2))

conn.close()