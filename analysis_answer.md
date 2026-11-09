This data records Instamart quick-commerce orders: 1,200 customers in six Indian cities buying 40 products across five categories between April and September 2025, with 11,110 completed orders worth INR 2,998,668.80.

Order amounts are strongly right-skewed. The mean (INR 269.91) is about double the median (INR 135.00), with skewness of 4.61, Q1 of 64.80 and Q3 of 297.00.

Using the 1.5×IQR rule, the upper fence is INR 645.30 and 897 orders lie above it; none fall below the lower fence. About 70% of these are Electronics purchases, so they look like genuine expensive orders, not data errors, and I kept them.

Correlations are weak. The largest in absolute value is order_amount with discount_pct (about -0.095), followed by order_amount with quantity (+0.087); delivery_minutes is essentially unrelated to everything else. Correlation of this size does not establish causation.