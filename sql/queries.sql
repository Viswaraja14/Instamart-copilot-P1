-- Q1: customers with more than 8000 in large completed orders
SELECT customer_id,
       COUNT(*) AS n_orders,
       ROUND(SUM(order_amount), 2) AS total
FROM orders
WHERE status = 'completed' AND order_amount > 500
GROUP BY customer_id
HAVING SUM(order_amount) > 8000
ORDER BY total DESC;

-- Q2: revenue by city and category
SELECT c.city,
       p.category,
       ROUND(SUM(o.order_amount), 2) AS revenue
FROM orders AS o
JOIN customers AS c ON o.customer_id = c.customer_id
JOIN products AS p ON o.product_id = p.product_id
WHERE o.status = 'completed'
GROUP BY c.city, p.category
ORDER BY revenue DESC;

-- Q3: products priced above the average catalogue price
SELECT product_id,
       product_name,
       category,
       price_inr
FROM products
WHERE price_inr > (SELECT AVG(price_inr) FROM products)
ORDER BY price_inr DESC;

-- Q4: customers with no completed order
SELECT c.customer_id,
       c.city,
       c.signup_date
FROM customers AS c
LEFT JOIN orders AS o
       ON o.customer_id = c.customer_id
      AND o.status = 'completed'
WHERE o.order_id IS NULL
ORDER BY c.customer_id;

-- Q5: customers whose total spend is above the average customer total
WITH customer_spend AS (
    SELECT customer_id,
           SUM(order_amount) AS total_spend
    FROM orders
    WHERE status = 'completed'
    GROUP BY customer_id
),
avg_spend AS (
    SELECT AVG(total_spend) AS avg_total
    FROM customer_spend
)
SELECT cs.customer_id,
       ROUND(cs.total_spend, 2) AS total_spend,
       ROUND(a.avg_total, 2) AS avg_total
FROM customer_spend AS cs
CROSS JOIN avg_spend AS a
WHERE cs.total_spend > a.avg_total
ORDER BY cs.total_spend DESC;
-- Q6: rank customers by spend within each city
WITH spend AS (
    SELECT c.customer_id,
           c.city,
           ROUND(SUM(o.order_amount), 2) AS total_spend
    FROM orders AS o
    JOIN customers AS c ON o.customer_id = c.customer_id
    WHERE o.status = 'completed'
    GROUP BY c.customer_id, c.city
)
SELECT customer_id,
       city,
       total_spend,
       RANK()       OVER (PARTITION BY city ORDER BY total_spend DESC) AS rnk,
       DENSE_RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) AS dense_rnk,
       ROW_NUMBER() OVER (PARTITION BY city ORDER BY total_spend DESC) AS row_num
FROM spend
ORDER BY city, row_num;

-- Q7: daily completed revenue with a running total
WITH daily AS (
    SELECT SUBSTR(order_ts, 1, 10) AS day,
           ROUND(SUM(order_amount), 2) AS revenue
    FROM orders
    WHERE status = 'completed'
    GROUP BY SUBSTR(order_ts, 1, 10)
)
SELECT day,
       revenue,
       ROUND(SUM(revenue) OVER (ORDER BY day), 2) AS cumulative_revenue
FROM daily
ORDER BY day;