-- ============================================================
--  Banking Intelligence Project — Banking Customer Risk & Revenue Intelligence
--  Module 6: SQL Analytical Queries (12 production-grade queries)
--  Engine  : PostgreSQL / Google BigQuery compatible
--  Purpose : Extract portfolio-level insights for senior management
-- ============================================================


-- ─────────────────────────────────────────────────────────
-- Q1. Portfolio Default Overview by Employment Type
-- KPI: Default rate breakdown — flags vulnerability segments
-- ─────────────────────────────────────────────────────────
SELECT
    employment_type,
    COUNT(*)                                         AS total_customers,
    SUM(defaulted)                                   AS total_defaults,
    ROUND(AVG(defaulted) * 100, 2)                   AS default_rate_pct,
    ROUND(AVG(credit_score), 0)                      AS avg_credit_score,
    ROUND(AVG(income), 2)                            AS avg_income,
    ROUND(AVG(loan_amount), 2)                       AS avg_loan_amount
FROM customers
GROUP BY employment_type
ORDER BY default_rate_pct DESC;


-- ─────────────────────────────────────────────────────────
-- Q2. High-Risk Customer Identification
-- KPI: Customers with high default probability for proactive outreach
-- ─────────────────────────────────────────────────────────
SELECT
    c.customer_id,
    c.city,
    c.employment_type,
    c.credit_score,
    c.income,
    c.loan_amount,
    rs.default_probability,
    rs.risk_tier,
    cs.segment
FROM customers          c
JOIN risk_scored        rs ON rs.customer_id   = c.customer_id
JOIN customer_segments  cs ON cs.customer_id   = c.customer_id
WHERE rs.risk_tier = 'High Risk'
  AND c.has_loan   = 1
ORDER BY rs.default_probability DESC
LIMIT 100;


-- ─────────────────────────────────────────────────────────
-- Q3. Monthly EMI Collection Efficiency
-- KPI: Repayment health across portfolio over last 12 months
-- ─────────────────────────────────────────────────────────
SELECT
    DATE_TRUNC('month', due_date::DATE)             AS repayment_month,
    COUNT(*)                                         AS total_emi_dues,
    SUM(emi_amount)                                  AS total_emi_billed,
    SUM(paid_amount)                                 AS total_collected,
    ROUND(SUM(paid_amount) / NULLIF(SUM(emi_amount), 0) * 100, 2)
                                                     AS collection_rate_pct,
    SUM(CASE WHEN payment_status = 'Missed' THEN 1 ELSE 0 END)
                                                     AS missed_payments,
    SUM(CASE WHEN payment_status = 'Late'   THEN 1 ELSE 0 END)
                                                     AS late_payments
FROM repayment_history
GROUP BY 1
ORDER BY 1 DESC;


-- ─────────────────────────────────────────────────────────
-- Q4. Credit Utilisation Risk Distribution
-- KPI: % of CC customers exceeding safe utilisation thresholds
-- ─────────────────────────────────────────────────────────
SELECT
    CASE
        WHEN cc_balance / NULLIF(credit_limit, 0) < 0.30 THEN 'Low (< 30%)'
        WHEN cc_balance / NULLIF(credit_limit, 0) < 0.60 THEN 'Medium (30–60%)'
        WHEN cc_balance / NULLIF(credit_limit, 0) < 0.90 THEN 'High (60–90%)'
        ELSE 'Very High (> 90%)'
    END                                              AS utilisation_band,
    COUNT(*)                                         AS customer_count,
    ROUND(AVG(defaulted) * 100, 2)                  AS default_rate_pct,
    ROUND(AVG(credit_score), 0)                      AS avg_credit_score
FROM customers
WHERE has_credit_card = 1
GROUP BY 1
ORDER BY default_rate_pct DESC;


-- ─────────────────────────────────────────────────────────
-- Q5. City-Level NPA Portfolio Exposure
-- KPI: Geographic concentration of NPA risk by city
-- ─────────────────────────────────────────────────────────
SELECT
    city,
    COUNT(*)                                         AS total_customers,
    SUM(loan_amount)                                 AS total_loan_exposure,
    SUM(CASE WHEN defaulted = 1 THEN loan_amount ELSE 0 END)
                                                     AS npa_exposure,
    ROUND(
        SUM(CASE WHEN defaulted = 1 THEN loan_amount ELSE 0 END)
        / NULLIF(SUM(loan_amount), 0) * 100, 2
    )                                                AS npa_rate_pct,
    ROUND(AVG(credit_score), 0)                      AS avg_credit_score
FROM customers
WHERE has_loan = 1
GROUP BY city
ORDER BY npa_exposure DESC;


-- ─────────────────────────────────────────────────────────
-- Q6. Spending Category Analysis (Transaction Level)
-- KPI: Revenue by transaction category — product opportunity mapping
-- ─────────────────────────────────────────────────────────
SELECT
    category,
    transaction_type,
    COUNT(*)                                         AS txn_count,
    ROUND(SUM(amount), 2)                            AS total_volume,
    ROUND(AVG(amount), 2)                            AS avg_txn_amount,
    ROUND(STDDEV(amount), 2)                         AS txn_amount_stddev
FROM transactions
GROUP BY category, transaction_type
ORDER BY total_volume DESC;


-- ─────────────────────────────────────────────────────────
-- Q7. Customer Lifetime Value Distribution by Segment
-- KPI: Revenue contribution per CLV tier
-- ─────────────────────────────────────────────────────────
SELECT
    clv_segment,
    COUNT(*)                                         AS customer_count,
    ROUND(AVG(estimated_clv), 2)                     AS avg_clv,
    ROUND(SUM(estimated_clv), 2)                     AS total_clv,
    ROUND(SUM(estimated_clv) / SUM(SUM(estimated_clv)) OVER () * 100, 2)
                                                     AS pct_of_total_clv,
    ROUND(AVG(default_probability) * 100, 2)         AS avg_default_prob_pct
FROM clv_estimates
GROUP BY clv_segment
ORDER BY avg_clv DESC;


-- ─────────────────────────────────────────────────────────
-- Q8. Rolling 3-Month Default Trend (Window Function)
-- KPI: Trend detection for portfolio stress monitoring
-- ─────────────────────────────────────────────────────────
WITH monthly_defaults AS (
    SELECT
        DATE_TRUNC('month', due_date::DATE)          AS month,
        COUNT(DISTINCT r.customer_id)                AS defaulting_customers
    FROM repayment_history r
    WHERE payment_status = 'Missed'
    GROUP BY 1
)
SELECT
    month,
    defaulting_customers,
    ROUND(AVG(defaulting_customers)
          OVER (ORDER BY month ROWS BETWEEN 2 PRECEDING AND CURRENT ROW), 1)
                                                     AS rolling_3m_avg
FROM monthly_defaults
ORDER BY month;


-- ─────────────────────────────────────────────────────────
-- Q9. Cohort Retention Analysis — Loan Repayment by Tenure
-- KPI: Repayment behaviour improves with customer tenure
-- ─────────────────────────────────────────────────────────
SELECT
    CASE
        WHEN c.tenure_months < 12  THEN '0–12 months'
        WHEN c.tenure_months < 36  THEN '1–3 years'
        WHEN c.tenure_months < 60  THEN '3–5 years'
        ELSE '5+ years'
    END                                              AS tenure_band,
    COUNT(DISTINCT c.customer_id)                    AS customers,
    ROUND(AVG(
        CASE WHEN r.payment_status = 'On-Time' THEN 1.0 ELSE 0.0 END
    ) * 100, 2)                                      AS on_time_rate_pct,
    ROUND(AVG(c.credit_score), 0)                    AS avg_credit_score
FROM customers       c
JOIN repayment_history r ON r.customer_id = c.customer_id
GROUP BY 1
ORDER BY on_time_rate_pct DESC;


-- ─────────────────────────────────────────────────────────
-- Q10. Top 20 Customers by Revenue Potential & Low Risk
-- KPI: Priority list for upsell / cross-sell campaigns
-- ─────────────────────────────────────────────────────────
SELECT
    c.customer_id,
    c.city,
    c.employment_type,
    c.credit_score,
    cv.estimated_clv,
    cv.clv_segment,
    rs.default_probability,
    rs.risk_tier
FROM customers       c
JOIN clv_estimates   cv ON cv.customer_id = c.customer_id
JOIN risk_scored     rs ON rs.customer_id = c.customer_id
WHERE rs.risk_tier   = 'Low Risk'
  AND cv.clv_segment IN ('Gold', 'Platinum')
ORDER BY cv.estimated_clv DESC
LIMIT 20;


-- ─────────────────────────────────────────────────────────
-- Q11. Risk Concentration by Income Band (NEW)
-- KPI: Identifies income strata most vulnerable to default
--      Useful for pre-approval credit policy tuning
-- ─────────────────────────────────────────────────────────
SELECT
    CASE
        WHEN income < 200000   THEN 'Low Income (< ₹2L)'
        WHEN income < 500000   THEN 'Lower Middle (₹2L–5L)'
        WHEN income < 1000000  THEN 'Middle (₹5L–10L)'
        WHEN income < 2000000  THEN 'Upper Middle (₹10L–20L)'
        ELSE 'High Income (> ₹20L)'
    END                                              AS income_band,
    COUNT(*)                                         AS total_customers,
    SUM(defaulted)                                   AS total_defaults,
    ROUND(AVG(defaulted) * 100, 2)                   AS default_rate_pct,
    ROUND(AVG(credit_score), 0)                      AS avg_credit_score,
    ROUND(AVG(loan_amount), 2)                       AS avg_loan_amount,
    ROUND(SUM(CASE WHEN defaulted = 1 THEN loan_amount ELSE 0 END), 2)
                                                     AS npa_exposure,
    ROUND(
        SUM(CASE WHEN defaulted=1 THEN loan_amount ELSE 0 END)
        / NULLIF(SUM(loan_amount),0) * 100, 2
    )                                                AS npa_rate_pct
FROM customers
WHERE has_loan = 1
GROUP BY 1
ORDER BY default_rate_pct DESC;


-- ─────────────────────────────────────────────────────────
-- Q12. Churner Segment Risk & CLV Impact Analysis (NEW)
-- KPI: Customers at risk of disengagement — revenue leakage alert
--      Combines K-Means segment 3 (Churner) with risk and CLV data
--      Enables proactive retention campaign prioritisation
-- ─────────────────────────────────────────────────────────
WITH churner_base AS (
    SELECT
        c.customer_id,
        c.city,
        c.employment_type,
        c.income,
        c.credit_score,
        c.tenure_months,
        rs.default_probability,
        rs.risk_tier,
        cv.estimated_clv,
        cv.clv_segment,
        cs.segment                                   AS kmeans_segment
    FROM customers          c
    JOIN risk_scored        rs ON rs.customer_id = c.customer_id
    JOIN clv_estimates      cv ON cv.customer_id = c.customer_id
    JOIN customer_segments  cs ON cs.customer_id = c.customer_id
    WHERE cs.segment = 3                             -- Churner Watch cluster
)
SELECT
    city,
    COUNT(*)                                         AS churner_count,
    ROUND(AVG(default_probability) * 100, 2)         AS avg_default_prob_pct,
    ROUND(AVG(estimated_clv), 2)                     AS avg_clv,
    ROUND(SUM(estimated_clv), 2)                     AS total_clv_at_risk,
    ROUND(AVG(credit_score), 0)                      AS avg_credit_score,
    ROUND(AVG(income), 2)                            AS avg_income,
    ROUND(AVG(tenure_months), 1)                     AS avg_tenure_months
FROM churner_base
GROUP BY city
ORDER BY total_clv_at_risk DESC;
