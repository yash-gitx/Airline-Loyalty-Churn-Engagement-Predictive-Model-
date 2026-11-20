-- ============================================================================
-- ANALYTICAL BASE TABLE (ABT) FOR LOYALTY CHURN & ENGAGEMENT MODELING
-- Strict Observation Cutoff: 2026-06-30 (Zero Target Leakage)
-- ============================================================================

DROP TABLE IF EXISTS abt_member_features;

CREATE TABLE abt_member_features AS
WITH flight_metrics AS (
    SELECT
        member_id,
        -- Recency (In SQL Server use: DATEDIFF(day, MAX(flight_date), '2026-06-30'))
        CAST(julianday('2026-06-30') - julianday(MAX(flight_date)) AS INTEGER) AS recency_days,

        -- 12-Month Rolling Activity (Jul 2025 - Jun 2026)
        SUM(CASE WHEN flight_date >= '2025-07-01' THEN pqf_earned ELSE 0 END) AS pqf_12m,
        SUM(CASE WHEN flight_date >= '2025-07-01' THEN pqp_earned ELSE 0 END) AS flight_pqp_12m,
        SUM(CASE WHEN flight_date >= '2025-07-01' THEN base_fare_usd ELSE 0 END) AS fare_spend_12m,

        -- Period Split for Engagement Decay Velocity (H1 2026 vs H2 2025)
        SUM(CASE WHEN flight_date >= '2026-01-01' THEN pqf_earned ELSE 0 END) AS pqf_recent_6m,
        SUM(CASE WHEN flight_date >= '2025-07-01' AND flight_date < '2026-01-01' THEN pqf_earned ELSE 0 END) AS pqf_prior_6m,
        SUM(CASE WHEN flight_date >= '2026-01-01' THEN pqp_earned ELSE 0 END) AS pqp_recent_6m,
        SUM(CASE WHEN flight_date >= '2025-07-01' AND flight_date < '2026-01-01' THEN pqp_earned ELSE 0 END) AS pqp_prior_6m,

        -- Operational Disruption Friction (Recent 6M)
        SUM(CASE WHEN flight_date >= '2026-01-01' THEN is_delayed_60m ELSE 0 END) AS delays_recent_6m,
        SUM(CASE WHEN flight_date >= '2026-01-01' THEN is_cancelled ELSE 0 END) AS cancels_recent_6m,

        -- Cabin & Route Mix
        AVG(CASE WHEN cabin_class IN ('Business/First', 'Premium Plus') THEN 1.0 ELSE 0.0 END) AS premium_cabin_share,
        AVG(CASE WHEN route_type = 'International' THEN 1.0 ELSE 0.0 END) AS intl_flight_share
    FROM fact_flights
    WHERE flight_date <= '2026-06-30'
    GROUP BY member_id
),

card_metrics AS (
    SELECT
        member_id,
        SUM(CASE WHEN spend_month >= '2025-07-01' THEN monthly_spend_usd ELSE 0 END) AS card_spend_12m,
        SUM(CASE WHEN spend_month >= '2026-01-01' THEN card_pqp_earned ELSE 0 END) AS card_pqp_recent_6m,
        SUM(CASE WHEN spend_month >= '2025-07-01' AND spend_month < '2026-01-01' THEN card_pqp_earned ELSE 0 END) AS card_pqp_prior_6m
    FROM fact_card_spend
    WHERE spend_month <= '2026-06-01'
    GROUP BY member_id
),

engagement_metrics AS (
    SELECT
        member_id,
        SUM(CASE WHEN event_type = 'Support_Complaint' AND event_date >= '2026-01-01' THEN 1 ELSE 0 END) AS complaints_recent_6m,
        SUM(CASE WHEN event_type = 'Miles_Redemption' AND event_date >= '2025-07-01' THEN 1 ELSE 0 END) AS redemptions_12m,
        SUM(CASE WHEN event_type = 'Miles_Redemption' AND event_date >= '2025-07-01' THEN miles_redeemed ELSE 0 END) AS miles_redeemed_12m,
        AVG(csat_score) AS avg_csat_score
    FROM fact_engagement_events
    WHERE event_date <= '2026-06-30'
    GROUP BY member_id
),

combined_base AS (
    SELECT
        m.member_id,
        m.home_hub,
        m.current_tier,
        m.age_band,
        m.has_cobrand_card,
        m.card_tier,
        m.annual_pqp_target,
        CAST((julianday('2026-06-30') - julianday(m.join_date)) / 365.25 AS REAL) AS tenure_years,

        COALESCE(f.recency_days, 365) AS recency_days,
        COALESCE(f.pqf_12m, 0) AS pqf_12m,
        COALESCE(f.flight_pqp_12m, 0) AS flight_pqp_12m,
        COALESCE(f.fare_spend_12m, 0) AS fare_spend_12m,
        COALESCE(f.pqf_recent_6m, 0) AS pqf_recent_6m,
        COALESCE(f.pqf_prior_6m, 0) AS pqf_prior_6m,
        COALESCE(f.pqp_recent_6m, 0) AS pqp_recent_6m,
        COALESCE(f.pqp_prior_6m, 0) AS pqp_prior_6m,

        -- Velocity & Decay Ratios (Smoothed to avoid zero-division)
        ROUND((COALESCE(f.pqp_recent_6m, 0) + 100.0) / (COALESCE(f.pqp_prior_6m, 0) + 100.0), 3) AS pqp_velocity_ratio,
        COALESCE(f.pqf_recent_6m, 0) - COALESCE(f.pqf_prior_6m, 0) AS pqf_delta_6m,

        -- Annualized PQP Pacing Toward Current Tier Status
        ROUND(((COALESCE(f.pqp_recent_6m, 0) + COALESCE(c.card_pqp_recent_6m, 0)) * 2.0) / m.annual_pqp_target, 3) AS tier_pacing_ratio,

        -- Disruption & Experience Friction
        ROUND(COALESCE(f.delays_recent_6m, 0) * 1.0 / NULLIF(COALESCE(f.pqf_recent_6m, 0), 0), 3) AS delay_rate_recent_6m,
        COALESCE(f.cancels_recent_6m, 0) AS cancels_recent_6m,
        ROUND(COALESCE(f.premium_cabin_share, 0), 3) AS premium_cabin_share,
        ROUND(COALESCE(f.intl_flight_share, 0), 3) AS intl_flight_share,

        -- Co-Brand Card Engagement
        COALESCE(c.card_spend_12m, 0) AS card_spend_12m,
        COALESCE(c.card_pqp_recent_6m, 0) AS card_pqp_recent_6m,
        ROUND((COALESCE(c.card_pqp_recent_6m, 0) + 10.0) / (COALESCE(c.card_pqp_prior_6m, 0) + 10.0), 3) AS card_velocity_ratio,

        -- Loyalty Ecosystem Engagement
        COALESCE(e.complaints_recent_6m, 0) AS complaints_recent_6m,
        COALESCE(e.redemptions_12m, 0) AS redemptions_12m,
        COALESCE(e.miles_redeemed_12m, 0) AS miles_redeemed_12m,
        ROUND(COALESCE(e.avg_csat_score, 4.2), 2) AS avg_csat_score,

        -- Outcome Label & Revenue Exposure
        o.is_churn_or_downgrade,
        o.projected_annual_revenue_usd
    FROM dim_members m
    LEFT JOIN flight_metrics f ON m.member_id = f.member_id
    LEFT JOIN card_metrics c ON m.member_id = c.member_id
    LEFT JOIN engagement_metrics e ON m.member_id = e.member_id
    INNER JOIN fact_member_outcomes o ON m.member_id = o.member_id
)

SELECT
    *,
    -- Window Function: Rank member value within their Premier Tier
    NTILE(4) OVER (PARTITION BY current_tier ORDER BY fare_spend_12m DESC) AS within_tier_value_quartile,
    ROUND(PERCENT_RANK() OVER (PARTITION BY current_tier ORDER BY pqp_velocity_ratio ASC), 3) AS decay_percentile_in_tier
FROM combined_base;

-- ============================================================================
-- MONTHLY COHORT TREND TABLE FOR POWER BI TIME-SERIES VISUALS
-- ============================================================================
DROP TABLE IF EXISTS agg_monthly_trends;

CREATE TABLE agg_monthly_trends AS
SELECT
    substr(f.flight_date, 1, 7) || '-01' AS month_date,
    m.current_tier,
    m.has_cobrand_card,
    COUNT(DISTINCT f.member_id) AS active_flyers,
    SUM(f.pqf_earned) AS total_flights,
    ROUND(SUM(f.base_fare_usd), 2) AS total_fare_revenue,
    SUM(f.pqp_earned) AS total_pqp,
    ROUND(AVG(f.is_delayed_60m), 4) AS monthly_delay_rate
FROM fact_flights f
INNER JOIN dim_members m ON f.member_id = m.member_id
WHERE f.flight_date <= '2026-06-30'
GROUP BY 1, 2, 3;