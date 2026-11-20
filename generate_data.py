import os
import sqlite3
import numpy as np
import pandas as pd

SEED = 42
rng = np.random.default_rng(SEED)

os.makedirs("data", exist_ok=True)
DB_PATH = "data/loyalty_analytics.db"


def generate_database(n_members: int = 5000):
    # 1. DIM_MEMBERS
    tiers = ["General", "Premier Silver", "Premier Gold", "Premier Platinum", "Premier 1K"]
    tier_probs = [0.42, 0.28, 0.16, 0.09, 0.05]
    pqp_targets = {
        "General": 3000,
        "Premier Silver": 6000,
        "Premier Gold": 12000,
        "Premier Platinum": 18000,
        "Premier 1K": 28000,
    }
    hubs = ["ORD", "EWR", "IAH", "DEN", "SFO", "IAD", "LAX"]
    age_bands = ["22-34", "35-49", "50-64", "65+"]

    member_ids = [f"MP{100000 + i}" for i in range(n_members)]
    member_tiers = rng.choice(tiers, size=n_members, p=tier_probs)
    
    # Higher tiers have higher co-brand credit card adoption
    card_prob_map = {
        "General": 0.30, "Premier Silver": 0.52, "Premier Gold": 0.68,
        "Premier Platinum": 0.78, "Premier 1K": 0.85,
    }
    has_card = np.array([rng.random() < card_prob_map[t] for t in member_tiers], dtype=int)
    card_tier = np.where(
        has_card == 1,
        rng.choice(["Explorer", "Quest", "Club Infinite"], size=n_members, p=[0.50, 0.32, 0.18]),
        "None"
    )

    join_offsets = rng.integers(365, 3650, size=n_members)
    join_dates = (pd.Timestamp("2025-01-01") - pd.to_timedelta(join_offsets, unit="D")).strftime("%Y-%m-%d")

    dim_members = pd.DataFrame({
        "member_id": member_ids,
        "join_date": join_dates,
        "home_hub": rng.choice(hubs, size=n_members),
        "current_tier": member_tiers,
        "annual_pqp_target": [pqp_targets[t] for t in member_tiers],
        "has_cobrand_card": has_card,
        "card_tier": card_tier,
        "age_band": rng.choice(age_bands, size=n_members, p=[0.24, 0.38, 0.28, 0.10]),
    })

    # Latent propensity for engagement decay in H1 2026
    base_decay_prob = 0.28 - 0.12 * has_card + np.where(member_tiers == "Premier Silver", 0.08, 0.0)
    is_decaying = rng.random(n_members) < np.clip(base_decay_prob, 0.05, 0.55)

    # 2. FACT_FLIGHTS (Observation: H1 2025, H2 2025, H1 2026)
    base_semiannual_flights = {
        "General": 2.5, "Premier Silver": 6.5, "Premier Gold": 12.0,
        "Premier Platinum": 17.5, "Premier 1K": 24.0,
    }
    avg_fare_map = {
        "General": 280, "Premier Silver": 410, "Premier Gold": 540,
        "Premier Platinum": 680, "Premier 1K": 890,
    }

    flight_records = []
    periods = [
        ("2025-01-01", "2025-06-30", 1.00),
        ("2025-07-01", "2025-12-31", 1.02),
        ("2026-01-01", "2026-06-30", 0.42),  # Decay multiplier applies only to decaying members
    ]

    for p_idx, (start_str, end_str, decay_mult) in enumerate(periods):
        start_dt = pd.Timestamp(start_str)
        n_days = (pd.Timestamp(end_str) - start_dt).days

        lam = np.array([base_semiannual_flights[t] for t in member_tiers])
        if p_idx == 2:
            lam = np.where(is_decaying, lam * decay_mult, lam * rng.uniform(0.90, 1.15, n_members))

        counts = rng.poisson(lam=np.clip(lam, 0.2, None))
        rep_idx = np.repeat(np.arange(n_members), counts)
        n_f = len(rep_idx)

        f_dates = (start_dt + pd.to_timedelta(rng.integers(0, n_days + 1, size=n_f), unit="D")).strftime("%Y-%m-%d")
        f_tiers = member_tiers[rep_idx]
        mean_fares = np.array([avg_fare_map[t] for t in f_tiers])
        fares = np.round(rng.lognormal(mean=np.log(mean_fares), sigma=0.35), 2)
        fares = np.clip(fares, 79.0, 6500.0)

        # Decaying members experience slightly higher disruption rates (realistic friction driver)
        delay_prob = np.where(is_decaying[rep_idx], 0.24, 0.11)
        is_delayed = (rng.random(n_f) < delay_prob).astype(int)
        is_cancelled = (rng.random(n_f) < np.where(is_delayed == 1, 0.14, 0.01)).astype(int)

        cabin = np.where(fares > 1100, "Business/First", np.where(fares > 520, "Premium Plus", "Economy"))
        route_type = np.where(fares > 750, "International", "Domestic")

        flight_records.append(pd.DataFrame({
            "member_id": np.array(member_ids)[rep_idx],
            "flight_date": f_dates,
            "route_type": route_type,
            "cabin_class": cabin,
            "base_fare_usd": fares,
            "pqp_earned": np.round(fares * 0.88, 0).astype(int),
            "pqf_earned": np.where(is_cancelled == 1, 0, 1),
            "is_delayed_60m": is_delayed,
            "is_cancelled": is_cancelled,
        }))

    fact_flights = pd.concat(flight_records, ignore_index=True)
    fact_flights.insert(0, "flight_id", np.arange(1, len(fact_flights) + 1))

    # 3. FACT_CARD_SPEND (Monthly statements Jan 2025 - Jun 2026 for cardholders)
    months = pd.date_range("2025-01-01", "2026-06-01", freq="MS").strftime("%Y-%m-%d").tolist()
    card_indices = np.where(has_card == 1)[0]
    c_rep = np.repeat(card_indices, len(months))
    m_rep = np.tile(months, len(card_indices))

    base_spend = rng.uniform(1200, 5500, size=n_members)[c_rep]
    is_recent = np.array([m >= "2026-01-01" for m in m_rep])
    spend_mult = np.where(is_recent & is_decaying[c_rep], rng.uniform(0.45, 0.75, len(c_rep)), rng.uniform(0.85, 1.20, len(c_rep)))
    monthly_spend = np.round(base_spend * spend_mult, 2)

    fact_card_spend = pd.DataFrame({
        "spend_id": np.arange(1, len(c_rep) + 1),
        "member_id": np.array(member_ids)[c_rep],
        "spend_month": m_rep,
        "monthly_spend_usd": monthly_spend,
        "card_pqp_earned": (monthly_spend // 20).astype(int),
        "miles_earned": (monthly_spend * 2.0).astype(int),
    })

    # 4. FACT_ENGAGEMENT_EVENTS (Redemptions & Support Tickets Jan 2025 - Jun 2026)
    n_events = 16000
    ev_member_idx = rng.integers(0, n_members, size=n_events)
    ev_dates = (pd.Timestamp("2025-01-01") + pd.to_timedelta(rng.integers(0, 545, size=n_events), unit="D")).strftime("%Y-%m-%d")
    complaint_prob = np.where(is_decaying[ev_member_idx], 0.45, 0.18)
    is_complaint = rng.random(n_events) < complaint_prob

    fact_engagement_events = pd.DataFrame({
        "event_id": np.arange(1, n_events + 1),
        "member_id": np.array(member_ids)[ev_member_idx],
        "event_date": ev_dates,
        "event_type": np.where(is_complaint, "Support_Complaint", "Miles_Redemption"),
        "miles_redeemed": np.where(is_complaint, 0, rng.integers(10000, 85000, size=n_events)),
        "csat_score": np.where(is_complaint, rng.integers(1, 4, size=n_events), rng.integers(4, 6, size=n_events)),
    })

    # 5. FACT_MEMBER_OUTCOMES (Strict Future Outcome Window: H2 2026)
    # Derive realistic outcome probabilities driven by behavioral signals + noise
    h1_2026_flights = fact_flights[fact_flights["flight_date"] >= "2026-01-01"].groupby("member_id").agg(
        f_h1=("pqf_earned", "sum"),
        pqp_h1=("pqp_earned", "sum"),
        delays_h1=("is_delayed_60m", "sum"),
    )
    h2_2025_flights = fact_flights[
        (fact_flights["flight_date"] >= "2025-07-01") & (fact_flights["flight_date"] < "2026-01-01")
    ].groupby("member_id").agg(pqp_prev=("pqp_earned", "sum"))

    outcome_df = dim_members[["member_id", "current_tier", "annual_pqp_target", "has_cobrand_card"]].copy()
    outcome_df = outcome_df.merge(h1_2026_flights, on="member_id", how="left").merge(h2_2025_flights, on="member_id", how="left").fillna(0)

    pqp_velocity = (outcome_df["pqp_h1"] + 1) / (outcome_df["pqp_prev"] + 1)
    delay_rate = outcome_df["delays_h1"] / np.maximum(outcome_df["f_h1"], 1)
    pacing_ratio = (outcome_df["pqp_h1"] * 2) / outcome_df["annual_pqp_target"]

    # Logistic data-generating process for churn / tier downgrade
    logit = (
        -0.45
        - 1.65 * np.clip(pqp_velocity - 0.85, -1.0, 1.5)
        - 0.95 * outcome_df["has_cobrand_card"]
        + 2.10 * delay_rate
        - 1.15 * np.clip(pacing_ratio - 0.75, -1.0, 1.5)
        + rng.normal(0, 0.85, size=n_members)
    )
    churn_prob = 1 / (1 + np.exp(-logit))
    is_churn = (rng.random(n_members) < churn_prob).astype(int)

    fact_member_outcomes = pd.DataFrame({
        "member_id": outcome_df["member_id"],
        "outcome_period": "2026-H2",
        "is_churn_or_downgrade": is_churn,
        "projected_annual_revenue_usd": np.round(np.maximum(outcome_df["pqp_prev"] * 2.15, 650.0), 2),
    })

    # Write all 5 tables to SQLite
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    with sqlite3.connect(DB_PATH) as conn:
        dim_members.to_sql("dim_members", conn, index=False)
        fact_flights.to_sql("fact_flights", conn, index=False)
        fact_card_spend.to_sql("fact_card_spend", conn, index=False)
        fact_engagement_events.to_sql("fact_engagement_events", conn, index=False)
        fact_member_outcomes.to_sql("fact_member_outcomes", conn, index=False)

    print(f"Database created at {DB_PATH}")
    print(f"Members: {len(dim_members):,} | Flights: {len(fact_flights):,} | Churn/Downgrade Rate: {is_churn.mean():.1%}")


if __name__ == "__main__":
    generate_database()