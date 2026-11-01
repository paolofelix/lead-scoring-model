import numpy as np
import pandas as pd

SEED = 42
N_LEADS = 20_000
START = pd.Timestamp("2025-01-01")
EXTRACT_DATE = pd.Timestamp("2026-06-30")   # the day we "pulled" the CRM export
WINDOW_DAYS = 90                            # target: closed-won within 90 days of lead creation

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def pick(rng, options, probs, size=None):
    return rng.choice(options, p=probs, size=size)

def simulate_leads(n=N_LEADS, seed=SEED):
    rng = np.random.default_rng(seed)

    # ---------- 1. Lead creation dates: growing volume, fewer leads on weekends ----------
    days = pd.date_range(START, EXTRACT_DATE, freq="D")
    w = (1 + 0.6 * np.arange(len(days)) / len(days)) * np.where(days.dayofweek >= 5, 0.35, 1.0)
    day_idx = rng.choice(len(days), size=n, p=w / w.sum())
    hour = pick(rng, np.arange(24),
                np.array([1,1,1,1,1,2,3,5,8,9,9,8,7,8,8,7,6,5,4,3,3,2,2,1], float) / 105, n)
    created_at = days[day_idx] + pd.to_timedelta(hour, "h") + pd.to_timedelta(rng.integers(0, 60, n), "m")
    is_weekend = created_at.dayofweek >= 5

    # ---------- 2. Acquisition source ----------
    channels = ["google_search", "meta", "linkedin", "microsoft", "organic", "referral"]
    channel = pick(rng, channels, [0.32, 0.22, 0.14, 0.08, 0.16, 0.08], n)

    campaign_type = np.full(n, "none", dtype=object)
    search = np.isin(channel, ["google_search", "microsoft"])
    social = np.isin(channel, ["meta", "linkedin"])
    campaign_type[search] = pick(rng, ["brand", "nonbrand"], [0.35, 0.65], search.sum())
    campaign_type[social] = pick(rng, ["prospecting", "retargeting"], [0.70, 0.30], social.sum())

    form_types = ["demo_request", "free_trial", "contact_sales", "content_download"]
    form_probs = {
        "google_search": [0.38, 0.30, 0.17, 0.15], "microsoft": [0.40, 0.28, 0.17, 0.15],
        "meta":          [0.15, 0.30, 0.05, 0.50], "linkedin":  [0.25, 0.15, 0.10, 0.50],
        "organic":       [0.30, 0.35, 0.10, 0.25], "referral":  [0.50, 0.20, 0.25, 0.05],
    }
    form_type = np.array([pick(rng, form_types, form_probs[c]) for c in channel])

    # ---------- 3. Firmographics (with realistic enrichment gaps) ----------
    free_p = np.select([channel == "meta", channel == "linkedin"], [0.40, 0.10], 0.18)
    email_type = np.where(rng.random(n) < free_p, "free", "business")

    sizes = ["1-10", "11-50", "51-200", "201-1000", "1000+"]
    size_probs = {"linkedin": [0.12, 0.22, 0.26, 0.24, 0.16], "meta": [0.40, 0.30, 0.16, 0.10, 0.04]}
    default_sizes = [0.30, 0.28, 0.20, 0.14, 0.08]
    company_size = np.array([pick(rng, sizes, size_probs.get(c, default_sizes)) for c in channel], dtype=object)

    industries = ["saas_tech", "finance", "healthcare", "retail_ecom", "manufacturing", "education", "other"]
    industry = pick(rng, industries, [0.26, 0.14, 0.12, 0.14, 0.12, 0.08, 0.14], n).astype(object)

    # Enrichment (e.g. Clearbit-style lookup) fails far more often on free email addresses
    enrich_fail = rng.random(n) < np.where(email_type == "free", 0.55, 0.05)
    company_size[enrich_fail] = np.nan
    industry[enrich_fail] = np.nan

    job_levels = ["c_level", "vp_director", "manager", "individual_contributor", "student_other"]
    job_level = pick(rng, job_levels, [0.08, 0.17, 0.30, 0.33, 0.12], n).astype(object)
    job_level[rng.random(n) < 0.12] = np.nan          # optional form field left blank

    region = pick(rng, ["NA", "EU", "APAC", "LATAM"], [0.55, 0.25, 0.12, 0.08], n)

    # ---------- 4. On-site behaviour before converting ----------
    lam = 2.0 + 1.5 * (campaign_type == "brand") + 1.0 * (form_type == "demo_request") \
          + 1.0 * (channel == "referral") - 0.8 * (form_type == "content_download")
    pages_viewed = 1 + rng.poisson(np.clip(lam, 0.5, None))
    visited_pricing = (rng.random(n) < sigmoid(-2.2 + 0.45 * pages_viewed)).astype(int)
    time_on_site_sec = np.round(np.exp(rng.normal(np.log(150) + 0.12 * pages_viewed, 0.8))).astype(int)

    # Speed-to-lead: known only AFTER the lead arrives -> NOT usable for scoring at creation time
    off_hours = is_weekend | (hour < 8) | (hour >= 18)
    first_response_hours = np.round(np.exp(rng.normal(np.log(3) + 1.2 * off_hours, 0.9)), 1)

    # ---------- 5. TRUE data-generating process for closed-won within 90 days ----------
    eff = lambda arr, d, default=0.0: np.array([d.get(v, default) if isinstance(v, str) else default for v in arr])
    size_known = pd.notna(company_size)
    mid_large = np.isin(company_size.astype(str), ["51-200", "201-1000", "1000+"])

    logit = (
        -4.4
        + eff(form_type, {"demo_request": 1.0, "contact_sales": 0.8, "free_trial": 0.3, "content_download": -0.9})
        + eff(np.char.add(channel.astype(str), "|" + campaign_type.astype(str)), {
            "google_search|brand": 0.5, "microsoft|brand": 0.6, "microsoft|nonbrand": 0.1,
            "linkedin|prospecting": 0.3, "linkedin|retargeting": 0.5,
            "meta|prospecting": -0.5, "meta|retargeting": 0.1,
            "organic|none": 0.2, "referral|none": 0.9})
        + np.where(size_known, eff(company_size, {"1-10": -0.6, "51-200": 0.4, "201-1000": 0.6, "1000+": 0.3}), -0.2)
        + eff(job_level, {"c_level": 0.6, "vp_director": 0.7, "manager": 0.3, "student_other": -1.2}, default=-0.2)
        - 0.8 * (email_type == "free")
        + eff(industry, {"saas_tech": 0.3, "finance": 0.2, "retail_ecom": 0.1, "manufacturing": -0.1,
                         "education": -0.4, "other": -0.2}, default=-0.2)
        + eff(region, {"NA": 0.2, "APAC": -0.3, "LATAM": -0.4})
        + 0.6 * np.log1p(pages_viewed)                                         # diminishing returns
        - 0.3 * (np.log(time_on_site_sec / 60) - np.log(6)) ** 2                # sweet spot ~6 minutes
        + 0.5 * visited_pricing
        + 0.5 * visited_pricing * mid_large                                     # interaction
        - 0.35 * np.log1p(first_response_hours)                                 # slow follow-up hurts
        - 0.2 * is_weekend
    )
    true_prob = sigmoid(logit)
    won = rng.random(n) < true_prob

    # ---------- 6. Funnel outcomes ----------
    is_sql = won | (rng.random(n) < np.clip(2.5 * true_prob + 0.04, 0, 0.9))
    size_slow = eff(company_size, {"201-1000": 10, "1000+": 20})
    days_to_close = np.clip(np.round(rng.gamma(4, 9, n) + size_slow), 7, WINDOW_DAYS).astype(int)
    close_date = pd.Series(created_at + pd.to_timedelta(days_to_close, "D"))

    acv_median = eff(company_size, {"1-10": 3_000, "11-50": 8_000, "51-200": 18_000,
                                    "201-1000": 40_000, "1000+": 90_000}, default=6_000)
    acv = np.round(acv_median * np.exp(rng.normal(0, 0.45, n)), -2)

    # Only leads older than 90 days at extract time have a final label
    outcome_matured = (created_at <= EXTRACT_DATE - pd.Timedelta(days=WINDOW_DAYS))
    observed_won = won & (close_date.values <= EXTRACT_DATE)

    leads = pd.DataFrame({
        "lead_id": [f"L{100000 + i}" for i in range(n)],
        "created_at": created_at,
        "channel": channel, "campaign_type": campaign_type, "form_type": form_type,
        "email_type": email_type, "company_size": company_size, "industry": industry,
        "job_level": job_level, "region": region,
        "pages_viewed": pages_viewed, "visited_pricing": visited_pricing,
        "time_on_site_sec": time_on_site_sec,
        "first_response_hours": first_response_hours,
        "is_sql": is_sql.astype(int),
        "closed_won": observed_won.astype(int),
        "close_date": close_date.where(observed_won),
        "acv_usd": np.where(observed_won, acv, np.nan),
        "outcome_matured": outcome_matured.astype(int),
    }).sort_values("created_at").reset_index(drop=True)

    truth = pd.DataFrame({"lead_id": [f"L{100000 + i}" for i in range(n)],
                          "true_logit": logit, "true_prob": true_prob,
                          "will_win": won.astype(int), "potential_acv_usd": acv})
    return leads, truth
