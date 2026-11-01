"""
Lead Scoring Model - interactive dashboard (Streamlit).

Uses the repo's published outputs (outputs/scored_new_leads.csv, scored by the XGBoost model in
notebooks 03-04) and the simulation's ground truth to show how much each prioritisation strategy
captures, plus segment drivers, a filterable lead queue and a live "score a lead" tool.

The live scorer is a logistic regression retrained inside this app on the matured leads, because
the saved .joblib pipelines are tied to the exact library versions used in Colab.

Run locally:  streamlit run dashboard/app.py      (from the repo root)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR, OUT_DIR = ROOT / "data", ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"
REPO_URL = "https://github.com/paolofelix/lead-scoring-model"

CAT = ["channel", "campaign_type", "form_type", "email_type", "company_size", "industry", "job_level", "region"]
CHANNEL_LABELS = {"google_search": "Google Search", "meta": "Meta", "microsoft": "Microsoft Ads",
                  "organic": "Organic", "referral": "Referral", "linkedin": "LinkedIn"}
CHANNEL_COLORS = {"Google Search": "#2a78d6", "Meta": "#eb6834", "Microsoft Ads": "#1baf7a",
                  "Organic": "#eda100", "Referral": "#e87ba4", "LinkedIn": "#008300"}
STRATEGY_COLORS = {"Model: chance of closing": "#1f4f8f", "Model: expected value": "#6f9bd6",
                   "Rule: demo/contact + business email": "#b7cdea", "Arrival order": "#d6d4ce"}
TIER_COLORS = {"A": "#1f4f8f", "B": "#6f9bd6", "C": "#b7cdea"}
SIZE_ORDER = ["1-10", "11-50", "51-200", "201-1000", "1000+", "missing"]

st.set_page_config(page_title="Lead Scoring Model", page_icon="🎯", layout="wide")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
@st.cache_data
def load():
    # keep_default_na=False so region "NA" (North America) is not read as missing
    leads = pd.read_csv(DATA_DIR / "leads.csv", keep_default_na=False, na_values=[""],
                        parse_dates=["created_at"])
    for c in CAT:
        leads[c] = leads[c].fillna("missing")
    scored = pd.read_csv(OUT_DIR / "scored_new_leads.csv", keep_default_na=False, na_values=[""],
                         parse_dates=["created_at"])
    truth = pd.read_csv(DATA_DIR / "ground_truth" / "true_lead_probabilities.csv")
    comparison = pd.read_csv(OUT_DIR / "model_comparison.csv", index_col=0)

    new = scored.merge(leads.drop(columns=["created_at", "channel", "form_type", "company_size", "job_level"]),
                       on="lead_id", how="left").merge(truth, on="lead_id", how="left")
    new["revenue"] = new["will_win"] * new["potential_acv_usd"]
    new["channel_label"] = new["channel"].map(CHANNEL_LABELS)
    leads["channel_label"] = leads["channel"].map(CHANNEL_LABELS)
    return leads, new, comparison


def wilson(k, n, z=1.96):
    p = k / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return centre - half, centre + half


def add_features(d):
    d = d.copy()
    d["log_pages"] = np.log1p(d["pages_viewed"])
    mins = d["time_on_site_sec"] / 60
    d["log_minutes"] = np.log1p(mins)
    d["log_minutes_sq"] = d["log_minutes"] ** 2          # sweet-spot shape
    d["size_x_pricing"] = d["company_size"] + "|" + d["visited_pricing"].astype(str)
    return d


NUM_FEATS = ["log_pages", "visited_pricing", "log_minutes", "log_minutes_sq"]
CAT_FEATS = CAT + ["size_x_pricing"]


@st.cache_resource
def train_live_model(leads):
    m = add_features(leads[leads["outcome_matured"] == 1])
    X, y = m[CAT_FEATS + NUM_FEATS], m["closed_won"]
    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATS),
                             ("num", StandardScaler(), NUM_FEATS)])
    pipe = Pipeline([("pre", pre), ("lr", LogisticRegression(C=0.5, max_iter=2000))])
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
    pipe.fit(X_tr, y_tr)
    auc = roc_auc_score(y_te, pipe.predict_proba(X_te)[:, 1])
    pipe.fit(X, y)  # refit on all matured leads for scoring
    return pipe, auc


def money(v):
    return f"${v:,.0f}"


leads, new, comparison = load()
matured = leads[leads["outcome_matured"] == 1]
cut_a = new.loc[new["tier"] == "A", "score"].min()
cut_b = new.loc[new["tier"] == "B", "score"].min()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("Lead Scoring: which B2B leads will close?")
st.markdown("Scores inbound SaaS leads by their chance of becoming a closed-won deal within 90 days, using only "
            "what is known when the form is submitted. *Simulated data, so every lead's true chance of "
            "closing is known and the model can be checked against it.*")
st.sidebar.header("About")
st.sidebar.markdown("**Data:** 20,000 simulated leads, Jan 2025 – Jun 2026. 16,071 matured leads trained the "
                    "model; the 3,929 newest leads were scored as new leads and are used on this page, checked "
                    "against the ground truth.")
st.sidebar.markdown(f"[GitHub repo]({REPO_URL}) · [Findings]({REPO_URL}/blob/main/FINDINGS.md)")

k = st.columns(5)
k[0].metric("Leads", f"{len(leads):,}")
k[1].metric("Overall win rate", f"{matured['closed_won'].mean():.1%}", help="Matured leads")
k[2].metric("XGBoost ROC AUC", f"{comparison.loc['XGBoost', 'ROC AUC']:.3f}",
            delta=f"ceiling {comparison.loc['Oracle (true prob.)', 'ROC AUC']:.3f}", delta_color="off", delta_arrow="off")
a = new[new["tier"] == "A"]
k[3].metric("Tier A: share of wins", f"{a['will_win'].sum() / new['will_win'].sum():.0%}",
            help="New leads, checked against ground truth")
k[4].metric("Tier A: true win rate", f"{a['will_win'].mean():.1%}",
            delta=f"predicted {a['score'].mean():.1%}", delta_color="off", delta_arrow="off")

tabs = st.tabs(["Prioritisation", "Lead quality by source", "What drives a win", "Lead queue",
                "Score a lead", "Model performance"])

# ---------------------------------------------------------------------------
# Tab: Prioritisation
# ---------------------------------------------------------------------------
with tabs[0]:
    st.subheader("If sales can only work X% of leads first, which ones should they be?")
    cap = st.slider("Sales capacity (share of new leads worked first)", 5, 60, 20, 5, format="%d%%")
    n_cap = int(round(len(new) * cap / 100))

    by_time = new.sort_values("created_at")
    rule_mask = by_time["form_type"].isin(["demo_request", "contact_sales"]) & (by_time["email_type"] == "business")
    orders = {
        "Model: chance of closing": new.sort_values("score", ascending=False),
        "Model: expected value": new.sort_values("expected_value", ascending=False),
        "Rule: demo/contact + business email": pd.concat([by_time[rule_mask], by_time[~rule_mask]]),
        "Arrival order": by_time,
    }
    tot_w, tot_r = new["will_win"].sum(), new["revenue"].sum()
    rows = []
    for name, o in orders.items():
        top = o.head(n_cap)
        rows.append({"Strategy": name, "Share of wins": top["will_win"].sum() / tot_w,
                     "Share of revenue": top["revenue"].sum() / tot_r, "Wins": int(top["will_win"].sum()),
                     "Revenue won": top["revenue"].sum(), "Win rate of worked leads": top["will_win"].mean()})
    res = pd.DataFrame(rows)

    c1, c2 = st.columns([1, 1])
    with c1:
        long = res.melt(id_vars="Strategy", value_vars=["Share of wins", "Share of revenue"],
                        var_name="Measure", value_name="Share")
        fig = px.bar(long, x="Measure", y="Share", color="Strategy", barmode="group",
                     color_discrete_map=STRATEGY_COLORS, text_auto=".0%")
        fig.update_traces(marker_line_color="white", marker_line_width=2, textposition="outside",
                          hovertemplate="%{fullData.name}<br>%{x}: %{y:.1%}<extra></extra>")
        fig.update_layout(height=420, yaxis=dict(tickformat=".0%", range=[0, 1.08], title=None), xaxis_title=None,
                          legend=dict(orientation="h", y=-0.15, title=None), margin=dict(t=20))
        st.plotly_chart(fig, width="stretch")
    with c2:
        metric = st.radio("Cumulative gains", ["Wins", "Revenue"], horizontal=True)
        col = "will_win" if metric == "Wins" else "revenue"
        fig = go.Figure()
        for name, o in orders.items():
            y = o[col].cumsum().to_numpy() / o[col].sum()
            x = np.arange(1, len(o) + 1) / len(o)
            fig.add_trace(go.Scatter(x=x, y=y, name=name, mode="lines",
                                     line=dict(color=STRATEGY_COLORS[name], width=2),
                                     hovertemplate=f"{name}<br>Top %{{x:.0%}} of leads → %{{y:.0%}} of {metric.lower()}<extra></extra>"))
        fig.add_vline(x=cap / 100, line_dash="dash", line_color="#52514e")
        fig.update_layout(height=380, xaxis=dict(title="Share of leads worked", tickformat=".0%"),
                          yaxis=dict(title=f"Share of {metric.lower()} captured", tickformat=".0%"),
                          showlegend=False, margin=dict(t=10))
        st.plotly_chart(fig, width="stretch")

    st.dataframe(res.set_index("Strategy").style.format({
        "Share of wins": "{:.0%}", "Share of revenue": "{:.0%}", "Revenue won": "${:,.0f}",
        "Win rate of worked leads": "{:.1%}"}), width="stretch")
    st.caption(f"Computed on the {len(new):,} new leads ({n_cap:,} worked first), scored by the XGBoost model "
               "and checked against how each lead really turned out in the simulation. The README's figures use "
               "the 4,018-lead test set, so they differ slightly.")
    st.info("**Read-out:** ranking by chance of closing wins the most deals; ranking by expected value "
            "(chance × typical deal size) wins fewer deals but more revenue. Which one to use is a business "
            "choice: a team paid on deal count and a team paid on revenue should rank differently.")

# ---------------------------------------------------------------------------
# Tab: Lead quality by source
# ---------------------------------------------------------------------------
with tabs[1]:
    st.subheader("Expected first-year revenue per lead, by source")
    src = new.groupby(["channel_label", "campaign_type"]).agg(
        leads=("lead_id", "size"), ev=("expected_value", "mean"),
        a_share=("tier", lambda t: (t == "A").mean())).reset_index()
    wr = matured.groupby(["channel_label", "campaign_type"])["closed_won"].agg(["mean", "size"]).reset_index()
    src = src.merge(wr.rename(columns={"mean": "win_rate", "size": "matured_leads"}),
                    on=["channel_label", "campaign_type"], how="left")
    src["Source"] = src["channel_label"] + np.where(src["campaign_type"] == "none", "",
                                                    " · " + src["campaign_type"])
    src = src.sort_values("ev")
    fig = px.bar(src, y="Source", x="ev", color="channel_label", orientation="h",
                 color_discrete_map=CHANNEL_COLORS, text_auto="$,.0f",
                 custom_data=["a_share", "win_rate", "leads"])
    fig.update_traces(marker_line_color="white", marker_line_width=2,
                      hovertemplate="%{y}<br>Expected value per lead: $%{x:,.0f}<br>A-tier share: "
                                    "%{customdata[0]:.0%}<br>Actual win rate: %{customdata[1]:.1%}<br>"
                                    "New leads: %{customdata[2]}<extra></extra>")
    fig.update_layout(height=480, yaxis=dict(categoryorder="array", categoryarray=src["Source"].tolist(), title=None),
                      xaxis=dict(title="Expected revenue per lead (score × typical deal size)", tickprefix="$"),
                      legend=dict(orientation="h", y=-0.15, title=None), margin=dict(t=10))
    st.plotly_chart(fig, width="stretch")
    st.dataframe(src.sort_values("ev", ascending=False)[["Source", "leads", "ev", "a_share", "win_rate"]]
                 .rename(columns={"leads": "New leads", "ev": "Expected revenue / lead",
                                  "a_share": "Share A-tier", "win_rate": "Actual win rate (matured)"})
                 .style.format({"Expected revenue / lead": "${:,.0f}", "Share A-tier": "{:.0%}",
                                "Actual win rate (matured)": "{:.1%}"}), width="stretch", hide_index=True)

    st.caption("Expected value and A-tier share are computed on the new leads, win rate on matured leads, "
               "so figures differ slightly from FINDINGS.md.")
    st.markdown("**Break-even cost per lead**: what each source can afford to pay per lead at a given return target.")
    target = st.slider("Target revenue per $1 of ad spend", 1.0, 10.0, 3.0, 0.5)
    be = src[src["channel_label"].isin(["Google Search", "Meta", "Microsoft Ads", "LinkedIn"])].copy()
    be["Max cost per lead"] = be["ev"] / target
    st.dataframe(be.sort_values("Max cost per lead", ascending=False)[["Source", "ev", "Max cost per lead"]]
                 .rename(columns={"ev": "Expected revenue / lead"})
                 .style.format({"Expected revenue / lead": "${:,.0f}", "Max cost per lead": "${:,.0f}"}),
                 width="stretch", hide_index=True)
    st.caption("This dataset has no ad cost, so compare these ceilings with each channel's real cost per lead.")

# ---------------------------------------------------------------------------
# Tab: What drives a win
# ---------------------------------------------------------------------------
with tabs[2]:
    st.subheader("Win rate by segment (matured leads, 95% Wilson intervals)")
    labels = {"form_type": "Form type", "email_type": "Email type", "company_size": "Company size",
              "job_level": "Job level", "channel_label": "Channel", "campaign_type": "Campaign type",
              "industry": "Industry", "region": "Region", "visited_pricing": "Visited pricing page"}
    field = st.selectbox("Segment by", list(labels), format_func=labels.get)
    g = matured.groupby(field)["closed_won"].agg(["sum", "size"]).reset_index()
    g["rate"] = g["sum"] / g["size"]
    g["lo"], g["hi"] = wilson(g["sum"], g["size"])
    g[field] = g[field].astype(str)
    g = g.sort_values("rate")
    fig = go.Figure(go.Bar(
        y=g[field], x=g["rate"], orientation="h", marker_color="#2a78d6", marker_line_color="white",
        marker_line_width=2,
        error_x=dict(type="data", symmetric=False, array=g["hi"] - g["rate"], arrayminus=g["rate"] - g["lo"],
                     color="#52514e"),
        customdata=g[["size", "lo", "hi"]],
        hovertemplate="%{y}: %{x:.1%} (95% CI %{customdata[1]:.1%}–%{customdata[2]:.1%})<br>"
                      "%{customdata[0]:,} leads<extra></extra>"))
    fig.add_vline(x=matured["closed_won"].mean(), line_dash="dash", line_color="#52514e",
                  annotation_text="overall", annotation_position="top")
    fig.update_layout(height=max(280, 45 * len(g)), xaxis=dict(title="Win rate", tickformat=".0%"),
                      yaxis_title=None, margin=dict(t=30))
    st.plotly_chart(fig, width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**A pricing-page visit matters most at larger companies**")
        inter = matured.pivot_table(index="company_size", columns="visited_pricing", values="closed_won",
                                    aggfunc="mean").reindex(SIZE_ORDER)
        inter.columns = ["No pricing visit", "Visited pricing"]
        il = inter.reset_index().melt(id_vars="company_size", var_name="Pricing", value_name="Win rate")
        fig = px.bar(il, x="company_size", y="Win rate", color="Pricing", barmode="group",
                     color_discrete_map={"No pricing visit": "#b7cdea", "Visited pricing": "#1f4f8f"},
                     text_auto=".1%")
        fig.update_traces(marker_line_color="white", marker_line_width=2,
                          hovertemplate="%{x} · %{fullData.name}: %{y:.1%}<extra></extra>")
        fig.update_layout(height=360, yaxis_tickformat=".0%", xaxis_title="Company size",
                          legend=dict(orientation="h", y=-0.25, title=None), margin=dict(t=10))
        st.plotly_chart(fig, width="stretch")
    with c2:
        st.markdown("**Speed to lead: a lever, not a model input**")
        bins = [0, 1, 4, 12, 24, np.inf]
        names = ["< 1h", "1–4h", "4–12h", "12–24h", "24h+"]
        sp = matured.assign(bucket=pd.cut(matured["first_response_hours"], bins, labels=names, right=False))
        sp = sp.groupby("bucket", observed=True)["closed_won"].mean().reset_index()
        fig = px.bar(sp, x="bucket", y="closed_won", text_auto=".1%", color_discrete_sequence=["#2a78d6"])
        fig.update_traces(hovertemplate="%{x}: %{y:.1%}<extra></extra>")
        fig.update_layout(height=360, yaxis=dict(title="Win rate", tickformat=".0%"),
                          xaxis_title="Time to first sales response", margin=dict(t=10))
        st.plotly_chart(fig, width="stretch")
        st.caption("Excluded from the score: response time only exists after the lead arrives, so using it "
                   "would leak information. The score says *who* to answer first; answering fast raises the odds.")

# ---------------------------------------------------------------------------
# Tab: Lead queue
# ---------------------------------------------------------------------------
with tabs[3]:
    st.subheader("Scored new leads, ready for sales")
    f = st.columns(4)
    tiers = f[0].multiselect("Tier", ["A", "B", "C"], default=["A"])
    chans = f[1].multiselect("Channel", sorted(new["channel_label"].unique()))
    forms = f[2].multiselect("Form type", sorted(new["form_type"].unique()))
    sort_by = f[3].radio("Rank by", ["Score", "Expected value"], horizontal=True)
    q = new[new["tier"].isin(tiers or ["A", "B", "C"])]
    if chans:
        q = q[q["channel_label"].isin(chans)]
    if forms:
        q = q[q["form_type"].isin(forms)]
    q = q.sort_values("score" if sort_by == "Score" else "expected_value", ascending=False)
    view = q[["lead_id", "created_at", "tier", "score", "expected_value", "channel_label", "form_type",
              "company_size", "job_level", "top_reasons"]].rename(columns={
        "lead_id": "Lead", "created_at": "Created", "tier": "Tier", "score": "Win probability",
        "expected_value": "Expected value", "channel_label": "Channel", "form_type": "Form",
        "company_size": "Company size", "job_level": "Job level", "top_reasons": "Top reasons"})
    st.caption(f"{len(view):,} leads match.")
    st.dataframe(view, width="stretch", hide_index=True, height=480, column_config={
        "Win probability": st.column_config.ProgressColumn(format="%.2f", min_value=0, max_value=1),
        "Expected value": st.column_config.NumberColumn(format="$%,.0f"),
        "Created": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm")})
    st.download_button("Download this list (CSV)", view.to_csv(index=False).encode(), "lead_queue.csv",
                       "text/csv")

# ---------------------------------------------------------------------------
# Tab: Score a lead
# ---------------------------------------------------------------------------
with tabs[4]:
    st.subheader("Score a new lead")
    model, live_auc = train_live_model(leads)
    st.caption(f"A logistic regression retrained in this app on the {len(matured):,} matured leads "
               f"(held-out ROC AUC {live_auc:.3f}; the repo's logistic baseline scored "
               f"{comparison.loc['Logistic regression', 'ROC AUC']:.3f}). Tier cut-offs are the published ones: "
               f"A ≥ {cut_a:.3f}, B ≥ {cut_b:.3f}.")

    def opts(c):
        vals = sorted(v for v in leads[c].unique() if v != "missing")
        return vals + (["missing"] if (leads[c] == "missing").any() else [])

    with st.form("score"):
        r1 = st.columns(4)
        channel = r1[0].selectbox("Channel", opts("channel"), format_func=lambda v: CHANNEL_LABELS.get(v, v))
        camp = r1[1].selectbox("Campaign type", opts("campaign_type"))
        form = r1[2].selectbox("Form type", opts("form_type"), index=opts("form_type").index("demo_request"))
        email = r1[3].selectbox("Email type", opts("email_type"))
        r2 = st.columns(4)
        size = r2[0].selectbox("Company size", [s for s in SIZE_ORDER if s in opts("company_size")], index=3)
        industry = r2[1].selectbox("Industry", opts("industry"))
        job = r2[2].selectbox("Job level", opts("job_level"), index=opts("job_level").index("vp_director"))
        region = r2[3].selectbox("Region", opts("region"))
        r3 = st.columns(3)
        pages = r3[0].slider("Pages viewed", 1, 30, 6)
        minutes = r3[1].slider("Time on site (minutes)", 0.0, 40.0, 5.0, 0.5)
        pricing = r3[2].toggle("Visited pricing page", value=True)
        submitted = st.form_submit_button("Score this lead", type="primary")

    lead = pd.DataFrame([{"channel": channel, "campaign_type": camp, "form_type": form, "email_type": email,
                          "company_size": size, "industry": industry, "job_level": job, "region": region,
                          "pages_viewed": pages, "visited_pricing": int(pricing),
                          "time_on_site_sec": minutes * 60}])
    X1 = add_features(lead)[CAT_FEATS + NUM_FEATS]
    p = float(model.predict_proba(X1)[0, 1])
    tier = "A" if p >= cut_a else "B" if p >= cut_b else "C"
    acv = matured.loc[matured["closed_won"] == 1].groupby("company_size")["acv_usd"].median()
    ev = p * acv.get(size, matured["acv_usd"].median())

    m = st.columns(4)
    m[0].metric("Win probability (90 days)", f"{p:.1%}",
                delta=f"{p / matured['closed_won'].mean():.1f}x average", delta_color="off", delta_arrow="off")
    m[1].metric("Tier", tier)
    m[2].metric("Expected first-year value", money(ev), help="Probability × median deal size for this company size")
    m[3].metric("Suggested action", {"A": "Call within 1h", "B": "Standard follow-up",
                                     "C": "Email nurture"}[tier],
                help={"A": "Senior rep, first response within the hour", "B": "Normal rep queue",
                      "C": "Automated nurture sequence instead of a rep's time"}[tier])

    # Contribution of each input to the log-odds, relative to an average matured lead
    pre, lr = model.named_steps["pre"], model.named_steps["lr"]
    names = pre.get_feature_names_out()
    xb = pre.transform(add_features(matured)[CAT_FEATS + NUM_FEATS])
    x1 = pre.transform(X1)
    xb = np.asarray(xb.mean(axis=0)).ravel()
    x1 = np.asarray(x1.todense() if hasattr(x1, "todense") else x1).ravel()
    contrib = (x1 - xb) * lr.coef_.ravel()
    pretty = {"size_x_pricing": "company size × pricing visit", "log_minutes_sq": "time on site",
              "log_minutes": "time on site", "log_pages": "pages viewed", "visited_pricing": "pricing visit"}
    group = []
    for n in names:
        raw = n.split("__", 1)[1]
        key = next(k_ for k_ in ["size_x_pricing", *CAT, *NUM_FEATS] if raw == k_ or raw.startswith(k_ + "_"))
        group.append(pretty.get(key, key.replace("_", " ")))
    cdf = pd.DataFrame({"Input": group, "Effect": contrib}).groupby("Input")["Effect"].sum().sort_values()
    fig = go.Figure(go.Bar(y=cdf.index, x=cdf.values, orientation="h",
                           marker_color=np.where(cdf.values >= 0, "#1f4f8f", "#eb6834"),
                           hovertemplate="%{y}: %{x:+.2f} log-odds<extra></extra>"))
    fig.add_vline(x=0, line_color="#52514e")
    fig.update_layout(height=380, xaxis_title="Pushes the score up (blue) or down (orange), vs an average lead",
                      yaxis_title=None, margin=dict(t=10))
    st.markdown("**Why this score**")
    st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------------------
# Tab: Model performance
# ---------------------------------------------------------------------------
with tabs[5]:
    st.subheader("Held-out test set (4,018 leads)")
    st.dataframe(comparison.style.format({"ROC AUC": "{:.3f}", "PR AUC": "{:.3f}", "Brier": "{:.4f}",
                                          "Log loss": "{:.4f}", "PR AUC lift vs random": "{:.1f}x"}),
                 width="stretch")
    st.caption("Oracle = the simulation's true probabilities, the best ranking any model could achieve.")

    st.markdown("**Calibration on new leads: predicted vs actual win rate by tier**")
    cal = new.groupby("tier").agg(leads=("lead_id", "size"), predicted=("score", "mean"),
                                  actual=("will_win", "mean"), wins=("will_win", "sum")).reset_index()
    cal["share_of_wins"] = cal["wins"] / cal["wins"].sum()
    fig = go.Figure([
        go.Bar(x=cal["tier"], y=cal["predicted"], name="Predicted", marker_color="#b7cdea",
               marker_line_color="white", marker_line_width=2, text=[f"{v:.1%}" for v in cal["predicted"]],
               textposition="outside"),
        go.Bar(x=cal["tier"], y=cal["actual"], name="Actual (ground truth)", marker_color="#1f4f8f",
               marker_line_color="white", marker_line_width=2, text=[f"{v:.1%}" for v in cal["actual"]],
               textposition="outside")])
    fig.update_layout(barmode="group", height=340, yaxis=dict(title="Win rate", tickformat=".0%"),
                      xaxis_title="Tier", legend=dict(orientation="h", y=-0.2), margin=dict(t=20))
    st.plotly_chart(fig, width="stretch")
    st.dataframe(cal.set_index("tier").style.format({"predicted": "{:.1%}", "actual": "{:.1%}",
                                                     "share_of_wins": "{:.0%}", "wins": "{:.0f}"}),
                 width="stretch")

    imgs = [("06_calibration.png", "Calibration curve"), ("07_cumulative_gains.png", "Cumulative gains"),
            ("08_shap_importance.png", "SHAP feature importance"), ("10_shap_dependence.png", "SHAP dependence")]
    cols = st.columns(2)
    for i, (fname, cap_) in enumerate(imgs):
        if (FIG_DIR / fname).exists():
            cols[i % 2].image(str(FIG_DIR / fname), caption=cap_, width="stretch")

with st.expander("About this dashboard"):
    st.markdown(f"""
- **Data:** 20,000 simulated B2B SaaS leads with built-in problems (enrichment gaps, leakage traps,
  unfinished outcomes, non-linear effects). True win probabilities are in `data/ground_truth/` and are
  used here only to check results.
- **Scores and tiers:** from the XGBoost model in the notebooks (`outputs/scored_new_leads.csv`).
  A = top 20%, B = next 30%, C = bottom 50%, cut-offs set on the test set.
- **Live scorer:** a logistic regression retrained in the app, for what-if scoring.
- **Code:** [{REPO_URL}]({REPO_URL})
""")
