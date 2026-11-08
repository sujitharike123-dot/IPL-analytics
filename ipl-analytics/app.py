"""IPL Analytics dashboard (Streamlit).

Run locally:   streamlit run app.py
Data:          data/processed/*.csv   (built by `python src/prepare_data.py`; committed so the app deploys as-is)
Model:         models/win_model.json  (built by `python src/train_win_model.py`)
"""
import json
import os

import numpy as np
import pandas as pd
import streamlit as st

import ipl_charts as ch
from ipl_utils import HYD, ROOT, load_win_model, predict_win_prob, wilson_ci

st.set_page_config(page_title="IPL Analytics 2008-2024", page_icon=":cricket_bat_and_ball:", layout="wide")


# ----------------------------------------------------------------------------- data
@st.cache_data
def load():
    p = os.path.join(ROOT, "data", "processed")
    matches = pd.read_csv(os.path.join(p, "matches_clean.csv"), parse_dates=["date"])
    innings = pd.read_csv(os.path.join(p, "innings_summary.csv"), parse_dates=["date"])
    bat = pd.read_csv(os.path.join(p, "batting_by_phase.csv"))
    bowl = pd.read_csv(os.path.join(p, "bowling_by_phase.csv"))
    states = pd.read_csv(os.path.join(p, "chase_states.csv"), parse_dates=["date"])
    with open(os.path.join(ROOT, "results", "model_metrics.json")) as f:
        metrics = json.load(f)
    return matches, innings, bat, bowl, states, metrics


matches, innings, bat, bowl, states, metrics = load()
model = load_win_model()
YEARS = sorted(matches["year"].unique())

# ----------------------------------------------------------------------------- sidebar
st.sidebar.title("IPL Analytics")
st.sidebar.caption("1,095 matches and 260,920 deliveries, 2008-2024. Built from ball-by-ball data with SQL, Pandas and a "
                   "chase win-probability model.")
y0, y1 = st.sidebar.select_slider("Seasons", options=YEARS, value=(YEARS[0], YEARS[-1]))
st.sidebar.markdown("The season filter applies to the Overview, Teams & grounds and Players tabs.")
st.sidebar.markdown("**Definitions.** Regular match = no rain rule, no result excluded. Powerplay = overs 1-6, middle = 7-15, "
                    "death = 16-20. Super overs are excluded from all innings stats.")

m = matches[matches["year"].between(y0, y1)]
inn = innings[innings["year"].between(y0, y1)]
reg_m = m[(~m["is_dl"]) & (~m["no_result"])]
reg_i = inn[(inn["is_dl"] == 0) & (inn["no_result"] == 0)]


def show(fig):
    st.pyplot(fig, use_container_width=True)


def pct_str(k, n):
    lo, hi = wilson_ci(k, n)
    return f"{100 * k / n:.1f}%  (95% CI {lo:.0f}-{hi:.0f}%, n={n})"


st.title("IPL Analytics, 2008-2024")
tab_over, tab_team, tab_play, tab_pred = st.tabs(["Overview", "Teams & grounds", "Players", "Chase predictor"])

# ============================================================================ overview
with tab_over:
    f1 = reg_i[reg_i["inning"] == 1]
    dec = m[m["decided"]]
    chased = dec[dec["chase_won"].notna()]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Matches", f"{len(m):,}")
    c2.metric("Avg first-innings score", f"{f1['runs'].mean():.0f}" if len(f1) else "-")
    c3.metric("Chasing team wins", f"{100 * chased['chase_won'].mean():.1f}%" if len(chased) else "-")
    c4.metric("Sixes per match", f"{inn['sixes'].sum() / len(m):.1f}" if len(m) else "-")
    st.markdown("**What the data says (all seasons).** Winning the toss barely matters: toss winners won 50.8% of 1,090 decided "
                "matches. Scoring has jumped since 2023, from an average first-innings total of 163 in 2008-22 to 186 in 2023-24. "
                "Chasing teams won 55.1% of matches before 2023 and 47.9% since, but with only 144 recent matches that gap is "
                "not statistically significant (p = 0.13). A fast powerplay goes with winning. Full numbers and confidence "
                "intervals are in the README.")
    if len(YEARS) and y1 > y0:
        a, b = st.columns(2)
        with a:
            r = (chased.groupby("year")["chase_won"].agg(["sum", "count"]).reset_index())
            r["chase_win_pct"] = 100 * r["sum"] / r["count"]
            show(ch.chart_chase_by_year(r[["year", "chase_win_pct"]]))
        with b:
            ph = reg_i.groupby("year").apply(lambda g: pd.Series({
                "powerplay_rpo": 6 * g["pp_runs"].sum() / g["pp_balls"].sum(),
                "middle_rpo": 6 * g["mid_runs"].sum() / g["mid_balls"].sum(),
                "death_rpo": 6 * g["death_runs"].sum() / g["death_balls"].sum()})).reset_index()
            show(ch.chart_phase_rates(ph))
        a, b = st.columns(2)
        with a:
            av = f1.groupby("year")["runs"].mean().reset_index().rename(columns={"runs": "avg_first_innings"})
            show(ch.chart_avg_first_innings(av))
        with b:
            t = (reg_i.assign(big=(reg_i["runs"] >= 200)).groupby("year")["big"].sum().reset_index()
                 .rename(columns={"big": "scores_of_200_plus"}))
            show(ch.chart_200_plus(t))
    else:
        st.info("Pick a range of two or more seasons in the sidebar to see the trend charts.")

# ============================================================================ teams & grounds
with tab_team:
    teams = sorted(set(m["team1"]).union(m["team2"]))
    default = teams.index("Sunrisers Hyderabad") if "Sunrisers Hyderabad" in teams else 0
    team = st.selectbox("Team", teams, index=default)
    tm = m[((m["team1"] == team) | (m["team2"] == team)) & m["decided"]]
    if len(tm) == 0:
        st.info("This team has no decided matches in the selected seasons.")
    else:
        wins = int((tm["winner"] == team).sum())
        hm = tm[tm["home_team"] == team]
        am = tm[tm["away_team"] == team]
        c1, c2, c3 = st.columns(3)
        c1.metric("Overall win rate", f"{100 * wins / len(tm):.1f}%", f"{wins} wins in {len(tm)} matches", delta_color="off")
        if len(hm) and len(am):
            hw, aw = int((hm["winner"] == team).sum()), int((am["winner"] == team).sum())
            c2.metric("At home", f"{100 * hw / len(hm):.1f}%", f"{hw} of {len(hm)}", delta_color="off")
            c3.metric("Away", f"{100 * aw / len(am):.1f}%", f"{aw} of {len(am)}", delta_color="off")
            st.caption(f"Home: {pct_str(hw, len(hm))}. Away: {pct_str(aw, len(am))}. Home and away are only counted in the "
                       "seasons played in a true home-and-away format (2008, 2010-19, 2023-24).")
        else:
            c2.metric("At home", "-")
            c3.metric("Away", "-")
            st.caption("No true home-and-away matches in the selected seasons (2009, 2020-22 were played at neutral venues).")
        by = (tm.assign(win=(tm["winner"] == team)).groupby("year")["win"].agg(["sum", "count"]).reset_index())
        by = by.rename(columns={"sum": "wins", "count": "matches"})
        by["win_pct"] = 100 * by["wins"] / by["matches"]
        show(ch.chart_team_seasons(by, team))

    st.subheader("Ground guide")
    min_n = st.slider("Minimum regular matches at the ground", 10, 60, 20, step=5)
    g = (reg_m.merge(reg_i[reg_i["inning"] == 1][["match_id", "runs"]].rename(columns={"runs": "f_runs"}), on="match_id")
         .groupby("venue").agg(matches=("match_id", "count"), avg_first_innings=("f_runs", "mean"),
                               chase_win_pct=("chase_won", lambda s: 100 * s.mean())).reset_index())
    g = g[g["matches"] >= min_n]
    if len(g):
        a, b = st.columns(2)
        with a:
            show(ch.chart_venues(g, highlight=HYD, subtitle=f"Grounds with {min_n}+ regular matches"))
        with b:
            show(ch.chart_chase_by_venue(g, highlight=HYD, min_n=min_n))
        st.caption("Uppal (Rajiv Gandhi International Stadium, Hyderabad) is highlighted.")
    else:
        st.info("No ground has that many regular matches in the selected seasons. Lower the minimum.")

# ============================================================================ players
with tab_play:
    who = st.radio("Look at", ["Batters", "Bowlers"], horizontal=True)
    phase = st.selectbox("Phase of the innings", ["All overs", "Powerplay", "Middle", "Death"])
    if who == "Batters":
        b = bat[bat["year"].between(y0, y1)]
        if phase != "All overs":
            b = b[b["phase"] == phase]
        p = (b.groupby("player").agg(runs=("runs", "sum"), balls=("balls", "sum"), fours=("fours", "sum"),
                                     sixes=("sixes", "sum"), dismissals=("dismissals", "sum")).reset_index())
        top_balls = int(p["balls"].max()) if len(p) else 100
        min_balls = st.slider("Minimum balls faced", 50, max(top_balls // 2, 100), min(300, max(top_balls // 2, 100)), step=50)
        p = p[p["balls"] >= min_balls].copy()
        p["strike_rate"] = (100 * p["runs"] / p["balls"]).round(1)
        p["average"] = (p["runs"] / p["dismissals"].replace(0, np.nan)).round(1)
        metric = st.radio("Rank by", ["Runs", "Strike rate"], horizontal=True)
        col = "runs" if metric == "Runs" else "strike_rate"
        top = p.sort_values(col, ascending=False).head(15)
        if len(top):
            show(ch.chart_top_players(top, "player", col, f"{metric}: top batters ({phase.lower()})",
                                      f"At least {min_balls} balls faced, {y0}-{y1}", fmt="{:,.0f}" if col == "runs" else "{:.1f}"))
            st.dataframe(top[["player", "runs", "balls", "strike_rate", "average", "fours", "sixes"]].reset_index(drop=True),
                         use_container_width=True, hide_index=True)
        else:
            st.info("No batter meets the minimum. Lower it.")
    else:
        b = bowl[bowl["year"].between(y0, y1)]
        if phase != "All overs":
            b = b[b["phase"] == phase]
        p = (b.groupby("player").agg(balls=("balls", "sum"), runs=("runs", "sum"), wickets=("wickets", "sum"),
                                     dots=("dots", "sum")).reset_index())
        top_balls = int(p["balls"].max()) if len(p) else 100
        min_balls = st.slider("Minimum balls bowled", 60, max(top_balls // 2, 120), min(300, max(top_balls // 2, 120)), step=30)
        p = p[p["balls"] >= min_balls].copy()
        p["economy"] = (6 * p["runs"] / p["balls"]).round(2)
        p["dot_ball_pct"] = (100 * p["dots"] / p["balls"]).round(1)
        metric = st.radio("Rank by", ["Economy (lowest first)", "Wickets"], horizontal=True)
        top = (p.sort_values("economy").head(15) if metric.startswith("Economy") else p.sort_values("wickets", ascending=False).head(15))
        col = "economy" if metric.startswith("Economy") else "wickets"
        if len(top):
            show(ch.chart_top_players(top, "player", col, f"{'Economy rate' if col == 'economy' else 'Wickets'}: top bowlers ({phase.lower()})",
                                      f"At least {min_balls} balls bowled, {y0}-{y1}", fmt="{:.2f}" if col == "economy" else "{:,.0f}"))
            st.dataframe(top[["player", "balls", "runs", "wickets", "economy", "dot_ball_pct"]].reset_index(drop=True),
                         use_container_width=True, hide_index=True)
        else:
            st.info("No bowler meets the minimum. Lower it.")
    st.caption("Wickets count bowler-credited dismissals only (no run-outs). Retired-hurt is not counted as a dismissal.")

# ============================================================================ chase predictor
with tab_pred:
    st.markdown("Enter the state of a chase at the end of an over and the model estimates the chasing team's chance of winning.")
    c1, c2, c3, c4 = st.columns(4)
    target = c1.number_input("Target", 80, 300, 180)
    overs = c2.number_input("Overs completed", 1, 19, 10)
    wk = c3.number_input("Wickets lost", 0, 9, 2)
    score = c4.number_input("Score", 0, 299, 90)
    if score >= target:
        st.success("The target has already been reached.")
    else:
        pr = predict_win_prob(model, target, score, wk, overs)
        need = target - score
        left = 120 - 6 * overs
        a, b, c = st.columns(3)
        a.metric("Chasing team win probability", f"{100 * pr:.0f}%")
        b.metric("Runs needed", need, f"{left} balls left", delta_color="off")
        c.metric("Required run rate", f"{need * 6 / left:.2f}", f"current {score * 6 / (6 * overs):.2f}", delta_color="off")
        lm = metrics["logistic_regression"]
        st.caption(f"Logistic regression on five features (runs needed, balls left, wickets lost, required and current run "
                   f"rate). On the unseen 2023-24 seasons it scores AUC {lm['auc']} (0.5 is a coin-flip) and a Brier score of "
                   f"{lm['brier']} versus {metrics['baseline_historical_rate']['brier']} for always guessing the historical "
                   "chase win rate. It ignores batters, bowlers, pitch and dew, so treat it as an estimate and not a forecast. "
                   "In the 2023-24 seasons the model was slightly too pessimistic about chasing teams.")
        wsweep = pd.DataFrame({"wickets_lost": range(0, 10),
                               "win_probability_pct": [round(100 * predict_win_prob(model, target, score, w, overs), 1) for w in range(10)]})
        st.markdown("**Same score and overs, different wickets lost**")
        st.dataframe(wsweep, hide_index=True, use_container_width=True)

    st.divider()
    st.subheader("Replay a real chase")
    yr = st.selectbox("Season", sorted(states["year"].unique(), reverse=True))
    sy = states[states["year"] == yr]
    mm = (matches[matches["match_id"].isin(sy["match_id"])].sort_values("date"))
    labels = {int(r.match_id): f"{r.date:%d %b %Y}: {r.bat_first} v {r.chase_team} ({'chase won' if r.chase_won == 1 else 'chase lost'})"
              for r in mm.itertuples()}
    mid = st.selectbox("Match", list(labels), format_func=lambda k: labels[k])
    s = sy[sy["match_id"] == mid].sort_values("overs_done")
    s = s.assign(prob=[predict_win_prob(model, r.target_runs, r.score, r.wickets_lost, r.overs_done) for r in s.itertuples()])
    curve = s[["overs_done", "prob"]].rename(columns={"overs_done": "over_label"})
    curve["over_label"] = curve["over_label"].astype(str)
    row = mm[mm["match_id"] == mid].iloc[0]
    show(ch.chart_win_curve(curve, title=f"{row['chase_team']} chasing {int(row['target_runs'])}: win probability by over",
                            subtitle=f"{row['venue']}. The curve ends when the chase finishes or is out of reach."))
    st.caption("The chart stops at the last over where the chase was still alive (overs 1-19). "
               "Because the model is trained on all seasons, this is a description of how the match looked to the model, "
               "not an out-of-sample test.")
