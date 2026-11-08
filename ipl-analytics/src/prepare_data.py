"""Clean the raw IPL files and build the processed tables used by the analysis and the dashboard.

Usage (from the repo root):
    python src/prepare_data.py                  # reads data/raw/matches.csv and data/raw/deliveries.csv
    python src/prepare_data.py path/to/raw_dir  # explicit folder

Outputs (data/processed/):
    matches_clean.csv      one row per match, standardised teams/venues, innings totals, toss/chase flags
    innings_summary.csv    one row per innings (runs, wickets, balls, phase splits, boundaries)
    batting_by_phase.csv   batter x season x team x phase
    bowling_by_phase.csv   bowler x season x team x phase
    data_quality.json      what was cleaned and which validation checks passed
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
from ipl_utils import (HOME_AWAY_YEARS, HOME_GROUND, PHASES, PROC, RAW, clean_team, clean_venue,  # noqa: E402
                       phase_of_over)

raw_dir = sys.argv[1] if len(sys.argv) > 1 else RAW
matches = pd.read_csv(os.path.join(raw_dir, "matches.csv"))
deliv = pd.read_csv(os.path.join(raw_dir, "deliveries.csv"))
os.makedirs(PROC, exist_ok=True)
dq = {"raw_matches": int(len(matches)), "raw_deliveries": int(len(deliv))}
print(f"Loaded {len(matches):,} matches and {len(deliv):,} deliveries")

# ============================================================================ matches
m = matches.rename(columns={"id": "match_id", "season": "season_label"}).copy()
m["date"] = pd.to_datetime(m["date"])
m["year"] = m["date"].dt.year  # the raw season labels ("2007/08", "2009/10", "2020/21") are inconsistent
for c in ["team1", "team2", "toss_winner", "winner"]:
    m[c] = m[c].map(lambda x: clean_team(x) if isinstance(x, str) else x)
dq["raw_venue_spellings"] = int(matches["venue"].nunique())
m["venue"] = m["venue"].map(clean_venue)
dq["clean_venues"] = int(m["venue"].nunique())
m["is_dl"] = m["method"].eq("D/L")
m["no_result"] = m["result"].eq("no result")
dq["season_labels_raw"] = int(matches["season"].nunique())
dq["seasons_clean"] = int(m["year"].nunique())
dq["no_result_matches"] = int(m["no_result"].sum())
dq["dl_matches"] = int(m["is_dl"].sum())
dq["super_over_matches"] = int(m["super_over"].eq("Y").sum())

# ============================================================================ deliveries
d = deliv.copy()
n_super = int((~d["inning"].isin([1, 2])).sum())
dq["super_over_deliveries_excluded"] = n_super
d = d[d["inning"].isin([1, 2])].copy()  # super overs are not part of the 20-over innings
d["batting_team"] = d["batting_team"].map(clean_team)
d["bowling_team"] = d["bowling_team"].map(clean_team)
d = d.merge(m[["match_id", "year"]], on="match_id", how="left")
assert d["year"].notna().all()
d["phase"] = d["over"].map(phase_of_over)

# --- ball-type flags
ext = d["extras_type"]
d["is_wide"] = ext.eq("wides")
d["is_noball"] = ext.eq("noballs")
d["is_penalty"] = ext.eq("penalty")
d["legal_cand"] = ~(d["is_wide"] | d["is_noball"] | d["is_penalty"])
d["faced"] = ~(d["is_wide"] | d["is_penalty"])  # a batter faces no-balls but not wides
d["bowler_runs"] = d["batsman_runs"] + np.where(d["is_wide"] | d["is_noball"], d["extra_runs"], 0)
d["is_four"] = d["batsman_runs"].eq(4)
d["is_six"] = d["batsman_runs"].eq(6)
kind = d["dismissal_kind"]
d["team_wicket"] = d["is_wicket"].eq(1) & ~kind.eq("retired hurt")  # retiring hurt is not a dismissal
bowler_kinds = ["bowled", "caught", "lbw", "stumped", "caught and bowled", "hit wicket"]
d["bowler_wicket"] = d["is_wicket"].eq(1) & kind.isin(bowler_kinds)
dq["retired_hurt_not_counted_as_wicket"] = int(kind.eq("retired hurt").sum())

# --- legal balls per over: an over has at most 6 legal balls. 28 overs show a 7th because a no-ball
#     was not tagged in the raw data, so each over is capped at 6 (the innings' last over keeps its true count).
ov = (d.groupby(["match_id", "inning", "over"])
        .agg(cand=("legal_cand", "sum"), runs=("total_runs", "sum"),
             wkts=("team_wicket", "sum"), sixes=("is_six", "sum"), fours=("is_four", "sum"))
        .reset_index())
last_over = ov.groupby(["match_id", "inning"])["over"].transform("max")
dq["overs_with_untagged_noball_capped_at_6"] = int((ov["cand"] > 6).sum())
ov["balls"] = np.where(ov["over"] < last_over, 6, np.minimum(ov["cand"], 6))
ov["phase"] = ov["over"].map(phase_of_over)

# ============================================================================ innings summary
team_rows = d.groupby(["match_id", "inning"]).agg(batting_team=("batting_team", "first"),
                                                  bowling_team=("bowling_team", "first")).reset_index()
tot = ov.groupby(["match_id", "inning"]).agg(runs=("runs", "sum"), wickets=("wkts", "sum"), balls=("balls", "sum"),
                                             sixes=("sixes", "sum"), fours=("fours", "sum")).reset_index()
ph = (ov.pivot_table(index=["match_id", "inning"], columns="phase", values=["runs", "wkts", "balls"],
                     aggfunc="sum", fill_value=0))
ph.columns = [f"{ {'Powerplay': 'pp', 'Middle': 'mid', 'Death': 'death'}[b] }_{ {'runs': 'runs', 'wkts': 'wkts', 'balls': 'balls'}[a] }"
              for a, b in ph.columns]
ph = ph.reset_index()
inn = team_rows.merge(tot, on=["match_id", "inning"]).merge(ph, on=["match_id", "inning"])
for c in ["pp_runs", "pp_wkts", "pp_balls", "mid_runs", "mid_wkts", "mid_balls", "death_runs", "death_wkts", "death_balls"]:
    if c not in inn.columns:
        inn[c] = 0
inn = inn.merge(m[["match_id", "year", "date", "venue", "winner", "is_dl", "no_result"]], on="match_id")
inn["won"] = np.where(inn["winner"].notna(), inn["batting_team"].eq(inn["winner"]), np.nan)
inn.to_csv(os.path.join(PROC, "innings_summary.csv"), index=False)

# ============================================================================ validation against the official result data
i1 = inn[inn["inning"] == 1].set_index("match_id")
i2 = inn[inn["inning"] == 2].set_index("match_id")
chk = m.set_index("match_id").join(i1["runs"].rename("first_runs")).join(i2["runs"].rename("second_runs"))
reg = chk[(~chk["is_dl"]) & (~chk["no_result"])]
dq["validation_target_matches_first_innings_plus_1"] = f"{int((reg['target_runs'] == reg['first_runs'] + 1).sum())} of {len(reg)}"
runs_res = reg[reg["result"] == "runs"]
dq["validation_runs_margin_matches"] = f"{int(((runs_res['first_runs'] - runs_res['second_runs']) == runs_res['result_margin']).sum())} of {len(runs_res)}"
assert (reg["target_runs"] == reg["first_runs"] + 1).all(), "ball-by-ball totals disagree with official targets"
assert ((runs_res["first_runs"] - runs_res["second_runs"]) == runs_res["result_margin"]).all()
print("Validation:", dq["validation_target_matches_first_innings_plus_1"], "targets and",
      dq["validation_runs_margin_matches"], "run margins match the official data")

# ============================================================================ matches_clean
first = i1[["batting_team", "runs", "wickets"]].rename(columns={"batting_team": "bat_first", "runs": "first_runs", "wickets": "first_wkts"})
second = i2[["batting_team", "runs", "wickets"]].rename(columns={"batting_team": "chase_team", "runs": "second_runs", "wickets": "second_wkts"})
mc = m.set_index("match_id").join(first).join(second).reset_index()
mc["decided"] = mc["winner"].notna()
mc["chase_won"] = np.where(mc["decided"] & mc["chase_team"].notna(), mc["chase_team"].eq(mc["winner"]), np.nan)
mc["toss_winner_won"] = np.where(mc["decided"], mc["toss_winner"].eq(mc["winner"]), np.nan)
mc["toss_chose_field"] = mc["toss_decision"].eq("field")
valid_year = mc["year"].isin(HOME_AWAY_YEARS)
h1 = valid_year & mc["team1"].map(HOME_GROUND).eq(mc["venue"])
h2 = valid_year & mc["team2"].map(HOME_GROUND).eq(mc["venue"])
mc["home_team"] = np.where(h1 & ~h2, mc["team1"], np.where(h2 & ~h1, mc["team2"], None))
mc["away_team"] = np.where(h1 & ~h2, mc["team2"], np.where(h2 & ~h1, mc["team1"], None))
keep = ["match_id", "date", "year", "season_label", "match_type", "city", "venue", "team1", "team2", "toss_winner",
        "toss_decision", "toss_chose_field", "winner", "result", "result_margin", "target_runs", "super_over", "is_dl",
        "no_result", "decided", "player_of_match", "bat_first", "chase_team", "first_runs", "first_wkts", "second_runs",
        "second_wkts", "chase_won", "toss_winner_won", "home_team", "away_team"]
mc = mc[keep].sort_values(["date", "match_id"])
mc.to_csv(os.path.join(PROC, "matches_clean.csv"), index=False)
dq["matches_with_a_home_team"] = int(mc["home_team"].notna().sum())

# ============================================================================ batting by phase
bat = (d[d["faced"] | d["batsman_runs"].gt(0)]
       .groupby(["batter", "year", "batting_team", "phase"])
       .agg(runs=("batsman_runs", "sum"), balls=("faced", "sum"), fours=("is_four", "sum"), sixes=("is_six", "sum"))
       .reset_index().rename(columns={"batter": "player", "batting_team": "team"}))
out = d[d["team_wicket"] & d["player_dismissed"].notna()]
dis = (out.groupby(["player_dismissed", "year", "batting_team", "phase"]).size().rename("dismissals").reset_index()
          .rename(columns={"player_dismissed": "player", "batting_team": "team"}))
bat = bat.merge(dis, on=["player", "year", "team", "phase"], how="outer")
bat[["runs", "balls", "fours", "sixes", "dismissals"]] = bat[["runs", "balls", "fours", "sixes", "dismissals"]].fillna(0).astype(int)
bat.to_csv(os.path.join(PROC, "batting_by_phase.csv"), index=False)
dq["batters"] = int(bat["player"].nunique())

# ============================================================================ bowling by phase
bw = d.copy()
bw["legal"] = bw["legal_cand"].astype(int)
bw["dot"] = (bw["legal_cand"] & bw["bowler_runs"].eq(0)).astype(int)
bowl = (bw.groupby(["bowler", "year", "bowling_team", "phase"])
          .agg(balls=("legal", "sum"), runs=("bowler_runs", "sum"), wickets=("bowler_wicket", "sum"), dots=("dot", "sum"))
          .reset_index().rename(columns={"bowler": "player", "bowling_team": "team"}))
bowl["wickets"] = bowl["wickets"].astype(int)
bowl.to_csv(os.path.join(PROC, "bowling_by_phase.csv"), index=False)
dq["bowlers"] = int(bowl["player"].nunique())

# ============================================================================ sanity checks on the player tables
assert int(bat["runs"].sum()) == int(d["batsman_runs"].sum()), "batting table loses runs"
assert int(bowl["wickets"].sum()) == int(d["bowler_wicket"].sum())
dq["batting_runs_total"] = int(bat["runs"].sum())
dq["sixes_total"] = int(inn["sixes"].sum())
dq["innings_rows"] = int(len(inn))
dq["matches_clean_rows"] = int(len(mc))

with open(os.path.join(PROC, "data_quality.json"), "w") as f:
    json.dump(dq, f, indent=2)
print(json.dumps(dq, indent=2))
print("Saved processed tables to", os.path.relpath(PROC, os.getcwd()))
