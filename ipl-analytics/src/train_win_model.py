"""Chase win-probability model.

At the end of every over of a chase, how likely is the chasing team to win, given the score, the wickets lost
and the target? Trained on 2008-2022, tested on the unseen 2023-24 seasons.

Usage (from the repo root, after src/prepare_data.py):
    python src/train_win_model.py

Writes:
    data/processed/chase_states.csv   one row per chase per completed over (used by the dashboard's match replay)
    models/win_model.json             logistic-regression coefficients (the app predicts with plain numpy)
    results/model_metrics.json        held-out performance vs. simple baselines
    figures/model_calibration.png
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
import ipl_charts as ch  # noqa: E402
from ipl_utils import PROC, RAW, load_matches  # noqa: E402

FEATURES = ["runs_needed", "balls_left", "wickets_lost", "req_rate", "cur_rate"]

# ------------------------------------------------------------------ build chase states from ball-by-ball data
d = pd.read_csv(os.path.join(RAW, "deliveries.csv"))
d = d[d["inning"] == 2].copy()  # the chase (super overs are inning 3+, first innings is inning 1)
d["wkt"] = ((d["is_wicket"] == 1) & (d["dismissal_kind"] != "retired hurt")).astype(int)
ov = d.groupby(["match_id", "over"]).agg(runs=("total_runs", "sum"), wkts=("wkt", "sum")).reset_index()
ov = ov.sort_values(["match_id", "over"])
ov["score"] = ov.groupby("match_id")["runs"].cumsum()
ov["wickets_lost"] = ov.groupby("match_id")["wkts"].cumsum()
ov["overs_done"] = ov["over"] + 1

m = load_matches()
reg = m[(m["decided"]) & (~m["is_dl"]) & (~m["no_result"]) & m["target_runs"].notna()]
reg = reg[["match_id", "year", "date", "chase_team", "bat_first", "venue", "target_runs", "chase_won"]]
st = ov.merge(reg, on="match_id")
# keep the states where the chase is still alive: overs 1-19 and the chase has neither finished nor been bowled out
st = st[(st["overs_done"] <= 19) & (st["score"] < st["target_runs"]) & (st["wickets_lost"] < 10)].copy()
st["runs_needed"] = st["target_runs"] - st["score"]
st["balls_left"] = 120 - 6 * st["overs_done"]
st["req_rate"] = st["runs_needed"] * 6 / st["balls_left"]
st["cur_rate"] = st["score"] * 6 / (6 * st["overs_done"])
st["chase_won"] = st["chase_won"].astype(int)
st = st[["match_id", "year", "date", "chase_team", "bat_first", "venue", "overs_done", "score", "wickets_lost",
         "target_runs", "runs_needed", "balls_left", "req_rate", "cur_rate", "chase_won"]]
st.to_csv(os.path.join(PROC, "chase_states.csv"), index=False)
print(f"{len(st):,} chase states from {st['match_id'].nunique()} chases")

train = st[st["year"] <= 2022]
test = st[st["year"] >= 2023]
print(f"train: {train['match_id'].nunique()} chases ({len(train):,} states) | test: {test['match_id'].nunique()} chases ({len(test):,} states)")

# ------------------------------------------------------------------ models
sc = StandardScaler().fit(train[FEATURES])
lr = LogisticRegression(C=1.0, max_iter=1000).fit(sc.transform(train[FEATURES]), train["chase_won"])
p_lr = lr.predict_proba(sc.transform(test[FEATURES]))[:, 1]
gb = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150, random_state=0).fit(train[FEATURES], train["chase_won"])
p_gb = gb.predict_proba(test[FEATURES])[:, 1]
base = np.full(len(test), train["chase_won"].mean())  # "always say the historical chase win rate"
# a no-model rule of thumb: the side needing under 8 an over is favoured
p_rule = np.where(test["req_rate"] < 8, 0.65, 0.35)


def metrics(p):
    return {"auc": round(float(roc_auc_score(test["chase_won"], p)), 3),
            "log_loss": round(float(log_loss(test["chase_won"], p)), 3),
            "brier": round(float(brier_score_loss(test["chase_won"], p)), 3)}


M = {"train_chases": int(train["match_id"].nunique()), "test_chases": int(test["match_id"].nunique()),
     "test_states": int(len(test)),
     "logistic_regression": metrics(p_lr), "gradient_boosting": metrics(p_gb),
     "baseline_historical_rate": metrics(base), "rule_of_thumb_rrr_under_8": metrics(p_rule)}

# performance by how far into the chase (honest: early overs are close to coin-flips)
rows = []
for lo, hi, name in [(1, 6, "overs 1-6"), (7, 12, "overs 7-12"), (13, 19, "overs 13-19")]:
    s = test["overs_done"].between(lo, hi)
    rows.append({"stage": name, "states": int(s.sum()),
                 "auc": round(float(roc_auc_score(test.loc[s, "chase_won"], p_lr[s.to_numpy()])), 3)})
M["logistic_auc_by_stage"] = rows

# calibration of the shipped (logistic) model on the held-out seasons
bins = pd.cut(p_lr, [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0], include_lowest=True)
cal = (pd.DataFrame({"p": p_lr, "y": test["chase_won"].to_numpy(), "bin": bins})
       .groupby("bin", observed=True).agg(pred_mean=("p", "mean"), actual_mean=("y", "mean"), n=("y", "size")).reset_index())
M["calibration"] = cal.drop(columns="bin").round(3).to_dict("records")
os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
with open(os.path.join(ROOT, "results", "model_metrics.json"), "w") as f:
    json.dump(M, f, indent=2)
print(json.dumps({k: M[k] for k in ["logistic_regression", "gradient_boosting", "baseline_historical_rate",
                                    "rule_of_thumb_rrr_under_8", "logistic_auc_by_stage"]}, indent=2))

# ------------------------------------------------------------------ ship the logistic model as JSON
# refit on every season so the dashboard uses all the data; the metrics above come from the train/test split
full_sc = StandardScaler().fit(st[FEATURES])
full = LogisticRegression(C=1.0, max_iter=1000).fit(full_sc.transform(st[FEATURES]), st["chase_won"])
os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)
with open(os.path.join(ROOT, "models", "win_model.json"), "w") as f:
    json.dump({"features": FEATURES, "mean": full_sc.mean_.tolist(), "scale": full_sc.scale_.tolist(),
               "coef": full.coef_[0].tolist(), "intercept": float(full.intercept_[0]),
               "trained_on": "all chases 2008-2024 (evaluation uses a 2008-22 train / 2023-24 test split)"}, f, indent=2)
ch.save(ch.chart_calibration(cal), os.path.join(ROOT, "figures", "model_calibration.png"))
print("Saved models/win_model.json, results/model_metrics.json, figures/model_calibration.png")
