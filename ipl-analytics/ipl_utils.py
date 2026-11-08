"""Shared cleaning rules, constants and helpers for the IPL analytics project.

Used by the data-prep pipeline (src/), the analysis scripts and the Streamlit app, so every
part of the project applies exactly the same definitions.
"""
import math
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(ROOT, "data", "raw")
PROC = os.path.join(ROOT, "data", "processed")

# ----------------------------------------------------------------------------- team names
# Franchises that were renamed are merged into their current name. Deccan Chargers, Gujarat Lions,
# Pune Warriors and Kochi Tuskers Kerala are different franchises and stay separate.
TEAM_MAP = {
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    "Rising Pune Supergiants": "Rising Pune Supergiant",
}


def clean_team(name):
    return TEAM_MAP.get(name, name)


# ----------------------------------------------------------------------------- venues
# 58 spellings in the raw file -> 36 real grounds.
VENUE_MAP = {
    "Eden Gardens": "Eden Gardens, Kolkata",
    "Eden Gardens, Kolkata": "Eden Gardens, Kolkata",
    "Wankhede Stadium": "Wankhede Stadium, Mumbai",
    "Wankhede Stadium, Mumbai": "Wankhede Stadium, Mumbai",
    "M Chinnaswamy Stadium": "M Chinnaswamy Stadium, Bengaluru",
    "M.Chinnaswamy Stadium": "M Chinnaswamy Stadium, Bengaluru",
    "M Chinnaswamy Stadium, Bengaluru": "M Chinnaswamy Stadium, Bengaluru",
    "Feroz Shah Kotla": "Arun Jaitley Stadium (Feroz Shah Kotla), Delhi",
    "Arun Jaitley Stadium": "Arun Jaitley Stadium (Feroz Shah Kotla), Delhi",
    "Arun Jaitley Stadium, Delhi": "Arun Jaitley Stadium (Feroz Shah Kotla), Delhi",
    "Rajiv Gandhi International Stadium, Uppal": "Rajiv Gandhi International Stadium, Hyderabad",
    "Rajiv Gandhi International Stadium": "Rajiv Gandhi International Stadium, Hyderabad",
    "Rajiv Gandhi International Stadium, Uppal, Hyderabad": "Rajiv Gandhi International Stadium, Hyderabad",
    "MA Chidambaram Stadium, Chepauk": "MA Chidambaram Stadium, Chennai",
    "MA Chidambaram Stadium, Chepauk, Chennai": "MA Chidambaram Stadium, Chennai",
    "MA Chidambaram Stadium": "MA Chidambaram Stadium, Chennai",
    "Sawai Mansingh Stadium": "Sawai Mansingh Stadium, Jaipur",
    "Sawai Mansingh Stadium, Jaipur": "Sawai Mansingh Stadium, Jaipur",
    "Dubai International Cricket Stadium": "Dubai International Cricket Stadium",
    "Punjab Cricket Association Stadium, Mohali": "PCA IS Bindra Stadium, Mohali",
    "Punjab Cricket Association IS Bindra Stadium, Mohali": "PCA IS Bindra Stadium, Mohali",
    "Punjab Cricket Association IS Bindra Stadium": "PCA IS Bindra Stadium, Mohali",
    "Punjab Cricket Association IS Bindra Stadium, Mohali, Chandigarh": "PCA IS Bindra Stadium, Mohali",
    "Sheikh Zayed Stadium": "Sheikh Zayed Stadium, Abu Dhabi",
    "Zayed Cricket Stadium, Abu Dhabi": "Sheikh Zayed Stadium, Abu Dhabi",
    "Sharjah Cricket Stadium": "Sharjah Cricket Stadium",
    "Narendra Modi Stadium, Ahmedabad": "Narendra Modi Stadium, Ahmedabad",
    "Sardar Patel Stadium, Motera": "Narendra Modi Stadium, Ahmedabad",  # same ground, renamed
    "Maharashtra Cricket Association Stadium": "MCA Stadium, Pune",
    "Maharashtra Cricket Association Stadium, Pune": "MCA Stadium, Pune",
    "Subrata Roy Sahara Stadium": "MCA Stadium, Pune",  # same ground, sponsor name
    "Dr DY Patil Sports Academy": "DY Patil Stadium, Navi Mumbai",
    "Dr DY Patil Sports Academy, Mumbai": "DY Patil Stadium, Navi Mumbai",
    "Brabourne Stadium": "Brabourne Stadium, Mumbai",
    "Brabourne Stadium, Mumbai": "Brabourne Stadium, Mumbai",
    "Kingsmead": "Kingsmead, Durban",
    "Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium, Lucknow": "Ekana Cricket Stadium, Lucknow",
    "Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium": "ACA-VDCA Stadium, Visakhapatnam",
    "Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium, Visakhapatnam": "ACA-VDCA Stadium, Visakhapatnam",
    "SuperSport Park": "SuperSport Park, Centurion",
    "Saurashtra Cricket Association Stadium": "Saurashtra Cricket Association Stadium, Rajkot",
    "Himachal Pradesh Cricket Association Stadium": "HPCA Stadium, Dharamsala",
    "Himachal Pradesh Cricket Association Stadium, Dharamsala": "HPCA Stadium, Dharamsala",
    "Holkar Cricket Stadium": "Holkar Cricket Stadium, Indore",
    "New Wanderers Stadium": "New Wanderers Stadium, Johannesburg",
    "Newlands": "Newlands, Cape Town",
    "St George's Park": "St George's Park, Gqeberha",
    "Barabati Stadium": "Barabati Stadium, Cuttack",
    "JSCA International Stadium Complex": "JSCA Stadium, Ranchi",
    "Shaheed Veer Narayan Singh International Stadium": "Shaheed Veer Narayan Singh Stadium, Raipur",
    "Nehru Stadium": "Nehru Stadium, Kochi",
    "Maharaja Yadavindra Singh International Cricket Stadium, Mullanpur": "Mullanpur Stadium, New Chandigarh",
    "Green Park": "Green Park, Kanpur",
    "Buffalo Park": "Buffalo Park, East London",
    "De Beers Diamond Oval": "De Beers Diamond Oval, Kimberley",
    "Vidarbha Cricket Association Stadium, Jamtha": "VCA Stadium, Nagpur",
    "Barsapara Cricket Stadium, Guwahati": "Barsapara Stadium, Guwahati",
    "OUTsurance Oval": "OUTsurance Oval, Bloemfontein",
}


def clean_venue(name):
    if name not in VENUE_MAP:
        raise KeyError(f"Unmapped venue: {name!r} - add it to VENUE_MAP in ipl_utils.py")
    return VENUE_MAP[name]


# ----------------------------------------------------------------------------- home grounds
HYD = "Rajiv Gandhi International Stadium, Hyderabad"
HOME_GROUND = {
    "Mumbai Indians": "Wankhede Stadium, Mumbai",
    "Chennai Super Kings": "MA Chidambaram Stadium, Chennai",
    "Kolkata Knight Riders": "Eden Gardens, Kolkata",
    "Royal Challengers Bengaluru": "M Chinnaswamy Stadium, Bengaluru",
    "Sunrisers Hyderabad": HYD,
    "Deccan Chargers": HYD,
    "Rajasthan Royals": "Sawai Mansingh Stadium, Jaipur",
    "Punjab Kings": "PCA IS Bindra Stadium, Mohali",
    "Delhi Capitals": "Arun Jaitley Stadium (Feroz Shah Kotla), Delhi",
    "Gujarat Titans": "Narendra Modi Stadium, Ahmedabad",
    "Lucknow Super Giants": "Ekana Cricket Stadium, Lucknow",
    "Gujarat Lions": "Saurashtra Cricket Association Stadium, Rajkot",
    "Pune Warriors": "MCA Stadium, Pune",
    "Rising Pune Supergiant": "MCA Stadium, Pune",
    "Kochi Tuskers Kerala": "Nehru Stadium, Kochi",
}
# Seasons played in a true home-and-away format in India. 2009 (South Africa), 2020 (UAE),
# 2021 (bubble venues) and 2022 (Maharashtra only) had no real home games.
HOME_AWAY_YEARS = {2008, *range(2010, 2020), 2023, 2024}

# ----------------------------------------------------------------------------- phases
PHASES = ["Powerplay", "Middle", "Death"]  # overs 1-6, 7-15, 16-20


def phase_of_over(over_index):
    """over_index is 0-based in the raw data (0 = first over)."""
    if over_index <= 5:
        return "Powerplay"
    if over_index <= 14:
        return "Middle"
    return "Death"


# ----------------------------------------------------------------------------- stats helpers
def wilson_ci(successes, n, z=1.96):
    """95% Wilson score interval for a proportion, returned as (low, high) in percent."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (100 * (centre - half), 100 * (centre + half))


def pct(a, b):
    return float("nan") if b == 0 else 100.0 * a / b


# ----------------------------------------------------------------------------- loaders
def load_processed(name):
    return pd.read_csv(os.path.join(PROC, name))


def load_matches():
    m = load_processed("matches_clean.csv")
    m["date"] = pd.to_datetime(m["date"])
    return m


# ----------------------------------------------------------------------------- win-probability model
def load_win_model():
    import json
    with open(os.path.join(ROOT, "models", "win_model.json")) as f:
        return json.load(f)


def predict_win_prob(model, target, score, wickets_lost, overs_done):
    """Chasing team's win probability at the end of a completed over (1-19). Pure numpy, no sklearn needed."""
    balls_left = 120 - 6 * overs_done
    runs_needed = target - score
    req_rate = runs_needed * 6 / balls_left
    cur_rate = score * 6 / (6 * overs_done)
    x = np.array([runs_needed, balls_left, wickets_lost, req_rate, cur_rate], dtype=float)
    z = (x - np.array(model["mean"])) / np.array(model["scale"])
    return float(1 / (1 + np.exp(-(z @ np.array(model["coef"]) + model["intercept"]))))
