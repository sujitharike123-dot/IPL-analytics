"""Chart functions shared by the analysis script (README figures) and the Streamlit dashboard.

Design rules (from the data-viz guidance used for this project):
  - one accent colour (blue) with gray for context, categorical colours in a fixed order (blue, orange, aqua)
  - thin marks, 2 px lines, hairline recessive gridlines, no dual axes
  - a legend whenever there are two or more series, plus direct labels where there is room
  - text in ink colours, never in a series colour
Every function takes a DataFrame and returns a matplotlib Figure.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
CONTEXT = "#c3c2b7"   # de-emphasised bars
BLUE = "#2a78d6"      # categorical slot 1 / accent
ORANGE = "#eb6834"    # slot 2
AQUA = "#1baf7a"      # slot 3
PHASE_COLORS = {"Powerplay": BLUE, "Middle": ORANGE, "Death": AQUA}


def _base(figsize=(8, 4.2), horizontal_grid=False):
    fig, ax = plt.subplots(figsize=figsize, dpi=110)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    if horizontal_grid:
        ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    else:
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    return fig, ax


def _titles(ax, title, subtitle=None):
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color=INK, pad=26 if subtitle else 12)
    if subtitle:
        ax.text(0, 1.035, subtitle, transform=ax.transAxes, fontsize=9, color=INK2, va="bottom")


def _done(fig):
    fig.tight_layout()
    return fig


def save(fig, path):
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- line charts by season
def _line_by_year(df, ycol, title, subtitle, ylabel, ref=None, ref_label=None, ylim=None, fmt="{:.1f}%"):
    fig, ax = _base()
    x, y = df["year"].to_numpy(), df[ycol].to_numpy()
    if ref is not None:
        ax.axhline(ref, color=MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
        ax.text(x.min() - 0.4, ref, f" {ref_label} ", color=MUTED, fontsize=8, va="center", ha="left", zorder=2,
                bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.5))
    ax.plot(x, y, color=BLUE, linewidth=2, zorder=3)
    ax.plot(x, y, "o", color=BLUE, markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=4)
    ax.annotate(fmt.format(y[-1]), (x[-1], y[-1]), textcoords="offset points", xytext=(8, 0), fontsize=9,
                color=INK, va="center")
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in x], rotation=45 if len(x) > 10 else 0)
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_xlim(x.min() - 0.5, x.max() + 1.2)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    _titles(ax, title, subtitle)
    return _done(fig)


def chart_chase_by_year(df):
    return _line_by_year(df, "chase_win_pct", "Chasing teams' win rate by season",
                         "Share of decided matches won by the team batting second", "Win rate (%)",
                         ref=50, ref_label="50% = no edge", ylim=(35, 75))


def chart_toss_decision_by_year(df):
    return _line_by_year(df, "chose_to_field_pct", "How often toss winners choose to field",
                         "Share of toss winners who chose to bowl first", "Chose to field (%)",
                         ref=50, ref_label="50% = even split", ylim=(20, 95))


def chart_phase_rates(df):
    fig, ax = _base()
    x = df["year"].to_numpy()
    cols = {"Powerplay": "powerplay_rpo", "Middle": "middle_rpo", "Death": "death_rpo"}
    for name, col in cols.items():
        y = df[col].to_numpy()
        ax.plot(x, y, color=PHASE_COLORS[name], linewidth=2, label=f"{name} (overs {'1-6' if name == 'Powerplay' else '7-15' if name == 'Middle' else '16-20'})", zorder=3)
        ax.plot(x, y, "o", color=PHASE_COLORS[name], markersize=5, markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=4)
        ax.annotate(f"{y[-1]:.1f}", (x[-1], y[-1]), textcoords="offset points", xytext=(8, 0), fontsize=9, color=INK, va="center")
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in x], rotation=45)
    ax.set_xlim(x.min() - 0.5, x.max() + 1.2)
    ax.set_ylabel("Runs per over", color=INK2, fontsize=9)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper left", ncol=3)
    _titles(ax, "Scoring rate by phase of the innings", "Runs per over, both innings, regular matches")
    ax.set_ylim(6, 12.3)
    return _done(fig)


def chart_bars_by_year(df, ycol, title, subtitle, ylabel, highlight_from=None, fmt="{:.0f}", ylim=None):
    fig, ax = _base()
    x, y = df["year"].to_numpy(), df[ycol].to_numpy()
    colors = [BLUE if (highlight_from is None or v >= highlight_from) else CONTEXT for v in x]
    ax.bar(x, y, width=0.62, color=colors, zorder=3)
    if highlight_from is not None:
        for xi, yi in zip(x, y):
            if xi >= highlight_from:
                ax.text(xi, yi, fmt.format(yi), ha="center", va="bottom", fontsize=9, color=INK)
    else:
        i = int(np.argmax(y))
        ax.text(x[i], y[i], fmt.format(y[i]), ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in x], rotation=45)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    if ylim:
        ax.set_ylim(*ylim)
    _titles(ax, title, subtitle)
    return _done(fig)


def chart_avg_first_innings(df):
    return chart_bars_by_year(df, "avg_first_innings", "Average first-innings score by season",
                              "Regular matches; 2023 and 2024 (Impact Player seasons) highlighted", "Runs",
                              highlight_from=2023, fmt="{:.0f}", ylim=(120, 200))


def chart_200_plus(df):
    return chart_bars_by_year(df, "scores_of_200_plus", "Scores of 200 or more per season",
                              "Innings totals of 200+, both innings; 2023 and 2024 highlighted", "Innings of 200+",
                              highlight_from=2023, fmt="{:.0f}")


def chart_sixes_per_match(df):
    return chart_bars_by_year(df, "sixes_per_match", "Sixes per match by season",
                              "Both innings combined; 2023 and 2024 highlighted", "Sixes per match",
                              highlight_from=2023, fmt="{:.1f}")


# --------------------------------------------------------------------------- bars
def chart_target_bands(df):
    fig, ax = _base()
    labels = [s.split(") ", 1)[1] for s in df["target_band"]]
    x = np.arange(len(df))
    ax.bar(x, df["chase_win_pct"], width=0.55, color=BLUE, zorder=3)
    ax.axhline(50, color=MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=2)
    ax.text(len(df) - 0.5, 51, "50%", color=MUTED, fontsize=8, ha="right", va="bottom")
    for xi, v, n in zip(x, df["chase_win_pct"], df["matches"]):
        ax.text(xi, v + 1.2, f"{v:.0f}%", ha="center", va="bottom", fontsize=10, color=INK, fontweight="bold")
        ax.text(xi, 2, f"{int(n)} matches", ha="center", va="bottom", fontsize=8, color="#ffffff")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Chasing team wins (%)", color=INK2, fontsize=9)
    ax.set_xlabel("Target set (runs)", color=INK2, fontsize=9)
    _titles(ax, "How often a chase succeeds, by the target it faces", "Regular matches 2008-2024, excluding rain-affected games")
    return _done(fig)


def chart_powerplay_vs_win(df):
    fig, ax = _base()
    bands = sorted(df["powerplay_runs"].unique())
    labels = [b.split(") ", 1)[1] for b in bands]
    x = np.arange(len(bands))
    w = 0.34
    for k, (inn, color, name) in enumerate([(1, BLUE, "Batting first"), (2, ORANGE, "Chasing")]):
        sub = df[df["inning"] == inn].set_index("powerplay_runs").reindex(bands)
        pos = x + (k - 0.5) * (w + 0.04)
        ax.bar(pos, sub["batting_team_win_pct"], width=w, color=color, label=name, zorder=3)
        for xi, v in zip(pos, sub["batting_team_win_pct"]):
            ax.text(xi, v + 1, f"{v:.0f}%", ha="center", va="bottom", fontsize=9, color=INK)
    ax.axhline(50, color=MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 90)
    ax.set_xlabel("Runs scored in the powerplay (overs 1-6)", color=INK2, fontsize=9)
    ax.set_ylabel("Team wins (%)", color=INK2, fontsize=9)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper left", ncol=2)
    _titles(ax, "A fast powerplay is linked to winning", "Win rate of the batting team by powerplay runs, regular matches")
    return _done(fig)


def chart_venues(df, highlight=None, value_col="avg_first_innings", title="Average first-innings score by ground",
                 subtitle="Grounds with 20+ regular matches", xlabel="Average first-innings score (runs)"):
    d = df.sort_values(value_col)
    fig, ax = _base(figsize=(8, 0.34 * len(d) + 1.6), horizontal_grid=True)
    ax.yaxis.grid(False)
    colors = [BLUE if (highlight and v == highlight) else CONTEXT for v in d["venue"]]
    ax.barh(d["venue"], d[value_col], height=0.62, color=colors, zorder=3)
    lo = max(0, d[value_col].min() * 0.8)
    ax.set_xlim(lo, d[value_col].max() * 1.06)
    for yi, v, n in zip(range(len(d)), d[value_col], d["matches"]):
        ax.text(v + (d[value_col].max() - lo) * 0.01, yi, f"{v:.1f}  (n={int(n)})", va="center", fontsize=8, color=INK2)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    _titles(ax, title, subtitle)
    return _done(fig)


def chart_chase_by_venue(df, highlight=None, min_n=20):
    return chart_venues(df, highlight, value_col="chase_win_pct", title="Chasing win rate by ground",
                        subtitle=f"Grounds with {min_n}+ regular matches; bars are chasing wins as a share of decided matches",
                        xlabel="Chasing team wins (%)")


# --------------------------------------------------------------------------- home advantage dumbbell
def chart_home_away(df, highlight=("Sunrisers Hyderabad",)):
    d = df.sort_values("home_boost_pts")
    fig, ax = _base(figsize=(8, 0.5 * len(d) + 1.8), horizontal_grid=True)
    ax.yaxis.grid(False)
    ys = np.arange(len(d))
    ax.hlines(ys, d["away_win_pct"], d["home_win_pct"], color=CONTEXT, linewidth=2, zorder=2)
    ax.scatter(d["away_win_pct"], ys, s=55, color=ORANGE, zorder=3, label="Away", edgecolor=SURFACE, linewidth=1.2)
    ax.scatter(d["home_win_pct"], ys, s=55, color=BLUE, zorder=3, label="Home", edgecolor=SURFACE, linewidth=1.2)
    for yi, (_, r) in zip(ys, d.iterrows()):
        ax.text(max(r["home_win_pct"], r["away_win_pct"]) + 2, yi, f"{r['home_boost_pts']:+.1f} pts", va="center",
                fontsize=8, color=INK2)
    ax.set_yticks(ys)
    ax.set_yticklabels(d["team"], fontsize=9)
    for lab in ax.get_yticklabels():
        if lab.get_text() in highlight:
            lab.set_fontweight("bold")
            lab.set_color(INK)
    ax.axvline(50, color=MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
    ax.set_xlim(max(0, min(d["away_win_pct"].min(), d["home_win_pct"].min()) - 6), d["home_win_pct"].max() + 14)
    ax.set_xlabel("Win rate (%)", color=INK2, fontsize=9)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="lower right", ncol=2)
    _titles(ax, "Home vs away win rate by franchise",
            "True home-and-away seasons only; franchises with 15+ home and 15+ away matches")
    return _done(fig)


def chart_team_seasons(df, team):
    """df: year, matches, wins, win_pct for one team."""
    fig, ax = _base()
    x, y = df["year"].to_numpy(), df["win_pct"].to_numpy()
    ax.bar(x, y, width=0.62, color=BLUE, zorder=3)
    ax.axhline(50, color=MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=2)
    for xi, yi, n in zip(x, y, df["matches"]):
        ax.text(xi, yi + 1, f"{yi:.0f}%", ha="center", va="bottom", fontsize=8, color=INK)
        ax.text(xi, 1.5, f"{int(n)}", ha="center", va="bottom", fontsize=7, color="#ffffff")
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in x], rotation=45)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Win rate (%)", color=INK2, fontsize=9)
    _titles(ax, f"{team}: win rate by season", "Share of decided matches won; white numbers at the base are matches played")
    return _done(fig)


# --------------------------------------------------------------------------- win-probability model
def chart_calibration(cal, title="Predicted vs actual chase win rate (held-out 2023-24 seasons)"):
    """cal: columns pred_mean, actual_mean, n."""
    fig, ax = _base(figsize=(5.4, 5.0))
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.plot([0, 100], [0, 100], color=MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=1, label="Perfect calibration")
    ax.plot(cal["pred_mean"] * 100, cal["actual_mean"] * 100, color=BLUE, linewidth=2, zorder=3, label="Model")
    ax.plot(cal["pred_mean"] * 100, cal["actual_mean"] * 100, "o", color=BLUE, markersize=6, markeredgecolor=SURFACE,
            markeredgewidth=1.4, zorder=4)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Model's predicted win probability (%)", color=INK2, fontsize=9)
    ax.set_ylabel("Share of chases actually won (%)", color=INK2, fontsize=9)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper left")
    _titles(ax, title, None)
    return _done(fig)


def chart_win_curve(curve, title="Chasing team win probability, over by over", subtitle=None):
    """curve: columns over_label (str), prob (0-1) - the chasing team's win probability by over."""
    fig, ax = _base(figsize=(7.5, 3.6))
    x = np.arange(len(curve))
    ax.axhline(50, color=MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
    ax.plot(x, curve["prob"] * 100, color=BLUE, linewidth=2, zorder=3)
    ax.plot(x[-1:], curve["prob"].to_numpy()[-1:] * 100, "o", color=BLUE, markersize=7, markeredgecolor=SURFACE,
            markeredgewidth=1.5, zorder=4)
    ax.set_xticks(x[::2])
    ax.set_xticklabels(curve["over_label"].to_numpy()[::2], fontsize=8)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Chasing team win probability (%)", color=INK2, fontsize=9)
    ax.set_xlabel("Overs completed", color=INK2, fontsize=9)
    _titles(ax, title, subtitle)
    return _done(fig)


# --------------------------------------------------------------------------- players
def chart_top_players(df, label_col, value_col, title, subtitle=None, fmt="{:.1f}", highlight=None):
    """Horizontal bars, best at the top. df must already be sorted best-first and trimmed."""
    fig, ax = _base(figsize=(8, 0.34 * len(df) + 1.6), horizontal_grid=True)
    ax.yaxis.grid(False)
    y = np.arange(len(df))[::-1]
    cols = [BLUE if (highlight is None or lab == highlight) else CONTEXT for lab in df[label_col]]
    ax.barh(y, df[value_col], height=0.62, color=cols, zorder=3)
    for yi, v in zip(y, df[value_col]):
        ax.text(v, yi, "  " + fmt.format(v), va="center", fontsize=8.5, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(df[label_col], fontsize=9)
    lo = float(df[value_col].min())
    ax.set_xlim(max(0, lo * 0.85), float(df[value_col].max()) * 1.12)
    _titles(ax, title, subtitle)
    return _done(fig)
