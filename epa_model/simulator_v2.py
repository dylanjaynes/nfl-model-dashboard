"""
simulator_v2.py — Calibrated Monte Carlo engine (no Vegas shrinkage)

Key differences from v1:
  • Zero Vegas shrinkage — score predictions are 100% from EPA signal
  • Pass/rush EPA split in scoring formula (empirically better)
  • EPA_TO_PTS = 0.70 (empirically derived from 2022-2024 regression)
  • Game shocks decomposed into independent team shocks + corr shock,
    calibrated to match NFL empirical margin SD ≈ 14.5 and total SD ≈ 13.1
  • Vegas spread/total are used ONLY for grading (ATS/total pick output),
    never as an input to score projections
"""

import numpy as np
from scipy import stats
from typing import Optional

# ── Empirically calibrated constants ──────────────────────────────────────
LEAGUE_AVG_PTS   = 22.1   # baseline from TN EPA regression (2022-24)
EPA_TO_PTS       = 0.70   # empirically derived (was 0.45, which was wrong)
AVG_PASS_PLAYS   = 35.0   # 2022-24 average offensive pass plays / game
AVG_RUSH_PLAYS   = 27.0   # 2022-24 average offensive rush plays / game
HFA              = 1.0    # per-team home field pts (2.0 pts net in spread)

# Game shock calibration
# Decompose into:
#   team_shock  ~ t(df=6, scale=7.5): independent per team, drives margin AND total
#   corr_shock  ~ t(df=8, scale=5.0): adds to margin only (explains margin > total SD)
# Result: margin SD ≈ 14.4 pts, total SD ≈ 12.9 pts  (empirical: 14.5 / 13.1)
TEAM_SHOCK_SCALE = 7.5
TEAM_SHOCK_DF    = 6
CORR_SHOCK_SCALE = 5.0
CORR_SHOCK_DF    = 8


def simulate_game_v2(
    home_metrics: dict,
    away_metrics: dict,
    vegas_spread: Optional[float] = None,
    vegas_total:  Optional[float] = None,
    sims: int = 5000,
    rng:  Optional[np.random.Generator] = None,
) -> dict:
    """
    Monte Carlo simulation — pure EPA model, no Vegas contamination.

    Vegas lines are accepted as parameters for grading output only.
    They do NOT influence home_scores or away_scores.
    """
    if rng is None:
        rng = np.random.default_rng()

    ht, at = home_metrics, away_metrics

    # ── Pass/rush matchup edges ────────────────────────────────────────────
    # NOTE: def_epa convention = EPA *allowed* per play (opponent's perspective).
    #   positive def_epa = bad defense (opponents gain EPA freely)
    #   negative def_epa = great defense (opponents lose EPA)
    # So scoring edge = home_offense_quality + away_defense_weakness (ADD, not subtract).
    # e.g. great away defense (def_epa = -0.12) REDUCES home scoring edge.
    home_pass_edge = ht["pass_off_epa"] + at["pass_def_epa"]
    home_rush_edge = ht["rush_off_epa"] + at["rush_def_epa"]
    away_pass_edge = at["pass_off_epa"] + ht["pass_def_epa"]
    away_rush_edge = at["rush_off_epa"] + ht["rush_def_epa"]

    # ── Base projected scores (EPA → points, empirically calibrated) ───────
    home_base = (LEAGUE_AVG_PTS
                 + home_pass_edge * AVG_PASS_PLAYS * EPA_TO_PTS
                 + home_rush_edge * AVG_RUSH_PLAYS * EPA_TO_PTS
                 + HFA)
    away_base = (LEAGUE_AVG_PTS
                 + away_pass_edge * AVG_PASS_PLAYS * EPA_TO_PTS
                 + away_rush_edge * AVG_RUSH_PLAYS * EPA_TO_PTS
                 - HFA)

    # ── EPA uncertainty noise (estimation error in team metrics) ──────────
    n_ht = max(ht["n_games"], 1)
    n_at = max(at["n_games"], 1)

    # Noise in EPA estimates (shrinks as we observe more games)
    h_off_noise = rng.normal(0, ht["off_epa_sd"] / np.sqrt(n_ht), sims)
    h_def_noise = rng.normal(0, ht["def_epa_sd"] / np.sqrt(n_ht), sims)
    a_off_noise = rng.normal(0, at["off_epa_sd"] / np.sqrt(n_at), sims)
    a_def_noise = rng.normal(0, at["def_epa_sd"] / np.sqrt(n_at), sims)

    # Plays-per-game noise
    h_pass_plays = np.clip(
        rng.normal(ht["avg_pass_plays"], ht["sd_pass_plays"], sims), 15, 55)
    h_rush_plays = np.clip(
        rng.normal(ht["avg_rush_plays"], ht["sd_rush_plays"], sims), 10, 45)
    a_pass_plays = np.clip(
        rng.normal(at["avg_pass_plays"], at["sd_pass_plays"], sims), 15, 55)
    a_rush_plays = np.clip(
        rng.normal(at["avg_rush_plays"], at["sd_rush_plays"], sims), 10, 45)

    home_scores = (home_base
                   + (h_off_noise - h_def_noise) * (h_pass_plays + h_rush_plays) * EPA_TO_PTS)
    away_scores = (away_base
                   + (a_off_noise - a_def_noise) * (a_pass_plays + a_rush_plays) * EPA_TO_PTS)

    # ── Game shocks ────────────────────────────────────────────────────────
    # Independent team shocks (t-dist, fat tails for blowouts/bad weather)
    seed1 = int(rng.integers(1_000_000))
    seed2 = int(rng.integers(1_000_000))
    seed3 = int(rng.integers(1_000_000))

    team_shock_h = stats.t.rvs(df=TEAM_SHOCK_DF, scale=TEAM_SHOCK_SCALE,
                                size=sims, random_state=seed1)
    team_shock_a = stats.t.rvs(df=TEAM_SHOCK_DF, scale=TEAM_SHOCK_SCALE,
                                size=sims, random_state=seed2)

    # Corr shock: adds to home-away margin but cancels in total
    # Models game-control effects (winning team runs clock, limits possessions)
    corr_shock   = stats.t.rvs(df=CORR_SHOCK_DF,  scale=CORR_SHOCK_SCALE,
                                size=sims, random_state=seed3)

    home_scores = home_scores + team_shock_h + 0.5 * corr_shock
    away_scores = away_scores + team_shock_a - 0.5 * corr_shock

    home_scores = np.clip(home_scores, 0, None)
    away_scores = np.clip(away_scores, 0, None)

    # ── Summary statistics ─────────────────────────────────────────────────
    spread_draws = home_scores - away_scores
    total_draws  = home_scores + away_scores

    spread_mean   = float(spread_draws.mean())
    spread_sd     = float(spread_draws.std())
    total_mean    = float(total_draws.mean())
    total_sd      = float(total_draws.std())
    home_win_prob = float((home_scores > away_scores).mean())
    away_win_prob = 1.0 - home_win_prob

    # ── ATS probabilities (Vegas spread used for grading ONLY) ────────────
    if vegas_spread is not None and np.isfinite(vegas_spread):
        # nflfastR convention: positive spread_line = home team favored
        # home covers when their margin exceeds the spread (e.g., spread=+3, home must win by >3)
        home_cover_prob = float((spread_draws > vegas_spread).mean())
        away_cover_prob = 1.0 - home_cover_prob
        ats_pick = ht["team"] if home_cover_prob >= away_cover_prob else at["team"]
        ats_prob = max(home_cover_prob, away_cover_prob)
    else:
        home_cover_prob = away_cover_prob = ats_prob = None
        ats_pick = None

    # ── Totals probabilities ───────────────────────────────────────────────
    if vegas_total is not None and np.isfinite(vegas_total):
        over_prob  = float((total_draws > vegas_total).mean())
        under_prob = 1.0 - over_prob
        total_pick = "Over" if over_prob >= under_prob else "Under"
        total_prob = max(over_prob, under_prob)
    else:
        over_prob = under_prob = total_prob = None
        total_pick = None

    def to_american(p):
        if p is None or not (0 < p < 1): return None
        return int(-round(p/(1-p)*100)) if p > 0.5 else int(round((1-p)/p*100))

    return {
        "home_team":        ht["team"],
        "away_team":        at["team"],
        "home_mean":        float(home_scores.mean()),
        "away_mean":        float(away_scores.mean()),
        "spread_mean":      spread_mean,
        "spread_sd":        spread_sd,
        "total_mean":       total_mean,
        "total_sd":         total_sd,
        "home_win_prob":    home_win_prob,
        "away_win_prob":    away_win_prob,
        "home_american":    to_american(home_win_prob),
        "away_american":    to_american(away_win_prob),
        "home_cover_prob":  home_cover_prob,
        "away_cover_prob":  away_cover_prob,
        "ats_pick":         ats_pick,
        "ats_prob":         ats_prob,
        "over_prob":        over_prob,
        "under_prob":       under_prob,
        "total_pick":       total_pick,
        "total_prob":       total_prob,
        "vegas_spread":     vegas_spread,
        "vegas_total":      vegas_total,
    }
