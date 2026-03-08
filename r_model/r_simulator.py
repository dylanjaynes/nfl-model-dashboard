"""
r_simulator.py  —  Python port of R/week_runner.R  run_one_game

Two-projector ensemble:
  1. PBP projector  (yards → points via yards-per-point)
  2. Eckel projector (drives × Eckel rate × pts/Eckel)
  Blend: home_base = (1 - w_eckel) * pbp + w_eckel * eckel

Game shocks (correlated t-distributed, matches R defaults):
  g_shock ~ t(df_total,  scale=sigma_total)   ← same for both teams (shifts total)
  s_shock ~ t(df_spread, scale=sigma_spread)  ← opposite sign (shifts spread only)

Bayesian shrinkage toward Vegas spread (Gaussian conjugate prior):
  mu_post = (mu_model/var_model + vegas_spread/tau^2) / (1/var_model + 1/tau^2)

ATS sign convention (nflfastR Python):
  positive spread_line = home team favored
  home covers when spread_draws > vegas_spread
"""

import numpy as np
from scipy import stats
from typing import Optional

# ── Default simulation constants (match R defaults in week_runner.R) ──────────
W_ECKEL      = 0.30    # Eckel projector weight
LEAGUE_AVG   = 22.5    # baseline pts per team per game
HFA          = 1.5     # home-field advantage pts (net 3 pt swing)
TAU          = 12.0    # Bayesian prior strength (tau_spread in R)

SIGMA_SPREAD = 6.5     # spread shock scale (sigma_spread in R)
DF_SPREAD    = 6       # spread shock df
SIGMA_TOTAL  = 3.0     # total shock scale (sigma_total in R)
DF_TOTAL     = 6       # total shock df

# Drives per game: R draws from N(11, 0.75), clipped [8,14]
AVG_DPG = 11.0
SD_DPG  = 0.75


# ── PBP Projector ─────────────────────────────────────────────────────────────
def _project_pbp(home: dict, away: dict, rng: np.random.Generator, sims: int) -> tuple:
    """
    Multiplicative YPC/YPA matchup (vectorized).

    proj_ypc_home = home_off_ypc * (1 + away_def_ypc_pct + noise)
    proj_ypa_home = home_off_ypa * (1 + away_def_ypa_pct + noise)
    total_yds_home = proj_ypc_home * rush_att + proj_ypa_home * pass_att
    score_home = total_yds_home / ypp_blended
    """
    # noise draws
    h_ypc_off_n = rng.normal(0, home["off_ypc_sd"], sims)
    h_ypa_off_n = rng.normal(0, home["off_ypa_sd"], sims)
    h_ra_n      = rng.normal(0, home["sd_rush_att"], sims)
    h_pa_n      = rng.normal(0, home["sd_pass_att"], sims)
    a_def_ypc_n = rng.normal(0, away["def_ypc_sd"], sims)
    a_def_ypa_n = rng.normal(0, away["def_ypa_sd"], sims)

    a_ypc_off_n = rng.normal(0, away["off_ypc_sd"], sims)
    a_ypa_off_n = rng.normal(0, away["off_ypa_sd"], sims)
    a_ra_n      = rng.normal(0, away["sd_rush_att"], sims)
    a_pa_n      = rng.normal(0, away["sd_pass_att"], sims)
    h_def_ypc_n = rng.normal(0, home["def_ypc_sd"], sims)
    h_def_ypa_n = rng.normal(0, home["def_ypa_sd"], sims)

    # projected YPC / YPA (multiplicative, matching R)
    h_proj_ypc = (home["off_ypc"] + h_ypc_off_n) * (1 + away["def_ypc_pct"] + a_def_ypc_n)
    h_proj_ypa = (home["off_ypa"] + h_ypa_off_n) * (1 + away["def_ypa_pct"] + a_def_ypa_n)

    a_proj_ypc = (away["off_ypc"] + a_ypc_off_n) * (1 + home["def_ypc_pct"] + h_def_ypc_n)
    a_proj_ypa = (away["off_ypa"] + a_ypa_off_n) * (1 + home["def_ypa_pct"] + h_def_ypa_n)

    # projected attempts
    h_ra = np.clip(home["avg_rush_att"] + h_ra_n, 10, 45)
    h_pa = np.clip(home["avg_pass_att"] + h_pa_n, 15, 55)
    a_ra = np.clip(away["avg_rush_att"] + a_ra_n, 10, 45)
    a_pa = np.clip(away["avg_pass_att"] + a_pa_n, 15, 55)

    h_rush_yds = h_proj_ypc * h_ra
    h_pass_yds = h_proj_ypa * h_pa
    a_rush_yds = a_proj_ypc * a_ra
    a_pass_yds = a_proj_ypa * a_pa

    h_tot = h_rush_yds + h_pass_yds
    a_tot = a_rush_yds + a_pass_yds

    # yards-per-point: blend home offense + away defense
    h_ypp = (home["off_ypp"] + away["def_ypp"]) / 2
    a_ypp = (away["off_ypp"] + home["def_ypp"]) / 2
    h_ypp = max(h_ypp, 8.0)
    a_ypp = max(a_ypp, 8.0)

    home_score = h_tot / h_ypp
    away_score = a_tot / a_ypp
    return home_score, away_score


# ── Eckel Projector ───────────────────────────────────────────────────────────
def _project_eckel(home: dict, away: dict, rng: np.random.Generator, sims: int) -> tuple:
    """
    Eckel rate / pts-per-Eckel blended from offense & defense.

    Matches R's project_matchup_eckel_vec:
      team1_eckel_rate = [(1 + away.eckel_vs_off_diff) * er_off_home
                         + (1 + home.eckel_vs_def_diff) * er_def_away] / 2
      score = drives_per_game * eckel_rate * ppe

    Uses truncated-normal draws for Eckel rates (clamped 0–0.85)
    and normal draws for PPE (clamped 2–5).
    """
    from scipy.stats import truncnorm

    def _tn(lo, hi, mu, sd, n):
        sd = max(sd, 1e-6)
        a, b = (lo - mu) / sd, (hi - mu) / sd
        return truncnorm.rvs(a, b, loc=mu, scale=sd, size=n)

    def _nr(mu, sd, n):
        return rng.normal(mu, max(sd, 1e-6), n)

    # Eckel rate draws
    er_off_h = _tn(0, 0.85, home["off_eckel_rate"], home["off_eckel_rate_sd"], sims)
    er_def_h = _tn(0, 0.85, home["def_eckel_rate"], home["def_eckel_rate_sd"], sims)
    er_off_a = _tn(0, 0.85, away["off_eckel_rate"], away["off_eckel_rate_sd"], sims)
    er_def_a = _tn(0, 0.85, away["def_eckel_rate"], away["def_eckel_rate_sd"], sims)

    # PPE draws (clipped 2–5 pts per Eckel drive)
    ppe_off_h = _tn(2, 5, home["off_ppe"], home["off_ppe_sd"], sims)
    ppe_def_h = _tn(2, 5, home["def_ppe"], home["def_ppe_sd"], sims)
    ppe_off_a = _tn(2, 5, away["off_ppe"], away["off_ppe_sd"], sims)
    ppe_def_a = _tn(2, 5, away["def_ppe"], away["def_ppe_sd"], sims)

    # Combine offense + defense effects (matches R formula)
    h_er = (
        (1 + away["eckel_vs_off_diff"]) * er_off_h
        + (1 + home["eckel_vs_def_diff"]) * er_def_a
    ) / 2
    a_er = (
        (1 + home["eckel_vs_off_diff"]) * er_off_a
        + (1 + away["eckel_vs_def_diff"]) * er_def_h
    ) / 2

    h_ppe = (
        (1 + home["ppe_vs_def_diff"]) * ppe_off_h
        + (1 + away["ppe_vs_off_diff"]) * ppe_def_a
    ) / 2
    a_ppe = (
        (1 + away["ppe_vs_def_diff"]) * ppe_off_a
        + (1 + home["ppe_vs_off_diff"]) * ppe_def_h
    ) / 2

    # Drives per game (same scalar for both teams, matches R)
    dpg = int(np.clip(round(rng.normal(AVG_DPG, SD_DPG)), 8, 14))

    home_score = dpg * np.clip(h_er, 0, 0.9) * np.clip(h_ppe, 1, 7)
    away_score = dpg * np.clip(a_er, 0, 0.9) * np.clip(a_ppe, 1, 7)
    return home_score, away_score


# ── Main simulation function ───────────────────────────────────────────────────
def simulate_r_game(
    home_m: dict,
    away_m: dict,
    vegas_spread: Optional[float] = None,
    vegas_total:  Optional[float] = None,
    sims:   int = 10_000,
    tau:    float = TAU,
    w_eckel: float = W_ECKEL,
    neutral: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> dict:
    """
    Monte Carlo R-model game simulation.

    vegas_spread: positive = home team favored (nflfastR convention).
    Bayesian shrinkage toward Vegas only if vegas_spread is provided.
    """
    if rng is None:
        rng = np.random.default_rng()

    # ── Base score projections ─────────────────────────────────────────────────
    hfa = 0.0 if neutral else HFA

    home_pbp, away_pbp = _project_pbp(home_m, away_m, rng, sims)
    home_eck, away_eck = _project_eckel(home_m, away_m, rng, sims)

    # Apply HFA to the PBP projector (Eckel drives are symmetric enough)
    home_pbp = home_pbp + hfa
    away_pbp = away_pbp - hfa

    # Blend
    home_blend = (1 - w_eckel) * home_pbp + w_eckel * home_eck
    away_blend = (1 - w_eckel) * away_pbp + w_eckel * away_eck

    # ── Correlated t-distributed shocks ───────────────────────────────────────
    seed1 = int(rng.integers(1_000_000))
    seed2 = int(rng.integers(1_000_000))

    g_shock = stats.t.rvs(df=DF_TOTAL,  scale=SIGMA_TOTAL,  size=sims, random_state=seed1)
    s_shock = stats.t.rvs(df=DF_SPREAD, scale=SIGMA_SPREAD, size=sims, random_state=seed2)

    home_blend = home_blend + g_shock + 0.5 * s_shock
    away_blend = away_blend + g_shock - 0.5 * s_shock

    # ── Bayesian shrinkage toward Vegas spread (Gaussian conjugate) ────────────
    spread_draws = home_blend - away_blend
    total_draws  = home_blend + away_blend

    if vegas_spread is not None and np.isfinite(vegas_spread):
        mu_model  = float(spread_draws.mean())
        var_model = float(spread_draws.var())
        tau2      = tau ** 2

        if var_model > 0:
            # precision-weighted average (matches R week_runner.R)
            prec_model = 1.0 / var_model
            prec_prior = 1.0 / tau2
            mu_post    = (mu_model * prec_model + vegas_spread * prec_prior) / (prec_model + prec_prior)
            delta      = mu_post - mu_model

            # shift spread draws; adjust implied team scores
            spread_draws = spread_draws + delta
            home_blend   = home_blend + 0.5 * delta
            away_blend   = away_blend - 0.5 * delta

    # Recompute total from adjusted team scores
    total_draws = home_blend + away_blend

    # Clip negative scores
    home_cal = np.clip(home_blend, 0, None)
    away_cal = np.clip(away_blend, 0, None)
    spread_draws = home_cal - away_cal
    total_draws  = home_cal + away_cal

    # ── Summary statistics ─────────────────────────────────────────────────────
    spread_mean   = float(spread_draws.mean())
    spread_sd     = float(spread_draws.std())
    total_mean    = float(total_draws.mean())
    total_sd      = float(total_draws.std())
    home_win_prob = float((home_cal > away_cal).mean())
    away_win_prob = 1.0 - home_win_prob

    # ── ATS probabilities ──────────────────────────────────────────────────────
    if vegas_spread is not None and np.isfinite(vegas_spread):
        # positive spread_line = home favored; home covers if margin > spread_line
        home_cover_prob = float((spread_draws > vegas_spread).mean())
        away_cover_prob = 1.0 - home_cover_prob
        ats_pick = home_m["team"] if home_cover_prob >= away_cover_prob else away_m["team"]
        ats_prob = max(home_cover_prob, away_cover_prob)
    else:
        home_cover_prob = away_cover_prob = ats_prob = None
        ats_pick = None

    # ── Totals probabilities ───────────────────────────────────────────────────
    if vegas_total is not None and np.isfinite(vegas_total):
        over_prob  = float((total_draws > vegas_total).mean())
        under_prob = 1.0 - over_prob
        total_pick = "Over" if over_prob >= under_prob else "Under"
        total_prob = max(over_prob, under_prob)
    else:
        over_prob = under_prob = total_prob = None
        total_pick = None

    def to_american(p):
        if p is None or not (0 < p < 1):
            return None
        return int(-round(p / (1 - p) * 100)) if p > 0.5 else int(round((1 - p) / p * 100))

    return {
        "home_team":       home_m["team"],
        "away_team":       away_m["team"],
        "home_mean":       float(home_cal.mean()),
        "away_mean":       float(away_cal.mean()),
        "spread_mean":     spread_mean,
        "spread_sd":       spread_sd,
        "total_mean":      total_mean,
        "total_sd":        total_sd,
        "home_win_prob":   home_win_prob,
        "away_win_prob":   away_win_prob,
        "home_american":   to_american(home_win_prob),
        "away_american":   to_american(away_win_prob),
        "home_cover_prob": home_cover_prob,
        "away_cover_prob": away_cover_prob,
        "ats_pick":        ats_pick,
        "ats_prob":        ats_prob,
        "over_prob":       over_prob,
        "under_prob":      under_prob,
        "total_pick":      total_pick,
        "total_prob":      total_prob,
        "vegas_spread":    vegas_spread,
        "vegas_total":     vegas_total,
    }
