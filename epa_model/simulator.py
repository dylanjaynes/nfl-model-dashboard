"""
simulator.py — Monte Carlo game simulation engine (EPA-based)
"""

import numpy as np
from scipy import stats
from typing import Optional


# ── Model constants ────────────────────────────────────────────────────────────
LEAGUE_AVG_PTS  = 23.0    # historical NFL mean points per team per game
EPA_TO_PTS      = 0.45    # EPA/play → points conversion (~0.42–0.48 in literature)
BASE_PLAYS      = 65.0    # average offensive plays per game
HFA             = 1.5     # home field advantage in points (net 3pt swing)
GAME_SHOCK_SCALE = 3.0    # scale of t-distributed game-level shock
GAME_SHOCK_DF   = 6       # degrees of freedom for shock (fat tails)
BAYESIAN_TAU    = 14.0    # prior SD for Bayesian shrinkage toward Vegas spread


def simulate_game(
    home_metrics: dict,
    away_metrics: dict,
    vegas_spread: Optional[float] = None,   # home team perspective (negative = home favored)
    vegas_total:  Optional[float] = None,
    sims: int = 5000,
    rng: Optional[np.random.Generator] = None,
) -> dict:
    """
    Run a Monte Carlo simulation for one game.

    Parameters
    ----------
    home_metrics : dict   output of metrics.compute_team_epa() for home team
    away_metrics : dict   output of metrics.compute_team_epa() for away team
    vegas_spread : float  closing spread, home perspective (e.g. -3.0 = home -3)
    vegas_total  : float  closing total
    sims         : int    number of simulation draws
    rng          : numpy Generator (for reproducibility)

    Returns
    -------
    dict with prediction summary
    """
    if rng is None:
        rng = np.random.default_rng()

    ht = home_metrics
    at = away_metrics

    # ── EPA matchup edges ──────────────────────────────────────────────────────
    home_epa_edge = ht["off_epa"] - at["def_epa"]   # home offense vs away defense
    away_epa_edge = at["off_epa"] - ht["def_epa"]   # away offense vs home defense

    # ── Base projected scores ──────────────────────────────────────────────────
    n_ht = max(ht["n_games"], 1)
    n_at = max(at["n_games"], 1)

    home_base = LEAGUE_AVG_PTS + home_epa_edge * BASE_PLAYS * EPA_TO_PTS + HFA
    away_base = LEAGUE_AVG_PTS + away_epa_edge * BASE_PLAYS * EPA_TO_PTS - HFA

    # ── Vectorized noise draws (sims) ─────────────────────────────────────────

    # EPA uncertainty scales with 1/sqrt(n_games) — more games = more confident
    home_off_noise = rng.normal(0, ht["off_epa_sd"] / np.sqrt(n_ht), sims)
    home_def_noise = rng.normal(0, at["def_epa_sd"] / np.sqrt(n_at), sims)
    away_off_noise = rng.normal(0, at["off_epa_sd"] / np.sqrt(n_at), sims)
    away_def_noise = rng.normal(0, ht["def_epa_sd"] / np.sqrt(n_ht), sims)

    # Plays per game noise (each game has different pace)
    home_plays = np.clip(rng.normal(ht["avg_plays"], ht["sd_plays"], sims), 40, 90)
    away_plays = np.clip(rng.normal(at["avg_plays"], at["sd_plays"], sims), 40, 90)

    # Translate noisy EPA → score adjustments
    home_scores = (
        home_base
        + (home_off_noise - home_def_noise) * home_plays * EPA_TO_PTS
    )
    away_scores = (
        away_base
        + (away_off_noise - away_def_noise) * away_plays * EPA_TO_PTS
    )

    # ── Fat-tailed game shock (same-direction = total shock, opposite = spread shock)
    total_shock  = stats.t.rvs(df=GAME_SHOCK_DF, scale=GAME_SHOCK_SCALE, size=sims, random_state=rng.integers(1e9))
    spread_shock = stats.t.rvs(df=GAME_SHOCK_DF, scale=GAME_SHOCK_SCALE, size=sims, random_state=rng.integers(1e9))

    home_scores += total_shock + 0.5 * spread_shock
    away_scores += total_shock - 0.5 * spread_shock

    # Clip to non-negative
    home_scores = np.clip(home_scores, 0, None)
    away_scores = np.clip(away_scores, 0, None)

    # ── Bayesian shrinkage toward Vegas spread (if provided) ───────────────────
    spread_draws = home_scores - away_scores

    if vegas_spread is not None and np.isfinite(vegas_spread):
        mu_model = float(spread_draws.mean())
        var_model = float(spread_draws.var())
        tau2 = BAYESIAN_TAU ** 2

        if var_model > 0:
            mu_post = (mu_model / var_model + vegas_spread / tau2) / (1 / var_model + 1 / tau2)
            delta = mu_post - mu_model
            home_scores += 0.5 * delta
            away_scores -= 0.5 * delta
            spread_draws = home_scores - away_scores

    total_draws = home_scores + away_scores

    # ── Summary statistics ─────────────────────────────────────────────────────
    spread_mean = float(spread_draws.mean())
    total_mean  = float(total_draws.mean())
    spread_sd   = float(spread_draws.std())
    total_sd    = float(total_draws.std())

    home_win_prob = float((home_scores > away_scores).mean())
    away_win_prob = 1.0 - home_win_prob

    # ── ATS probabilities ──────────────────────────────────────────────────────
    if vegas_spread is not None and np.isfinite(vegas_spread):
        # Home covers if actual margin > -spread_line (spread_line is negative for home fav)
        home_cover_prob = float((spread_draws > -vegas_spread).mean())
        away_cover_prob = 1.0 - home_cover_prob
        ats_pick = ht["team"] if home_cover_prob >= away_cover_prob else at["team"]
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

    # ── American odds helper ───────────────────────────────────────────────────
    def to_american(p: float) -> Optional[int]:
        if p is None or not (0 < p < 1):
            return None
        return int(-round(p / (1 - p) * 100)) if p > 0.5 else int(round((1 - p) / p * 100))

    return {
        "home_team":         ht["team"],
        "away_team":         at["team"],
        "home_mean":         float(home_scores.mean()),
        "away_mean":         float(away_scores.mean()),
        "spread_mean":       spread_mean,
        "total_mean":        total_mean,
        "spread_sd":         spread_sd,
        "total_sd":          total_sd,
        "home_win_prob":     home_win_prob,
        "away_win_prob":     away_win_prob,
        "home_american":     to_american(home_win_prob),
        "away_american":     to_american(away_win_prob),
        "home_cover_prob":   home_cover_prob,
        "away_cover_prob":   away_cover_prob,
        "ats_pick":          ats_pick,
        "ats_prob":          ats_prob,
        "over_prob":         over_prob,
        "under_prob":        under_prob,
        "total_pick":        total_pick,
        "total_prob":        total_prob,
        "vegas_spread":      vegas_spread,
        "vegas_total":       vegas_total,
    }
