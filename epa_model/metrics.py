"""
metrics.py — EPA team metric computation
Computes offensive and defensive EPA/play per team from nfl_data_py PBP.
"""

import numpy as np
import pandas as pd


def compute_team_epa(team: str, pbp: pd.DataFrame, through_week: int) -> dict:
    """
    Compute EPA-based team metrics through a given week.

    Parameters
    ----------
    team : str
        NFL team abbreviation (e.g. 'KC', 'BUF')
    pbp : pd.DataFrame
        Full season play-by-play from nfl_data_py
    through_week : int
        Only use plays from weeks <= this value (no lookahead)

    Returns
    -------
    dict with keys:
        off_epa        : mean offensive EPA per play
        def_epa        : mean defensive EPA per play allowed
        off_epa_sd     : std of game-level mean off EPA (game-to-game variance)
        def_epa_sd     : std of game-level mean def EPA
        avg_plays      : mean offensive plays per game
        sd_plays       : std of offensive plays per game
        n_games        : number of games in sample
    """
    # Filter to requested week range and valid regular-season plays (exclude playoffs)
    reg_mask = pbp["season_type"] == "REG" if "season_type" in pbp.columns else pd.Series(True, index=pbp.index)
    df = pbp[
        reg_mask &
        (pbp["week"] <= through_week) &
        (pbp["epa"].notna()) &
        (
            (pbp["rush_attempt"] == 1) |
            (pbp["pass_attempt"] == 1)
        )
    ].copy()

    # ----- OFFENSE -----
    off_plays = df[df["posteam"] == team]

    if len(off_plays) == 0:
        return _empty_metrics(team)

    # Game-level offensive EPA means
    off_by_game = (
        off_plays.groupby("game_id")["epa"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "epa_mean", "count": "n_plays"})
    )

    off_epa     = float(off_plays["epa"].mean())
    off_epa_sd  = float(off_by_game["epa_mean"].std()) if len(off_by_game) > 1 else 0.05
    avg_plays   = float(off_by_game["n_plays"].mean())
    sd_plays    = float(off_by_game["n_plays"].std()) if len(off_by_game) > 1 else 5.0
    n_games     = len(off_by_game)

    # ----- DEFENSE -----
    def_plays = df[df["defteam"] == team]

    if len(def_plays) == 0:
        def_epa    = 0.0
        def_epa_sd = 0.05
    else:
        def_by_game = (
            def_plays.groupby("game_id")["epa"]
            .mean()
        )
        def_epa    = float(def_plays["epa"].mean())
        def_epa_sd = float(def_by_game.std()) if len(def_by_game) > 1 else 0.05

    return {
        "team":       team,
        "off_epa":    off_epa,
        "def_epa":    def_epa,
        "off_epa_sd": max(off_epa_sd, 0.02),   # floor to avoid zero noise
        "def_epa_sd": max(def_epa_sd, 0.02),
        "avg_plays":  avg_plays,
        "sd_plays":   max(sd_plays, 3.0),
        "n_games":    n_games,
    }


def _empty_metrics(team: str) -> dict:
    """League-average fallback when no data exists (e.g. Week 1 expansion teams)."""
    return {
        "team":       team,
        "off_epa":    0.0,
        "def_epa":    0.0,
        "off_epa_sd": 0.08,
        "def_epa_sd": 0.08,
        "avg_plays":  65.0,
        "sd_plays":   5.0,
        "n_games":    0,
    }


def compute_all_teams(pbp: pd.DataFrame, through_week: int) -> dict:
    """
    Compute EPA metrics for every team in the PBP through a given week.
    Returns a dict keyed by team abbreviation.
    """
    teams = set(pbp["posteam"].dropna().unique()) | set(pbp["defteam"].dropna().unique())
    teams = {t for t in teams if isinstance(t, str) and len(t) <= 3}

    return {
        team: compute_team_epa(team, pbp, through_week)
        for team in sorted(teams)
    }
