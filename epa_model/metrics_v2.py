"""
metrics_v2.py — Enhanced EPA team metrics

Improvements over v1:
  • Recency-weighted (recent games count more, half-life ~5.8 weeks)
  • Pass/rush EPA split (separately predictive signals)
  • Turnover-neutral (strip INTs and fumbles lost — these are noise, not skill)
  • Strength-of-schedule adjusted (one-pass opponent quality correction)
"""

import numpy as np
import pandas as pd

DECAY_RATE   = 0.12   # exponential decay rate (weeks); half-life = ln2/0.12 ≈ 5.8 wks
MIN_SD       = 0.02   # floor on EPA standard deviation
FALLBACK_SD  = 0.08   # SD for teams with < 2 games


def compute_all_teams_v2(pbp: pd.DataFrame, through_week: int,
                          decay: float = DECAY_RATE) -> dict:
    """
    Compute SOS-adjusted, recency-weighted, turnover-neutral EPA metrics
    for every team with data through `through_week`.

    Returns dict keyed by team abbreviation.
    """
    # ── Filter PBP ─────────────────────────────────────────────────────────
    reg_mask = pbp["season_type"] == "REG" if "season_type" in pbp.columns \
               else pd.Series(True, index=pbp.index)
    df = pbp[
        reg_mask &
        (pbp["week"] <= through_week) &
        pbp["epa"].notna() &
        ((pbp["rush_attempt"] == 1) | (pbp["pass_attempt"] == 1))
    ].copy()

    df["week"] = pd.to_numeric(df["week"], errors="coerce")

    if df.empty:
        return {}

    # ── Turnover mask (plays ending in turnovers — noisy, not skill) ───────
    to_mask = pd.Series(False, index=df.index)
    if "interception" in df.columns:
        to_mask |= (df["interception"] == 1)
    if "fumble_lost" in df.columns:
        to_mask |= (df["fumble_lost"] == 1)
    df_tn = df[~to_mask]   # turnover-neutral plays

    # ── Recency weights at the game level ──────────────────────────────────
    # Assign each play the weight of its game (exp decay by weeks ago)
    df_tn = df_tn.copy()
    df_tn["_wt"] = np.exp(-decay * (through_week - df_tn["week"]))

    # ── Identify all teams ─────────────────────────────────────────────────
    teams = (set(df_tn["posteam"].dropna().unique()) |
             set(df_tn["defteam"].dropna().unique()))
    teams = {t for t in teams if isinstance(t, str) and 1 < len(t) <= 3}

    # ── Compute raw (pre-SOS) metrics for every team ───────────────────────
    raw = {t: _compute_raw(t, df_tn) for t in sorted(teams)}

    # ── SOS adjustment (one pass) ─────────────────────────────────────────
    # For each team's offense, reduce/inflate by the mean defensive quality
    # of the opponents they faced.
    _sos_adjust(raw, df_tn)

    return raw


# ── Internal helpers ────────────────────────────────────────────────────────

def _compute_raw(team: str, df_tn: pd.DataFrame) -> dict:
    """Recency-weighted, turnover-neutral metrics for one team (pre-SOS)."""

    # ── OFFENSE ──────────────────────────────────────────────────────────
    off = df_tn[df_tn["posteam"] == team]
    pass_off = off[off["pass_attempt"] == 1]
    rush_off = off[off["rush_attempt"] == 1]

    if off.empty:
        return _empty(team)

    # Game-level weighted EPA (weighted mean within each game, then weighted across games)
    def _game_weighted_epa(plays):
        if plays.empty:
            return pd.Series(dtype=float)
        gdf = plays.groupby("game_id").apply(
            lambda g: pd.Series({
                "epa_mean": float(np.average(g["epa"], weights=g["_wt"])),
                "game_wt":  float(g["_wt"].mean()),   # representative game weight
                "n_plays":  len(g),
            })
        )
        return gdf

    off_games     = _game_weighted_epa(off)
    pass_games    = _game_weighted_epa(pass_off)
    rush_games    = _game_weighted_epa(rush_off)

    n_games = len(off_games)

    def _wmean(gdf):
        if gdf.empty: return 0.0
        return float(np.average(gdf["epa_mean"], weights=gdf["game_wt"]))

    def _wsd(gdf):
        if len(gdf) < 2: return FALLBACK_SD
        mu = _wmean(gdf)
        w  = gdf["game_wt"].values
        v  = gdf["epa_mean"].values
        return float(np.sqrt(np.average((v - mu) ** 2, weights=w)))

    off_epa      = _wmean(off_games)
    pass_off_epa = _wmean(pass_games) if not pass_games.empty else 0.0
    rush_off_epa = _wmean(rush_games) if not rush_games.empty else 0.0
    off_epa_sd   = _wsd(off_games)

    avg_pass_plays = float(pass_games["n_plays"].mean()) if not pass_games.empty else 35.0
    avg_rush_plays = float(rush_games["n_plays"].mean()) if not rush_games.empty else 27.0
    sd_pass_plays  = float(pass_games["n_plays"].std())  if len(pass_games) > 1 else 8.0
    sd_rush_plays  = float(rush_games["n_plays"].std())  if len(rush_games) > 1 else 7.0

    # ── DEFENSE ──────────────────────────────────────────────────────────
    def_ = df_tn[df_tn["defteam"] == team]
    pass_def = def_[def_["pass_attempt"] == 1]
    rush_def = def_[def_["rush_attempt"] == 1]

    def_games     = _game_weighted_epa(def_)
    pass_def_games = _game_weighted_epa(pass_def)
    rush_def_games = _game_weighted_epa(rush_def)

    def_epa      = _wmean(def_games)      if not def_games.empty      else 0.0
    pass_def_epa = _wmean(pass_def_games) if not pass_def_games.empty else 0.0
    rush_def_epa = _wmean(rush_def_games) if not rush_def_games.empty else 0.0
    def_epa_sd   = _wsd(def_games)        if not def_games.empty      else FALLBACK_SD

    return {
        "team":           team,
        "off_epa":        off_epa,
        "def_epa":        def_epa,
        "pass_off_epa":   pass_off_epa,
        "rush_off_epa":   rush_off_epa,
        "pass_def_epa":   pass_def_epa,
        "rush_def_epa":   rush_def_epa,
        "off_epa_sd":     max(off_epa_sd,  MIN_SD),
        "def_epa_sd":     max(def_epa_sd,  MIN_SD),
        "avg_pass_plays": avg_pass_plays,
        "avg_rush_plays": avg_rush_plays,
        "sd_pass_plays":  max(sd_pass_plays, 5.0),
        "sd_rush_plays":  max(sd_rush_plays, 5.0),
        "n_games":        n_games,
        # raw copies preserved for SOS delta computation
        "_raw_off_epa":       off_epa,
        "_raw_pass_off_epa":  pass_off_epa,
        "_raw_rush_off_epa":  rush_off_epa,
        "_raw_def_epa":       def_epa,
        "_raw_pass_def_epa":  pass_def_epa,
        "_raw_rush_def_epa":  rush_def_epa,
    }


def _sos_adjust(raw: dict, df_tn: pd.DataFrame):
    """
    One-pass SOS adjustment (in-place).

    For each team's offense: subtract the mean defensive EPA of opponents
    they faced (higher opp def_epa = weaker opponents → reduce credit).
    """
    # Build opponent lookup from matchup pairs
    matchups = df_tn[["game_id", "posteam", "defteam"]].drop_duplicates()

    for team, m in raw.items():
        # Opponents this team faced as offense
        opp_as_def = matchups.loc[matchups["posteam"] == team, "defteam"].unique()
        opp_def_epas = [raw[o]["_raw_def_epa"] for o in opp_as_def if o in raw]
        sos_off = float(np.mean(opp_def_epas)) if opp_def_epas else 0.0

        # Opponents this team faced as defense
        opp_as_off = matchups.loc[matchups["defteam"] == team, "posteam"].unique()
        opp_off_epas = [raw[o]["_raw_off_epa"] for o in opp_as_off if o in raw]
        sos_def = float(np.mean(opp_off_epas)) if opp_off_epas else 0.0

        # Adjust offense: subtract opponent def quality
        # (if faced strong defenses → boost; weak defenses → reduce)
        m["off_epa"]      = m["_raw_off_epa"]      - sos_off
        m["pass_off_epa"] = m["_raw_pass_off_epa"]  - sos_off
        m["rush_off_epa"] = m["_raw_rush_off_epa"]  - sos_off * 0.5

        # Adjust defense: subtract opponent off quality
        m["def_epa"]      = m["_raw_def_epa"]      - sos_def
        m["pass_def_epa"] = m["_raw_pass_def_epa"]  - sos_def
        m["rush_def_epa"] = m["_raw_rush_def_epa"]  - sos_def * 0.5


def _empty(team: str) -> dict:
    """League-average fallback for teams with no data."""
    return {
        "team":           team,
        "off_epa":        0.0,  "def_epa":        0.0,
        "pass_off_epa":   0.0,  "rush_off_epa":   0.0,
        "pass_def_epa":   0.0,  "rush_def_epa":   0.0,
        "off_epa_sd":     FALLBACK_SD,
        "def_epa_sd":     FALLBACK_SD,
        "avg_pass_plays": 35.0, "avg_rush_plays": 27.0,
        "sd_pass_plays":  8.0,  "sd_rush_plays":  7.0,
        "n_games":        0,
        "_raw_off_epa": 0.0, "_raw_pass_off_epa": 0.0, "_raw_rush_off_epa": 0.0,
        "_raw_def_epa": 0.0, "_raw_pass_def_epa": 0.0, "_raw_rush_def_epa": 0.0,
    }
