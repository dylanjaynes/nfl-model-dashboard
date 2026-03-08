"""
r_metrics.py  —  Python port of R/metrics.R  calculate_team_metrics

Computes all NFL team metrics needed for the R-model simulator:
  • PBP side:  YPC, YPA (adjusted net), yards-per-point, attempt counts
  • Eckel side: Eckel rate, pts/Eckel drive, turnover rate, field position

All metrics are computed through `through_week` (no lookahead).
Drive identity uses composite key: game_id + "_" + fixed_drive.
"""

import os
import numpy as np
import pandas as pd
import streamlit as st

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE      = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "epa_model", "cache")

# ── Drive-result → points mapping (same as epa_model/app.py) ─────────────────
DRIVE_PTS = {
    "Touchdown": 6.8, "Field goal": 3.0,
    "Opp touchdown": 0.0, "Safety": 0.0,
    "End of half": 0.0, "Turnover on downs": 0.0,
    "Punt": 0.0, "Turnover": 0.0, "Missing": 0.0,
}

# ── Team / conference helpers ─────────────────────────────────────────────────
ALIASES = {"OAK": "LV", "SD": "LAC", "STL": "LA"}
CONF_MAP = {
    "BUF":"AFC","MIA":"AFC","NE":"AFC","NYJ":"AFC",
    "BAL":"AFC","CIN":"AFC","CLE":"AFC","PIT":"AFC",
    "HOU":"AFC","IND":"AFC","JAX":"AFC","TEN":"AFC",
    "DEN":"AFC","KC":"AFC","LV":"AFC","LAC":"AFC",
    "DAL":"NFC","NYG":"NFC","PHI":"NFC","WAS":"NFC",
    "CHI":"NFC","DET":"NFC","GB":"NFC","MIN":"NFC",
    "ATL":"NFC","CAR":"NFC","NO":"NFC","TB":"NFC",
    "ARI":"NFC","LA":"NFC","SF":"NFC","SEA":"NFC",
}
NFL_TEAMS = [
    "ARI","ATL","BAL","BUF","CAR","CHI","CIN","CLE",
    "DAL","DEN","DET","GB","HOU","IND","JAX","KC",
    "LA","LAC","LV","MIA","MIN","NE","NO","NYG",
    "NYJ","PHI","PIT","SEA","SF","TB","TEN","WAS",
]

def _norm(t):
    return ALIASES.get(t, t) if isinstance(t, str) else t


# ── Data loader ───────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading play-by-play data…")
def load_pbp(season: int) -> pd.DataFrame:
    f = os.path.join(CACHE_DIR, f"pbp_{season}.parquet")
    if os.path.exists(f):
        pbp = pd.read_parquet(f)
    else:
        try:
            import nfl_data_py as nfl
            pbp = nfl.import_pbp_data([season], downcast=True)
            os.makedirs(CACHE_DIR, exist_ok=True)
            pbp.to_parquet(f, index=False)
        except Exception as e:
            st.error(f"Could not load PBP for {season}: {e}")
            return pd.DataFrame()
    for col in ["posteam", "defteam", "home_team", "away_team"]:
        if col in pbp.columns:
            pbp[col] = pbp[col].map(_norm)
    return pbp


@st.cache_data(show_spinner="Loading schedule…")
def load_schedule(season: int) -> pd.DataFrame:
    for stem in [f"schedule_{season}_full", f"schedule_{season}"]:
        f = os.path.join(CACHE_DIR, f"{stem}.parquet")
        if os.path.exists(f):
            df = pd.read_parquet(f)
            df["home_team"] = df["home_team"].map(_norm)
            df["away_team"] = df["away_team"].map(_norm)
            return df
    try:
        import nfl_data_py as nfl
        df = nfl.import_schedules([season])
        os.makedirs(CACHE_DIR, exist_ok=True)
        df.to_parquet(os.path.join(CACHE_DIR, f"schedule_{season}_full.parquet"), index=False)
        return df
    except Exception as e:
        st.error(f"Could not load schedule for {season}: {e}")
        return pd.DataFrame()


# ── Helpers ───────────────────────────────────────────────────────────────────
def _safe_float(x, default=0.0):
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def _safe_div(num, den, default=0.0):
    """Element-wise safe division (numpy/scalar)."""
    try:
        d = np.asarray(den, dtype=float)
        n = np.asarray(num, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            result = np.where(np.abs(d) > 1e-10, n / d, default)
        return float(result) if result.ndim == 0 else result
    except Exception:
        return default


# ── Main metrics function ─────────────────────────────────────────────────────
@st.cache_data(show_spinner="Computing R-model team metrics…")
def compute_r_metrics(season: int, through_week: int):
    """
    Returns (team_metrics: dict, league_avgs: dict, rankings: dict).

    team_metrics[team] = flat dict of all metrics for the simulator.
    league_avgs = scalar league-wide averages used in the simulator.
    rankings[team][metric] = integer rank (1 = best).
    """
    pbp = load_pbp(season)
    if pbp.empty:
        return {}, {}, {}

    # ── Filter ────────────────────────────────────────────────────────────────
    reg = pbp[pbp["season_type"] == "REG"].copy()
    if through_week > 0:
        reg = reg[reg["week"] <= through_week]
    if reg.empty:
        return {}, {}, {}

    # Composite drive key (fixed_drive resets to 1 at start of every game)
    reg["_drv"] = reg["game_id"].astype(str) + "_" + reg["fixed_drive"].astype(str)

    # Rush/pass plays
    plays = reg[
        (reg["rush_attempt"].fillna(0) == 1) | (reg["pass_attempt"].fillna(0) == 1)
    ].copy()

    # Sack yards (negative yards_gained on sack plays)
    has_sack = "sack" in plays.columns
    plays["_is_sack"]  = plays["sack"].fillna(0).astype(int) if has_sack else 0
    plays["_sack_yds"] = np.where(plays["_is_sack"] == 1, plays["yards_gained"].fillna(0), 0.0)

    # Touchdown indicator (covers pass_touchdown, rush_touchdown, or combined)
    if "touchdown" in plays.columns:
        plays["_td"] = plays["touchdown"].fillna(0).astype(int)
    elif "pass_touchdown" in plays.columns and "rush_touchdown" in plays.columns:
        plays["_td"] = (plays["pass_touchdown"].fillna(0) + plays["rush_touchdown"].fillna(0)).clip(upper=1).astype(int)
    else:
        plays["_td"] = 0

    if "interception" in plays.columns:
        plays["_int"] = plays["interception"].fillna(0).astype(int)
    else:
        plays["_int"] = 0

    # ── PART 1 : PBP yards-based metrics ──────────────────────────────────────

    # — Offensive YPC per game (all teams) —
    rush_g = (
        plays[plays["rush_attempt"].fillna(0) == 1]
        .groupby(["posteam", "game_id", "defteam"])
        .agg(rush_yds=("yards_gained", "sum"), rush_att=("rush_attempt", "sum"))
        .reset_index()
    )
    rush_g["off_ypc"] = rush_g["rush_yds"] / rush_g["rush_att"].clip(lower=1)

    # — Offensive adjusted-net YPA per game —
    pass_g_off = (
        plays[plays["pass_attempt"].fillna(0) == 1]
        .groupby(["posteam", "game_id", "defteam"])
        .agg(
            pass_yds=("yards_gained", "sum"),
            pass_att=("pass_attempt", "sum"),
            pass_td=("_td", "sum"),
            pass_int=("_int", "sum"),
            sack_yds=("_sack_yds", "sum"),
            sacks=("_is_sack", "sum"),
        )
        .reset_index()
    )
    pass_g_off["off_ypa"] = (
        pass_g_off["pass_yds"]
        + 20 * pass_g_off["pass_td"]
        - 45 * pass_g_off["pass_int"]
        - pass_g_off["sack_yds"].abs()
    ) / (pass_g_off["pass_att"] + pass_g_off["sacks"]).clip(lower=1)

    # Merge rush + pass per game
    off_game = rush_g[["posteam", "game_id", "defteam", "off_ypc", "rush_att"]].merge(
        pass_g_off[["posteam", "game_id", "off_ypa", "pass_att", "sacks"]],
        on=["posteam", "game_id"], how="outer"
    ).fillna({"off_ypc": 0.0, "off_ypa": 0.0, "rush_att": 0.0, "pass_att": 0.0, "sacks": 0.0})

    # Season-level offense summaries
    off_season = off_game.groupby("posteam").agg(
        off_ypc=("off_ypc", "mean"),
        off_ypa=("off_ypa", "mean"),
        off_ypc_sd=("off_ypc", "std"),
        off_ypa_sd=("off_ypa", "std"),
        avg_rush_att=("rush_att", "mean"),
        avg_pass_att=("pass_att", "mean"),
        sd_rush_att=("rush_att", "std"),
        sd_pass_att=("pass_att", "std"),
        n_games=("game_id", "nunique"),
    ).reset_index().fillna(0.0)

    # — Defensive YPC and YPA per game —
    rush_g_def = (
        plays[plays["rush_attempt"].fillna(0) == 1]
        .groupby(["defteam", "game_id", "posteam"])
        .agg(rush_yds=("yards_gained", "sum"), rush_att=("rush_attempt", "sum"))
        .reset_index()
    )
    rush_g_def["def_ypc_raw"] = rush_g_def["rush_yds"] / rush_g_def["rush_att"].clip(lower=1)

    pass_g_def = (
        plays[plays["pass_attempt"].fillna(0) == 1]
        .groupby(["defteam", "game_id", "posteam"])
        .agg(
            pass_yds=("yards_gained", "sum"),
            pass_att=("pass_attempt", "sum"),
            pass_td=("_td", "sum"),
            pass_int=("_int", "sum"),
            sack_yds=("_sack_yds", "sum"),
            sacks=("_is_sack", "sum"),
        )
        .reset_index()
    )
    pass_g_def["def_ypa_raw"] = (
        pass_g_def["pass_yds"]
        + 20 * pass_g_def["pass_td"]
        - 45 * pass_g_def["pass_int"]
        - pass_g_def["sack_yds"].abs()
    ) / (pass_g_def["pass_att"] + pass_g_def["sacks"]).clip(lower=1)

    def_game = rush_g_def[["defteam", "game_id", "posteam", "def_ypc_raw"]].merge(
        pass_g_def[["defteam", "game_id", "def_ypa_raw"]],
        on=["defteam", "game_id"], how="outer"
    ).fillna({"def_ypc_raw": 0.0, "def_ypa_raw": 0.0})

    # Defensive percentage modifier vs opponent's seasonal average
    # ypc_diff_allowed = (def_ypc - opp_avg_ypc) / opp_avg_ypc
    opp_off_avg = off_season[["posteam", "off_ypc", "off_ypa"]].rename(
        columns={"posteam": "posteam", "off_ypc": "opp_avg_ypc", "off_ypa": "opp_avg_ypa"}
    )
    def_game_adj = def_game.merge(opp_off_avg, left_on="posteam", right_on="posteam", how="left")
    def_game_adj["ypc_diff_pct"] = (def_game_adj["def_ypc_raw"] - def_game_adj["opp_avg_ypc"]) / \
                                    def_game_adj["opp_avg_ypc"].clip(lower=1e-6)
    def_game_adj["ypa_diff_pct"] = (def_game_adj["def_ypa_raw"] - def_game_adj["opp_avg_ypa"]) / \
                                    def_game_adj["opp_avg_ypa"].clip(lower=1e-6)

    def_season = def_game_adj.groupby("defteam").agg(
        def_ypc_pct=("ypc_diff_pct", "mean"),
        def_ypa_pct=("ypa_diff_pct", "mean"),
        def_ypc_sd=("ypc_diff_pct", "std"),
        def_ypa_sd=("ypa_diff_pct", "std"),
        def_ypc_raw=("def_ypc_raw", "mean"),
        def_ypa_raw=("def_ypa_raw", "mean"),
    ).reset_index().fillna(0.0)

    # — Yards per point (off/def) —
    # Offense: total yards / total points scored per game, then average
    if "posteam_score" in reg.columns:
        ypp_off_g = (
            reg[reg["posteam"].notna()]
            .groupby(["posteam", "game_id"])
            .agg(tot_yds=("yards_gained", "sum"), pts=("posteam_score", "max"))
            .reset_index()
        )
        ypp_off_g["ypp"] = ypp_off_g["tot_yds"] / ypp_off_g["pts"].clip(lower=1.0)
        off_ypp_s = ypp_off_g.groupby("posteam")["ypp"].mean().reset_index()
        off_ypp_s.columns = ["posteam", "off_ypp"]

        ypp_def_g = (
            reg[reg["defteam"].notna()]
            .groupby(["defteam", "game_id"])
            .agg(tot_yds=("yards_gained", "sum"), pts=("posteam_score", "max"))
            .reset_index()
        )
        ypp_def_g["ypp"] = ypp_def_g["tot_yds"] / ypp_def_g["pts"].clip(lower=1.0)
        def_ypp_s = ypp_def_g.groupby("defteam")["ypp"].mean().reset_index()
        def_ypp_s.columns = ["defteam", "def_ypp"]
    else:
        off_ypp_s = off_season[["posteam"]].assign(off_ypp=14.0)
        def_ypp_s = def_season[["defteam"]].assign(def_ypp=14.0)

    # Yards per play (for display)
    off_ypp_play = plays.groupby("posteam")["yards_gained"].mean().reset_index()
    off_ypp_play.columns = ["posteam", "off_ypp_play"]
    def_ypp_play = plays.groupby("defteam")["yards_gained"].mean().reset_index()
    def_ypp_play.columns = ["defteam", "def_ypp_play"]

    # ── PART 2 : Drive-based Eckel metrics ────────────────────────────────────
    drive_plays = reg.copy()
    if "kickoff_attempt" in drive_plays.columns:
        drive_plays = drive_plays[drive_plays["kickoff_attempt"].fillna(0) == 0]
    if "punt_attempt" in drive_plays.columns:
        drive_plays = drive_plays[drive_plays["punt_attempt"].fillna(0) == 0]

    # Eckel event per play
    if "first_down" in drive_plays.columns:
        _fd = drive_plays["first_down"].fillna(0)
    elif "first_down_rush" in drive_plays.columns and "first_down_pass" in drive_plays.columns:
        _fd = (drive_plays["first_down_rush"].fillna(0) + drive_plays["first_down_pass"].fillna(0)).clip(upper=1)
    else:
        _fd = pd.Series(0, index=drive_plays.index)

    if "touchdown" in drive_plays.columns:
        _tdc = drive_plays["touchdown"].fillna(0)
    else:
        _tdc = pd.Series(0, index=drive_plays.index)

    drive_plays["_eckel"] = (
        ((_fd == 1) & (drive_plays["yardline_100"].fillna(99) <= 40)) | (_tdc == 1)
    ).astype(int)

    has_fumble_lost = "fumble_lost" in drive_plays.columns
    if has_fumble_lost:
        drive_plays["_to"] = (
            (drive_plays["interception"].fillna(0) == 1) |
            (drive_plays["fumble_lost"].fillna(0) == 1)
        ).astype(int)
    else:
        drive_plays["_to"] = (drive_plays["interception"].fillna(0) == 1).astype(int)

    # Aggregate to drive level
    drv_agg = (
        drive_plays.groupby(["_drv", "posteam", "defteam", "game_id"])
        .agg(
            eckel=("_eckel", "max"),
            turnover=("_to", "max"),
            start_yl=("yardline_100", "first"),
            drive_result=("fixed_drive_result", "first"),
        )
        .reset_index()
    )
    drv_agg["drive_pts"] = drv_agg["drive_result"].map(DRIVE_PTS).fillna(0.0)

    # — Offensive Eckel per game —
    off_drv_g = (
        drv_agg.groupby(["posteam", "game_id", "defteam"])
        .agg(
            total_drv=("_drv", "count"),
            eckel_drv=("eckel", "sum"),
            total_pts=("drive_pts", "sum"),
            turnovers=("turnover", "sum"),
            avg_fp=("start_yl", "mean"),
        )
        .reset_index()
    )
    off_drv_g["eckel_rate"] = off_drv_g["eckel_drv"] / off_drv_g["total_drv"].clip(lower=1)
    off_drv_g["ppe"] = np.where(
        off_drv_g["eckel_drv"] > 0,
        off_drv_g["total_pts"] / off_drv_g["eckel_drv"], 0.0
    )
    off_drv_g["to_rate"] = off_drv_g["turnovers"] / off_drv_g["total_drv"].clip(lower=1)

    # — Defensive Eckel per game —
    def_drv_g = (
        drv_agg.groupby(["defteam", "game_id", "posteam"])
        .agg(
            total_drv=("_drv", "count"),
            eckel_drv=("eckel", "sum"),
            total_pts=("drive_pts", "sum"),
            turnovers=("turnover", "sum"),
            avg_fp=("start_yl", "mean"),
        )
        .reset_index()
    )
    def_drv_g["eckel_rate_allow"] = def_drv_g["eckel_drv"] / def_drv_g["total_drv"].clip(lower=1)
    def_drv_g["ppe_allow"] = np.where(
        def_drv_g["eckel_drv"] > 0,
        def_drv_g["total_pts"] / def_drv_g["eckel_drv"], 0.0
    )
    def_drv_g["to_forced"] = def_drv_g["turnovers"] / def_drv_g["total_drv"].clip(lower=1)

    # Season-level eckel summaries
    off_eck_s = off_drv_g.groupby("posteam").agg(
        off_eckel_rate=("eckel_rate", "mean"),
        off_ppe=("ppe", "mean"),
        off_to_rate=("to_rate", "mean"),
        off_fp=("avg_fp", "mean"),
        avg_off_drives=("total_drv", "mean"),
    ).reset_index()

    def_eck_s = def_drv_g.groupby("defteam").agg(
        def_eckel_rate=("eckel_rate_allow", "mean"),
        def_ppe=("ppe_allow", "mean"),
        def_to_rate=("to_forced", "mean"),
        def_fp=("avg_fp", "mean"),
        avg_def_drives=("total_drv", "mean"),
    ).reset_index()

    # — Cross-matchup Eckel relative diffs (opponent-adjusted, like R model) —
    opp_def_eck_avg = def_eck_s.rename(columns={
        "defteam": "defteam", "def_eckel_rate": "opp_def_er", "def_ppe": "opp_def_ppe"
    })[["defteam", "opp_def_er", "opp_def_ppe"]]

    off_eck_adj = off_drv_g.merge(opp_def_eck_avg, on="defteam", how="left")
    off_eck_adj["er_diff"] = (off_eck_adj["eckel_rate"] - off_eck_adj["opp_def_er"]) / \
                              off_eck_adj["opp_def_er"].clip(lower=1e-6)
    off_eck_adj["ppe_diff"] = (off_eck_adj["ppe"] - off_eck_adj["opp_def_ppe"]) / \
                               off_eck_adj["opp_def_ppe"].clip(lower=1e-6)

    eckel_vs_def = off_eck_adj.groupby("posteam").agg(
        eckel_vs_def_diff=("er_diff", "mean"),
        ppe_vs_def_diff=("ppe_diff", "mean"),
        off_eckel_rate_sd=("er_diff", "std"),
        off_ppe_sd=("ppe_diff", "std"),
    ).reset_index().fillna(0.0)

    opp_off_eck_avg = off_eck_s.rename(columns={
        "posteam": "posteam", "off_eckel_rate": "opp_off_er", "off_ppe": "opp_off_ppe"
    })[["posteam", "opp_off_er", "opp_off_ppe"]]

    def_eck_adj = def_drv_g.merge(opp_off_eck_avg, on="posteam", how="left")
    def_eck_adj["er_allow_diff"] = (def_eck_adj["eckel_rate_allow"] - def_eck_adj["opp_off_er"]) / \
                                    def_eck_adj["opp_off_er"].clip(lower=1e-6)
    def_eck_adj["ppe_allow_diff"] = (def_eck_adj["ppe_allow"] - def_eck_adj["opp_off_ppe"]) / \
                                     def_eck_adj["opp_off_ppe"].clip(lower=1e-6)

    eckel_vs_off = def_eck_adj.groupby("defteam").agg(
        eckel_vs_off_diff=("er_allow_diff", "mean"),
        ppe_vs_off_diff=("ppe_allow_diff", "mean"),
        def_eckel_rate_sd=("er_allow_diff", "std"),
        def_ppe_sd=("ppe_allow_diff", "std"),
    ).reset_index().fillna(0.0)

    # ── PART 3 : Merge everything ──────────────────────────────────────────────
    # Rename all "defteam" keys to "posteam" before merging to avoid duplicate columns
    def _rk(df, col="defteam"):
        return df.rename(columns={col: "posteam"}) if col in df.columns else df

    all_m = (
        off_season
        .merge(_rk(def_season),   on="posteam", how="left")
        .merge(off_ypp_s,         on="posteam", how="left")
        .merge(_rk(def_ypp_s),    on="posteam", how="left")
        .merge(off_eck_s,         on="posteam", how="left")
        .merge(_rk(def_eck_s),    on="posteam", how="left")
        .merge(eckel_vs_def,      on="posteam", how="left")
        .merge(_rk(eckel_vs_off), on="posteam", how="left")
        .merge(off_ypp_play,      on="posteam", how="left")
        .merge(_rk(def_ypp_play), on="posteam", how="left")
    ).fillna(0.0)

    # ── Build team_metrics dict ────────────────────────────────────────────────
    def _g(row, key, default=0.0):
        return float(row[key]) if key in row.index else default

    team_metrics = {}
    for _, row in all_m.iterrows():
        team = row["posteam"]
        team_metrics[team] = {
            "team": team,
            # PBP offense
            "off_ypc":      _g(row, "off_ypc"),
            "off_ypa":      _g(row, "off_ypa"),
            "off_ypc_sd":   max(_g(row, "off_ypc_sd"), 0.1),
            "off_ypa_sd":   max(_g(row, "off_ypa_sd"), 0.3),
            "avg_rush_att": _g(row, "avg_rush_att", 27.0),
            "avg_pass_att": _g(row, "avg_pass_att", 35.0),
            "sd_rush_att":  max(_g(row, "sd_rush_att"), 3.0),
            "sd_pass_att":  max(_g(row, "sd_pass_att"), 3.0),
            "off_ypp":      max(_g(row, "off_ypp", 14.0), 8.0),
            # PBP defense
            "def_ypc_pct":  _g(row, "def_ypc_pct"),
            "def_ypa_pct":  _g(row, "def_ypa_pct"),
            "def_ypc_sd":   max(_g(row, "def_ypc_sd"), 0.05),
            "def_ypa_sd":   max(_g(row, "def_ypa_sd"), 0.05),
            "def_ypp":      max(_g(row, "def_ypp", 14.0), 8.0),
            "def_ypc_raw":  _g(row, "def_ypc_raw"),
            "def_ypa_raw":  _g(row, "def_ypa_raw"),
            # Eckel offense
            "off_eckel_rate":   _g(row, "off_eckel_rate"),
            "off_ppe":          _g(row, "off_ppe"),
            "off_to_rate":      _g(row, "off_to_rate"),
            "off_fp":           _g(row, "off_fp"),
            "avg_off_drives":   _g(row, "avg_off_drives", 11.0),
            "eckel_vs_def_diff":_g(row, "eckel_vs_def_diff"),
            "ppe_vs_def_diff":  _g(row, "ppe_vs_def_diff"),
            "off_eckel_rate_sd":max(_g(row, "off_eckel_rate_sd"), 0.02),
            "off_ppe_sd":       max(_g(row, "off_ppe_sd"), 0.05),
            # Eckel defense
            "def_eckel_rate":   _g(row, "def_eckel_rate"),
            "def_ppe":          _g(row, "def_ppe"),
            "def_to_rate":      _g(row, "def_to_rate"),
            "def_fp":           _g(row, "def_fp"),
            "eckel_vs_off_diff":_g(row, "eckel_vs_off_diff"),
            "ppe_vs_off_diff":  _g(row, "ppe_vs_off_diff"),
            "def_eckel_rate_sd":max(_g(row, "def_eckel_rate_sd"), 0.02),
            "def_ppe_sd":       max(_g(row, "def_ppe_sd"), 0.05),
            # General
            "n_games":      int(_g(row, "n_games", 1)),
            "off_ypp_play": _g(row, "off_ypp_play", 5.5),
            "def_ypp_play": _g(row, "def_ypp_play", 5.5),
        }

    # ── League averages ────────────────────────────────────────────────────────
    lg = {}
    for col in ["off_eckel_rate", "def_eckel_rate", "off_ppe", "def_ppe",
                "avg_off_drives", "off_ypc", "off_ypa", "off_ypp"]:
        if col in all_m.columns:
            lg[f"lg_{col}"] = float(all_m[col].mean())
        else:
            lg[f"lg_{col}"] = 0.0

    # ── Rankings ───────────────────────────────────────────────────────────────
    # direction: True = ascending (lower=better), False = descending (higher=better)
    rank_cfg = {
        "off_ypc":        False,
        "off_ypa":        False,
        "off_eckel_rate": False,
        "off_ppe":        False,
        "off_ypp_play":   False,
        "def_ypc_pct":    True,   # lower% modifier = better defense
        "def_ypa_pct":    True,
        "def_eckel_rate": True,
        "def_ppe":        True,
        "def_ypp_play":   True,
        "off_to_rate":    True,   # lower TO rate = better offense
        "def_to_rate":    False,  # higher forced TO = better defense
    }

    rankings = {team: {} for team in all_m["posteam"]}
    for metric, asc in rank_cfg.items():
        if metric not in all_m.columns:
            continue
        rk = all_m[metric].rank(ascending=asc, method="min").astype(int)
        for team, r in zip(all_m["posteam"], rk):
            rankings[team][metric] = int(r)

    return team_metrics, lg, rankings
