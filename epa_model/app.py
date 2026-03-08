"""
app.py — NFL EPA Model  ·  Streamlit Dashboard

Run locally:
    streamlit run epa_model/app.py

Deploy to web:
    Push repo to GitHub → share.streamlit.io → connect → done
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Setup ─────────────────────────────────────────────────────────────────────
HERE  = Path(__file__).parent
CACHE = HERE / "cache"
CACHE.mkdir(exist_ok=True)

st.set_page_config(
    page_title="NFL EPA Model",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Module loader ──────────────────────────────────────────────────────────────
@st.cache_resource
def _load_modules():
    def _load(name):
        spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    return _load("metrics_v2"), _load("simulator_v2")

m2, s2 = _load_modules()

# ── Constants ─────────────────────────────────────────────────────────────────
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
AFC_COLOR = "#1a6fc4"
NFC_COLOR = "#c41a1a"
SEASONS   = list(range(2019, 2026))
NFL_TEAMS = [
    "ARI","ATL","BAL","BUF","CAR","CHI","CIN","CLE",
    "DAL","DEN","DET","GB","HOU","IND","JAX","KC",
    "LA","LAC","LV","MIA","MIN","NE","NO","NYG",
    "NYJ","PHI","PIT","SEA","SF","TB","TEN","WAS",
]
DRIVE_PTS_MAP = {
    "Touchdown": 6.8, "Field goal": 3.0,
}  # all other drive results → 0 pts

def _norm(t):
    return ALIASES.get(t, t) if isinstance(t, str) else t

# ── Data loaders (cached) ──────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading play-by-play data…")
def load_pbp(season: int) -> pd.DataFrame:
    f = CACHE / f"pbp_{season}.parquet"
    if not f.exists():
        try:
            import nfl_data_py as nfl
            pbp = nfl.import_pbp_data([season], downcast=True)
            pbp.to_parquet(f, index=False)
        except Exception as e:
            st.error(f"Could not load PBP for {season}: {e}")
            return pd.DataFrame()
    pbp = pd.read_parquet(f)
    for col in ["posteam","defteam","home_team","away_team"]:
        if col in pbp.columns:
            pbp[col] = pbp[col].map(lambda x: ALIASES.get(x,x) if isinstance(x,str) else x)
    return pbp

@st.cache_data(show_spinner="Loading schedule…")
def load_schedule(season: int) -> pd.DataFrame:
    for stem in [f"schedule_{season}_full", f"schedule_{season}"]:
        f = CACHE / f"{stem}.parquet"
        if f.exists():
            df = pd.read_parquet(f)
            df["home_team"] = df["home_team"].map(_norm)
            df["away_team"] = df["away_team"].map(_norm)
            return df
    try:
        import nfl_data_py as nfl
        df = nfl.import_schedules([season])
        df.to_parquet(CACHE / f"schedule_{season}_full.parquet", index=False)
        return df
    except Exception as e:
        st.error(f"Could not load schedule for {season}: {e}")
        return pd.DataFrame()

@st.cache_data(show_spinner="Computing EPA metrics…")
def build_metrics(season: int, through_week: int) -> dict:
    if through_week == 0:
        pbp_prior = load_pbp(season - 1)
        pbp_prior = pbp_prior[pbp_prior["season_type"] == "REG"].copy()
        return m2.compute_all_teams_v2(pbp_prior, through_week=17)
    pbp = load_pbp(season)
    pbp_reg = pbp[pbp["season_type"] == "REG"].copy()
    return m2.compute_all_teams_v2(pbp_reg, through_week=through_week)

@st.cache_data(show_spinner="Running power rankings simulations…")
def compute_power_rankings(season: int, through_week: int, sims: int) -> pd.DataFrame:
    metrics  = build_metrics(season, through_week)
    avg_team = _build_average_team(metrics)
    rng      = np.random.default_rng(42)
    rng2     = np.random.default_rng(43)
    rows = []
    for team, tm in metrics.items():
        r_home = s2.simulate_game_v2(tm, avg_team, sims=sims, rng=rng)
        r_away = s2.simulate_game_v2(avg_team, tm, sims=sims, rng=rng)
        r_away2 = s2.simulate_game_v2(avg_team, tm, sims=sims, rng=rng2)
        neutral_margin  = (r_home["spread_mean"] - r_away["spread_mean"]) / 2
        neutral_win_pct = (r_home["home_win_prob"] + r_away["away_win_prob"]) / 2
        pts_for   = r_home["home_mean"]
        pts_agnst = r_away2["home_mean"]
        rows.append({
            "team":      team,
            "conf":      CONF_MAP.get(team, ""),
            "power":     neutral_margin,
            "win_pct":   neutral_win_pct * 100,
            "pts_for":   pts_for,
            "pts_agnst": pts_agnst,
            "off_epa":   tm["off_epa"],
            "def_epa":   tm["def_epa"],
            "net_epa":   tm["off_epa"] - tm["def_epa"],
            "n_games":   tm["n_games"],
        })
    df = pd.DataFrame(rows).sort_values("power", ascending=False).reset_index(drop=True)
    df["rank"]     = df.index + 1
    df["off_rank"] = df["off_epa"].rank(ascending=False).astype(int)
    df["def_rank"] = df["def_epa"].rank(ascending=True).astype(int)
    return df

@st.cache_data(show_spinner="Simulating games…")
def compute_weekly_predictions(season: int, week: int, sims: int) -> list[dict]:
    through_week = week - 1 if week > 1 else 0
    metrics  = build_metrics(season, through_week)
    sched    = load_schedule(season)
    reg      = sched[sched["game_type"] == "REG"]
    games    = reg[reg["week"] == week].copy()
    rng      = np.random.default_rng(42)
    results  = []
    for _, row in games.sort_values("gameday").iterrows():
        ht, at = _norm(row["home_team"]), _norm(row["away_team"])
        hm, am = metrics.get(ht), metrics.get(at)
        if hm is None or am is None:
            continue
        vs = _safe_float(row.get("spread_line"))
        vt = _safe_float(row.get("total_line"))
        res = s2.simulate_game_v2(hm, am, vegas_spread=vs, vegas_total=vt,
                                  sims=sims, rng=rng)
        ah = _safe_float(row.get("home_score"))
        aa = _safe_float(row.get("away_score"))
        results.append({
            **res,
            "gameday":     str(row.get("gameday",""))[:10],
            "actual_home": ah,
            "actual_away": aa,
            "scored":      ah is not None and aa is not None,
        })
    return results

@st.cache_data(show_spinner="Computing extended stats…")
def compute_extended_metrics(season: int, through_week: int) -> dict:
    """Success rates, pts/drive, 3rd-down conv, red zone TD%, early-down EPA."""
    if through_week == 0:
        pbp = load_pbp(season - 1)
        reg = pbp[pbp["season_type"] == "REG"].copy()
    else:
        pbp = load_pbp(season)
        reg = pbp[(pbp["season_type"] == "REG") & (pbp["week"] <= through_week)].copy()

    if reg.empty:
        return {}

    # ── Composite drive ID ─────────────────────────────────────────────────────
    # fixed_drive resets to 1 at the start of every game, so we must combine it
    # with game_id to uniquely identify each drive across the full season.
    reg = reg.copy()
    if "game_id" in reg.columns and "fixed_drive" in reg.columns:
        reg["_drv"] = reg["game_id"].astype(str) + "_" + reg["fixed_drive"].astype(str)
    elif "fixed_drive" in reg.columns:
        reg["_drv"] = reg["fixed_drive"].astype(str)

    plays     = reg[(reg["rush_attempt"] == 1) | (reg["pass_attempt"] == 1)].copy()
    plays_p   = plays[plays["pass_attempt"] == 1]
    plays_r   = plays[plays["rush_attempt"] == 1]
    all_teams = sorted(reg["posteam"].dropna().unique().tolist())

    result = {}
    for team in all_teams:
        off   = plays[plays["posteam"] == team]
        def_  = plays[plays["defteam"] == team]
        off_p = plays_p[plays_p["posteam"] == team]
        off_r = plays_r[plays_r["posteam"] == team]
        def_p = plays_p[plays_p["defteam"] == team]
        def_r = plays_r[plays_r["defteam"] == team]

        def _sr(df):
            return float(df["success"].mean()) if len(df) > 0 else 0.0

        off_pass_sr = _sr(off_p)
        off_rush_sr = _sr(off_r)
        def_pass_sr = _sr(def_p)
        def_rush_sr = _sr(def_r)

        # Early-down EPA (downs 1 & 2)
        off_ed = off[off["down"].isin([1, 2])]
        def_ed = def_[def_["down"].isin([1, 2])]
        off_early_epa = float(off_ed["epa"].mean())  if len(off_ed) > 0 else 0.0
        def_early_epa = float(def_ed["epa"].mean())  if len(def_ed) > 0 else 0.0

        # 3rd-down conversion
        off_3 = off[off["down"] == 3]
        def_3 = def_[def_["down"] == 3]
        col3 = "third_down_converted" if "third_down_converted" in off.columns else "first_down"
        off_3rd_conv = float(off_3[col3].mean()) if len(off_3) > 0 else 0.0
        def_3rd_conv = float(def_3[col3].mean()) if len(def_3) > 0 else 0.0

        # Red-zone TD% (yardline_100 ≤ 20)
        if "_drv" in reg.columns and "yardline_100" in reg.columns:
            off_rz = reg[(reg["posteam"] == team) & (reg["yardline_100"] <= 20)]
            def_rz = reg[(reg["defteam"]  == team) & (reg["yardline_100"] <= 20)]
            off_rz_td = float(off_rz.groupby("_drv")["touchdown"].max().mean()) \
                        if len(off_rz) > 0 else 0.0
            def_rz_td = float(def_rz.groupby("_drv")["touchdown"].max().mean()) \
                        if len(def_rz) > 0 else 0.0
        else:
            off_rz_td = def_rz_td = 0.0

        # Points per drive
        if "_drv" in reg.columns and "fixed_drive_result" in reg.columns:
            off_drv = reg[reg["posteam"] == team].drop_duplicates("_drv")["fixed_drive_result"]
            def_drv = reg[reg["defteam"]  == team].drop_duplicates("_drv")["fixed_drive_result"]
            off_pts_drv = float(off_drv.map(lambda x: DRIVE_PTS_MAP.get(x, 0.0)).mean()) if len(off_drv) > 0 else 0.0
            def_pts_drv = float(def_drv.map(lambda x: DRIVE_PTS_MAP.get(x, 0.0)).mean()) if len(def_drv) > 0 else 0.0
        else:
            off_pts_drv = def_pts_drv = 0.0

        # ── Eckel metrics ──────────────────────────────────────────────────────
        # Eckel event = big-play TD  OR  first down inside the opponent's 40
        #   "big-play TD"      → touchdown == 1 on an offensive rush/pass play
        #   "first down/40"    → first_down == 1 AND yardline_100 <= 40
        #
        # Eckel Rate        = % of drives containing ≥1 eckel event
        # Pts Per Eckel     = avg pts scored on drives that contained an eckel event
        # Eckel Ratio       = team's eckel events / (team's + opponent's eckel events)
        #                     measures which team controls the big-play/field-position battle
        _fd_col = "first_down"
        _has_eckel = (
            "_drv" in reg.columns
            and "fixed_drive_result" in reg.columns
            and _fd_col in plays.columns
        )
        if _has_eckel:
            off_pl = plays[plays["posteam"] == team].copy()
            def_pl = plays[plays["defteam"] == team].copy()

            def _eckel_mask(df):
                td = df["touchdown"].fillna(0) == 1
                fd40 = (df[_fd_col].fillna(0) == 1) & (df["yardline_100"].fillna(99) <= 40)
                return td | fd40

            off_eck = off_pl[_eckel_mask(off_pl)]
            def_eck = def_pl[_eckel_mask(def_pl)]

            # Eckel Rate: drives with ≥1 eckel event / total drives
            # Use composite _drv key (game_id + fixed_drive) — fixed_drive resets each game
            off_drv_n = reg[reg["posteam"] == team]["_drv"].nunique()
            def_drv_n = reg[reg["defteam"] == team]["_drv"].nunique()
            off_eckel_rate = off_eck["_drv"].nunique() / off_drv_n if off_drv_n > 0 else 0.0
            def_eckel_rate = def_eck["_drv"].nunique() / def_drv_n if def_drv_n > 0 else 0.0

            # Pts Per Eckel: avg pts on drives that had an eckel event
            off_eck_ids = set(off_eck["_drv"].unique())
            def_eck_ids = set(def_eck["_drv"].unique())
            _off_drv_df = reg[reg["posteam"] == team].drop_duplicates("_drv")
            _def_drv_df = reg[reg["defteam"] == team].drop_duplicates("_drv")
            off_eck_drv = _off_drv_df[_off_drv_df["_drv"].isin(off_eck_ids)]["fixed_drive_result"]
            def_eck_drv = _def_drv_df[_def_drv_df["_drv"].isin(def_eck_ids)]["fixed_drive_result"]
            off_pts_eckel = float(off_eck_drv.map(lambda x: DRIVE_PTS_MAP.get(x, 0.0)).mean()) if len(off_eck_drv) > 0 else 0.0
            def_pts_eckel = float(def_eck_drv.map(lambda x: DRIVE_PTS_MAP.get(x, 0.0)).mean()) if len(def_eck_drv) > 0 else 0.0

            # Eckel Ratio: share of total eckel events in all games
            off_n = len(off_eck);  def_n = len(def_eck)
            eckel_ratio = off_n / (off_n + def_n) if (off_n + def_n) > 0 else 0.5
        else:
            off_eckel_rate = def_eckel_rate = 0.0
            off_pts_eckel  = def_pts_eckel  = 0.0
            eckel_ratio    = 0.5

        result[team] = {
            "off_pass_sr":   off_pass_sr,
            "off_rush_sr":   off_rush_sr,
            "def_pass_sr":   def_pass_sr,
            "def_rush_sr":   def_rush_sr,
            "off_early_epa": off_early_epa,
            "def_early_epa": def_early_epa,
            "off_3rd_conv":  off_3rd_conv,
            "def_3rd_conv":  def_3rd_conv,
            "off_rz_td_pct": off_rz_td,
            "def_rz_td_pct": def_rz_td,
            "off_pts_drive":  off_pts_drv,
            "def_pts_drive":  def_pts_drv,
            "off_eckel_rate": off_eckel_rate,
            "def_eckel_rate": def_eckel_rate,
            "off_pts_eckel":  off_pts_eckel,
            "def_pts_eckel":  def_pts_eckel,
            "eckel_ratio":    eckel_ratio,
        }
    return result


@st.cache_data(show_spinner="Building national rankings…")
def build_all_rankings(season: int, through_week: int) -> dict:
    """Return {team: {metric: rank_int}} for all 32 teams, all metrics."""
    epa_m = build_metrics(season, through_week)
    ext_m = compute_extended_metrics(season, through_week)

    rows = []
    for team, m in epa_m.items():
        e = ext_m.get(team, {})
        rows.append({
            "team":          team,
            "off_epa":       m.get("off_epa",       0.0),
            "def_epa":       m.get("def_epa",        0.0),
            "pass_off_epa":  m.get("pass_off_epa",   0.0),
            "pass_def_epa":  m.get("pass_def_epa",   0.0),
            "rush_off_epa":  m.get("rush_off_epa",   0.0),
            "rush_def_epa":  m.get("rush_def_epa",   0.0),
            "off_pass_sr":   e.get("off_pass_sr",    0.0),
            "off_rush_sr":   e.get("off_rush_sr",    0.0),
            "def_pass_sr":   e.get("def_pass_sr",    0.0),
            "def_rush_sr":   e.get("def_rush_sr",    0.0),
            "off_early_epa": e.get("off_early_epa",  0.0),
            "def_early_epa": e.get("def_early_epa",  0.0),
            "off_3rd_conv":  e.get("off_3rd_conv",   0.0),
            "def_3rd_conv":  e.get("def_3rd_conv",   0.0),
            "off_rz_td_pct": e.get("off_rz_td_pct",  0.0),
            "def_rz_td_pct": e.get("def_rz_td_pct",  0.0),
            "off_pts_drive":  e.get("off_pts_drive",  0.0),
            "def_pts_drive":  e.get("def_pts_drive",  0.0),
            "off_eckel_rate": e.get("off_eckel_rate", 0.0),
            "def_eckel_rate": e.get("def_eckel_rate", 0.0),
            "off_pts_eckel":  e.get("off_pts_eckel",  0.0),
            "def_pts_eckel":  e.get("def_pts_eckel",  0.0),
            "eckel_ratio":    e.get("eckel_ratio",    0.5),
        })

    if not rows:
        return {}

    df = pd.DataFrame(rows).set_index("team")
    # Offense metrics: rank 1 = highest (better)
    off_cols = ["off_epa","pass_off_epa","rush_off_epa",
                "off_pass_sr","off_rush_sr","off_early_epa",
                "off_3rd_conv","off_rz_td_pct","off_pts_drive",
                "off_eckel_rate","off_pts_eckel","eckel_ratio"]
    # Defense metrics: rank 1 = lowest (allows least = better)
    def_cols = ["def_epa","pass_def_epa","rush_def_epa",
                "def_pass_sr","def_rush_sr","def_early_epa",
                "def_3rd_conv","def_rz_td_pct","def_pts_drive",
                "def_eckel_rate","def_pts_eckel"]

    rk = pd.DataFrame(index=df.index)
    for c in off_cols:
        if c in df.columns:
            rk[c] = df[c].rank(ascending=False, method="min").astype(int)
    for c in def_cols:
        if c in df.columns:
            rk[c] = df[c].rank(ascending=True, method="min").astype(int)

    return rk.to_dict(orient="index")


# ── Helper functions ───────────────────────────────────────────────────────────
def _safe_float(v):
    try:
        f = float(v)
        return f if np.isfinite(f) else None
    except (TypeError, ValueError):
        return None

def _build_average_team(metrics: dict) -> dict:
    keys = ["off_epa","def_epa","pass_off_epa","rush_off_epa","pass_def_epa",
            "rush_def_epa","off_epa_sd","def_epa_sd","avg_pass_plays",
            "avg_rush_plays","sd_pass_plays","sd_rush_plays","n_games"]
    avg = {k: float(np.mean([m[k] for m in metrics.values() if k in m]))
           for k in keys}
    avg["team"] = "AVG"
    for rk in ["_raw_off_epa","_raw_pass_off_epa","_raw_rush_off_epa",
               "_raw_def_epa","_raw_pass_def_epa","_raw_rush_def_epa"]:
        avg[rk] = avg.get(rk.replace("_raw_",""), 0.0)
    return avg

def _latest_completed_week(sched: pd.DataFrame) -> int:
    reg = sched[sched["game_type"] == "REG"]
    played = reg[reg["home_score"].notna()]
    return int(played["week"].max()) if not played.empty else 1

def _ats_grade(actual_margin, vegas_spread, model_pick, home_team, away_team):
    """Returns (ats_result, correct) where result is '✅'/'❌'/'⬛'
    nflfastR convention: positive spread_line = home team is favored.
    Home covers when actual_margin > spread_line (wins by more than the spread).
    """
    if vegas_spread is None:
        return "—", None
    if actual_margin > vegas_spread + 0.5:
        winner = home_team
    elif actual_margin < vegas_spread - 0.5:
        winner = away_team
    else:
        return "⬛", None
    correct = (model_pick == winner)
    return "✅" if correct else "❌", correct

# ── Win probability bar ────────────────────────────────────────────────────────
def _win_prob_bar(prob: float, team: str, color: str) -> str:
    pct = int(prob * 100)
    bar = "█" * (pct // 10) + "░" * (10 - pct // 10)
    return f"{bar}  {pct}%"

# ── Matchup tab HTML helpers ───────────────────────────────────────────────────
def _rank_badge(rank: int) -> str:
    if rank <= 10:   bg = "#1565c0"
    elif rank <= 22: bg = "#555555"
    else:            bg = "#c62828"
    return (f'<span style="background:{bg};color:#fff;padding:2px 6px;'
            f'border-radius:3px;font-weight:700;font-size:0.72rem;'
            f'display:inline-block;min-width:26px;text-align:center">#{rank}</span>')

def _fe(v):   # format EPA
    if v is None or (isinstance(v, float) and (v != v)): return "—"
    return f"{v:+.3f}"

def _fp(v):   # format percentage
    if v is None or (isinstance(v, float) and (v != v)): return "—"
    return f"{v*100:.1f}%"

def _fd(v):   # format pts/drive
    if v is None or (isinstance(v, float) and (v != v)): return "—"
    return f"{v:.2f}"

def _get(key: str, merged: dict) -> float:
    return merged.get(key, 0.0)

def _rk(key: str, ranks: dict) -> int:
    return int(ranks.get(key, 16))

def _mu_table_html(away_m: dict, home_m: dict,
                   away_rk: dict, home_rk: dict,
                   away_team: str, home_team: str,
                   table_type: str) -> str:
    """Center matchup table: off_vs_def or def_vs_off."""
    if table_type == "off_vs_def":
        title  = f"&nbsp;&nbsp;{away_team} OFFENSE  ·  vs  ·  {home_team} DEFENSE&nbsp;&nbsp;"
        hdr_bg = "#2e7d32"
        # (label, away_key [offense], home_key [defense], fmt)
        cfg = [
            ("EPA / DROPBACK",   "pass_off_epa",   "pass_def_epa",   _fe),
            ("EPA / RUSH",       "rush_off_epa",   "rush_def_epa",   _fe),
            ("PASS SUCCESS %",   "off_pass_sr",    "def_pass_sr",    _fp),
            ("RUSH SUCCESS %",   "off_rush_sr",    "def_rush_sr",    _fp),
            ("ECKEL RATE",       "off_eckel_rate", "def_eckel_rate", _fp),
            ("PTS / ECKEL",      "off_pts_eckel",  "def_pts_eckel",  _fd),
            ("EARLY DOWN EPA",   "off_early_epa",  "def_early_epa",  _fe),
            ("3RD DOWN CONV",    "off_3rd_conv",   "def_3rd_conv",   _fp),
            ("RED ZONE TD %",    "off_rz_td_pct",  "def_rz_td_pct",  _fp),
            ("PTS / DRIVE",      "off_pts_drive",  "def_pts_drive",  _fd),
        ]
    else:
        title  = f"&nbsp;&nbsp;{away_team} DEFENSE  ·  vs  ·  {home_team} OFFENSE&nbsp;&nbsp;"
        hdr_bg = "#1a3a6b"
        cfg = [
            ("EPA / DROPBACK",   "pass_def_epa",   "pass_off_epa",   _fe),
            ("EPA / RUSH",       "rush_def_epa",   "rush_off_epa",   _fe),
            ("PASS SUCCESS %",   "def_pass_sr",    "off_pass_sr",    _fp),
            ("RUSH SUCCESS %",   "def_rush_sr",    "off_rush_sr",    _fp),
            ("ECKEL RATE",       "def_eckel_rate", "off_eckel_rate", _fp),
            ("PTS / ECKEL",      "def_pts_eckel",  "off_pts_eckel",  _fd),
            ("EARLY DOWN EPA",   "def_early_epa",  "off_early_epa",  _fe),
            ("3RD DOWN CONV",    "def_3rd_conv",   "off_3rd_conv",   _fp),
            ("RED ZONE TD %",    "def_rz_td_pct",  "off_rz_td_pct",  _fp),
            ("PTS / DRIVE",      "def_pts_drive",  "off_pts_drive",  _fd),
        ]

    rows_html = ""
    for label, ak, hk, fmt in cfg:
        av = fmt(_get(ak, away_m));  hv = fmt(_get(hk, home_m))
        ab = _rank_badge(_rk(ak, away_rk));  hb = _rank_badge(_rk(hk, home_rk))
        rows_html += (
            f"<tr style='border-top:1px solid #2a2a40'>"
            f"<td style='text-align:right;padding:5px 8px;font-size:0.9rem;font-weight:600'>{av}</td>"
            f"<td style='text-align:right;padding:5px 4px'>{ab}</td>"
            f"<td style='text-align:center;padding:5px 8px;color:#9ca3af;"
            f"font-size:0.72rem;font-weight:700;text-transform:uppercase;"
            f"white-space:nowrap;letter-spacing:.3px'>{label}</td>"
            f"<td style='text-align:left;padding:5px 4px'>{hb}</td>"
            f"<td style='text-align:left;padding:5px 8px;font-size:0.9rem;font-weight:600'>{hv}</td>"
            f"</tr>"
        )

    return (
        f"<div style='margin-bottom:8px;border-radius:6px;overflow:hidden'>"
        f"<div style='background:{hdr_bg};color:#fff;text-align:center;"
        f"padding:7px 4px;font-weight:700;font-size:0.8rem;letter-spacing:.5px'>{title}</div>"
        f"<table style='width:100%;border-collapse:collapse;background:#111827'>"
        f"<colgroup><col style='width:18%'><col style='width:10%'>"
        f"<col style='width:44%'><col style='width:10%'><col style='width:18%'></colgroup>"
        f"{rows_html}"
        f"</table></div>"
    )


def _team_panel_html(team: str, epa_m: dict, ext_m: dict,
                     ranks: dict, win_prob: float, proj_pts: float) -> str:
    """Left/right team info panel with win prob, proj pts, and stat rows."""
    conf       = CONF_MAP.get(team, "")
    conf_color = AFC_COLOR if conf == "AFC" else NFC_COLOR
    bar_n      = max(0, min(20, int(win_prob * 20)))
    bar        = "█" * bar_n + "░" * (20 - bar_n)
    wp_color   = "#4ade80" if win_prob > 0.55 else "#facc15" if win_prob > 0.45 else "#f87171"

    merged = {**epa_m, **ext_m}

    def stat_row(label, key):
        v   = merged.get(key, 0.0)
        rk  = _rk(key, ranks)
        # format by key type
        val_str = _fp(v) if ("sr" in key or "conv" in key or "rz" in key or "eckel_rate" in key or "eckel_ratio" in key) else \
                  _fd(v) if ("drive" in key or "pts_eckel" in key) else _fe(v)
        return (
            f"<tr><td style='color:#9ca3af;font-size:0.75rem;padding:4px 6px;"
            f"text-transform:uppercase;white-space:nowrap'>{label}</td>"
            f"<td style='text-align:right;padding:4px 4px;font-weight:600;"
            f"font-size:0.88rem'>{val_str}</td>"
            f"<td style='text-align:right;padding:4px 6px'>{_rank_badge(rk)}</td></tr>"
        )

    sections = (
        f"<tr><td colspan='3' style='padding:8px 6px 3px;font-size:0.68rem;color:#6b7280;"
        f"text-transform:uppercase;letter-spacing:1px;font-weight:700'>── PASSING ──</td></tr>"
        + stat_row("EPA/Dropback",  "pass_off_epa")
        + stat_row("Pass Success%", "off_pass_sr")
        + f"<tr><td colspan='3' style='padding:8px 6px 3px;font-size:0.68rem;color:#6b7280;"
        f"text-transform:uppercase;letter-spacing:1px;font-weight:700'>── RUSHING ──</td></tr>"
        + stat_row("EPA/Rush",      "rush_off_epa")
        + stat_row("Rush Success%", "off_rush_sr")
        + f"<tr><td colspan='3' style='padding:8px 6px 3px;font-size:0.68rem;color:#6b7280;"
        f"text-transform:uppercase;letter-spacing:1px;font-weight:700'>── OFFENSE ──</td></tr>"
        + stat_row("Off EPA/Play",  "off_epa")
        + stat_row("Eckel Rate",    "off_eckel_rate")
        + stat_row("Pts/Eckel",     "off_pts_eckel")
        + stat_row("Eckel Ratio",   "eckel_ratio")
        + stat_row("3rd Dn Conv%",  "off_3rd_conv")
        + stat_row("Red Zone TD%",  "off_rz_td_pct")
        + stat_row("Pts/Drive",     "off_pts_drive")
        + f"<tr><td colspan='3' style='padding:8px 6px 3px;font-size:0.68rem;color:#6b7280;"
        f"text-transform:uppercase;letter-spacing:1px;font-weight:700'>── DEFENSE ──</td></tr>"
        + stat_row("Def EPA/Play",  "def_epa")
        + stat_row("Eckel Rate Alw","def_eckel_rate")
        + stat_row("Pts/Eckel Alw", "def_pts_eckel")
        + stat_row("Pass Def SR",   "def_pass_sr")
        + stat_row("Rush Def SR",   "def_rush_sr")
    )

    return (
        f"<div style='background:#1a1a2e;border-radius:8px;padding:14px'>"
        f"<div style='text-align:center;margin-bottom:10px'>"
        f"<div style='font-size:2.2rem;font-weight:800;letter-spacing:2px'>{team}</div>"
        f"<div style='color:{conf_color};font-size:0.78rem;font-weight:700;"
        f"text-transform:uppercase;letter-spacing:1px'>{conf}</div>"
        f"</div>"
        f"<div style='background:#111827;border-radius:6px;padding:10px;margin-bottom:8px;text-align:center'>"
        f"<div style='color:#9ca3af;font-size:0.68rem;text-transform:uppercase;letter-spacing:1px'>Win Probability</div>"
        f"<div style='font-size:2rem;font-weight:800;color:{wp_color}'>{win_prob*100:.1f}%</div>"
        f"<div style='font-family:monospace;font-size:0.62rem;color:#374151;margin-top:2px'>{bar}</div>"
        f"</div>"
        f"<div style='background:#111827;border-radius:6px;padding:8px;margin-bottom:10px;text-align:center'>"
        f"<div style='color:#9ca3af;font-size:0.68rem;text-transform:uppercase;letter-spacing:1px'>Proj Points</div>"
        f"<div style='font-size:1.5rem;font-weight:700'>{proj_pts:.1f}</div>"
        f"</div>"
        f"<table style='width:100%;border-collapse:collapse'>{sections}</table>"
        f"</div>"
    )

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🏈 NFL EPA Model")
    st.markdown("*Pure signal · Zero Vegas shrinkage*")
    st.divider()

    season = st.selectbox("Season", options=SEASONS[::-1], index=0)

    # Auto-detect week
    sched_sidebar = load_schedule(season)
    reg_sidebar   = sched_sidebar[sched_sidebar["game_type"] == "REG"] if not sched_sidebar.empty else pd.DataFrame()
    max_week      = int(reg_sidebar["week"].max()) if not reg_sidebar.empty else 18
    default_week  = _latest_completed_week(sched_sidebar) if not sched_sidebar.empty else 18

    week = st.slider("Through Week", min_value=1, max_value=max_week,
                     value=min(default_week, max_week))

    sims = st.select_slider(
        "Monte Carlo Sims",
        options=[1000, 2000, 3000, 5000, 8000],
        value=3000,
        help="More sims = more accurate but slower (~1s per 1000)"
    )

    st.divider()
    st.caption(
        "**Model**: SOS-adjusted · recency-weighted\n"
        "turnover-neutral EPA · 0% Vegas shrinkage\n\n"
        "**Data**: nflfastR via nfl_data_py\n\n"
        "**Backtest**: 72.5% ATS (2019–2024)"
    )

# ═══════════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════════
tab_pr, tab_wk, tab_rec, tab_mu = st.tabs([
    "🏈  Power Rankings",
    "📅  Weekly Picks",
    "📊  Season Record",
    "🏟️  Matchup",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — POWER RANKINGS
# ─────────────────────────────────────────────────────────────────────────────
with tab_pr:
    st.markdown(f"### Power Rankings · {season} Season · Through Week {week}")
    st.caption("Power = expected point margin vs average NFL team at a neutral site")

    pr_df = compute_power_rankings(season, week, sims)

    # ── Spotlight cards ──────────────────────────────────────────────────────
    top_team   = pr_df.iloc[0]
    top_off    = pr_df.loc[pr_df["off_rank"] == 1].iloc[0]
    top_def    = pr_df.loc[pr_df["def_rank"] == 1].iloc[0]

    c1, c2, c3 = st.columns(3)
    with c1:
        conf_color = AFC_COLOR if CONF_MAP.get(top_team["team"]) == "AFC" else NFC_COLOR
        st.markdown(
            f"""<div style='background:#1e1e2e;border-left:4px solid {conf_color};
            padding:14px 18px;border-radius:6px;'>
            <div style='color:#aaa;font-size:0.75rem;text-transform:uppercase;letter-spacing:1px'>
            #1 Overall</div>
            <div style='font-size:2rem;font-weight:700;margin:4px 0'>{top_team['team']}</div>
            <div style='color:#4ade80;font-size:1.1rem'>+{top_team['power']:.1f} pts vs avg</div>
            <div style='color:#aaa;font-size:0.85rem'>{top_team['win_pct']:.0f}% vs avg team</div>
            </div>""", unsafe_allow_html=True)
    with c2:
        conf_color2 = AFC_COLOR if CONF_MAP.get(top_off["team"]) == "AFC" else NFC_COLOR
        st.markdown(
            f"""<div style='background:#1e1e2e;border-left:4px solid {conf_color2};
            padding:14px 18px;border-radius:6px;'>
            <div style='color:#aaa;font-size:0.75rem;text-transform:uppercase;letter-spacing:1px'>
            #1 Offense</div>
            <div style='font-size:2rem;font-weight:700;margin:4px 0'>{top_off['team']}</div>
            <div style='color:#60a5fa;font-size:1.1rem'>OFF EPA {top_off['off_epa']:+.4f}</div>
            <div style='color:#aaa;font-size:0.85rem'>{top_off['pts_for']:.1f} pts/gm vs avg</div>
            </div>""", unsafe_allow_html=True)
    with c3:
        conf_color3 = AFC_COLOR if CONF_MAP.get(top_def["team"]) == "AFC" else NFC_COLOR
        st.markdown(
            f"""<div style='background:#1e1e2e;border-left:4px solid {conf_color3};
            padding:14px 18px;border-radius:6px;'>
            <div style='color:#aaa;font-size:0.75rem;text-transform:uppercase;letter-spacing:1px'>
            #1 Defense</div>
            <div style='font-size:2rem;font-weight:700;margin:4px 0'>{top_def['team']}</div>
            <div style='color:#f87171;font-size:1.1rem'>DEF EPA {top_def['def_epa']:+.4f}</div>
            <div style='color:#aaa;font-size:0.85rem'>{top_def['pts_agnst']:.1f} pts allowed vs avg</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("")

    # ── Power score bar chart ─────────────────────────────────────────────────
    colors = [AFC_COLOR if CONF_MAP.get(t) == "AFC" else NFC_COLOR
              for t in pr_df["team"]]
    bar_colors = [
        "#4ade80" if p > 7 else
        "#a3e635" if p > 3 else
        "#facc15" if p > -3 else
        "#f87171"
        for p in pr_df["power"]
    ]

    fig = go.Figure(go.Bar(
        x=pr_df["power"],
        y=pr_df["team"],
        orientation="h",
        marker_color=bar_colors,
        text=[f"{p:+.1f}" for p in pr_df["power"]],
        textposition="outside",
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Power: %{x:+.1f}<br>"
            "<extra></extra>"
        ),
    ))
    fig.add_vline(x=0, line_color="rgba(255,255,255,0.3)", line_width=1)
    fig.update_layout(
        height=900,
        margin=dict(l=60, r=80, t=20, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white", size=12),
        xaxis=dict(
            title="Expected margin vs avg team (neutral site)",
            gridcolor="rgba(255,255,255,0.08)",
            zeroline=False,
        ),
        yaxis=dict(
            autorange="reversed",
            gridcolor="rgba(0,0,0,0)",
        ),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Rankings table ────────────────────────────────────────────────────────
    st.markdown("#### Full Rankings Table")

    display_df = pr_df[[
        "rank","team","conf","power","win_pct",
        "pts_for","pts_agnst","off_epa","def_epa","off_rank","def_rank","n_games"
    ]].copy()
    display_df.columns = [
        "Rk","Team","Conf","Power","Win%",
        "PF","PA","Off EPA","Def EPA","Off Rk","Def Rk","GP"
    ]
    display_df["Power"]   = display_df["Power"].map(lambda x: f"{x:+.1f}")
    display_df["Win%"]    = display_df["Win%"].map(lambda x: f"{x:.1f}%")
    display_df["PF"]      = display_df["PF"].map(lambda x: f"{x:.1f}")
    display_df["PA"]      = display_df["PA"].map(lambda x: f"{x:.1f}")
    display_df["Off EPA"] = display_df["Off EPA"].map(lambda x: f"{x:+.4f}")
    display_df["Def EPA"] = display_df["Def EPA"].map(lambda x: f"{x:+.4f}")

    def _color_power(val):
        try:
            v = float(val)
        except (ValueError, TypeError):
            return ""
        if v > 7:  return "color: #4ade80; font-weight:600"
        if v > 3:  return "color: #a3e635"
        if v > -3: return "color: #facc15"
        return "color: #f87171"

    def _color_epa(val):
        try:
            v = float(val)
        except (ValueError, TypeError):
            return ""
        return "color: #4ade80" if v > 0 else "color: #f87171"

    styled = (
        display_df.style
        .applymap(_color_power, subset=["Power"])
        .applymap(_color_epa,   subset=["Off EPA"])
        .applymap(lambda v: "color: #4ade80" if float(v) < 0 else "color: #f87171",
                  subset=["Def EPA"])
        .set_properties(**{"font-size": "0.9rem"})
    )
    st.dataframe(styled, use_container_width=True, hide_index=True, height=1150)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — WEEKLY PICKS
# ─────────────────────────────────────────────────────────────────────────────
with tab_wk:
    st.markdown(f"### Weekly Picks · {season} Season")

    week_pick = st.select_slider(
        "Select Week",
        options=list(range(1, max_week + 1)),
        value=min(week, max_week),
        key="week_pick_slider",
    )

    results = compute_weekly_predictions(season, week_pick, sims)

    if not results:
        st.warning("No games found for this week.")
    else:
        has_scores = any(r["scored"] for r in results)
        through_wk = week_pick - 1 if week_pick > 1 else 0

        if has_scores:
            st.caption(f"✅ Scores available — showing graded results · "
                       f"Metrics computed through Week {through_wk}")
        else:
            st.caption(f"⏳ Upcoming games — predictions only · "
                       f"Metrics computed through Week {through_wk}")

        # ── Running totals ────────────────────────────────────────────────────
        ats_w = ats_l = ats_p = tot_w = tot_l = ml_w = ml_l = 0

        rows = []
        for r in results:
            ht, at = r["home_team"], r["away_team"]
            spr    = r["spread_mean"]
            fav    = ht if spr >= 0 else at
            dog    = at if spr >= 0 else ht
            spr_abs = abs(spr)

            mdl_spr_str = f"{fav} -{spr_abs:.1f}"
            veg_spr = r["vegas_spread"]
            veg_str = f"{veg_spr:+.1f}" if veg_spr is not None else "N/A"
            mdl_tot = f"{r['total_mean']:.1f}"
            veg_tot = r["vegas_total"]
            veg_tot_str = f"{veg_tot:.1f}" if veg_tot is not None else "N/A"
            win_pct = f"{r['home_win_prob']*100:.1f}%"
            ml      = f"{r['home_american']:+d}" if r["home_american"] else "N/A"
            ats_pick = r["ats_pick"] or "—"
            ats_conf = f"{r['ats_prob']*100:.0f}%" if r["ats_prob"] else "—"

            row = {
                "Matchup":      f"{ht} vs {at}",
                "Model Spread": mdl_spr_str,
                "Vegas Spread": veg_str,
                "Model Total":  mdl_tot,
                "Vegas Total":  veg_tot_str,
                f"{ht} Win%":   win_pct,
                "ML":           ml,
                "ATS Pick":     ats_pick,
                "Confidence":   ats_conf,
            }

            if has_scores and r["scored"]:
                ah, aa = r["actual_home"], r["actual_away"]
                row["Score"] = f"{ht} {int(ah)}-{int(aa)} {at}"
                actual_margin = ah - aa

                # ATS
                ats_res, ats_ok = _ats_grade(actual_margin, veg_spr, ats_pick, ht, at)
                row["ATS"] = ats_res
                if ats_ok is True:   ats_w += 1
                elif ats_ok is False: ats_l += 1
                elif ats_res == "⬛": ats_p += 1

                # Totals
                if veg_tot is not None:
                    actual_total = ah + aa
                    if actual_total > veg_tot + 0.5:
                        tot_res = "✅" if r["total_pick"] == "Over" else "❌"
                        if r["total_pick"] == "Over": tot_w += 1
                        else: tot_l += 1
                    elif actual_total < veg_tot - 0.5:
                        tot_res = "✅" if r["total_pick"] == "Under" else "❌"
                        if r["total_pick"] == "Under": tot_w += 1
                        else: tot_l += 1
                    else:
                        tot_res = "⬛"
                else:
                    tot_res = "—"
                row["TOT"] = tot_res

                # ML
                if ah > aa:
                    ml_res = "✅" if r["home_win_prob"] > 0.5 else "❌"
                    if r["home_win_prob"] > 0.5: ml_w += 1
                    else: ml_l += 1
                else:
                    ml_res = "✅" if r["home_win_prob"] < 0.5 else "❌"
                    if r["home_win_prob"] < 0.5: ml_w += 1
                    else: ml_l += 1
                row["ML"] = ml_res

            rows.append(row)

        games_df = pd.DataFrame(rows)
        st.dataframe(games_df, use_container_width=True, hide_index=True)

        # ── Summary cards ─────────────────────────────────────────────────────
        if has_scores:
            st.markdown("")
            c1, c2, c3, c4 = st.columns(4)
            ats_tot = ats_w + ats_l
            tot_tot = tot_w + tot_l
            ml_tot  = ml_w  + ml_l

            push_str = f"-{ats_p}P" if ats_p else ""
            ats_pct  = ats_w / ats_tot * 100 if ats_tot else 0
            tot_pct  = tot_w / tot_tot * 100 if tot_tot else 0
            ml_pct   = ml_w  / ml_tot  * 100 if ml_tot  else 0

            def _card(label, record, pct, color):
                return f"""<div style='background:#1e1e2e;padding:14px;border-radius:6px;text-align:center'>
                <div style='color:#aaa;font-size:0.75rem;text-transform:uppercase'>{label}</div>
                <div style='font-size:1.6rem;font-weight:700;color:{color}'>{record}</div>
                <div style='color:#aaa'>{pct:.1f}%</div></div>"""

            col_color = "#4ade80" if ats_pct >= 55 else "#facc15" if ats_pct >= 50 else "#f87171"
            with c1:
                st.markdown(_card(
                    f"ATS ({ats_tot}{push_str})", f"{ats_w}-{ats_l}", ats_pct, col_color
                ), unsafe_allow_html=True)
            with c2:
                tc = "#4ade80" if tot_pct >= 55 else "#facc15" if tot_pct >= 50 else "#f87171"
                st.markdown(_card(f"Totals ({tot_tot})", f"{tot_w}-{tot_l}", tot_pct, tc),
                            unsafe_allow_html=True)
            with c3:
                mc = "#4ade80" if ml_pct >= 60 else "#facc15" if ml_pct >= 50 else "#f87171"
                st.markdown(_card(f"ML ({ml_tot})", f"{ml_w}-{ml_l}", ml_pct, mc),
                            unsafe_allow_html=True)
            with c4:
                profit = ats_w * 90.91 - ats_l * 100
                profit_color = "#4ade80" if profit >= 0 else "#f87171"
                st.markdown(
                    f"""<div style='background:#1e1e2e;padding:14px;border-radius:6px;text-align:center'>
                    <div style='color:#aaa;font-size:0.75rem;text-transform:uppercase'>ATS Profit ($100)</div>
                    <div style='font-size:1.6rem;font-weight:700;color:{profit_color}'>
                    {'$'+f'{profit:,.0f}' if profit>=0 else '-$'+f'{abs(profit):,.0f}'}</div>
                    <div style='color:#aaa'>at -110</div></div>""",
                    unsafe_allow_html=True,
                )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — SEASON RECORD
# ─────────────────────────────────────────────────────────────────────────────
with tab_rec:
    st.markdown(f"### Season Record · {season}")
    st.caption("Model ATS/TOT/ML performance across all completed weeks")

    sched_full = load_schedule(season)
    reg_full   = sched_full[sched_full["game_type"] == "REG"].copy() if not sched_full.empty else pd.DataFrame()

    if reg_full.empty:
        st.warning("No schedule data available.")
    else:
        # Accumulate week-by-week
        weeks_played = sorted(reg_full[reg_full["home_score"].notna()]["week"].unique().tolist())

        if not weeks_played:
            st.info("No completed games yet this season.")
        else:
            week_rows = []
            cumulative = {"ats_w":0,"ats_l":0,"ats_p":0,"tot_w":0,"tot_l":0,"ml_w":0,"ml_l":0}

            for wk in weeks_played:
                wk_results = compute_weekly_predictions(season, int(wk), sims)
                wk_ats_w = wk_ats_l = wk_ats_p = 0
                wk_tot_w = wk_tot_l = wk_ml_w = wk_ml_l = 0

                for r in wk_results:
                    if not r["scored"]: continue
                    ah, aa  = r["actual_home"], r["actual_away"]
                    am      = ah - aa
                    vs      = r["vegas_spread"]
                    vt      = r["vegas_total"]

                    if vs is not None:
                        # nflfastR: positive spread_line = home favored
                        # home covers if margin > spread_line
                        if am > vs + 0.5:
                            (wk_ats_w := wk_ats_w+1) if r["ats_pick"] == r["home_team"] else (wk_ats_l := wk_ats_l+1)
                        elif am < vs - 0.5:
                            (wk_ats_w := wk_ats_w+1) if r["ats_pick"] == r["away_team"] else (wk_ats_l := wk_ats_l+1)
                        else:
                            wk_ats_p += 1

                    if vt is not None:
                        at_ = ah + aa
                        if at_ > vt + 0.5:
                            (wk_tot_w := wk_tot_w+1) if r["total_pick"]=="Over" else (wk_tot_l := wk_tot_l+1)
                        elif at_ < vt - 0.5:
                            (wk_tot_w := wk_tot_w+1) if r["total_pick"]=="Under" else (wk_tot_l := wk_tot_l+1)

                    if ah > aa:
                        (wk_ml_w := wk_ml_w+1) if r["home_win_prob"]>0.5 else (wk_ml_l := wk_ml_l+1)
                    else:
                        (wk_ml_w := wk_ml_w+1) if r["home_win_prob"]<0.5 else (wk_ml_l := wk_ml_l+1)

                cumulative["ats_w"] += wk_ats_w; cumulative["ats_l"] += wk_ats_l; cumulative["ats_p"] += wk_ats_p
                cumulative["tot_w"] += wk_tot_w; cumulative["tot_l"] += wk_tot_l
                cumulative["ml_w"]  += wk_ml_w;  cumulative["ml_l"]  += wk_ml_l

                ats_tot_ = cumulative["ats_w"] + cumulative["ats_l"]
                tot_tot_ = cumulative["tot_w"] + cumulative["tot_l"]
                ml_tot_  = cumulative["ml_w"]  + cumulative["ml_l"]

                week_rows.append({
                    "week": int(wk),
                    "wk_ats": f"{wk_ats_w}-{wk_ats_l}",
                    "wk_ats_pct": wk_ats_w/(wk_ats_w+wk_ats_l)*100 if (wk_ats_w+wk_ats_l) else 0,
                    "cum_ats_pct": cumulative["ats_w"]/ats_tot_*100 if ats_tot_ else 0,
                    "cum_tot_pct": cumulative["tot_w"]/tot_tot_*100 if tot_tot_ else 0,
                    "cum_ml_pct":  cumulative["ml_w"] /ml_tot_ *100 if ml_tot_  else 0,
                    "cum_ats_w": cumulative["ats_w"], "cum_ats_l": cumulative["ats_l"],
                    "cum_tot_w": cumulative["tot_w"], "cum_tot_l": cumulative["tot_l"],
                    "cum_ml_w":  cumulative["ml_w"],  "cum_ml_l":  cumulative["ml_l"],
                    "cum_profit": cumulative["ats_w"]*90.91 - cumulative["ats_l"]*100,
                })

            if not week_rows:
                st.info("No graded weeks yet.")
            else:
                wr_df = pd.DataFrame(week_rows)

                # ── Summary cards ─────────────────────────────────────────────
                last = wr_df.iloc[-1]
                c1,c2,c3,c4 = st.columns(4)
                with c1:
                    ap = last["cum_ats_pct"]
                    ac = "#4ade80" if ap>=55 else "#facc15" if ap>=50 else "#f87171"
                    st.markdown(
                        f"""<div style='background:#1e1e2e;padding:14px;border-radius:6px;text-align:center'>
                        <div style='color:#aaa;font-size:0.75rem;text-transform:uppercase'>Season ATS</div>
                        <div style='font-size:1.6rem;font-weight:700;color:{ac}'>
                        {last['cum_ats_w']}-{last['cum_ats_l']}</div>
                        <div style='color:#aaa'>{ap:.1f}%</div></div>""", unsafe_allow_html=True)
                with c2:
                    tp = last["cum_tot_pct"]
                    tc = "#4ade80" if tp>=55 else "#facc15" if tp>=50 else "#f87171"
                    st.markdown(
                        f"""<div style='background:#1e1e2e;padding:14px;border-radius:6px;text-align:center'>
                        <div style='color:#aaa;font-size:0.75rem;text-transform:uppercase'>Season Totals</div>
                        <div style='font-size:1.6rem;font-weight:700;color:{tc}'>
                        {last['cum_tot_w']}-{last['cum_tot_l']}</div>
                        <div style='color:#aaa'>{tp:.1f}%</div></div>""", unsafe_allow_html=True)
                with c3:
                    mp = last["cum_ml_pct"]
                    mc = "#4ade80" if mp>=60 else "#facc15" if mp>=50 else "#f87171"
                    st.markdown(
                        f"""<div style='background:#1e1e2e;padding:14px;border-radius:6px;text-align:center'>
                        <div style='color:#aaa;font-size:0.75rem;text-transform:uppercase'>Season ML</div>
                        <div style='font-size:1.6rem;font-weight:700;color:{mc}'>
                        {last['cum_ml_w']}-{last['cum_ml_l']}</div>
                        <div style='color:#aaa'>{mp:.1f}%</div></div>""", unsafe_allow_html=True)
                with c4:
                    pr_ = last["cum_profit"]
                    pc  = "#4ade80" if pr_>=0 else "#f87171"
                    st.markdown(
                        f"""<div style='background:#1e1e2e;padding:14px;border-radius:6px;text-align:center'>
                        <div style='color:#aaa;font-size:0.75rem;text-transform:uppercase'>ATS Profit ($100)</div>
                        <div style='font-size:1.6rem;font-weight:700;color:{pc}'>
                        {'$'+f'{pr_:,.0f}' if pr_>=0 else '-$'+f'{abs(pr_):,.0f}'}</div>
                        <div style='color:#aaa'>at -110</div></div>""", unsafe_allow_html=True)

                st.markdown("")

                # ── Cumulative ATS% line chart ─────────────────────────────────
                fig2 = go.Figure()
                fig2.add_hline(y=52.4, line_dash="dash",
                               line_color="rgba(255,255,255,0.25)",
                               annotation_text="Break-even (52.4%)",
                               annotation_position="bottom right")
                fig2.add_trace(go.Scatter(
                    x=wr_df["week"], y=wr_df["cum_ats_pct"],
                    mode="lines+markers", name="ATS %",
                    line=dict(color="#4ade80", width=2.5),
                    marker=dict(size=7),
                    hovertemplate="Week %{x}<br>ATS: %{y:.1f}%<extra></extra>",
                ))
                fig2.add_trace(go.Scatter(
                    x=wr_df["week"], y=wr_df["cum_tot_pct"],
                    mode="lines+markers", name="TOT %",
                    line=dict(color="#60a5fa", width=2, dash="dot"),
                    marker=dict(size=6),
                    hovertemplate="Week %{x}<br>TOT: %{y:.1f}%<extra></extra>",
                ))
                fig2.add_trace(go.Scatter(
                    x=wr_df["week"], y=wr_df["cum_ml_pct"],
                    mode="lines+markers", name="ML %",
                    line=dict(color="#f59e0b", width=2, dash="dot"),
                    marker=dict(size=6),
                    hovertemplate="Week %{x}<br>ML: %{y:.1f}%<extra></extra>",
                ))
                fig2.update_layout(
                    title="Cumulative Win % by Week",
                    height=380,
                    xaxis=dict(title="Week", dtick=1,
                               gridcolor="rgba(255,255,255,0.08)"),
                    yaxis=dict(title="Win %", range=[35, 90],
                               gridcolor="rgba(255,255,255,0.08)"),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="white"),
                    legend=dict(bgcolor="rgba(0,0,0,0)"),
                )
                st.plotly_chart(fig2, use_container_width=True)

                # ── Profit line chart ──────────────────────────────────────────
                fig3 = go.Figure()
                fig3.add_hline(y=0, line_color="rgba(255,255,255,0.25)")
                fig3.add_trace(go.Scatter(
                    x=wr_df["week"], y=wr_df["cum_profit"],
                    mode="lines+markers", name="Profit",
                    fill="tozeroy",
                    fillcolor="rgba(74,222,128,0.15)",
                    line=dict(color="#4ade80", width=2.5),
                    marker=dict(size=7),
                    hovertemplate="Week %{x}<br>Profit: $%{y:,.0f}<extra></extra>",
                ))
                fig3.update_layout(
                    title="Cumulative ATS Profit ($100 flat bets at -110)",
                    height=300,
                    xaxis=dict(title="Week", dtick=1,
                               gridcolor="rgba(255,255,255,0.08)"),
                    yaxis=dict(title="Profit ($)",
                               gridcolor="rgba(255,255,255,0.08)"),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="white"),
                    showlegend=False,
                )
                st.plotly_chart(fig3, use_container_width=True)

                # ── Per-week breakdown table ───────────────────────────────────
                st.markdown("#### Week-by-Week Breakdown")
                wk_display = wr_df[[
                    "week","wk_ats","wk_ats_pct",
                    "cum_ats_w","cum_ats_l","cum_ats_pct","cum_profit"
                ]].copy()
                wk_display.columns = [
                    "Week","ATS (wk)","Wk ATS%",
                    "Cum W","Cum L","Cum ATS%","Cum Profit"
                ]
                wk_display["Wk ATS%"]  = wk_display["Wk ATS%"].map(lambda x: f"{x:.1f}%")
                wk_display["Cum ATS%"] = wk_display["Cum ATS%"].map(lambda x: f"{x:.1f}%")
                wk_display["Cum Profit"] = wk_display["Cum Profit"].map(
                    lambda x: f"${x:,.0f}" if x>=0 else f"-${abs(x):,.0f}")
                st.dataframe(wk_display, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — MATCHUP PREVIEW
# ─────────────────────────────────────────────────────────────────────────────
with tab_mu:
    st.markdown(f"### 🏟️ Matchup Preview · {season} Season · Through Week {week}")
    st.caption("Side-by-side advanced stats with national rank badges · Monte Carlo win probability")

    # ── Team selectors ────────────────────────────────────────────────────────
    sc1, sc2, sc3 = st.columns([4, 4, 2])
    with sc1:
        away_team = st.selectbox("Away Team", NFL_TEAMS, index=0, key="mu_away")
    with sc2:
        home_idx  = 1 if len(NFL_TEAMS) > 1 else 0
        home_team = st.selectbox("Home Team", NFL_TEAMS, index=home_idx, key="mu_home")
    with sc3:
        st.markdown("<div style='margin-top:28px'></div>", unsafe_allow_html=True)
        neutral_site = st.checkbox("Neutral Site", key="mu_neutral")

    if away_team == home_team:
        st.warning("Please select two different teams.")
    else:
        with st.spinner("Loading stats and running simulation…"):
            epa_all  = build_metrics(season, week)
            ext_all  = compute_extended_metrics(season, week)
            all_rks  = build_all_rankings(season, week)

        away_epa = epa_all.get(away_team, {})
        home_epa = epa_all.get(home_team, {})
        away_ext = ext_all.get(away_team, {})
        home_ext = ext_all.get(home_team, {})
        away_rks = all_rks.get(away_team, {})
        home_rks = all_rks.get(home_team, {})
        away_mrg = {**away_epa, **away_ext}
        home_mrg = {**home_epa, **home_ext}

        if not away_epa or not home_epa:
            st.error("Metrics not available for one or both teams — check that the season has play data.")
        else:
            rng_a = np.random.default_rng(42)
            rng_b = np.random.default_rng(43)

            if neutral_site:
                # Cancel HFA by averaging both venue perspectives
                r1 = s2.simulate_game_v2(away_epa, home_epa, sims=sims, rng=rng_a)
                r2 = s2.simulate_game_v2(home_epa, away_epa, sims=sims, rng=rng_b)
                # from away_team's perspective as "home" (r1) and truly away (r2)
                away_proj = (r1["home_mean"]    + r2["away_mean"])    / 2
                home_proj = (r1["away_mean"]    + r2["home_mean"])    / 2
                away_wp   = (r1["home_win_prob"] + r2["away_win_prob"]) / 2
                home_wp   = 1.0 - away_wp
                total_proj  = away_proj + home_proj
                spread_proj = home_proj - away_proj   # positive = home favoured
            else:
                sim = s2.simulate_game_v2(home_epa, away_epa, sims=sims, rng=rng_a)
                home_proj   = sim["home_mean"]
                away_proj   = sim["away_mean"]
                home_wp     = sim["home_win_prob"]
                away_wp     = sim["away_win_prob"]
                total_proj  = sim["total_mean"]
                spread_proj = sim["spread_mean"]   # home - away

            fav_team   = home_team if spread_proj >= 0 else away_team
            margin_val = abs(spread_proj)

            # ── 3-column layout ───────────────────────────────────────────────
            col_a, col_c, col_h = st.columns([3, 4, 3])

            with col_a:
                st.markdown(
                    _team_panel_html(away_team, away_epa, away_ext,
                                     away_rks, away_wp, away_proj),
                    unsafe_allow_html=True,
                )

            with col_c:
                st.markdown(
                    _mu_table_html(away_mrg, home_mrg, away_rks, home_rks,
                                   away_team, home_team, "off_vs_def"),
                    unsafe_allow_html=True,
                )
                st.markdown(
                    _mu_table_html(away_mrg, home_mrg, away_rks, home_rks,
                                   away_team, home_team, "def_vs_off"),
                    unsafe_allow_html=True,
                )

            with col_h:
                st.markdown(
                    _team_panel_html(home_team, home_epa, home_ext,
                                     home_rks, home_wp, home_proj),
                    unsafe_allow_html=True,
                )

            # ── Bottom summary bar ────────────────────────────────────────────
            site_tag = "🌐 Neutral Site" if neutral_site else "🏠 Home Field"
            fav_color = AFC_COLOR if CONF_MAP.get(fav_team) == "AFC" else NFC_COLOR
            st.markdown(
                f"""<div style='background:#1e1e2e;border-radius:8px;padding:14px 20px;
                margin-top:14px;display:flex;justify-content:center;
                align-items:center;gap:32px;flex-wrap:wrap;text-align:center'>
                <div>
                  <div style='color:#6b7280;font-size:0.7rem;text-transform:uppercase;
                  letter-spacing:1px'>Projected Total</div>
                  <div style='font-size:1.4rem;font-weight:800;color:#60a5fa'>
                  {total_proj:.1f}</div>
                </div>
                <div style='color:#374151;font-size:1.5rem'>·</div>
                <div>
                  <div style='color:#6b7280;font-size:0.7rem;text-transform:uppercase;
                  letter-spacing:1px'>Projected Margin</div>
                  <div style='font-size:1.4rem;font-weight:800;color:{fav_color}'>
                  {fav_team} by {margin_val:.1f}</div>
                </div>
                <div style='color:#374151;font-size:1.5rem'>·</div>
                <div>
                  <div style='color:#6b7280;font-size:0.7rem;text-transform:uppercase;
                  letter-spacing:1px'>Site</div>
                  <div style='font-size:1rem;font-weight:600;color:#9ca3af'>{site_tag}</div>
                </div>
                </div>""",
                unsafe_allow_html=True,
            )
