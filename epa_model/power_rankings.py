"""
power_rankings.py — EPA-based NFL Power Rankings

Rates every team by simulating them against a neutral "league average" opponent
at a neutral site (home-field advantage cancels by averaging both perspectives).

Usage:
    python3 epa_model/power_rankings.py                      # auto week, 2025
    python3 epa_model/power_rankings.py --season 2025 --week 14
    python3 epa_model/power_rankings.py --season 2025 --week 14 --sims 5000
    python3 epa_model/power_rankings.py --season 2025 --week 14 --show-epa

Power score = expected point margin vs an average NFL team at a neutral site.
A team rated +7 would be expected to beat an average team by 7 points.
"""

import argparse
import importlib.util
import os
import sys
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

# ── resolve paths ──────────────────────────────────────────────────────────────
HERE  = Path(__file__).parent
CACHE = HERE / "cache"
CACHE.mkdir(exist_ok=True)

def _load(name):
    p = HERE / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, p)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def _ensure_nfl():
    try:
        import nfl_data_py
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "nfl_data_py", "-q"])

ALIASES = {"OAK": "LV", "SD": "LAC", "STL": "LA"}

def _norm(t):
    return ALIASES.get(t, t) if isinstance(t, str) else t

def _load_pbp(season: int) -> pd.DataFrame:
    f = CACHE / f"pbp_{season}.parquet"
    if f.exists():
        pbp = pd.read_parquet(f)
    else:
        _ensure_nfl()
        import nfl_data_py as nfl
        print(f"  Downloading {season} PBP (one-time)…")
        pbp = nfl.import_pbp_data([season], downcast=True)
        pbp.to_parquet(f, index=False)
    for col in ["posteam", "defteam", "home_team", "away_team"]:
        if col in pbp.columns:
            pbp[col] = pbp[col].map(lambda x: ALIASES.get(x, x) if isinstance(x, str) else x)
    return pbp

def _load_schedule(season: int) -> pd.DataFrame:
    for stem in [f"schedule_{season}_full", f"schedule_{season}"]:
        f = CACHE / f"{stem}.parquet"
        if f.exists():
            sched = pd.read_parquet(f)
            sched["home_team"] = sched["home_team"].map(_norm)
            sched["away_team"] = sched["away_team"].map(_norm)
            return sched
    _ensure_nfl()
    import nfl_data_py as nfl
    sched = nfl.import_schedules([season])
    sched.to_parquet(CACHE / f"schedule_{season}_full.parquet", index=False)
    return sched

def _latest_completed_week(sched: pd.DataFrame) -> int:
    reg = sched[sched["game_type"] == "REG"]
    played = reg[reg["home_score"].notna()]
    if played.empty:
        return 0
    return int(played["week"].max())

# ── average team builder ───────────────────────────────────────────────────────
def _build_average_team(metrics: dict) -> dict:
    """
    Compute a 'league average' team as the mean of all 32 teams' metrics.
    This is the neutral benchmark every team is rated against.
    """
    keys = [
        "off_epa", "def_epa",
        "pass_off_epa", "rush_off_epa",
        "pass_def_epa", "rush_def_epa",
        "off_epa_sd",   "def_epa_sd",
        "avg_pass_plays", "avg_rush_plays",
        "sd_pass_plays",  "sd_rush_plays",
        "n_games",
    ]
    avg = {}
    for k in keys:
        vals = [m[k] for m in metrics.values() if k in m and m[k] is not None]
        avg[k] = float(np.mean(vals)) if vals else 0.0

    avg["team"] = "AVG"
    # copy raw EPA fields so SOS code doesn't break if called
    for raw_k in ["_raw_off_epa","_raw_pass_off_epa","_raw_rush_off_epa",
                  "_raw_def_epa","_raw_pass_def_epa","_raw_rush_def_epa"]:
        main_k = raw_k.replace("_raw_", "")
        avg[raw_k] = avg.get(main_k, 0.0)
    return avg


def _neutral_power(team_metrics: dict, avg_metrics: dict,
                   sim_fn, sims: int, rng) -> dict:
    """
    Simulate team vs average at a neutral site.
    HFA cancels by averaging both home/away perspectives:
      neutral_margin = (margin_as_home + margin_as_away) / 2
    """
    # Team as home
    r_home = sim_fn(team_metrics, avg_metrics,
                    vegas_spread=None, vegas_total=None,
                    sims=sims, rng=rng)

    # Team as away (average is "home", but we want team's perspective)
    r_away = sim_fn(avg_metrics, team_metrics,
                    vegas_spread=None, vegas_total=None,
                    sims=sims, rng=rng)

    # r_home["spread_mean"] = team_home_pts - avg_pts  (positive = team winning)
    # r_away["spread_mean"] = avg_home_pts - team_pts  (positive = avg winning)
    # team's neutral margin = (r_home + -r_away) / 2

    neutral_margin  = (r_home["spread_mean"] - r_away["spread_mean"]) / 2
    neutral_win_pct = (r_home["home_win_prob"] + r_away["away_win_prob"]) / 2
    neutral_total   = (r_home["total_mean"]    + r_away["total_mean"])   / 2

    return {
        "power":       neutral_margin,
        "win_vs_avg":  neutral_win_pct,
        "pts_scored":  neutral_total / 2,       # team's expected pts/game vs avg
        "off_score":   r_home["home_mean"],      # offensive output as home
        "def_allowed": r_home["away_mean"],      # pts allowed as home
    }


# ── formatting helpers ────────────────────────────────────────────────────────
CONF_MAP = {
    # AFC East
    "BUF": "AFC", "MIA": "AFC", "NE": "AFC",  "NYJ": "AFC",
    # AFC North
    "BAL": "AFC", "CIN": "AFC", "CLE": "AFC", "PIT": "AFC",
    # AFC South
    "HOU": "AFC", "IND": "AFC", "JAX": "AFC", "TEN": "AFC",
    # AFC West
    "DEN": "AFC", "KC":  "AFC", "LV":  "AFC", "LAC": "AFC",
    # NFC East
    "DAL": "NFC", "NYG": "NFC", "PHI": "NFC", "WAS": "NFC",
    # NFC North
    "CHI": "NFC", "DET": "NFC", "GB":  "NFC", "MIN": "NFC",
    # NFC South
    "ATL": "NFC", "CAR": "NFC", "NO":  "NFC", "TB":  "NFC",
    # NFC West
    "ARI": "NFC", "LA":  "NFC", "SF":  "NFC", "SEA": "NFC",
}

def _bar(val, width=10, scale=14.0):
    """ASCII progress bar centred at 0. Scale = pts per full width."""
    pct   = max(-1.0, min(1.0, val / scale))
    pos   = int(round((pct + 1.0) / 2 * width))
    chars = ["·"] * width
    mid   = width // 2
    if pct >= 0:
        for i in range(mid, min(pos, width)):
            chars[i] = "█"
    else:
        for i in range(max(pos, 0), mid):
            chars[i] = "█"
    chars[mid] = "|"
    return "".join(chars)


def main():
    ap = argparse.ArgumentParser(description="EPA-based NFL Power Rankings")
    ap.add_argument("--season",   type=int,  default=2025)
    ap.add_argument("--week",     type=int,  default=None,
                    help="Through which week (default: latest completed)")
    ap.add_argument("--sims",     type=int,  default=4000)
    ap.add_argument("--show-epa", action="store_true",
                    help="Also print raw EPA columns (off / def / net)")
    args = ap.parse_args()

    m2 = _load("metrics_v2")
    s2 = _load("simulator_v2")

    print(f"\nLoading {args.season} data…")
    sched = _load_schedule(args.season)

    through_week = args.week if args.week else _latest_completed_week(sched)
    if through_week == 0:
        print("No completed games yet — cannot build rankings.")
        return

    print(f"  Building metrics through Week {through_week}…")

    if through_week == 0:
        # No current-season data — use prior season
        pbp_prior = _load_pbp(args.season - 1)
        pbp_prior = pbp_prior[pbp_prior["season_type"] == "REG"].copy()
        metrics   = m2.compute_all_teams_v2(pbp_prior, through_week=17)
    else:
        pbp = _load_pbp(args.season)
        pbp_reg = pbp[pbp["season_type"] == "REG"].copy()
        metrics = m2.compute_all_teams_v2(pbp_reg, through_week=through_week)

    avg_team = _build_average_team(metrics)
    rng      = np.random.default_rng(42)

    print(f"  Simulating {len(metrics)} teams vs league average ({args.sims:,} sims each)…\n")

    rows = []
    for team, tm in metrics.items():
        pwr = _neutral_power(tm, avg_team, s2.simulate_game_v2,
                             args.sims, rng)
        rows.append({
            "team":      team,
            "conf":      CONF_MAP.get(team, "   "),
            "power":     pwr["power"],
            "win_pct":   pwr["win_vs_avg"] * 100,
            "pts_for":   pwr["pts_scored"],
            "pts_agnst": avg_team["n_games"],   # placeholder — replaced below
            "off_epa":   tm["off_epa"],
            "def_epa":   tm["def_epa"],
            "net_epa":   tm["off_epa"] - tm["def_epa"],
            "n_games":   tm["n_games"],
        })

    # Fix pts_against: simulate avg vs team for real pts-allowed
    rng2 = np.random.default_rng(43)
    for row in rows:
        tm = metrics[row["team"]]
        r_away = s2.simulate_game_v2(avg_team, tm, sims=args.sims, rng=rng2)
        row["pts_agnst"] = r_away["home_mean"]   # pts avg scores against this team

    df = pd.DataFrame(rows).sort_values("power", ascending=False).reset_index(drop=True)
    df["rank"]     = df.index + 1
    df["off_rank"] = df["off_epa"].rank(ascending=False).astype(int)
    df["def_rank"] = df["def_epa"].rank(ascending=True).astype(int)   # lower def_epa = better

    # ── print ──────────────────────────────────────────────────────────────────
    title = (f"  NFL Power Rankings  ·  Season {args.season}  ·  "
             f"Through Week {through_week}  ·  EPA Model")
    width = 82 if not args.show_epa else 98
    print("═" * width)
    print(title)
    print("═" * width)

    if args.show_epa:
        hdr = (f"{'RK':>3}  {'TEAM':<5} {'CONF':>4}  {'POWER':>6}  "
               f"{'WIN%':>5}  {'PF':>5}  {'PA':>5}  {'BAR':<12}  "
               f"{'OFF EPA':>7}  {'DEF EPA':>7}  {'NET EPA':>7}  "
               f"{'OFF RK':>6}  {'DEF RK':>6}  {'GP':>4}")
    else:
        hdr = (f"{'RK':>3}  {'TEAM':<5} {'CONF':>4}  {'POWER':>6}  "
               f"{'WIN%':>5}  {'PF':>5}  {'PA':>5}  {'BAR':<12}  "
               f"{'OFF RK':>6}  {'DEF RK':>6}  {'GP':>4}")
    print(hdr)
    print("─" * width)

    prev_tier = None
    for _, r in df.iterrows():
        # Tier breaks (purely cosmetic)
        tier = ("Elite" if r["power"] > 7 else
                "Contender" if r["power"] > 3 else
                "Fringe" if r["power"] > -3 else
                "Rebuilding")
        if tier != prev_tier:
            if prev_tier is not None:
                print()
            prev_tier = tier

        bar = _bar(r["power"])
        tag = "◀" if r["power"] >= 0 else ""

        if args.show_epa:
            line = (f"  {r['rank']:>2}.  {r['team']:<5} {r['conf']:>4}  "
                    f"{r['power']:>+6.1f}  {r['win_pct']:>4.1f}%  "
                    f"{r['pts_for']:>5.1f}  {r['pts_agnst']:>5.1f}  "
                    f"[{bar}]  "
                    f"{r['off_epa']:>+7.4f}  {r['def_epa']:>+7.4f}  {r['net_epa']:>+7.4f}  "
                    f"{'#'+str(r['off_rank']):>6}  {'#'+str(r['def_rank']):>6}  "
                    f"{int(r['n_games']):>4}")
        else:
            line = (f"  {r['rank']:>2}.  {r['team']:<5} {r['conf']:>4}  "
                    f"{r['power']:>+6.1f}  {r['win_pct']:>4.1f}%  "
                    f"{r['pts_for']:>5.1f}  {r['pts_agnst']:>5.1f}  "
                    f"[{bar}]  "
                    f"{'#'+str(r['off_rank']):>6}  {'#'+str(r['def_rank']):>6}  "
                    f"{int(r['n_games']):>4}")
        print(line)

    print("─" * width)
    print(
        f"\n  POWER  = expected margin vs avg NFL team at neutral site\n"
        f"  WIN%   = win probability vs avg NFL team\n"
        f"  PF/PA  = avg points scored/allowed vs avg team\n"
        f"  OFF/DEF RK = rank among all teams this season\n"
        f"  Metrics: SOS-adjusted · recency-weighted · turnover-neutral EPA\n"
    )
    if args.show_epa:
        print(
            f"  OFF EPA  = SOS-adj recency-weighted turnover-neutral offensive EPA/play\n"
            f"  DEF EPA  = SOS-adj recency-weighted turnover-neutral defensive EPA allowed/play\n"
            f"  NET EPA  = off_epa − def_epa (higher = better overall team)\n"
        )


if __name__ == "__main__":
    main()
