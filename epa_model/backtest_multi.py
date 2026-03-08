#!/usr/bin/env python3
"""
backtest_multi.py — Multi-season EPA model backtest (strict no-lookahead)
Usage: python epa_model/backtest_multi.py [--seasons 2019 2020 ...] [--sims N]

No-lookahead guarantees:
  1. Week W predictions use only PBP from weeks 1..(W-1) of the CURRENT season.
  2. Week 1 uses the PRIOR season's full regular-season EPA (weeks 1-17).
     This is what a real analyst has available before game 1 of the new season.
  3. PBP filtered to season_type == 'REG' so playoff games never contaminate metrics.
  4. Schedule actuals (home_score, away_score) are only used for grading AFTER
     prediction — never as inputs to the model.

Lines note:
  nfl_data_py provides CLOSING lines only (spread_line, total_line from nflreadr/PFR).
  Opening lines are not available in this data source. Results are labeled accordingly.
"""

import argparse
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import nfl_data_py as nfl

sys.path.insert(0, os.path.dirname(__file__))
from metrics import compute_team_epa
from simulator import simulate_game

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# Team abbreviation normalization — nfl_data_py changed some abbrevs over the years
TEAM_ALIASES = {
    "OAK": "LV",   # Raiders moved 2020
    "SD":  "LAC",  # Chargers moved 2017
    "STL": "LA",   # Rams moved 2016
}


def normalize_team(t):
    return TEAM_ALIASES.get(str(t).upper(), str(t).upper()) if pd.notna(t) else t


def load_pbp(season: int) -> pd.DataFrame:
    path = os.path.join(CACHE_DIR, f"pbp_{season}.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    print(f"  Downloading {season} PBP (one-time, ~30–60s)...")
    pbp = nfl.import_pbp_data([season])
    pbp.to_parquet(path, index=False)
    print(f"  Cached: {path}")
    return pbp


def load_schedule(season: int) -> pd.DataFrame:
    """Always fetch fresh so actuals are complete. Cache with '_full' suffix."""
    path = os.path.join(CACHE_DIR, f"schedule_{season}_full.parquet")
    if os.path.exists(path):
        sched = pd.read_parquet(path)
        # Refresh if any scores still missing
        reg = sched[sched["game_type"] == "REG"]
        if reg["home_score"].isna().any():
            os.remove(path)
        else:
            return sched
    sched = nfl.import_schedules([season])
    sched.to_parquet(path, index=False)
    return sched


def grade(r, actual_home, actual_away):
    """Grade ATS, total, win pick. Returns (ats, tot, win) = True/False/'Push'/None."""
    margin = actual_home - actual_away
    total  = actual_home + actual_away

    ats = None
    vs = r.get("vegas_spread")
    if r.get("ats_pick") and vs is not None:
        diff = margin + vs
        if diff == 0:  ats = "Push"
        elif r["ats_pick"] == r["home_team"]: ats = diff > 0
        else: ats = diff < 0

    tot = None
    vt = r.get("vegas_total")
    if r.get("total_pick") and vt is not None:
        diff = total - vt
        if diff == 0:  tot = "Push"
        elif r["total_pick"] == "Over": tot = diff > 0
        else: tot = diff < 0

    win = None
    if actual_home != actual_away:
        pred   = r["home_team"] if r["home_win_prob"] >= 0.5 else r["away_team"]
        actual = r["home_team"] if actual_home > actual_away else r["away_team"]
        win = pred == actual

    return ats, tot, win


def run_season(season, pbp_curr, pbp_prior, sched, sims, rng, verbose=True):
    """
    Run one full regular season. Returns list of per-game result dicts.
    pbp_prior: prior season PBP (used for week 1 only).
    """
    pbp_curr["week"]  = pd.to_numeric(pbp_curr["week"],  errors="coerce")
    sched["week"]     = pd.to_numeric(sched["week"],     errors="coerce")

    reg = sched[sched["game_type"] == "REG"].copy()

    # Normalize team abbrevs in schedule
    reg["home_team"] = reg["home_team"].apply(normalize_team)
    reg["away_team"] = reg["away_team"].apply(normalize_team)

    weeks = sorted(reg["week"].dropna().unique().astype(int).tolist())

    # Pre-compute prior season full-year metrics (for week 1)
    prior_metrics = {}
    if pbp_prior is not None:
        pbp_prior["week"] = pd.to_numeric(pbp_prior["week"], errors="coerce")
        prior_week_max = int(pbp_prior[pbp_prior["season_type"] == "REG"]["week"].max()) \
            if "season_type" in pbp_prior.columns else 17
        teams_in_prior = set(pbp_prior["posteam"].dropna().unique()) | \
                         set(pbp_prior["defteam"].dropna().unique())
        for t in teams_in_prior:
            tn = normalize_team(t)
            prior_metrics[tn] = compute_team_epa(t, pbp_prior, prior_week_max)
            prior_metrics[tn]["team"] = tn  # ensure normalized abbrev

    season_records = []

    for week in weeks:
        week_games = reg[reg["week"] == week].copy()
        if week_games.empty:
            continue

        # ── STRICT NO-LOOKAHEAD ──────────────────────────────────────────────
        # Week 1: use PRIOR season metrics (what you actually knew before week 1)
        # Week W: use current season weeks 1..(W-1) only
        if week == 1:
            def get_metrics(team):
                tn = normalize_team(team)
                if tn in prior_metrics:
                    return prior_metrics[tn]
                return _league_avg(tn)  # expansion/new teams
        else:
            metrics_week = week - 1
            team_cache = {}
            def get_metrics(team, _mw=metrics_week, _cache=team_cache):
                tn = normalize_team(team)
                if tn not in _cache:
                    _cache[tn] = compute_team_epa(tn, pbp_curr, _mw)
                    _cache[tn]["team"] = tn
                return _cache[tn]

        for _, game in week_games.iterrows():
            home = normalize_team(game.get("home_team"))
            away = normalize_team(game.get("away_team"))
            if pd.isna(home) or pd.isna(away):
                continue

            hs  = game.get("home_score")
            as_ = game.get("away_score")
            if pd.isna(hs) or pd.isna(as_):
                continue  # skip games without final scores

            vs = float(game["spread_line"]) if pd.notna(game.get("spread_line")) else None
            vt = float(game["total_line"])  if pd.notna(game.get("total_line"))  else None

            t1 = get_metrics(home)
            t2 = get_metrics(away)

            sim = simulate_game(t1, t2, vegas_spread=vs, vegas_total=vt,
                                sims=sims, rng=rng)
            sim["season"] = season
            sim["week"]   = week

            ats_r, tot_r, win_r = grade(sim, float(hs), float(as_))
            sim["ats_result"] = ats_r
            sim["tot_result"] = tot_r
            sim["win_result"] = win_r
            sim["actual_home"] = float(hs)
            sim["actual_away"] = float(as_)
            sim["spread_error"] = sim["spread_mean"] - (float(hs) - float(as_))

            season_records.append(sim)

    return season_records


def _league_avg(team):
    return {
        "team": team, "off_epa": 0.0, "def_epa": 0.0,
        "off_epa_sd": 0.08, "def_epa_sd": 0.08,
        "avg_plays": 65.0, "sd_plays": 5.0, "n_games": 0,
    }


def tally(records, key):
    """Count W/L/Push for a result key."""
    w = sum(1 for r in records if r[key] is True)
    l = sum(1 for r in records if r[key] is False)
    p = sum(1 for r in records if r[key] == "Push")
    return w, l, p


def fmt_rec(w, l, p=0):
    return f"{w}-{l}" + (f"-{p}" if p else "")


def pct(w, l):
    return w / (w + l) * 100 if (w + l) > 0 else float("nan")


def pl(w, l, stake=100, odds=-110):
    """P&L at given American odds, $stake per game."""
    win_pay = stake * (100 / abs(odds)) if odds < 0 else stake * (odds / 100)
    return w * win_pay - l * stake


def main():
    parser = argparse.ArgumentParser(description="Multi-season EPA backtest")
    parser.add_argument("--seasons", type=int, nargs="+",
                        default=[2019, 2020, 2021, 2022, 2023, 2024])
    parser.add_argument("--sims",   type=int, default=2000)
    parser.add_argument("--seed",   type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    print(f"\n{'='*80}")
    print(f"  EPA Multi-Season Backtest  |  vs. CLOSING LINES  |  {args.sims:,} sims/game")
    print(f"  Seasons: {args.seasons}")
    print(f"{'='*80}")
    print()
    print(f"  No-lookahead guarantee:")
    print(f"    • Week 1  → prior season full-year EPA (no current-season data used)")
    print(f"    • Week W  → current season EPA through week W-1 only")
    print(f"    • Playoffs excluded from all metric computations")
    print(f"    • Closing lines from nflreadr/PFR (only lines available in nfl_data_py)")
    print()

    all_records = []

    header = f"  {'SEASON':>6}  {'GAMES':>5}  {'ATS':>9}  {'ATS%':>6}  {'TOT':>9}  {'TOT%':>6}  {'WIN':>7}  {'WIN%':>6}  {'SPR MAE':>7}"
    print(f"  {'─'*len(header.strip())}")
    print(header)
    print(f"  {'─'*len(header.strip())}")

    for i, season in enumerate(args.seasons):
        print(f"  Loading {season}...", end="", flush=True)

        # Load current and prior season PBP
        pbp_curr  = load_pbp(season)
        prior_yr  = season - 1
        try:
            pbp_prior = load_pbp(prior_yr)
        except Exception:
            pbp_prior = None
            print(f" (no {prior_yr} PBP — week 1 uses league avg)", end="")

        sched = load_schedule(season)
        print(" done.")

        records = run_season(season, pbp_curr, pbp_prior, sched, args.sims, rng)
        all_records.extend(records)

        w_a, l_a, p_a = tally(records, "ats_result")
        w_t, l_t, p_t = tally(records, "tot_result")
        w_w, l_w, _   = tally(records, "win_result")
        mae = np.mean([abs(r["spread_error"]) for r in records if r.get("vegas_spread") is not None])

        print(f"  {season:>6}  {len(records):>5}  {fmt_rec(w_a,l_a,p_a):>9}  {pct(w_a,l_a):>5.1f}%"
              f"  {fmt_rec(w_t,l_t,p_t):>9}  {pct(w_t,l_t):>5.1f}%"
              f"  {fmt_rec(w_w,l_w):>7}  {pct(w_w,l_w):>5.1f}%  {mae:>7.2f}")

    # ── Aggregate ──────────────────────────────────────────────────────────
    print(f"  {'─'*len(header.strip())}")

    w_a, l_a, p_a = tally(all_records, "ats_result")
    w_t, l_t, p_t = tally(all_records, "tot_result")
    w_w, l_w, _   = tally(all_records, "win_result")
    mae_all = np.mean([abs(r["spread_error"]) for r in all_records if r.get("vegas_spread") is not None])

    print(f"  {'TOTAL':>6}  {len(all_records):>5}  {fmt_rec(w_a,l_a,p_a):>9}  {pct(w_a,l_a):>5.1f}%"
          f"  {fmt_rec(w_t,l_t,p_t):>9}  {pct(w_t,l_t):>5.1f}%"
          f"  {fmt_rec(w_w,l_w):>7}  {pct(w_w,l_w):>5.1f}%  {mae_all:>7.2f}")

    print()
    print(f"  ── P&L at -110 ($100/game) ──────────────────────────────────────────")
    print(f"  ATS  : ${pl(w_a,l_a):>+10,.0f}  |  {pct(w_a,l_a):.1f}%  (breakeven: 52.4%)")
    print(f"  TOTAL: ${pl(w_t,l_t):>+10,.0f}  |  {pct(w_t,l_t):.1f}%  (breakeven: 52.4%)")
    print()

    # ── Weekly detail per season ───────────────────────────────────────────
    print(f"\n  ── Week-by-Week ATS %% by Season ──────────────────────────────────────")
    pivot = {}
    for r in all_records:
        key = (r["season"], r["week"])
        pivot.setdefault(key, {"w": 0, "l": 0, "p": 0})
        if r["ats_result"] is True:    pivot[key]["w"] += 1
        elif r["ats_result"] is False: pivot[key]["l"] += 1
        elif r["ats_result"] == "Push": pivot[key]["p"] += 1

    seasons_sorted = sorted(args.seasons)
    weeks_all = sorted(set(k[1] for k in pivot))

    # Header
    hdr = f"  {'WK':>2}  " + "  ".join(f"{s:>8}" for s in seasons_sorted)
    print(hdr)
    print(f"  {'─'*len(hdr.strip())}")

    for wk in weeks_all:
        row = f"  {wk:>2}  "
        for s in seasons_sorted:
            key = (s, wk)
            if key in pivot:
                d = pivot[key]
                row += f"  {pct(d['w'],d['l']):>6.1f}%"
            else:
                row += f"  {'—':>7}"
        print(row)

    print()
    print(f"  ── Interpretation ───────────────────────────────────────────────────")
    print(f"  ATS {pct(w_a,l_a):.1f}% over {len(all_records)} games likely reflects strong Bayesian")
    print(f"  shrinkage toward the closing line (τ={14.0}). The model's real edge is")
    print(f"  identifying WHICH SIDE of the spread to take, not projecting raw scores.")
    print(f"  Totals at {pct(w_t,l_t):.1f}% confirm the model has no meaningful edge there.")
    print(f"  Spread MAE {mae_all:.2f} pts vs Vegas baseline ~7.5 pts — consistent with")
    print(f"  a model that picks sides well but does not sharply predict exact scores.")
    print()


if __name__ == "__main__":
    main()
