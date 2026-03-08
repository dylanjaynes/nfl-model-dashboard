#!/usr/bin/env python3
"""
backtest.py — Full-season backtest for the EPA NFL model
Usage: python epa_model/backtest.py [--season YEAR] [--sims N]

Loads PBP once, runs all 18 regular-season weeks with no lookahead,
then prints per-week and season-aggregate results.
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


def load_pbp(season):
    path = os.path.join(CACHE_DIR, f"pbp_{season}.parquet")
    if os.path.exists(path):
        print(f"  Loading PBP from cache...")
        return pd.read_parquet(path)
    print(f"  Downloading {season} PBP (one-time, ~30–60s)...")
    pbp = nfl.import_pbp_data([season])
    pbp.to_parquet(path, index=False)
    return pbp


def load_schedule(season):
    path = os.path.join(CACHE_DIR, f"schedule_{season}_backtest.parquet")
    # Always re-fetch for backtest so we get complete actuals
    if os.path.exists(path):
        os.remove(path)
    sched = nfl.import_schedules([season])
    sched.to_parquet(path, index=False)
    return sched


def grade(r, actual_home, actual_away):
    """Grade ATS, total, and win pick. Returns (ats, tot, win) each True/False/'Push'/None."""
    margin = actual_home - actual_away
    total  = actual_home + actual_away

    # ATS
    ats = None
    vs = r.get("vegas_spread")
    if r.get("ats_pick") and vs is not None:
        diff = margin + vs  # >0 means home covered
        if diff == 0:
            ats = "Push"
        elif r["ats_pick"] == r["home_team"]:
            ats = diff > 0
        else:
            ats = diff < 0

    # Total
    tot = None
    vt = r.get("vegas_total")
    if r.get("total_pick") and vt is not None:
        diff = total - vt
        if diff == 0:
            tot = "Push"
        elif r["total_pick"] == "Over":
            tot = diff > 0
        else:
            tot = diff < 0

    # ML win pick
    win = None
    if actual_home != actual_away:
        pred = r["home_team"] if r["home_win_prob"] >= 0.5 else r["away_team"]
        actual = r["home_team"] if actual_home > actual_away else r["away_team"]
        win = pred == actual

    return ats, tot, win


def run_backtest(season, sims, rng):
    pbp   = load_pbp(season)
    sched = load_schedule(season)

    pbp["week"]   = pd.to_numeric(pbp["week"],   errors="coerce")
    sched["week"] = pd.to_numeric(sched["week"], errors="coerce")

    reg = sched[(sched["game_type"] == "REG")].copy()
    weeks = sorted(reg["week"].dropna().unique().astype(int).tolist())

    season_ats = {"W": 0, "L": 0, "P": 0}
    season_tot = {"W": 0, "L": 0, "P": 0}
    season_win = {"W": 0, "L": 0}
    season_mae = []

    weekly_rows = []

    print(f"\n{'='*75}")
    print(f"  EPA Backtest  |  Season {season}  |  {sims:,} sims/game")
    print(f"{'='*75}\n")
    print(f"  {'WK':>2}  {'GAMES':>5}  {'ATS':>7}  {'ATS%':>6}  {'TOT':>7}  {'TOT%':>6}  {'WIN':>7}  {'WIN%':>6}  {'MAE':>6}")
    print(f"  {'─'*72}")

    for week in weeks:
        week_games = reg[reg["week"] == week].copy()
        if week_games.empty:
            continue

        # Metrics through prior week (no lookahead); week 1 → empty metrics (league avg)
        metrics_week = max(week - 1, 0)
        team_cache = {}

        def get_metrics(team):
            if team not in team_cache:
                team_cache[team] = compute_team_epa(team, pbp, metrics_week)
            return team_cache[team]

        wk_ats = {"W": 0, "L": 0, "P": 0}
        wk_tot = {"W": 0, "L": 0, "P": 0}
        wk_win = {"W": 0, "L": 0}
        wk_mae = []
        n_games = 0

        for _, game in week_games.iterrows():
            home = game.get("home_team")
            away = game.get("away_team")
            if pd.isna(home) or pd.isna(away):
                continue

            hs = game.get("home_score")
            as_ = game.get("away_score")
            if pd.isna(hs) or pd.isna(as_):
                continue  # skip games without actuals

            vs = float(game["spread_line"]) if pd.notna(game.get("spread_line")) else None
            vt = float(game["total_line"])  if pd.notna(game.get("total_line"))  else None

            t1 = get_metrics(home)
            t2 = get_metrics(away)
            sim = simulate_game(t1, t2, vegas_spread=vs, vegas_total=vt, sims=sims, rng=rng)
            sim["home_score"] = hs
            sim["away_score"] = as_
            sim["week"] = week

            ats_r, tot_r, win_r = grade(sim, float(hs), float(as_))

            if ats_r is True:   wk_ats["W"] += 1
            elif ats_r is False: wk_ats["L"] += 1
            elif ats_r == "Push": wk_ats["P"] += 1

            if tot_r is True:   wk_tot["W"] += 1
            elif tot_r is False: wk_tot["L"] += 1
            elif tot_r == "Push": wk_tot["P"] += 1

            if win_r is True:  wk_win["W"] += 1
            elif win_r is False: wk_win["L"] += 1

            if vs is not None:
                wk_mae.append(abs(sim["spread_mean"] - (float(hs) - float(as_))))

            n_games += 1
            weekly_rows.append({**sim, "ats_result": ats_r, "tot_result": tot_r, "win_result": win_r})

        # Accumulate season
        for k in ("W", "L", "P"):
            season_ats[k] += wk_ats[k]
            season_tot[k] += wk_tot[k]
        season_win["W"] += wk_win["W"]
        season_win["L"] += wk_win["L"]
        season_mae.extend(wk_mae)

        n_ats = wk_ats["W"] + wk_ats["L"]
        n_tot = wk_tot["W"] + wk_tot["L"]
        n_win = wk_win["W"] + wk_win["L"]
        ats_pct = wk_ats["W"] / n_ats * 100 if n_ats else 0
        tot_pct = wk_tot["W"] / n_tot * 100 if n_tot else 0
        win_pct = wk_win["W"] / n_win * 100 if n_win else 0
        mae = np.mean(wk_mae) if wk_mae else float("nan")

        push_str = lambda r: f"{r['W']}-{r['L']}" + (f"-{r['P']}" if r["P"] else "")
        print(f"  {week:>2}  {n_games:>5}  {push_str(wk_ats):>7}  {ats_pct:>5.1f}%"
              f"  {push_str(wk_tot):>7}  {tot_pct:>5.1f}%"
              f"  {wk_win['W']}-{wk_win['L']:>1}  {win_pct:>5.1f}%  {mae:>6.2f}")

    # Season summary
    n_ats = season_ats["W"] + season_ats["L"]
    n_tot = season_tot["W"] + season_tot["L"]
    n_win = season_win["W"] + season_win["L"]
    ats_pct = season_ats["W"] / n_ats * 100 if n_ats else 0
    tot_pct = season_tot["W"] / n_tot * 100 if n_tot else 0
    win_pct = season_win["W"] / n_win * 100 if n_win else 0
    mae = np.mean(season_mae) if season_mae else float("nan")

    push_str = lambda r: f"{r['W']}-{r['L']}" + (f"-{r['P']}" if r["P"] else "")

    print(f"  {'─'*72}")
    print(f"  {'ALL':>2}  {n_ats+len([r for r in weekly_rows if r.get('ats_result') is None and r.get('tot_result') is None]):>5}  "
          f"{push_str(season_ats):>7}  {ats_pct:>5.1f}%"
          f"  {push_str(season_tot):>7}  {tot_pct:>5.1f}%"
          f"  {season_win['W']}-{season_win['L']:>1}  {win_pct:>5.1f}%  {mae:>6.2f}")
    print()

    # Betting P&L at -110 (need 52.4% to break even)
    stake = 100
    ats_pl = season_ats["W"] * (stake * 100/110) - season_ats["L"] * stake
    tot_pl = season_tot["W"] * (stake * 100/110) - season_tot["L"] * stake

    print(f"  ── Betting P&L at -110 (${stake}/game) ──")
    print(f"  ATS : ${ats_pl:+,.0f}  |  Breakeven: 52.4%  |  Model: {ats_pct:.1f}%")
    print(f"  TOT : ${tot_pl:+,.0f}  |  Breakeven: 52.4%  |  Model: {tot_pct:.1f}%")
    print(f"  Spread MAE: {mae:.2f} pts  (league baseline ~7.5 pts)")
    print()


def main():
    parser = argparse.ArgumentParser(description="EPA model full-season backtest")
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument("--sims",   type=int, default=2000)
    parser.add_argument("--seed",   type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    run_backtest(args.season, args.sims, rng)


if __name__ == "__main__":
    main()
