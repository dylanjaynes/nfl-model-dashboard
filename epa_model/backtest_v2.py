#!/usr/bin/env python3
"""
backtest_v2.py — 6-season backtest of the improved EPA model (no Vegas shrinkage)
Usage: python epa_model/backtest_v2.py [--seasons 2019 2020 ...] [--sims N]

This is the clean version:
  • No Bayesian shrinkage toward Vegas — pure EPA predictions
  • Recency-weighted, turnover-neutral, pass/rush split, SOS-adjusted metrics
  • EPA_TO_PTS calibrated empirically (0.70, not the guessed 0.45)
  • Week 1 uses prior season full-year metrics (no current-season lookahead)
  • vs. CLOSING lines (only lines available in nfl_data_py)
"""

import argparse, os, sys, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import nfl_data_py as nfl

sys.path.insert(0, os.path.dirname(__file__))
from metrics_v2  import compute_all_teams_v2
from simulator_v2 import simulate_game_v2

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

TEAM_ALIASES = {"OAK": "LV", "SD": "LAC", "STL": "LA"}

def norm(t):
    return TEAM_ALIASES.get(str(t).upper(), str(t).upper()) if pd.notna(t) else t

def load_pbp(season):
    path = os.path.join(CACHE_DIR, f"pbp_{season}.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    print(f"  Downloading {season} PBP...")
    pbp = nfl.import_pbp_data([season])
    pbp.to_parquet(path, index=False)
    return pbp

def load_schedule(season):
    path = os.path.join(CACHE_DIR, f"schedule_{season}_full.parquet")
    if os.path.exists(path):
        s = pd.read_parquet(path)
        if s[s["game_type"]=="REG"]["home_score"].isna().any():
            os.remove(path)
        else:
            return s
    s = nfl.import_schedules([season])
    s.to_parquet(path, index=False)
    return s

def grade(r, hs, as_):
    margin, total = hs - as_, hs + as_
    ats = tot = win = None
    vs, vt = r.get("vegas_spread"), r.get("vegas_total")
    if r.get("ats_pick") and vs is not None:
        d = margin + vs
        if d == 0:   ats = "Push"
        elif r["ats_pick"] == r["home_team"]: ats = d > 0
        else: ats = d < 0
    if r.get("total_pick") and vt is not None:
        d = total - vt
        if d == 0:   tot = "Push"
        elif r["total_pick"] == "Over": tot = d > 0
        else: tot = d < 0
    if hs != as_:
        pred   = r["home_team"] if r["home_win_prob"] >= 0.5 else r["away_team"]
        actual = r["home_team"] if hs > as_ else r["away_team"]
        win = pred == actual
    return ats, tot, win

def tally(records, key):
    w = sum(1 for r in records if r[key] is True)
    l = sum(1 for r in records if r[key] is False)
    p = sum(1 for r in records if r[key] == "Push")
    return w, l, p

def fmt(w, l, p=0): return f"{w}-{l}" + (f"-{p}" if p else "")
def pct(w, l): return w/(w+l)*100 if w+l else float("nan")
def pl(w, l, stake=100): return w*(stake*100/110) - l*stake

def run_season(season, pbp_curr, pbp_prior, sched, sims, rng):
    pbp_curr["week"] = pd.to_numeric(pbp_curr["week"], errors="coerce")
    sched["week"]    = pd.to_numeric(sched["week"],    errors="coerce")

    reg = sched[sched["game_type"] == "REG"].copy()
    reg["home_team"] = reg["home_team"].apply(norm)
    reg["away_team"] = reg["away_team"].apply(norm)
    weeks = sorted(reg["week"].dropna().unique().astype(int))

    # Pre-compute prior season metrics for week 1
    prior_all = {}
    if pbp_prior is not None:
        pbp_prior["week"] = pd.to_numeric(pbp_prior["week"], errors="coerce")
        max_reg_wk = 17
        prior_all = compute_all_teams_v2(pbp_prior, max_reg_wk)
        # Normalize team abbrevs
        prior_all = {norm(k): v for k, v in prior_all.items()}
        for v in prior_all.values():
            v["team"] = norm(v["team"])

    records = []
    curr_all_cache = {}   # cache metrics by through_week to avoid recomputing per game

    for week in weeks:
        wk_games = reg[reg["week"] == week].dropna(subset=["home_score","away_score"])
        if wk_games.empty:
            continue

        metrics_week = max(week - 1, 0)

        if week == 1:
            all_metrics = prior_all
        else:
            if metrics_week not in curr_all_cache:
                curr_all_cache[metrics_week] = compute_all_teams_v2(pbp_curr, metrics_week)
                # Normalize
                nm = {}
                for k, v in curr_all_cache[metrics_week].items():
                    nk = norm(k)
                    v["team"] = nk
                    nm[nk] = v
                curr_all_cache[metrics_week] = nm
            all_metrics = curr_all_cache[metrics_week]

        def get_m(team):
            tn = norm(team)
            return all_metrics.get(tn, _league_avg(tn))

        for _, game in wk_games.iterrows():
            home = norm(game["home_team"])
            away = norm(game["away_team"])
            hs   = float(game["home_score"])
            as_  = float(game["away_score"])
            vs   = float(game["spread_line"]) if pd.notna(game.get("spread_line")) else None
            vt   = float(game["total_line"])  if pd.notna(game.get("total_line"))  else None

            sim = simulate_game_v2(get_m(home), get_m(away),
                                   vegas_spread=vs, vegas_total=vt,
                                   sims=sims, rng=rng)
            sim.update({"season": season, "week": week,
                        "actual_home": hs, "actual_away": as_,
                        "spread_error": sim["spread_mean"] - (hs - as_)})
            ar, tr, wr = grade(sim, hs, as_)
            sim.update({"ats_result": ar, "tot_result": tr, "win_result": wr})
            records.append(sim)

    return records

def _league_avg(team):
    return {"team": team,
            "off_epa": 0.0, "def_epa": 0.0,
            "pass_off_epa": 0.0, "rush_off_epa": 0.0,
            "pass_def_epa": 0.0, "rush_def_epa": 0.0,
            "off_epa_sd": 0.08, "def_epa_sd": 0.08,
            "avg_pass_plays": 35.0, "avg_rush_plays": 27.0,
            "sd_pass_plays": 8.0, "sd_rush_plays": 7.0,
            "n_games": 0}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", type=int, nargs="+",
                        default=[2019, 2020, 2021, 2022, 2023, 2024])
    parser.add_argument("--sims",   type=int, default=2000)
    parser.add_argument("--seed",   type=int, default=42)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    print(f"\n{'='*80}")
    print(f"  EPA v2 Backtest — PURE MODEL, ZERO VEGAS SHRINKAGE")
    print(f"  Seasons: {args.seasons}  |  {args.sims:,} sims/game  |  vs. closing lines")
    print(f"{'='*80}")
    print(f"  Improvements vs v1:")
    print(f"    EPA_TO_PTS: 0.45 → 0.70 (empirically regressed from 2022-24 data)")
    print(f"    Shrinkage: τ=14 toward Vegas → ZERO (model is now truly independent)")
    print(f"    EPA signal: single all-play EPA → pass/rush split + turnover-neutral")
    print(f"    Weighting: equal → recency-weighted (half-life 5.8 weeks)")
    print(f"    Opponents: raw EPA → SOS-adjusted (one-pass opponent correction)")
    print(f"    Week 1: league average → prior season full-year EPA\n")

    all_records = []

    hdr = f"  {'SEASON':>6}  {'GAMES':>5}  {'ATS':>9}  {'ATS%':>6}  {'TOT':>9}  {'TOT%':>6}  {'WIN':>7}  {'WIN%':>6}  {'MAE':>6}"
    div = f"  {'─'*len(hdr.strip())}"
    print(div); print(hdr); print(div)

    for season in args.seasons:
        print(f"  Loading {season}...", end="", flush=True)
        pbp_curr  = load_pbp(season)
        try:    pbp_prior = load_pbp(season - 1)
        except: pbp_prior = None
        sched = load_schedule(season)
        print(" running...", end="", flush=True)

        recs = run_season(season, pbp_curr, pbp_prior, sched, args.sims, rng)
        all_records.extend(recs)

        wa, la, pa = tally(recs, "ats_result")
        wt, lt, pt = tally(recs, "tot_result")
        ww, lw, _  = tally(recs, "win_result")
        mae = np.mean([abs(r["spread_error"]) for r in recs if r.get("vegas_spread") is not None])

        print(f"\r  {season:>6}  {len(recs):>5}  {fmt(wa,la,pa):>9}  {pct(wa,la):>5.1f}%"
              f"  {fmt(wt,lt,pt):>9}  {pct(wt,lt):>5.1f}%"
              f"  {fmt(ww,lw):>7}  {pct(ww,lw):>5.1f}%  {mae:>6.2f}")

    print(div)
    wa, la, pa = tally(all_records, "ats_result")
    wt, lt, pt = tally(all_records, "tot_result")
    ww, lw, _  = tally(all_records, "win_result")
    mae_all = np.mean([abs(r["spread_error"]) for r in all_records if r.get("vegas_spread") is not None])

    print(f"  {'TOTAL':>6}  {len(all_records):>5}  {fmt(wa,la,pa):>9}  {pct(wa,la):>5.1f}%"
          f"  {fmt(wt,lt,pt):>9}  {pct(wt,lt):>5.1f}%"
          f"  {fmt(ww,lw):>7}  {pct(ww,lw):>5.1f}%  {mae_all:>6.2f}")
    print()

    print(f"  ── P&L at -110 ($100/game) ─────────────────────────────────────────")
    print(f"  ATS  : ${pl(wa,la):>+10,.0f}  ({pct(wa,la):.1f}%)   breakeven: 52.4%")
    print(f"  TOTAL: ${pl(wt,lt):>+10,.0f}  ({pct(wt,lt):.1f}%)   breakeven: 52.4%")
    print(f"  Spread MAE: {mae_all:.2f} pts  (Vegas baseline ≈ 7.5 pts)")

    be = 52.4
    print(f"\n  ── Honest interpretation ────────────────────────────────────────────")
    ats_pct = pct(wa, la)
    if ats_pct > 55:
        print(f"  ATS {ats_pct:.1f}% with ZERO Vegas input is a genuine signal worth investigating.")
        print(f"  Next steps: test on opening lines, add QB-level data, validate on 2026.")
    elif ats_pct > be:
        print(f"  ATS {ats_pct:.1f}% > breakeven ({be}%) suggests a marginal edge.")
        print(f"  Sample variance over 1 season could flip this — more seasons needed.")
    else:
        print(f"  ATS {ats_pct:.1f}% ≤ breakeven. Model has no detectable edge without Vegas shrinkage.")
        print(f"  The v1 73% was entirely driven by shrinkage, not EPA insight.")
    print()


if __name__ == "__main__":
    main()
