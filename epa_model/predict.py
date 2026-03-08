#!/usr/bin/env python3
"""
predict.py — EPA-based NFL game prediction model
Usage: python epa_model/predict.py [--season YEAR] [--week N] [--sims N]

Fetches PBP and schedule data via nfl_data_py, runs Monte Carlo simulation,
prints predictions, and grades against actuals if the week is complete.
"""

import argparse
import os
import sys
import warnings
warnings.filterwarnings("ignore")

# ── Dependency check ───────────────────────────────────────────────────────────
def ensure_deps():
    import importlib.util, subprocess
    required = ["nfl_data_py", "numpy", "pandas", "scipy"]
    for pkg in required:
        import_name = pkg.replace("-", "_")
        if importlib.util.find_spec(import_name) is None:
            print(f"  Installing {pkg}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

ensure_deps()

import numpy as np
import pandas as pd
import nfl_data_py as nfl

# Add parent dir so we can import from this subfolder when run from project root
sys.path.insert(0, os.path.dirname(__file__))
from metrics import compute_team_epa
from simulator import simulate_game


# ── Cache helpers ──────────────────────────────────────────────────────────────
CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)


def load_pbp(season: int) -> pd.DataFrame:
    cache_path = os.path.join(CACHE_DIR, f"pbp_{season}.parquet")
    if os.path.exists(cache_path):
        print(f"  Loading PBP from cache ({cache_path})...")
        return pd.read_parquet(cache_path)
    print(f"  Downloading {season} PBP (one-time, ~30–60s)...")
    pbp = nfl.import_pbp_data([season])
    pbp.to_parquet(cache_path, index=False)
    print(f"  Cached to {cache_path}")
    return pbp


def load_schedule(season: int) -> pd.DataFrame:
    cache_path = os.path.join(CACHE_DIR, f"schedule_{season}.parquet")
    if os.path.exists(cache_path):
        sched = pd.read_parquet(cache_path)
        # Refresh if it looks stale (scores not filled for recent weeks)
        if sched["home_score"].isna().any():
            os.remove(cache_path)
        else:
            return sched
    print(f"  Loading {season} schedule...")
    sched = nfl.import_schedules([season])
    sched.to_parquet(cache_path, index=False)
    return sched


# ── Formatting helpers ─────────────────────────────────────────────────────────
def fmt_prob(p) -> str:
    return f"{p*100:.0f}%" if p is not None else "  N/A"

def fmt_odds(o) -> str:
    if o is None:
        return "  N/A"
    return f"+{o}" if o > 0 else str(o)

def fmt_spread(s) -> str:
    if s is None:
        return " N/A"
    return f"+{s:.1f}" if s > 0 else f"{s:.1f}"

def grade_bet(result, pick, actual_home, actual_away, vegas_spread, vegas_total):
    """Returns True/False/None (push) for ATS and Total bets."""
    ats_result = total_result = None

    actual_margin = actual_home - actual_away
    actual_total  = actual_home + actual_away

    if result["ats_pick"] and vegas_spread is not None:
        ats_margin = actual_margin + vegas_spread   # > 0 = home covered
        if ats_margin == 0:
            ats_result = "Push"
        elif result["ats_pick"] == result["home_team"]:
            ats_result = ats_margin > 0
        else:
            ats_result = ats_margin < 0

    if result["total_pick"] and vegas_total is not None:
        tot_diff = actual_total - vegas_total
        if tot_diff == 0:
            total_result = "Push"
        elif result["total_pick"] == "Over":
            total_result = tot_diff > 0
        else:
            total_result = tot_diff < 0

    win_result = None
    if actual_home != actual_away:
        predicted_winner = result["home_team"] if result["home_win_prob"] >= 0.5 else result["away_team"]
        actual_winner    = result["home_team"] if actual_home > actual_away else result["away_team"]
        win_result = predicted_winner == actual_winner

    return ats_result, total_result, win_result


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="EPA NFL prediction model")
    parser.add_argument("--season", type=int, default=2024)
    parser.add_argument("--week",   type=int, default=18)
    parser.add_argument("--sims",   type=int, default=5000)
    parser.add_argument("--seed",   type=int, default=42)
    args = parser.parse_args()

    season, week, sims = args.season, args.week, args.sims
    rng = np.random.default_rng(args.seed)

    print(f"\n{'='*65}")
    print(f"  EPA Model  |  Season {season}  |  Week {week}  |  {sims:,} sims")
    print(f"{'='*65}\n")

    # ── Load data ──────────────────────────────────────────────────────────────
    pbp   = load_pbp(season)
    sched = load_schedule(season)

    # Standardize week column type
    pbp["week"]   = pd.to_numeric(pbp["week"], errors="coerce")
    sched["week"] = pd.to_numeric(sched["week"], errors="coerce")

    # Filter schedule to target week, regular season only
    week_sched = sched[
        (sched["week"] == week) &
        (sched["game_type"] == "REG")
    ].copy()

    if week_sched.empty:
        print(f"No regular season games found for {season} Week {week}.")
        sys.exit(1)

    # ── Compute team EPA through week - 1 (no lookahead) ─────────────────────
    metrics_week = week - 1
    if metrics_week < 1:
        print("Warning: Week 1 — using prior season not implemented, using week 1 data.")
        metrics_week = 1

    print(f"  Computing team EPA metrics through Week {metrics_week}...")
    team_cache = {}

    def get_metrics(team):
        if team not in team_cache:
            team_cache[team] = compute_team_epa(team, pbp, metrics_week)
        return team_cache[team]

    # ── Simulate all games ─────────────────────────────────────────────────────
    results = []
    for _, game in week_sched.iterrows():
        home = game.get("home_team")
        away = game.get("away_team")
        if pd.isna(home) or pd.isna(away):
            continue

        vegas_spread = game.get("spread_line")   # home perspective
        vegas_total  = game.get("total_line")

        # Convert to float, treat 0 as None (missing)
        vegas_spread = float(vegas_spread) if pd.notna(vegas_spread) else None
        vegas_total  = float(vegas_total)  if pd.notna(vegas_total)  else None

        t1 = get_metrics(home)
        t2 = get_metrics(away)

        sim = simulate_game(t1, t2, vegas_spread=vegas_spread, vegas_total=vegas_total,
                            sims=sims, rng=rng)
        sim["game_id"]     = game.get("game_id", f"{season}_{week:02d}_{home}_{away}")
        sim["gameday"]     = str(game.get("gameday", ""))[:10]
        sim["home_score"]  = game.get("home_score")
        sim["away_score"]  = game.get("away_score")
        results.append(sim)

    # ── Print predictions table ────────────────────────────────────────────────
    has_actuals = all(
        r["home_score"] is not None and pd.notna(r.get("home_score", None))
        for r in results
    )

    header = (
        f"  {'MATCHUP':<28} {'MDL SPR':>7}  {'VEG SPR':>7}  "
        f"{'MDL TOT':>7}  {'ATS PICK':>10}  {'ATS%':>5}  "
        f"{'TOT':>5}  {'TOT%':>5}  {'WIN%':>5}"
    )
    if has_actuals:
        header += f"  {'ACTUAL':>11}  {'ATS':>4}  {'TOT':>4}"

    print(f"\n{'─'*len(header)}")
    print(header)
    print(f"{'─'*len(header)}")

    ats_record   = {"W": 0, "L": 0, "P": 0}
    tot_record   = {"W": 0, "L": 0, "P": 0}
    win_record   = {"W": 0, "L": 0}
    spread_errors = []

    for r in sorted(results, key=lambda x: x["gameday"]):
        home = r["home_team"]
        away = r["away_team"]
        hs   = r["home_score"]
        as_  = r["away_score"]

        matchup = f"{away} @ {home}"
        model_spread = r["spread_mean"]
        vegas_spread = r["vegas_spread"]
        model_total  = r["total_mean"]

        ats_pick  = r["ats_pick"]  or "—"
        ats_pct   = fmt_prob(r["ats_prob"])
        tot_pick  = r["total_pick"] or "—"
        tot_pct   = fmt_prob(r["total_prob"])
        win_pct   = fmt_prob(r["home_win_prob"])

        line = (
            f"  {matchup:<28} {fmt_spread(model_spread):>7}  "
            f"{fmt_spread(vegas_spread):>7}  "
            f"{model_total:>7.1f}  {ats_pick:>10}  {ats_pct:>5}  "
            f"{tot_pick:>5}  {tot_pct:>5}  {win_pct:>5}"
        )

        if has_actuals and pd.notna(hs) and pd.notna(as_):
            ats_res, tot_res, win_res = grade_bet(
                r, ats_pick, float(hs), float(as_), vegas_spread, r["vegas_total"]
            )
            actual_str = f"{int(hs):>3}-{int(as_):<3}"

            ats_str = "W" if ats_res is True else ("L" if ats_res is False else ("P" if ats_res == "Push" else "?"))
            tot_str = "W" if tot_res is True else ("L" if tot_res is False else ("P" if tot_res == "Push" else "?"))

            if ats_res is True:  ats_record["W"] += 1
            elif ats_res is False: ats_record["L"] += 1
            elif ats_res == "Push": ats_record["P"] += 1

            if tot_res is True:  tot_record["W"] += 1
            elif tot_res is False: tot_record["L"] += 1
            elif tot_res == "Push": tot_record["P"] += 1

            if win_res is True:  win_record["W"] += 1
            elif win_res is False: win_record["L"] += 1

            if vegas_spread is not None:
                spread_errors.append(abs(model_spread - (float(hs) - float(as_))))

            line += f"  {actual_str}  {ats_str:>4}  {tot_str:>4}"

        print(line)

    print(f"{'─'*len(header)}\n")

    # ── Grading summary ────────────────────────────────────────────────────────
    if has_actuals:
        n_ats = ats_record["W"] + ats_record["L"]
        n_tot = tot_record["W"] + tot_record["L"]
        n_win = win_record["W"] + win_record["L"]

        ats_rate = ats_record["W"] / n_ats if n_ats > 0 else 0
        tot_rate = tot_record["W"] / n_tot if n_tot > 0 else 0
        win_rate = win_record["W"] / n_win if n_win > 0 else 0
        mae      = np.mean(spread_errors) if spread_errors else float("nan")

        print(f"  {'RESULTS':}")
        print(f"  ATS :  {ats_record['W']}-{ats_record['L']}"
              + (f"-{ats_record['P']}" if ats_record['P'] else "")
              + f"  ({ats_rate*100:.1f}%)")
        print(f"  TOT :  {tot_record['W']}-{tot_record['L']}"
              + (f"-{tot_record['P']}" if tot_record['P'] else "")
              + f"  ({tot_rate*100:.1f}%)")
        print(f"  WIN :  {win_record['W']}-{win_record['L']}  ({win_rate*100:.1f}%)")
        print(f"  Spread MAE: {mae:.2f} pts")
        print()

    print(f"  Model params: EPA→pts={0.45}, HFA={1.5}pts, τ={14.0}, "
          f"t-shock(df={6}, σ={3.0})")
    print()


if __name__ == "__main__":
    main()
