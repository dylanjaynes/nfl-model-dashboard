"""
weekly.py — Live weekly predictions for the current NFL season

Usage:
    python3 epa_model/weekly.py                        # auto-detect next week
    python3 epa_model/weekly.py --season 2025 --week 14
    python3 epa_model/weekly.py --season 2025 --week 14 --sims 5000

How it works:
  • Loads PBP from local cache (or downloads fresh if missing)
  • Computes SOS-adjusted, recency-weighted, turnover-neutral EPA through week-1
  • Runs Monte Carlo simulation (zero Vegas shrinkage — pure EPA signal)
  • If game scores are available (past week): grades ATS / Total / ML
  • If no scores yet (upcoming week): prints predictions only
"""

import argparse
import importlib.util
import os
import sys
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ── resolve paths ──────────────────────────────────────────────────────────────
HERE   = Path(__file__).parent
CACHE  = HERE / "cache"
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
        print("Installing nfl_data_py…")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "nfl_data_py", "-q"])

# ── team alias normalization ───────────────────────────────────────────────────
ALIASES = {"OAK": "LV", "SD": "LAC", "STL": "LA"}

def _norm(t):
    if not isinstance(t, str): return t
    return ALIASES.get(t, t)

# ── PBP / schedule loaders ────────────────────────────────────────────────────
def _load_pbp(season: int) -> pd.DataFrame:
    cache_file = CACHE / f"pbp_{season}.parquet"
    if cache_file.exists():
        pbp = pd.read_parquet(cache_file)
    else:
        _ensure_nfl()
        import nfl_data_py as nfl
        print(f"  Downloading {season} PBP data (one-time, ~60–90 s)…")
        pbp = nfl.import_pbp_data([season], downcast=True)
        pbp.to_parquet(cache_file, index=False)
        print(f"  Cached to {cache_file}")
    # normalise team names
    for col in ["posteam", "defteam", "home_team", "away_team"]:
        if col in pbp.columns:
            pbp[col] = pbp[col].map(lambda x: ALIASES.get(x, x) if isinstance(x, str) else x)
    return pbp

def _load_schedule(season: int) -> pd.DataFrame:
    for stem in [f"schedule_{season}_full", f"schedule_{season}"]:
        f = CACHE / f"{stem}.parquet"
        if f.exists():
            sched = pd.read_parquet(f)
            break
    else:
        _ensure_nfl()
        import nfl_data_py as nfl
        print(f"  Downloading {season} schedule…")
        sched = nfl.import_schedules([season])
        sched.to_parquet(CACHE / f"schedule_{season}_full.parquet", index=False)
    sched["home_team"] = sched["home_team"].map(_norm)
    sched["away_team"] = sched["away_team"].map(_norm)
    return sched

# ── auto-detect next upcoming week ────────────────────────────────────────────
def _detect_week(sched: pd.DataFrame) -> int:
    """Return lowest week that has unplayed games (no home_score yet)."""
    reg = sched[sched["game_type"] == "REG"].copy()
    unplayed = reg[reg["home_score"].isna()]
    if unplayed.empty:
        # Season complete — return last week
        return int(reg["week"].max())
    return int(unplayed["week"].min())

# ── main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="EPA-based weekly NFL predictions")
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--week",   type=int, default=None,
                    help="Week to predict (default: auto-detect next week)")
    ap.add_argument("--sims",   type=int, default=5000)
    args = ap.parse_args()

    m2 = _load("metrics_v2")
    s2 = _load("simulator_v2")

    print(f"\nLoading {args.season} data…")
    pbp   = _load_pbp(args.season)
    sched = _load_schedule(args.season)

    reg   = sched[sched["game_type"] == "REG"].copy()

    week  = args.week if args.week else _detect_week(reg)
    print(f"→ Predicting Season {args.season}  Week {week}\n")

    # Determine metrics source
    if week == 1:
        print(f"  Week 1: loading {args.season - 1} prior-season metrics…")
        pbp_prior = _load_pbp(args.season - 1)
        pbp_prior = pbp_prior[pbp_prior["season_type"] == "REG"].copy()
        metrics   = m2.compute_all_teams_v2(pbp_prior, through_week=17)
    else:
        pbp_reg  = pbp[pbp["season_type"] == "REG"].copy()
        metrics  = m2.compute_all_teams_v2(pbp_reg, through_week=week - 1)

    week_games = reg[reg["week"] == week].copy()
    if week_games.empty:
        print(f"No games found for Week {week}. Check --season / --week.")
        return

    has_scores = week_games["home_score"].notna().any()

    rng = np.random.default_rng(42)

    results = []
    for _, row in week_games.sort_values("gameday").iterrows():
        ht, at = _norm(row["home_team"]), _norm(row["away_team"])
        hm = metrics.get(ht)
        am = metrics.get(at)

        if hm is None or am is None:
            missing = ht if hm is None else at
            print(f"  ⚠  No metrics for {missing} — skipping")
            continue

        vs  = row.get("vegas_spread") if "vegas_spread" in row.index else None
        vt  = row.get("vegas_total")  if "vegas_total"  in row.index else None
        if vs is None or not isinstance(vs, float) or not np.isfinite(vs):
            vs = row.get("spread_line")
        if vt is None or not isinstance(vt, float) or not np.isfinite(vt):
            vt = row.get("total_line")

        try:
            vs = float(vs) if vs is not None and pd.notna(vs) else None
        except (TypeError, ValueError):
            vs = None
        try:
            vt = float(vt) if vt is not None and pd.notna(vt) else None
        except (TypeError, ValueError):
            vt = None

        res = s2.simulate_game_v2(hm, am, vegas_spread=vs, vegas_total=vt,
                                  sims=args.sims, rng=rng)

        actual_home = row.get("home_score")
        actual_away = row.get("away_score")
        scored = (pd.notna(actual_home) and pd.notna(actual_away))

        results.append({
            **res,
            "gameday":     row.get("gameday", ""),
            "actual_home": float(actual_home) if scored else None,
            "actual_away": float(actual_away) if scored else None,
            "scored":      scored,
        })

    if not results:
        print("No valid games to simulate.")
        return

    # ── Print table ───────────────────────────────────────────────────────────
    hdr_pred = (
        f"{'MATCHUP':<22} {'MDL SPR':>8} {'VEG SPR':>8} "
        f"{'MDL TOT':>8} {'VEG TOT':>8} {'WIN%':>6} {'ML':>6}"
    )
    if has_scores:
        hdr_pred += f"  {'ACTUAL':>12}  ATS  TOT  ML"

    print(f"{'═'*len(hdr_pred)}")
    print(f"  Season {args.season}  ·  Week {week}  ·  EPA Model (pure signal, zero shrinkage)")
    print(f"{'═'*len(hdr_pred)}")
    print(hdr_pred)
    print(f"{'─'*len(hdr_pred)}")

    ats_w = ats_l = ats_p = 0
    tot_w = tot_l           = 0
    ml_w  = ml_l            = 0

    for r in results:
        ht, at  = r["home_team"], r["away_team"]
        mdl_spr = r["spread_mean"]
        veg_spr = r["vegas_spread"]
        mdl_tot = r["total_mean"]
        veg_tot = r["vegas_total"]
        win_pct = r["home_win_prob"] * 100
        ml_str  = (f"{r['home_american']:+d}" if r["home_american"] is not None
                   else "N/A")

        fav     = ht if mdl_spr >= 0 else at
        dog     = at if mdl_spr >= 0 else ht
        spr_val = abs(mdl_spr)
        matchup = f"{fav} -{spr_val:.1f} {dog}"

        vs_str  = f"{veg_spr:+.1f}" if veg_spr is not None else "  N/A"
        vt_str  = f"{veg_tot:.1f}"  if veg_tot is not None else " N/A"

        row_str = (
            f"  {matchup:<22} {mdl_spr:>+8.1f} {vs_str:>8} "
            f"{mdl_tot:>8.1f} {vt_str:>8} {win_pct:>5.1f}% {ml_str:>6}"
        )

        if has_scores and r["scored"]:
            ah, aa = r["actual_home"], r["actual_away"]
            score_str = f"{ht} {int(ah)}-{int(aa)} {at}"

            # ATS grade
            actual_margin = ah - aa
            if veg_spr is not None:
                if actual_margin > -veg_spr + 0.5:
                    ats_res = "✅" if r["ats_pick"] == ht else "❌"
                    ats_w += 1 if r["ats_pick"] == ht else 0
                    ats_l += 0 if r["ats_pick"] == ht else 1
                elif actual_margin < -veg_spr - 0.5:
                    ats_res = "✅" if r["ats_pick"] == at else "❌"
                    ats_w += 0 if r["ats_pick"] == ht else 1
                    ats_l += 1 if r["ats_pick"] == ht else 0
                else:
                    ats_res = "⬛"
                    ats_p  += 1
            else:
                ats_res = "  "

            # Totals grade
            actual_total = ah + aa
            if veg_tot is not None:
                if actual_total > veg_tot + 0.5:
                    tot_res = "✅" if r["total_pick"] == "Over"  else "❌"
                    tot_w  += 1 if r["total_pick"] == "Over" else 0
                    tot_l  += 0 if r["total_pick"] == "Over" else 1
                elif actual_total < veg_tot - 0.5:
                    tot_res = "✅" if r["total_pick"] == "Under" else "❌"
                    tot_w  += 0 if r["total_pick"] == "Over" else 1
                    tot_l  += 1 if r["total_pick"] == "Over" else 0
                else:
                    tot_res = "⬛"
            else:
                tot_res = "  "

            # ML grade
            if ah > aa:
                ml_res = "✅" if r["home_win_prob"] > 0.5 else "❌"
                ml_w  += 1 if r["home_win_prob"] > 0.5 else 0
                ml_l  += 0 if r["home_win_prob"] > 0.5 else 1
            else:
                ml_res = "✅" if r["home_win_prob"] < 0.5 else "❌"
                ml_w  += 0 if r["home_win_prob"] > 0.5 else 1
                ml_l  += 1 if r["home_win_prob"] > 0.5 else 0

            row_str += f"  {score_str:<14}  {ats_res}   {tot_res}   {ml_res}"

        print(row_str)

    print(f"{'─'*len(hdr_pred)}")

    if has_scores:
        ats_tot = ats_w + ats_l
        tot_tot = tot_w + tot_l
        ml_tot  = ml_w  + ml_l
        ats_pct = ats_w / ats_tot * 100 if ats_tot else 0
        tot_pct = tot_w / tot_tot * 100 if tot_tot else 0
        ml_pct  = ml_w  / ml_tot  * 100 if ml_tot  else 0

        push_str = f" {ats_p}P" if ats_p else ""
        print(f"\n  ATS: {ats_w}-{ats_l}{push_str} ({ats_pct:.1f}%)   "
              f"TOT: {tot_w}-{tot_l} ({tot_pct:.1f}%)   "
              f"ML:  {ml_w}-{ml_l} ({ml_pct:.1f}%)")
    else:
        print(f"\n  {len(results)} games — scores not yet available (upcoming week)")

    print()


if __name__ == "__main__":
    main()
