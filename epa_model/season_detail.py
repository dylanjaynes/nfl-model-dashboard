#!/usr/bin/env python3
"""
season_detail.py — Game-by-game predictions for a full season (v2 model)
Usage: python epa_model/season_detail.py [--season YEAR] [--sims N]
"""

import argparse, os, sys, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import nfl_data_py as nfl

sys.path.insert(0, os.path.dirname(__file__))
from metrics_v2   import compute_all_teams_v2
from simulator_v2 import simulate_game_v2

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
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

def _league_avg(team):
    return {"team": team,
            "off_epa": 0.0, "def_epa": 0.0,
            "pass_off_epa": 0.0, "rush_off_epa": 0.0,
            "pass_def_epa": 0.0, "rush_def_epa": 0.0,
            "off_epa_sd": 0.08, "def_epa_sd": 0.08,
            "avg_pass_plays": 35.0, "avg_rush_plays": 27.0,
            "sd_pass_plays": 8.0,  "sd_rush_plays": 7.0,
            "n_games": 0}

def grade(r, hs, as_):
    margin, total = hs - as_, hs + as_
    ats = tot = win = None
    vs, vt = r.get("vegas_spread"), r.get("vegas_total")
    if r.get("ats_pick") and vs is not None:
        d = margin + vs
        ats = "P" if d == 0 else ("W" if (r["ats_pick"]==r["home_team"]) == (d>0) else "L")
    if r.get("total_pick") and vt is not None:
        d = total - vt
        tot = "P" if d == 0 else ("W" if (r["total_pick"]=="Over") == (d>0) else "L")
    if hs != as_:
        pred   = r["home_team"] if r["home_win_prob"] >= 0.5 else r["away_team"]
        actual = r["home_team"] if hs > as_ else r["away_team"]
        win = "W" if pred == actual else "L"
    return ats or "?", tot or "?", win or "-"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument("--sims",   type=int, default=3000)
    parser.add_argument("--seed",   type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    season = args.season

    print(f"\n{'='*105}")
    print(f"  EPA v2 Model  |  {season} Season Game-by-Game  |  {args.sims:,} sims  |  Zero Vegas shrinkage")
    print(f"{'='*105}\n")

    pbp_curr  = load_pbp(season)
    pbp_prior = load_pbp(season - 1)
    sched     = load_schedule(season)

    pbp_curr["week"] = pd.to_numeric(pbp_curr["week"], errors="coerce")
    sched["week"]    = pd.to_numeric(sched["week"],    errors="coerce")

    reg = sched[sched["game_type"] == "REG"].copy()
    reg["home_team"] = reg["home_team"].apply(norm)
    reg["away_team"] = reg["away_team"].apply(norm)
    weeks = sorted(reg["week"].dropna().unique().astype(int))

    pbp_prior["week"] = pd.to_numeric(pbp_prior["week"], errors="coerce")
    prior_all = {norm(k): {**v, "team": norm(v["team"])}
                 for k, v in compute_all_teams_v2(pbp_prior, 17).items()}

    metrics_cache = {}
    def get_week_metrics(week):
        if week not in metrics_cache:
            raw = compute_all_teams_v2(pbp_curr, week)
            metrics_cache[week] = {norm(k): {**v, "team": norm(v["team"])} for k, v in raw.items()}
        return metrics_cache[week]

    # Column header
    H = (f"  {'MATCHUP':<26} {'DATE':>6}  {'MDL':>6} {'VEG':>6}  "
         f"{'MDL TOT':>7} {'VEG TOT':>7}  "
         f"{'WIN%':>5}  {'ATS PICK':>8}  {'ACTUAL':>8}  "
         f"{'ATS':>3} {'TOT':>3} {'ML':>3}")
    DIV = "  " + "─" * (len(H) - 2)

    # Season totals
    s_ats = {"W":0,"L":0,"P":0}
    s_tot = {"W":0,"L":0,"P":0}
    s_win = {"W":0,"L":0}

    for week in weeks:
        wk_games = reg[reg["week"] == week].copy()
        has_scores = wk_games["home_score"].notna().all()

        all_m = prior_all if week == 1 else get_week_metrics(week - 1)

        print(f"  {'─'*10}  WEEK {week}  {'─'*10}")
        print(H)
        print(DIV)

        wk_ats = {"W":0,"L":0,"P":0}
        wk_tot = {"W":0,"L":0,"P":0}
        wk_win = {"W":0,"L":0}

        for _, game in wk_games.sort_values("gametime").iterrows():
            home = norm(game["home_team"])
            away = norm(game["away_team"])
            hs   = game.get("home_score")
            as_  = game.get("away_score")
            vs   = float(game["spread_line"]) if pd.notna(game.get("spread_line")) else None
            vt   = float(game["total_line"])  if pd.notna(game.get("total_line"))  else None
            date = str(game.get("gameday",""))[:5]

            hm = all_m.get(home, _league_avg(home))
            am = all_m.get(away, _league_avg(away))

            sim = simulate_game_v2(hm, am, vegas_spread=vs, vegas_total=vt,
                                   sims=args.sims, rng=rng)

            mdl_spr = sim["spread_mean"]
            mdl_tot = sim["total_mean"]
            win_pct = sim["home_win_prob"] * 100
            ats_pk  = sim.get("ats_pick") or "—"
            tot_pk  = sim.get("total_pick") or "—"

            matchup = f"{away} @ {home}"
            mdl_spr_s = (f"+{mdl_spr:.1f}" if mdl_spr > 0 else f"{mdl_spr:.1f}")
            veg_spr_s = (f"+{vs:.1f}" if vs and vs > 0 else (f"{vs:.1f}" if vs else "  N/A"))

            if has_scores and pd.notna(hs) and pd.notna(as_):
                hs_i, as_i = int(hs), int(as_)
                a_str, t_str, w_str = grade(sim, float(hs), float(as_))
                actual_s = f"{hs_i:>3}-{as_i:<3}"
                line = (f"  {matchup:<26} {date:>6}  {mdl_spr_s:>6} {veg_spr_s:>6}  "
                        f"{mdl_tot:>7.1f} {vt:>7.1f}  "
                        f"{win_pct:>4.0f}%  {ats_pk:>8}  {actual_s:>8}  "
                        f"{a_str:>3} {t_str:>3} {w_str:>3}")
                for k in ("W","L","P"):
                    if a_str == k: wk_ats[k] += 1
                    if t_str == k: wk_tot[k] += 1
                if w_str == "W": wk_win["W"] += 1
                elif w_str == "L": wk_win["L"] += 1
            else:
                vt_s = f"{vt:.1f}" if vt else "  N/A"
                line = (f"  {matchup:<26} {date:>6}  {mdl_spr_s:>6} {veg_spr_s:>6}  "
                        f"{mdl_tot:>7.1f} {vt_s:>7}  "
                        f"{win_pct:>4.0f}%  {ats_pk:>8}  {'TBD':>8}  "
                        f"{'—':>3} {'—':>3} {'—':>3}")
            print(line)

        # Week summary
        if has_scores:
            na = wk_ats["W"] + wk_ats["L"]
            nt = wk_tot["W"] + wk_tot["L"]
            nw = wk_win["W"] + wk_win["L"]
            push_s = lambda d: f"{d['W']}-{d['L']}" + (f"-{d['P']}" if d["P"] else "")
            ats_p = wk_ats["W"]/na*100 if na else 0
            tot_p = wk_tot["W"]/nt*100 if nt else 0
            win_p = wk_win["W"]/nw*100 if nw else 0
            print(DIV)
            print(f"  Week {week:>2} summary:  "
                  f"ATS {push_s(wk_ats)} ({ats_p:.0f}%)  "
                  f"TOT {push_s(wk_tot)} ({tot_p:.0f}%)  "
                  f"ML {wk_win['W']}-{wk_win['L']} ({win_p:.0f}%)")
            for k in ("W","L","P"):
                s_ats[k] += wk_ats[k]
                s_tot[k] += wk_tot[k]
            s_win["W"] += wk_win["W"]
            s_win["L"] += wk_win["L"]
        print()

    # Season summary
    na = s_ats["W"] + s_ats["L"]
    nt = s_tot["W"] + s_tot["L"]
    nw = s_win["W"] + s_win["L"]
    push_s = lambda d: f"{d['W']}-{d['L']}" + (f"-{d['P']}" if d["P"] else "")
    print(f"\n{'='*60}")
    print(f"  {season} SEASON FINAL")
    print(f"  ATS : {push_s(s_ats)}  ({s_ats['W']/na*100:.1f}%)" if na else "")
    print(f"  TOT : {push_s(s_tot)}  ({s_tot['W']/nt*100:.1f}%)" if nt else "")
    print(f"  ML  : {s_win['W']}-{s_win['L']}  ({s_win['W']/nw*100:.1f}%)" if nw else "")
    apl = s_ats["W"]*(100*100/110) - s_ats["L"]*100
    tpl = s_tot["W"]*(100*100/110) - s_tot["L"]*100
    print(f"  ATS P&L @-110 $100/game: ${apl:+,.0f}")
    print(f"  TOT P&L @-110 $100/game: ${tpl:+,.0f}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
