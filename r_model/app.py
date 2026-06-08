"""
app.py  —  NFL R-Model Streamlit Dashboard (port 8502)

Run locally:
    streamlit run r_model/app.py --server.port 8502

Mirrors the structure of epa_model/app.py but uses the R-model
two-projector ensemble (YPC/YPA + Eckel) with Bayesian Vegas shrinkage.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Setup ──────────────────────────────────────────────────────────────────────
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

st.set_page_config(
    page_title="NFL R-Model",
    page_icon="🏟️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Module loader ──────────────────────────────────────────────────────────────
@st.cache_resource
def _load_modules():
    def _load(name, fpath):
        spec = importlib.util.spec_from_file_location(name, fpath)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    rm = _load("r_metrics",   HERE / "r_metrics.py")
    rs = _load("r_simulator", HERE / "r_simulator.py")
    return rm, rs

rm, rs = _load_modules()

# ── Constants ──────────────────────────────────────────────────────────────────
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

def _norm(t):
    return ALIASES.get(t, t) if isinstance(t, str) else t

def _safe_float(x, default=None):
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default


# ── Data loaders ───────────────────────────────────────────────────────────────
load_pbp      = rm.load_pbp
load_schedule = rm.load_schedule

@st.cache_data(show_spinner="Computing R-model metrics…")
def build_metrics(season: int, through_week: int):
    """Returns (team_metrics, league_avgs, rankings)."""
    return rm.compute_r_metrics(season, through_week)


@st.cache_data(show_spinner="Computing power rankings…")
def compute_power_rankings(season: int, through_week: int, sims: int) -> pd.DataFrame:
    team_metrics, lg, rankings = build_metrics(season, through_week)
    if not team_metrics:
        return pd.DataFrame()

    # Build an "average team" for SOS-neutral power ranking
    all_vals = {k: [tm[k] for tm in team_metrics.values() if isinstance(tm.get(k), (int, float))]
                for k in next(iter(team_metrics.values())).keys()}
    avg_team = {k: float(np.mean(v)) if v else 0.0 for k, v in all_vals.items()}
    avg_team["team"] = "AVG"
    avg_team["n_games"] = 1

    rng = np.random.default_rng(42)
    rows = []
    for team, tm in team_metrics.items():
        r_home = rs.simulate_r_game(tm, avg_team, sims=sims, neutral=True, rng=rng)
        r_away = rs.simulate_r_game(avg_team, tm, sims=sims, neutral=True, rng=rng)
        power    = (r_home["spread_mean"] - r_away["spread_mean"]) / 2
        win_pct  = (r_home["home_win_prob"] + r_away["away_win_prob"]) / 2 * 100
        rows.append({
            "team":      team,
            "conf":      CONF_MAP.get(team, ""),
            "power":     power,
            "win_pct":   win_pct,
            "pts_for":   r_home["home_mean"],
            "pts_agnst": r_away["home_mean"],
            "off_ypc":   tm["off_ypc"],
            "off_ypa":   tm["off_ypa"],
            "off_eckel": tm["off_eckel_rate"],
            "off_ppe":   tm["off_ppe"],
            "def_ypc":   tm["def_ypc_raw"],
            "def_ypa":   tm["def_ypa_raw"],
            "n_games":   tm["n_games"],
        })
    df = pd.DataFrame(rows).sort_values("power", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    return df


@st.cache_data(show_spinner="Simulating weekly games…")
def compute_weekly_predictions(season: int, week: int, sims: int) -> list:
    through_week = week - 1 if week > 1 else 0
    team_metrics, lg, rankings = build_metrics(season, through_week)
    sched = load_schedule(season)
    if sched.empty or not team_metrics:
        return []
    reg   = sched[sched["game_type"] == "REG"]
    games = reg[reg["week"] == week].copy()
    rng   = np.random.default_rng(42)
    results = []
    for _, row in games.sort_values("gameday").iterrows():
        ht, at = _norm(row["home_team"]), _norm(row["away_team"])
        hm, am = team_metrics.get(ht), team_metrics.get(at)
        if hm is None or am is None:
            continue
        vs = _safe_float(row.get("spread_line"))
        vt = _safe_float(row.get("total_line"))
        res = rs.simulate_r_game(hm, am, vegas_spread=vs, vegas_total=vt,
                                 sims=sims, rng=rng)
        ah  = _safe_float(row.get("home_score"))
        aa  = _safe_float(row.get("away_score"))
        results.append({
            **res,
            "gameday":     str(row.get("gameday", ""))[:10],
            "actual_home": ah,
            "actual_away": aa,
            "scored":      ah is not None and aa is not None,
        })
    return results


# ── ATS grader ─────────────────────────────────────────────────────────────────
def _ats_grade(actual_margin, vegas_spread, model_pick, home_team, away_team):
    """
    nflfastR: positive spread_line = home favored.
    Home covers if actual_margin > spread_line (wins by more).
    """
    if vegas_spread is None:
        return "—", None
    if actual_margin > vegas_spread + 0.5:
        winner = home_team
    elif actual_margin < vegas_spread - 0.5:
        winner = away_team
    else:
        return "⬛", None
    correct = model_pick == winner
    return "✅" if correct else "❌", correct


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🏟️ NFL R-Model")
    st.caption("YPC · YPA · Eckel rate · Bayesian shrink")
    season = st.selectbox("Season", SEASONS[::-1], index=0)
    week   = st.selectbox(
        "Week",
        ["Full Season"] + [f"Week {w}" for w in range(1, 23)],
        index=0,
    )
    sims   = st.select_slider("Simulations", [1000, 5000, 10000, 20000], value=10000)
    through_week = 0 if week == "Full Season" else int(str(week).split()[-1])
    sel_week     = through_week  # for weekly picks tab

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_pr, tab_wk, tab_rec, tab_mu = st.tabs([
    "🏈 Power Rankings", "📅 Weekly Picks", "📊 Season Record", "🏟️ Matchup"
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 – POWER RANKINGS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_pr:
    st.subheader(f"Power Rankings — {season} {'Full Season' if through_week == 0 else f'Through Week {through_week}'}")
    with st.spinner("Computing power rankings…"):
        pr_df = compute_power_rankings(season, through_week, sims)

    if pr_df.empty:
        st.warning("No data available for this season/week combination.")
    else:
        conf_filter = st.radio("Conference", ["All", "AFC", "NFC"], horizontal=True)
        if conf_filter != "All":
            pr_df = pr_df[pr_df["conf"] == conf_filter]

        def color_conf(v):
            c = AFC_COLOR if v == "AFC" else NFC_COLOR if v == "NFC" else "#666"
            return f"color: {c}; font-weight: 600"

        display = pr_df.rename(columns={
            "rank": "#", "team": "Team", "conf": "Conf",
            "power": "Power", "win_pct": "Win%",
            "pts_for": "Proj PF", "pts_agnst": "Proj PA",
            "off_ypc": "Off YPC", "off_ypa": "Off YPA",
            "off_eckel": "Eckel Rate", "off_ppe": "PPE",
            "def_ypc": "Def YPC", "def_ypa": "Def YPA",
            "n_games": "Gms",
        })
        styled = (
            display.style
            .format({
                "Power":     "{:+.1f}",
                "Win%":      "{:.1f}%",
                "Proj PF":   "{:.1f}",
                "Proj PA":   "{:.1f}",
                "Off YPC":   "{:.2f}",
                "Off YPA":   "{:.2f}",
                "Eckel Rate":"{:.1%}",
                "PPE":       "{:.2f}",
                "Def YPC":   "{:.2f}",
                "Def YPA":   "{:.2f}",
            })
            .map(color_conf, subset=["Conf"])
        )
        st.dataframe(styled, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 – WEEKLY PICKS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_wk:
    pick_week = st.number_input("Pick Week", min_value=1, max_value=22,
                                value=max(sel_week, 1), key="pick_week")
    with st.spinner(f"Simulating Week {pick_week}…"):
        preds = compute_weekly_predictions(season, pick_week, sims)

    if not preds:
        st.info("No games found for this week / season combination.")
    else:
        rows_wk = []
        ats_w = ats_l = ats_p = 0
        tot_w = tot_l = tot_p = 0

        for r in preds:
            ht, at = r["home_team"], r["away_team"]
            sp = r.get("vegas_spread")
            vt = r.get("vegas_total")
            sp_str = f"{sp:+.1f}" if sp is not None else "—"
            vt_str = f"{vt:.1f}" if vt is not None else "—"
            hm = r["home_mean"]; am = r["away_mean"]
            hwp = r["home_win_prob"]; awp = r["away_win_prob"]
            hw = r.get("home_american"); aw = r.get("away_american")
            hw_str = f"{hw:+d}" if hw is not None else "—"
            aw_str = f"{aw:+d}" if aw is not None else "—"
            spread_str = f"{r['spread_mean']:+.1f}"
            total_str  = f"{r['total_mean']:.1f}"
            ats_pick   = r.get("ats_pick") or "—"
            ats_pct    = f"{r['ats_prob']*100:.1f}%" if r.get("ats_prob") else "—"
            tot_pick   = r.get("total_pick") or "—"
            tot_pct    = f"{r['total_prob']*100:.1f}%" if r.get("total_prob") else "—"

            if r.get("scored"):
                am_h = r["actual_home"]; am_a = r["actual_away"]
                act_str  = f"{am_h:.0f}–{am_a:.0f}"
                margin   = am_h - am_a
                ats_icon, ats_ok = _ats_grade(margin, sp, ats_pick, ht, at)
                # Totals
                if vt is not None and am_h is not None and am_a is not None:
                    tot_actual = am_h + am_a
                    if tot_actual > vt + 0.5:
                        actual_tot = "Over"
                    elif tot_actual < vt - 0.5:
                        actual_tot = "Under"
                    else:
                        actual_tot = "Push"
                    if actual_tot == "Push":
                        tot_icon = "⬛"; tot_ok = None
                    else:
                        tot_ok   = tot_pick == actual_tot
                        tot_icon = "✅" if tot_ok else "❌"
                else:
                    tot_icon = "—"; tot_ok = None

                if ats_ok is True:  ats_w += 1
                elif ats_ok is False: ats_l += 1
                else:               ats_p += 1
                if tot_ok is True:  tot_w += 1
                elif tot_ok is False: tot_l += 1
                else:               tot_p += 1
            else:
                act_str = ats_icon = tot_icon = "—"

            rows_wk.append({
                "Date":    r.get("gameday", "")[:10],
                "Matchup": f"{at} @ {ht}",
                "Spread":  sp_str,
                "Total":   vt_str,
                "Home%":   f"{hwp*100:.0f}%",
                "Away%":   f"{awp*100:.0f}%",
                "Odds":    f"{aw_str} / {hw_str}",
                "ProjScr": f"{at} {am:.1f}  –  {ht} {hm:.1f}",
                "ProjSpd": spread_str,
                "ProjTot": total_str,
                "ATS Pick":  f"{ats_pick} ({ats_pct})",
                "ATS ✓":     ats_icon,
                "Tot Pick":  f"{tot_pick} ({tot_pct})",
                "Tot ✓":     tot_icon,
                "Actual":    act_str,
            })

        st.dataframe(pd.DataFrame(rows_wk), use_container_width=True, hide_index=True)

        # Summary
        total_graded = ats_w + ats_l + ats_p
        if total_graded > 0:
            c1, c2 = st.columns(2)
            c1.metric("ATS Record", f"{ats_w}–{ats_l}{'–'+str(ats_p) if ats_p else ''}",
                      f"{ats_w/max(ats_w+ats_l,1)*100:.1f}% win rate")
            c2.metric("Totals Record", f"{tot_w}–{tot_l}{'–'+str(tot_p) if tot_p else ''}",
                      f"{tot_w/max(tot_w+tot_l,1)*100:.1f}% win rate")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 – SEASON RECORD
# ═══════════════════════════════════════════════════════════════════════════════
with tab_rec:
    st.subheader(f"Season ATS & Totals Record — {season}")
    with st.spinner("Loading all week results…"):
        sched = load_schedule(season)

    if sched.empty:
        st.warning("No schedule available.")
    else:
        max_wk = int(sched[sched["game_type"] == "REG"]["week"].max()) if not sched.empty else 18
        weekly_rows = []
        cum_ats_w = cum_ats_l = 0
        cum_tot_w = cum_tot_l = 0

        prog = st.progress(0, text="Simulating season…")
        for wk in range(1, max_wk + 1):
            prog.progress(wk / max_wk, text=f"Week {wk}…")
            through_wk = wk - 1 if wk > 1 else 0
            team_metrics, lg, _ = build_metrics(season, through_wk)
            if not team_metrics:
                continue
            reg   = sched[sched["game_type"] == "REG"]
            games = reg[reg["week"] == wk].copy()
            rng   = np.random.default_rng(42)
            wk_ats_w = wk_ats_l = wk_ats_p = 0
            wk_tot_w = wk_tot_l = wk_tot_p = 0

            for _, row in games.iterrows():
                ht, at = _norm(row["home_team"]), _norm(row["away_team"])
                hm, am = team_metrics.get(ht), team_metrics.get(at)
                ah = _safe_float(row.get("home_score"))
                aa = _safe_float(row.get("away_score"))
                if hm is None or am is None or ah is None or aa is None:
                    continue
                vs = _safe_float(row.get("spread_line"))
                vt = _safe_float(row.get("total_line"))
                res = rs.simulate_r_game(hm, am, vegas_spread=vs, vegas_total=vt,
                                         sims=sims, rng=rng)
                actual_margin = ah - aa

                # ATS
                if vs is not None:
                    # positive spread_line = home favored
                    if actual_margin > vs + 0.5:
                        cover = ht
                    elif actual_margin < vs - 0.5:
                        cover = at
                    else:
                        cover = None
                    ap = res.get("ats_pick")
                    if cover is None:
                        wk_ats_p += 1
                    elif cover == ap:
                        wk_ats_w += 1
                    else:
                        wk_ats_l += 1

                # Totals
                if vt is not None:
                    actual_tot = ah + aa
                    if actual_tot > vt + 0.5:
                        actual_pick = "Over"
                    elif actual_tot < vt - 0.5:
                        actual_pick = "Under"
                    else:
                        actual_pick = None
                    tp = res.get("total_pick")
                    if actual_pick is None:
                        wk_tot_p += 1
                    elif actual_pick == tp:
                        wk_tot_w += 1
                    else:
                        wk_tot_l += 1

            cum_ats_w += wk_ats_w; cum_ats_l += wk_ats_l
            cum_tot_w += wk_tot_w; cum_tot_l += wk_tot_l

            wk_d = wk_ats_w + wk_ats_l
            td_d = wk_tot_w + wk_tot_l
            weekly_rows.append({
                "Week":     wk,
                "ATS W-L":  f"{wk_ats_w}–{wk_ats_l}{'–'+str(wk_ats_p) if wk_ats_p else ''}",
                "ATS%":     f"{wk_ats_w/max(wk_d,1)*100:.0f}%" if wk_d else "—",
                "Tot W-L":  f"{wk_tot_w}–{wk_tot_l}{'–'+str(wk_tot_p) if wk_tot_p else ''}",
                "Tot%":     f"{wk_tot_w/max(td_d,1)*100:.0f}%" if td_d else "—",
                "Cum ATS":  f"{cum_ats_w}–{cum_ats_l}",
                "Cum Tot":  f"{cum_tot_w}–{cum_tot_l}",
            })

        prog.empty()
        if weekly_rows:
            st.dataframe(pd.DataFrame(weekly_rows), use_container_width=True, hide_index=True)
            tot_d = cum_ats_w + cum_ats_l
            totd2 = cum_tot_w + cum_tot_l
            c1, c2 = st.columns(2)
            c1.metric("Season ATS", f"{cum_ats_w}–{cum_ats_l}",
                      f"{cum_ats_w/max(tot_d,1)*100:.1f}%")
            c2.metric("Season Totals", f"{cum_tot_w}–{cum_tot_l}",
                      f"{cum_tot_w/max(totd2,1)*100:.1f}%")

            # Cumulative ATS win chart
            fig = go.Figure()
            rows_df = pd.DataFrame(weekly_rows)
            cum_ats = [int(r.split("–")[0]) for r in rows_df["Cum ATS"]]
            cum_tot_wins = [int(r.split("–")[0]) for r in rows_df["Cum Tot"]]
            fig.add_trace(go.Scatter(x=rows_df["Week"], y=cum_ats, name="ATS Wins", mode="lines+markers"))
            fig.add_trace(go.Scatter(x=rows_df["Week"], y=cum_tot_wins, name="Total Wins", mode="lines+markers"))
            fig.update_layout(title="Cumulative Season W (ATS & Totals)", xaxis_title="Week",
                              yaxis_title="Cumulative Wins", height=350)
            st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 – MATCHUP PREVIEW
# ═══════════════════════════════════════════════════════════════════════════════
with tab_mu:
    st.subheader("🏟️ Matchup Preview")

    # ── Team selectors ─────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns([4, 4, 2, 1])
    with c1:
        away_team = st.selectbox("Away Team", NFL_TEAMS, index=NFL_TEAMS.index("KC"), key="mu_away")
    with c2:
        home_team = st.selectbox("Home Team", NFL_TEAMS, index=NFL_TEAMS.index("BUF"), key="mu_home")
    with c3:
        neutral = st.checkbox("Neutral Site", value=False, key="mu_neutral")
        vegas_sp = st.number_input("Vegas Spread (home)", value=0.0, step=0.5, key="mu_vs",
                                   help="Enter as home perspective: +3 = home favored by 3")
        vegas_tot = st.number_input("Vegas Total", value=48.0, step=0.5, key="mu_vt")
    with c4:
        st.write("")
        st.write("")
        if st.button("✕ Clear", key="mu_clear"):
            st.rerun()

    if not away_team or not home_team or away_team == home_team:
        st.info("Select two different teams to see the matchup preview.")
        st.stop()

    # ── Load metrics ───────────────────────────────────────────────────────────
    mu_through = through_week  # use sidebar week
    team_metrics, lg, rankings = build_metrics(season, mu_through)

    hm = team_metrics.get(home_team)
    am = team_metrics.get(away_team)
    if hm is None or am is None:
        st.warning(f"No data for {home_team} or {away_team} at this week/season.")
        st.stop()

    # ── Run simulation ─────────────────────────────────────────────────────────
    vs = vegas_sp if vegas_sp != 0.0 else None
    vt = vegas_tot if vegas_tot > 0 else None
    sim = rs.simulate_r_game(hm, am,
                              vegas_spread=vs,
                              vegas_total=vt,
                              sims=sims,
                              neutral=neutral,
                              rng=np.random.default_rng(99))

    # ── HTML helpers ───────────────────────────────────────────────────────────
    def rank_badge(rank: int) -> str:
        if rank <= 10:   color = "#1565c0"
        elif rank <= 22: color = "#444444"
        else:            color = "#c62828"
        return (
            f'<span style="background:{color};color:white;'
            f'padding:1px 7px;border-radius:3px;'
            f'font-weight:600;font-size:0.8rem;">{rank}</span>'
        )

    def fmt_pct(v):  return f"{v:.1%}"
    def fmt_pts(v):  return f"{v:.2f}"
    def fmt_yds(v):  return f"{v:.2f}"
    def fmt_wp(p):
        odds = int(-round(p/(1-p)*100)) if p > 0.5 else int(round((1-p)/p*100))
        sign = "+" if odds > 0 else ""
        return f"{p*100:.1f}% ({sign}{odds})"

    # ── Side panel renderer ────────────────────────────────────────────────────
    SIDE_METRICS = [
        ("Off YPC",       "off_ypc",       fmt_yds, False),
        ("Def YPC Allow", "def_ypc_raw",   fmt_yds, True),
        ("Off YPA",       "off_ypa",       fmt_yds, False),
        ("Def YPA Allow", "def_ypa_raw",   fmt_yds, True),
        ("Off Eckel Rate","off_eckel_rate",fmt_pct, False),
        ("Def Eckel Rate","def_eckel_rate",fmt_pct, True),
        ("Pts/Eckel Drv", "off_ppe",       fmt_pts, False),
        ("TO Rate Off",   "off_to_rate",   fmt_pct, True),
        ("TO Forced Def", "def_to_rate",   fmt_pct, False),
        ("Off Yds/Play",  "off_ypp_play",  fmt_yds, False),
        ("Def Yds Allow", "def_ypp_play",  fmt_yds, True),
    ]

    def render_team_panel(team: str, metrics: dict, rnk: dict, win_prob: float,
                          proj_pts: float, side: str):
        rks = rnk.get(team, {})
        align = "left" if side == "right" else "right"
        st.markdown(
            f'<div style="text-align:{align}">'
            f'<span style="font-size:2rem;font-weight:700">{team}</span>'
            f'</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            f'<div style="text-align:{align}">'
            f'<span style="font-size:1.1rem;color:#888">WIN PROB</span><br>'
            f'<span style="font-size:1.8rem;font-weight:700">{win_prob*100:.1f}%</span>'
            f'</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            f'<div style="text-align:{align}">'
            f'<span style="font-size:1.1rem;color:#888">PROJ PTS</span><br>'
            f'<span style="font-size:1.4rem;font-weight:600">{proj_pts:.1f}</span>'
            f'</div>',
            unsafe_allow_html=True
        )
        st.markdown("---")

        html_rows = []
        for label, key, fmt, lower_better in SIDE_METRICS:
            val = metrics.get(key, 0.0)
            rk  = rks.get(key, "–")
            badge = rank_badge(rk) if isinstance(rk, int) else f"<b>–</b>"
            val_str = fmt(val)
            if side == "right":
                html_rows.append(
                    f'<tr><td style="text-align:left;padding:2px 4px">{badge}</td>'
                    f'<td style="text-align:left;color:#aaa;font-size:0.85rem">{label}</td>'
                    f'<td style="text-align:left;font-weight:600">{val_str}</td></tr>'
                )
            else:
                html_rows.append(
                    f'<tr><td style="text-align:right;font-weight:600">{val_str}</td>'
                    f'<td style="text-align:right;color:#aaa;font-size:0.85rem">{label}</td>'
                    f'<td style="text-align:right;padding:2px 4px">{badge}</td></tr>'
                )

        tbl_align = "right" if side == "left" else "left"
        st.markdown(
            f'<table style="width:100%;border-collapse:collapse;text-align:{tbl_align}">'
            + "".join(html_rows)
            + "</table>",
            unsafe_allow_html=True
        )

    # ── Center matchup table renderer ─────────────────────────────────────────
    CENTER_METRICS = [
        ("Adj Net YPA",    "off_ypa",       "def_ypa_raw",      fmt_yds, False, True),
        ("YPC",            "off_ypc",       "def_ypc_raw",      fmt_yds, False, True),
        ("Eckel Rate",     "off_eckel_rate","def_eckel_rate",   fmt_pct, False, True),
        ("Pts/Eckel",      "off_ppe",       "def_ppe",          fmt_pts, False, True),
        ("Turnover Rate",  "off_to_rate",   "def_to_rate",      fmt_pct, True,  False),
        ("Yards/Play",     "off_ypp_play",  "def_ypp_play",     fmt_yds, False, True),
        ("Avg Field Pos",  "off_fp",        "def_fp",           lambda v: f"{v:.0f}",
                                                                          True,  True),
    ]
    # center_metrics entry: (label, away_off_key, home_def_key, fmt, away_lower_better, home_lower_better)

    def matchup_table_html(away_m, home_m, metrics_cfg, away_rnk, home_rnk,
                           title: str, header_color: str) -> str:
        rows = []
        for label, a_off_key, h_def_key, fmt, a_lo, h_lo in metrics_cfg:
            av = away_m.get(a_off_key, 0.0)
            hv = home_m.get(h_def_key, 0.0)
            # ranks: for off metrics use off rank; for def metrics use def rank
            ar = away_rnk.get(a_off_key, 16)
            hr = home_rnk.get(h_def_key, 16)
            rows.append(
                f'<tr>'
                f'<td style="text-align:right;padding:3px 6px;font-weight:600">{fmt(av)}</td>'
                f'<td style="padding:2px 4px">{rank_badge(ar) if isinstance(ar,int) else "–"}</td>'
                f'<td style="text-align:center;padding:3px 8px;color:#aaa;font-size:0.85rem;white-space:nowrap">{label}</td>'
                f'<td style="padding:2px 4px">{rank_badge(hr) if isinstance(hr,int) else "–"}</td>'
                f'<td style="text-align:left;padding:3px 6px;font-weight:600">{fmt(hv)}</td>'
                f'</tr>'
            )

        header = (
            f'<tr style="background:{header_color};color:white">'
            f'<th colspan="2" style="text-align:right;padding:6px 8px">{away_team} OFF</th>'
            f'<th style="text-align:center;padding:6px 8px">{title}</th>'
            f'<th colspan="2" style="text-align:left;padding:6px 8px">{home_team} DEF</th>'
            f'</tr>'
        )
        return (
            '<table style="width:100%;border-collapse:collapse;margin-bottom:12px">'
            + header + "".join(rows) + "</table>"
        )

    # ── 3-column layout ────────────────────────────────────────────────────────
    col_a, col_c, col_h = st.columns([3, 4, 3])

    away_rnk = rankings.get(away_team, {})
    home_rnk = rankings.get(home_team, {})

    with col_a:
        render_team_panel(away_team, am, rankings, sim["away_win_prob"], sim["away_mean"], "left")

    with col_c:
        # Win prob gauge
        hwp = sim["home_win_prob"] * 100
        awp = sim["away_win_prob"] * 100
        fig_wp = go.Figure(go.Bar(
            x=[awp, hwp],
            y=["", ""],
            orientation="h",
            text=[f"{away_team} {awp:.0f}%", f"{home_team} {hwp:.0f}%"],
            textposition="inside",
            marker_color=["#1565c0", "#c62828"],
        ))
        fig_wp.update_layout(
            barmode="stack", showlegend=False,
            height=55, margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=False, showticklabels=False, range=[0, 100]),
            yaxis=dict(showticklabels=False),
        )
        st.plotly_chart(fig_wp, use_container_width=True, config={"displayModeBar": False})

        # Center matchup tables
        # Table 1: Away OFF vs Home DEF
        tbl1 = matchup_table_html(am, hm, CENTER_METRICS, away_rnk, home_rnk,
                                   "AWAY OFF vs HOME DEF", "#2e7d32")
        # Table 2: Home OFF vs Away DEF (flip teams)
        center_flip = [
            (label, h_def_key, a_off_key, fmt, h_lo, a_lo)
            for label, a_off_key, h_def_key, fmt, a_lo, h_lo in CENTER_METRICS
        ]
        tbl2 = matchup_table_html(hm, am, center_flip, home_rnk, away_rnk,
                                   "HOME OFF vs AWAY DEF", "#1a237e")
        # Swap column headers for table 2
        tbl2 = tbl2.replace(
            f"{away_team} OFF", f"{home_team} OFF"
        ).replace(
            f"{home_team} DEF", f"{away_team} DEF"
        )

        st.markdown(tbl1, unsafe_allow_html=True)
        st.markdown(tbl2, unsafe_allow_html=True)

        # ATS / Totals block
        if vs is not None:
            ats_pick = sim.get("ats_pick", "—")
            ats_pct  = sim.get("ats_prob", 0.0)
            hcp = sim.get("home_cover_prob", 0.5) or 0.5
            acp = sim.get("away_cover_prob", 0.5) or 0.5
            st.markdown(
                f"<div style='text-align:center;margin-top:8px'>"
                f"<b>ATS Pick:</b> <span style='font-size:1.1rem'>{ats_pick}</span> "
                f"({ats_pct*100:.1f}%) &nbsp;|&nbsp; "
                f"{home_team}: {hcp*100:.1f}%  {away_team}: {acp*100:.1f}%"
                f"</div>",
                unsafe_allow_html=True,
            )
        if vt is not None:
            tp = sim.get("total_pick", "—")
            tpp = sim.get("total_prob", 0.0)
            op = sim.get("over_prob", 0.5) or 0.5
            up = sim.get("under_prob", 0.5) or 0.5
            st.markdown(
                f"<div style='text-align:center'>"
                f"<b>Total Pick:</b> {tp} ({tpp*100:.1f}%) &nbsp;|&nbsp; "
                f"Over: {op*100:.1f}%  Under: {up*100:.1f}%"
                f"</div>",
                unsafe_allow_html=True,
            )

    with col_h:
        render_team_panel(home_team, hm, rankings, sim["home_win_prob"], sim["home_mean"], "right")

    # ── Bottom summary bar ─────────────────────────────────────────────────────
    fav = home_team if sim["spread_mean"] > 0 else away_team
    margin_abs = abs(sim["spread_mean"])
    site_note = " (Neutral Site)" if neutral else ""

    st.markdown("---")
    st.markdown(
        f"<div style='text-align:center;font-size:1.1rem'>"
        f"<b>PROJECTED TOTAL: {sim['total_mean']:.1f}</b>"
        f" &nbsp;·&nbsp; "
        f"<b>PROJ MARGIN: {fav} BY {margin_abs:.1f}</b>"
        f"{site_note}"
        f"</div>",
        unsafe_allow_html=True,
    )

    # Spread distribution chart
    st.markdown("")
    spread_mean = sim["spread_mean"]
    spread_sd   = sim["spread_sd"]
    x = np.linspace(spread_mean - 4*spread_sd, spread_mean + 4*spread_sd, 300)
    from scipy.stats import norm as sp_norm
    y = sp_norm.pdf(x, loc=spread_mean, scale=spread_sd)
    fig_sp = go.Figure()
    fig_sp.add_trace(go.Scatter(x=x, y=y, fill="tozeroy", name="Spread distribution",
                                line_color="#1565c0"))
    if vs is not None:
        fig_sp.add_vline(x=vs, line_dash="dash", line_color="orange",
                         annotation_text=f"Vegas {vs:+.1f}", annotation_position="top right")
    fig_sp.add_vline(x=0, line_dash="dot", line_color="#888")
    fig_sp.update_layout(
        title=f"Spread Distribution — {away_team} @ {home_team}",
        xaxis_title=f"← {away_team} wins  |  {home_team} wins →",
        showlegend=False, height=280,
        margin=dict(l=40, r=40, t=40, b=40),
    )
    st.plotly_chart(fig_sp, use_container_width=True)
