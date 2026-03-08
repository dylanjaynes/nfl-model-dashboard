#' @import magrittr
#' @import dplyr
NULL


# --- priors you liked from bins_cmp ---
PRI_SIG <- list(tau_spread=15, sigma_spread=7.5, df_spread=5, sigma_total=3, df_total=6)
PRI_TAU <- list(tau_spread=16, sigma_spread=7.0, df_spread=5, sigma_total=3, df_total=6)
PRI_DF  <- list(tau_spread=15, sigma_spread=7.0, df_spread=4, sigma_total=3, df_total=6)
PRI_ALL <- list(tau_spread=16, sigma_spread=7.5, df_spread=4, sigma_total=3, df_total=6)

# --- tiny Kelly helper (half Kelly by default) ---
.kelly_risk_frac <- function(p, american_odds = -110, kelly_fraction = 0.5) {
  b <- if (american_odds > 0) american_odds/100 else 100/abs(american_odds)  # net odds per $1
  frac <- (b*p - (1 - p)) / b                      # Kelly fraction of bankroll
  pmax(0, kelly_fraction * frac)
}

# --- main wrapper ---
pick_bets_by_bucket <- function(season, week,
                                targets,
                                sims = 2000,
                                w_eckel = 0.3,
                                injuries = NULL,
                                vegas_lines = NULL,   # optional tibble (home_team, away_team, week, spread_line, total_line)
                                thresholds = list(
                                  tot_high = 0.70,
                                  tot_low  = c(0.55, 0.60),
                                  ats_high = 0.70,
                                  ats_mid  = c(0.65, 0.70)
                                ),
                                priors_map = list(sig=PRI_SIG, tau=PRI_TAU, df=PRI_DF, all=PRI_ALL),
                                routing = list(       # which prior to use for each band
                                  tot_high = "sig",   # best Totals 70%+ bucket
                                  tot_low  = "tau",   # best Totals 55–60% bucket
                                  ats_high = "df",    # best ATS 70%+ bucket
                                  ats_mid  = "all"    # best ATS 65–70% bucket
                                ),
                                bankroll = NULL,          # e.g., 5000 to get stake sizes
                                kelly_fraction = 0.5,
                                max_risk_by_type = c(ATS = 0.02, Total = 0.03),   # per-bet caps
                                outfile = NULL) {

  # null-coalesce helper (local)
  `%||%` <- function(x, y) if (is.null(x)) y else x

  need <- unique(unlist(routing, use.names = FALSE))
  runs <- setNames(vector("list", length(need)), need)

  # run each needed prior once
  for (tag in need) {
    pr <- priors_map[[tag]]
    runs[[tag]] <- run_week(
      season_target = season, week = week,
      sims = sims, w_eckel = w_eckel, targets = targets,
      tau_spread = pr$tau_spread, sigma_spread = pr$sigma_spread, df_spread = pr$df_spread,
      sigma_total = pr$sigma_total, df_total = pr$df_total,
      injuries = injuries, vegas_lines = vegas_lines, outfile = NULL
    ) |>
      dplyr::mutate(priors = tag)
  }

  # helper: slice games for a bucket from the run used for that bucket
  pick_block <- function(type, rng, tag_used) {
    d <- runs[[tag_used]]
    if (type == "Total") {
      if (length(rng) == 1) d <- d |> dplyr::filter(Total_prob >= rng)
      else                  d <- d |> dplyr::filter(Total_prob >= rng[1], Total_prob < rng[2])
      d |>
        dplyr::mutate(bet_type = "Total", pick = Total_pick, prob = Total_prob, line = vegas_total_line)
    } else {
      if (length(rng) == 1) d <- d |> dplyr::filter(ATS_prob >= rng)
      else                  d <- d |> dplyr::filter(ATS_prob >= rng[1], ATS_prob < rng[2])
      d |>
        dplyr::mutate(bet_type = "ATS", pick = ATS_pick, prob = ATS_prob, line = vegas_spread_line)
    }
  }

  # ---- bet card (only qualifying bets) ----
  card <- dplyr::bind_rows(
    pick_block("Total", thresholds$tot_high, routing$tot_high),
    pick_block("Total", thresholds$tot_low,  routing$tot_low),
    pick_block("ATS",   thresholds$ats_high, routing$ats_high),
    pick_block("ATS",   thresholds$ats_mid,  routing$ats_mid)
  ) |>
    dplyr::select(
      game_id, season_target, week, gameday, gametime,
      home_team, away_team,
      bet_type, pick, prob, line,
      ATS_pick, ATS_prob, Total_pick, Total_prob,
      Win_pick, Win_prob, Win_prob_american, Win_edge,
      home_win_prob, away_win_prob,
      home_american_odds, away_american_odds,
      vegas_home_win_prob, vegas_away_win_prob,
      vegas_home_odds, vegas_away_odds,
      vegas_win_pick, vegas_win_prob, vegas_win_odds,
      prob_edge, odds_edge, priors
    )


  # optional Kelly stake suggestion
  if (!is.null(bankroll)) {
    card <- card |>
      dplyr::rowwise() |>
      dplyr::mutate(
        risk_frac_raw = .kelly_risk_frac(prob, american_odds = -110, kelly_fraction = kelly_fraction),
        risk_frac_cap = pmin(risk_frac_raw, max_risk_by_type[[bet_type]] %||% 0.03),
        stake_dollars = round(bankroll * risk_frac_cap, 2)
      ) |>
      dplyr::ungroup()
  }

  # ---- all_games table (every matchup + flags; uses your routing) ----
  # base schedule from any run (they all have the same rows)
  base_key <- names(runs)[1]
  base <- runs[[base_key]] |>
    dplyr::select(
      game_id, season_target, week, gameday, gametime,
      home_team, away_team,
      vegas_spread_line, vegas_total_line,
      spread_mean, total_mean,
      ATS_pick, ATS_prob, Total_pick, Total_prob,
      Win_pick, Win_prob, Win_prob_american, Win_edge,
      home_win_prob, away_win_prob,
      home_american_odds, away_american_odds,
      vegas_home_win_prob, vegas_away_win_prob,
      vegas_home_odds, vegas_away_odds,
      vegas_win_pick, vegas_win_prob, vegas_win_odds,
      prob_edge, odds_edge
    )


  # pick the specific runs used by each bucket
  th_tot_hi <- runs[[ routing$tot_high ]] |>
    dplyr::select(game_id, Total_pick_high = Total_pick, Total_prob_high = Total_prob)
  th_tot_lo <- runs[[ routing$tot_low  ]] |>
    dplyr::select(game_id, Total_prob_low = Total_prob)
  th_ats_hi <- runs[[ routing$ats_high ]] |>
    dplyr::select(game_id, ATS_pick_high = ATS_pick, ATS_prob_high = ATS_prob)
  th_ats_mid<- runs[[ routing$ats_mid  ]] |>
    dplyr::select(game_id, ATS_prob_mid = ATS_prob)

  all_games <- base |>
    dplyr::left_join(th_tot_hi, by = "game_id") |>
    dplyr::left_join(th_tot_lo, by = "game_id") |>
    dplyr::left_join(th_ats_hi, by = "game_id") |>
    dplyr::left_join(th_ats_mid, by = "game_id") |>
    dplyr::mutate(
      recommend_total =
        (Total_prob_high >= thresholds$tot_high) |
        (Total_prob_low  >= thresholds$tot_low[1] & Total_prob_low < thresholds$tot_low[2]),
      recommend_ats =
        (ATS_prob_high   >= thresholds$ats_high)  |
        (ATS_prob_mid    >= thresholds$ats_mid[1] & ATS_prob_mid < thresholds$ats_mid[2]),
      recommended = recommend_total | recommend_ats
    ) |>
    # merge stake suggestions from card (0 if not recommended)
    dplyr::left_join(card |> dplyr::select(game_id, bet_type, pick, prob, priors, stake_dollars),
                     by = "game_id") |>
    dplyr::mutate(stake_dollars = dplyr::coalesce(stake_dollars, 0)) |>
    dplyr::arrange(gameday, gametime)

  # optional write-out
  if (!is.null(outfile)) {
    writexl::write_xlsx(list(card = card, all_games = all_games), outfile)
    message("Saved bet card & all-games to: ", outfile)
  }

  # return both objects (what run_weekly_report expects)
  return(list(card = card, all_games = all_games))
}


run_and_save_week <- function(season, week,
                              targets,
                              sims = 2000,
                              bankroll = 5000,
                              injuries = NULL,
                              base_dir = "weekly_output",
                              thresholds = list(
                                tot_high = 0.70, tot_low = c(0.55, 0.60),
                                ats_high = 0.70, ats_mid = c(0.65, 0.70)
                              )) {
  dir.create(base_dir, showWarnings = FALSE, recursive = TRUE)

  # produce bet card (recs) + all-games table
  out <- pick_bets_by_bucket(
    season = season, week = week, targets = targets,
    sims = sims, injuries = injuries,
    bankroll = bankroll,
    thresholds = thresholds,
    outfile = NULL  # we'll write two files below
  )

  ts <- format(Sys.time(), "%Y%m%d_%H%M%S")
  card_file <- file.path(base_dir, sprintf("card_%d_w%02d_%s.xlsx", season, week, ts))
  full_file <- file.path(base_dir, sprintf("allgames_%d_w%02d_%s.xlsx", season, week, ts))

  writexl::write_xlsx(out$card, card_file)
  writexl::write_xlsx(out$all_games, full_file)

  message("Saved:\n- ", card_file, "\n- ", full_file)

  # Quick console summary
  msg <- out$card |>
    dplyr::count(bet_type) |>
    dplyr::mutate(n = as.integer(n)) |>
    utils::capture.output()
  cat("Recs by type:\n", paste(msg, collapse = "\n"), "\n", sep = "")

  invisible(out)
}


settle_week <- function(card, season, log_csv = "weekly_output/bets_log.csv",
                        american_odds = -110) {
  # load actuals for this season
  actual <- nflreadr::load_schedules(season) |>
    dplyr::filter(game_type == "REG") |>
    dplyr::select(game_id, home_team, away_team, home_score, away_score,
                  spread_line, total_line) |>
    dplyr::rename(vegas_spread_line = spread_line,
                  vegas_total_line  = total_line)

  b <- if (american_odds > 0) american_odds/100 else 100/abs(american_odds)

  settled <- card |>
    dplyr::left_join(actual, by = "game_id") |>
    dplyr::mutate(
      # settle
      spread_actual = home_score - away_score,
      total_actual  = home_score + away_score,

      won = dplyr::case_when(
        bet_type == "ATS"   & pick == home_team ~ (spread_actual - vegas_spread_line > 0),
        bet_type == "ATS"   & pick == away_team ~ ((-spread_actual) + vegas_spread_line > 0),
        bet_type == "Total" & pick == "Over"    ~ (total_actual > vegas_total_line),
        bet_type == "Total" & pick == "Under"   ~ (total_actual < vegas_total_line),
        TRUE ~ NA
      ),

      risk   = stake_dollars,
      profit = dplyr::case_when(
        is.na(won)        ~ 0,
        won               ~ risk * b,      # win returns b per 1 risk
        !won              ~ -risk
      )
    )

  # append to a CSV log
  if (file.exists(log_csv)) {
    old <- readr::read_csv(log_csv, show_col_types = FALSE)
    new <- dplyr::bind_rows(old, settled)
  } else {
    new <- settled
  }
  readr::write_csv(new, log_csv)
  new
}

run_weekly_report <- function(season, week,
                              targets,
                              sims = 2000,
                              bankroll = 500,
                              injuries = NULL,
                              vegas_lines = NULL,
                              thresholds = list(
                                tot_high = 0.70, tot_low = c(0.55, 0.60),
                                ats_high = 0.70, ats_mid = c(0.65, 0.70)
                              ),
                              base_dir = "weekly_output",
                              american_odds = -110,
                              settle_now = FALSE) {
  `%||%` <- function(x, y) if (is.null(x)) y else x
  kNet <- function(odds) if (odds > 0) odds/100 else 100/abs(odds)

  # --- run card (+ maybe all_games) ---
  out_raw <- pick_bets_by_bucket(
    season = season, week = week, targets = targets,
    sims = sims, injuries = injuries, vegas_lines = vegas_lines,
    bankroll = bankroll, thresholds = thresholds
  )

  # Accept either a tibble (card only) or list(card=..., all_games=...)
  if (is.data.frame(out_raw)) {
    card <- out_raw
    all_games <- NULL
  } else if (is.list(out_raw) && !is.null(out_raw$card)) {
    card <- out_raw$card
    all_games <- out_raw$all_games %||% NULL
  } else {
    stop("pick_bets_by_bucket() returned an unexpected structure.")
  }

  # FS paths
  subdir <- file.path(base_dir, sprintf("%d_w%02d", season, week))
  dir.create(subdir, recursive = TRUE, showWarnings = FALSE)
  ts <- format(Sys.time(), "%Y%m%d_%H%M%S")
  card_file <- file.path(subdir, sprintf("card_%d_w%02d_%s.xlsx", season, week, ts))
  full_file <- file.path(subdir, sprintf("allgames_%d_w%02d_%s.xlsx", season, week, ts))

  # Write files
  if (!is.data.frame(card)) stop("`card` is not a data frame; check pick_bets_by_bucket().")
  writexl::write_xlsx(card, card_file)
  if (is.data.frame(all_games)) writexl::write_xlsx(all_games, full_file)

  message("Saved:\n- ", card_file, if (is.data.frame(all_games)) paste0("\n- ", full_file) else "")

  # ---- settle inline (optional) ----
  settle_week_inline <- function(card_df) {
    actual <- nflreadr::load_schedules(season) |>
      dplyr::filter(game_type == "REG", week == !!week) |>
      dplyr::select(game_id, home_team, away_team, home_score, away_score,
                    spread_line, total_line) |>
      dplyr::rename(vegas_spread_line = spread_line,
                    vegas_total_line  = total_line)

    b <- kNet(american_odds)
    card_df |>
      dplyr::left_join(actual, by = "game_id") |>
      dplyr::mutate(
        spread_actual = home_score - away_score,
        total_actual  = home_score + away_score,
        won = dplyr::case_when(
          bet_type == "ATS"   & pick == home_team ~ (spread_actual - vegas_spread_line > 0),
          bet_type == "ATS"   & pick == away_team ~ ((-spread_actual) + vegas_spread_line > 0),
          bet_type == "Total" & pick == "Over"    ~ (total_actual > vegas_total_line),
          bet_type == "Total" & pick == "Under"   ~ (total_actual < vegas_total_line),
          TRUE ~ NA
        ),
        risk   = card_df$stake_dollars %||% 0,
        profit = dplyr::case_when(
          is.na(won) ~ 0,
          won        ~ risk * b,
          !won       ~ -risk
        )
      )
  }

  summarize_log_inline <- function(df) {
    df |>
      dplyr::filter(!is.na(won)) |>
      dplyr::mutate(win = as.integer(won)) |>
      dplyr::group_by(bet_type) |>
      dplyr::summarise(
        n_bets       = dplyr::n(),
        hit_rate     = mean(win, na.rm = TRUE),
        total_staked = sum(risk, na.rm = TRUE),
        total_profit = sum(profit, na.rm = TRUE),
        ROI          = ifelse(total_staked > 0, total_profit / total_staked, NA_real_),
        .groups = "drop"
      )
  }

  settled <- weekly_summary <- lifetime_summary <- NULL
  if (settle_now) {
    settled <- settle_week_inline(card)
    log_csv <- file.path(base_dir, "bets_log.csv")
    if (file.exists(log_csv)) {
      old <- readr::read_csv(log_csv, show_col_types = FALSE)
      new <- dplyr::bind_rows(old, settled)
    } else {
      new <- settled
    }
    readr::write_csv(new, log_csv)

    weekly_summary   <- summarize_log_inline(settled)
    lifetime_summary <- summarize_log_inline(new)

    writexl::write_xlsx(
      list(weekly = weekly_summary, lifetime = lifetime_summary, settled = settled),
      file.path(subdir, sprintf("summary_%d_w%02d_%s.xlsx", season, week, ts))
    )
  }

  cat("\nRecs by type this week:\n")
  print(card |>
          dplyr::count(bet_type, name = "n") |>
          dplyr::arrange(dplyr::desc(n)))

  invisible(list(card = card, all_games = all_games,
                 settled = settled,
                 weekly_summary = weekly_summary,
                 lifetime_summary = lifetime_summary,
                 files = list(card = card_file,
                              all_games = if (is.data.frame(all_games)) full_file else NA)))
}



