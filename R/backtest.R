#' @import magrittr
#' @import dplyr
NULL


# Compact backtester comparing spread/total errors and Brier scores vs closing lines
backtest_projectors <- function(seasons = 2023:2024, sims_per_game = 2, w_eckel = 0.5, quiet = TRUE) {

  schedules <- load_schedules_cached(seasons) %>%
    dplyr::mutate(week = as.numeric(week)) %>%
    dplyr::filter(!is.na(home_team), !is.na(away_team),
                  !is.na(spread_line), !is.na(total_line),
                  !is.na(home_score), !is.na(away_score))

  # iterate games
  per_game <- lapply(seq_len(nrow(schedules)), function(i) {
    g <- schedules[i, ]
    if (!quiet) message(sprintf("Sim %s (Wk %s)", g$game_id, g$week))
    season_target <- g$season; week <- g$week

    # prepare inputs (prior year if week < 4)
    prep <- prepare_game_inputs(season_target, week, g$home_team, g$away_team)
    pbp <- prep$pbp; season_use <- prep$season_use; week_use <- prep$week_use
    t1 <- prep$t1;   t2 <- prep$t2

    # PBP sims
    mat_pbp <- simulate_game_matrix(project_matchup, g$home_team, g$away_team, season_use, week_use, pbp, t1, t2, sims_per_game)
    sp_pbp <- mat_pbp["home", ] - mat_pbp["away", ]
    tt_pbp <- mat_pbp["home", ] + mat_pbp["away", ]

    # Eckel sims
    mat_eck <- simulate_game_matrix(project_matchup_eckel, g$home_team, g$away_team, season_use, week_use, pbp, t1, t2, sims_per_game)
    sp_eck <- mat_eck["home", ] - mat_eck["away", ]
    tt_eck <- mat_eck["home", ] + mat_eck["away", ]

    # means
    sp_pbp_m <- mean(sp_pbp, na.rm = TRUE);  tt_pbp_m <- mean(tt_pbp, na.rm = TRUE)
    sp_eck_m <- mean(sp_eck, na.rm = TRUE);  tt_eck_m <- mean(tt_eck, na.rm = TRUE)

    # actuals + lines
    margin <- g$home_score - g$away_score
    total  <- g$home_score + g$away_score
    v_spread <- g$spread_line; v_total <- g$total_line

    # errors
    tibble::tibble(
      season = g$season, week = week, game_id = g$game_id,
      spread_error_pbp  = margin - sp_pbp_m,
      spread_error_eckel= margin - sp_eck_m,
      total_error_pbp   = total  - tt_pbp_m,
      total_error_eckel = total  - tt_eck_m,
      # probs vs line (Brier components)
      cover_true = as.integer(margin > -v_spread),
      over_true  = as.integer(total  >  v_total),
      cover_prob_pbp = mean(sp_pbp > -v_spread),
      cover_prob_eck = mean(sp_eck > -v_spread),
      over_prob_pbp  = mean(tt_pbp >  v_total),
      over_prob_eck  = mean(tt_eck >  v_total)
    )
  })

  df <- dplyr::bind_rows(per_game)

  week_summary <- df %>%
    dplyr::group_by(week) %>%
    dplyr::summarise(
      n = dplyr::n(),
      spread_mae_pbp   = mean(abs(spread_error_pbp), na.rm = TRUE),
      spread_mae_eckel = mean(abs(spread_error_eckel), na.rm = TRUE),
      spread_rmse_pbp   = sqrt(mean(spread_error_pbp^2, na.rm = TRUE)),
      spread_rmse_eckel = sqrt(mean(spread_error_eckel^2, na.rm = TRUE)),
      total_mae_pbp   = mean(abs(total_error_pbp), na.rm = TRUE),
      total_mae_eckel = mean(abs(total_error_eckel), na.rm = TRUE),
      total_rmse_pbp   = sqrt(mean(total_error_pbp^2, na.rm = TRUE)),
      total_rmse_eckel = sqrt(mean(total_error_eckel^2, na.rm = TRUE)),
      brier_cover_pbp  = mean((cover_prob_pbp - cover_true)^2, na.rm = TRUE),
      brier_cover_eckel= mean((cover_prob_eck - cover_true)^2, na.rm = TRUE),
      brier_over_pbp   = mean((over_prob_pbp - over_true)^2, na.rm = TRUE),
      brier_over_eckel = mean((over_prob_eck - over_true)^2, na.rm = TRUE),
      .groups = "drop"
    )

  list(per_game = df, week_summary = week_summary)
}
backtest_model <- function(model_results, actual_results,
                           bins = c(0.5, 0.55, 0.60, 0.65, 0.70, 1.0)) {
  df <- model_results %>%
    dplyr::left_join(actual_results, by = "game_id", suffix = c(".model", ".actual")) %>%
    dplyr::mutate(
      home_team = home_team.actual,
      away_team = away_team.actual,
      vegas_spread_line = vegas_spread_line.actual,
      vegas_total_line  = vegas_total_line.actual,

      # Actual outcomes
      actual_spread_cover = dplyr::case_when(
        (home_score - away_score + vegas_spread_line) > 0 ~ home_team,
        (home_score - away_score + vegas_spread_line) < 0 ~ away_team,
        TRUE ~ "Push"
      ),
      actual_total = dplyr::case_when(
        (home_score + away_score) > vegas_total_line ~ "Over",
        (home_score + away_score) < vegas_total_line ~ "Under",
        TRUE ~ "Push"
      ),

      ATS_correct   = (ATS_pick == actual_spread_cover),
      Total_correct = (Total_pick == actual_total),

      ATS_profit = dplyr::case_when(
        actual_spread_cover == "Push" ~ 0,
        ATS_correct ~ 100, TRUE ~ -110
      ),
      Total_profit = dplyr::case_when(
        actual_total == "Push" ~ 0,
        Total_correct ~ 100, TRUE ~ -110
      ),

      # New: numeric result flags for Brier scoring
      ATS_Result   = ifelse(actual_spread_cover == ATS_pick, 1,
                            ifelse(actual_spread_cover == "Push", NA, 0)),
      Total_Result = ifelse(actual_total == Total_pick, 1,
                            ifelse(actual_total == "Push", NA, 0)),

      ATS_bin   = cut(ATS_prob,   breaks = bins, include.lowest = TRUE),
      Total_bin = cut(Total_prob, breaks = bins, include.lowest = TRUE)
    )

  ats_summary <- df %>%
    dplyr::group_by(ATS_bin) %>%
    dplyr::summarize(n = dplyr::n(),
                     win_rate = mean(ATS_correct, na.rm = TRUE),
                     ROI = mean(ATS_profit, na.rm = TRUE) / 110,
                     .groups = "drop")

  total_summary <- df %>%
    dplyr::group_by(Total_bin) %>%
    dplyr::summarize(n = dplyr::n(),
                     win_rate = mean(Total_correct, na.rm = TRUE),
                     ROI = mean(Total_profit, na.rm = TRUE) / 110,
                     .groups = "drop")

  list(games = df, ats_summary = ats_summary, total_summary = total_summary)
}


backtest_all_seasons <- function(seasons = 2018:2024, weeks = 1:18,
                                 sims = 2000,
                                 priors = list(tau_spread = 12, sigma_spread = 6.5, df_spread = 6,
                                               sigma_total = 3,  df_total = 6),
                                 injuries = NULL,
                                 cache_dir = "cache/backtests",
                                 use_parallel = TRUE) {
  if (!dir.exists(cache_dir)) dir.create(cache_dir, recursive = TRUE)

  tag <- sprintf("ts%s_ss%s_ds%s_st%s_dt%s_sims%s",
                 priors$tau_spread, priors$sigma_spread, priors$df_spread,
                 priors$sigma_total, priors$df_total, sims)

  run_week_cached <- function(season, week) {
    cache_file <- file.path(cache_dir, sprintf("%s_season_%d_week_%02d.rds", tag, season, week))
    if (file.exists(cache_file)) return(readRDS(cache_file))

    message(sprintf("Running season %d, week %d...", season, week))
    res <- tryCatch({
      run_week(
        season_target = season, week = week, sims = sims,
        tau_spread   = priors$tau_spread,
        sigma_spread = priors$sigma_spread,
        df_spread    = priors$df_spread,
        sigma_total  = priors$sigma_total,
        df_total     = priors$df_total,
        injuries     = injuries,
        outfile      = NULL
      )
    }, error = function(e) {
      warning(sprintf("Failed %d-%02d: %s", season, week, e$message))
      NULL
    })

    if (!is.null(res)) saveRDS(res, cache_file)
    res
  }

  jobs <- expand.grid(season = seasons, week = weeks)

  if (use_parallel) {
    future::plan(future::multisession, workers = max(1, parallel::detectCores() - 1))
    out <- future.apply::future_mapply(run_week_cached,
                                       season = jobs$season, week = jobs$week,
                                       SIMPLIFY = FALSE, USE.NAMES = FALSE)
    future::plan(future::sequential)
  } else {
    out <- purrr::map2(jobs$season, jobs$week, run_week_cached)
  }

  results <- out |>
    purrr::compact() |>
    dplyr::bind_rows() |>
    dplyr::arrange(season_target, week, gameday, gametime, home_team)

  attr(results, "priors") <- priors
  attr(results, "sims")   <- sims
  message("Backtest rows: ", nrow(results))
  results
}







