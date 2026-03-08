# run_week.R

#' @import magrittr
#' @import dplyr
NULL

#' Run all games for a given NFL week
#'
#' @param season_target integer season (e.g. 2025)
#' @param week integer week
#' @param sims integer simulations per game
#' @param w_eckel numeric, weight on Eckel model [0, 1]
#' @param targets optional variance calibration object (or NULL)
#' @param vegas_lines optional tibble with vegas lines, or NULL
#' @param outfile optional .xlsx path to write results
#' @param tau_spread prior SD for spread mean (Bayesian shrink)
#' @param sigma_spread scale of t-shock for spread
#' @param df_spread df of t-shock for spread
#' @param sigma_total scale of t-shock for total
#' @param df_total df of t-shock for total
#' @param w_hybrid blend weight of model vs market SD
#' @param injuries optional injury list
#' @param pbp_cache_dir dir for cached PBP
#' @param precompute_cache_dir dir for precomputed season metrics
#'
#' @return tibble with per-game simulation summaries
#' @export
run_week <- function(season_target,
                     week,
                     sims                 = 10000,
                     w_eckel              = 0.3,
                     targets              = NULL,
                     vegas_lines          = NULL,   # optional tibble
                     outfile              = NULL,
                     tau_spread           = 12,
                     sigma_spread         = 6.5,
                     df_spread            = 6,
                     sigma_total          = 3,
                     df_total             = 6,
                     w_hybrid             = 0.5,
                     injuries             = NULL,
                     pbp_cache_dir        = "cache/pbp",
                     precompute_cache_dir = "cache/pre") {

  pbp_season_to_use <- if (!is.na(week) && week < 4) season_target - 1 else season_target
  through_req <- if (pbp_season_to_use == season_target) week else Inf

  pbp_src <- get_pbp_cached(
    season      = pbp_season_to_use,
    through_week= through_req,
    cache_dir   = pbp_cache_dir,
    refresh     = "auto"
  )
  # ---- Load schedule for this week ----
  sched <- nflreadr::load_schedules(season_target) %>%
    dplyr::filter(week == !!week, game_type == "REG") %>%
    dplyr::arrange(gameday, gametime, home_team)

  # ---- Join in vegas_lines if provided ----
  if (!is.null(vegas_lines)) {
    sched <- sched %>%
      dplyr::select(
        game_id, season, week, gameday, gametime,
        home_team, away_team, spread_line, total_line
      ) %>%
      dplyr::left_join(
        vegas_lines %>%
          dplyr::select(home_team, away_team, week, spread_line, total_line),
        by = c("home_team", "away_team", "week"),
        relationship = "many-to-many"
      ) %>%
      dplyr::mutate(
        spread_line = dplyr::coalesce(spread_line.y, spread_line.x),
        total_line  = dplyr::coalesce(total_line.y,  total_line.x)
      ) %>%
      dplyr::select(-dplyr::ends_with(".x"), -dplyr::ends_with(".y"))
  } else {
    sched <- sched %>%
      dplyr::select(
        game_id, season, week, gameday, gametime,
        home_team, away_team, spread_line, total_line
      )
  }

  # ---- Simulate all games in the week ----
  results <- purrr::map_dfr(seq_len(nrow(sched)), function(i) {
    row <- sched[i, ]

    run_one_game(
      season_target       = season_target,
      week                = week,
      home_team           = row$home_team,
      away_team           = row$away_team,
      sims                = sims,
      w_eckel             = w_eckel,
      targets             = targets,
      vegas_spread_line   = row$spread_line,
      vegas_total_line    = row$total_line,
      sched_row           = row,
      tau_spread          = tau_spread,
      sigma_spread        = sigma_spread,
      df_spread           = df_spread,
      sigma_total         = sigma_total,
      df_total            = df_total,
      w_hybrid            = w_hybrid,
      injuries            = injuries,
      pbp_season_to_use   = pbp_season_to_use,
      precompute_cache_dir = precompute_cache_dir,
      pbp_cache_dir        = pbp_cache_dir,
      pbp_src              = pbp_src
    )
  })

  # ---- Add cover prob for sorting ----
  results <- results %>%
    dplyr::mutate(
      cover_prob = dplyr::case_when(
        ATS_pick == home_team ~ home_cover_prob,
        ATS_pick == away_team ~ away_cover_prob,
        TRUE ~ NA_real_
      )
    ) %>%
    dplyr::arrange(dplyr::desc(cover_prob), gameday, gametime, home_team)

  # ---- Variance diagnostics ----
  spread_pre  <- mean(results$pre_sd_spread, na.rm = TRUE)
  spread_post <- mean(results$spread_sd,      na.rm = TRUE)
  total_pre   <- mean(results$pre_sd_total,   na.rm = TRUE)
  total_post  <- mean(results$total_sd,       na.rm = TRUE)

  cat(sprintf(
    "\n=== Week %d variance check ===\nSpread SD (pre/post): %.2f → %.2f\nTotal SD  (pre/post): %.2f → %.2f\n",
    week, spread_pre, spread_post, total_pre, total_post
  ))

  # ---- Backward compatibility for Win_* if missing ----
  if (!"home_win_prob" %in% names(results)) {
    results <- results %>%
      dplyr::mutate(
        home_win_prob = ifelse(home_mean > away_mean, 0.55, 0.45),
        away_win_prob = 1 - home_win_prob,
        Win_pick      = ifelse(home_win_prob >= away_win_prob, home_team, away_team),
        Win_prob      = pmax(home_win_prob, away_win_prob)
      )
  }

  # ensure season_target column present
  results <- results %>%
    dplyr::mutate(season_target = !!season_target)

  # ---- Optional: write week results to Excel ----
  if (!is.null(outfile)) {
    writexl::write_xlsx(
      results %>% dplyr::mutate(across(where(is.numeric), round, 4)),
      outfile
    )
    message("Saved week results to: ", outfile)
  }

  results
}
