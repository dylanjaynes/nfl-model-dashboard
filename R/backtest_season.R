# R/backtest_season.R

# ==========================
# Backtest Helpers
# ==========================

library(dplyr)
library(tibble)

# --- wrapper to run a season backtest
backtest_season <- function(season_target, weeks, sims,
                            w_eckel, tau_spread, sigma_spread, w_hybrid) {
  # grab cached targets once
  targets <- get_targets_cache(2018:2024)

  results <- purrr::map_dfr(weeks, function(w) {
    wk <- run_week(
      season_target = season_target,
      week = w,
      sims = sims,
      w_eckel = w_eckel,
      targets = targets,              # <- use cache here
      tau_spread = tau_spread,
      sigma_spread = sigma_spread,
      w_hybrid = w_hybrid
    )
    evaluate_week(wk, season_target, w)$summary
  })

  results %>%
    summarise(
      MAE   = mean(MAE, na.rm = TRUE),
      RMSE  = mean(RMSE, na.rm = TRUE),
      Brier = mean(Brier, na.rm = TRUE)
    )
}


# --- multi-season wrapper
backtest_multi <- function(seasons, weeks = 1:1,
                           sims = 20,
                           w_eckel = 0.6,
                           tau_spread = 8,
                           sigma_spread = 6.5,
                           w_hybrid = 0.25) {
  targets <- get_targets_cache(2018:2024)

  with_progress({
    p <- progressor(steps = length(seasons))

    per_season <- purrr::map_dfr(seasons, function(season_target) {
      res <- backtest_season(
        season_target = season_target,
        weeks = weeks,
        sims = sims,
        w_eckel = w_eckel,
        tau_spread = tau_spread,
        sigma_spread = sigma_spread,
        w_hybrid = w_hybrid
      )
      p(message = paste("Finished season", season_target))
      dplyr::mutate(res, season = season_target)
    })

    summary <- per_season %>%
      dplyr::summarise(
        MAE   = mean(MAE, na.rm = TRUE),
        RMSE  = mean(RMSE, na.rm = TRUE),
        Brier = mean(Brier, na.rm = TRUE)
      ) %>%
      dplyr::mutate(season = "ALL")

    dplyr::bind_rows(per_season, summary)
  })
}

