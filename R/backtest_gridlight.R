# ==========================
# Lightweight Grid Search (sequential, Mac-safe)
# ==========================

grid_search_light <- function(
    season_target = 2024,
    weeks = 1:5,         # keep short for quick runs
    sims = 1000,         # smaller sims for speed
    w_eckel = 0.6
) {
  # define parameter grid
  param_grid <- expand.grid(
    tau_spread   = c(6, 8, 10),
    sigma_spread = c(5.5, 6.5, 7.5),
    w_hybrid     = c(0.25, 0.5)
  )

  results <- list()

  for (i in seq_len(nrow(param_grid))) {
    p <- param_grid[i, ]
    message(sprintf(">>> Testing tau=%s, sigma=%s, w_hybrid=%s",
                    p$tau_spread, p$sigma_spread, p$w_hybrid))

    res <- backtest_season(
      season_target = season_target,
      weeks = weeks,
      sims = sims,
      w_eckel = w_eckel,
      tau_spread = p$tau_spread,
      sigma_spread = p$sigma_spread,
      w_hybrid = p$w_hybrid
    )

    results[[i]] <- cbind(p, res)
  }

  dplyr::bind_rows(results) %>%
    dplyr::arrange(Brier)  # lower Brier is better
}
