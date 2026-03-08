#' @import magrittr
#' @import dplyr
NULL


# run_grid_search.R

run_grid_search <- function(season_target = 2024,
                            weeks = 1:1,
                            sims = 2,
                            w_eckel = 0.6) {
  library(furrr)
  library(progressr)
  library(dplyr)
  library(tibble)

  plan(multisession, workers = parallel::detectCores() - 1)
  handlers(global = TRUE)
  handlers("progress")

  # --- parameter grids
  param_grid <- expand.grid(
    tau_spread   = c(4, 6),
    sigma_spread = c(6.5, 7.5, 9),
    w_hybrid     = c(0.1, 0.25)
  )

  with_progress({
    p <- progressor(steps = nrow(param_grid))

    grid_results <- future_pmap_dfr(param_grid, function(tau_spread, sigma_spread, w_hybrid) {
      res <- backtest_multi(
        seasons = 2024:2024,
        weeks = 1:1,
        sims = 2,
        w_eckel = 0.6,
        tau_spread = tau_spread,
        sigma_spread = sigma_spread,
        w_hybrid = w_hybrid
      )
      p(message = sprintf("τ=%s σ=%s w=%s", tau_spread, sigma_spread, w_hybrid))
      tibble::tibble(
        tau_spread = tau_spread,
        sigma_spread = sigma_spread,
        w_hybrid = w_hybrid,
        MAE   = mean(res$MAE),
        RMSE  = mean(res$RMSE),
        Brier = mean(res$Brier)
      )
    })
  })

  grid_results %>%
    arrange(Brier)
}
