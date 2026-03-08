#proper_grid_backtest.R

#' @import dplyr
#' @import future
#' @import future.apply
NULL

# ------------------ GRID DEFINITIONS ------------------

GRID_ATS <- expand.grid(
  tau_spread   = c(8, 12, 18),
  sigma_spread = c(4.5, 6.0, 7.5),
  df_spread    = c(3, 6, 12)
) %>%
  dplyr::sample_n(12)

GRID_TOTAL <- expand.grid(
  sigma_total = c(3.0, 4.0, 5.0),
  df_total    = c(4, 8, 16)
)

# ------------------ ONE GRID ROW WORKER ------------------

.tune_grid_one_row <- function(row_idx,
                               grid,
                               seasons,
                               weeks,
                               sims_tune,
                               kind = c("ATS", "TOTAL"),
                               verbose = TRUE) {

  kind <- match.arg(kind)
  p <- grid[row_idx, , drop = FALSE]

  if (verbose) {
    msg <- switch(
      kind,
      ATS   = sprintf("[ATS] grid row %d / %d: tau=%g, sigma=%g, df=%g",
                      row_idx, nrow(grid),
                      p$tau_spread, p$sigma_spread, p$df_spread),
      TOTAL = sprintf("[TOTAL] grid row %d / %d: sigma_total=%g, df_total=%g",
                      row_idx, nrow(grid),
                      p$sigma_total, p$df_total)
    )
    message(msg)
  }

  max_chunks <- length(seasons) * length(weeks)
  chunks <- vector("list", max_chunks)
  k <- 1L

  for (ss in seasons) {
    for (wk in weeks) {

      # -------------- RUN MODEL ----------------

      if (kind == "ATS") {
        preds <- nflproj::run_week_prior(
          season_target = ss,
          week          = wk,
          p             = p,
          sims          = sims_tune
        )

      } else {
        preds <- nflproj::run_week_prior(
          season_target = ss,
          week          = wk,
          p             = p,
          sims          = sims_tune
        )

      }

      # -------------- EVALUATE ----------------

      ev <- nflproj::eval_week_prior(preds, ss, wk)

      if (!is.null(ev) && !is.null(ev$summary) && nrow(ev$summary) > 0) {
        chunks[[k]] <- dplyr::mutate(ev$summary,
                                     season = ss,
                                     week   = wk)
        k <- k + 1L
      }
    }
  }

  metrics <- if (k == 1L) dplyr::tibble() else dplyr::bind_rows(chunks[seq_len(k - 1L)])

  list(params = p, metrics = metrics)
}

# ------------------ ATS TUNING ------------------

#' @export
tune_ATS_parallel <- function(seasons,
                              weeks,
                              grid      = GRID_ATS,
                              sims_tune = 4000,
                              n_workers = 6,
                              verbose   = TRUE) {

  future::plan(multisession, workers = n_workers)

  start.time <- Sys.time()

  results <- future.apply::future_lapply(
    X = seq_len(nrow(grid)),
    FUN = .tune_grid_one_row,
    grid      = grid,
    seasons   = seasons,
    weeks     = weeks,
    sims_tune = sims_tune,
    kind      = "ATS",
    verbose   = verbose,
    future.packages = c("nflproj", "dplyr")
  )

  end.time <- Sys.time()
  if (verbose) message("ATS tuning completed in ", round(end.time - start.time, 2))

  results
}

# ------------------ TOTALS TUNING ------------------

#' @export
tune_TOTAL_parallel <- function(seasons,
                                weeks,
                                grid      = GRID_TOTAL,
                                sims_tune = 4000,
                                n_workers = 6,
                                verbose   = TRUE) {

  future::plan(multisession, workers = n_workers)

  start.time <- Sys.time()

  results <- future.apply::future_lapply(
    X = seq_len(nrow(grid)),
    FUN = .tune_grid_one_row,
    grid      = grid,
    seasons   = seasons,
    weeks     = weeks,
    sims_tune = sims_tune,
    kind      = "TOTAL",
    verbose   = verbose,
    future.packages = c("nflproj", "dplyr")
  )

  end.time <- Sys.time()
  if (verbose) message("TOTAL tuning completed in ", round(end.time - start.time, 2))

  results
}

# ------------------ SELECTION HELPERS ------------------

#' @export
select_best_ATS <- function(results) {
  df <- dplyr::bind_rows(
    lapply(results, function(x) {
      if (is.null(x$metrics) || nrow(x$metrics) == 0) return(NULL)
      cbind(as.data.frame(x$params), x$metrics)
    })
  )

  df %>%
    dplyr::group_by(tau_spread, sigma_spread, df_spread) %>%
    dplyr::summarise(
      ATS_Brier = mean(ATS_Brier, na.rm = TRUE),
      ATS_Hit   = mean(ATS_Hit,   na.rm = TRUE),
      ATS_RMSE  = mean(ATS_RMSE,  na.rm = TRUE),
      ATS_MAE   = mean(ATS_MAE,   na.rm = TRUE),
      .groups   = "drop"
    ) %>%
    dplyr::arrange(ATS_Brier, dplyr::desc(ATS_Hit), ATS_RMSE) %>%
    dplyr::slice(1)
}

#' @export
select_best_TOTAL <- function(results) {
  df <- dplyr::bind_rows(
    lapply(results, function(x) {
      if (is.null(x$metrics) || nrow(x$metrics) == 0) return(NULL)
      cbind(as.data.frame(x$params), x$metrics)
    })
  )

  df %>%
    dplyr::group_by(sigma_total, df_total) %>%
    dplyr::summarise(
      Total_Brier = mean(Total_Brier, na.rm = TRUE),
      Total_Hit   = mean(Total_Hit,   na.rm = TRUE),
      Total_RMSE  = mean(Total_RMSE,  na.rm = TRUE),
      Total_MAE   = mean(Total_MAE,   na.rm = TRUE),
      .groups     = "drop"
    ) %>%
    dplyr::arrange(Total_Brier, dplyr::desc(Total_Hit), Total_RMSE) %>%
    dplyr::slice(1)
}
