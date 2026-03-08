# run_week_prior.R

#' @import dplyr
NULL

# ---------------------------------------------------------------------
# Safe fallback operator
# ---------------------------------------------------------------------
`%||%` <- function(x, y) if (is.null(x) || is.na(x)) y else x

# ---------------------------------------------------------------------
# Wrapper for running a week with a given parameter list `p`
# Compatible with tune_ATS_parallel() and tune_TOTAL_parallel()
# ---------------------------------------------------------------------

#' Run a week using a prior-parameter list `p` (for grid tuning)
#'
#' This wrapper ensures:
#' - Arguments match CURRENT run_week() exactly
#' - No old / unused args like `pbp_src`
#' - `p` may contain ATS or TOTAL hyperparameters
#'
#' @param season_target integer
#' @param week integer
#' @param p named list or data.frame row with prior params
#' @param sims simulations per game
#'
#' @export
run_week_prior <- function(season_target,
                           week,
                           p,
                           sims = 2500) {

  # Extract hyperparameters if present, otherwise fallback defaults
  tau_spread   <- p$tau_spread   %||% 12
  sigma_spread <- p$sigma_spread %||% 6
  df_spread    <- p$df_spread    %||% 6

  sigma_total  <- p$sigma_total  %||% 4
  df_total     <- p$df_total     %||% 8

  # -----------------------------------------------------------------
  # EXACT match to your run_week() signature (NO pbp_src)
  # -----------------------------------------------------------------
  run_week(
    season_target = season_target,
    week          = week,
    sims          = sims,
    w_eckel       = 0.30,
    targets       = NULL,
    vegas_lines   = NULL,
    outfile       = NULL,

    # Priors (ATS or TOTAL depending on grid)
    tau_spread    = tau_spread,
    sigma_spread  = sigma_spread,
    df_spread     = df_spread,
    sigma_total   = sigma_total,
    df_total      = df_total,

    # Keep your defaults
    w_hybrid      = 0.4,
    injuries      = NULL,

    # REQUIRED BY YOUR run_week()
    precompute_cache_dir = "cache/pre",
    pbp_cache_dir        = "cache/pbp"
  )
}

# ---------------------------------------------------------------------
# Evaluation wrapper (unchanged)
# ---------------------------------------------------------------------

#' Evaluate predictions for a given week (for tuning)
#'
#' @export
eval_week_prior <- function(preds, season, week) {
  evaluate_week(
    model_df      = preds,
    season_target = season,
    week_target   = week
  )
}
