# ---- helpers ---------------------------------------------------------------
# ---- Enable full core access on M2 ----
.libPaths(c(
  "/Users/dylanjaynes/Library/R/arm64/4.5/library",
  "/Library/Frameworks/R.framework/Versions/4.5-arm64/Resources/library"
))

# Core & memory settings
Sys.setenv(
  R_PARALLELLY_LOCALHOST_MAX_WORKERS = "12",
  R_PARALLELLY_AVAILABLECORES_FALLBACK = "12",
  R_FUTURE_GLOBALS_MAXSIZE = as.character(20 * 1024^3)  # 20 GB
)

options(
  parallelly.maxWorkers.localhost = Inf,
  parallelly.availableCores.fallback = 12,
  future.globals.maxSize = 20 * 1024^3,
  future.fork.enable = TRUE,
  future.rng.onMisuse = "ignore"
)

library(future)
library(furrr)
library(parallelly)

# One global plan
future::plan(multisession, workers = 10)


# Safe loader: prefer your cached helper if you have it; else nflfastR directly.
.get_pbp <- function(season, through_week = Inf) {
  if (exists("get_pbp_cached", mode = "function")) {
    get_pbp_cached(season, through_week = through_week,
                   cache_dir = "cache/pbp", refresh = "auto")
  } else {
    # Fallback: load full season, then trim weeks if asked.
    pbp <- nflfastR::load_pbp(season)
    if (is.finite(through_week)) {
      pbp <- pbp |> dplyr::mutate(week = as.numeric(week)) |> dplyr::filter(week <= through_week)
    }
    pbp
  }
}

# Try run_week with preloaded pbp; if your run_week version doesn't accept it,
# retry without (so this stays compatible with older signatures).
.run_week_safe <- function(season, week, sims, priors, injuries, vegas_lines,
                           w_eckel, w_hybrid, pbp_src, pbp_season_to_use) {
  # Try new signature, non-exported
  out <- tryCatch(
    nflproj:::run_week(
      season_target = season,
      week = week,
      sims = sims,
      w_eckel = w_eckel,
      w_hybrid = w_hybrid,
      tau_spread = priors$tau_spread,
      sigma_spread = priors$sigma_spread,
      df_spread = priors$df_spread,
      sigma_total = priors$sigma_total,
      df_total = priors$df_total,
      injuries = injuries,
      vegas_lines = vegas_lines,
      outfile = NULL,
      pbp_src = pbp_src,
      pbp_season_to_use = pbp_season_to_use
    ),
    error = function(e) e
  )
  if (!inherits(out, "error")) return(out)

  # Fallback (older run_week, no pbp_* args)
  nflproj:::run_week(
    season_target = season, week = week, sims = sims,
    w_eckel = w_eckel, w_hybrid = w_hybrid,
    tau_spread   = priors$tau_spread,
    sigma_spread = priors$sigma_spread,
    df_spread    = priors$df_spread,
    sigma_total  = priors$sigma_total,
    df_total     = priors$df_total,
    injuries     = injuries,
    vegas_lines  = vegas_lines,
    outfile      = NULL
  )
}


# ---- main: one future per SEASON ------------------------------------------
#' @export
backtest_all_seasons_fast <- function(
    seasons        = 2018:2024,
    weeks          = 1:18,
    sims           = 2000,
    priors         = list(tau_spread=12, sigma_spread=6.5, df_spread=6, sigma_total=3, df_total=6),
    injuries       = NULL,
    cache_root     = "cache/backtests",
    use_parallel   = TRUE,
    w_eckel        = 0.3,
    w_hybrid       = 0.5,
    vegas_lines    = NULL
) {

  ## cap math library threads
  Sys.setenv(OMP_NUM_THREADS = "1", MKL_NUM_THREADS = "1")
  if (requireNamespace("RhpcBLASctl", quietly = TRUE)) {
    RhpcBLASctl::blas_set_num_threads(1)
    RhpcBLASctl::omp_set_num_threads(1)
  }

  on.exit(future::plan(future::sequential), add = TRUE)

  dir.create(cache_root, recursive = TRUE, showWarnings = FALSE)

  tag <- sprintf("v2_ts%s_ss%s_ds%s_st%s_dt%s_sims%s_we%.2f_wh%.2f",
                 priors$tau_spread, priors$sigma_spread, priors$df_spread,
                 priors$sigma_total, priors$df_total, sims, w_eckel, w_hybrid)
  message(">>> Running tag: ", tag)

  season_fun <- function(season) {
    season_dir <- file.path(cache_root, tag, paste0("season_", season))
    dir.create(season_dir, recursive = TRUE, showWarnings = FALSE)
    logf <- file.path(season_dir, "backtest.log")

    # Small in-memory PBP pool keyed by season we actually need (season or season-1 for early weeks)
    pbp_pool <- new.env(parent = emptyenv())

    # Run each week serially inside this season
    out <- lapply(weeks, function(week) {
      cf <- file.path(season_dir, sprintf("%s_%d_%02d.rds", tag, season, week))
      if (file.exists(cf)) return(readRDS(cf))

      # Decide which PBP season to use this week
      pbp_season_to_use <- if (!is.na(week) && week < 4) season - 1 else season

      # Load PBP for that season once; if using current season, only up to this week
      key <- as.character(pbp_season_to_use)
      if (!exists(key, envir = pbp_pool, inherits = FALSE)) {
        through_req <- if (pbp_season_to_use == season) week else Inf
        assign(key, .get_pbp(pbp_season_to_use, through_week = through_req), envir = pbp_pool)
      } else {
        # If we already loaded current-season PBP for a lower week, expand to this week if needed
        if (pbp_season_to_use == season) {
          cur <- get(key, envir = pbp_pool, inherits = FALSE)
          max_have <- suppressWarnings(max(as.numeric(cur$week), na.rm = TRUE))
          if (is.finite(max_have) && week > max_have) {
            assign(key, .get_pbp(pbp_season_to_use, through_week = week), envir = pbp_pool)
          }
        }
      }
      pbp_src <- get(key, envir = pbp_pool, inherits = FALSE)

      message(sprintf("Season %d, week %02d...", season, week))

      res <- tryCatch(
        .run_week_safe(
          season, week, sims, priors, injuries, vegas_lines,
          w_eckel, w_hybrid,
          pbp_src = pbp_src, pbp_season_to_use = pbp_season_to_use
        ),
        error = function(e) {
          msg <- sprintf("[%s] FAIL %d-%02d: %s", format(Sys.time(), "%F %T"), season, week, e$message)
          cat(msg, "\n", file = logf, append = TRUE)
          message(msg)
          NULL
        }
      )

      if (!is.null(res)) saveRDS(res, cf)
      res
    })

    purrr::compact(out) |> dplyr::bind_rows()
  }



  if (use_parallel) {
    requireNamespace("furrr"); requireNamespace("future")
    future::plan(future::multisession, workers = max(1, parallel::detectCores()-1))
    # Avoid over-scheduling; let each worker chew a whole season.
    out <- furrr::future_map(
      seasons, season_fun,
      .options = furrr::furrr_options(
        seed = TRUE,
        packages = c("nflproj","dplyr","purrr","nflreadr","nflfastR")
      ),
      .progress = TRUE
    )
    future::plan(future::sequential)
  } else {
    out <- lapply(seasons, season_fun)
  }

  out <- purrr::compact(out)
  if (!length(out)) stop("No season produced results (check logs and that nflproj::run_week is exported).")

  results <- dplyr::bind_rows(out) |>
    dplyr::arrange(season_target, week, gameday, gametime, home_team)

  attr(results, "priors")    <- priors
  attr(results, "sims")      <- sims
  attr(results, "w_eckel")   <- w_eckel
  attr(results, "w_hybrid")  <- w_hybrid
  message("Backtest rows: ", nrow(results))
  results
}

#' @export
backtest_all_seasons2 <- function(
    seasons = 2018:2024, weeks = 1:17, sims = 2000,
    priors = list(tau_spread=12, sigma_spread=6.5, df_spread=6, sigma_total=3, df_total=6),
    injuries = NULL,
    cache_dir = "cache/backtests",
    use_parallel = TRUE,
    w_eckel = 0.3,
    w_hybrid = 0.5,
    vegas_lines = NULL
) {
  # ensure cache root exists
  dir.create(cache_dir, recursive = TRUE, showWarnings = FALSE)

  tag <- sprintf("v2_ts%s_ss%s_ds%s_st%s_dt%s_sims%s_we%.2f_wh%.2f",
                 priors$tau_spread, priors$sigma_spread, priors$df_spread,
                 priors$sigma_total, priors$df_total, sims, w_eckel, w_hybrid)

  run_week_cached <- function(season, week) {
    # each worker uses same tag subfolder
    dir.create(cache_dir, recursive = TRUE, showWarnings = FALSE)
    cf <- file.path(cache_dir, sprintf("%s_%d_%02d.rds", tag, season, week))
    if (file.exists(cf)) return(readRDS(cf))

    message(sprintf("Running season %d, week %02d...", season, week))
    logf <- file.path(cache_dir, "backtest.log")

    res <- tryCatch({
      # IMPORTANT: use ::: if run_week is not exported
      nflproj:::run_week(
        season_target = season, week = week, sims = sims,
        w_eckel = w_eckel, w_hybrid = w_hybrid,
        tau_spread   = priors$tau_spread,
        sigma_spread = priors$sigma_spread,
        df_spread    = priors$df_spread,
        sigma_total  = priors$sigma_total,
        df_total     = priors$df_total,
        injuries     = injuries,
        vegas_lines  = vegas_lines,
        outfile      = NULL
      )
    }, error = function(e) {
      msg <- sprintf("[%s] FAIL %d-%02d: %s",
                     format(Sys.time(), "%F %T"), season, week, e$message)
      dir.create(dirname(logf), showWarnings = FALSE, recursive = TRUE)
      cat(msg, "\n", file = logf, append = TRUE)
      message(msg)
      NULL
    })

    if (!is.null(res)) saveRDS(res, cf)
    res
  }

  jobs <- expand.grid(season = seasons, week = weeks, KEEP.OUT.ATTRS = FALSE, stringsAsFactors = FALSE)

  if (use_parallel) {
    if (!requireNamespace("furrr", quietly = TRUE) || !requireNamespace("future", quietly = TRUE)) {
      stop("Install {furrr} and {future} or set use_parallel = FALSE.")
    }
    # good default: cores - 1
    future::plan(future::multisession, workers = max(1, parallel::detectCores() - 1))
    on.exit(future::plan(future::sequential), add = TRUE)

    out <- furrr::future_pmap(
      list(jobs$season, jobs$week),
      function(season, week) run_week_cached(season, week),
      .options = furrr::furrr_options(
        seed = TRUE,
        scheduling = Inf,                      # let workers steal more work
        packages = c("nflproj","dplyr","purrr","nflreadr","nflfastR")
      )
    )
  } else {
    out <- purrr::pmap(list(jobs$season, jobs$week), run_week_cached)
  }

  out <- purrr::compact(out)
  if (!length(out)) stop("All tasks failed—workers probably couldn’t see run_week.")

  # bind & order; all frames should have season_target if run_week succeeded
  results <- dplyr::bind_rows(out) |>
    dplyr::arrange(season_target, week, gameday, gametime, home_team)

  attr(results, "priors")   <- priors
  attr(results, "sims")     <- sims
  attr(results, "w_eckel")  <- w_eckel
  attr(results, "w_hybrid") <- w_hybrid
  message("Backtest rows: ", nrow(results))
  results
}

#' @export
score_backtest <- function(bt_res, seasons) {
  actual <- nflreadr::load_schedules(seasons) |>
    dplyr::filter(game_type == "REG") |>
    dplyr::transmute(
      game_id, season, week, home_team, away_team,
      home_score, away_score,
      vegas_spread_line = spread_line,
      vegas_total_line  = total_line
    ) |>
    dplyr::mutate(
      vegas_spread_line.actual = vegas_spread_line,
      vegas_total_line.actual  = vegas_total_line
    )

  bm <- backtest_model(bt_res, actual)

  games <- bm$games
  ats_brier <- with(games, mean((ATS_prob - ATS_Result)^2, na.rm = TRUE))
  tot_brier <- with(games, mean((Total_prob - Total_Result)^2, na.rm = TRUE))

  tibble::tibble(
    ATS_ROI   = stats::weighted.mean(bm$ats_summary$ROI,   bm$ats_summary$n),
    Tot_ROI   = stats::weighted.mean(bm$total_summary$ROI, bm$total_summary$n),
    ATS_Hit   = with(bm$ats_summary,   sum(win_rate * n) / sum(n)),
    Tot_Hit   = with(bm$total_summary, sum(win_rate * n) / sum(n)),
    ATS_Brier = ats_brier,
    Tot_Brier = tot_brier
  )
}
# --- CPU & Parallel Plan Summary --------------------------------------------
library(parallel)
library(future)



# Detect total logical cores
total_cores <- parallel::detectCores(logical = TRUE)
usable_cores <- max(1, floor(total_cores / 2))   # conservative default

message("──────────────────────────────────────────────")
message("💻 System Parallel Summary")
message("Detected Cores: ", total_cores)
message("Reserved for OS: ", total_cores - usable_cores)
message("Grid Workers (outer loop): ", usable_cores)
message("Season Workers (inner loop via backtest_all_seasons_fast): ",
        max(1, usable_cores - 1))
message("──────────────────────────────────────────────")


furrr::future_map(1:6, ~{
  list(
    pid = Sys.getpid(),
    maxWorkers = getOption("parallelly.maxWorkers.localhost"),
    availableCores = parallelly::availableCores()
  )
})
message("Future plan: ", paste(class(future::plan()), collapse = " / "))
message("Workers: ", future::nbrOfWorkers())

#' @export
run_priors_grid_totals <- function(
    seasons        = 2018:2024,
    weeks          = 1:18,
    sims           = 2000,
    tau_spread     = 14,
    sigma_spread   = 6.5,
    df_spread      = 4,
    sigT_vals      = c(2.0, 2.5, 3.0, 3.5, 4.0),
    dfT_vals       = c(4, 5, 6, 8),
    w_eckel_vals   = 0.5,
    w_hybrid_vals  = 0.5,
    cache_root     = "cache/backtests_grid_totals",
    results_csv    = "grid_results_totals.csv",
    use_parallel   = TRUE,
    max_workers    = max(1, floor(parallel::detectCores() / 2))
) {
  if (!requireNamespace("tictoc", quietly = TRUE)) install.packages("tictoc")
  if (!requireNamespace("furrr", quietly = TRUE)) install.packages("furrr")
  if (!requireNamespace("progressr", quietly = TRUE)) install.packages("progressr")

  library(tictoc); library(progressr)

  furrr::future_map(1:6, ~{
    list(
      pid = Sys.getpid(),
      maxWorkers = getOption("parallelly.maxWorkers.localhost"),
      availableCores = parallelly::availableCores()
    )
  })


  grid <- expand.grid(
    sigma_total = sigT_vals,
    df_total    = dfT_vals,
    w_eckel     = w_eckel_vals,
    w_hybrid    = w_hybrid_vals,
    KEEP.OUT.ATTRS = FALSE, stringsAsFactors = FALSE
  )

  message("Parameter combos: ", nrow(grid))
  done <- if (file.exists(results_csv)) readr::read_csv(results_csv, show_col_types = FALSE) else NULL
  remaining <- if (!is.null(done)) {
    dplyr::anti_join(grid,
                     done |> dplyr::select(sigma_total, df_total, w_eckel, w_hybrid),
                     by = c("sigma_total", "df_total", "w_eckel", "w_hybrid")
    )
  } else grid
  if (!nrow(remaining)) {
    message("✅ All combos completed.")
    return(readr::read_csv(results_csv, show_col_types = FALSE))
  }

  message("Remaining combos: ", nrow(remaining))
  if (use_parallel) {
    future::plan(future::multisession, workers = max_workers)
    on.exit(future::plan(future::sequential), add = TRUE)

    if (inherits(future::plan(), "multisession")) {
      workers <- parallelly::availableCores()
      cl <- parallel::makeCluster(workers)
      on.exit(parallel::stopCluster(cl), add = TRUE)

      parallel::clusterEvalQ(cl, {
        suppressPackageStartupMessages({
          if ("devtools" %in% rownames(installed.packages())) {
            devtools::load_all(quiet = TRUE)
          } else {
            library(nflproj, character.only = TRUE)
          }
        })
        TRUE
      })
    }
  }

  #' @export
  combo_fun <- function(sigma_total, df_total, w_eckel, w_hybrid, combo_index, total_combos, start_time_global) {
    tag <- sprintf(
      "ts%s_ss%s_ds%s_st%s_dt%s_we%.2f_wh%.2f_s%s",
      tau_spread, sigma_spread, df_spread,
      sigma_total, df_total, w_eckel, w_hybrid, sims
    )
    combo_start <- Sys.time()
    message(sprintf("\n▶ [%02d/%02d] Starting %s", combo_index, total_combos, tag))

    priors <- list(
      tau_spread   = tau_spread,
      sigma_spread = sigma_spread,
      df_spread    = df_spread,
      sigma_total  = sigma_total,
      df_total     = df_total
    )

    bt <- backtest_all_seasons_fast(
      seasons = seasons, weeks = weeks, sims = sims,
      priors = priors,
      cache_root = file.path(cache_root, tag),
      use_parallel = FALSE,
      w_eckel = w_eckel, w_hybrid = w_hybrid
    )

    met <- score_backtest(bt, seasons)
    row <- dplyr::bind_cols(
      tag = tag,
      tibble::tibble(
        sigma_total = sigma_total,
        df_total = df_total,
        w_eckel = w_eckel,
        w_hybrid = w_hybrid,
        sims = sims
      ),
      met
    )

    readr::write_csv(row, results_csv, append = file.exists(results_csv))

    # Timing and ETA tracking
    combo_duration <- as.numeric(difftime(Sys.time(), combo_start, units = "mins"))
    elapsed_total <- as.numeric(difftime(Sys.time(), start_time_global, units = "mins"))
    combos_done <- combo_index
    combos_left <- total_combos - combos_done
    avg_per_combo <- elapsed_total / combos_done
    eta_minutes <- combos_left * avg_per_combo
    eta_finish <- Sys.time() + as.difftime(eta_minutes, units = "mins")

    message(sprintf("✅ Finished %s (%.1f min)", tag, combo_duration))
    message(sprintf("Progress: %d/%d | Avg %.1f min/combo | ETA: %s (%.1f hrs remaining)",
                    combos_done, total_combos, avg_per_combo,
                    format(eta_finish, '%H:%M:%S'),
                    eta_minutes / 60))

    return(row)
  }

  # ─── Run grid ──────────────────────────────────────────────────────────────
  total_combos <- nrow(remaining)
  start_time_global <- Sys.time()
  message("\n⏳ Starting grid search across ", total_combos, " combos ...")

  tictoc::tic("Total grid search")
  progressr::with_progress({
    p <- progressr::progressor(steps = total_combos)
    results <- if (use_parallel) {
      furrr::future_pmap(
        cbind(remaining, combo_index = seq_len(total_combos)),
        function(sigma_total, df_total, w_eckel, w_hybrid, combo_index) {
          require(nflproj)  # ensures all functions visible inside each worker
          out <- combo_fun(sigma_total, df_total, w_eckel, w_hybrid,
                           combo_index, total_combos, start_time_global)
          p(sprintf("Combo %d/%d done", combo_index, total_combos))
          out
        },
        .options = furrr::furrr_options(
          seed = TRUE,
          packages = c("nflproj","dplyr","purrr","furrr","nflreadr","nflfastR")
        )
      )
    } else {
      purrr::pmap(cbind(remaining, combo_index = seq_len(total_combos)),
                  function(sigma_total, df_total, w_eckel, w_hybrid, combo_index) {
                    require(nflproj)  # 🧠 ensure nflproj is loaded inside each worker
                    out <- combo_fun(sigma_total, df_total, w_eckel, w_hybrid,
                                     combo_index, total_combos, start_time_global)
                    p(sprintf("Combo %d/%d done", combo_index, total_combos))
                    out
                  })
    }
  })
  tictoc::toc()

  # Combine & summarize
  all_results <- readr::read_csv(results_csv, show_col_types = FALSE) |>
    dplyr::arrange(dplyr::desc(Tot_ROI))
  message("\n✅ Grid search complete: ", nrow(all_results), " total rows.")
  return(all_results)
}



# HOW TO RUN:

# (optional) reproducible RNG per-session
# set.seed(20251015)
#
# # prevent nested future RNG warnings
# options(future.rng.onMisuse = "ignore")
#
# # kick it off
# grid_out <- run_priors_grid(
#   seasons = 2018:2024,
#   weeks   = 1:18,
#   sims    = 1000,          # bump to 2000+ later if you want tighter noise
#   cache_root = "cache/backtests_grid",
#   results_csv = "overnight_grid_results.csv",
#   use_parallel = TRUE
# )
#
# # see top rows
# dplyr::slice_head(grid_out, n = 20)
