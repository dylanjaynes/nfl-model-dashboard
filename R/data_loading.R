# Cached loaders (optional but nice). Switch to nflfastR/nflreadr directly if preferred.

load_pbp_cached <- function(season, cache_dir = tools::R_user_dir("nflproj", which = "cache")) {
  if (!requireNamespace("nflfastR", quietly = TRUE)) stop("Please install 'nflfastR'.")
  dir.create(cache_dir, showWarnings = FALSE, recursive = TRUE)
  f <- file.path(cache_dir, sprintf("pbp_%d.rds", season))
  if (file.exists(f)) return(readRDS(f))
  x <- nflfastR::load_pbp(season)
  saveRDS(x, f); x
}

load_schedules_cached <- function(seasons, cache_dir = tools::R_user_dir("nflproj", which = "cache")) {
  if (!requireNamespace("nflreadr", quietly = TRUE)) stop("Please install 'nflreadr'.")
  dir.create(cache_dir, showWarnings = FALSE, recursive = TRUE)
  key <- paste(range(seasons), collapse = "_")
  f <- file.path(cache_dir, sprintf("schedules_%s.rds", key))
  if (file.exists(f)) return(readRDS(f))
  x <- nflreadr::load_schedules(seasons)
  saveRDS(x, f); x
}
