# ---- small helpers ----
safe_div <- function(a, b, fallback = NA_real_) {
  if (!is.finite(a) || !is.finite(b) || b == 0) return(fallback)
  a / b
}

# Always give rtruncnorm a sane scalar
sd_safe <- function(s) {
  s <- as.numeric(s)
  if (!length(s) || !is.finite(s) || s <= 0) return(1e-3)
  s
}

nz_mean <- function(x) {
  x <- as.numeric(x)
  if (!length(x) || all(!is.finite(x))) return(0)
  mean(x, na.rm = TRUE)
}

tn1 <- function(a, b, mean, sd) {
  m <- as.numeric(mean)
  if (!length(m) || !is.finite(m)) m <- 0

  s <- sd_safe(sd)

  truncnorm::rtruncnorm(
    1,
    a = a,
    b = b,
    mean = m,
    sd  = s
  )
}


# percentiles + “chance to hit” (>= each percentile and in IQR)
pct_summary <- function(x, prefix) {
  x <- as.numeric(x)
  x <- x[is.finite(x)]
  if (!length(x)) {
    nm <- c("_p25","_p50","_p75","_prob_ge_p25","_prob_ge_p50","_prob_ge_p75","_prob_in_IQR")
    out <- setNames(as.list(rep(NA_real_, length(nm))), paste0(prefix, nm))
    return(out)
  }
  q <- as.numeric(stats::quantile(x, probs = c(.25, .5, .75), na.rm = TRUE, names = FALSE))
  names(q) <- paste0(prefix, c("_p25","_p50","_p75"))
  p_ge <- setNames(c(mean(x >= q[1]), mean(x >= q[2]), mean(x >= q[3])),
                   paste0(prefix, c("_prob_ge_p25","_prob_ge_p50","_prob_ge_p75")))
  iqr <- setNames(mean(x >= q[1] & x <= q[3]), paste0(prefix, "_prob_in_IQR"))
  c(as.list(q), as.list(p_ge), as.list(iqr))
}


# R/targets_cache.R

# cache global for target SDs
.targets_cache <- NULL

get_targets_cache <- function(seasons = 2018:2024, refresh = FALSE) {
  if (refresh || is.null(.targets_cache)) {
    message("Learning target SDs across seasons: ", paste(seasons, collapse = ", "))
    .targets_cache <<- learn_target_sds(seasons)
  }
  .targets_cache
}


.pbp_cache_env <- new.env(parent = emptyenv())

get_pbp_cached <- function(season,
                           through_week = Inf,          # e.g. 6 for Wk6; Inf for prior seasons
                           cache_dir = "cache/pbp",
                           refresh = c("auto","force","never")) {
  refresh <- match.arg(refresh)
  if (!dir.exists(cache_dir)) dir.create(cache_dir, recursive = TRUE)
  path <- file.path(cache_dir, sprintf("pbp_%d.rds", season))

  need_fetch <- TRUE
  meta <- list(max_week = NA_integer_, fetched_at = NA)

  if (file.exists(path) && refresh != "force") {
    obj <- readRDS(path)
    pbp <- obj$pbp
    if (!is.null(obj$meta)) meta <- obj$meta
    cached_max <- suppressWarnings(max(as.numeric(pbp$week), na.rm = TRUE))
    # auto-refresh if we need later weeks than the cache has
    need_fetch <- (refresh == "auto" && is.finite(through_week) && cached_max < through_week)
    if (!need_fetch) {
      return(
        pbp |>
          dplyr::mutate(week = as.numeric(week)) |>
          dplyr::filter(week != 18, week <= through_week)
      )
    }
  }

  if (need_fetch) {
    message(sprintf("Loading nflfastR PBP for season %d ...", season))
    pbp <- nflfastR::load_pbp(season)
    meta <- list(max_week = suppressWarnings(max(as.numeric(pbp$week), na.rm = TRUE)),
                 fetched_at = Sys.time())
    saveRDS(list(pbp = pbp, meta = meta), path)
    return(
      pbp |>
        dplyr::mutate(week = as.numeric(week)) |>
        dplyr::filter(week != 18, week <= through_week)
    )
  }
}


