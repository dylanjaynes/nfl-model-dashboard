#' @import magrittr
#' @import dplyr
NULL


# Learn historical target SDs of (margin - spread_line) and (total - total_line)
learn_target_sds <- function(seasons = 2018:2024) {
  sched <- load_schedules_cached(seasons) %>%
    dplyr::mutate(week = as.numeric(week)) %>%
    dplyr::filter(!is.na(home_score), !is.na(away_score), !is.na(spread_line), !is.na(total_line)) %>%
    dplyr::transmute(
      season, week,
      margin = home_score - away_score,
      total  = home_score + away_score,
      spread_line = spread_line,
      total_line  = total_line,
      spread_error = margin - spread_line,
      total_error  = total - total_line
    )

  trim_sd <- function(x, trim = 0.02) {
    q <- stats::quantile(x, c(trim, 1 - trim), na.rm = TRUE)
    stats::sd(x[x >= q[1] & x <= q[2]], na.rm = TRUE)
  }

  by_week <- sched %>%
    dplyr::group_by(week) %>%
    dplyr::summarise(
      spread_sd_target = trim_sd(spread_error),
      total_sd_target  = trim_sd(total_error),
      .groups = "drop"
    )

  overall <- dplyr::summarise(sched,
                              spread_sd_overall = trim_sd(spread_error),
                              total_sd_overall  = trim_sd(total_error))

  list(by_week = by_week, overall = overall)
}

# Add zero-mean noise to match target SDs (per week)
calibrate_distributions <- function(spread, total, week, targets) {
  wk_row <- targets$by_week %>% dplyr::filter(week == !!week)
  s_target_spread <- if (nrow(wk_row) && is.finite(wk_row$spread_sd_target)) wk_row$spread_sd_target else targets$overall$spread_sd_overall
  s_target_total  <- if (nrow(wk_row) && is.finite(wk_row$total_sd_target))  wk_row$total_sd_target  else targets$overall$total_sd_overall

  s_model_spread <- sd(spread, na.rm = TRUE)
  s_model_total  <- sd(total,  na.rm = TRUE)

  s_extra_spread <- sqrt(pmax(0, s_target_spread^2 - s_model_spread^2))
  s_extra_total  <- sqrt(pmax(0, s_target_total^2  - s_model_total^2))

  spread_cal <- spread + rnorm(length(spread), 0, s_extra_spread)
  total_cal  <- total  + rnorm(length(total),  0, s_extra_total)

  list(spread = spread_cal, total = total_cal,
       s_target_spread = s_target_spread, s_target_total = s_target_total,
       s_model_spread = s_model_spread, s_model_total = s_model_total)
}
