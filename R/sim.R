# Build inputs per game (handles early-week prior-year PBP)
prepare_game_inputs <- function(season_target, week, home_team, away_team) {
  if (!is.na(week) && week < 4) {
    # Blend last season + current season until Week 4
    pbp_prev <- load_pbp_cached(season_target - 1) %>%
      dplyr::mutate(week = as.numeric(week)) %>%
      dplyr::filter(week != 18)

    pbp_curr <- load_pbp_cached(season_target) %>%
      dplyr::mutate(week = as.numeric(week)) %>%
      dplyr::filter(week != 18)

    pbp <- dplyr::bind_rows(pbp_prev, pbp_curr)
  } else {
    # Week >= 4 → just current season
    pbp <- load_pbp_cached(season_target) %>%
      dplyr::mutate(week = as.numeric(week)) %>%
      dplyr::filter(week != 18)
  }

  # metrics based on current season week
  metrics_week <- week

  t1 <- calculate_team_metrics(home_team, season_target, metrics_week, pbp)
  t2 <- calculate_team_metrics(away_team, season_target, metrics_week, pbp)

  list(
    pbp = pbp,
    season_use = season_target,
    week_use = metrics_week,
    t1 = t1,
    t2 = t2
  )
}


# Simulate scores matrix using a given projector function
simulate_game_matrix <- function(projector, home, away, season_use, week_use, pbp, t1, t2, sims) {
  draws <- replicate(sims, {
    s <- projector(home, away, season_use, week_use, pbp, t1, t2)
    c(home = s["projected_score_team1"], away = s["projected_score_team2"])
  })
  if (is.null(colnames(draws))) colnames(draws) <- paste0("sim", seq_len(ncol(draws)))
  if (is.null(rownames(draws))) rownames(draws) <- c("home","away")
  draws
}

summarize_sims_basic <- function(home, away) {
  spread <- home - away
  total  <- home + away
  tibble::tibble(
    home_mean   = mean(home, na.rm = TRUE),
    away_mean   = mean(away, na.rm = TRUE),
    spread_mean = mean(spread, na.rm = TRUE),
    total_mean  = mean(total,  na.rm = TRUE),
    home_sd     = sd(home, na.rm = TRUE),
    away_sd     = sd(away, na.rm = TRUE),
    spread_sd   = sd(spread, na.rm = TRUE),
    total_sd    = sd(total,  na.rm = TRUE),
    na_rate_home   = mean(!is.finite(home)),
    na_rate_away   = mean(!is.finite(away)),
    na_rate_spread = mean(!is.finite(spread)),
    na_rate_total  = mean(!is.finite(total))
  )
}
