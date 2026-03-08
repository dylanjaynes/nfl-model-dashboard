# R/compare_configs.R

compare_configs <- function(season_target, weeks, sims = 2000, w_eckel = 0.6) {
  configs <- list(
    conservative = list(tau_spread = 12, sigma_spread = 6.5, w_hybrid = 0.5),
    cons_less_hybrid = list(tau_spread = 12, sigma_spread = 6.5, w_hybrid = 0.25),
    cons_little_hybrid = list(tau_spread = 12, sigma_spread = 6.5, w_hybrid = 0.1)
  )

  results <- purrr::imap_dfr(configs, function(params, name) {
    wk_results <- purrr::map_dfr(weeks, function(w) {
      wk <- run_week(
        season_target = season_target,
        week = w,
        sims = sims,
        w_eckel = w_eckel,
        targets = learn_target_sds(2018:2024),
        tau_spread = params$tau_spread,
        sigma_spread = params$sigma_spread,
        w_hybrid = params$w_hybrid
      )
      wk %>%
        mutate(config = name, week = w,
               ATS_edge_110 = ATS_prob - 0.524)  # 52.4% break-even vs -110
    })
    wk_results
  })

  # summary stats
  summary <- results %>%
    group_by(config) %>%
    summarise(
      n_games = n(),
      mean_prob = mean(ATS_prob, na.rm = TRUE),
      mean_edge_110 = mean(ATS_edge_110, na.rm = TRUE),
      pct_favorites = mean(ATS_pick == home_team, na.rm = TRUE),
      `edges_≥1pp` = sum(ATS_edge_110 >= 0.01, na.rm = TRUE),
      `edges_≥2pp` = sum(ATS_edge_110 >= 0.02, na.rm = TRUE),
      `edges_≥3pp` = sum(ATS_edge_110 >= 0.03, na.rm = TRUE),
      `edges_≥5pp` = sum(ATS_edge_110 >= 0.05, na.rm = TRUE),
      .groups = "drop"
    )

  top_plays <- results %>%
    arrange(config, desc(ATS_edge_110)) %>%
    group_by(config) %>%
    slice_head(n = 10) %>%
    ungroup()

  list(summary = summary, top_plays = top_plays, raw = results)
}
