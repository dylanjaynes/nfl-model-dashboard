#projectors.R
# ---- PBP projector (your YPP-style) ----
project_matchup <- function(team1_abbr, team2_abbr, season, week, pbp, team1_metrics, team2_metrics) {

  # noises
  ypc_noise_1 <- rnorm(1, 0, team2_metrics$defense_sd$sd_ypc_diff_allowed)
  ypa_noise_1 <- rnorm(1, 0, team2_metrics$defense_sd$sd_ypa_diff_allowed)
  rush_att_noise_1 <- rnorm(1, 0, team1_metrics$attempt_summary$sd_rush_attempts_per_game)
  pass_att_noise_1 <- rnorm(1, 0, team1_metrics$attempt_summary$sd_pass_attempts_per_game)
  ypc_off_noise_1 <- rnorm(1, 0, team1_metrics$offense_sd$sd_ypc_diff)
  ypa_off_noise_1 <- rnorm(1, 0, team1_metrics$offense_sd$sd_ypa_diff)

  ypc_noise_2 <- rnorm(1, 0, team1_metrics$defense_sd$sd_ypc_diff_allowed)
  ypa_noise_2 <- rnorm(1, 0, team1_metrics$defense_sd$sd_ypa_diff_allowed)
  rush_att_noise_2 <- rnorm(1, 0, team2_metrics$attempt_summary$sd_rush_attempts_per_game)
  pass_att_noise_2 <- rnorm(1, 0, team2_metrics$attempt_summary$sd_pass_attempts_per_game)
  ypc_off_noise_2 <- rnorm(1, 0, team2_metrics$offense_sd$sd_ypc_diff)
  ypa_off_noise_2 <- rnorm(1, 0, team2_metrics$offense_sd$sd_ypa_diff)

  # projections
  projected_rushing_ypc_team1 <- (mean(team1_metrics$offense_vs_defense$offensive_ypc, na.rm = TRUE) + ypc_off_noise_1) *
    (1 + mean(team2_metrics$defense_vs_offense$ypc_diff_allowed, na.rm = TRUE) + ypc_noise_1)

  projected_passing_ypa_team1 <- (mean(team1_metrics$offense_vs_defense$offensive_ypa, na.rm = TRUE) + ypa_off_noise_1) *
    (1 + mean(team2_metrics$defense_vs_offense$ypa_diff_allowed, na.rm = TRUE) + ypa_noise_1)

  projected_rushing_ypc_team2 <- (mean(team2_metrics$offense_vs_defense$offensive_ypc, na.rm = TRUE) + ypc_off_noise_2) *
    (1 + mean(team1_metrics$defense_vs_offense$ypc_diff_allowed, na.rm = TRUE) + ypc_noise_2)

  projected_passing_ypa_team2 <- (mean(team2_metrics$offense_vs_defense$offensive_ypa, na.rm = TRUE) + ypa_off_noise_2) *
    (1 + mean(team1_metrics$defense_vs_offense$ypa_diff_allowed, na.rm = TRUE) + ypa_noise_2)

  team1_ypp <- mean(c(team1_metrics$offense_ypp$yards_per_point_offense,
                      team2_metrics$defense_ypp$yards_per_point_defense), na.rm = TRUE)
  team2_ypp <- mean(c(team2_metrics$offense_ypp$yards_per_point_offense,
                      team1_metrics$defense_ypp$yards_per_point_defense), na.rm = TRUE)

  projected_rushing_yards_team1 <- projected_rushing_ypc_team1 * (team1_metrics$attempt_summary$avg_rush_attempts_per_game + rush_att_noise_1)
  projected_passing_yards_team1 <- projected_passing_ypa_team1 * (team1_metrics$attempt_summary$avg_pass_attempts_per_game + pass_att_noise_1)
  projected_total_yards_team1   <- projected_rushing_yards_team1 + projected_passing_yards_team1

  projected_rushing_yards_team2 <- projected_rushing_ypc_team2 * (team2_metrics$attempt_summary$avg_rush_attempts_per_game + rush_att_noise_2)
  projected_passing_yards_team2 <- projected_passing_ypa_team2 * (team2_metrics$attempt_summary$avg_pass_attempts_per_game + pass_att_noise_2)
  projected_total_yards_team2   <- projected_rushing_yards_team2 + projected_passing_yards_team2

  projected_score_team1 <- (projected_total_yards_team1 / team1_ypp)
  projected_score_team2 <- (projected_total_yards_team2 / team2_ypp)

  c(projected_score_team1 = projected_score_team1,
    projected_score_team2 = projected_score_team2)
}

# ---- Eckel projector (NA-safe) ----
project_matchup_eckel <- function(team1_abbr, team2_abbr, season, week, pbp, team1_metrics, team2_metrics) {

  drives_per_game <- round(rnorm(1, mean = 11, sd = 0.75))
  drives_per_game <- min(max(drives_per_game, 8), 14)

  # truncated draws
  sim_eckel_rate_off_team1 <- tn1(0, .85,
                                  team1_metrics$offense_average_game$average_eckel_rate,
                                  team1_metrics$eckel_off_sd$sd_eckel_rate_diff)
  sim_eckel_rate_off_team2 <- tn1(0, .85,
                                  team2_metrics$offense_average_game$average_eckel_rate,
                                  team2_metrics$eckel_off_sd$sd_eckel_rate_diff)
  sim_eckel_rate_def_team1 <- tn1(0, .85,
                                  team1_metrics$defense_average_game$average_eckel_rate,
                                  team1_metrics$eckel_def_sd$sd_eckel_allowed_diff)
  sim_eckel_rate_def_team2 <- tn1(0, .85,
                                  team2_metrics$defense_average_game$average_eckel_rate,
                                  team2_metrics$eckel_def_sd$sd_eckel_allowed_diff)

  sim_point_eckel_off_team1 <- tn1(2, 5,
                                   team1_metrics$offense_average_game$average_points_per_eckel,
                                   team1_metrics$eckel_off_sd$sd_points_per_eckel_diff)
  sim_point_eckel_off_team2 <- tn1(2, 5,
                                   team2_metrics$offense_average_game$average_points_per_eckel,
                                   team2_metrics$eckel_off_sd$sd_points_per_eckel_diff)
  sim_point_eckel_def_team1 <- tn1(2, 5,
                                   team1_metrics$defense_average_game$average_points_per_eckel,
                                   team1_metrics$eckel_def_sd$sd_points_per_eckel_allow_diff)
  sim_point_eckel_def_team2 <- tn1(2, 5,
                                   team2_metrics$defense_average_game$average_points_per_eckel,
                                   team2_metrics$eckel_def_sd$sd_points_per_eckel_allow_diff)

  rn <- function(s) { s <- sd_safe(s); if (s == 1e-3) 0 else rnorm(1, 0, s) }

  points_noise_off_1   <- rn(team1_metrics$eckel_off_sd$sd_points_per_eckel_diff)
  field_noise_off_1    <- rn(team1_metrics$eckel_off_sd$sd_field_position_diff)
  turnover_noise_off_1 <- rn(team1_metrics$eckel_off_sd$sd_turnover_rate_diff)

  points_noise_allow_1   <- rn(team1_metrics$eckel_def_sd$sd_points_per_eckel_allow_diff)
  field_noise_allow_1    <- rn(team1_metrics$eckel_def_sd$sd_field_position_allow_diff)
  turnover_noise_allow_1 <- rn(team1_metrics$eckel_def_sd$sd_turnover_rate_def_diff)

  points_noise_off_2   <- rn(team2_metrics$eckel_off_sd$sd_points_per_eckel_diff)
  field_noise_off_2    <- rn(team2_metrics$eckel_off_sd$sd_field_position_diff)
  turnover_noise_off_2 <- rn(team2_metrics$eckel_off_sd$sd_turnover_rate_diff)

  points_noise_allow_2   <- rn(team2_metrics$eckel_def_sd$sd_points_per_eckel_allow_diff)
  field_noise_allow_2    <- rn(team2_metrics$eckel_def_sd$sd_field_position_allow_diff)
  turnover_noise_allow_2 <- rn(team2_metrics$eckel_def_sd$sd_turnover_rate_def_diff)

  team1_eckel_rate <- (
    (1 + nz_mean(team2_metrics$eckel_vs_offense$eckel_rate_allowed_diff)) * sim_eckel_rate_off_team1 +
      (1 + nz_mean(team1_metrics$eckel_vs_defense$eckel_rate_diff))        * sim_eckel_rate_def_team2
  ) / 2

  team1_points_per_eckel <- (
    (1 + nz_mean(team1_metrics$eckel_vs_defense$points_per_eckel_diff))      * sim_point_eckel_off_team1 +
      (1 + nz_mean(team2_metrics$eckel_vs_offense$points_per_eckel_allowed_diff)) * sim_point_eckel_def_team2
  ) / 2

  team2_eckel_rate <- (
    (1 + nz_mean(team1_metrics$eckel_vs_offense$eckel_rate_allowed_diff)) * sim_eckel_rate_off_team2 +
      (1 + nz_mean(team2_metrics$eckel_vs_defense$eckel_rate_diff))       * sim_eckel_rate_def_team1
  ) / 2

  team2_points_per_eckel <- (
    (1 + nz_mean(team2_metrics$eckel_vs_defense$points_per_eckel_diff))        * sim_point_eckel_off_team2 +
      (1 + nz_mean(team1_metrics$eckel_vs_offense$points_per_eckel_allowed_diff)) * sim_point_eckel_def_team1
  ) / 2

  projected_score_team1 <- drives_per_game * team1_eckel_rate * team1_points_per_eckel
  projected_score_team2 <- drives_per_game * team2_eckel_rate * team2_points_per_eckel

  c(projected_score_team1 = projected_score_team1,
    projected_score_team2 = projected_score_team2)
}

# ---- PBP PROJECTOR (VECTORIZED) ----
project_matchup_vec <- function(team1_abbr, team2_abbr,
                                season, week, pbp,
                                t1, t2,
                                sims) {

  # ---- all noise draws vectorized ----
  ypc_noise_1      <- rnorm(sims, 0, t2$defense_sd$sd_ypc_diff_allowed)
  ypa_noise_1      <- rnorm(sims, 0, t2$defense_sd$sd_ypa_diff_allowed)
  rush_att_noise_1 <- rnorm(sims, 0, t1$attempt_summary$sd_rush_attempts_per_game)
  pass_att_noise_1 <- rnorm(sims, 0, t1$attempt_summary$sd_pass_attempts_per_game)
  ypc_off_noise_1  <- rnorm(sims, 0, t1$offense_sd$sd_ypc_diff)
  ypa_off_noise_1  <- rnorm(sims, 0, t1$offense_sd$sd_ypa_diff)

  ypc_noise_2      <- rnorm(sims, 0, t1$defense_sd$sd_ypc_diff_allowed)
  ypa_noise_2      <- rnorm(sims, 0, t1$defense_sd$sd_ypa_diff_allowed)
  rush_att_noise_2 <- rnorm(sims, 0, t2$attempt_summary$sd_rush_attempts_per_game)
  pass_att_noise_2 <- rnorm(sims, 0, t2$attempt_summary$sd_pass_attempts_per_game)
  ypc_off_noise_2  <- rnorm(sims, 0, t2$offense_sd$sd_ypc_diff)
  ypa_off_noise_2  <- rnorm(sims, 0, t2$offense_sd$sd_ypa_diff)

  # ---- constants (scalar means) ----
  t1_ypc <- mean(t1$offense_vs_defense$offensive_ypc, na.rm = TRUE)
  t2_ypc <- mean(t2$offense_vs_defense$offensive_ypc, na.rm = TRUE)
  t1_ypa <- mean(t1$offense_vs_defense$offensive_ypa, na.rm = TRUE)
  t2_ypa <- mean(t2$offense_vs_defense$offensive_ypa, na.rm = TRUE)

  t1_def_ypc <- mean(t2$defense_vs_offense$ypc_diff_allowed, na.rm = TRUE)
  t2_def_ypc <- mean(t1$defense_vs_offense$ypc_diff_allowed, na.rm = TRUE)

  t1_def_ypa <- mean(t2$defense_vs_offense$ypa_diff_allowed, na.rm = TRUE)
  t2_def_ypa <- mean(t1$defense_vs_offense$ypa_diff_allowed, na.rm = TRUE)

  # ---- YPC/ YPA projections (vectorized) ----
  proj_ypc_1 <- (t1_ypc + ypc_off_noise_1) * (1 + t1_def_ypc + ypc_noise_1)
  proj_ypa_1 <- (t1_ypa + ypa_off_noise_1) * (1 + t1_def_ypa + ypa_noise_1)

  proj_ypc_2 <- (t2_ypc + ypc_off_noise_2) * (1 + t2_def_ypc + ypc_noise_2)
  proj_ypa_2 <- (t2_ypa + ypa_off_noise_2) * (1 + t2_def_ypa + ypa_noise_2)

  # ---- yards-per-point constants ----
  ypp1 <- mean(c(t1$offense_ypp$yards_per_point_offense,
                 t2$defense_ypp$yards_per_point_defense), na.rm = TRUE)
  ypp2 <- mean(c(t2$offense_ypp$yards_per_point_offense,
                 t1$defense_ypp$yards_per_point_defense), na.rm = TRUE)

  # ---- attempts (scalars) ----
  ra1 <- t1$attempt_summary$avg_rush_attempts_per_game
  pa1 <- t1$attempt_summary$avg_pass_attempts_per_game
  ra2 <- t2$attempt_summary$avg_rush_attempts_per_game
  pa2 <- t2$attempt_summary$avg_pass_attempts_per_game

  # ---- projected yards (vectorized) ----
  rush_yds_1 <- proj_ypc_1 * (ra1 + rush_att_noise_1)
  pass_yds_1 <- proj_ypa_1 * (pa1 + pass_att_noise_1)
  tot_yds_1  <- rush_yds_1 + pass_yds_1

  rush_yds_2 <- proj_ypc_2 * (ra2 + rush_att_noise_2)
  pass_yds_2 <- proj_ypa_2 * (pa2 + pass_att_noise_2)
  tot_yds_2  <- rush_yds_2 + pass_yds_2

  # ---- final scoring (vectorized) ----
  score1 <- (tot_yds_1 / ypp1)
  score2 <- (tot_yds_2 / ypp2)

  list(home = score1, away = score2)
}

# ---- Eckel PROJECTOR (VECTORIZED) ----
project_matchup_eckel_vec <- function(team1_abbr, team2_abbr,
                                      season, week, pbp,
                                      t1, t2,
                                      sims) {

  # ---- drives per game, drawn once (scalar) ----
  dpg <- round(rnorm(1, 11, 0.75))
  dpg <- min(max(dpg, 8), 14)

  # ----- vector-safe truncated normal -----
  tn_vec <- function(a, b, mean, sd, n) {
    sd <- sd_safe(sd)
    if (!is.finite(sd) || sd == 0) sd <- 1e-6
    truncnorm::rtruncnorm(n, a = a, b = b, mean = mean, sd = sd)
  }

  # ---- vector draws ----
  er_off_1 <- tn_vec(0, .85, t1$offense_average_game$average_eckel_rate,
                     t1$eckel_off_sd$sd_eckel_rate_diff, sims)
  er_off_2 <- tn_vec(0, .85, t2$offense_average_game$average_eckel_rate,
                     t2$eckel_off_sd$sd_eckel_rate_diff, sims)

  er_def_1 <- tn_vec(0, .85, t1$defense_average_game$average_eckel_rate,
                     t1$eckel_def_sd$sd_eckel_allowed_diff, sims)
  er_def_2 <- tn_vec(0, .85, t2$defense_average_game$average_eckel_rate,
                     t2$eckel_def_sd$sd_eckel_allowed_diff, sims)

  ppe_off_1 <- tn_vec(2, 5, t1$offense_average_game$average_points_per_eckel,
                      t1$eckel_off_sd$sd_points_per_eckel_diff, sims)
  ppe_off_2 <- tn_vec(2, 5, t2$offense_average_game$average_points_per_eckel,
                      t2$eckel_off_sd$sd_points_per_eckel_diff, sims)

  ppe_def_1 <- tn_vec(2, 5, t1$defense_average_game$average_points_per_eckel,
                      t1$eckel_def_sd$sd_points_per_eckel_allow_diff, sims)
  ppe_def_2 <- tn_vec(2, 5, t2$defense_average_game$average_points_per_eckel,
                      t2$eckel_def_sd$sd_points_per_eckel_allow_diff, sims)

  # ---- combine effects (vectorized) ----
  team1_eckel_rate <- (
    (1 + nz_mean(t2$eckel_vs_offense$eckel_rate_allowed_diff)) * er_off_1 +
      (1 + nz_mean(t1$eckel_vs_defense$eckel_rate_diff))         * er_def_2
  ) / 2

  team2_eckel_rate <- (
    (1 + nz_mean(t1$eckel_vs_offense$eckel_rate_allowed_diff)) * er_off_2 +
      (1 + nz_mean(t2$eckel_vs_defense$eckel_rate_diff))         * er_def_1
  ) / 2

  team1_points <- (
    (1 + nz_mean(t1$eckel_vs_defense$points_per_eckel_diff))          * ppe_off_1 +
      (1 + nz_mean(t2$eckel_vs_offense$points_per_eckel_allowed_diff))  * ppe_def_2
  ) / 2

  team2_points <- (
    (1 + nz_mean(t2$eckel_vs_defense$points_per_eckel_diff))          * ppe_off_2 +
      (1 + nz_mean(t1$eckel_vs_offense$points_per_eckel_allowed_diff))  * ppe_def_1
  ) / 2

  # ---- FINAL SCORES (VECTORIZED) ----
  score1 <- dpg * team1_eckel_rate * team1_points
  score2 <- dpg * team2_eckel_rate * team2_points

  list(home = score1, away = score2)
}
