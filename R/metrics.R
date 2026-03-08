calculate_team_metrics <- function(team_abbr, season, week, pbp) {

  # local helper
  safe_div <- function(a, b) ifelse(is.finite(a) & is.finite(b) & abs(b) > 0, a / b, NA_real_)

  # only use data up to the requested week
  pbp <- pbp %>% dplyr::filter(week <= !!week)

  ## -------------------- PBP Yards-based side --------------------
  # OFFENSE weekly by opponent
  offense_stats <- pbp %>%
    dplyr::filter(posteam == team_abbr, (rush_attempt == 1 | pass_attempt == 1), !is.na(yards_gained)) %>%
    dplyr::group_by(week, defteam) %>%
    dplyr::summarize(
      rush_yards = sum(yards_gained[rush_attempt == 1], na.rm = TRUE),
      rush_attempts = sum(rush_attempt, na.rm = TRUE),
      pass_yards = sum(yards_gained[pass_attempt == 1], na.rm = TRUE),
      pass_attempts = sum(pass_attempt, na.rm = TRUE),
      offensive_ypc = rush_yards / pmax(rush_attempts, 1),
      offensive_ypa = (sum(yards_gained[pass_attempt == 1], na.rm = TRUE) +
                         20 * sum(pass_touchdown == 1, na.rm = TRUE) -
                         45 * sum(interception == 1, na.rm = TRUE) -
                         sum(yards_gained[sack == 1], na.rm = TRUE)) /
        pmax(sum(pass_attempt == 1, na.rm = TRUE) + sum(sack == 1, na.rm = TRUE), 1),
      .groups = "drop"
    )

  attempts_per_game <- pbp %>%
    dplyr::filter(posteam == team_abbr, rush_attempt == 1 | pass_attempt == 1, !is.na(rush_attempt)) %>%
    dplyr::group_by(game_id, week) %>%
    dplyr::summarize(
      rush_attempts = sum(rush_attempt, na.rm = TRUE),
      pass_attempts = sum(pass_attempt, na.rm = TRUE),
      .groups = "drop"
    )

  attempt_summary <- attempts_per_game %>%
    dplyr::summarize(
      total_games = dplyr::n(),
      avg_rush_attempts_per_game = mean(rush_attempts, na.rm = TRUE),
      avg_pass_attempts_per_game = mean(pass_attempts, na.rm = TRUE),
      sd_rush_attempts_per_game = if (dplyr::n() > 1) stats::sd(rush_attempts, na.rm = TRUE) else 0,
      sd_pass_attempts_per_game = if (dplyr::n() > 1) stats::sd(pass_attempts, na.rm = TRUE) else 0
    )

  # yards-per-point (offense & defense)
  offense_ypp <- pbp %>%
    dplyr::filter(posteam == team_abbr) %>%
    dplyr::group_by(game_id) %>%
    dplyr::summarize(total_offensive_yards = sum(yards_gained, na.rm = TRUE),
                     points_in_game = max(posteam_score, na.rm = TRUE), .groups = "drop") %>%
    dplyr::summarize(total_points_season = sum(points_in_game, na.rm = TRUE),
                     total_yards_season = sum(total_offensive_yards, na.rm = TRUE),
                     yards_per_point_offense = safe_div(total_yards_season, total_points_season))

  defense_ypp <- pbp %>%
    dplyr::filter(defteam == team_abbr) %>%
    dplyr::group_by(game_id) %>%
    dplyr::summarize(total_offensive_yards_allowed = sum(yards_gained, na.rm = TRUE),
                     points_in_game_allowed = max(posteam_score, na.rm = TRUE), .groups = "drop") %>%
    dplyr::summarize(total_points_season_allowed = sum(points_in_game_allowed, na.rm = TRUE),
                     total_yards_season_allowed = sum(total_offensive_yards_allowed, na.rm = TRUE),
                     yards_per_point_defense = safe_div(total_yards_season_allowed, total_points_season_allowed))

  # league averages (for diffs)
  defense_averages <- pbp %>%
    dplyr::filter(defteam %in% offense_stats$defteam,
                  rush_attempt == 1 | pass_attempt == 1, !is.na(yards_gained)) %>%
    dplyr::group_by(defteam) %>%
    dplyr::summarize(
      avg_def_ypc = mean(yards_gained[rush_attempt == 1], na.rm = TRUE),
      avg_def_ypa = (sum(yards_gained[pass_attempt == 1], na.rm = TRUE) +
                       20 * sum(pass_touchdown == 1, na.rm = TRUE) -
                       45 * sum(interception == 1, na.rm = TRUE) -
                       sum(yards_gained[sack == 1], na.rm = TRUE)) /
        pmax(sum(pass_attempt == 1, na.rm = TRUE) + sum(sack == 1, na.rm = TRUE), 1),
      .groups = "drop"
    )

  offense_vs_defense <- offense_stats %>%
    dplyr::left_join(defense_averages, by = "defteam") %>%
    dplyr::mutate(
      ypc_diff = (offensive_ypc - avg_def_ypc) / pmax(avg_def_ypc, 1e-6),
      ypa_diff = (offensive_ypa - avg_def_ypa) / pmax(avg_def_ypa, 1e-6)
    )

  defense_stats <- pbp %>%
    dplyr::filter(defteam == team_abbr, (rush_attempt == 1 | pass_attempt == 1), !is.na(yards_gained)) %>%
    dplyr::group_by(week, posteam) %>%
    dplyr::summarize(
      rush_yards_allowed = sum(yards_gained[rush_attempt == 1], na.rm = TRUE),
      rush_attempts_faced = sum(rush_attempt, na.rm = TRUE),
      sack_yards = sum(yards_gained[sack == 1], na.rm = TRUE),
      pass_yards_allowed = sum(yards_gained[pass_attempt == 1], na.rm = TRUE) + sack_yards,
      pass_attempts_faced = sum(pass_attempt, na.rm = TRUE),
      defensive_ypa = (sum(yards_gained[pass_attempt == 1], na.rm = TRUE) +
                         20 * sum(pass_touchdown == 1, na.rm = TRUE) -
                         45 * sum(interception == 1, na.rm = TRUE) -
                         sum(yards_gained[sack == 1], na.rm = TRUE)) /
        pmax(sum(pass_attempt == 1, na.rm = TRUE) + sum(sack == 1, na.rm = TRUE), 1),
      defensive_ypc = rush_yards_allowed / pmax(rush_attempts_faced, 1),
      .groups = "drop"
    )

  offense_averages <- pbp %>%
    dplyr::filter(posteam %in% defense_stats$posteam,
                  rush_attempt == 1 | pass_attempt == 1, !is.na(yards_gained)) %>%
    dplyr::group_by(posteam) %>%
    dplyr::summarize(
      avg_off_ypc = mean(yards_gained[rush_attempt == 1], na.rm = TRUE),
      avg_off_ypa = (sum(yards_gained[pass_attempt == 1], na.rm = TRUE) +
                       20 * sum(pass_touchdown == 1, na.rm = TRUE) -
                       45 * sum(interception == 1, na.rm = TRUE) -
                       sum(yards_gained[sack == 1], na.rm = TRUE)) /
        pmax(sum(pass_attempt == 1, na.rm = TRUE) + sum(sack == 1, na.rm = TRUE), 1),
      .groups = "drop"
    )

  defense_vs_offense <- defense_stats %>%
    dplyr::left_join(offense_averages, by = "posteam") %>%
    dplyr::mutate(
      ypc_diff_allowed = (defensive_ypc - avg_off_ypc) / pmax(avg_off_ypc, 1e-6),
      ypa_diff_allowed = (defensive_ypa - avg_off_ypa) / pmax(avg_off_ypa, 1e-6)
    )

  # variability (SDs for noise)
  offense_sd <- offense_vs_defense %>%
    dplyr::summarize(
      sd_ypc_diff = if (length(stats::na.omit(ypc_diff)) > 1) stats::sd(ypc_diff, na.rm = TRUE) else 0,
      sd_ypa_diff = if (length(stats::na.omit(ypa_diff)) > 1) stats::sd(ypa_diff, na.rm = TRUE) else 0
    )
  defense_sd <- defense_vs_offense %>%
    dplyr::summarize(
      sd_ypc_diff_allowed = if (length(stats::na.omit(ypc_diff_allowed)) > 1) stats::sd(ypc_diff_allowed, na.rm = TRUE) else 0,
      sd_ypa_diff_allowed = if (length(stats::na.omit(ypa_diff_allowed)) > 1) stats::sd(ypa_diff_allowed, na.rm = TRUE) else 0
    )

  ## -------------------- Drive-based (Eckel) side --------------------
  # OFFENSIVE DRIVES for this team
  offensive_drives_team <- pbp %>%
    dplyr::filter(posteam == team_abbr, (kickoff_attempt == 0 & punt_attempt == 0)) %>%
    dplyr::group_by(game_id, drive) %>%
    dplyr::summarize(
      posteam = dplyr::first(posteam),
      defteam = dplyr::first(defteam),
      week = dplyr::first(week),
      start_yardline = dplyr::first(yardline_100),
      fumble = any(fumble_lost == 1, na.rm = TRUE),
      interception = any(interception == 1, na.rm = TRUE),
      score_gained = max(posteam_score_post, na.rm = TRUE) - min(posteam_score_post, na.rm = TRUE),
      eckel = any((yardline_100 <= 40 & first_down == 1) | touchdown == 1),
      .groups = "drop"
    ) %>%
    dplyr::mutate(turnover = ifelse(fumble | interception, 1, 0))

  offensive_drive_summary <- offensive_drives_team %>%
    dplyr::group_by(week, defteam) %>%
    dplyr::summarize(
      total_drives = dplyr::n(),
      eckel_drives = sum(eckel, na.rm = TRUE),
      total_points = sum(score_gained, na.rm = TRUE),
      turnovers = sum(turnover, na.rm = TRUE),
      avg_start_yardline = mean(start_yardline, na.rm = TRUE),
      eckel_rate = safe_div(eckel_drives, total_drives),
      points_per_eckel = ifelse(eckel_drives > 0, total_points / eckel_drives, 0),
      turnover_rate = safe_div(turnovers, total_drives),
      .groups = "drop"
    )

  offense_average_game <- offensive_drive_summary %>%
    dplyr::summarize(
      avg_start_yardline = mean(avg_start_yardline, na.rm = TRUE),
      average_eckel_rate = mean(eckel_rate, na.rm = TRUE),
      average_points_per_eckel = mean(points_per_eckel, na.rm = TRUE),
      average_turnover_rate = mean(turnover_rate, na.rm = TRUE),
      .groups = "drop"
    )

  # LEAGUE DEFENSIVE DRIVES (for opponent baselines)
  defensive_drives_league <- pbp %>%
    dplyr::filter(!is.na(posteam), !is.na(defteam), (kickoff_attempt == 0 & punt_attempt == 0)) %>%
    dplyr::group_by(game_id, drive, defteam) %>%
    dplyr::summarize(
      posteam = dplyr::first(posteam),
      defteam = dplyr::first(defteam),
      week = dplyr::first(week),
      start_yardline = dplyr::first(yardline_100),
      fumble = any(fumble_lost == 1, na.rm = TRUE),
      interception = any(interception == 1, na.rm = TRUE),
      score_gained = max(posteam_score_post, na.rm = TRUE) - min(posteam_score_post, na.rm = TRUE),
      eckel = any((yardline_100 <= 40 & first_down == 1) | touchdown == 1),
      .groups = "drop"
    ) %>%
    dplyr::mutate(turnover = ifelse(fumble | interception, 1, 0))

  defensive_drives_league_summary <- defensive_drives_league %>%
    dplyr::group_by(week, defteam) %>%
    dplyr::summarize(
      total_drives = dplyr::n(),
      eckel_drives = sum(eckel, na.rm = TRUE),
      total_points = sum(score_gained, na.rm = TRUE),
      turnovers = sum(turnover, na.rm = TRUE),
      avg_start_yardline = mean(start_yardline, na.rm = TRUE),
      eckel_rate = safe_div(eckel_drives, total_drives),
      points_per_eckel = ifelse(eckel_drives > 0, total_points / eckel_drives, 0),
      turnover_rate = safe_div(turnovers, total_drives),
      .groups = "drop"
    )

  defensive_drive_averages <- defensive_drives_league_summary %>%
    dplyr::group_by(defteam) %>%
    dplyr::summarize(
      total_games = dplyr::n(),
      total_drives = sum(total_drives, na.rm = TRUE),  # ✅ fixed
      avg_def_eckel_rate = mean(eckel_rate, na.rm = TRUE),
      avg_def_points_per_eckel = mean(points_per_eckel, na.rm = TRUE),
      avg_def_fp = mean(avg_start_yardline, na.rm = TRUE),
      avg_def_turnover_rate = mean(turnover_rate, na.rm = TRUE),
      .groups = "drop"
    )

  eckel_vs_defense <- offensive_drive_summary %>%
    dplyr::left_join(defensive_drive_averages, by = "defteam") %>%
    dplyr::mutate(
      eckel_rate_diff       = (eckel_rate       - avg_def_eckel_rate)       / pmax(avg_def_eckel_rate, 1e-6),
      points_per_eckel_diff = (points_per_eckel - avg_def_points_per_eckel) / pmax(avg_def_points_per_eckel, 1e-6),
      field_position_diff   = ((avg_start_yardline - avg_def_fp) / pmax(avg_def_fp, 1e-6)) * -1,
      turnover_rate_diff    = (turnover_rate    - avg_def_turnover_rate)    / pmax(avg_def_turnover_rate, 1e-6)
    )

  # TEAM DEFENSIVE DRIVES (this team on defense; per opponent posteam)
  defensive_drives_team <- pbp %>%
    dplyr::filter(defteam == team_abbr, (kickoff_attempt == 0 & punt_attempt == 0)) %>%
    dplyr::group_by(game_id, drive) %>%
    dplyr::summarize(
      posteam = dplyr::first(posteam),
      defteam = dplyr::first(defteam),
      week = dplyr::first(week),
      start_yardline = dplyr::first(yardline_100),
      fumble = any(fumble_lost == 1, na.rm = TRUE),
      interception = any(interception == 1, na.rm = TRUE),
      score_gained = max(posteam_score, na.rm = TRUE) - min(posteam_score, na.rm = TRUE),
      eckel = any((yardline_100 <= 40 & first_down == 1) | touchdown == 1),
      .groups = "drop"
    ) %>%
    dplyr::mutate(turnover = ifelse(fumble | interception, 1, 0))

  defensive_drive_summary <- defensive_drives_team %>%
    dplyr::group_by(week, posteam) %>%
    dplyr::summarize(
      total_drives = dplyr::n(),
      eckel_drives = sum(eckel, na.rm = TRUE),
      total_points = sum(score_gained, na.rm = TRUE),
      turnovers = sum(turnover, na.rm = TRUE),
      avg_start_yardline = mean(start_yardline, na.rm = TRUE),
      eckel_rate_allowed = safe_div(eckel_drives, total_drives),
      points_per_eckel_allowed = ifelse(eckel_drives > 0, total_points / eckel_drives, 0),
      turnover_rate_forced = safe_div(turnovers, total_drives),
      .groups = "drop"
    )

  defense_average_game <- defensive_drive_summary %>%
    dplyr::summarize(
      avg_start_yardline = mean(avg_start_yardline, na.rm = TRUE),
      average_eckel_rate = mean(eckel_rate_allowed, na.rm = TRUE),
      average_points_per_eckel = mean(points_per_eckel_allowed, na.rm = TRUE),
      average_turnover_rate = mean(turnover_rate_forced, na.rm = TRUE),
      .groups = "drop"
    )

  # LEAGUE OFFENSIVE DRIVE AVERAGES (for opponent baselines)
  offensive_drives_league <- pbp %>%
    dplyr::filter(!is.na(posteam), !is.na(defteam), (kickoff_attempt == 0 & punt_attempt == 0)) %>%
    dplyr::group_by(game_id, drive, posteam) %>%
    dplyr::summarize(
      posteam = dplyr::first(posteam),
      defteam = dplyr::first(defteam),
      week = dplyr::first(week),
      start_yardline = dplyr::first(yardline_100),
      fumble = any(fumble_lost == 1, na.rm = TRUE),
      interception = any(interception == 1, na.rm = TRUE),
      score_gained = max(posteam_score, na.rm = TRUE) - min(posteam_score, na.rm = TRUE),
      eckel = any((yardline_100 <= 40 & first_down == 1) | touchdown == 1),
      .groups = "drop"
    ) %>%
    dplyr::mutate(turnover = ifelse(fumble | interception, 1, 0))

  offensive_drives_league_summary <- offensive_drives_league %>%
    dplyr::group_by(week, posteam) %>%
    dplyr::summarize(
      total_drives = dplyr::n(),
      eckel_drives = sum(eckel, na.rm = TRUE),
      total_points = sum(score_gained, na.rm = TRUE),
      turnovers = sum(turnover, na.rm = TRUE),
      avg_start_yardline = mean(start_yardline, na.rm = TRUE),
      eckel_rate = safe_div(eckel_drives, total_drives),
      points_per_eckel = ifelse(eckel_drives > 0, total_points / eckel_drives, 0),
      turnover_rate = safe_div(turnovers, total_drives),
      .groups = "drop"
    )

  offensive_drive_averages <- offensive_drives_league_summary %>%
    dplyr::group_by(posteam) %>%
    dplyr::summarize(
      avg_off_eckel_rate = mean(eckel_rate, na.rm = TRUE),
      avg_off_points_per_eckel = mean(points_per_eckel, na.rm = TRUE),
      avg_off_fp = mean(avg_start_yardline, na.rm = TRUE),
      avg_off_turnover_rate = mean(turnover_rate, na.rm = TRUE),
      .groups = "drop"
    )

  eckel_vs_offense <- defensive_drive_summary %>%
    dplyr::left_join(offensive_drive_averages, by = "posteam") %>%
    dplyr::mutate(
      eckel_rate_allowed_diff        = (eckel_rate_allowed        - avg_off_eckel_rate)        / pmax(avg_off_eckel_rate, 1e-6),
      points_per_eckel_allowed_diff  = (points_per_eckel_allowed  - avg_off_points_per_eckel)  / pmax(avg_off_points_per_eckel, 1e-6),
      field_position_allowed_diff    = ((avg_start_yardline       - avg_off_fp)                / pmax(avg_off_fp, 1e-6)) * -1,
      turnover_rate_forced_diff      = (turnover_rate_forced      - avg_off_turnover_rate)     / pmax(avg_off_turnover_rate, 1e-6)
    )

  # SDs for Eckel noise
  eckel_off_sd <- eckel_vs_defense %>% dplyr::summarize(
    sd_eckel_rate_diff        = if (length(stats::na.omit(eckel_rate_diff)) > 1)        stats::sd(eckel_rate_diff, na.rm = TRUE) else 0,
    sd_points_per_eckel_diff  = if (length(stats::na.omit(points_per_eckel_diff)) > 1)  stats::sd(points_per_eckel_diff, na.rm = TRUE) else 0,
    sd_field_position_diff    = if (length(stats::na.omit(field_position_diff)) > 1)    stats::sd(field_position_diff, na.rm = TRUE) else 0,
    sd_turnover_rate_diff     = if (length(stats::na.omit(turnover_rate_diff)) > 1)     stats::sd(turnover_rate_diff, na.rm = TRUE) else 0
  )
  eckel_def_sd <- eckel_vs_offense %>% dplyr::summarize(
    sd_eckel_allowed_diff           = if (length(stats::na.omit(eckel_rate_allowed_diff)) > 1)          stats::sd(eckel_rate_allowed_diff, na.rm = TRUE) else 0,
    sd_points_per_eckel_allow_diff  = if (length(stats::na.omit(points_per_eckel_allowed_diff)) > 1)    stats::sd(points_per_eckel_allowed_diff, na.rm = TRUE) else 0,
    sd_field_position_allow_diff    = if (length(stats::na.omit(field_position_allowed_diff)) > 1)      stats::sd(field_position_allowed_diff, na.rm = TRUE) else 0,
    sd_turnover_rate_def_diff       = if (length(stats::na.omit(turnover_rate_forced_diff)) > 1)        stats::sd(turnover_rate_forced_diff, na.rm = TRUE) else 0
  )

  # Return all the pieces used elsewhere
  list(
    offense_vs_defense = offense_vs_defense,
    defense_vs_offense = defense_vs_offense,
    attempts_per_game = attempts_per_game,
    attempt_summary = attempt_summary,
    offense_ypp = offense_ypp,
    defense_ypp = defense_ypp,
    offense_sd = offense_sd,
    defense_sd = defense_sd,
    # Eckel tables
    eckel_vs_defense = eckel_vs_defense,
    eckel_vs_offense = eckel_vs_offense,
    offensive_drives = offensive_drives_team,
    offensive_drive_summary = offensive_drive_summary,
    defensive_drives = defensive_drives_team,
    defensive_drive_summary = defensive_drive_summary,
    defensive_drive_averages = defensive_drive_averages,
    offensive_drive_averages = offensive_drive_averages,
    # Averages you reference in the projector (these include average_eckel_rate)
    offense_average_game = offense_average_game,
    defense_average_game = defense_average_game,
    # SDs for Eckel projector noise
    eckel_def_sd = eckel_def_sd,
    eckel_off_sd = eckel_off_sd
  )
}
