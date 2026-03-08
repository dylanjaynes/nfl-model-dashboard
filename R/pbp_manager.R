#' @importFrom data.table data.table setDT rbindlist :=
NULL



##############################################
# pbp_manager.R
# Fast PBP loading + season precompute system
# Used by run_one_game() for ultra-fast metrics
##############################################

#########################################################
# 1. get_pbp_cached_fast()
#########################################################
# Wrapper around your get_pbp_cached() but ensures:
# - data.table
# - numeric week
# - week 18 removed
# - sorted and ready for precompute
#########################################################

get_pbp_cached_fast <- function(season,
                                through_week = Inf,
                                cache_dir = "cache/pbp") {

  pbp <- get_pbp_cached(
    season = season,
    through_week = through_week,
    cache_dir = cache_dir,
    refresh = "auto"
  )

  data.table::setDT(pbp)
  pbp[, week := as.integer(week)]
  pbp <- pbp[week != 18]
  data.table::setorder(pbp, game_id, play_id)
  return(pbp)
}



#########################################################
# 2. precompute_season_metrics_fast()
#########################################################
# Build the FAST season-wide precompute object:
#
#   - pbp_off[[team]]  : all offensive plays for that team
#   - pbp_def[[team]]  : all defensive plays for that team
#   - drive_by_posteam : drive table keyed by posteam
#   - drive_by_defteam : drive table keyed by defteam
#   - league_def_avg   : league ypc/ypa allowed
#   - league_off_avg   : league ypc/ypa generated
#   - league_drive_def : league eckel/drive metrics allowed
#   - league_drive_off : league eckel/drive metrics generated
#
# This is called once per season or when precompute not found.
#########################################################

precompute_season_metrics_fast <- function(pbp) {

  message("[Precompute] Building season metrics…")

  # ensure DT
  data.table::setDT(pbp)
  pbp[, week := as.integer(week)]
  pbp <- pbp[week != 18]   # remove week 18 noise

  # --------------------------------------------------------------------
  # REQUIRED COLUMNS for fast metrics (matching your original model)
  # --------------------------------------------------------------------
  required_cols <- c(
    "game_id", "drive", "posteam", "defteam", "week",
    "yards_gained", "rush_attempt", "pass_attempt",
    "pass_touchdown", "interception", "sack",
    "posteam_score", "posteam_score_post",
    "kickoff_attempt", "punt_attempt",
    "yardline_100", "fumble_lost", "first_down", "touchdown"
  )

  # Fill missing columns safely
  for (col in required_cols) {
    if (!col %in% names(pbp)) {
      pbp[, (col) := NA]
    }
  }

  # Keep only required columns
  pbp <- pbp[, ..required_cols]

  # --------------------------------------------------------------------
  # SPLIT PBP INTO OFFENSE + DEFENSE (ALL REQUIRED FIELDS INCLUDED)
  # --------------------------------------------------------------------
  pbp_off <- split(pbp, by = "posteam", keep.by = TRUE)
  pbp_def <- split(pbp, by = "defteam", keep.by = TRUE)

  # --------------------------------------------------------------------
  # DRIVE TABLE (ONCE PER SEASON)
  # --------------------------------------------------------------------
  drive_table <- pbp[
    kickoff_attempt == 0 & punt_attempt == 0 & !is.na(drive),
    .(
      posteam          = first(posteam),
      defteam          = first(defteam),
      week             = first(week),
      start_yardline   = first(yardline_100),
      fumble           = any(fumble_lost == 1, na.rm = TRUE),
      interception     = any(interception == 1, na.rm = TRUE),
      score_gained     = max(posteam_score_post, na.rm = TRUE) -
        min(posteam_score_post, na.rm = TRUE),
      eckel            = any(
        (yardline_100 <= 40 & first_down == 1) |
          touchdown == 1,
        na.rm = TRUE
      )
    ),
    by = .(game_id, drive)
  ][, turnover := as.integer(fumble | interception)]

  drive_by_posteam <- split(drive_table, by = "posteam", keep.by = TRUE)
  drive_by_defteam <- split(drive_table, by = "defteam", keep.by = TRUE)

  # --------------------------------------------------------------------
  # LEAGUE PBP DEFENSE AVERAGES
  # --------------------------------------------------------------------
  league_def_avg <- pbp[
    (rush_attempt == 1 | pass_attempt == 1),
    .(
      avg_def_ypc = mean(yards_gained[rush_attempt == 1], na.rm = TRUE),
      avg_def_ypa = (
        sum(yards_gained[pass_attempt == 1], na.rm = TRUE) +
          20 * sum(pass_touchdown == 1, na.rm = TRUE) -
          45 * sum(interception == 1, na.rm = TRUE) -
          sum(yards_gained[sack == 1], na.rm = TRUE)
      ) / max(sum(pass_attempt == 1, na.rm = TRUE) +
                sum(sack == 1, na.rm = TRUE), 1)
    ),
    by = defteam
  ]

  # --------------------------------------------------------------------
  # LEAGUE PBP OFFENSE AVERAGES
  # --------------------------------------------------------------------
  league_off_avg <- pbp[
    (rush_attempt == 1 | pass_attempt == 1),
    .(
      avg_off_ypc = mean(yards_gained[rush_attempt == 1], na.rm = TRUE),
      avg_off_ypa = (
        sum(yards_gained[pass_attempt == 1], na.rm = TRUE) +
          20 * sum(pass_touchdown == 1, na.rm = TRUE) -
          45 * sum(interception == 1, na.rm = TRUE) -
          sum(yards_gained[sack == 1], na.rm = TRUE)
      ) / max(sum(pass_attempt == 1, na.rm = TRUE) +
                sum(sack == 1, na.rm = TRUE), 1)
    ),
    by = posteam
  ]

  # --------------------------------------------------------------------
  # LEAGUE DRIVE DEFENSE AVERAGES
  # --------------------------------------------------------------------
  league_drive_def <- drive_table[
    , .(
      avg_def_eckel_rate        = mean(eckel, na.rm = TRUE),
      avg_def_points_per_eckel  = mean(score_gained[eckel == 1], na.rm = TRUE),
      avg_def_fp                = mean(start_yardline, na.rm = TRUE),
      avg_def_turnover_rate     = mean(turnover, na.rm = TRUE)
    ),
    by = defteam
  ]

  # --------------------------------------------------------------------
  # LEAGUE DRIVE OFFENSE AVERAGES
  # --------------------------------------------------------------------
  league_drive_off <- drive_table[
    , .(
      avg_off_eckel_rate       = mean(eckel, na.rm = TRUE),
      avg_off_points_per_eckel = mean(score_gained[eckel == 1], na.rm = TRUE),
      avg_off_fp               = mean(start_yardline, na.rm = TRUE),
      avg_off_turnover_rate    = mean(turnover, na.rm = TRUE)
    ),
    by = posteam
  ]

  # --------------------------------------------------------------------
  # RETURN FULL PRECOMPUTE OBJECT
  # --------------------------------------------------------------------
  list(
    pbp_off            = pbp_off,
    pbp_def            = pbp_def,
    drive_by_posteam   = drive_by_posteam,
    drive_by_defteam   = drive_by_defteam,
    league_def_avg     = league_def_avg,
    league_off_avg     = league_off_avg,
    league_drive_def   = league_drive_def,
    league_drive_off   = league_drive_off,
    max_week           = max(pbp$week, na.rm = TRUE)
  )
}



#########################################################
# 3. update_precomputed_season_metrics_fast()
#########################################################
# Incrementally appends new PBP rows + drives + updates
# league averages, without rebuilding the season object.
#########################################################

update_precomputed_season_metrics_fast <- function(pre, pbp_new) {

  if (nrow(pbp_new) == 0) return(pre)

  data.table::setDT(pbp_new)
  pbp_new[, week := as.integer(week)]
  pbp_new <- pbp_new[week != 18]

  # ---- append new offensive PBP ----
  for (tm in unique(pbp_new$posteam)) {
    if (tm %in% names(pre$pbp_off)) {
      pre$pbp_off[[tm]] <- data.table::rbindlist(
        list(pre$pbp_off[[tm]], pbp_new[posteam == tm]),
        fill = TRUE
      )
    } else {
      pre$pbp_off[[tm]] <- pbp_new[posteam == tm]
    }
  }

  # ---- append new defensive PBP ----
  for (tm in unique(pbp_new$defteam)) {
    if (tm %in% names(pre$pbp_def)) {
      pre$pbp_def[[tm]] <- data.table::rbindlist(
        list(pre$pbp_def[[tm]], pbp_new[defteam == tm]),
        fill = TRUE
      )
    } else {
      pre$pbp_def[[tm]] <- pbp_new[defteam == tm]
    }
  }

  # ---- append new drive rows ----
  drives_new <- pbp_new[
    kickoff_attempt == 0 & punt_attempt == 0 & !is.na(drive),
    .(
      posteam = data.table::first(posteam),
      defteam = data.table::first(defteam),
      week    = data.table::first(week),
      start_yardline = data.table::first(yardline_100),
      fumble = any(fumble_lost == 1),
      interception = any(interception == 1),
      score_gained = max(posteam_score_post, na.rm = TRUE) -
        min(posteam_score_post, na.rm = TRUE),
      eckel = any((yardline_100 <= 40 & first_down == 1) | touchdown == 1)
    ),
    by = .(game_id, drive)
  ][, turnover := as.integer(fumble | interception)]

  if (nrow(drives_new) > 0) {

    for (tm in unique(drives_new$posteam)) {
      pre$drive_by_posteam[[tm]] <-
        data.table::rbindlist(
          list(pre$drive_by_posteam[[tm]], drives_new[posteam == tm]),
          fill = TRUE
        )
    }

    for (tm in unique(drives_new$defteam)) {
      pre$drive_by_defteam[[tm]] <-
        data.table::rbindlist(
          list(pre$drive_by_defteam[[tm]], drives_new[defteam == tm]),
          fill = TRUE
        )
    }
  }

  # ---- recompute league PBP averages quickly ----
  all_pbp <- data.table::rbindlist(pre$pbp_off, fill = TRUE)

  pre$league_def_avg <- all_pbp[
    (rush_attempt==1 | pass_attempt==1),
    .(
      avg_def_ypc = mean(yards_gained[rush_attempt == 1], na.rm=TRUE),
      avg_def_ypa = (
        sum(yards_gained[pass_attempt == 1], na.rm=TRUE) +
          20*sum(pass_touchdown, na.rm=TRUE) -
          45*sum(interception, na.rm=TRUE) -
          sum(yards_gained[sack==1], na.rm=TRUE)
      ) / max(sum(pass_attempt==1) + sum(sack==1), 1)
    ),
    by=defteam
  ]

  pre$league_off_avg <- all_pbp[
    (rush_attempt==1 | pass_attempt==1),
    .(
      avg_off_ypc = mean(yards_gained[rush_attempt==1], na.rm=TRUE),
      avg_off_ypa = (
        sum(yards_gained[pass_attempt==1]) +
          20*sum(pass_touchdown) -
          45*sum(interception) -
          sum(yards_gained[sack==1])
      ) / max(sum(pass_attempt==1) + sum(sack==1), 1)
    ),
    by=posteam
  ]

  # ---- recompute drive averages ----
  all_drives <- data.table::rbindlist(pre$drive_by_posteam, fill = TRUE)

  pre$league_drive_def <- all_drives[
    , .(
      avg_def_eckel_rate = mean(eckel),
      avg_def_points_per_eckel = mean(score_gained[eckel==1], na.rm=TRUE),
      avg_def_fp = mean(start_yardline),
      avg_def_turnover_rate = mean(turnover)
    ),
    by = defteam
  ]

  pre$league_drive_off <- all_drives[
    , .(
      avg_off_eckel_rate = mean(eckel),
      avg_off_points_per_eckel = mean(score_gained[eckel==1], na.rm=TRUE),
      avg_off_fp = mean(start_yardline),
      avg_off_turnover_rate = mean(turnover)
    ),
    by = posteam
  ]

  return(pre)
}



#########################################################
# 4. ensure_precompute_upto_week()
#########################################################
# Master controller:
# - Loads precompute (if exists)
# - Detects if missing weeks
# - Fetches PBP through required_week
# - Builds or updates precompute
#########################################################

ensure_precompute_upto_week <- function(season,
                                        required_week,
                                        cache_dir = "cache/pre",
                                        pbp_cache_dir = "cache/pbp") {

  pre_path <- file.path(cache_dir, sprintf("pre_%d.rds", season))

  if (!dir.exists(cache_dir)) dir.create(cache_dir, recursive = TRUE)

  # ---- Load precompute if available ----
  if (file.exists(pre_path)) {
    pre <- readRDS(pre_path)
    current_max_week <- pre$max_week
  } else {
    pre <- NULL
    current_max_week <- 0
  }

  # Up to date?
  if (current_max_week >= required_week)
    return(pre)

  # ---- Fetch PBP through required week ----
  pbp <- get_pbp_cached_fast(
    season = season,
    through_week = required_week,
    cache_dir = pbp_cache_dir
  )

  # ---- First build ----
  if (is.null(pre)) {
    pre <- precompute_season_metrics_fast(pbp)
    pre$max_week <- max(pbp$week, na.rm = TRUE)
    saveRDS(pre, pre_path)
    return(pre)
  }

  # ---- Incremental update ----
  new_only <- pbp[week > current_max_week]
  if (nrow(new_only) > 0) {
    pre <- update_precomputed_season_metrics_fast(pre, new_only)
    pre$max_week <- required_week
    saveRDS(pre, pre_path)
  }

  return(pre)
}


make_empty_team_metrics <- function() {
  zero <- 0
  list(
    offense_vs_defense = list(offensive_ypc = zero, offensive_ypa = zero),
    defense_vs_offense = list(ypc_diff_allowed = zero, ypa_diff_allowed = zero),

    attempt_summary = list(
      avg_rush_attempts_per_game = 20,
      avg_pass_attempts_per_game = 30,
      sd_rush_attempts_per_game  = 5,
      sd_pass_attempts_per_game  = 8
    ),

    offense_ypp = list(yards_per_point_offense = 12),
    defense_ypp = list(yards_per_point_defense = 12),

    offense_average_game = list(average_eckel_rate = .3, average_points_per_eckel = 4),
    defense_average_game = list(average_eckel_rate = .3, average_points_per_eckel = 4),

    offense_sd = list(sd_ypc_diff = .01, sd_ypa_diff = .01),
    defense_sd = list(sd_ypc_diff_allowed = .01, sd_ypa_diff_allowed = .01),

    eckel_off_sd = list(
      sd_eckel_rate_diff = .01,
      sd_points_per_eckel_diff = .02,
      sd_field_position_diff = .01,
      sd_turnover_rate_diff = .01
    ),
    eckel_def_sd = list(
      sd_eckel_allowed_diff = .01,
      sd_points_per_eckel_allow_diff = .02,
      sd_field_position_allow_diff = .01,
      sd_turnover_rate_def_diff = .01
    ),

    eckel_vs_offense = list(
      eckel_rate_allowed_diff = zero,
      points_per_eckel_allowed_diff = zero
    ),
    eckel_vs_defense = list(
      eckel_rate_diff = zero,
      points_per_eckel_diff = zero
    )
  )
}



#########################################################
# 5. calculate_team_metrics_fast()
#########################################################
# Ultra-fast metrics:
# - Only uses O(1) lookups
# - Replaces calculate_team_metrics()
# - Compatible with .sim_pbp_once() and .sim_eckel_once()
#########################################################
#########################################################
# FIXED calculate_team_metrics_fast()
# Safe under all priors, all weeks, all seasons
# Never returns NULL or empty fields
#########################################################

calculate_team_metrics_fast <- function(team_abbr, season, week, pre) {

  safe_div <- function(a, b) {
    ifelse(is.finite(a) & is.finite(b) & abs(b) > 0, a / b, NA_real_)
  }

  # -----------------------------
  # Pull precomputed components
  # -----------------------------
  po   <- pre$pbp_off[[team_abbr]]
  pd   <- pre$pbp_def[[team_abbr]]
  doff <- pre$drive_by_posteam[[team_abbr]]
  ddef <- pre$drive_by_defteam[[team_abbr]]

  # If absolutely no data exists for this team → return full defaults
  if (is.null(po) || is.null(pd) || nrow(po) == 0 || nrow(pd) == 0) {
    return(make_empty_team_metrics())
  }

  data.table::setDT(po)
  data.table::setDT(pd)
  data.table::setDT(doff)
  data.table::setDT(ddef)

  # Trim to requested week
  po   <- po[week <= week]
  pd   <- pd[week <= week]
  doff <- doff[week <= week]
  ddef <- ddef[week <= week]

  # If trimmed tables become empty → fallback
  if (nrow(po) == 0 || nrow(pd) == 0) {
    return(make_empty_team_metrics())
  }

  # ============================================================
  # == PBP OFFENSE METRICS ==
  # ============================================================

  offense_stats_dt <- po[
    (rush_attempt == 1 | pass_attempt == 1) & !is.na(yards_gained),
    {
      rush_yards    <- sum(yards_gained[rush_attempt == 1], na.rm = TRUE)
      rush_attempts <- sum(rush_attempt, na.rm = TRUE)
      pass_yards    <- sum(yards_gained[pass_attempt == 1], na.rm = TRUE)
      pass_attempts <- sum(pass_attempt, na.rm = TRUE)

      offensive_ypc <- safe_div(rush_yards, rush_attempts)

      offensive_ypa <- (
        sum(yards_gained[pass_attempt == 1], na.rm = TRUE) +
          20 * sum(pass_touchdown == 1, na.rm = TRUE) -
          45 * sum(interception == 1, na.rm = TRUE) -
          sum(yards_gained[sack == 1], na.rm = TRUE)
      ) / max(sum(pass_attempt == 1, na.rm = TRUE) +
                sum(sack == 1, na.rm = TRUE), 1)

      data.table::data.table(
        week,
        defteam,
        offensive_ypc,
        offensive_ypa
      )
    },
    by = .(week, defteam)
  ]

  if (nrow(offense_stats_dt) == 0)
    return(make_empty_team_metrics())

  offense_stats <- tibble::as_tibble(offense_stats_dt)

  # Attempts per game
  attempts_per_game_dt <- po[
    (rush_attempt == 1 | pass_attempt == 1),
    .(
      rush_attempts = sum(rush_attempt),
      pass_attempts = sum(pass_attempt)
    ),
    by = .(game_id, week)
  ]

  if (nrow(attempts_per_game_dt) == 0)
    attempts_per_game_dt <- data.table::data.table(
      rush_attempts = 20,
      pass_attempts = 30,
      week = week,
      game_id = "placeholder"
    )

  attempts_per_game <- tibble::as_tibble(attempts_per_game_dt)

  attempt_summary_dt <- attempts_per_game_dt[
    ,
    .(
      avg_rush_attempts_per_game =
        mean(rush_attempts, na.rm = TRUE),
      avg_pass_attempts_per_game =
        mean(pass_attempts, na.rm = TRUE),
      sd_rush_attempts_per_game =
        if (.N > 1) sd(rush_attempts) else 5,
      sd_pass_attempts_per_game =
        if (.N > 1) sd(pass_attempts) else 8
    )
  ]

  attempt_summary <- tibble::as_tibble(attempt_summary_dt)

  # ============================================================
  # == OFFENSE YARDS PER POINT ==
  # ============================================================
  offense_ypp_step1 <- po[
    ,
    .(
      total_yards = sum(yards_gained, na.rm = TRUE),
      pts         = max(posteam_score, na.rm = TRUE)
    ),
    by = game_id
  ]

  offense_ypp_dt <- offense_ypp_step1[
    ,
    .(
      yards_per_point_offense =
        safe_div(sum(total_yards), sum(pts))
    )
  ]

  # If this produces NA → fallback to original default = 12
  if (is.na(offense_ypp_dt$yards_per_point_offense))
    offense_ypp_dt$yards_per_point_offense <- 12

  offense_ypp <- tibble::as_tibble(offense_ypp_dt)

  # ============================================================
  # == DEFENSE YARDS PER POINT ==
  # ============================================================
  defense_ypp_step1 <- pd[
    ,
    .(
      total_yards = sum(yards_gained, na.rm = TRUE),
      pts         = max(posteam_score, na.rm = TRUE)
    ),
    by = game_id
  ]

  defense_ypp_dt <- defense_ypp_step1[
    ,
    .(
      yards_per_point_defense =
        safe_div(sum(total_yards), sum(pts))
    )
  ]

  if (is.na(defense_ypp_dt$yards_per_point_defense))
    defense_ypp_dt$yards_per_point_defense <- 12

  defense_ypp <- tibble::as_tibble(defense_ypp_dt)

  # ============================================================
  # == LEAGUE AVERAGES (from precompute) ==
  # ============================================================

  league_def <- data.table::as.data.table(pre$league_def_avg)
  league_off <- data.table::as.data.table(pre$league_off_avg)

  defense_averages_dt <- league_def[
    defteam %in% offense_stats_dt$defteam,
    .(defteam, avg_def_ypc, avg_def_ypa)
  ]

  # If missing → fallback to league-wide neutral
  if (nrow(defense_averages_dt) == 0)
    defense_averages_dt <- data.table::data.table(
      defteam = offense_stats_dt$defteam,
      avg_def_ypc = 4.3,
      avg_def_ypa = 6.4
    )

  offense_averages_dt <- league_off[
    posteam %in% pd$posteam,
    .(posteam, avg_off_ypc, avg_off_ypa)
  ]

  if (nrow(offense_averages_dt) == 0)
    offense_averages_dt <- data.table::data.table(
      posteam = team_abbr,
      avg_off_ypc = 4.3,
      avg_off_ypa = 6.4
    )

  # ============================================================
  # == Offense vs Defense ==
  # ============================================================
  offense_vs_defense_dt <- merge(
    offense_stats_dt, defense_averages_dt,
    by = "defteam", all.x = TRUE
  )[
    ,
    `:=`(
      ypc_diff = safe_div(offensive_ypc - avg_def_ypc, avg_def_ypc),
      ypa_diff = safe_div(offensive_ypa - avg_def_ypa, avg_def_ypa)
    )
  ]

  offense_vs_defense <- tibble::as_tibble(offense_vs_defense_dt)

  # ============================================================
  # == Defense vs Offense ==
  # ============================================================
  defense_stats_dt <- pd[
    (rush_attempt == 1 | pass_attempt == 1),
    {
      rush_yards <- sum(yards_gained[rush_attempt == 1], na.rm = TRUE)
      rush_att   <- sum(rush_attempt, na.rm = TRUE)
      pass_att   <- sum(pass_attempt, na.rm = TRUE)

      defensive_ypc <- safe_div(rush_yards, rush_att)

      defensive_ypa <- (
        sum(yards_gained[pass_attempt == 1], na.rm = TRUE) +
          20 * sum(pass_touchdown == 1, na.rm = TRUE) -
          45 * sum(interception == 1, na.rm = TRUE) -
          sum(yards_gained[sack == 1], na.rm = TRUE)
      ) / max(sum(pass_attempt == 1) + sum(sack == 1), 1)

      data.table::data.table(posteam, defensive_ypc, defensive_ypa)
    },
    by = .(week, posteam)
  ]

  if (nrow(defense_stats_dt) == 0)
    return(make_empty_team_metrics())

  defense_vs_offense_dt <- merge(
    defense_stats_dt,
    offense_averages_dt,
    by = "posteam",
    all.x = TRUE
  )[
    ,
    `:=`(
      ypc_diff_allowed =
        safe_div(defensive_ypc - avg_off_ypc, avg_off_ypc),
      ypa_diff_allowed =
        safe_div(defensive_ypa - avg_off_ypa, avg_off_ypa)
    )
  ]

  defense_vs_offense <- tibble::as_tibble(defense_vs_offense_dt)

  # SDs always at least something
  sd_or_default <- function(x, def) {
    x <- na.omit(x)
    if (length(x) <= 1) def else stats::sd(x)
  }

  offense_sd <- tibble::tibble(
    sd_ypc_diff = sd_or_default(offense_vs_defense_dt$ypc_diff, 0.01),
    sd_ypa_diff = sd_or_default(offense_vs_defense_dt$ypa_diff, 0.01)
  )

  defense_sd <- tibble::tibble(
    sd_ypc_diff_allowed =
      sd_or_default(defense_vs_offense_dt$ypc_diff_allowed, 0.01),
    sd_ypa_diff_allowed =
      sd_or_default(defense_vs_offense_dt$ypa_diff_allowed, 0.01)
  )

  # ============================================================
  # == Drive-Based Eckel Metrics ==
  # ============================================================

  # If doff/ddef are empty → fallback immediately
  if (is.null(doff) || nrow(doff) == 0 ||
      is.null(ddef) || nrow(ddef) == 0) {
    return(make_empty_team_metrics())
  }

  # OFFENSIVE DRIVES
  offensive_drives_team_dt <- doff[
    ,
    .(
      posteam, defteam, week,
      start_yardline, fumble, interception,
      score_gained, eckel, turnover
    ),
    by = .(game_id, drive)
  ]

  if (nrow(offensive_drives_team_dt) == 0) {
    return(make_empty_team_metrics())
  }

  # Summary with safe fallbacks
  offensive_drive_summary_dt <- offensive_drives_team_dt[
    ,
    .(
      total_drives       = .N,
      eckel_drives       = sum(eckel),
      total_points       = sum(score_gained),
      turnovers          = sum(turnover),
      avg_start_yardline = mean(start_yardline)
    ),
    by = .(week, defteam)
  ][
    ,
    `:=`(
      eckel_rate = safe_div(eckel_drives, total_drives),
      points_per_eckel =
        ifelse(eckel_drives > 0,
               total_points / eckel_drives, 4),
      turnover_rate = safe_div(turnovers, total_drives)
    )
  ]

  offense_average_game_dt <- offensive_drive_summary_dt[
    ,
    .(
      avg_start_yardline      = mean(avg_start_yardline),
      average_eckel_rate      = mean(eckel_rate),
      average_points_per_eckel =
        mean(points_per_eckel),
      average_turnover_rate   = mean(turnover_rate)
    )
  ]

  # If empty → fallback
  if (nrow(offense_average_game_dt) == 0) {
    return(make_empty_team_metrics())
  }

  # League defense drive avgs (already computed)
  defensive_drive_averages_dt <- pre$league_drive_def

  # eckel_vs_defense
  eckel_vs_defense_dt <- merge(
    offensive_drive_summary_dt,
    defensive_drive_averages_dt,
    by = "defteam",
    all.x = TRUE
  )[
    ,
    `:=`(
      eckel_rate_diff =
        safe_div(eckel_rate - avg_def_eckel_rate, avg_def_eckel_rate),
      points_per_eckel_diff =
        safe_div(points_per_eckel - avg_def_points_per_eckel,
                 avg_def_points_per_eckel),
      field_position_diff =
        safe_div(avg_start_yardline - avg_def_fp, avg_def_fp) * -1,
      turnover_rate_diff =
        safe_div(turnover_rate - avg_def_turnover_rate,
                 avg_def_turnover_rate)
    )
  ]

  if (nrow(eckel_vs_defense_dt) == 0)
    return(make_empty_team_metrics())

  # DEFENSIVE DRIVES
  defensive_drives_team_dt <- ddef[
    ,
    .(
      posteam, defteam, week,
      start_yardline, fumble, interception,
      score_gained, eckel, turnover
    ),
    by = .(game_id, drive)
  ]

  if (nrow(defensive_drives_team_dt) == 0)
    return(make_empty_team_metrics())

  defensive_drive_summary_dt <- defensive_drives_team_dt[
    ,
    .(
      total_drives       = .N,
      eckel_drives       = sum(eckel),
      total_points       = sum(score_gained),
      turnovers          = sum(turnover),
      avg_start_yardline = mean(start_yardline)
    ),
    by = .(week, posteam)
  ][
    ,
    `:=`(
      eckel_rate_allowed =
        safe_div(eckel_drives, total_drives),
      points_per_eckel_allowed =
        ifelse(eckel_drives > 0,
               total_points / eckel_drives, 4),
      turnover_rate_forced =
        safe_div(turnovers, total_drives)
    )
  ]

  if (nrow(defensive_drive_summary_dt) == 0)
    return(make_empty_team_metrics())

  offensive_drive_averages_dt <- pre$league_drive_off

  eckel_vs_offense_dt <- merge(
    defensive_drive_summary_dt,
    offensive_drive_averages_dt,
    by = "posteam",
    all.x = TRUE
  )[
    ,
    `:=`(
      eckel_rate_allowed_diff =
        safe_div(eckel_rate_allowed - avg_off_eckel_rate,
                 avg_off_eckel_rate),
      points_per_eckel_allowed_diff =
        safe_div(points_per_eckel_allowed - avg_off_points_per_eckel,
                 avg_off_points_per_eckel),
      field_position_allowed_diff =
        safe_div(avg_start_yardline - avg_off_fp, avg_off_fp) * -1,
      turnover_rate_forced_diff =
        safe_div(turnover_rate_forced - avg_off_turnover_rate,
                 avg_off_turnover_rate)
    )
  ]

  # SDs for Eckel components
  sd_eckel <- function(x, default = 0.01) {
    x <- na.omit(x)
    if (length(x) <= 1) default else stats::sd(x)
  }

  eckel_off_sd_dt <- tibble::tibble(
    sd_eckel_rate_diff =
      sd_eckel(eckel_vs_defense_dt$eckel_rate_diff),
    sd_points_per_eckel_diff =
      sd_eckel(eckel_vs_defense_dt$points_per_eckel_diff, 0.02),
    sd_field_position_diff =
      sd_eckel(eckel_vs_defense_dt$field_position_diff),
    sd_turnover_rate_diff =
      sd_eckel(eckel_vs_defense_dt$turnover_rate_diff)
  )

  eckel_def_sd_dt <- tibble::tibble(
    sd_eckel_allowed_diff =
      sd_eckel(eckel_vs_offense_dt$eckel_rate_allowed_diff),
    sd_points_per_eckel_allow_diff =
      sd_eckel(eckel_vs_offense_dt$points_per_eckel_allowed_diff, 0.02),
    sd_field_position_allow_diff =
      sd_eckel(eckel_vs_offense_dt$field_position_allowed_diff),
    sd_turnover_rate_def_diff =
      sd_eckel(eckel_vs_offense_dt$turnover_rate_forced_diff)
  )

  # ============================================================
  # Return metrics in EXACT structure run_one_game() expects
  # ============================================================

  list(
    offense_vs_defense        = offense_vs_defense,
    defense_vs_offense        = defense_vs_offense,
    attempts_per_game         = attempts_per_game,
    attempt_summary           = attempt_summary,
    offense_ypp               = offense_ypp,
    defense_ypp               = defense_ypp,
    offense_sd                = offense_sd,
    defense_sd                = defense_sd,
    eckel_vs_defense          = tibble::as_tibble(eckel_vs_defense_dt),
    eckel_vs_offense          = tibble::as_tibble(eckel_vs_offense_dt),
    offensive_drives          = tibble::as_tibble(offensive_drives_team_dt),
    offensive_drive_summary   = tibble::as_tibble(offensive_drive_summary_dt),
    defensive_drives          = tibble::as_tibble(defensive_drives_team_dt),
    defensive_drive_summary   = tibble::as_tibble(defensive_drive_summary_dt),
    defensive_drive_averages  = tibble::as_tibble(pre$league_drive_def),
    offensive_drive_averages  = tibble::as_tibble(pre$league_drive_off),
    offense_average_game      = tibble::as_tibble(offense_average_game_dt),
    defense_average_game      = tibble::as_tibble(
      defensive_drive_summary_dt[, .(
        avg_start_yardline      = mean(avg_start_yardline),
        average_eckel_rate      = mean(eckel_rate_allowed),
        average_points_per_eckel =
          mean(points_per_eckel_allowed),
        average_turnover_rate   = mean(turnover_rate_forced)
      )]
    ),
    eckel_def_sd              = eckel_def_sd_dt,
    eckel_off_sd              = eckel_off_sd_dt
  )
}



#########################################################
# END pbp_manager.R
#########################################################
