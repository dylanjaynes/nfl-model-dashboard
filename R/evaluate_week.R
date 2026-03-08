#' @import magrittr
#' @import dplyr
NULL


# ---------- helpers ----------
safe_read_excel <- function(path, sheet = 1) {
  if (requireNamespace("readxl", quietly = TRUE)) {
    readxl::read_excel(path, sheet = sheet)
  } else if (requireNamespace("openxlsx", quietly = TRUE)) {
    openxlsx::read.xlsx(path, sheet = sheet)
  } else {
    stop("Install either {readxl} or {openxlsx}.")
  }
}

# parse "67%" -> 0.67, "0.6715" -> 0.6715
.parse_prob <- function(x) {
  if (is.numeric(x)) return(x)
  x <- as.character(x)
  suppressWarnings(readr::parse_number(x) / ifelse(grepl("%", x), 100, 1))
}

# ---------- read a bet card (long or wide) from Excel ----------
read_betcard_excel <- function(path, sheet = 1) {
  raw <- safe_read_excel(path, sheet = sheet) |> janitor::clean_names()

  # If you wrote the “_high / _low” wide columns, map to canonical fields
  if (!"ats_pick" %in% names(raw) && "ats_pick_high" %in% names(raw)) {
    raw <- raw |>
      dplyr::mutate(ats_pick = ats_pick_high,
                    ats_prob = .parse_prob(ats_prob_high))
  }
  if (!"total_pick" %in% names(raw) && "total_pick_high" %in% names(raw)) {
    raw <- raw |>
      dplyr::mutate(total_pick = total_pick_high,
                    total_prob = .parse_prob(total_prob_high))
  }

  # Long format (bet_type/pick/prob present) -> collapse to one row per game
  if (all(c("bet_type","pick","prob") %in% names(raw))) {
    dplyr::bind_rows(raw) |>
      dplyr::mutate(
        prob      = .parse_prob(prob),
        ATS_pick  = dplyr::if_else(tolower(bet_type) == "ats",   as.character(pick), NA_character_),
        ATS_prob  = dplyr::if_else(tolower(bet_type) == "ats",   prob, NA_real_),
        Total_pick= dplyr::if_else(tolower(bet_type) == "total", as.character(pick), NA_character_),
        Total_prob= dplyr::if_else(tolower(bet_type) == "total", prob, NA_real_)
      ) |>
      dplyr::group_by(game_id) |>
      dplyr::summarise(
        season_target    = dplyr::first(season_target),
        week             = dplyr::first(week),
        gameday          = dplyr::first(gameday),
        gametime         = dplyr::first(gametime),
        home_team        = dplyr::first(home_team),
        away_team        = dplyr::first(away_team),
        vegas_spread_line= suppressWarnings(as.numeric(dplyr::first(vegas_spread_line))),
        vegas_total_line = suppressWarnings(as.numeric(dplyr::first(vegas_total_line))),
        ATS_pick   = dplyr::coalesce(dplyr::first(na.omit(ATS_pick)),   NA_character_),
        ATS_prob   = dplyr::coalesce(dplyr::first(na.omit(ATS_prob)),   NA_real_),
        Total_pick = dplyr::coalesce(dplyr::first(na.omit(Total_pick)), NA_character_),
        Total_prob = dplyr::coalesce(dplyr::first(na.omit(Total_prob)), NA_real_),
        .groups = "drop"
      )
  } else {
    # Already wide: normalize names/types the evaluator expects
    raw |>
      dplyr::mutate(
        ATS_prob        = if ("ats_prob"   %in% names(.)) .parse_prob(ats_prob)   else ATS_prob,
        Total_prob      = if ("total_prob" %in% names(.)) .parse_prob(total_prob) else Total_prob,
        vegas_spread_line = suppressWarnings(as.numeric(vegas_spread_line)),
        vegas_total_line  = suppressWarnings(as.numeric(vegas_total_line))
      ) |>
      dplyr::rename(
        ATS_pick   = dplyr::any_of("ats_pick"),
        Total_pick = dplyr::any_of("total_pick")
      )
  }
}

# ---------- evaluate a week from a model dataframe ----------
evaluate_week <- function(model_df, season_target, week_target) {
  stopifnot(is.data.frame(model_df))

  library(dplyr)

  model_week <- model_df |>
    dplyr::filter(.data$season_target == !!season_target,
                  .data$week          == !!week_target)

  # Ensure pick/prob columns exist (rebuild if possible; otherwise NA)
  if (!"ATS_pick" %in% names(model_week))  model_week$ATS_pick <- NA_character_
  if (!"ATS_prob" %in% names(model_week))  model_week$ATS_prob <- NA_real_
  if (!"Total_pick" %in% names(model_week))model_week$Total_pick <- NA_character_
  if (!"Total_prob" %in% names(model_week))model_week$Total_prob <- NA_real_
  if (!"spread_mean" %in% names(model_week)) model_week$spread_mean <- NA_real_
  if (!"total_mean" %in% names(model_week)) model_week$total_mean <- NA_real_

  if (all(c("home_cover_prob","away_cover_prob","home_team","away_team") %in% names(model_week))) {
    model_week <- model_week |>
      mutate(
        ATS_pick = dplyr::if_else(home_cover_prob >= away_cover_prob, home_team, away_team),
        ATS_prob = pmax(home_cover_prob, away_cover_prob)
      )
  }
  if (all(c("over_prob","under_prob") %in% names(model_week))) {
    model_week <- model_week |>
      mutate(
        Total_pick = dplyr::if_else(over_prob >= under_prob, "Over", "Under"),
        Total_prob = pmax(over_prob, under_prob)
      )
  }

  # Actuals + closing lines
  sched <- nflreadr::load_schedules(season_target) |>
    filter(.data$week == !!week_target, game_type == "REG") |>
    select(game_id, home_score, away_score, spread_line, total_line)

  week_eval <- model_week |>
    left_join(sched, by = "game_id") |>
    mutate(
      actual_margin   = home_score - away_score,
      actual_total    = home_score + away_score,
      spread_line_use = dplyr::coalesce(vegas_spread_line, spread_line),
      total_line_use  = dplyr::coalesce(vegas_total_line,  total_line),

      # ATS grading (+ push)
      ats_diff   = actual_margin + spread_line_use,
      ats_push   = is.finite(ats_diff) & ats_diff == 0,
      cover_home = ats_diff > 0,
      actual_cover_team = case_when(
        ats_push            ~ NA_character_,
        cover_home          ~ home_team,
        is.finite(ats_diff) ~ away_team,
        TRUE                ~ NA_character_
      ),
      ats_correct = dplyr::case_when(
        ats_push                         ~ NA,
        !is.na(ATS_pick) & ATS_pick == actual_cover_team ~ TRUE,
        !is.na(ATS_pick) & ATS_pick != actual_cover_team ~ FALSE,
        TRUE                              ~ NA
      ),

      spread_error = spread_mean - actual_margin,
      abs_error    = abs(spread_error),

      ATS_outcome = dplyr::case_when(
        ats_push    ~ NA_real_,
        ats_correct ~ 1,
        !ats_correct~ 0,
        TRUE        ~ NA_real_
      ),
      brier_ATS = ifelse(is.na(ATS_outcome) | is.na(ATS_prob), NA_real_, (ATS_prob - ATS_outcome)^2),

      # Totals grading (+ push)
      tot_diff  = actual_total - total_line_use,
      tot_push  = is.finite(tot_diff) & tot_diff == 0,
      went_over = tot_diff > 0,

      total_correct = dplyr::case_when(
        tot_push ~ NA,
        !is.na(Total_pick) & Total_pick == "Over"  & went_over  ~ TRUE,
        !is.na(Total_pick) & Total_pick == "Under" & !went_over ~ TRUE,
        !is.na(Total_pick)                                      ~ FALSE,
        TRUE ~ NA
      ),

      total_error     = total_mean - actual_total,
      abs_total_error = abs(total_error),

      p_over_for_brier = dplyr::case_when(
        !is.na(Total_pick) & Total_pick == "Over"  ~ Total_prob,
        !is.na(Total_pick) & Total_pick == "Under" ~ 1 - Total_prob,
        TRUE ~ NA_real_
      ),
      Total_outcome = dplyr::case_when(
        tot_push  ~ NA_real_,
        went_over ~ 1,
        !went_over~ 0,
        TRUE      ~ NA_real_
      ),
      brier_Total = ifelse(is.na(Total_outcome) | is.na(p_over_for_brier),
                           NA_real_, (p_over_for_brier - Total_outcome)^2)
    )

  ats_non_push <- dplyr::filter(week_eval, !is.na(ats_correct))
  tot_non_push <- dplyr::filter(week_eval, !is.na(total_correct))

  summary_stats <- tibble::tibble(
    ATS_MAE   = mean(week_eval$abs_error, na.rm = TRUE),
    ATS_RMSE  = sqrt(mean(week_eval$spread_error^2, na.rm = TRUE)),
    ATS_Brier = mean(week_eval$brier_ATS, na.rm = TRUE),
    ATS_Hit   = mean(ats_non_push$ats_correct, na.rm = TRUE),
    ATS_Bets  = nrow(ats_non_push),
    ATS_Push  = sum(week_eval$ats_push, na.rm = TRUE),

    Total_MAE   = mean(week_eval$abs_total_error, na.rm = TRUE),
    Total_RMSE  = sqrt(mean(week_eval$total_error^2, na.rm = TRUE)),
    Total_Brier = mean(week_eval$brier_Total, na.rm = TRUE),
    Total_Hit   = mean(tot_non_push$total_correct, na.rm = TRUE),
    Total_Bets  = nrow(tot_non_push),
    Total_Push  = sum(week_eval$tot_push, na.rm = TRUE)
  )

  list(
    per_game = week_eval |>
      dplyr::select(
        game_id, gameday, gametime, home_team, away_team,
        spread_line_use, total_line_use,
        spread_mean, actual_margin, spread_error, abs_error,
        ATS_pick, ATS_prob, ats_correct, ats_push, brier_ATS,
        total_mean, actual_total, total_error, abs_total_error,
        Total_pick, Total_prob, total_correct, tot_push, brier_Total
      ),
    summary = summary_stats
  )
}

# ---------- convenience: Excel -> evaluate ----------
# kind = "allgames" if you're pointing at the allgames.xlsx (already wide model output)
# kind = "card"     if you're pointing at a shorter bet-card export
evaluate_week_from_excel <- function(path, season_target, week_target, sheet = 1,
                                     kind = c("allgames","card")) {
  kind <- match.arg(kind)
  df <- if (kind == "allgames") {
    safe_read_excel(path, sheet = sheet) |> janitor::clean_names()
  } else {
    read_betcard_excel(path, sheet = sheet)
  }
  evaluate_week(df, season_target, week_target)
}

run_full_evaluation <- function(
    seasons,
    weeks,
    targets,
    sims = 2000,
    bankroll = 500,
    injuries = NULL,
    vegas_lines = NULL,
    settle_now = FALSE,
    thresholds = list(
      tot_high = 0.70, tot_low = c(0.55, 0.60),
      ats_high = 0.70, ats_mid = c(0.65, 0.70)
    ),
    base_dir = "weekly_output",
    american_odds = -110
) {
  library(dplyr)
  library(purrr)

  weekly_reports <- list()
  weekly_eval    <- list()

  for (ss in seasons) {
    for (wk in weeks) {
      message("\n=============================")
      message("Running Season ", ss, " Week ", wk)
      message("=============================\n")

      # ------------------------------------------------
      # 1. RUN WEEKLY REPORT (produces card + all_games)
      # ------------------------------------------------
      rr <- run_weekly_report(
        season       = ss,
        week         = wk,
        targets      = targets,
        sims         = sims,
        bankroll     = bankroll,
        injuries     = injuries,
        vegas_lines  = vegas_lines,
        settle_now   = settle_now,
        thresholds   = thresholds,
        base_dir     = base_dir,
        american_odds = american_odds
      )

      weekly_reports[[paste0(ss, "_", wk)]] <- rr

      # ------------------------------------------------
      # 2. EVALUATE THE WEEK using rr$all_games
      # ------------------------------------------------
      model_df <- rr$all_games

      if (!is.null(model_df) && is.data.frame(model_df)) {
        ew <- tryCatch(
          evaluate_week(
            model_df     = model_df,
            season_target = ss,
            week_target   = wk
          ),
          error = function(e) {
            warning(paste("evaluate_week failed for", ss, wk, ":", e$message))
            return(NULL)
          }
        )
      } else {
        warning(paste("all_games missing for", ss, wk, " → skipping evaluation"))
        ew <- NULL
      }

      weekly_eval[[paste0(ss, "_", wk)]] <- ew
    }
  }

  # ----------------------------------------------------
  # 3. COMPILE WEEK-BY-WEEK STATS INTO ONE TABLE
  # ----------------------------------------------------
  weekly_summary <- purrr::map_dfr(
    names(weekly_eval),
    function(key) {
      ew <- weekly_eval[[key]]
      if (is.null(ew) || is.null(ew$summary)) return(NULL)

      parts <- strsplit(key, "_")[[1]]
      season_val <- as.integer(parts[1])
      week_val   <- as.integer(parts[2])

      ew$summary %>%
        mutate(season = season_val, week = week_val, .before = 1)
    }
  ) %>% arrange(season, week)

  message("\n===== WEEKLY SUMMARY =====\n")
  print(weekly_summary)

  # ----------------------------------------------------
  # 4. OVERALL AGGREGATED SUMMARY
  # ----------------------------------------------------
  if (nrow(weekly_summary) > 0) {
    overall_summary <- tibble(
      ATS_HitRate      = mean(weekly_summary$ATS_Hit, na.rm = TRUE),
      ATS_MAE          = mean(weekly_summary$ATS_MAE, na.rm = TRUE),
      ATS_RMSE         = mean(weekly_summary$ATS_RMSE, na.rm = TRUE),
      ATS_Brier        = mean(weekly_summary$ATS_Brier, na.rm = TRUE),

      Total_HitRate    = mean(weekly_summary$Total_Hit, na.rm = TRUE),
      Total_MAE        = mean(weekly_summary$Total_MAE, na.rm = TRUE),
      Total_RMSE       = mean(weekly_summary$Total_RMSE, na.rm = TRUE),
      Total_Brier      = mean(weekly_summary$Total_Brier, na.rm = TRUE),

      Total_ATS_Bets   = sum(weekly_summary$ATS_Bets, na.rm = TRUE),
      Total_TOT_Bets   = sum(weekly_summary$Total_Bets, na.rm = TRUE)
    )

    message("\n===== OVERALL SUMMARY =====\n")
    print(overall_summary)
  } else {
    overall_summary <- NULL
    message("\n(No valid summaries accumulated.)")
  }

  return(list(
    weekly_reports  = weekly_reports,
    weekly_eval     = weekly_eval,
    weekly_summary  = weekly_summary,
    overall_summary = overall_summary
  ))
}


