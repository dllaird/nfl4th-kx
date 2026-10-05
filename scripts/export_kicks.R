################################################################################
# Purpose: export every FG and extra-point attempt 2014-2025 for the kicker model
# Usage:   Rscript scripts/export_kicks.R   (run from the repo root)
# Writes:  data_cache/kicks_2014_2025.parquet
################################################################################
suppressMessages(library(dplyr))
pbp <- nflreadr::load_pbp(2014:2025)
kicks <- pbp |>
  filter(play_type %in% c("field_goal", "extra_point"), !is.na(kick_distance), !is.na(kicker_player_id)) |>
  transmute(
    game_id, play_id, season, week, season_type, game_date, posteam, defteam, home_team, away_team,
    kicker_player_id, kicker_player_name, kick_distance,
    is_xp = as.integer(play_type == "extra_point"),
    made = as.integer(if_else(play_type == "field_goal", field_goal_result == "made", extra_point_result == "good")),
    blocked = as.integer(coalesce(field_goal_result == "blocked", FALSE) | coalesce(extra_point_result == "blocked", FALSE)),
    roof, surface, temp, wind, qtr, game_seconds_remaining, score_differential
  )
dir.create("data_cache", showWarnings = FALSE)
nanoparquet::write_parquet(as.data.frame(kicks), "data_cache/kicks_2014_2025.parquet")
cat(nrow(kicks), "kicks;", sum(kicks$is_xp == 0), "FG attempts;", n_distinct(kicks$kicker_player_id), "kickers\n")
print(kicks |> filter(is_xp == 0) |> mutate(b = cut(kick_distance, c(0, 39, 49, 54, 59, 64, 80))) |> count(b, wt = NULL) |> left_join(kicks |> filter(is_xp == 0) |> mutate(b = cut(kick_distance, c(0, 39, 49, 54, 59, 64, 80))) |> group_by(b) |> summarize(rate = mean(made), max = max(kick_distance)), by = "b"))
