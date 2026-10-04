################################################################################
# Purpose: build parity fixtures for the Python port and run the R package on them
# Usage:   Rscript scripts/make_fixtures.R   (run from the repo root)
# Writes:  tests/fixtures/fixtures_4th.parquet  inputs (add_4th_probs Path A columns)
#          tests/fixtures/games.parquet         schedule snapshot used by R
#          tests/fixtures/r_outputs.parquet     add_4th_probs() outputs
#          tests/fixtures/r_outputs_2pt.parquet add_2pt_probs() outputs
#          tests/fixtures/wp_states.parquet     calculate_win_probability() on
#                                               perturbed states (WP unit parity)
################################################################################

suppressMessages({
  library(dplyr)
})
set.seed(4)

fix_dir <- file.path("tests", "fixtures")
dir.create(fix_dir, recursive = TRUE, showWarnings = FALSE)

nfl4th_ns <- asNamespace("nfl4th")
games <- get(".games_nfl4th", nfl4th_ns)()

pbp <- nflreadr::load_pbp(2023:2025) |>
  filter(down == 4, !is.na(posteam), !is.na(yardline_100), qtr <= 5) |>
  mutate(type = if_else(season_type == "REG", "reg", "post")) |>
  select(
    game_id, play_id, home_team, away_team, posteam, type, season, qtr,
    quarter_seconds_remaining, game_seconds_remaining, ydstogo, yardline_100,
    score_differential, home_opening_kickoff, posteam_timeouts_remaining,
    defteam_timeouts_remaining
  ) |>
  filter(!is.na(score_differential), !is.na(posteam_timeouts_remaining))

reg <- pbp |> filter(qtr <= 4, game_seconds_remaining > 15)

strata <- list(
  random = reg |> slice_sample(n = 1200),
  end_q2 = reg |> filter(qtr == 2, quarter_seconds_remaining <= 120) |> slice_sample(n = 150),
  end_q4 = reg |> filter(qtr == 4, quarter_seconds_remaining <= 180) |> slice_sample(n = 250),
  goal_to_go = reg |> filter(ydstogo == yardline_100) |> slice_sample(n = 150),
  backed_up = reg |> filter(yardline_100 >= 90) |> slice_sample(n = 120),
  long_fg = reg |> filter(between(yardline_100, 35, 52)) |> slice_sample(n = 200),
  no_fg = reg |> filter(yardline_100 >= 53, yardline_100 < 90) |> slice_sample(n = 60),
  ot = pbp |> filter(qtr == 5) |> slice_sample(n = 40)
)

fixtures <- bind_rows(strata, .id = "stratum") |>
  distinct(game_id, play_id, .keep_all = TRUE) |>
  mutate(runoff = 0)

# synthetic runoff rows: late-game conversions with clock run-off
runoff_rows <- reg |>
  filter(qtr == 4, quarter_seconds_remaining <= 300) |>
  slice_sample(n = 100) |>
  mutate(stratum = "runoff", runoff = 40)

fixtures <- bind_rows(fixtures, runoff_rows) |>
  mutate(
    fixture_id = row_number(),
    r_supported = qtr <= 4
  ) |>
  select(-game_seconds_remaining)

cli::cli_alert_info("{nrow(fixtures)} fixture plays")
print(count(fixtures, stratum))

r_in <- fixtures |> select(-stratum, -fixture_id, -r_supported, -game_id, -play_id)
out <- nfl4th::add_4th_probs(r_in)
r_outputs <- bind_cols(
  fixtures |> select(fixture_id),
  out |> select(
    go_boost, first_down_prob, wp_fail, wp_succeed, go_wp, fg_make_prob,
    miss_fg_wp, make_fg_wp, fg_wp, punt_wp
  )
)

out2 <- nfl4th::add_2pt_probs(r_in |> filter(fixtures$r_supported) |> slice_head(n = 600))
r_outputs_2pt <- bind_cols(
  fixtures |> filter(r_supported) |> slice_head(n = 600) |> select(fixture_id),
  out2 |> select(wp_0, wp_1, wp_2, conv_1pt, conv_2pt, wp_go1, wp_go2)
)

# WP unit parity: perturbed post-play-like states through calculate_win_probability
prepared <- r_in |>
  filter(fixtures$r_supported) |>
  get("prepare_df", nfl4th_ns)()
n_states <- 2000
wp_states <- prepared |>
  slice_sample(n = n_states, replace = TRUE) |>
  mutate(
    state_id = row_number(),
    flip = rbinom(n(), 1, 0.4),
    posteam = if_else(flip == 1, if_else(posteam == home_team, away_team, home_team), posteam),
    score_differential = if_else(flip == 1, -score_differential, score_differential) +
      sample(c(-7, -3, 0, 0, 3, 7), n(), replace = TRUE),
    down = 1,
    yardline_100 = sample(1:99, n(), replace = TRUE),
    ydstogo = if_else(yardline_100 < 10, yardline_100, 10L),
    half_seconds_remaining = pmax(half_seconds_remaining - sample(0:40, n(), replace = TRUE), 0),
    game_seconds_remaining = if_else(qtr <= 2, half_seconds_remaining + 1800, half_seconds_remaining)
  ) |>
  select(-flip)

wp_out <- get("calculate_win_probability", nfl4th_ns)(wp_states)

wp_cols <- c(
  "state_id", "home_team", "away_team", "posteam", "original_posteam", "season",
  "qtr", "down", "ydstogo", "yardline_100", "score_differential",
  "half_seconds_remaining", "game_seconds_remaining", "elapsed_share",
  "spread_line", "home_opening_kickoff", "home_receive_2h_ko",
  "home_timeouts_remaining", "away_timeouts_remaining",
  "posteam_timeouts_remaining", "defteam_timeouts_remaining", "roof", "vegas_wp"
)
wp_fixture <- wp_out |> select(all_of(wp_cols))

nanoparquet::write_parquet(as.data.frame(fixtures), file.path(fix_dir, "fixtures_4th.parquet"))
nanoparquet::write_parquet(
  as.data.frame(games |> filter(season >= 2023) |> mutate(fg_model_roof = as.character(fg_model_roof))),
  file.path(fix_dir, "games.parquet")
)
nanoparquet::write_parquet(as.data.frame(r_outputs), file.path(fix_dir, "r_outputs.parquet"))
nanoparquet::write_parquet(as.data.frame(r_outputs_2pt), file.path(fix_dir, "r_outputs_2pt.parquet"))
nanoparquet::write_parquet(as.data.frame(wp_fixture), file.path(fix_dir, "wp_states.parquet"))
cli::cli_alert_success("fixtures written to {fix_dir}")
