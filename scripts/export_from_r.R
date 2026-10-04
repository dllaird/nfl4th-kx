################################################################################
# Purpose: export nfl4th / nflfastR model artifacts for the Python port
# Usage:   Rscript scripts/export_from_r.R   (run from the repo root)
# Writes:  nfl4th_py/data/*.json.gz (xgboost JSON), *.parquet, MANIFEST.json
#
# xgboost boosters carry no feature names, so feature order is positional.
# The orders below are copied from the select() calls in R/decision_functions.R,
# R/apply_win_prob.R and nflfastR 6.0.0 (wp_spread_model_select, ep_model_select).
################################################################################

suppressMessages({
  library(xgboost)
  library(dplyr)
})

out_dir <- file.path("nfl4th_py", "data")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

sysdata <- new.env()
load("R/sysdata.rda", envir = sysdata)

features <- list(
  fd_model = c(
    "down", "ydstogo", "yardline_100", "era3", "era4", "outdoors",
    "retractable", "dome", "posteam_spread", "total_line", "posteam_total"
  ),
  two_pt_model = c(
    "era2", "era3", "era4", "outdoors", "retractable", "dome",
    "posteam_spread", "total_line", "posteam_total"
  ),
  # select() de-duplicates the repeated home_timeouts_remaining, so 11 features
  home_wp_model = c(
    "home_receive_2h_ko", "spread_time", "home_posteam",
    "half_seconds_remaining", "game_seconds_remaining", "Diff_Time_Ratio",
    "home_score_differential", "home_ep", "ydstogo", "home_yardline_100",
    "home_timeouts_remaining"
  ),
  ep_model = c(
    "half_seconds_remaining", "yardline_100", "home", "retractable", "dome",
    "outdoors", "ydstogo", "era0", "era1", "era2", "era3", "era4", "down1",
    "down2", "down3", "down4", "posteam_timeouts_remaining",
    "defteam_timeouts_remaining"
  ),
  wp_spread_model = c(
    "receive_2h_ko", "spread_time", "home", "half_seconds_remaining",
    "game_seconds_remaining", "Diff_Time_Ratio", "score_differential", "down",
    "ydstogo", "yardline_100", "posteam_timeouts_remaining",
    "defteam_timeouts_remaining"
  )
)

nfl4th_ns <- asNamespace("nfl4th")
raw_models <- list(
  fd_model = get("load_fd_model", nfl4th_ns)(),
  two_pt_model = sysdata$two_pt_model,
  home_wp_model = get("load_wp_model", nfl4th_ns)(),
  ep_model = fastrmodels::ep_model,
  wp_spread_model = fastrmodels::wp_model_spread
)

sha256 <- function(path) {
  # tools::md5sum is base R; use openssl-free sha256 via digest if present
  if (requireNamespace("digest", quietly = TRUE)) {
    digest::digest(file = path, algo = "sha256")
  } else {
    unname(tools::md5sum(path))
  }
}

manifest <- list(
  generated_at = format(Sys.time(), tz = "UTC", usetz = TRUE),
  r_version = R.version.string,
  packages = list(
    nfl4th = as.character(packageVersion("nfl4th")),
    nflfastR = as.character(packageVersion("nflfastR")),
    fastrmodels = as.character(packageVersion("fastrmodels")),
    xgboost = as.character(packageVersion("xgboost")),
    mgcv = as.character(packageVersion("mgcv"))
  ),
  models = list()
)

for (name in names(raw_models)) {
  booster <- xgb.load.raw(raw_models[[name]])
  n_feat <- as.integer(
    xgb.config(booster)$learner$learner_model_param$num_feature
  )
  stopifnot(n_feat == length(features[[name]]))
  # gzip the JSON: fd_model.json alone is ~60 MB uncompressed
  json_path <- tempfile(fileext = ".json")
  xgb.save(booster, json_path)
  path <- file.path(out_dir, paste0(name, ".json.gz"))
  con <- gzfile(path, "wb", compression = 9)
  writeBin(readBin(json_path, "raw", file.info(json_path)$size), con)
  close(con)
  unlink(json_path)
  manifest$models[[name]] <- list(
    file = basename(path),
    features = features[[name]],
    num_feature = n_feat,
    objective = xgb.config(booster)$learner$objective$name,
    num_class = as.integer(
      xgb.config(booster)$learner$learner_model_param$num_class
    ),
    rounds = xgb.get.num.boosted.rounds(booster),
    hash = sha256(path)
  )
  cli::cli_alert_success("{name}: {n_feat} features -> {path}")
}

# FG GAM -> exact lookup table. Inputs are an integer yardline and a 4-level
# factor, so evaluating every combination reproduces the model exactly.
fg_grid <- expand.grid(
  yardline_100 = 1:99,
  fg_model_roof = levels(sysdata$fg_model$model$fg_model_roof),
  stringsAsFactors = FALSE
) |>
  mutate(fg_model_roof = factor(fg_model_roof, levels = levels(sysdata$fg_model$model$fg_model_roof)))
fg_grid$fg_make_prob <- as.numeric(
  mgcv::predict.bam(sysdata$fg_model, newdata = fg_grid, type = "response")
)
fg_grid$fg_model_roof <- as.character(fg_grid$fg_model_roof)
nanoparquet::write_parquet(fg_grid, file.path(out_dir, "fg_lookup.parquet"))
manifest$fg_lookup <- list(
  file = "fg_lookup.parquet",
  rows = nrow(fg_grid),
  hash = sha256(file.path(out_dir, "fg_lookup.parquet"))
)

punt <- as.data.frame(sysdata$punt_df)
nanoparquet::write_parquet(punt, file.path(out_dir, "punt_df.parquet"))
manifest$punt_df <- list(
  file = "punt_df.parquet",
  rows = nrow(punt),
  hash = sha256(file.path(out_dir, "punt_df.parquet"))
)

jsonlite::write_json(
  manifest,
  file.path(out_dir, "MANIFEST.json"),
  auto_unbox = TRUE,
  pretty = TRUE
)
cli::cli_alert_success("wrote {file.path(out_dir, 'MANIFEST.json')}")
