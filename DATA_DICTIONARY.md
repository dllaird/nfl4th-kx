# DATA_DICTIONARY — `data_cache/dataset/plays_2025.parquet`

One row per play (see DATASET.md for scope and timing definitions). Unless noted:
- "posteam" is the **pre-play** possession team;
- prices are P(team wins) + ½·P(tie), on a 0–1 scale;
- times are Unix seconds (UTC).

## Identifiers and game context (from nflfastR)
| column | description |
|---|---|
| game_id, play_id | nflfastR identifiers |
| season, week, season_type, game_date | game info |
| home_team, away_team, posteam, defteam | nflverse team codes |
| roof | nflfastR roof |
| home_opening_kickoff | 1 if the home team received the opening kickoff |
| home_ticker, away_ticker | the Kalshi winner markets used for this game |

## Pre-play state (nflfastR)
| column | description |
|---|---|
| qtr, down, ydstogo, yardline_100, goal_to_go | situation (`yardline_100` = yards from the opponent's end zone) |
| quarter_seconds_remaining, half_seconds_remaining, game_seconds_remaining | clock |
| posteam_score, defteam_score, score_differential | score before the play (differential = posteam − defteam) |
| total_home_score, total_away_score | score before the play |
| posteam_timeouts_remaining, defteam_timeouts_remaining | timeouts |
| receive_2h_ko | 1 if posteam receives the 2nd-half kickoff (1st half only) |
| is_home_pos | posteam is the home team |

## Play and outcome (nflfastR)
| column | description |
|---|---|
| play_type, play_type_nfl, desc | play type and description |
| yards_gained, first_down, touchdown, td_team, pass_touchdown, rush_touchdown, return_touchdown | gain and scoring |
| interception, fumble_lost, safety | turnovers and safeties |
| penalty, penalty_team, penalty_yards | penalties |
| fourth_down_converted, fourth_down_failed, third_down_converted, third_down_failed | conversions |
| field_goal_result, kick_distance, extra_point_result, two_point_conv_result | kicking outcomes |
| punt_blocked, punt_inside_twenty, touchback | punting outcomes |
| aborted_play, qb_kneel, qb_spike | special plays |
| spread_line, total_line | **betting** lines: reference only, never a model input |
| ep, epa, wp, vegas_wp, vegas_home_wp, wpa | nflfastR model values: benchmark/reference only |

## Post-play state (the next play's pre-play state, same game)
| column | description |
|---|---|
| post_posteam, post_defteam, post_qtr, post_down, post_ydstogo, post_yardline_100, post_goal_to_go | situation at the next snap (from the **post** posteam's view) |
| post_quarter/half/game_seconds_remaining | clock at the next snap |
| post_posteam_score, post_defteam_score, post_score_differential, post_total_home_score, post_total_away_score | score at the next snap |
| post_posteam_timeouts_remaining, post_defteam_timeouts_remaining | timeouts at the next snap |
| post_same_posteam, post_pre_pos_has_ball | the pre-play posteam has the ball at the next snap |
| post_score_differential_pre_view | post score differential from the pre-play posteam's view |
| next_play_id | play_id of the next play |

## Timing
| column | description |
|---|---|
| time_of_day, snap_ts | snap time (ISO string / Unix seconds) |
| end_clock_time, end_ts | whistle time (NaN for 10.8% of plays) |
| next_snap_ts | next play's snap |
| prev_end_ts | previous play's whistle |
| settle_time_s | next snap − whistle (or − snap if the whistle time is missing) |

## Winner-market prices
| column | description |
|---|---|
| pre_price, post_price | **targets:** P(pre-play posteam wins) just before the snap / once settled before the next snap |
| pre_logit, post_logit | logit of the above, clipped to [0.001, 0.999] |
| pre_price_source, post_price_source | `trades_vwap`, `last_trade` or `none`. Evaluate models by source. |
| pre_home_vwap, post_home_vwap | home-team VWAP in the window (raw, home view) |
| pre_home_last, post_home_last | last home-view trade price before the reference time |
| pre_n_trades, post_n_trades, pre_volume, post_volume | trades and contracts in the window |
| pre_age_s, post_age_s | age of the newest trade in the window |
| pre_last_age_s, post_last_age_s | age of the last trade before the reference time |
| pre_dispersion, post_dispersion | volume-weighted sd of trade prices in the window (bid/ask bounce) |
| candle_bid, candle_ask, candle_spread, candle_age_s | home-market 1-minute candle quote closed at or before the snap: best YES bid/ask, width and age |
| candle_mid_pos | candle midpoint converted to posteam's view |

## Kalshi pregame team strength (fixed for the game; Phase 3)
| column | description |
|---|---|
| kalshi_spread_line | implied mean final margin, home − away (same sign as nflverse `spread_line`) |
| kalshi_total_line | implied mean final total |
| kalshi_posteam_spread | implied margin from posteam's view (conversion-model `posteam_spread`) |
| kalshi_posteam_total | (total + posteam_spread) / 2 (conversion-model `posteam_total`) |
| kalshi_spread_quality, kalshi_total_quality | 0–1 ladder fit quality |

Per-game fit details (sd, model, RMSE, strikes used, quote age, winner-vs-ladder gap) are
in `data_cache/kalshi_pregame_2025.parquet`.

## Flags
| column | true when |
|---|---|
| usable | pre and post prices exist, pre price not stale, times monotonic, not unsettled |
| flag_unsettled | next snap < 25 s after the whistle |
| flag_pre_stale, flag_post_stale | no trade within 60 s before the reference time (proxy for a halt or thin market) |
| flag_nonmonotonic_time | snap times out of order with neighboring plays |
| flag_coarse_time | snap time has whole-second precision (likely hand-entered) |
| flag_missing_end | no whistle time |
| flag_replay, flag_reversed | review mentioned / ruling reversed |
| flag_penalty, flag_no_play | penalty on the play / no-play |
| flag_qtr_change | the next snap is in a different quarter (includes halftime) |
| flag_last_play | last play of the game: no post state or post price |
