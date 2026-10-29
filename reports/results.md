### strategies_all_gws

| strategy | mean_pts_per_gw | total_pts | uplift_vs_last_week_% | uplift_vs_mean_last5_% | uplift_vs_top_price_% | uplift_vs_random_% |
|---|---|---|---|---|---|---|
| model | 58.474 | 2222 | 25.537 | 25.892 | 73.458 | 319.404 |
| model_robust_cal | 57.526 | 2186 | 23.503 | 23.853 | 70.648 | 312.609 |
| model_single | 53.132 | 2019 | 14.068 | 14.391 | 57.611 | 281.087 |
| mean_last5 | 46.447 | 1765 | -0.282 | 0 | 37.783 | 233.145 |
| last_week | 46.579 | 1770 | 0 | 0.283 | 38.173 | 234.088 |
| top_price | 33.711 | 1281 | -27.627 | -27.422 | 0 | 141.789 |
| random | 13.942 | 529.8 | -70.068 | -69.983 | -58.642 | 0 |
| oracle | 155 | 5890 | 232.768 | 233.711 | 359.797 | 1011.74 |

### strategies_gw6_38

| strategy | mean_pts_per_gw | total_pts | uplift_vs_last_week_% | uplift_vs_mean_last5_% | uplift_vs_top_price_% | uplift_vs_random_% |
|---|---|---|---|---|---|---|
| model | 58.333 | 1925 | 25.735 | 24.515 | 76.768 | 329.783 |
| model_robust_cal | 56.788 | 1874 | 22.404 | 21.216 | 72.084 | 318.397 |
| model_single | 54.939 | 1813 | 18.419 | 17.27 | 66.483 | 304.778 |
| mean_last5 | 46.848 | 1546 | 0.98 | 0 | 41.965 | 245.166 |
| last_week | 46.394 | 1531 | 0 | -0.97 | 40.588 | 241.817 |
| top_price | 33 | 1089 | -28.87 | -29.56 | 0 | 143.135 |
| random | 13.573 | 447.9 | -70.745 | -71.028 | -58.871 | 0 |
| oracle | 155.636 | 5136 | 235.467 | 232.212 | 371.625 | 1046.685 |

### significance_all_gws

| model_vs | mean_diff_pts_per_gw | ci95_low | ci95_high | model_better_gws | gws |
|---|---|---|---|---|---|
| last_week | 11.895 | 5.157 | 18.289 | 25 | 38 |
| mean_last5 | 12.026 | 6.157 | 17.632 | 27 | 38 |
| top_price | 24.763 | 18.5 | 31.026 | 33 | 38 |
| random | 44.532 | 38.018 | 51.039 | 37 | 38 |
| model_single | 5.342 | 0.763 | 10.026 | 23 | 38 |

### significance_gw6_38

| model_vs | mean_diff_pts_per_gw | ci95_low | ci95_high | model_better_gws | gws |
|---|---|---|---|---|---|
| last_week | 11.939 | 4.242 | 19.394 | 21 | 33 |
| mean_last5 | 11.485 | 4.818 | 18.001 | 23 | 33 |
| top_price | 25.333 | 18.121 | 32.636 | 28 | 33 |
| random | 44.761 | 37.412 | 52.249 | 32 | 33 |
| model_single | 3.394 | -1.485 | 8.546 | 18 | 33 |

### forecast_accuracy

| forecast | rmse | mae | rmse_GK | rmse_DEF | rmse_MID | rmse_FWD | rmse_played | rmse_zeros | rmse_blanks | rmse_tickers | rmse_haulers |
|---|---|---|---|---|---|---|---|---|---|---|---|
| two_stage | 1.938 | 0.978 | 1.409 | 2.124 | 1.871 | 2.1 | 2.97 | 0.746 | 1.559 | 1.471 | 5.592 |
| single_gbm | 1.942 | 0.99 | 1.413 | 2.127 | 1.873 | 2.109 | 2.974 | 0.75 | 1.552 | 1.495 | 5.605 |
| mean_last5 | 2.144 | 1.066 | 1.592 | 2.344 | 2.073 | 2.306 | 3.244 | 0.923 | 2.027 | 1.946 | 5.754 |
| last_week | 2.613 | 1.163 | 1.962 | 2.895 | 2.491 | 2.808 | 4.034 | 0.928 | 3.123 | 3.297 | 6.187 |

### forecast_dm

| model_vs | dm_se | p_se | model_better_gws_se | dm_ae | p_ae | model_better_gws_ae | gws |
|---|---|---|---|---|---|---|---|
| single_gbm | 0.907 | 0.37 | 22 | 4.512 | 0 | 33 | 38 |
| mean_last5 | 11.332 | 0 | 38 | 9.519 | 0 | 38 | 38 |
| last_week | 33.015 | 0 | 38 | 17.584 | 0 | 38 | 38 |

### interval_coverage

| group | coverage | raw_coverage | mean_width | mean_cqr_q | rows |
|---|---|---|---|---|---|
| ALL | 0.918 | 0.918 | 2.644 | 0 | 29338 |
| likely starters (P60 >= 0.5) | 0.846 | 0.846 | 6.799 | 0 | 7717 |
| others (P60 < 0.5) | 0.943 | 0.943 | 1.161 | 0 | 21621 |
| DEF | 0.908 | 0.908 | 2.854 | 0 | 9601 |
| FWD | 0.925 | 0.925 | 3.15 | 0 | 3235 |
| GK | 0.946 | 0.946 | 1.56 | 0 | 3383 |
| MID | 0.916 | 0.916 | 2.646 | 0 | 13119 |
| played (known only afterwards) | 0.808 | 0.808 | 5.523 | 0 | 11361 |

### cqr_blocks

| block | likely_starters | others |
|---|---|---|
| GW1-4 | 0 | 0 |
| GW5-8 | 0 | 0 |
| GW9-12 | 0 | 0 |
| GW13-16 | 0 | 0 |
| GW17-20 | 0 | 0 |
| GW21-24 | 0 | 0 |
| GW25-28 | 0 | 0 |
| GW29-32 | 0 | 0 |
| GW33-36 | 0 | 0 |
| GW37-38 | 0 | 0 |

### robust_calibrated

| strategy | mean | p10 | worst | sd |
|---|---|---|---|---|
| risk-neutral (Gamma = 0) | 58.474 | 33.7 | 18 | 19.874 |
| calibrated Gamma (80th pct of past budget used) | 57.526 | 36.4 | 18 | 19.17 |

### xp_audit

| forecast | corr_points_this_gw | corr_points_next_gw | share_of_edge_kept_next_gw | rmse_points_this_gw |
|---|---|---|---|---|
| two-stage model (pre-match) | 0.557 | 0.527 | 0.946 | 2.013 |
| FPL xP column | 0.734 | 0.478 | 0.651 | 1.648 |
| mean of last 5 (pre-match) | 0.461 | 0.46 | 0.998 | 2.229 |

### xp_partial

| xP vs | partial_corr_this_gw | partial_corr_next_gw |
|---|---|---|
| goals | 0.348 | 0.021 |
| bonus points | 0.43 | 0.042 |

### robust_frontier

| gamma | mean | p10 | worst | sd |
|---|---|---|---|---|
| 0 | 58.474 | 33.7 | 18 | 19.874 |
| 1 | 58.316 | 33.7 | 18 | 19.82 |
| 2 | 58.316 | 33.7 | 18 | 19.82 |
| 3 | 58.026 | 33.7 | 18 | 19.552 |
| 4 | 58.921 | 37.1 | 18 | 19.344 |
| 6 | 58.105 | 36.4 | 18 | 18.828 |
| 8 | 57.132 | 35.5 | 18 | 19.196 |

### sensitivity

| metric | value |
|---|---|
| gameweeks where the budget binds | 21 |
| budget shadow price, predicted pts per extra GBP 1m (median, all GWs) | 0.04 |
| budget shadow price (median, binding GWs only) | 0.092 |
| gameweeks where the club cap binds | 11 |
| club-cap cost, predicted pts (mean, all GWs) | 0.852 |
| club-cap cost (mean, binding GWs only) | 2.945 |
| club-cap cost (worst GW) | 15.546 |
| most players from one club with no cap, in the worst GW | 9 |
| mean XI cost (GBP m) | 80.487 |

### ewma_signals

| position | first_signal_gw | mean_error |
|---|---|---|
| DEF | 3 | 0.083 |
| FWD | nan | -0.019 |
| GK | nan | -0.005 |
| MID | nan | -0.024 |
