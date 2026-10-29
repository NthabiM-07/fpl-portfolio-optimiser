### strategies_all_gws

| strategy | mean_pts_per_gw | total_pts | uplift_vs_last_week_% | uplift_vs_mean_last5_% | uplift_vs_top_price_% | uplift_vs_random_% |
|---|---|---|---|---|---|---|
| model | 58.474 | 2222 | 25.537 | 25.892 | 73.458 | 319.404 |
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

### forecast_accuracy

| forecast | rmse | mae | rmse_GK | rmse_DEF | rmse_MID | rmse_FWD | rmse_played | rmse_0-2 pts | rmse_3-4 pts | rmse_5+ pts |
|---|---|---|---|---|---|---|---|---|---|---|
| two_stage | 1.938 | 0.978 | 1.409 | 2.124 | 1.871 | 2.1 | 2.97 | 1.044 | 1.471 | 5.592 |
| single_gbm | 1.942 | 0.99 | 1.413 | 2.127 | 1.873 | 2.109 | 2.974 | 1.042 | 1.495 | 5.605 |
| mean_last5 | 2.144 | 1.066 | 1.592 | 2.344 | 2.073 | 2.306 | 3.244 | 1.333 | 1.946 | 5.754 |
| last_week | 2.613 | 1.163 | 1.962 | 2.895 | 2.491 | 2.808 | 4.034 | 1.84 | 3.297 | 6.187 |

### interval_coverage

| group | coverage | raw_coverage | mean_width |
|---|---|---|---|
| ALL | 0.918 | 0.918 | 2.644 |
| DEF | 0.908 | 0.908 | 2.854 |
| FWD | 0.925 | 0.925 | 3.15 |
| GK | 0.946 | 0.946 | 1.56 |
| MID | 0.916 | 0.916 | 2.646 |
| played only | 0.808 | 0.808 | 5.523 |

### robust_frontier

| gamma | mean | p10 | worst | sd |
|---|---|---|---|---|
| 0 | 58.474 | 33.7 | 18 | 19.874 |
| 1 | 58.474 | 33.7 | 18 | 19.874 |
| 2 | 58.474 | 34.7 | 18 | 19.659 |
| 3 | 57.711 | 37.1 | 18 | 19.081 |
| 4 | 58.184 | 37.7 | 18 | 19.115 |
| 6 | 57.632 | 34.4 | 18 | 18.771 |
| 8 | 56.816 | 35.5 | 18 | 18.605 |

### sensitivity

| metric | value |
|---|---|
| budget shadow price, predicted pts per extra GBP 1m (median) | 0.04 |
| gameweeks where budget binds | 21 |
| club-cap cost, predicted pts (mean) | 0.852 |
| gameweeks where club cap binds | 11 |
| mean XI cost (GBP m) | 80.487 |

### ewma_signals

| position | first_signal_gw | mean_error |
|---|---|---|
| DEF | 3 | 0.083 |
| FWD | nan | -0.019 |
| GK | nan | -0.005 |
| MID | nan | -0.024 |
