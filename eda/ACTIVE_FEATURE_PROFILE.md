# Active Feature Profile

## Sample

- main sample: 2,000
- feature rows matched to current sample: 2,000
- monthly raw rows matched to current sample: 24,000
- active collection filter: access observed in all checkpoint months `2025-06`, `2025-12`, `2026-05`

### Level Bands

| level_band | n |
| --- | --- |
| 270-279 | 665 |
| 280-285 | 665 |
| 286-290 | 670 |

### Class Groups

| class_group | n |
| --- | --- |
| 전사 | 400 |
| 마법사 | 400 |
| 궁수 | 400 |
| 도적 | 400 |
| 해적 | 400 |

### Checkpoint Access Rate In Current Sample

| year_month | access_rate_pct |
| --- | --- |
| 2025-06 | 100.000 |
| 2025-12 | 100.000 |
| 2026-05 | 100.000 |

## H1 Feature Summary

| feature | non_null | missing_pct | zero_pct | mean | std | min | p25 | median | p75 | p95 | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| log1p_avg_monthly_delta_cumexp | 1999 | 0.050 | 0.450 | 29.444 | 2.911 | 0.000 | 28.704 | 29.944 | 30.963 | 31.725 | 32.329 |
| avg_monthly_delta_union_level | 1999 | 0.050 | 1.550 | 62.040 | 99.192 | -163.182 | 18.091 | 32.364 | 52.636 | 257.555 | 860.100 |
| access_ratio | 2000 | 0.000 | 0.000 | 0.834 | 0.219 | 0.083 | 0.729 | 0.938 | 1.000 | 1.000 | 1.000 |

## Support Feature Summary

| feature | non_null | missing_pct | zero_pct | mean | std | min | p25 | median | p75 | p95 | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| avg_monthly_delta_level | 2000 | 0.000 | 7.900 | 0.638 | 1.003 | 0.000 | 0.273 | 0.455 | 0.818 | 1.640 | 26.000 |
| avg_monthly_delta_combat_power | 2000 | 0.000 | 0.250 | 2106286.824 | 5718249.749 | -60876555.455 | -226.636 | 883207.409 | 3866586.273 | 11464830.955 | 51911829.545 |
| avg_monthly_delta_authentic_symbol | 2000 | 0.000 | 18.400 | 0.836 | 0.976 | -2.182 | 0.091 | 0.545 | 1.182 | 2.727 | 6.545 |
| avg_monthly_delta_hexa | 2000 | 0.000 | 7.250 | 2.341 | 2.861 | -2.636 | 0.455 | 1.364 | 3.182 | 7.727 | 30.091 |
| avg_monthly_delta_hexa_frag | 2000 | 0.000 | 7.950 | 152.139 | 235.455 | -72.273 | 25.886 | 68.136 | 180.727 | 575.614 | 3246.364 |
| access_active_months | 2000 | 0.000 | 0.000 | 11.232 | 1.546 | 3.000 | 11.000 | 12.000 | 12.000 | 12.000 | 12.000 |
| access_active_weeks | 2000 | 0.000 | 0.000 | 39.971 | 10.474 | 4.000 | 35.000 | 45.000 | 48.000 | 48.000 | 48.000 |
| num_valid_months | 2000 | 0.000 | 0.000 | 12.000 | 0.000 | 12.000 | 12.000 | 12.000 | 12.000 | 12.000 | 12.000 |
| character_age_months | 2000 | 0.000 | 0.000 | 64.996 | 46.599 | 11.000 | 27.000 | 53.000 | 94.000 | 161.000 | 269.000 |

## Generated Figures

- `eda/figures/active_h1_feature_distributions.png`
- `eda/figures/active_support_feature_distributions.png`

## Notes

- `features_monthly.csv` and `hexa_fragments.csv` can contain old OCIDs because collectors append and deduplicate rather than pruning removed samples. This profile filters all analysis to the current `main_characters.csv` OCID set.
- `avg_monthly_delta_combat_power` is heavy-tailed and can be negative because monthly stat snapshots are noisy; downstream analysis should keep the existing winsorization/clipping step.
- H1 (재설계 2026-06) uses `cp_slog` (= `sign(x)·log1p(|x|)` of winsorized `avg_monthly_delta_combat_power`) and `hexa_avg` (= clip≥0 `avg_monthly_delta_hexa`). Access is a **control variable** (sample restricted to `access_active_months ≥ 10`), not a clustering feature.
