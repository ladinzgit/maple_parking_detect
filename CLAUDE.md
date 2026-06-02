# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Applied Data Analysis term project (응용데이터분석 텀프로젝트) that detects MapleStory "parking users" (주차 유저) — players who freeze their character's growth at a specific level to farm boss drops repeatedly — using Nexon OpenAPI data and unsupervised clustering.

Full project plan and hypothesis details are in `docs/메이플스토리 주차 유저 클러스터링.md`.

**Three research hypotheses:**
1. Parking users form a distinct cluster separable by K-Means/DBSCAN on growth-change features
2. Parking users are unevenly distributed across level brackets and job classes (Chi-Square test)
3. A rule derived from Feature Importance can identify parking users with an acceptable False Positive Rate

## Running the Scripts

```powershell
# Install dependencies
pip install requests pandas python-dotenv scikit-learn scipy xgboost matplotlib seaborn numpy statsmodels

# Collect main characters (ranking/overall, level 270–290, 5계열×400=2,000명, writes data/main_characters.csv)
python scripts/collect_main_characters.py

# Collect 12-month monthly snapshots and compute delta features (writes data/features_monthly.csv, ~11 min)
python scripts/collect_features.py

# EDA (read-only — do NOT add experiment code here)
jupyter notebook eda/eda.ipynb

# H1 clustering experiment
jupyter notebook h1_clustering/h1_clustering.ipynb
```

**Data collection is complete.** Both CSV files exist in `data/` (gitignored).

## API Configuration

The Nexon OpenAPI key is stored in `.env` as `MAPLE_API_KEY`. All scripts load it via `python-dotenv`.

**Critical API constraints:**
- Rate limit: **500 req/s**, **20,000,000 req/day** — both scripts use a `RateLimiter` capped at 400 req/s with 30 concurrent threads
- Data availability: last 2 years only; snapshots refresh daily around 08:00 KST
- `date` parameter format: `YYYY-MM-DD` (use yesterday or earlier)
- History APIs (cube, starforce, potential) require account-owner authentication → excluded

## Data Collection Architecture

Both scripts share the same `RateLimiter` + `ThreadPoolExecutor(30)` pattern and a persistent `requests.Session` with a 60-connection pool.

**`collect_main_characters.py` (v2)** — `ranking/overall`-based main character collection, level 270–290 (`LEVEL_MIN=270`, `LEVEL_MAX=290`)

1. Binary-search page ranges per job class to locate the 270–290 band in `ranking/overall`
2. 5계열 × 400명 = 2,000명 target; per-계열: 3 level bins (270–279/280–285/286–290) each ~133명
3. Phase 1 (diversity): collect up to `max(MIN_PER_CLASS=10, 400÷job_count)` per job; Phase 2 (fill): round-robin up to `MAX_PER_CLASS=100`
4. For each candidate: `ranking/overall` → `id` → `character/basic` (create-date filter) → `user/union-raider` (main-char check)
5. Main-char filter: `max(union_block.block_level) <= character_level`
6. Create-date filter: `character_date_create > CREATE_CUTOFF=2025-06-30` → skip (new chars lack 12-month window)
7. Saves to `main_characters.csv` incrementally (UTF-8-BOM, deduped by OCID)

Design rationale: 270–290 covers the parking-signal window (low brackets carry the strongest stagnation signal) plus a high-level active baseline (286–290) for contrast. Note: a few stragglers down to level 260 remain in the data (12 rows <270), dropped by the 270–290 analysis filter.

**`collect_features.py` (v2.2)** — 12-month monthly snapshot collection

1. Reads `main_characters.csv`, skips OCIDs already in `features_monthly.csv`
2. 12-month window (not 24): current parking behavior matters; 24mo risks classifying "parked-then-returned" users
3. Per character × 12 months: `character/basic` + `character/stat` × 7 days (→ max combat power) + `user/union` + `character/symbol-equipment` + `character/hexamatrix`
4. Computes `avg_monthly_delta_*` (12mo) and `recent{3,6}_delta_*` (short-window slope)
5. Saves to `features_monthly.csv` incrementally every 100 characters

Total API calls: ~2,000 × 12 × 11 ≈ 264,000 → ~660 s at 400 req/s

### Output files

| File | Description |
|---|---|
| `data/main_characters.csv` | ~2,000 main characters, level 270–290, 5계열 균등 |
| `data/features_monthly.csv` | ~2,000 rows; 12-month delta features per character |
| `data/cluster_labels.csv` | `cluster_km`, `is_stagnant_cluster` per character (성장 정체 군집, written by H1 notebook) |
| `data/h1_current_candidates.csv` | `prior_stagnant_cluster`, `is_current_parking_candidate`, `is_high_confidence_candidate` (written by `temporal_external_validation.ipynb`; H2/H3 입력) |

### Feature columns in `features_monthly.csv`

| Column | Description |
|---|---|
| `level`, `union_level` | Latest snapshot value |
| `arcane_symbol_score`, `authentic_symbol_score` | Sum of symbol levels (latest) |
| `hexa_level_sum` | Sum of HEXA core levels (latest) — clean monotonic activity signal at 270+ |
| `avg_monthly_delta_level` | Parking signal: near-zero for parked users. **Replaced by cumEXP in clustering** (level-biased at high level) |
| `avg_monthly_delta_cumexp`, `log1p_avg_monthly_delta_cumexp` | Cumulative-EXP delta (via `exp_requirement_table.csv`) — level-unbiased parking signal. `log1p` form is the H1 Set A feature. cumEXP=0: parked 97.3% vs active 9.0% |
| `avg_monthly_delta_combat_power` | Key parking signal |
| `avg_monthly_delta_union_level` | Key parking signal |
| `avg_monthly_delta_arcane_symbol` | Symbol growth rate |
| `avg_monthly_delta_authentic_symbol` | Symbol growth rate — high-level (280+) discriminator |
| `avg_monthly_delta_hexa` | HEXA growth rate |
| `recent3_delta_*`, `recent6_delta_*` | Short-window (3/6 mo) slopes — age-debiased parking signal |
| `access_active_months`, `access_ratio`, `access_recent` | Recent login activity (from `access_flag` in `character/basic`) |
| `character_age_months`, `created_in_window` | Age diagnostics; `created_in_window=1` = new class (렌 cohort) |
| `first_valid_month`, `last_valid_month`, `num_valid_months` | Valid data window |

## Notebook Structure

**Rule: experiment code goes in hypothesis folders, never in `eda/eda.ipynb`.**

| Notebook | Role |
|---|---|
| `eda/eda.ipynb` | EDA only — distributions, correlations, preprocessing decisions. Do not add H1/H2/H3 code. |
| `h1_clustering/feature_selection.ipynb` | H1 피처 조합 완전탐색+greedy → 최적셋 선정 → `optimal_feature_set.json` |
| `h1_clustering/h1_clustering.ipynb` | H1: K-Means + DBSCAN clustering (최적셋 3피처) |
| `h1_clustering/temporal_external_validation.ipynb` | H1 외부검증: 시간분할 + 접속 활동 → `h1_current_candidates.csv` |
| `h2_distribution/` | H2: Chi-Square distribution tests (notebook not yet created) |
| `h3_rule/` | H3: Random Forest / XGBoost rule extraction (notebook not yet created) |

Each hypothesis notebook is self-contained: loads `data/features_monthly.csv` and reproduces preprocessing inline (no intermediate CSV hand-off from eda.ipynb).

## Clustering Decisions (H1 — adopted set confirmed 2026-06-02 via feature_selection.ipynb; clustering re-run complete on 270-290 data)

### Feature sets

- **Adopted clustering set (family-diverse, 3 features)** — `[log1p_avg_monthly_delta_cumexp, delta_union(=avg_monthly_delta_union_level, clip≥0), delta_hfrag(=avg_monthly_delta_hexa_frag, clip≥0)]`. Selected by `feature_selection.ipynb` (22-feature exhaustive + greedy under near-constant(dom_frac>0.70) + 비퇴화(maxfrac≤0.9) + family-당-1개 + pairwise|corr|<0.85 gates). K-Means **k=4, silhouette 0.6430**, 성장 정체 군집 **394명 (20.0%)**. Stored in `optimal_feature_set.json`. cumEXP=parked≈0, union=계정성장, hfrag=고렙 활성.
- **Set A (5-feature, v3) — superseded baseline**: `[log1p_avg_monthly_delta_cumexp, avg_monthly_delta_combat_power(raw), avg_monthly_delta_union_level, avg_monthly_delta_authentic_symbol, delta_hfrag]`. Earlier hand-curated set (silhouette 0.473). feature_selection 결과 3피처 셋이 **+0.17 silhouette** 우위 → cp/authentic는 lv270-290에서 잡음축이라 채택셋에서 제거. 비교용으로만 유지.
- **Feature Set A' (recent6) — age-debias 보조**: recent6 변형(cumexp/union/hfrag), 렌 코호트(2025-06 launch) age bias 점검용. clustering 채택 아님.

`arcane_stagnant` binary is **excluded from clustering** — binary feature dominates K-Means axis after StandardScaler, effectively reproducing a single boolean split rather than multi-dimensional clustering. The continuous `avg_monthly_delta_arcane_symbol` is **also excluded (v3)** — at lv270-290 arcane symbols are maxed (82.6% saturated) so Δarcane≈0 is near-constant noise. Both arcane features are retained for `stagnation_score` computation and post-hoc validation only.

`normalized_delta_level` is **excluded** — empirical P75 normalization is biased by parking-user concentration in the low-level brackets (wiki vs empirical mismatch). Superseded by `log1p_avg_monthly_delta_cumexp` for level-bias removal. Only propose it if the user explicitly asks.

`exp_rank_within_level` is **excluded** — Spearman r=−0.094 (p=0.001) with Δlevel: statistically significant at n=1,192 but negligible effect size (|r|<0.1).

`avg_monthly_delta_hexa_frag` (→ `delta_hfrag`) is **included in clustering (v3)** — the r≈0.93 with `avg_monthly_delta_hexa` is irrelevant because Δhexa is NOT in the feature set (no redundancy created); serves as high-level activity signal replacing arcane. `hexa_fragments_total` retained as H3 derived-feature candidate.

### Preprocessing

- `delta_cp`: clustering uses **raw** `avg_monthly_delta_combat_power` (no winsorize — preserve the 32.3% negative values as a distinct declining cluster). `delta_cp_winsor` = `winsorize(limits=[0.05, 0.05])` is still computed but used only for `stagnation_score`/validation.
- `delta_hfrag`: `avg_monthly_delta_hexa_frag` clamped ≥ 0 (from `hexa_fragments.csv`, merged in `preprocess()`); A' uses `recent6_delta_hexa_frag`
- `delta_cumexp`: clamp negative → 0 (calc artifact), then `log1p`
- `delta_union`, `delta_arcane`, `delta_authentic`: clamp to 0 (tiny negative artifact)
- `union_level` NaN → 0 (유니온 미가입)
- `arcane_stagnant` binary (validation only): `arcane_symbol_score < 120 AND avg_monthly_delta_arcane == 0`

### H1 Results (confirmed 2026-06-02, level 270-290)

- Adopted 3-feature set (cumEXP·union·hfrag), K-Means **k=4, silhouette 0.6430**, 성장 정체 군집 **394명 (20.0%)** of df_final 1,967 (클러스터링 표본 1,965).
- 성장 정체 군집 외부신호: park(stag≥4) **enrich 2.77x, recall 55.6%**. **성장 정체 군집 ≠ 주차 후보** — 휴면 캐릭터 포함(최근 접속 5.8%), 그대로 주차로 해석 금지.
- DBSCAN: 단일 군집 + noise 1.7% → 밀도 분리 없음 (K-Means는 거리 partition 효과). 별도 corroboration 아님.
- 외부검증(`temporal_external_validation.ipynb`): 분기 시간분할. 최신 분기만 유의 (OR 2.19, Fisher p=0.0195); 과거 분기 OR<1 → **시간 민감도 높음**. 결론 = 현재 시점 후보 탐색에 유효.

### v2 EDA Key Numbers (re-run 2026-06-02, level 270-290)

- **df_final (모델링 본판)**: 1,967명 (원본 2,000 → 270-290 필터 후 1,988 → 핵심 delta NaN 제거 후 1,967) | 3피처 dropna **클러스터링 표본 1,965명**
- **Δlevel=0 비율**: 35.6%
- **Triple-zero parked proxy**: 13.0% (255명, df_clean)
- **stagnation_score=5**: 63명 (3.2%)
- **created_in_window=1** (렌 코호트): 326명 / 2,000명 (16.3%)
- **cumEXP 채택**: `avg_monthly_delta_level` → `log1p_avg_monthly_delta_cumexp` (cumEXP=0: 파킹 후보 97.3% vs 활성 9.0%)
- **H2 사전 결과**: 레벨구간×파킹 χ²=167.5, p=4.28e-37 (**기각**, 25.1%→12.3%→1.1% 단조감소) / 직업×파킹 χ²=3.55, p=0.470 (**미기각** — 260-285 데이터의 직업 유의성은 270-290 필터에서 소멸)
- **hexa_fragments EDA**: r(delta_hexa_frag, delta_hexa) ≈ 0.93이나 Δhexa 미사용 → 중복 아님 → v3에서 `delta_hfrag` **클러스터링 채택** (arcane 대체)
- **exp_rank_within_level**: Δlevel과 r=−0.094 (무시 가능) → 피처 제외
- **hexa_fragments.csv** in `data/` — H3 추가 파생 피처 활용 가능 (hexa_fragments_total)

### Output files (updated)

| File | Description |
|---|---|
| `data/hexa_fragments.csv` | 2,000명 헥사 조각 소비량 (12개월 스냅샷 기반) |

## Remaining Analysis Phases

| Phase | Notebook | Status |
|---|---|---|
| H1 Clustering | `h1_clustering/` (3 notebooks) | **Done** — k=4, sil 0.6430, 정체군집 394명/20.0%, 외부검증 완료 |
| H2 Distribution test | `h2_distribution/` (create notebook) | Not started |
| H3 Rule evaluation | `h3_rule/` (create notebook) | Not started |

H2 uses `h1_current_candidates.csv` (from `temporal_external_validation.ipynb`) × level_band/class_group → Chi-Square (α=0.05).  
H3 uses `is_high_confidence_candidate` / `is_current_parking_candidate` (from `h1_current_candidates.csv`) as pseudo-labels → Random Forest/XGBoost → threshold rules → Precision/Recall/FPR/ROC-AUC.

## Other Files

- `docs/PLAN.md`: Historical design document. Superseded by implemented code — reference only.
- `h1_clustering/optimal_feature_set.json`: Adopted feature set + k + silhouette (written by `feature_selection.ipynb`, read by `h1_clustering.ipynb`).
- `assets/NanumSquareNeo-bRg.ttf`: Korean font used in all notebooks.
