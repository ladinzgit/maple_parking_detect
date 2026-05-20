# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Applied Data Analysis term project (응용데이터분석 텀프로젝트) that detects MapleStory "parking users" (주차 유저) — players who freeze their character's growth at a specific level to farm boss drops repeatedly — using Nexon OpenAPI data and unsupervised clustering.

Full project plan and hypothesis details are in `메이플스토리 주차 유저 클러스터링.md`.

**Three research hypotheses:**
1. Parking users form a distinct cluster separable by K-Means/DBSCAN on growth-change features
2. Parking users are unevenly distributed across level brackets and job classes (Chi-Square test)
3. A rule derived from Feature Importance can identify parking users with an acceptable False Positive Rate

## Running the Scripts

```powershell
# Install dependencies
pip install requests pandas python-dotenv scikit-learn scipy xgboost matplotlib seaborn

# Quick API connectivity test (collects 5 characters, sleeps 0.2s — use sparingly)
python test_collection.py

# Full main-character collection (collects 10 characters to main_characters.csv, sleeps 1.0s)
python collect_main_characters.py
```

## API Configuration

The Nexon OpenAPI key is stored in `.env` as `MAPLE_API_KEY`. All scripts load it via `python-dotenv`.

**Critical API constraints:**
- Rate limit: **500 req/s**, **20,000,000 req/day** — `collect_main_characters.py` uses a `RateLimiter` capped at 400 req/s (80% of limit) with 30 concurrent threads
- Data availability: last 2 years only; snapshots refresh daily around 08:00 KST
- `date` parameter format: `YYYY-MM-DD` (use yesterday or earlier — today's data may not be ready)
- History APIs (cube, starforce, potential) require account-owner authentication → **excluded from this project**

## Data Collection Architecture

### Current scripts

Both existing scripts follow the same 3-step pipeline per character:

1. **Fetch union ranking** (`ranking/union`) to get candidate character names
2. **Resolve OCID** (`id` endpoint) — permanent character identifier required for all other endpoints
3. **Identify main character** — fetch `character/basic` for current level, then `user/union-raider` for union block levels; a character is the "main" if no union block has a higher level than the character itself

Script differences:
- `test_collection.py` uses a single generic `get_data(endpoint, params)` wrapper with 429 retry logic
- `collect_main_characters.py` has per-endpoint functions and stricter 1.0s sleep

Output files: `main_characters.csv` (production), `test_main_characters.csv` (test run), both UTF-8 with BOM for Korean Excel compatibility.

### Next phase (not yet implemented)

Feature collection scripts need to be written. They must:
- Sample from `ranking/overall` (not just `ranking/union`) to cover diverse level brackets (240–260, 260–280, 280–300+)
- Collect multi-date snapshots using the `date` parameter to compute Δ features
- Call `character/stat` (전투력), `user/union`, `character/dojang`, `character/symbol-equipment` per character per date

Target sample size: 200–500 characters per level bracket, 2-year window (2023-05 to 2025-05).

## Planned Analysis Stack

| Phase | Endpoints Used | Output |
|---|---|---|
| Feature collection | `character/basic`, `character/stat`, `user/union`, `character/dojang`, `character/symbol-equipment` | Snapshot CSVs per date |
| Clustering (H1) | — | K-Means / DBSCAN labels, Elbow / Silhouette plots |
| Distribution test (H2) | — | Chi-Square p-values by level bracket and job class |
| Rule evaluation (H3) | — | Random Forest / XGBoost feature importance, Precision/Recall/FPR/ROC-AUC |

Key features for clustering: Δlevel, Δ전투력 (combat power), Δunion level, Δdojang floor, symbol growth — all measured as change over the 2-year collection window.
