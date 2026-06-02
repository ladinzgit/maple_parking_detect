# Repository Guidelines

## Project Structure & Module Organization

This repository is a Python and Jupyter analysis project for detecting MapleStory parking candidates from Nexon OpenAPI data.

- `scripts/`: API-backed data collectors. Run these in order when rebuilding datasets.
- `eda/`: exploratory notebook, exported Markdown, and generated figures.
- `h1_clustering/`: feature selection, clustering, and temporal validation notebooks plus committed analysis figures.
- `h2_distribution/` and `h3_rule/`: hypothesis-specific design notes; add notebooks and outputs here as those analyses are implemented.
- `docs/`: project design, collection plan, and reference data.
- `assets/`: committed fonts, spreadsheets, and report-ready charts.
- `data/`: local generated CSV files. CSV files are ignored by Git because they may be large or contain sensitive data.

## Build, Test, and Development Commands

Create a virtual environment and install the analysis stack:

```bash
pip install requests pandas python-dotenv scikit-learn scipy xgboost matplotlib seaborn statsmodels numpy openpyxl jupyter
```

Common workflows:

```bash
python scripts/collect_main_characters.py
python scripts/collect_features.py --refresh-raw
python scripts/collect_hexa_fragments.py
jupyter notebook eda/eda.ipynb
jupyter nbconvert --to notebook --execute --inplace h1_clustering/h1_clustering.ipynb
```

Collectors call the live API and can take several minutes. Reuse existing local CSV files unless a fresh snapshot is required.

## Coding Style & Naming Conventions

Use Python 3, four-space indentation, UTF-8 source files, and `snake_case` for functions, variables, and CSV columns. Keep configuration constants near the top of collector scripts in `UPPER_SNAKE_CASE`. Prefer `pathlib.Path` for repository-relative paths. Notebook names should describe their analysis stage, such as `feature_selection.ipynb`.

No formatter or linter is configured. Keep edits focused and follow the surrounding style.

## Testing Guidelines

There is currently no automated test suite. Validate script changes with `python -m compileall -q scripts` and run the affected collector on a limited sample when possible. For notebook changes, execute the notebook with `jupyter nbconvert --execute --inplace` and review generated tables and figures.

## Commit & Pull Request Guidelines

Recent commits use short imperative summaries, often with Conventional Commit prefixes such as `feat:`, `fix:`, and `docs:`. Keep commits scoped to one analysis or documentation change. Pull requests should explain the hypothesis or pipeline stage affected, list validation commands, and include updated charts when results change. Do not commit `.env` or generated CSV files.

## Security & Configuration

Set `MAPLE_API_KEY=...` in a local `.env` file. Never log, commit, or paste API keys. Treat `data/` exports as sensitive research artifacts.
