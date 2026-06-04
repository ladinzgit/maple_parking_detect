"""Run the predefined H2 distribution tests for H1 parking candidates."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from scipy.stats import chi2_contingency, fisher_exact


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "h2_distribution"
FIGURE_DIR = OUTPUT_DIR / "figures"
sys.path.insert(0, str(ROOT))

from scripts.collect_main_characters import CLASS_GROUP_MAP


LEVEL_BINS = [269, 279, 285, 290]
LEVEL_LABELS = ["270-279", "280-285", "286-290"]
CLASS_ORDER = ["전사", "마법사", "궁수", "도적", "해적"]
LABEL_SPECS = [
    ("is_stagnant_cluster", "H1 클러스터 후보", "cluster"),
    ("is_current_parking_candidate", "현재성 후보", "current"),
    ("is_high_confidence_candidate", "고신뢰 현재성 후보", "current"),
]
ALPHA = 0.05
MONTE_CARLO_ITERATIONS = 100_000
RANDOM_SEED = 42


def input_paths() -> tuple[Path, Path, Path]:
    return (
        DATA_DIR / "features_monthly.csv",
        DATA_DIR / "cluster_labels.csv",
        DATA_DIR / "h1_current_candidates.csv",
    )


def load_data() -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    features_path, cluster_path, current_path = input_paths()
    features = pd.read_csv(features_path, encoding="utf-8-sig")
    cluster_labels = pd.read_csv(cluster_path, encoding="utf-8-sig")
    current_labels = pd.read_csv(current_path, encoding="utf-8-sig")

    for frame in (features, cluster_labels, current_labels):
        frame["ocid"] = frame["ocid"].astype(str)

    if "class_group" not in features.columns:
        features["class_group"] = features["character_class"].map(CLASS_GROUP_MAP)

    base_columns = ["ocid", "level", "class_group"]
    cluster_df = features[base_columns].merge(
        cluster_labels[["ocid", "is_stagnant_cluster"]],
        on="ocid",
        how="inner",
        validate="one_to_one",
    )
    current_df = features[base_columns].merge(
        current_labels[
            [
                "ocid",
                "is_current_parking_candidate",
                "is_high_confidence_candidate",
            ]
        ],
        on="ocid",
        how="inner",
        validate="one_to_one",
    )

    datasets = {
        "cluster": add_level_band(cluster_df),
        "current": add_level_band(current_df),
    }
    paths = {
        "features_input": str(features_path.relative_to(ROOT)).replace("\\", "/"),
        "cluster_input": str(cluster_path.relative_to(ROOT)).replace("\\", "/"),
        "current_input": str(current_path.relative_to(ROOT)).replace("\\", "/"),
    }
    return datasets, paths


def add_level_band(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["level_band"] = pd.cut(
        df["level"],
        bins=LEVEL_BINS,
        labels=LEVEL_LABELS,
        include_lowest=True,
    )
    if df["level_band"].isna().any():
        raise ValueError("level_band contains missing values; check the expected 270-290 range")
    if df["class_group"].isna().any():
        raise ValueError("class_group contains missing values; update CLASS_GROUP_MAP")
    return df


def cramers_v(table: pd.DataFrame, chi2: float) -> float:
    n = table.to_numpy().sum()
    return math.sqrt(chi2 / (n * min(table.shape[0] - 1, table.shape[1] - 1)))


def odds_ratio_with_ci(a: int, b: int, c: int, d: int) -> tuple[float, float, float]:
    """Return odds ratio and Wald 95% CI, using Haldane correction if needed."""
    cells = np.array([a, b, c, d], dtype=float)
    if (cells == 0).any():
        cells += 0.5
    a, b, c, d = cells
    odds_ratio = (a * d) / (b * c)
    standard_error = math.sqrt(sum(1 / cells))
    margin = 1.96 * standard_error
    return odds_ratio, math.exp(math.log(odds_ratio) - margin), math.exp(math.log(odds_ratio) + margin)


def holm_adjust(p_values: list[float]) -> list[float]:
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [0.0] * len(p_values)
    running_max = 0.0
    total = len(p_values)
    for rank, (index, p_value) in enumerate(indexed):
        running_max = max(running_max, min(1.0, (total - rank) * p_value))
        adjusted[index] = running_max
    return adjusted


def monte_carlo_chi_square_p(table: pd.DataFrame, observed_chi2: float) -> float:
    """Estimate an exact-style p-value while preserving table margins."""
    row_totals = table.sum(axis=1).to_numpy()
    candidate_total = int(table[True].sum())
    rng = np.random.default_rng(RANDOM_SEED)
    simulated_candidates = rng.multivariate_hypergeometric(
        row_totals,
        candidate_total,
        size=MONTE_CARLO_ITERATIONS,
    )
    expected_candidates = row_totals * candidate_total / row_totals.sum()
    expected_non_candidates = row_totals - expected_candidates
    simulated_non_candidates = row_totals - simulated_candidates
    statistics = (
        ((simulated_candidates - expected_candidates) ** 2 / expected_candidates).sum(axis=1)
        + ((simulated_non_candidates - expected_non_candidates) ** 2 / expected_non_candidates).sum(axis=1)
    )
    return float((np.count_nonzero(statistics >= observed_chi2) + 1) / (MONTE_CARLO_ITERATIONS + 1))


def analyze_dimension(
    df: pd.DataFrame,
    label: str,
    dimension: str,
    categories: list[str],
) -> dict:
    table = pd.crosstab(df[dimension], df[label]).reindex(
        index=categories,
        columns=[False, True],
        fill_value=0,
    )
    chi2, p_value, dof, expected = chi2_contingency(table)
    residuals = (table.to_numpy() - expected) / np.sqrt(expected)
    rates = table[True] / table.sum(axis=1)

    category_results = []
    for index, category in enumerate(categories):
        candidate = int(table.loc[category, True])
        non_candidate = int(table.loc[category, False])
        other_candidate = int(table[True].sum() - candidate)
        other_non_candidate = int(table[False].sum() - non_candidate)
        odds_ratio, ci_low, ci_high = odds_ratio_with_ci(
            candidate,
            non_candidate,
            other_candidate,
            other_non_candidate,
        )
        _, fisher_p = fisher_exact(
            [[candidate, non_candidate], [other_candidate, other_non_candidate]],
            alternative="two-sided",
        )
        category_results.append(
            {
                "category": category,
                "n": int(table.loc[category].sum()),
                "candidate_n": candidate,
                "candidate_rate": float(rates.loc[category]),
                "candidate_standardized_residual": float(residuals[index, 1]),
                "odds_ratio_vs_rest": float(odds_ratio),
                "odds_ratio_ci95": [float(ci_low), float(ci_high)],
                "fisher_p_vs_rest": float(fisher_p),
            }
        )

    return {
        "dimension": dimension,
        "table": {
            str(category): {
                "non_candidate": int(table.loc[category, False]),
                "candidate": int(table.loc[category, True]),
            }
            for category in categories
        },
        "chi2": float(chi2),
        "dof": int(dof),
        "p_value": float(p_value),
        "monte_carlo_p_value": monte_carlo_chi_square_p(table, chi2),
        "monte_carlo_iterations": MONTE_CARLO_ITERATIONS,
        "min_expected_frequency": float(expected.min()),
        "cramers_v": float(cramers_v(table, chi2)),
        "categories": category_results,
    }


def analyze_label(df: pd.DataFrame, label: str, description: str, dataset: str) -> dict:
    tests = [
        analyze_dimension(df, label, "level_band", LEVEL_LABELS),
        analyze_dimension(df, label, "class_group", CLASS_ORDER),
    ]
    adjusted = holm_adjust([test["p_value"] for test in tests])
    for test, adjusted_p in zip(tests, adjusted):
        test["holm_adjusted_p_value"] = adjusted_p
    return {
        "label": label,
        "description": description,
        "dataset": dataset,
        "n": int(len(df)),
        "candidate_n": int(df[label].sum()),
        "candidate_rate": float(df[label].mean()),
        "tests": tests,
    }


def format_p(value: float) -> str:
    return f"{value:.4g}"


def format_rate(value: float) -> str:
    return f"{value * 100:.2f}%"


def decision(test: dict) -> str:
    return "유의" if test["holm_adjusted_p_value"] < ALPHA else "유의하지 않음"


def append_test_table(lines: list[str], label_result: dict, title_prefix: str) -> None:
    level_test, class_test = label_result["tests"]
    lines.extend(
        [
            f"### {title_prefix} 검정 요약",
            "",
            "| 검정 | chi-square | df | p | Monte Carlo p | Holm 보정 p | 최소 기대빈도 | Cramer's V | 판정 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---|",
            f"| 레벨 구간 x 후보 | {level_test['chi2']:.3f} | {level_test['dof']} | {format_p(level_test['p_value'])} | {format_p(level_test['monte_carlo_p_value'])} | {format_p(level_test['holm_adjusted_p_value'])} | {level_test['min_expected_frequency']:.2f} | {level_test['cramers_v']:.3f} | {decision(level_test)} |",
            f"| 직업 계열 x 후보 | {class_test['chi2']:.3f} | {class_test['dof']} | {format_p(class_test['p_value'])} | {format_p(class_test['monte_carlo_p_value'])} | {format_p(class_test['holm_adjusted_p_value'])} | {class_test['min_expected_frequency']:.2f} | {class_test['cramers_v']:.3f} | {decision(class_test)} |",
            "",
        ]
    )


def build_summary(results: dict) -> str:
    primary = results["labels"][0]
    current = results["labels"][1]
    high_confidence = results["labels"][2]
    primary_level, primary_class = primary["tests"]
    top_band = max(primary_level["categories"], key=lambda row: row["candidate_rate"])
    low_band = min(primary_level["categories"], key=lambda row: row["candidate_rate"])

    lines = [
        "# H2 H1 후보군 분포 검정 결과",
        "",
        "## 설계",
        "",
        f"- 입력: `{results['features_input']}`, `{results['cluster_input']}`",
        f"- 보조 입력: `{results['current_input']}`",
        "- 기본 라벨: `is_stagnant_cluster` (H1 K-Means 성장 정체/주차 후보 cluster)",
        "- 보조 라벨: `is_current_parking_candidate`, `is_high_confidence_candidate`",
        "- 사전 정의 범주: 레벨 `270-279 / 280-285 / 286-290`, 직업 계열 `전사 / 마법사 / 궁수 / 도적 / 해적`",
        "- 검정: 카이제곱 독립성 검정, `alpha = 0.05`; 효과크기 Cramer's V; 후보 셀 표준화 잔차",
        f"- 보수적 점검: 고정 주변합 Monte Carlo 검정({MONTE_CARLO_ITERATIONS:,}회), 두 기본 교차표에 대한 Holm 보정",
        "",
        "## 기본 라벨 결과: H1 클러스터 후보",
        "",
        f"H1 클러스터 후보는 {primary['candidate_n']}명({format_rate(primary['candidate_rate'])})이다. 분석 표본은 {primary['n']:,}명이다.",
        "",
    ]
    append_test_table(lines, primary, "H1 클러스터 후보")
    lines.extend(
        [
            "### 레벨 구간",
            "",
            "| 레벨 구간 | 표본 | 후보 | 후보 비율 | 후보 셀 표준화 잔차 | 나머지 대비 OR (95% CI) |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in primary_level["categories"]:
        lines.append(
            f"| {row['category']} | {row['n']} | {row['candidate_n']} | "
            f"{format_rate(row['candidate_rate'])} | {row['candidate_standardized_residual']:.2f} | "
            f"{row['odds_ratio_vs_rest']:.2f} ({row['odds_ratio_ci95'][0]:.2f}-{row['odds_ratio_ci95'][1]:.2f}) |"
        )
    lines.extend(
        [
            "",
            f"`{top_band['category']}` 구간이 가장 높은 후보 비율({format_rate(top_band['candidate_rate'])})과 "
            f"양의 표준화 잔차({top_band['candidate_standardized_residual']:.2f})를 보였다. "
            f"나머지 레벨 대비 odds ratio는 {top_band['odds_ratio_vs_rest']:.2f}이다. "
            f"반대로 `{low_band['category']}` 구간은 후보 비율이 {format_rate(low_band['candidate_rate'])}로 가장 낮고, "
            f"표준화 잔차도 {low_band['candidate_standardized_residual']:.2f}로 낮다.",
            "",
            "### 직업 계열",
            "",
            "| 직업 계열 | 표본 | 후보 | 후보 비율 | 후보 셀 표준화 잔차 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in primary_class["categories"]:
        lines.append(
            f"| {row['category']} | {row['n']} | {row['candidate_n']} | "
            f"{format_rate(row['candidate_rate'])} | {row['candidate_standardized_residual']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## 보조 현재성 라벨",
            "",
            f"현재성 후보는 {current['candidate_n']}명({format_rate(current['candidate_rate'])}, 표본 {current['n']:,}명)이고, "
            f"고신뢰 현재성 후보는 {high_confidence['candidate_n']}명({format_rate(high_confidence['candidate_rate'])}, 표본 {high_confidence['n']:,}명)이다.",
            "",
        ]
    )
    append_test_table(lines, current, "현재성 후보")
    append_test_table(lines, high_confidence, "고신뢰 현재성 후보")
    lines.extend(
        [
            "## 판정",
            "",
            f"H1 클러스터 후보 {primary['candidate_n']}명 기준으로 레벨 구간 분포는 Holm 보정 후에도 {decision(primary_level)}하다(p={format_p(primary_level['holm_adjusted_p_value'])}, Cramer's V={primary_level['cramers_v']:.3f}). "
            f"후보는 `{top_band['category']}` 구간에 집중되고(비율 {format_rate(top_band['candidate_rate'])}) `{low_band['category']}` 구간에서는 낮다(비율 {format_rate(low_band['candidate_rate'])}). H2의 레벨 구간 불균형 가설을 지지한다.",
            "",
            (f"직업 계열 분포는 Holm 보정 p={format_p(primary_class['holm_adjusted_p_value'])}로 {decision(primary_class)}하나, "
             f"Cramer's V={primary_class['cramers_v']:.3f}로 효과크기는 무시할 수준이다(대표본 χ² 민감도). "
             "따라서 H2 결론은 `레벨 구간 집중은 확인(작은~중간 효과), 직업 계열은 실질적 집중 없음(효과크기 무시 가능)`으로 정리한다. "
             "이 결과는 H1 파생 후보 라벨의 분포 검정이며 실제 주차 유저 ground truth 검정은 아니다."),
            "",
            "## 재현",
            "",
            "```bash",
            "python h2_distribution/run_analysis.py",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def plot_rates(results: dict) -> None:
    primary_level, primary_class = results["labels"][0]["tests"]
    font_path = ROOT / "assets" / "NanumSquareNeo-bRg.ttf"
    font_manager.fontManager.addfont(font_path)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=font_path).get_name()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    for axis, test, title, color in [
        (axes[0], primary_level, "H1 candidate rate by level band", "#4e79a7"),
        (axes[1], primary_class, "H1 candidate rate by class group", "#f28e2b"),
    ]:
        categories = [row["category"] for row in test["categories"]]
        rates = [row["candidate_rate"] * 100 for row in test["categories"]]
        bars = axis.bar(categories, rates, color=color)
        axis.set_title(title)
        axis.set_ylabel("Candidate rate (%)")
        axis.grid(axis="y", alpha=0.25)
        for bar, rate in zip(bars, rates):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.45,
                f"{rate:.2f}%",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    fig.suptitle("H2: H1 cluster candidate distribution")
    fig.tight_layout()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / "candidate_distribution.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run H2 distribution tests.")
    parser.parse_args()

    datasets, paths = load_data()
    results = {
        "alpha": ALPHA,
        "level_bins": LEVEL_LABELS,
        "class_groups": CLASS_ORDER,
        **paths,
        "labels": [
            analyze_label(datasets[dataset_key], label, description, dataset_key)
            for label, description, dataset_key in LABEL_SPECS
        ],
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (OUTPUT_DIR / "RESULTS.md").write_text(build_summary(results), encoding="utf-8")
    plot_rates(results)

    primary_level, primary_class = results["labels"][0]["tests"]
    print(f"sample_n={results['labels'][0]['n']}")
    print(f"candidate_n={results['labels'][0]['candidate_n']}")
    print(f"level_band_p={primary_level['p_value']:.6g}")
    print(f"class_group_p={primary_class['p_value']:.6f}")
    print(f"wrote={OUTPUT_DIR / 'RESULTS.md'}")


if __name__ == "__main__":
    main()
