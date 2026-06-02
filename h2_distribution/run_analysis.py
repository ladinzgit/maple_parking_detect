"""Run the predefined H2 distribution tests for current parking candidates."""

from __future__ import annotations

import json
import math
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

LEVEL_BINS = [269, 279, 285, 290]
LEVEL_LABELS = ["270-279", "280-285", "286-290"]
CLASS_ORDER = ["전사", "마법사", "궁수", "도적", "해적"]
LABELS = [
    ("is_current_parking_candidate", "기본 라벨"),
    ("is_high_confidence_candidate", "고신뢰 민감도 라벨"),
]
ALPHA = 0.05
MONTE_CARLO_ITERATIONS = 100_000
RANDOM_SEED = 42


def load_data() -> pd.DataFrame:
    main = pd.read_csv(DATA_DIR / "main_characters.csv")
    candidates = pd.read_csv(DATA_DIR / "h1_current_candidates.csv")
    df = main[["ocid", "level", "class_group"]].merge(
        candidates[["ocid", *[label for label, _ in LABELS]]],
        on="ocid",
        how="inner",
        validate="one_to_one",
    )
    df["level_band"] = pd.cut(
        df["level"],
        bins=LEVEL_BINS,
        labels=LEVEL_LABELS,
        include_lowest=True,
    )
    if df["level_band"].isna().any():
        raise ValueError("level_band contains missing values; check the expected 270-290 range")
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


def analyze_label(df: pd.DataFrame, label: str, description: str) -> dict:
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
        "n": int(len(df)),
        "candidate_n": int(df[label].sum()),
        "candidate_rate": float(df[label].mean()),
        "tests": tests,
    }


def format_p(value: float) -> str:
    return f"{value:.4g}"


def build_summary(results: dict) -> str:
    primary = results["labels"][0]
    sensitivity = results["labels"][1]
    primary_level, primary_class = primary["tests"]
    sensitivity_level, sensitivity_class = sensitivity["tests"]
    middle_band = primary_level["categories"][1]

    lines = [
        "# H2 현재 주차 후보 분포 검정 결과",
        "",
        "## 설계",
        "",
        "- 입력: `data/main_characters.csv`, `data/h1_current_candidates.csv`",
        f"- 분석 표본: {primary['n']:,}명",
        "- 기본 라벨: `is_current_parking_candidate`",
        "- 사전 정의 범주: 레벨 `270-279 / 280-285 / 286-290`, 직업 계열 `전사 / 마법사 / 궁수 / 도적 / 해적`",
        "- 검정: 카이제곱 독립성 검정, `alpha = 0.05`; 효과크기 Cramer's V; 후보 셀 표준화 잔차",
        f"- 보수적 점검: 고정 주변합 Monte Carlo 검정({MONTE_CARLO_ITERATIONS:,}회), 두 기본 교차표에 대한 Holm 보정, `is_high_confidence_candidate` 민감도 분석",
        "",
        "## 기본 라벨 결과",
        "",
        f"현재 후보는 {primary['candidate_n']}명({primary['candidate_rate'] * 100:.2f}%)이다.",
        "",
        "| 검정 | chi-square | df | p | Monte Carlo p | Holm 보정 p | 최소 기대빈도 | Cramer's V | 판정 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
        f"| 레벨 구간 x 후보 | {primary_level['chi2']:.3f} | {primary_level['dof']} | {format_p(primary_level['p_value'])} | {format_p(primary_level['monte_carlo_p_value'])} | {format_p(primary_level['holm_adjusted_p_value'])} | {primary_level['min_expected_frequency']:.2f} | {primary_level['cramers_v']:.3f} | 원검정 유의 |",
        f"| 직업 계열 x 후보 | {primary_class['chi2']:.3f} | {primary_class['dof']} | {format_p(primary_class['p_value'])} | {format_p(primary_class['monte_carlo_p_value'])} | {format_p(primary_class['holm_adjusted_p_value'])} | {primary_class['min_expected_frequency']:.2f} | {primary_class['cramers_v']:.3f} | 유의하지 않음 |",
        "",
        "### 레벨 구간",
        "",
        "| 레벨 구간 | 표본 | 후보 | 후보 비율 | 후보 셀 표준화 잔차 | 나머지 대비 OR (95% CI) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in primary_level["categories"]:
        lines.append(
            f"| {row['category']} | {row['n']} | {row['candidate_n']} | "
            f"{row['candidate_rate'] * 100:.2f}% | {row['candidate_standardized_residual']:.2f} | "
            f"{row['odds_ratio_vs_rest']:.2f} ({row['odds_ratio_ci95'][0]:.2f}-{row['odds_ratio_ci95'][1]:.2f}) |"
        )
    lines.extend(
        [
            "",
            f"`280-285` 구간이 가장 높은 후보 비율({middle_band['candidate_rate'] * 100:.2f}%)과 "
            f"양의 표준화 잔차({middle_band['candidate_standardized_residual']:.2f})를 보였다. "
            f"나머지 레벨 대비 odds ratio는 {middle_band['odds_ratio_vs_rest']:.2f}이다.",
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
            f"{row['candidate_rate'] * 100:.2f}% | {row['candidate_standardized_residual']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## 민감도 분석",
            "",
            f"고신뢰 후보는 {sensitivity['candidate_n']}명({sensitivity['candidate_rate'] * 100:.2f}%)이다.",
            "",
            "| 검정 | chi-square | p | Monte Carlo p | Holm 보정 p | 최소 기대빈도 | Cramer's V |",
            "|---|---:|---:|---:|---:|---:|---:|",
            f"| 레벨 구간 x 고신뢰 후보 | {sensitivity_level['chi2']:.3f} | {format_p(sensitivity_level['p_value'])} | {format_p(sensitivity_level['monte_carlo_p_value'])} | {format_p(sensitivity_level['holm_adjusted_p_value'])} | {sensitivity_level['min_expected_frequency']:.2f} | {sensitivity_level['cramers_v']:.3f} |",
            f"| 직업 계열 x 고신뢰 후보 | {sensitivity_class['chi2']:.3f} | {format_p(sensitivity_class['p_value'])} | {format_p(sensitivity_class['monte_carlo_p_value'])} | {format_p(sensitivity_class['holm_adjusted_p_value'])} | {sensitivity_class['min_expected_frequency']:.2f} | {sensitivity_class['cramers_v']:.3f} |",
            "",
            "## 판정",
            "",
            "사전 정의한 기본 레벨 구간 검정은 `p < 0.05`로 H2 수용 기준을 충족한다. "
            "현재 후보는 균일하게 분포하지 않으며 `280-285` 구간에 상대적으로 집중된다. "
            "직업 계열 집중은 확인되지 않았다.",
            "",
            "해석은 제한적으로 유지해야 한다. 레벨 효과의 Cramer's V는 작고, 두 기본 교차표에 Holm 보정을 "
            "적용하면 레벨 검정도 `p >= 0.05`이다. 고신뢰 라벨 분석 역시 유의하지 않다. "
            "고신뢰 직업 교차표는 최소 기대빈도가 5보다 작아 Monte Carlo p를 우선 참고한다. "
            "따라서 결과는 현재 표본에서의 탐색적 운영 신호이며, 독립 시점 데이터로 재검증해야 한다.",
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
        (axes[0], primary_level, "Candidate rate by level band", "#4e79a7"),
        (axes[1], primary_class, "Candidate rate by class group", "#f28e2b"),
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
                bar.get_height() + 0.08,
                f"{rate:.2f}%",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    fig.suptitle("H2: current parking candidate distribution")
    fig.tight_layout()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / "candidate_distribution.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df = load_data()
    results = {
        "alpha": ALPHA,
        "level_bins": LEVEL_LABELS,
        "class_groups": CLASS_ORDER,
        "labels": [analyze_label(df, label, description) for label, description in LABELS],
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (OUTPUT_DIR / "RESULTS.md").write_text(build_summary(results), encoding="utf-8")
    plot_rates(results)

    primary_level, primary_class = results["labels"][0]["tests"]
    print(f"sample_n={len(df)}")
    print(f"candidate_n={int(df[LABELS[0][0]].sum())}")
    print(f"level_band_p={primary_level['p_value']:.6f}")
    print(f"class_group_p={primary_class['p_value']:.6f}")
    print(f"wrote={OUTPUT_DIR / 'RESULTS.md'}")


if __name__ == "__main__":
    main()
