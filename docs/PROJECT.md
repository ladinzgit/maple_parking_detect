# 메이플스토리 성장 정체 기반 주차 후보 탐색

## 연구 목적

Nexon OpenAPI로 직접 관측 가능한 성장 이력과 접속 행동을 이용해 주차 후보를 탐색한다.
공개 API로 주간 보스 수행 기록과 메소 생산량을 직접 조회할 수 없으므로 결과는 주차 유저 확정
라벨이 아니라 운영 검토용 후보 라벨이다.

## H1 설계

H1은 두 단계로 검증한다.

1. 누적 경험치, 유니온, HEXA 조각 소비 성장량으로 성장 정체 군집을 탐색한다.
2. 시간 분할 검증 구간에서 성장 정체와 반복 접속이 함께 관측되는 현재 후보를 선별한다.

현재 클러스터링 피처:

```python
CLUSTER_FEATURES = [
    "log1p_avg_monthly_delta_cumexp",
    "avg_monthly_delta_union_level",
    "avg_monthly_delta_hexa_frag",
]
```

접속 행동은 `character/basic`의 최근 7일 접속 여부를 월 `1, 8, 15, 22일`에 조회한다.
월내 한 번이라도 접속이 관측되면 해당 월을 활동 월로 처리한다.

## H1 결과

전체 12개월 K-Means 결과:

| 항목 | 값 |
|---|---:|
| 분석 표본 | 1,965명 |
| 최적 k | 4 |
| silhouette | 0.6430 |
| 성장 정체 군집 | 394명 (20.0%) |

현재 지향 분기 외부 검증:

| 학습 구간 | 검증 구간 | 엄격 후보 | 직전 정체 군집 포착 | Odds ratio | Fisher 단측 p |
|---|---|---:|---:|---:|---:|
| 2025-06 ~ 2025-08 | 2025-09 ~ 2025-11 | 34명 | 9명 | 0.99 | 0.5730 |
| 2025-09 ~ 2025-11 | 2025-12 ~ 2026-02 | 36명 | 10명 | 0.57 | 0.9588 |
| 2025-12 ~ 2026-02 | 2026-03 ~ 2026-05 | 35명 | 23명 | 2.19 | 0.0195 |

최신 분기에서는 H1이 지지되지만 과거 분기 재현성은 일관되지 않다. 따라서 H1 결과는 현재 시점
후보 탐색에 사용하고, 시간 민감도를 명시한다.

## 출력 계약

| 파일 | 역할 |
|---|---|
| `data/cluster_labels.csv` | 전체 기간 성장 정체 군집 라벨 |
| `data/h1_current_candidates.csv` | 최신 분기 현재 후보와 고신뢰 후보 라벨 |
| `h1_clustering/temporal_external_validation.ipynb` | 시간 분할 외부 검증 |

H2에서 주차 후보 분포를 분석할 때는 `is_current_parking_candidate` 또는
`is_high_confidence_candidate`를 명시적으로 선택한다. H3에서는 `is_stagnant_cluster`를
1단계 성장 정체 Rule 학습 타깃으로 사용하고, 최종 Rule을 현재 후보 라벨로 평가한다.
`is_stagnant_cluster` 자체를 주차 후보 확정 라벨로 해석하지 않는다.

## H2 결과

`data/h1_current_candidates.csv`의 현재 후보 35명을 기준으로 사전 정의한 분포 검정을 수행했다.

| 검정 | chi-square | p | Holm 보정 p | Cramer's V | 판정 |
|---|---:|---:|---:|---:|---|
| 레벨 구간 × 현재 후보 | 6.995 | 0.0303 | 0.0606 | 0.060 | 원검정 유의 |
| 직업 계열 × 현재 후보 | 3.034 | 0.5522 | 0.5522 | 0.040 | 유의하지 않음 |

레벨 `280-285` 구간의 후보 비율은 2.90%로 가장 높고, 후보 셀 표준화 잔차는 2.08이다.
고신뢰 후보 23명으로 다시 분석하면 레벨 구간 검정은 `p=0.0637`로 유의하지 않다. 따라서 H2는
사전 정의한 원검정 수용 기준을 충족하지만 효과가 작고 보수적 점검에서 약해지는 탐색적 결과로
해석한다. 상세 결과와 재현 명령은 `h2_distribution/RESULTS.md`에 기록한다.

## H3 설계 (지도 학습) — 성장 정체 Rule + 접속 게이트

주차는 *성장 정체 + 활성 접속*의 2차원 신호다. 성장 정체 군집만으로는 주차 후보와 휴면이
분리되지 않는다(정체 군집의 약 94%가 휴면). 따라서 휴면을 **수집·전처리 단계에서 사전 필터로
제거하지 않고**, H1 성장 정체 군집을 먼저 다른 관측 피처로 근사한 뒤 최근 분기 반복 접속 게이트를
결합한다. 접속 피처는 분류기 입력에도 넣어 상대 중요도를 확인한다.

- **분류기**: Random Forest / XGBoost, 5-fold stratified CV.
- **1단계 supervised 타깃**: `is_stagnant_cluster` (성장 정체 군집, 394명).
- **입력 피처**: H1 미사용 성장·상태 피처(Δlevel, Δcombat_power, Δauthentic_symbol, Δhexa,
  union_level, level, authentic_symbol_score, hexa_level_sum, character_age_months) + 접속 피처
  (`access_active_months`, `access_active_weeks`, `access_ratio`, `access_recent`).
- **순환 방지**: 클러스터링 3피처(cumEXP·union delta·hexa_frag delta)와 동일 신호의
  recent3/recent6 파생값을 제외한다.
- **2단계 최종 Rule**: 1단계 성장 정체 Rule AND 최근 분기 반복 접속
  (`valid_access_active_months >= 2`).
- **평가**: 1단계 Rule은 `is_stagnant_cluster`, 최종 Rule은 `is_current_parking_candidate`,
  보수적 민감도 분석은 `is_high_confidence_candidate` 대비 Precision / Recall / F1 / FPR /
  ROC-AUC를 보고한다. 수용 기준은 최종 Rule Precision > 0.95 AND FPR < 5%.
- **한계**: ground truth 없음 → 지표는 H1 파생 라벨 대비값이다. H3는 H1 군집을 근사하고
  휴면을 제외하는 운영 Rule 실험이며, 실제 주차 유저를 독립적으로 검증하지 않는다.
