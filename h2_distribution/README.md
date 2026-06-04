# H2: H1 후보군 분포 검정

## 목적

H2는 H1 클러스터링으로 도출한 주차 후보군이 특정 레벨 구간 또는 직업 계열에 불균일하게 분포하는지 검정한다.

## 입력

- `data/features_monthly.csv`
- `data/cluster_labels.csv` (기본 라벨)
- 보조: `data/h1_current_candidates.csv`

기본/보조 라벨:

- `is_stagnant_cluster`: H1 K-Means 주차 후보 cluster, **471명**
- `is_current_parking_candidate`: 시간분할 현재성 보조 후보, **52명**
- `is_high_confidence_candidate`: 현재성 고신뢰 부분집합, **39명**

## 현재 결과 (재실행 2026-06-04)

- 기본 분석 표본: 1,999명 / H1 클러스터 후보: **471명, 23.6%**

| 검정 | chi² | p | Holm 보정 p | Cramer's V | 판정 |
|---|---:|---:|---:|---:|---|
| 레벨 구간 x 후보 | 37.75 | 6.36e-9 | 1.27e-8 | 0.137 | 유의 |
| 직업 계열 x 후보 | 9.59 | 0.048 | 0.048 | 0.069 | 경계 유의(효과 무시) |

- **레벨 구간**: 후보 비율 `270-279` 30.1% → `280-285` 24.7% → `286-290` 16.0% **단조 감소**(저레벨 집중). Cramer's V 0.137(작은~중간 효과).
- **직업 계열**: Holm p=0.048로 경계 유의이나 **Cramer's V 0.069로 효과크기 무시 수준**(대표본 χ² 민감도). 실질적 집중 없음.

상세 수치는 `RESULTS.md`와 `results.json`.

## 실행

```bash
python h2_distribution/run_analysis.py
jupyter nbconvert --to notebook --execute --inplace h2_distribution/h2_distribution.ipynb
```
