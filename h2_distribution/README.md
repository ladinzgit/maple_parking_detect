# 가설 2: 현재 주차 후보 분포 검정

현재 주차 후보가 특정 레벨 구간과 직업 계열에 통계적으로 유의하게 집중되는지 검정한다.

## 입력

- `data/h1_current_candidates.csv`
  - 기본 라벨: `is_current_parking_candidate`
  - 민감도 라벨: `is_high_confidence_candidate`
- `data/main_characters.csv`
  - `level`, `class_group`

`is_stagnant_cluster`는 미접속 캐릭터를 포함하므로 주차 후보 분포 검정 라벨로 사용하지 않는다.

## 방법

1. `candidate_label × level_band` 카이제곱 독립성 검정
2. `candidate_label × class_group` 카이제곱 독립성 검정
3. 기대 빈도 5 미만 셀이 있으면 Fisher exact test 또는 범주 병합 검토
4. Cramer's V와 표준화 잔차를 함께 보고

## 수용 기준

최소 한 교차표에서 `p < 0.05`이고, 효과 크기와 표준화 잔차 해석이 함께 제시되어야 한다.
