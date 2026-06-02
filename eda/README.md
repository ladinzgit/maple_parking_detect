# EDA 요약

## 현재 H1 입력

H1의 주 입력은 성장 정체를 직접 표현하는 세 피처다.

| 피처 | 역할 |
|---|---|
| `log1p_avg_monthly_delta_cumexp` | 레벨 편향을 줄인 경험치 성장량 |
| `avg_monthly_delta_union_level` | 계정 성장량 |
| `avg_monthly_delta_hexa_frag` | HEXA 조각 소비 성장량 |

전투력과 어센틱심볼은 보조 EDA 및 민감도 분석에만 사용한다.

## 접속 행동 보완

기존 월 1회 접속 관측은 활동성을 과소 측정했다. `character/basic`을 월 `1, 8, 15, 22일`에
조회하도록 수집기를 수정했다.

- 원시 이력: `access_active_weeks`, `access_observed_weeks`
- 월별 활동: 월내 한 번이라도 접속이 관측되면 `access_flag=1`
- 전체 요약: `access_active_months`, `access_ratio`, `access_recent`

## 해석 제한

성장 정체 군집은 주차 후보 확정 라벨이 아니다. 미접속 캐릭터가 포함되므로 시간 분할 검증에서
반복 접속 조건을 추가해야 한다. 현재 결과는 `h1_clustering/README.md`와
`h1_clustering/temporal_external_validation.ipynb`를 기준으로 해석한다.
