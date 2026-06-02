# 데이터 수집 및 H1 검증 계획

## 수집

`scripts/collect_features.py --refresh-raw`로 2,000명 × 12개월 이력을 수집한다.

월별 수집 항목:

| 항목 | 기준일 | 용도 |
|---|---|---|
| 레벨, 경험치 | 1일 | 누적 경험치 성장량 |
| 접속 여부 | 1, 8, 15, 22일 | 월내 활동 관측 |
| 전투력 | 1~7일 | 월 최대 전투력 |
| 유니온 레벨 | 1일 | 계정 성장량 |
| 어센틱심볼 | 1일 | 보조 성장량 |
| HEXA 코어 | 1일 | 보조 성장량 |

HEXA 조각 소비 이력은 `scripts/collect_hexa_fragments.py`로 별도 수집한다.

## H1 실행

```bash
python scripts/collect_features.py --refresh-raw
jupyter nbconvert --to notebook --execute --inplace h1_clustering/feature_selection.ipynb
jupyter nbconvert --to notebook --execute --inplace h1_clustering/h1_clustering.ipynb
jupyter nbconvert --to notebook --execute --inplace h1_clustering/temporal_external_validation.ipynb
```

## 판정

1. 전체 기간 성장 피처로 정체 군집을 탐색한다.
2. 6개월 분할로 장기 고정 상태 해석 가능성을 확인한다.
3. 분기 순차 검증으로 현재 시점 후보 탐색 성능을 확인한다.
4. 최신 분기 후보 파일을 H2 입력과 H3 최종 Rule 평가에 사용한다.
5. 전체 기간 성장 정체 군집 라벨을 H3 1단계 supervised 타깃으로 사용한다.

## H3 입력 규약 (지도 학습)

- **1단계 supervised 타깃** = `data/cluster_labels.csv`의 `is_stagnant_cluster` (성장 정체 군집, 394명).
- **1단계 입력 피처** = H1 미사용 성장·상태 피처 + 접속 피처(`access_active_months`, `access_ratio`, `access_recent`). 접속 중요도는 참고하되, 휴면을 사전 필터로 제거하지 않는다.
- **순환 방지 제외 피처** = H1 클러스터링 3피처(cumEXP·union delta·hexa_frag delta)와 동일 신호의 recent3/recent6 파생값. H1 군집 경계를 직접 복제하는 피처는 사용하지 않는다.
- **2단계 최종 Rule** = 1단계 성장 정체 Rule AND 최근 분기 반복 접속 게이트(`valid_access_active_months >= 2`).
- **평가 분리**:
  - 1단계 성장 정체 Rule: `is_stagnant_cluster` 대비 Precision / Recall / F1 / FPR / ROC-AUC.
  - 최종 주차 후보 Rule: `is_current_parking_candidate` 대비 평가. `is_high_confidence_candidate`는 보수적 민감도 분석에 사용한다.
- H3는 H1 군집을 다른 관측 피처로 근사하고 휴면을 제외하는 운영 Rule 실험이다. 실제 주차 유저 ground truth 검증으로 해석하지 않는다.
