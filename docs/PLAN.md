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
4. 최신 분기 후보 파일을 H2/H3 입력으로 사용한다.
