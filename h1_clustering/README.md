# H1: 성장 정체 기반 주차 후보 탐색 (재설계 2026-06-04)

## 목적

H1은 **≥10/12 접속 통제** 270~290 본캐 표본에서 파워 재투자 하위 군집을 찾는다. 공개 API로 주간 보스 수행·메소 생산을 직접 볼 수 없으므로, H1 결과는 확정 주차 유저가 아니라 H2/H3에서 사용할 **후보 라벨**이다.

## 핵심 설계

- **접속 = 통제변인**: 표본 전원 `access_active_months ≥ 10` → 접속이 군집 축이 되지 못함. access family는 클러스터링 피처에서 제외.
- **주차 후보군 = 클러스터링 결과(`is_stagnant_cluster`) 자체** (절대-0 동결 게이트 폐기).

## 표본

- 본캐 표본: 2,000명 (전원 ≥10/12 접속) / 클러스터링 유효 표본: 1,999명

## 채택 피처

| Alias | 실제 값 |
|---|---|
| `cp_slog` | `sign(Δcp)·log1p(\|Δcp\|)` — 전투력 월평균 증가량(winsor) signed-log |
| `hexa_avg` | `avg_monthly_delta_hexa` clip≥0 (헥사 코어 레벨 합) |

- 전투력 Δ는 음수 24%(감소=주차) → **signed-log** 로 부호 보존+압축해야 살아남음(단독 sil 0.92, recall 0.99). 클러스터링 전 `StandardScaler`. 성장 정체 군집 = 스케일 중심좌표 합 최저.

## 결과

| 항목 | 값 |
|---|---:|
| K-Means k | 3 |
| Silhouette | 0.6431 |
| 주차 후보 수 | **471명 (23.6%)** |
| park(stag≥4) enrich / recall | 4.20x / **99.1%** |

| Cluster | 인원 | 해석 |
|---|---:|---|
| 비후보 | 1,528명 | 일반 성장 군집 (전투력·헥사 증가) |
| 주차 후보 | 471명 | **전투력 99.4% 감소** + 헥사 코어 미성장, 단 접속 12개월 만점(고활성) |

탐색 1위는 `[cp_slog, level_avg]`(sil 0.696)이나 해석 일관성 위해 `[cp_slog, hexa_avg]`(sil 0.643) 채택. 3+ 피처는 단조 열위. DBSCAN은 11군집·noise 5%로 약한 2-덩어리. 상세는 `RESULT.md`.

## 실행

```bash
jupyter nbconvert --to notebook --execute --inplace h1_clustering/feature_selection.ipynb
jupyter nbconvert --to notebook --execute --inplace h1_clustering/h1_clustering.ipynb
jupyter nbconvert --to notebook --execute --inplace h1_clustering/temporal_external_validation.ipynb
```

## 산출물

| 파일 | 역할 |
|---|---|
| `optimal_feature_set.json` | 채택셋 (`[cp_slog, hexa_avg]`, k=3, user-pinned) |
| `data/cluster_labels.csv` | `cluster_km`, `is_stagnant_cluster` (주차 후보군 471명) |
| `data/h1_current_candidates.csv` | 시간분할 안정성 부분집합 (현재성 52 / 고신뢰 39) |
| `figures/` | silhouette, PCA, DBSCAN 진단 그림 |
| `RESULT.md` | 최종 H1 결과 요약 |
