# H3: Feature Importance 기반 Rule 평가

## 상태

H3는 아직 구현 예정 단계다.

## 목적

H3는 H1의 성장 정체 후보 cluster를 다른 관측 피처로 근사하고, 현재성 접속 조건을 결합해 운영 가능한 rule을 만드는 실험이다. 분류기 성능 자체가 목적이 아니라, feature importance를 통해 단순하고 설명 가능한 rule 후보를 도출하는 것이 목적이다.

## 입력 라벨

| 라벨 | 파일 | 현재 인원 | 용도 |
|---|---|---:|---|
| `is_stagnant_cluster` | `data/cluster_labels.csv` | 471명 | 1단계 성장 정체 rule 학습 target |
| `is_current_parking_candidate` | `data/h1_current_candidates.csv` | 52명 | 최종 rule 평가 target |
| `is_high_confidence_candidate` | `data/h1_current_candidates.csv` | 39명 | 민감도 분석 target |

## 설계 원칙

- H1에서 직접 선택한 `cp_slog`(전투력 Δ signed-log), `hexa_avg`(헥사 코어 Δ) 및 동일 신호 파생(combat_power/hexa family)을 그대로 복제하는 피처는 leakage 위험이 있으므로 입력에서 제외한다.
- 입력 후보는 전투력, 레벨, 어센틱, HEXA, 캐릭터 나이, 접속 행동 등 H1을 간접적으로 근사할 수 있는 피처로 구성한다.
- 접속 여부는 수집 단계의 사전 필터로 추가 제거하지 않는다. 최종 rule에서 별도 현재성 게이트로 사용한다.

## 예정 방법

1. Random Forest / XGBoost 학습
2. 5-fold stratified CV
3. class imbalance 처리
4. SHAP 또는 permutation importance 계산
5. 상위 2~3개 피처로 단순 rule 후보 생성
6. 최종 rule = 성장 정체 rule AND 현재성 접속 게이트
7. Precision, Recall, F1, FPR, ROC-AUC 및 threshold sweep 보고

## 해석 주의

본 프로젝트에는 확정 주차 유저 ground truth가 없다. 따라서 H3의 모든 지표는 H1/H2에서 만든 후보 라벨 대비 성능이며, 실제 주차 유저 검증 성능으로 해석하지 않는다.
