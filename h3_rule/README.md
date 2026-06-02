# 가설 3 (지도 학습) — Feature Importance 기반 Rule 평가

Feature Importance로 도출한 단순 Rule이 H1 성장 정체 군집을 근사하고, 반복 접속 게이트를 결합해 일반 유저 피해를 최소화하면서 현재 주차 후보를 선별할 수 있는지 평가한다. 분류기는 핵심 피처 식별 도구이며, **본 평가 대상은 2단계 Rule** 이다.

## 설계 원칙 — 성장 정체 Rule과 접속 게이트 분리

주차의 정의는 **성장 정체 + 활성 접속**의 2차원 신호다. 성장 정체 군집만으로는 주차 후보와 휴면(접속 끊긴) 캐릭터가 분리되지 않는다(H1 결과: 정체 군집의 약 94%가 휴면).

→ H3는 먼저 H1 성장 정체 군집을 H1 미사용 피처로 근사한다. 접속 피처(`access_active_months`, `access_ratio`, `access_recent`)도 분류기 입력에 넣어 중요도를 확인하되, 최종 주차 후보 Rule에서는 최근 분기 반복 접속을 별도 게이트로 명시한다. 휴면을 **수집·전처리 단계에서 사전 필터로 제거하지 않는다.**

## 입력

- **1단계 supervised 타깃 (positive)**: `data/cluster_labels.csv`의 `is_stagnant_cluster` (H1 성장 정체 군집, 394명).
- **최종 Rule 평가 라벨**: `data/h1_current_candidates.csv`의 `is_current_parking_candidate`. 보수적 민감도 분석은 `is_high_confidence_candidate`.
- **입력 피처**: `data/features_monthly.csv`의 H1 미사용 관측 피처 — `avg_monthly_delta_level`, `avg_monthly_delta_combat_power`, `avg_monthly_delta_authentic_symbol`, `avg_monthly_delta_hexa`, `union_level`, `level`, `authentic_symbol_score`, `hexa_level_sum`, `character_age_months`, 그리고 접속 피처 `access_active_months` / `access_active_weeks` / `access_ratio` / `access_recent`.
- **순환 방지 제외 피처**:
  - H1 직접 사용: `log1p_avg_monthly_delta_cumexp`, `avg_monthly_delta_union_level`, `avg_monthly_delta_hexa_frag`.
  - 동일 신호 파생값: `avg_monthly_delta_cumexp`, `recent3_delta_cumexp`, `log1p_recent3_delta_cumexp`, `recent6_delta_cumexp`, `log1p_recent6_delta_cumexp`, `recent3_delta_union_level`, `recent6_delta_union_level`, `hexa_fragments_total`, `recent3_delta_hexa_frag`, `recent6_delta_hexa_frag`.
- 전제: H1 K-Means 성장 정체 군집은 운영 Rule 근사 대상으로 사용한다. H1의 제한된 시간 재현성과 DBSCAN 불일치는 H3 해석 한계로 유지한다.

## 방법

1. **분류기 학습** (importance 추출 용도, 지도 학습)
   - Random Forest, XGBoost
   - 5-fold stratified CV
   - supervised 타깃 = H1 `is_stagnant_cluster`
   - **class imbalance 처리**: `class_weight='balanced'` / `scale_pos_weight` 적용, stratified fold로 fold별 positive 보존
2. **Feature Importance 산출**
   - SHAP values (mean |SHAP|) 또는 permutation importance
   - H1 미사용 성장·상태 피처와 접속 피처의 상대 중요도를 함께 확인
   - 상위 2~3개 핵심 피처 선정
3. **1단계 성장 정체 Rule 도출**
   - H1 미사용 피처만 사용한 규칙, 예: `Δlevel <= threshold AND Δcombat_power <= threshold`
   - 임계값은 핵심 피처 분포의 분위수에서 후보 → grid search
4. **2단계 최종 주차 후보 Rule 도출**
   - `stagnant_growth_rule AND valid_access_active_months >= 2`
5. **Rule 단독 평가** (분류기 성능과 별도)
   - 1단계 Rule: `is_stagnant_cluster` 대비 평가
   - 최종 Rule: `is_current_parking_candidate` 대비 평가
   - 보수적 민감도 분석: `is_high_confidence_candidate` 대비 평가
   - Precision, Recall, F1, **FPR (오타겟팅률)**, ROC-AUC
6. **Threshold sweep**
   - 임계값 변경에 따른 Precision / Recall / FPR 곡선
   - Precision-Recall curve로 운영 임계값 권고

## 수용 기준

**Precision > 0.95 AND FPR < 5%** (디렉터의 "일반 유저 피해 최소화" 요구 직접 반영)

## 출력

- Feature Importance 차트 (SHAP summary plot 또는 bar chart) — 접속·성장 피처 비교 포함
- 1단계 성장 정체 Rule과 2단계 최종 주차 후보 Rule 정의
- 평가 지표 테이블 (1단계 / 최종 Rule별 Precision / Recall / F1 / FPR / ROC-AUC)
- Threshold sweep 시각화 (PR curve, FPR-Recall trade-off)
- 운영 임계값 권고 (디렉터 타겟팅 시나리오 별)

## 주의

- 분류기 자체 성능 보고는 부차적 (importance 추출의 sanity check 용도)
- Rule의 단순성 ↔ 성능 trade-off를 threshold sweep으로 명시적으로 제시
- **Ground truth 없음**: 모든 지표는 H1 파생 라벨 대비값이다. 최종 Rule의 FPR은 휴면/일반 캐릭터 오분류율로 해석한다.
- `is_stagnant_cluster`는 성장 정체 Rule 학습 타깃이지 주차 후보 확정 라벨이 아니다.
- 접속을 수집·전처리 사전 필터로 사용하지 않는다. 최종 Rule의 별도 게이트로 명시한다.
