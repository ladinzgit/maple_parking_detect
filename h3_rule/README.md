# 가설 3 (지도) — Feature Importance 기반 Rule 평가

Feature Importance로 도출한 단순 Rule이 일반 유저 피해를 최소화하면서 주차 후보를 식별할 수 있는지 평가한다. 분류기는 핵심 피처 식별 도구이며, **본 평가 대상은 Rule** 이다.

## 입력

- `data/h1_current_candidates.csv` (`is_current_parking_candidate`, 민감도 분석은 `is_high_confidence_candidate`)
- `data/features_monthly.csv` 전체 피처
- 전제: H1 수용 기준(Silhouette ≥ 0.4, ARI ≥ 0.7) 충족

## 방법

1. **분류기 학습** (importance 추출 용도)
   - Random Forest, XGBoost
   - 5-fold stratified CV (class imbalance 고려)
   - pseudo-label = H1 현재 후보 레이블
2. **Feature Importance 산출**
   - SHAP values (mean |SHAP|) 또는 permutation importance
   - 상위 2~3개 핵심 피처 선정
3. **단순 Rule 도출**
   - 예: `ΔcumEXP = 0 AND Δunion_level = 0 AND access_active_months >= 2`
   - 임계값 a, b는 핵심 피처 분포의 분위수에서 후보 → grid search
4. **Rule 단독 평가** (분류기 성능과 별도)
   - Precision, Recall, F1, **FPR (오타겟팅률)**, ROC-AUC
5. **Threshold sweep**
   - 임계값 a, b 변경에 따른 Precision / Recall / FPR 곡선
   - Precision-Recall curve로 운영 임계값 권고

## 수용 기준

**Precision > 0.95 AND FPR < 5%** (디렉터의 "일반 유저 피해 최소화" 요구 직접 반영)

## 출력

- Feature Importance 차트 (SHAP summary plot 또는 bar chart)
- 최종 Rule 정의 (임계값 포함)
- 평가 지표 테이블 (Precision / Recall / F1 / FPR / ROC-AUC)
- Threshold sweep 시각화 (PR curve, FPR-Recall trade-off)
- 운영 임계값 권고 (디렉터 타겟팅 시나리오 별)

## 주의

- 분류기 자체 성능 보고는 부차적 (importance 추출의 sanity check 용도)
- Rule의 단순성 ↔ 성능 trade-off를 threshold sweep으로 명시적으로 제시
