# 메이플스토리 성장 정체 기반 주차 후보군 탐지

> 응용데이터분석 텀프로젝트 | 소프트웨어융합학과 3학년

Nexon OpenAPI 데이터를 활용하여 메이플스토리 **주차 후보군**을 비지도 클러스터링으로 탐색하고, 디렉터가 직면한 타겟팅 문제에 대한 Data-Driven 대안을 제시하는 프로젝트입니다.

---

## 배경

> *"성장 수준을 고정하고 시장에 공급하는 경우, 소위 주차 유저들의 메소 생산량만 줄이려고 해봤지만 그분들만 타겟팅해서 줄이는 게 현재 시점에서는 어려웠다."*
> — 메이플스토리 디렉터, 라이브 방송 중

**주차 유저**란 캐릭터의 성장을 의도적으로 멈추고 특정 레벨에 고정된 채 주간 보스 레이드를 반복하여 메소를 대량 생산하는 유저입니다. 일반 유저와 행동 패턴이 외관상 유사해 규칙 기반 타겟팅이 어려운 상황입니다.

> **연구 범위:** 공개 API에서는 주간 보스 수행 기록과 메소 생산량을 직접 조회할 수 없습니다. 따라서 본 연구는 주차 유저를 확정적으로 식별하지 않으며, 장기간 성장 정체와 접속 활동이 함께 관측되는 캐릭터를 **주차 후보**로 정의합니다.

---

## 연구 가설

세 가설은 각각 **비지도 학습 / 통계 / 지도 학습** 방법론을 적용하여 주차 후보군 탐색 문제를 세 각도에서 검증한다.

### 가설 1 (비지도 학습) — 클러스터링으로 주차 후보군 분리
> 성장 정체와 접속 활동이 장기간 함께 관측되는 주차 후보군은 일반 성장 캐릭터와 구별되는 군집을 형성한다.

- **방법**: K-Means + DBSCAN (Elbow Method / Silhouette Score로 최적 k 선정)
- **핵심 Feature**: 누적 경험치·유니온·HEXA 조각 소비 성장량
  - 성장 정체 군집 탐색 후 월내 반복 접속 관측으로 현재 주차 후보를 선별
- **수용 기준**: K-Means **Silhouette ≥ 0.4** AND K-Means/DBSCAN **ARI ≥ 0.7** (단일 알고리즘 artifact가 아님을 확인)
- **출력**: `data/cluster_labels.csv` (성장 정체 군집), `data/h1_current_candidates.csv` (H2/H3 입력)

### 가설 2 (통계) — 특정 레벨 구간·직업군에 불균형 집중
> 주차 후보군은 특정 레벨 구간 및 직업 계열에서 통계적으로 유의하게 높은 비율로 나타난다.

- **방법**: 카이제곱 검정 (Chi-Square Test), α = 0.05
- **검정 대상**:
  - `candidate_label × level_band` (270~279 / 280~285 / 286~290)
  - `candidate_label × class_group` (전사 / 마법사 / 궁수 / 도적 / 해적)
- **추가 분석**: 표준화 잔차(standardized residuals)로 over/under-represented 셀 식별 → 어느 레벨/직업이 주차 후보 핫스팟인지 특정

### 가설 3 (지도 학습) — Rule-Based 기준의 낮은 오분류율
> Feature Importance 기반으로 도출한 Rule-Based 타겟팅 기준은 주차 후보를 높은 정밀도와 낮은 오타겟팅률로 식별한다.

- **방법**:
  1. Random Forest / XGBoost 학습 (H1 클러스터 레이블 = pseudo-label, 5-fold stratified CV)
  2. SHAP / permutation importance → 상위 2~3개 핵심 피처 선정
  3. 임계값 기반 단순 Rule 도출 (예: `ΔcumEXP = 0 AND Δunion = 0 AND access_months >= 2`)
  4. Rule 단독 평가: Precision, Recall, F1, **FPR (오타겟팅률)**, ROC-AUC
  5. Threshold sweep으로 Precision-Recall trade-off 시각화
- **수용 기준**: **Precision > 0.95 AND FPR < 5%**
- **전제**: H1 수용 기준 충족 시 pseudo-label 신뢰도 확보

---

## 데이터 수집 설계

### 수집 소스
- **Nexon OpenAPI** (`https://openapi.nexon.com/`)
- API Rate Limit: 500 req/s, 20,000,000 req/day
- 수집 가능 기간: 최근 2년 이내

### 샘플링 설계 (v2 — 종합 랭킹 기반)

| 항목 | 내용 |
|---|---|
| 엔드포인트 | `ranking/overall` (레벨 내림차순, `class` 파라미터) |
| 대상 레벨 | **260 ~ 285** |
| 목표 인원 | **5계열 × 400명 = 2,000명** (H3 supervised CV/Rule threshold 안정성 확보) |
| 레벨 균등 배분 | 260~269 / 270~279 / 280~285 각 133/133/134명 (계열별) |
| 페이지 탐색 | 직업명별 이진 탐색으로 260~285 페이지 범위 확정 |

> **레벨 상한 285 설정 근거**: 280→285 구간 레벨업에 70~99시간 소요. 285+ 캐릭터는 활성 유저도 `delta_level ≈ 0` → 주차 후보 신호와 구분 불가능.

### 사용 엔드포인트

| 엔드포인트 | 수집 항목 |
|---|---|
| `ranking/overall` | 종합 랭킹 (캐릭터 샘플링, 직업별 필터) |
| `id` | OCID 조회 |
| `character/stat` | 전투력 (월 1~7일 × 7회 → max) |
| `user/union` | 유니온 레벨 |
| `character/symbol-equipment` | 어센틱심볼 합산 |
| `user/union-raider` | 본캐 판별 (유니온 블록 레벨) |

> ※ 큐브/스타포스 이력(History API)은 계정주 본인만 조회 가능하여 **제외**

### 본캐 판별 로직

`user/union-raider`의 `union_block` 중 `block_level` 최댓값이 현재 캐릭터 레벨 이하인 경우에만 본캐로 인정합니다.

### 피처 수집 방식 (월별 스냅샷)

- 수집 기간: **2025-06 ~ 2026-05 (12개월)** — 최근 행동 신호 가중, API 2년 한계 마진 확보
- 각 캐릭터 × 월당 14회 API 호출
  - `character/basic`: 레벨, 경험치, 접속 여부 (월 1·8·15·22일 기준)
  - `character/stat`: 전투력 × 7일(1~7일) → max 사용
  - `user/union`: 유니온 레벨 (월 1일 기준)
  - `character/symbol-equipment`: 어센틱심볼 합산 (월 1일 기준)
- 유효 데이터가 2개월 이상인 구간에서 **월평균 변화량** 계산

---

## 피처 목록 (`features_monthly.csv`)

| 컬럼 | 설명 |
|---|---|
| `character_name`, `ocid`, `character_class`, `world_name` | 식별 정보 |
| `level`, `union_level` | 최신(마지막 유효월) 값 |
| `authentic_symbol_score` | 어센틱심볼 레벨 합산 최신 값 |
| `hexa_level_sum` | HEXA 코어 레벨 합산 최신 값 (260+ 단조 증가 활동 신호) |
| `exp`, `log_exp` | 최신 월 경험치 및 log 변환값 |
| `avg_monthly_delta_level` | 월평균 레벨 증가량 (주차 후보 핵심 신호) |
| `avg_monthly_delta_combat_power` | 월평균 전투력 증가량 (주차 후보 핵심 신호) |
| `avg_monthly_delta_union_level` | 월평균 유니온 레벨 증가량 (주차 후보 핵심 신호) |
| `avg_monthly_delta_authentic_symbol` | 월평균 어센틱심볼 합산 증가량 |
| `avg_monthly_delta_hexa` | 월평균 HEXA 레벨 합산 증가량 |
| `recent3_delta_*`, `recent6_delta_*` | 3개월/6개월 단기 기울기 — 나이 편향 보정용 (렌 코호트 대응) |
| `access_active_months`, `access_active_weeks`, `access_ratio`, `access_recent` | 최근 로그인 활동 (`character/basic` `access_flag` 기반) |
| `character_age_months`, `created_in_window` | 캐릭터 연령; `created_in_window=1` = 새 직업 코호트 (렌 등) |
| `first_valid_month`, `last_valid_month`, `num_valid_months` | 유효 월 범위 |

---

## 현재 진행 상황

| 단계 | 상태 |
|---|---|
| 주제 확정 및 가설 설계 | ✅ 완료 |
| Nexon OpenAPI 탐색 및 수집 스크립트 개발 | ✅ 완료 (v2 재설계) |
| 본캐릭터 샘플링 | ✅ 완료 (260~285, 직업별 균등 2,000명) |
| 월별 피처 수집 | ✅ 완료 (`data/features_monthly.csv`, `data/hexa_fragments.csv`) |
| 데이터 전처리 및 EDA | ✅ 완료 (v2, df_final 1,337명) |
| 클러스터링 (가설 1) | 🔄 v2 데이터 기준 재실행 필요 (v1 결과 보유) |
| 카이제곱 검정 (가설 2) | ⏳ H1 현재 후보 라벨 생성 후 |
| Feature Importance 및 Rule 평가 (가설 3) | ⏳ 예정 |
| 보고서 작성 | ⏳ 예정 |

---

## 실행 방법

```bash
# 의존성 설치
pip install requests pandas python-dotenv scikit-learn scipy xgboost matplotlib seaborn statsmodels numpy

# .env 파일에 API 키 설정
# MAPLE_API_KEY=your_api_key_here

# 본캐릭터 수집 (data/main_characters.csv 생성, 예상 20~40분)
python scripts/collect_main_characters.py

# 월별 피처 수집 (data/features_monthly.csv 생성, 예상 10분)
python scripts/collect_features.py

# EDA 노트북 실행
jupyter notebook eda/eda.ipynb
```

---

## 파일 구조

```
maple_parking_detect/
├── data/                           # 수집 데이터 (gitignored)
│   ├── main_characters.csv         #   본캐릭터 2,000명 (260~285, 직업별 균등)
│   ├── features_monthly.csv        #   월별 피처 테이블
│   ├── cluster_labels.csv          #   H1 성장 정체 군집 레이블
│   └── h1_current_candidates.csv   #   H1 현재 후보 레이블 (H2/H3 입력)
├── scripts/
│   ├── collect_main_characters.py  #   종합 랭킹 기반 샘플링 (v2)
│   └── collect_features.py         #   12개월 월별 스냅샷 수집
├── eda/                            # 탐색적 데이터 분석
│   └── eda.ipynb
├── h1_clustering/                  # 가설 1: K-Means / DBSCAN
│   └── h1_clustering.ipynb
├── h2_distribution/                # 가설 2: Chi-Square 검정
│   └── h2_distribution.ipynb       #   (작성 예정)
├── h3_rule/                        # 가설 3: Feature Importance & Rule
│   └── h3_rule.ipynb               #   (작성 예정)
├── docs/
│   ├── PROJECT.md                  #   프로젝트 상세 기록
│   └── level.txt                   #   레벨별 경험치 참고 데이터
└── .env                            # API 키 (미포함)
```

---

## 분석 스택

| 단계 | 도구 |
|---|---|
| 데이터 수집 | Python, requests, python-dotenv |
| 전처리 | pandas, numpy, scipy (Winsorize) |
| 클러스터링 | scikit-learn (K-Means, DBSCAN) |
| 분포 검정 | scipy.stats |
| 분류 평가 | scikit-learn, xgboost |
| 시각화 | matplotlib, seaborn |

---

## 참고 자료

- [Nexon OpenAPI 공식 문서](https://openapi.nexon.com/)
