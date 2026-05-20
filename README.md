# 메이플스토리 주차 유저 클러스터링

> 응용데이터분석 텀프로젝트 | 소프트웨어융합학과 3학년

Nexon OpenAPI 데이터를 활용하여 메이플스토리 **주차 유저**를 비지도 클러스터링으로 검출하고, 디렉터가 직면한 타겟팅 문제에 대한 Data-Driven 대안을 제시하는 프로젝트입니다.

---

## 배경

> *"성장 수준을 고정하고 시장에 공급하는 경우, 소위 주차 유저들의 메소 생산량만 줄이려고 해봤지만 그분들만 타겟팅해서 줄이는 게 현재 시점에서는 어려웠다."*
> — 메이플스토리 디렉터, 라이브 방송 중

**주차 유저**란 캐릭터의 성장을 의도적으로 멈추고 특정 레벨에 고정된 채 주간 보스 레이드를 반복하여 메소를 대량 생산하는 유저입니다. 일반 유저와 행동 패턴이 외관상 유사해 규칙 기반 타겟팅이 어려운 상황입니다.

---

## 연구 가설

### 가설 1 — 클러스터링으로 주차 유저 집단 분리
> 주차 유저는 일반 유저와 플레이 패턴 feature 상에서 구별되는 군집을 형성할 것이다.

- **방법**: K-Means / DBSCAN (Elbow Method / Silhouette Score로 최적 k 선정)
- **핵심 Feature**: 월평균 레벨 변화량, 월평균 전투력 변화량, 월평균 유니온 변화량, 심볼 성장 수준
- **기대**: 성장 지표 변화가 거의 없는 군집이 별도 형성

### 가설 2 — 특정 레벨 구간·직업군에 불균형 집중
> 주차 유저 군집은 특정 레벨 구간 및 직업군에서 통계적으로 유의미하게 높은 비율로 나타날 것이다.

- **방법**: 카이제곱 검정 (Chi-Square Test), α = 0.05
- **검정 대상**: 레벨 구간별(240~260, 260~280, 280~300+), 직업군별 주차 유저 비율

### 가설 3 — Rule-Based 기준의 낮은 오분류율
> Feature Importance 기반으로 도출한 Rule-Based 타겟팅 기준은 일반 유저 피해를 최소화하면서 주차 유저를 식별할 수 있다.

- **방법**: Random Forest / XGBoost Feature Importance → Rule 도출 → Precision / Recall / FPR / ROC-AUC 평가

---

## 데이터 수집 설계

### 수집 소스
- **Nexon OpenAPI** (`https://openapi.nexon.com/`)
- API Rate Limit: 500 req/s, 20,000,000 req/day
- 수집 가능 기간: 최근 2년 이내

### 사용 엔드포인트

| 엔드포인트 | 수집 항목 |
|---|---|
| `ranking/union` | 유니온 랭킹 (캐릭터 샘플링) |
| `character/basic` | 레벨, 직업, 월드 |
| `character/stat` | 전투력 |
| `user/union` | 유니온 레벨 |
| `character/symbol-equipment` | 아케인/어센틱 심볼 합산 |
| `user/union-raider` | 본캐 판별 (유니온 블록 레벨) |

> ※ 큐브/스타포스 이력(History API)은 계정주 본인만 조회 가능하여 **제외**

### 본캐 판별 로직

`user/union-raider`의 `union_block` 중 `block_level` 최댓값이 현재 캐릭터 레벨 이하인 경우에만 본캐로 인정합니다. 더 높은 레벨의 블록이 존재하면 해당 캐릭터는 부캐로 분류되어 제외됩니다.

### 피처 수집 방식 (월별 스냅샷)

- 수집 기간: **2024-06 ~ 2026-05 (24개월)**
- 각 캐릭터 × 월당 10회 API 호출
  - `character/basic`: 레벨, 경험치 (월 1일 기준)
  - `character/stat`: 전투력 × 7일(1~7일) → max 사용
  - `user/union`: 유니온 레벨 (월 1일 기준)
  - `character/symbol-equipment`: 아케인/어센틱 심볼 합산 (월 1일 기준)
- 유효 데이터가 2개월 이상인 구간에서 **월평균 변화량** 계산

---

## 피처 목록 (`features_monthly.csv`)

| 컬럼 | 설명 |
|---|---|
| `character_name`, `ocid`, `character_class`, `world_name` | 식별 정보 |
| `level`, `union_level` | 최신(마지막 유효월) 값 |
| `arcane_symbol_score`, `authentic_symbol_score` | 심볼 레벨 합산 최신 값 |
| `exp`, `log_exp` | 최신 월 경험치 및 log 변환값 |
| `avg_monthly_delta_level` | 월평균 레벨 증가량 |
| `avg_monthly_delta_combat_power` | 월평균 전투력 증가량 |
| `avg_monthly_delta_union_level` | 월평균 유니온 레벨 증가량 |
| `avg_monthly_delta_arcane_symbol` | 월평균 아케인심볼 합산 증가량 |
| `avg_monthly_delta_authentic_symbol` | 월평균 어센틱심볼 합산 증가량 |
| `first_valid_month`, `last_valid_month`, `num_valid_months` | 유효 월 범위 |

---

## 현재 진행 상황

| 단계 | 상태 |
|---|---|
| 주제 확정 및 가설 설계 | ✅ 완료 |
| Nexon OpenAPI 탐색 및 수집 스크립트 개발 | ✅ 완료 |
| 본캐릭터 샘플링 (`collect_main_characters.py`) | ✅ 완료 — **1,497명** 수집 |
| 월별 피처 수집 (`collect_features.py`) | ✅ 완료 — **1,497행** (24개월 스냅샷 기반) |
| 데이터 전처리 및 EDA | 예정 |
| 클러스터링 (가설 1) | 예정 |
| 카이제곱 검정 (가설 2) | 예정 |
| Feature Importance 및 Rule 평가 (가설 3) | 예정 |
| 보고서 작성 | 예정 |

---

## 앞으로의 계획

### 1단계 — 전처리 및 EDA
- `num_valid_months >= 2` 필터링 후 결측치 처리
- `avg_monthly_delta_*` 컬럼 분포 시각화 (히스토그램, 박스플롯)
- 레벨 구간별 / 직업군별 기초 통계 확인

### 2단계 — 클러스터링 (가설 1)
- 입력 피처: `avg_monthly_delta_level`, `avg_monthly_delta_combat_power`, `avg_monthly_delta_union_level`, `avg_monthly_delta_arcane_symbol`, `avg_monthly_delta_authentic_symbol`
- StandardScaler 정규화 후 K-Means (Elbow Method / Silhouette Score로 k 선정)
- DBSCAN으로 보조 검증
- 클러스터별 delta 분포 시각화 → 주차 유저 군집 식별

### 3단계 — 분포 검정 (가설 2)
- 클러스터 레이블과 레벨 구간 / 직업군 교차 집계
- Chi-Square Test (α = 0.05) 로 유의성 검정
- 주차 유저 비율이 높은 레벨 구간 및 직업군 특정

### 4단계 — Rule 도출 및 평가 (가설 3)
- Random Forest / XGBoost로 Feature Importance 산출
- 핵심 feature 기반 Rule 도출 (예: `Δlevel < N AND Δ전투력 < M`)
- Precision / Recall / F1 / False Positive Rate / ROC-AUC 평가
- Threshold별 Trade-off 시각화

---

## 분석 스택

| 단계 | 도구 |
|---|---|
| 데이터 수집 | Python, requests, python-dotenv |
| 전처리 | pandas, numpy |
| 클러스터링 | scikit-learn (K-Means, DBSCAN) |
| 분포 검정 | scipy.stats |
| 분류 평가 | scikit-learn, xgboost |
| 시각화 | matplotlib, seaborn |

---

## 실행 방법

```bash
# 의존성 설치
pip install requests pandas python-dotenv scikit-learn scipy xgboost matplotlib seaborn

# .env 파일에 API 키 설정
echo "MAPLE_API_KEY=your_api_key_here" > .env

# 본캐릭터 수집 (main_characters.csv 생성)
python collect_main_characters.py

# 월별 피처 수집 (features_monthly.csv 생성, ~13분 소요)
python collect_features.py
```

---

## 파일 구조

```
maple_parking_detect/
├── collect_main_characters.py   # 본캐릭터 샘플링 (층화 랜덤, 유니온 랭킹 기반)
├── collect_features.py          # 24개월 월별 스냅샷 수집 및 피처 계산
├── main_characters.csv          # 수집된 본캐릭터 1,497명
├── features_monthly.csv         # 월별 피처 테이블 1,497행
└── .env                         # API 키 (미포함)
```

---

## 참고 자료

- [Nexon OpenAPI 공식 문서](https://openapi.nexon.com/)
