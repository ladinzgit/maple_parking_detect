# PLAN.md — 주차 후보군 재설계 실행 지시서 (작성 2026-06-04)

> **다음 세션 agent에게:** 이 문서는 단독 실행용 지시서다. 이전 대화 없이 이 문서 + `CLAUDE.md` + 코드만으로 진행하라.
> 아래 **0. 의도/맥락**을 먼저 읽고 **왜** 바꾸는지 이해한 뒤 **2. 실행 단계**를 순서대로 수행하라.
> (주의: `docs/PLAN.md`는 무관한 historical 문서. 이 파일은 루트의 새 지시서다.)

---

## 0. 의도 / 맥락 (WHY) — 반드시 먼저 읽을 것

### 0.1 주차(parking)의 조작적 정의 — 상대적 정체, 절대 0 아님

- 메이플은 **모든 유저에게 이벤트 재화**(성장의 불꽃, 경험치 쿠폰 등)를 지급한다. 따라서 **접속만 해도 누구나 최소 성장률(baseline floor)** 을 가진다. "성장 0"인 유저는 사실상 없다.
- 주차 유저는 **레벨/성장을 동결**한 채 **보스 주간 재화(솔 에르다 조각 등)를 벌지만, 그 재화를 자기 성장에 재투자하지 않는** 유저다. 즉 **"이벤트 재화로 인한 baseline 성장만 있고, 보스 재화는 벌기만 하고 안 씀"**.
- 따라서 탐지 대상은 **절대적으로 멈춘 유저가 아니라**, 모집단 성장률 분포에서 **상대적으로 하위(floor 근처)에 있으면서 여전히 활발히 접속하는** 유저다.
- **결론: 클러스터링이 잡아내는 "성장 하위 군집" 자체 = 주차 후보군.** 별도의 절대-임계 게이트(예: Δexp==0)로 거르는 방식은 폐기한다.

### 0.2 이전 설계의 문제 (왜 재작업하나)

1. **클러스터링에 `access_ratio`가 피처로 들어가 있었다** (`optimal_feature_set.json`, `h1_clustering.ipynb`). 이건 군집을 **낮은 접속(준-휴면)** 쪽으로 끌어당겨, "active 보스파밍 주차"와 "접속 뜸한 준-휴면"을 한 군집에 섞었다.
   - 검증: access **독립** 성장 프록시(cumexp·union·hfrag 모두 하위30%)의 접속 median = **11개월**. 반면 access_ratio 오염 군집(412명) median = **9개월**. → 진짜 주차는 접속을 **많이** 한다. access_ratio가 군집을 오염시키고 있었다.
2. **절대-0 게이트(`is_current_parking_candidate`, 63명 / `is_high_confidence_candidate`, 35명)** 는 `Δcumexp<=0 AND Δunion<=0 AND Δhfrag<=0 AND access>=2`로 정의됐다. 이는 "접속하는데 경험치/조각 증가가 0" = **자기모순**(보스·일퀘·이벤트로 미량 exp/조각은 항상 들어옴). 진짜 보스파밍 주차를 배제한다. → **폐기/강등.**
3. `CLAUDE.md`가 기술한 채택셋(cumexp·union·**hfrag**, k=4, sil 0.6430, 394명)과 **실제 아티팩트**(cumexp·union·**access_ratio**, k=3, sil 0.5446, 412명)가 불일치. 문서가 stale.

### 0.3 해결 설계

- **접속(access)을 통제변인으로 고정**: 표본을 **12개월 중 10개월 이상 접속(≥10/12)** 유저로 제한. → 표본 전체가 "활발히 접속" 상태로 균일 → 접속이 더 이상 교란/군집축이 될 수 없음.
- **클러스터링 피처에서 access family 전면 제외**. 성장/재투자 축(cumexp, union, hfrag-소비)으로만 군집.
- 그 결과 **고활성 유저 중 성장 하위 군집 = active 주차 후보군**. 0.1 정의와 정합.

### 0.4 사용자가 확정한 결정

| 항목 | 결정 |
|---|---|
| 접속 통제 임계 | **≥10/12 개월** (사용자 선택. 참고: clean 프록시 기준 ≥10은 주차의 ~35% 손실 — 정밀도 우선 trade-off로 수용) |
| 표본 크기 | **정확히 2000명** (in-range 270-290) — 채워질 때까지 재수집 |
| 주차 후보군 | **클러스터링 결과(`is_stagnant_cluster`) 자체** |
| access_ratio | 클러스터링 피처에서 제외 (단 수집·검증용으로 CSV에는 계산 유지) |
| 63/35 (절대-0 게이트) | 폐기 또는 민감도 각주로 강등 |

---

## 1. 참고 — 현재(재작업 전) 검증 수치

> 재실행 후 새 값으로 대체될 기준선. 재현 확인용.

- `main_characters.csv` = 2000행, `features_monthly.csv` = 2000행. 단 **레벨 270-290 in-range = 1980** (20명이 level<270 = collect_main 페이지경계 누수). cluster_labels = 1979.
- 접속 분포(현 2000명): ≥10/12 = **1754명(88%)**, ≥9 = 1840, ≥8 = 1903. =12 = 1428(71%).
- `hexa_fragments.csv`의 `avg_monthly_delta_hexa_frag` = **누적 솔 에르다 조각 소비(투자)량의 월평균 증가율**. 월별값은 단조증가 후 plateau. **낮을수록 = 보스 재화 벌어도 코어에 안 박음 = 주차 신호.** (`collect_hexa_fragments.py` 헤더 참조)
- 현 군집(412, access_ratio 오염) 프로파일: 성장 3축 모두 전체분포 P18~P35(상대 하위), cumexp의 98.1%·union 93%가 >0 → **상대 정체이지 절대 0 아님**(이벤트 baseline). hfrag이 최강 판별축(정체 median 22.5 vs 나머지 95.5), cumexp는 270-290서 약한 축.
- H2 사전(이전 데이터): 레벨구간×파킹 단조감소(저레벨 집중). 직업×파킹 미유의.

---

## 2. 실행 단계 (순서 엄수)

### Step 1 — `scripts/collect_main_characters.py` 수정 + 재수집

**목표:** level 270-290, 5계열×400=2000명, **전원 ≥10/12 접속**, 채워질 때까지 수집.

수정 사항:
1. **접속 필터 강화 (checkpoint → ≥10/12):**
   - `CHECKPOINT_MONTHS`를 관측 12개월 전체(`2025-06`~`2026-05`)로, `MIN_CHECKPOINT_MONTHS = 10`으로 변경. → `passes_checkpoint_access`가 12개월 중 10개월 이상 접속을 요구하게 됨.
   - **API 비용 주의:** 후보당 접속 probe가 현재 3개월×4일=12콜 → 12개월×4일=48콜로 증가. `month_has_access` 호출 루프에 **조기 종료** 추가: 누적 active_months가 10에 도달하면 즉시 통과, 남은 달을 다 더해도 10 불가능하면 즉시 탈락. (불필요한 월 probe 절감)
2. **레벨 누수 수정:** 최종 표본에 level<270이 섞이지 않도록. (현재 20명 누수 — 원인: 페이지경계/랭킹 레벨 vs 스냅샷 레벨 불일치 추정. `process_candidate`의 `char_level` 검사가 collection 시점 기준이므로, 수집 후 in-range 재확인 로직 또는 누수 원인 조사 후 차단.)
3. **2000명 보장:** 강화 필터로 후보 탈락률↑ → `collect_group`이 페이지를 더 깊이 소비해야 함. 계열별 400명이 안 채워지면 페이지 풀 소진 가능 → 경고 출력하고, 부족분 재시도/추가 페이지 확보. (필요시 `TARGET_PER_GROUP` 유지하되 페이지 범위 확장.) 5계열 균등 + 3 level_bin 균등(H2용) 유지.
4. 기존 `main_characters.csv` 등 데이터 파일은 재수집으로 덮어씀 (gitignored).

> **대안(검토용):** collect_main에서 12개월 probe가 너무 비싸면, collect_main은 초과수집(예 ~2400)만 하고 ≥10 필터를 collect_features 이후 다운샘플 단계에서 적용하는 방법도 있음. 단 사용자는 "collect_main에서 엄격 필터"를 명시 → 위 1번(checkpoint ≥10/12 + 조기종료)을 우선 구현.

### Step 2 — `scripts/collect_features.py` + `scripts/collect_hexa_fragments.py` 재수집

1. `collect_features.py` 실행 → `features_monthly.csv`, `monthly_snapshots_raw.csv` 재생성.
   - **`access_ratio`/`access_active_months`/`access_recent` 계산 코드는 그대로 유지** (수집 0 추가비용, ≥10 통제 검증 + 사후검증에 필요). **지우지 말 것.** 제거는 클러스터링 피처풀에서만(Step 4).
2. `collect_hexa_fragments.py` 실행 → `hexa_fragments.csv` 재생성 (`avg_monthly_delta_hexa_frag` 등).
3. 완료 후 확인: 표본 전원 `access_active_months >= 10` 인지, level 270-290 in-range == 2000 인지 검증 출력.

### Step 3 — EDA (`eda/eda.ipynb`, read-only)

- **프로젝트 규칙: eda.ipynb에는 H1/H2/H3 실험 코드 금지.** 분포·상관·전처리 결정만.
- **사용할 모든 피처의 가공 방법 선택:** clip(≥0), log1p, winsorize 등 각 delta 피처별 전처리 확정.
  - 최소: `log1p_avg_monthly_delta_cumexp`, `avg_monthly_delta_union_level`(clip≥0), `avg_monthly_delta_hexa_frag`(clip≥0). 그 외 후보(combat_power, authentic_symbol, recent3/6 변형 등)도 분포 점검.
- 새 표본에서 핵심 수치 재확인: 레벨대 분포, 성장률 분포, 접속이 ≥10로 통제돼 near-constant인지, 주차 프록시 비율.

### Step 4 — `h1_clustering/feature_selection.ipynb` 재실행

- **access family(`access_ratio`, `access_active_months`, `access_recent`)를 클러스터링 후보 피처풀에서 전면 제외.**
- 기존 게이트 유지: near-constant(dom_frac>0.70) 제외, 비퇴화(maxfrac≤0.9), family당 1개, pairwise |corr|<0.85.
- 완전탐색+greedy로 최적셋 재선정 → `optimal_feature_set.json` 갱신.
- **예상 수렴: `[cumexp, union, hfrag]`** (0.1 정의에 정합한 성장/재투자 축). 단 데이터가 다른 셋을 가리키면 그 근거를 기록.

### Step 5 — `h1_clustering/h1_clustering.ipynb` 재실행 (클러스터링)

- 선정셋으로 K-Means(+DBSCAN) → `cluster_labels.csv` 재생성. `is_stagnant_cluster` = **주차 후보군(헤드라인)**.
- **사후 검증(post-hoc):**
  - 군집이 상대 floor(성장 하위 백분위)에 위치하는가.
  - 군집이 **고활성**인가(접속 통제했으니 당연히 높아야 — 확인). access를 군집 정의에 안 썼는데도 active한지 = 설계 성공 증거.
  - hfrag(보스재화 비소비)이 주 판별축인가.
  - 레벨대 단조감소(주차 분포)인가.
  - DBSCAN 보강.
- **`temporal_external_validation.ipynb` 재배치:** 절대-0 게이트(63/35)는 **헤드라인 후보 정의에서 폐기**. 이 노트북은 "시간분할로 군집이 **안정적 trait인지** 검증"(과거 분기 군집이 미래 분기 성장정체를 예측?)으로 **재프레이밍**. 63/35는 쓰더라도 "보수적 고안정 부분집합 / 민감도"로만.

### Step 6 — `CLAUDE.md` 문서 반영

갱신할 내용:
- **주차 조작적 정의**(0.1: 상대적 정체, 이벤트 baseline, 보스재화 비재투자) 추가.
- **주차 후보군 = `is_stagnant_cluster`(클러스터)** 명시. 헤드라인 숫자 = 새 군집 크기.
- **접속 = 통제변인(≥10/12)**, 클러스터링 피처 아님. access_ratio-as-feature 설명 삭제.
- 채택 피처셋 / k / silhouette / 군집 n을 **새 실행값**으로 교체.
- stale 수치 정정: "394명", "휴면 5.8%"(틀림 — 통제 후 전원 고활성), access_ratio 셋 기술 등.
- 63/35 → 민감도/안정성 각주로 강등. temporal validation 역할 재기술.
- 출력파일 표(`cluster_labels.csv` = 주차 후보군, `h1_current_candidates.csv` = 안정성 부분집합) 갱신.

### Step 7 — H2 입력 갱신

- H2(Chi-Square)는 **새 클러스터 라벨(주차 후보군) × level_band / class_group**으로 수행 (이전엔 `h1_current_candidates.csv`의 63명 → **클러스터 라벨로 교체**). 셀 기대빈도↑로 검정 안정.
- `h2_distribution/` 노트북 생성(미착수). α=0.05.

---

## 3. 하지 말 것 (anti-goals)

- 클러스터링 피처에 access 계열 넣지 말 것.
- `Δexp==0`/`Δcumexp<=0` 같은 **절대-0 동결 게이트로 주차 후보를 정의하지 말 것** (보스·이벤트 미량 성장 때문에 자기모순).
- `collect_features.py`에서 access 계산 **삭제하지 말 것** (통제 검증·사후검증에 필요; 제외는 피처풀에서만).
- 클러스터(주차 후보군)를 "성장 정체 = 곧 주차 확정"으로 과대 해석하지 말 것. 공개 API엔 메소·보스기록 없음 → **"후보군"**(확정 아님)이 정확한 용어. 보스재화 "벌이" 측면은 직접 관측 불가 → 레벨+접속+저소비로 **추론**.

## 4. 제약 / 주의

- Nexon API: 500 req/s·20M/day. RateLimiter 400 req/s·30스레드 유지. `date`는 어제 이전. `END_YEAR_MONTH=2026-05` 유지(최신 완전월).
- 데이터는 매일 갱신 → 재수집 시 일관성 위해 한 번에 완주 권장.
- 실험 코드는 가설 폴더에. `eda.ipynb` 수정 금지(분포/전처리만).
- 한국어 폰트 `assets/NanumSquareNeo-bRg.ttf`.
