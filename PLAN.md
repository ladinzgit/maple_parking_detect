# Plan: collect_features.py 재설계 (월별 스냅샷 방식)

## Context

기존 T1(2024-05-16)/T2(2026-05-16) 두 시점 비교 방식은 다음 문제가 있음:
- T1 이전에 캐릭터가 없으면 delta 계산 불가 (338명, 32% 제외됨)
- 전투력을 단일 날짜 기준으로 측정 → 사냥 템셋팅 시 실제보다 낮게 측정될 위험
- 아케인/어센틱 심볼을 합산해 정보 손실
- dojang은 도메인 지식상 유의미하지 않음 (제외)

새 설계: **2024-06 ~ 2026-05 (24개월)** 월별 스냅샷을 수집하고, 데이터가 존재하는 구간에서 **월평균 변화량**을 계산.

클러스터링 입력 데이터 최소 **1,000행** 확보 필요.

---

## 수정 대상 파일

- **`collect_features.py`** — 전면 재작성
- `collect_main_characters.py` — TARGET_COUNT 1000 → **1300** 으로 변경 후 재실행 (1000명 확보 버퍼)
- `main_characters.csv` — 기존 1,086명 + 추가 수집분 (입력)
- `features.csv` → **`features_monthly.csv`** (새 출력)

---

## 설계 상세

### 수집 월 목록

```python
MONTHS = []  # "2024-06" ~ "2026-05", 24개
y, m = 2024, 6
for _ in range(24):
    MONTHS.append(f"{y:04d}-{m:02d}")
    m += 1
    if m > 12: m = 1; y += 1
```

- 2024-06-01 기준: API 2년 제한(기준일 2026-05-19)에서 안전하게 접근 가능
- 2026-05: 오늘(2026-05-19) 기준 1~7일 모두 과거 → 전투력 max 수집 가능

### API 호출 구조 (캐릭터 1명 × 1개월 = 10회)

| 데이터 | 날짜 | 호출수 | 비고 |
|--------|------|--------|------|
| level + exp | `YYYY-MM-01` | 1 | character/basic |
| combat_power | `YYYY-MM-01 ~ 07` | 7 | character/stat × 7일, **max** 사용 |
| union_level | `YYYY-MM-01` | 1 | user/union |
| arcane_symbol_score | `YYYY-MM-01` | 1 (심볼 공유) | character/symbol-equipment, `아케인심볼` 합산 |
| authentic_symbol_score | 위와 동일 | 0 추가 | 같은 응답에서 `어센틱심볼` 합산 |

- `basic` 응답이 None → 해당 월 전체 skip (캐릭터 미존재 혹은 API 범위 초과)
- 총 호출: ~1,300명 × 24개월 × 10회 = **~312,000회** → 400 req/s 기준 **~780초(~13분)**

### 심볼 분리 로직

```python
def get_symbol_scores(symbol_data):
    arcane, authentic = 0, 0
    for s in symbol_data.get("symbol", []):
        name = s.get("symbol_name", "")
        lv = int(s.get("symbol_level") or 0)
        if "아케인심볼" in name:
            arcane += lv
        elif "어센틱심볼" in name:
            authentic += lv
    return arcane, authentic
```

### 경험치 처리

- `character_exp` 필드 (int) → `exp` 컬럼 저장
- 레벨업 시 exp 리셋 → 월간 exp 변화는 의미 없음, **최종 월 exp만 저장**
- `log_exp = log1p(exp)` (numpy)

### 월평균 변화량 계산

```python
def avg_monthly_delta(values_by_month_idx):
    # values_by_month_idx: [(idx, value), ...] — None 제외
    if len(valid) < 2: return None
    first_idx, first_val = valid[0]
    last_idx, last_val = valid[-1]
    months_elapsed = last_idx - first_idx
    return (last_val - first_val) / months_elapsed
```

적용 피처: level, combat_power, union_level, arcane_symbol, authentic_symbol

### 출력 컬럼 (`features_monthly.csv`)

| 컬럼 | 설명 |
|------|------|
| character_name, ocid, character_class, world_name | 식별 정보 |
| level, union_level | 최신(마지막 유효월) 값 |
| arcane_symbol_score, authentic_symbol_score | 최신 값 |
| exp, log_exp | 최신 월 경험치 |
| avg_monthly_delta_level | 월평균 레벨 증가 |
| avg_monthly_delta_combat_power | 월평균 전투력 증가 |
| avg_monthly_delta_union_level | 월평균 유니온 증가 |
| avg_monthly_delta_arcane_symbol | 월평균 아케인심볼 합산 증가 |
| avg_monthly_delta_authentic_symbol | 월평균 어센틱심볼 합산 증가 |
| first_valid_month | 첫 데이터 존재 월 |
| last_valid_month | 마지막 데이터 존재 월 |
| num_valid_months | 유효 월 수 |

---

## 구현 구조

```
collect_features.py
├── RateLimiter(400)            # collect_main_characters.py와 동일 패턴
├── MONTHS = [...]              # 24개월 목록 (2024-06 ~ 2026-05)
├── fetch_month_snapshot(ocid, year_month)
│   ├── basic → level, exp (None이면 전체 None 반환)
│   ├── stat × 7일 → max(combat_power)
│   ├── union → union_level
│   └── symbol-equipment → arcane_score, authentic_score
├── compute_features(monthly_snaps)
│   ├── avg_monthly_delta per field
│   └── last valid값 → current snapshot
├── process_character(row)
│   ├── loop 25개월 → fetch_month_snapshot
│   └── compute_features → dict 반환
├── save_results(rows, filepath) # 기존 CSV append, ocid 중복 제거
└── collect()                    # ThreadPoolExecutor(30) + 진행 출력
```

---

## 주의 사항

- 2024-06 시작: API 2년 제한(기준일 2026-05-19) 기준 안전권. 400 에러는 None 처리로 자동 skip.
- **1,000행 확보 전략**: collect_main_characters.py TARGET_COUNT=1300으로 먼저 추가 수집 → features 수집 후 num_valid_months≥2 필터 적용 시 1,000행 이상 기대.
- 전투력 7일 중 하루라도 값이 있으면 그 max를 사용; 전부 None이면 해당 월 combat_power = None
- 재시작 지원: `features_monthly.csv` 존재 시 완료 ocid 로드 후 skip
- 진행 출력: 100명마다 경과 시간·속도 출력

---

## 검증

1. `collect_main_characters.py` (TARGET_COUNT=1300) 실행 → main_characters.csv ≥1,300명 확인
2. `python collect_features.py` 실행 후 약 13분 대기
3. `features_monthly.csv` 행 수 확인, `num_valid_months >= 2` 필터 후 **≥1,000행** 확인
4. `avg_monthly_delta_combat_power` describe() — 중앙값 양수, 이상치 없는지 확인
5. `num_valid_months` 분포 — 24에 가까운 캐릭터가 다수인지 확인
6. arcane/authentic 심볼 점수 레벨 구간별 확인 (260대: authentic=0 예상)
