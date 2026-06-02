"""
collect_features.py — 월별 스냅샷 기반 피처 수집 (v2.2)

main_characters.csv 의 각 캐릭터에 대해 SNAPSHOT_MONTHS 개월치 월별 스냅샷을 수집하고
월평균 변화량(avg_monthly_delta)을 계산하여 features_monthly.csv 에 저장한다.

v2.1 추가 (EDA 후속, 라이브 검증 완료):
  - access_flag (최근 7일 접속): 월별 → access_active_months / access_ratio / access_recent
      → 방치 캐릭 vs 주차 후보(활동 중) 구분 신호. character/basic 에 이미 포함 (추가 호출 0)
  - character_date_create: 캐릭터 생성일 (character/basic 에 포함, 추가 호출 0)
  - character/hexamatrix → hexa_level_sum + avg_monthly_delta_hexa
      → 270+ 클린 단조 활동 신호 (HEXA 코어 레벨 합)
v2.2 추가 (age 교란 제거 — 신규 클래스 렌 포함 유지하며 편향 완화):
  - recent{3,6}_delta_*: 최근 3·6개월 기울기. 12개월 delta는 6월 생성 신규캐(렌 등)를
      항상 '활성'으로 강제 분류 → 캐릭터 연령(age) 교란. 최근-구간 기울기는 연령과 무관하게
      '현재 주차 후보'을 정의 → H1 클러스터링 / H2 class_group 분포 편향 동시 제거. (추가 호출 0)
  - character_age_months / created_in_window: 편향 진단·민감도 분석용. cutoff(2025-06-30)는
      유지(렌=현 직업분포 1위, 대표성). created_in_window 제외 재실행 = 강건성 체크.

스냅샷 기간 선택 근거 (12개월):
  - 주차 후보은 "현재 진행 중인 행동" 으로 탐지 대상 — 최근 1년 신호가 더 적합
  - 24개월: 1년 전 주차 후보 후 복귀 유저까지 주차 후보으로 분류될 위험 (디렉터 타겟팅 정책 위배)
  - 6개월: 활성-슬로우 유저(286~290 자연스러운 성장 둔화)와 주차 후보 분리 어려움
  - 12개월: 1년 무성장 = 명백한 주차 후보 신호 + 일시 정지(1~2개월) 노이즈 흡수
  - Nexon API 2년 한계에서 1년 마진 확보 → 경계 케이스 회피

API 호출: ~2,000명 × SNAPSHOT_MONTHS × 14회 → 400 req/s 기준 소요
  - 12개월: ~264,000회 → ~660초 (~11분)
"""

import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from dotenv import load_dotenv
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
load_dotenv()

API_KEY = os.getenv("MAPLE_API_KEY")
BASE_URL = "https://open.api.nexon.com/maplestory/v1"

_session = requests.Session()
_session.mount("https://", HTTPAdapter(pool_connections=60, pool_maxsize=60))
_session.headers.update({"x-nxopen-api-key": API_KEY})

# ── 설정 ─────────────────────────────────────────────────────────────────────
_DATA_DIR   = Path(__file__).resolve().parent.parent / "data"
INPUT_FILE  = str(_DATA_DIR / "main_characters.csv")
OUTPUT_FILE = str(_DATA_DIR / "features_monthly.csv")
RAW_FILE    = str(_DATA_DIR / "monthly_snapshots_raw.csv")
EXP_REQUIREMENT_FILE = str(_DATA_DIR / "exp_requirement_table.csv")
CONCURRENCY = 30
MAX_RPS     = 400

# 스냅샷 기간: 변경 시 SNAPSHOT_MONTHS 만 조정 (Nexon API 최대 24)
SNAPSHOT_MONTHS = 12
END_YEAR_MONTH  = "2026-05"   # 마지막 수집 월 (최신 가용 데이터)

# MONTHS 자동 생성: END_YEAR_MONTH 에서 SNAPSHOT_MONTHS 만큼 거슬러 올라간 후 오름차순 정렬
_end_y, _end_m = map(int, END_YEAR_MONTH.split("-"))
MONTHS = []
_y, _m = _end_y, _end_m
for _ in range(SNAPSHOT_MONTHS):
    MONTHS.append(f"{_y:04d}-{_m:02d}")
    _m -= 1
    if _m < 1:
        _m = 12
        _y -= 1
MONTHS.reverse()   # 오래된 → 최신

# 최근-구간 기울기 창: age(캐릭터 연령) 교란 제거용 '현재 주차 후보' 신호 (추가 API 호출 0)
RECENT_WINDOWS = [3, 6]   # 최근 3·6개월 두 창 모두 산출 → EDA에서 Feature Set 선택

# delta 필드명 → 컬럼 접미사 (12mo avg_monthly_delta_* 명명과 일치)
_DELTA_SUFFIX = {
    "level":                  "level",
    "combat_power":           "combat_power",
    "union_level":            "union_level",
    "authentic_symbol_score": "authentic_symbol",
    "hexa_level_sum":         "hexa",
    "cumexp":                 "cumexp",
}
_RECENT_DELTA_KEYS = [
    f"recent{w}_delta_{suf}" for w in RECENT_WINDOWS for suf in _DELTA_SUFFIX.values()
]
_RECENT_COUNT_KEYS = [f"num_recent{w}_valid_months" for w in RECENT_WINDOWS]
# ─────────────────────────────────────────────────────────────────────────────


class RateLimiter:
    def __init__(self, calls_per_second):
        self._interval = 1.0 / calls_per_second
        self._lock = threading.Lock()
        self._last = 0.0

    def acquire(self):
        with self._lock:
            now = time.monotonic()
            wait = self._last + self._interval - now
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()


_limiter = RateLimiter(MAX_RPS)


def api_get(endpoint, params):
    _limiter.acquire()
    try:
        r = _session.get(f"{BASE_URL}/{endpoint}", params=params, timeout=10)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None  # 400(날짜 범위 초과), 404, 네트워크 오류 모두 None


def get_authentic_symbol_score(symbol_data):
    """어센틱심볼 레벨 합산을 반환."""
    authentic = 0
    if not symbol_data:
        return authentic
    for s in symbol_data.get("symbol", []):
        name = s.get("symbol_name", "")
        lv   = int(s.get("symbol_level") or 0)
        if "어센틱심볼" in name:
            authentic += lv
    return authentic


def get_hexa_level_sum(hexa_data):
    """HEXA 코어 레벨 합 반환. 데이터 없으면 None (코어 0개면 0)."""
    if not hexa_data:
        return None
    cores = hexa_data.get("character_hexa_core_equipment") or []
    return sum(int(c.get("hexa_core_level") or 0) for c in cores)


def fetch_month_snapshot(ocid, year_month):
    """
    캐릭터 1명의 특정 월 스냅샷 수집 (14회 API 호출: basic4 + stat7 + union1 + symbol1 + hexa1).
    basic이 None이면 전체 None 반환 (캐릭터 미존재 or API 범위 초과).
    반환: dict(level, exp, exp_rate, combat_power, union_level,
               authentic_symbol_score)
    exp_rate: 현재 레벨 내 경험치 비율 (float, %) — character/basic 에 포함, 추가 호출 0
    """
    date_01 = f"{year_month}-01"

    basic = api_get("character/basic", {"ocid": ocid, "date": date_01})
    if not basic:
        return None

    level = int(basic.get("character_level") or 0) or None
    exp_raw = basic.get("character_exp")
    exp = int(exp_raw) if exp_raw is not None else None

    exp_rate_raw = basic.get("character_exp_rate")
    try:
        exp_rate = float(exp_rate_raw) if exp_rate_raw is not None else None
    except (ValueError, TypeError):
        exp_rate = None

    date_create = basic.get("character_date_create")          # ISO, 캐릭터 메타(월 무관 상수)
    access_flags = []
    for day in [1, 8, 15, 22]:
        weekly_basic = basic if day == 1 else api_get(
            "character/basic", {"ocid": ocid, "date": f"{year_month}-{day:02d}"}
        )
        if weekly_basic:
            access_raw = weekly_basic.get("access_flag")
            access = 1 if access_raw == "true" else (0 if access_raw == "false" else None)
            if access is not None:
                access_flags.append(access)
    access_active_weeks = sum(access_flags) if access_flags else None
    access_observed_weeks = len(access_flags)
    access = int(access_active_weeks > 0) if access_active_weeks is not None else None

    # stat × 7일 (MM-01 ~ MM-07) → max(combat_power)
    cp_values = []
    year, month = map(int, year_month.split("-"))
    for day in range(1, 8):
        d = f"{year:04d}-{month:02d}-{day:02d}"
        stat = api_get("character/stat", {"ocid": ocid, "date": d})
        if stat:
            for s in stat.get("final_stat", []):
                if s.get("stat_name") == "전투력":
                    try:
                        cp_values.append(int(str(s.get("stat_value", "0")).replace(",", "")))
                    except (ValueError, TypeError):
                        pass
    combat_power = max(cp_values) if cp_values else None

    union = api_get("user/union", {"ocid": ocid, "date": date_01})
    union_level = union.get("union_level") if union else None

    symbol_data = api_get("character/symbol-equipment", {"ocid": ocid, "date": date_01})
    authentic = get_authentic_symbol_score(symbol_data)

    hexa_data = api_get("character/hexamatrix", {"ocid": ocid, "date": date_01})
    hexa_level_sum = get_hexa_level_sum(hexa_data)

    return {
        "level":                  level,
        "exp":                    exp,
        "exp_rate":               exp_rate,
        "combat_power":           combat_power,
        "union_level":            union_level,
        "authentic_symbol_score": authentic,
        "hexa_level_sum":         hexa_level_sum,
        "access_flag":            access,
        "access_active_weeks":    access_active_weeks,
        "access_observed_weeks":  access_observed_weeks,
        "date_create":            date_create,
    }


def avg_monthly_delta(indexed_values):
    """
    [(month_idx, value), ...] 에서 월평균 변화량 계산.
    None 값 제외 후 유효값 2개 미만이면 None 반환.
    """
    valid = [(i, v) for i, v in indexed_values if v is not None and not pd.isna(v)]
    if len(valid) < 2:
        return None
    first_idx, first_val = valid[0]
    last_idx,  last_val  = valid[-1]
    months_elapsed = last_idx - first_idx
    if months_elapsed == 0:
        return None
    return (last_val - first_val) / months_elapsed


def load_cumulative_exp_base():
    req = pd.read_csv(EXP_REQUIREMENT_FILE, encoding="utf-8-sig")
    req_map = req.set_index("level")["requirement"].to_dict()
    base = {260: 0.0}
    for level in range(261, 292):
        base[level] = base[level - 1] + float(req_map[level - 1])
    return base


_CUMEXP_BASE = load_cumulative_exp_base()


def cumulative_exp(level, exp):
    if level is None or exp is None or pd.isna(level) or pd.isna(exp):
        return None
    return _CUMEXP_BASE.get(int(level), None) + float(exp) if int(level) in _CUMEXP_BASE else None


def log1p_nonnegative(value):
    return float(np.log1p(max(0, value))) if value is not None and not pd.isna(value) else None


def character_age_months(date_create):
    """생성일 ISO 문자열 → END_YEAR_MONTH 기준 캐릭터 연령(개월). 파싱 실패 시 None."""
    if not date_create or len(date_create) < 7:
        return None
    try:
        cy, cm = int(date_create[:4]), int(date_create[5:7])
    except ValueError:
        return None
    return (_end_y - cy) * 12 + (_end_m - cm)


def compute_features(monthly_snaps):
    """
    monthly_snaps: [(month_idx, snap_dict | None), ...] SNAPSHOT_MONTHS 개
    유효 월에서 현재값(마지막)과 월평균 변화량을 계산.
    유효 월이 전혀 없으면 None 반환.
    """
    valid = [(i, s) for i, s in monthly_snaps if s is not None]
    if not valid:
        return None

    first_valid_idx = valid[0][0]
    last_valid_idx  = valid[-1][0]
    last_snap       = valid[-1][1]

    exp_val = last_snap.get("exp")
    log_exp = float(np.log1p(exp_val)) if exp_val is not None else None
    for _, snap in valid:
        snap["cumexp"] = cumulative_exp(snap.get("level"), snap.get("exp"))

    fields = [
        "level", "combat_power", "union_level",
        "authentic_symbol_score", "hexa_level_sum",
    ]
    indexed = {f: [(i, s.get(f)) for i, s in valid] for f in fields}

    # 접속 활동: 유효 월 중 access_flag 가 관측된 월 기준 집계
    access_vals = [s.get("access_flag") for _, s in valid if s.get("access_flag") is not None]
    access_active = sum(access_vals) if access_vals else None
    access_active_weeks = sum(
        s.get("access_active_weeks") for _, s in valid
        if s.get("access_active_weeks") is not None
    )
    access_observed_weeks = sum(s.get("access_observed_weeks", 0) for _, s in valid)
    access_ratio = (
        round(access_active_weeks / access_observed_weeks, 4)
        if access_observed_weeks else None
    )

    # ── 최근-구간 기울기 (age 불변 '현재 주차 후보' 신호; 추가 API 호출 0) ──────────
    # 최근 w개월 = 월 인덱스 ≥ (전체 개월 수 - w). 유효값 <2 면 avg_monthly_delta 가 None 반환.
    n_months = len(MONTHS)
    recent = {
        f"recent{w}_delta_{_DELTA_SUFFIX[f]}": avg_monthly_delta(
            [(i, v) for i, v in indexed[f] if i >= n_months - w]
        )
        for w in RECENT_WINDOWS for f in fields
    }
    recent_counts = {
        f"num_recent{w}_valid_months": sum(1 for i, _ in valid if i >= n_months - w)
        for w in RECENT_WINDOWS
    }
    avg_cumexp = avg_monthly_delta(indexed["cumexp"])
    recent_log_cumexp = {
        f"log1p_recent{w}_delta_cumexp": log1p_nonnegative(recent[f"recent{w}_delta_cumexp"])
        for w in RECENT_WINDOWS
    }

    # ── 캐릭터 연령 / 관측창 내 생성 플래그 (편향 진단·민감도 분석용) ──────────
    # created_in_window: 생성일이 관측창 시작(MONTHS[0]-01) 이후 → 관측창 truncate 된 신규 코호트
    #   (cutoff 가 2025-06-30 이라 사실상 2025-06 생성 = 렌 등 신규 클래스 코호트와 일치).
    date_create = last_snap.get("date_create")
    created_in_window = (
        1 if (date_create and date_create[:10] >= f"{MONTHS[0]}-01")
        else (0 if date_create else None)
    )

    return {
        "level":                              last_snap.get("level"),
        "union_level":                        last_snap.get("union_level"),
        "authentic_symbol_score":             last_snap.get("authentic_symbol_score"),
        "hexa_level_sum":                     last_snap.get("hexa_level_sum"),
        "exp":                                exp_val,
        "log_exp":                            log_exp,
        "avg_monthly_delta_level":            avg_monthly_delta(indexed["level"]),
        "avg_monthly_delta_combat_power":     avg_monthly_delta(indexed["combat_power"]),
        "avg_monthly_delta_union_level":      avg_monthly_delta(indexed["union_level"]),
        "avg_monthly_delta_authentic_symbol": avg_monthly_delta(indexed["authentic_symbol_score"]),
        "avg_monthly_delta_hexa":             avg_monthly_delta(indexed["hexa_level_sum"]),
        "avg_monthly_delta_cumexp":           avg_cumexp,
        "log1p_avg_monthly_delta_cumexp":     log1p_nonnegative(avg_cumexp),
        **recent,
        **recent_log_cumexp,
        "access_active_months":               access_active,
        "access_active_weeks":                access_active_weeks,
        "access_observed_weeks":              access_observed_weeks,
        "access_ratio":                       access_ratio,
        "access_recent":                      last_snap.get("access_flag"),
        "character_date_create":              date_create,
        "character_age_months":               character_age_months(date_create),
        "created_in_window":                  created_in_window,
        **recent_counts,
        "first_valid_month":                  MONTHS[first_valid_idx],
        "last_valid_month":                   MONTHS[last_valid_idx],
        "num_valid_months":                   len(valid),
    }


_FEATURE_KEYS = [
    "level", "union_level", "authentic_symbol_score",
    "hexa_level_sum", "exp", "log_exp",
    "avg_monthly_delta_level", "avg_monthly_delta_combat_power",
    "avg_monthly_delta_union_level",
    "avg_monthly_delta_authentic_symbol", "avg_monthly_delta_hexa",
    "avg_monthly_delta_cumexp", "log1p_avg_monthly_delta_cumexp",
    *_RECENT_DELTA_KEYS,
    *[f"log1p_recent{w}_delta_cumexp" for w in RECENT_WINDOWS],
    "access_active_months", "access_active_weeks", "access_observed_weeks",
    "access_ratio", "access_recent",
    "character_date_create", "character_age_months", "created_in_window",
    *_RECENT_COUNT_KEYS,
    "first_valid_month", "last_valid_month", "num_valid_months",
]


def process_character(row):
    """캐릭터 1명: SNAPSHOT_MONTHS 개월 스냅샷 수집 → 피처 계산 → dict 반환.
    반환: {"features": {...}, "raw_rows": [{ocid, year_month, 월별 원시 스냅샷}, ...]}
    """
    ocid = str(row["ocid"])

    monthly_snaps = []
    raw_rows = []
    for i, ym in enumerate(MONTHS):
        snap = fetch_month_snapshot(ocid, ym)
        monthly_snaps.append((i, snap))
        # 시간 분할 외부검증을 위해 0 경험치·미접속 월도 보존한다.
        # 과거 버전은 exp_rate > 0인 월만 저장해 정체 구간 재구성이 불가능했다.
        if snap is not None:
            raw_rows.append({
                "ocid":                       ocid,
                "year_month":                 ym,
                "level":                      snap.get("level"),
                "exp":                        snap.get("exp"),
                "exp_rate":                   snap.get("exp_rate"),
                "combat_power":               snap.get("combat_power"),
                "union_level":                snap.get("union_level"),
                "authentic_symbol_score":     snap.get("authentic_symbol_score"),
                "hexa_level_sum":             snap.get("hexa_level_sum"),
                "access_flag":                snap.get("access_flag"),
                "access_active_weeks":        snap.get("access_active_weeks"),
                "access_observed_weeks":      snap.get("access_observed_weeks"),
            })

    features = compute_features(monthly_snaps)
    if features is None:
        features = {k: None for k in _FEATURE_KEYS}

    feat_dict = {
        "character_name":  row["character_name"],
        "ocid":            ocid,
        "character_class": row.get("character_class", ""),
        "world_name":      row.get("world_name", ""),
        **features,
    }
    return {"features": feat_dict, "raw_rows": raw_rows}


def load_done_ocids(filepath):
    if not os.path.exists(filepath):
        return set()
    df = pd.read_csv(filepath, encoding="utf-8-sig")
    return set(df["ocid"].dropna().astype(str))


def save_results(rows, filepath):
    if not rows:
        return
    new_df = pd.DataFrame(rows)
    if os.path.exists(filepath):
        existing = pd.read_csv(filepath, encoding="utf-8-sig")
        existing = existing.reindex(columns=new_df.columns)
        new_df = pd.concat([existing, new_df], ignore_index=True)
        new_df = new_df.drop_duplicates(subset="ocid", keep="last")
    new_df.to_csv(filepath, index=False, encoding="utf-8-sig")
    print(f"  [저장] {filepath} — 누적 {len(new_df)}행")


def save_raw_results(rows, filepath):
    if not rows:
        return
    new_df = pd.DataFrame(rows)
    if os.path.exists(filepath):
        existing = pd.read_csv(filepath, encoding="utf-8-sig")
        existing = existing.reindex(columns=new_df.columns)
        new_df = pd.concat([existing, new_df], ignore_index=True)
        # --refresh-raw 재수집 시 과거 최소 스키마 행을 확장 스키마 행으로 교체한다.
        new_df = new_df.drop_duplicates(subset=["ocid", "year_month"], keep="last")
    new_df.to_csv(filepath, index=False, encoding="utf-8-sig")
    print(f"  [저장] {filepath} — 누적 {len(new_df)}행")


def collect(refresh_raw=False):
    chars_df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")
    done_ocids = load_done_ocids(OUTPUT_FILE)
    targets = chars_df if refresh_raw else chars_df[~chars_df["ocid"].astype(str).isin(done_ocids)]
    total = len(targets)

    calls_est = total * SNAPSHOT_MONTHS * 14
    print(f"=== 월별 피처 수집 시작 ===")
    print(f"수집 월: {MONTHS[0]} ~ {MONTHS[-1]} ({len(MONTHS)}개월)")
    mode = "월별 원시 이력 갱신" if refresh_raw else f"기수집 {len(done_ocids)}명 제외"
    print(f"대상: {total}명 ({mode})")
    print(f"예상 API 호출: ~{calls_est:,}회 | 예상 소요: ~{calls_est / MAX_RPS:.0f}초\n")

    run_start = time.monotonic()
    results = []
    raw_results = []
    done = 0

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        futures = {
            executor.submit(process_character, row): row["character_name"]
            for _, row in targets.iterrows()
        }
        for future in as_completed(futures):
            result = future.result()
            done += 1
            if result is not None:
                results.append(result["features"])
                raw_results.extend(result["raw_rows"])

            if done % 100 == 0 or done == total:
                elapsed = time.monotonic() - run_start
                rate = done / elapsed if elapsed > 0 else 0
                valid_cnt = sum(1 for r in results if r.get("num_valid_months"))
                print(f"  [{done}/{total}] {elapsed:.0f}s 경과 | {rate:.1f}명/s "
                      f"| 유효 {valid_cnt}/{len(results)}명(미저장분)")
                save_results(results, OUTPUT_FILE)
                save_raw_results(raw_results, RAW_FILE)
                results = []
                raw_results = []

    if results:
        save_results(results, OUTPUT_FILE)
    if raw_results:
        save_raw_results(raw_results, RAW_FILE)

    elapsed = time.monotonic() - run_start
    print(f"\n=== 완료 | 소요: {elapsed:.1f}초 ===")

    if os.path.exists(OUTPUT_FILE):
        df = pd.read_csv(OUTPUT_FILE, encoding="utf-8-sig")
        valid = df[df["num_valid_months"].notna() & (df["num_valid_months"] >= 2)]
        print(f"{OUTPUT_FILE}: {len(df)}행 | num_valid_months≥2: {len(valid)}행")
        if "created_in_window" in df.columns and len(df) > 0:
            n_new = int((df["created_in_window"] == 1).sum())
            print(f"[편향 진단] created_in_window(관측창 내 생성 ≈ 신규클래스 렌): "
                  f"{n_new}명 ({100 * n_new / len(df):.1f}%) — 민감도 분석 시 이 그룹 제외 재실행")
        if len(valid) > 0:
            delta_cols = [
                c for c in df.columns
                if c.startswith("avg_monthly_delta") or c.startswith("recent")
            ]
            print(f"\n[월평균/최근 변화량 요약]")
            print(valid[delta_cols].describe().round(2).to_string())


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="월별 성장 피처 수집")
    parser.add_argument(
        "--refresh-raw",
        action="store_true",
        help="기수집 OCID도 다시 조회해 monthly_snapshots_raw.csv의 월별 접속·유니온 이력을 보완",
    )
    args = parser.parse_args()
    collect(refresh_raw=args.refresh_raw)
