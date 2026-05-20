"""
collect_features.py — 월별 스냅샷 기반 피처 수집 (2024-06 ~ 2026-05)

main_characters.csv 의 각 캐릭터에 대해 24개월치 월별 스냅샷을 수집하고
월평균 변화량(avg_monthly_delta)을 계산하여 features_monthly.csv 에 저장한다.

API 호출: ~1,300명 × 24개월 × 10회 = ~312,000회 → 400 req/s 기준 ~780초(~13분)
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

sys.stdout.reconfigure(encoding="utf-8")
load_dotenv()

API_KEY = os.getenv("MAPLE_API_KEY")
BASE_URL = "https://open.api.nexon.com/maplestory/v1"

_session = requests.Session()
_session.mount("https://", HTTPAdapter(pool_connections=60, pool_maxsize=60))
_session.headers.update({"x-nxopen-api-key": API_KEY})

# ── 설정 ─────────────────────────────────────────────────────────────────────
INPUT_FILE  = "main_characters.csv"
OUTPUT_FILE = "features_monthly.csv"
CONCURRENCY = 30
MAX_RPS     = 400

# 수집 월 목록: 2024-06 ~ 2026-05 (24개월)
MONTHS = []
_y, _m = 2024, 6
for _ in range(24):
    MONTHS.append(f"{_y:04d}-{_m:02d}")
    _m += 1
    if _m > 12:
        _m = 1
        _y += 1
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


def get_symbol_scores(symbol_data):
    """아케인심볼/어센틱심볼 레벨 합산을 각각 반환."""
    arcane, authentic = 0, 0
    if not symbol_data:
        return arcane, authentic
    for s in symbol_data.get("symbol", []):
        name = s.get("symbol_name", "")
        lv   = int(s.get("symbol_level") or 0)
        if "아케인심볼" in name:
            arcane += lv
        elif "어센틱심볼" in name:
            authentic += lv
    return arcane, authentic


def fetch_month_snapshot(ocid, year_month):
    """
    캐릭터 1명의 특정 월 스냅샷 수집 (10회 API 호출).
    basic이 None이면 전체 None 반환 (캐릭터 미존재 or API 범위 초과).
    반환: dict(level, exp, combat_power, union_level,
               arcane_symbol_score, authentic_symbol_score)
    """
    date_01 = f"{year_month}-01"

    basic = api_get("character/basic", {"ocid": ocid, "date": date_01})
    if not basic:
        return None

    level = int(basic.get("character_level") or 0) or None
    exp_raw = basic.get("character_exp")
    exp = int(exp_raw) if exp_raw is not None else None

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
    arcane, authentic = get_symbol_scores(symbol_data)

    return {
        "level":                  level,
        "exp":                    exp,
        "combat_power":           combat_power,
        "union_level":            union_level,
        "arcane_symbol_score":    arcane,
        "authentic_symbol_score": authentic,
    }


def avg_monthly_delta(indexed_values):
    """
    [(month_idx, value), ...] 에서 월평균 변화량 계산.
    None 값 제외 후 유효값 2개 미만이면 None 반환.
    """
    valid = [(i, v) for i, v in indexed_values if v is not None]
    if len(valid) < 2:
        return None
    first_idx, first_val = valid[0]
    last_idx,  last_val  = valid[-1]
    months_elapsed = last_idx - first_idx
    if months_elapsed == 0:
        return None
    return (last_val - first_val) / months_elapsed


def compute_features(monthly_snaps):
    """
    monthly_snaps: [(month_idx, snap_dict | None), ...] 24개
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

    fields = [
        "level", "combat_power", "union_level",
        "arcane_symbol_score", "authentic_symbol_score",
    ]
    indexed = {f: [(i, s.get(f)) for i, s in valid] for f in fields}

    return {
        "level":                              last_snap.get("level"),
        "union_level":                        last_snap.get("union_level"),
        "arcane_symbol_score":                last_snap.get("arcane_symbol_score"),
        "authentic_symbol_score":             last_snap.get("authentic_symbol_score"),
        "exp":                                exp_val,
        "log_exp":                            log_exp,
        "avg_monthly_delta_level":            avg_monthly_delta(indexed["level"]),
        "avg_monthly_delta_combat_power":     avg_monthly_delta(indexed["combat_power"]),
        "avg_monthly_delta_union_level":      avg_monthly_delta(indexed["union_level"]),
        "avg_monthly_delta_arcane_symbol":    avg_monthly_delta(indexed["arcane_symbol_score"]),
        "avg_monthly_delta_authentic_symbol": avg_monthly_delta(indexed["authentic_symbol_score"]),
        "first_valid_month":                  MONTHS[first_valid_idx],
        "last_valid_month":                   MONTHS[last_valid_idx],
        "num_valid_months":                   len(valid),
    }


_FEATURE_KEYS = [
    "level", "union_level", "arcane_symbol_score", "authentic_symbol_score",
    "exp", "log_exp",
    "avg_monthly_delta_level", "avg_monthly_delta_combat_power",
    "avg_monthly_delta_union_level", "avg_monthly_delta_arcane_symbol",
    "avg_monthly_delta_authentic_symbol",
    "first_valid_month", "last_valid_month", "num_valid_months",
]


def process_character(row):
    """캐릭터 1명: 24개월 스냅샷 수집 → 피처 계산 → dict 반환."""
    ocid = str(row["ocid"])

    monthly_snaps = []
    for i, ym in enumerate(MONTHS):
        snap = fetch_month_snapshot(ocid, ym)
        monthly_snaps.append((i, snap))

    features = compute_features(monthly_snaps)
    if features is None:
        features = {k: None for k in _FEATURE_KEYS}

    return {
        "character_name":  row["character_name"],
        "ocid":            ocid,
        "character_class": row.get("character_class", ""),
        "world_name":      row.get("world_name", ""),
        **features,
    }


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
        new_df = pd.concat([existing, new_df], ignore_index=True)
        new_df = new_df.drop_duplicates(subset="ocid", keep="first")
    new_df.to_csv(filepath, index=False, encoding="utf-8-sig")
    print(f"  [저장] {filepath} — 누적 {len(new_df)}행")


def collect():
    chars_df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")
    done_ocids = load_done_ocids(OUTPUT_FILE)
    targets = chars_df[~chars_df["ocid"].astype(str).isin(done_ocids)]
    total = len(targets)

    calls_est = total * 24 * 10
    print(f"=== 월별 피처 수집 시작 ===")
    print(f"수집 월: {MONTHS[0]} ~ {MONTHS[-1]} ({len(MONTHS)}개월)")
    print(f"대상: {total}명 (기수집 {len(done_ocids)}명 제외)")
    print(f"예상 API 호출: ~{calls_est:,}회 | 예상 소요: ~{calls_est / MAX_RPS:.0f}초\n")

    run_start = time.monotonic()
    results = []
    done = 0

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        futures = {
            executor.submit(process_character, row): row["character_name"]
            for _, row in targets.iterrows()
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            done += 1

            if done % 100 == 0 or done == total:
                elapsed = time.monotonic() - run_start
                rate = done / elapsed if elapsed > 0 else 0
                valid_cnt = sum(1 for r in results if r.get("num_valid_months"))
                print(f"  [{done}/{total}] {elapsed:.0f}s 경과 | {rate:.1f}명/s "
                      f"| 유효 {valid_cnt}/{len(results)}명")
                save_results(results, OUTPUT_FILE)
                results = []

    if results:
        save_results(results, OUTPUT_FILE)

    elapsed = time.monotonic() - run_start
    print(f"\n=== 완료 | 소요: {elapsed:.1f}초 ===")

    if os.path.exists(OUTPUT_FILE):
        df = pd.read_csv(OUTPUT_FILE, encoding="utf-8-sig")
        valid = df[df["num_valid_months"].notna() & (df["num_valid_months"] >= 2)]
        print(f"{OUTPUT_FILE}: {len(df)}행 | num_valid_months≥2: {len(valid)}행")
        if len(valid) > 0:
            delta_cols = [c for c in df.columns if c.startswith("avg_monthly_delta")]
            print(f"\n[월평균 변화량 요약]")
            print(valid[delta_cols].describe().round(2).to_string())


if __name__ == "__main__":
    collect()
