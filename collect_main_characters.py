"""
collect_main_characters.py - 메이플스토리 본캐릭터 수집

API 제한: 500 req/s, 20,000,000 req/일
병렬화: ThreadPoolExecutor + RateLimiter 로 API 한도 최대 활용
목표: 층화 랜덤 샘플링으로 1,000명 본캐 수집 (약 100~200초 예상)

본캐 판별:
  user/union-raider 의 union_block 중 block_level 최댓값 <= 현재 캐릭터 레벨
  이면 본캐 (본인 블록이 항상 포함되므로, 더 높은 블록 = 더 높은 레벨 부캐 존재)
"""

import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding="utf-8")
load_dotenv()

API_KEY = os.getenv("MAPLE_API_KEY")
BASE_URL = "https://open.api.nexon.com/maplestory/v1"

# 커넥션 풀 확대: 스레드 수만큼 동시 연결 허용
_session = requests.Session()
_session.mount("https://", HTTPAdapter(pool_connections=60, pool_maxsize=60))
_session.headers.update({"x-nxopen-api-key": API_KEY})

# ── 설정 ─────────────────────────────────────────────────────────────────────
TARGET_DATE = "2026-05-16"
OUTPUT_FILE = "main_characters.csv"
TARGET_COUNT = 1300    # 수집 목표 인원 (달성 시 조기 종료)
CONCURRENCY = 30       # 동시 처리 스레드 수
MAX_RPS = 400          # req/s 상한 (API 한도 500의 80%, 안전 마진)

# 유니온 랭킹 페이지 층화 구간
# 상위 페이지 = 높은 유니온 레벨 / 하위 페이지 = 낮은 유니온 레벨
# 주차 유저는 Tier 4 (중위) 에 밀집 예상
PAGE_TIERS = [
    (1,    50),    # Tier 1: 최상위 유니온 (~9,000+)
    (51,   300),   # Tier 2: 고유니온 (~7,000-9,000)
    (301,  1000),  # Tier 3: 중상위 (~5,000-7,000)
    (1001, 3000),  # Tier 4: 중위 (~3,000-5,000)
    (3001, 6000),  # Tier 5: 중하위 (~1,000-3,000)
]
PAGES_PER_TIER = 5     # 구간별 랜덤 선택 페이지 수 (총 25 페이지 x 100명)
# ─────────────────────────────────────────────────────────────────────────────


class RateLimiter:
    """
    초당 호출 수를 MAX_RPS 이하로 유지하는 토큰 버킷 (스레드 안전).
    모든 스레드가 단일 인스턴스를 공유한다.
    """
    def __init__(self, calls_per_second: int):
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


def api_get(endpoint: str, params: dict):
    """속도 제한이 적용된 단일 API 호출. 성공 시 dict, 실패 시 None."""
    _limiter.acquire()
    url = f"{BASE_URL}/{endpoint}"
    try:
        r = _session.get(url, params=params, timeout=10)
    except Exception as e:
        print(f"  [ERR] {endpoint}: {e}")
        return None

    if r.status_code == 200:
        try:
            return r.json()
        except ValueError:
            return None
    if r.status_code == 429:
        print("  [429] 속도 제한 초과 - 2s 대기")
        time.sleep(2)
    else:
        print(f"  [HTTP {r.status_code}] {endpoint}")
    return None


def process_candidate(entry: dict, seen_ocids: set, lock: threading.Lock):
    """
    랭킹 항목 1개 처리. 최대 3회 순차 API 호출 (id -> basic -> union-raider).
    본캐이면 dict, 부캐/실패이면 None 반환.
    """
    char_name = entry.get("character_name", "")
    union_lv = entry.get("union_level", 0)

    # 호출 1: OCID 조회
    id_data = api_get("id", {"character_name": char_name})
    if not id_data:
        return None
    ocid = id_data.get("ocid", "")
    if not ocid:
        return None

    # OCID 중복 확인 및 원자적 예약 (다른 스레드가 동일 OCID 처리하지 않도록)
    with lock:
        if ocid in seen_ocids:
            return None
        seen_ocids.add(ocid)

    # 호출 2: 캐릭터 기본 정보 (레벨, 직업, 월드)
    basic = api_get("character/basic", {"ocid": ocid, "date": TARGET_DATE})
    if not basic:
        return None
    char_level = int(basic.get("character_level") or 0)
    char_class = basic.get("character_class", "")
    world_name = basic.get("world_name", "")
    if char_level <= 0 or not char_class:
        return None  # 날짜 기준 데이터 미존재 캐릭터 제외

    # 호출 3: 유니온 공격대 블록으로 본캐 여부 확인
    raider = api_get("user/union-raider", {"ocid": ocid, "date": TARGET_DATE})
    if not raider:
        return None
    max_block_lv = max(
        (int(b.get("block_level") or 0) for b in raider.get("union_block", [])),
        default=0,
    )
    if max_block_lv > char_level:
        return None  # 더 높은 레벨 캐릭터(부캐) 존재

    return {
        "character_name": char_name,
        "ocid": ocid,
        "level": char_level,
        "character_class": char_class,
        "world_name": world_name,
        "union_level": union_lv,
    }


def load_existing(filepath: str):
    """기존 CSV 로드. (DataFrame, ocid_set) 반환. 없으면 빈 값."""
    if not os.path.exists(filepath):
        return pd.DataFrame(), set()
    df = pd.read_csv(filepath, encoding="utf-8-sig")
    ocids = set(df["ocid"].dropna().astype(str)) if "ocid" in df.columns else set()
    return df, ocids


def save_append(new_rows: list, filepath: str):
    """new_rows 를 기존 CSV 에 누적 저장 (OCID 기준 중복 제거)."""
    if not new_rows:
        return
    existing_df, _ = load_existing(filepath)
    combined = pd.concat([existing_df, pd.DataFrame(new_rows)], ignore_index=True)
    combined = combined.drop_duplicates(subset="ocid", keep="first")
    combined.to_csv(filepath, index=False, encoding="utf-8-sig")
    print(f"  [저장] +{len(new_rows)}명 -> {filepath} 누적 {len(combined)}명")


def build_page_list() -> list:
    """각 Tier 에서 PAGES_PER_TIER 개 페이지를 랜덤 선택해 섞어서 반환."""
    pages = []
    for start, end in PAGE_TIERS:
        n = min(PAGES_PER_TIER, end - start + 1)
        pages.extend(random.sample(range(start, end + 1), n))
    random.shuffle(pages)
    return pages


def collect():
    lock = threading.Lock()
    _, seen_ocids = load_existing(OUTPUT_FILE)
    total_found = len(seen_ocids)  # 기존 수집 인원 포함

    print(f"=== 수집 시작 | 기존 {total_found}명 | 목표 {TARGET_COUNT}명 ===")
    print(f"병렬 처리: {CONCURRENCY} 워커 | 속도 제한: {MAX_RPS} req/s\n")

    pages = build_page_list()
    # 페이지당 100 후보 x 3회 + 랭킹 1회
    est_calls = len(pages) * (1 + 100 * 3)
    est_sec = est_calls / MAX_RPS
    print(f"방문 예정: {len(pages)} 페이지 ({PAGES_PER_TIER}개/Tier)")
    print(f"예상 API 호출: ~{est_calls}회 | 예상 소요: ~{est_sec:.0f}초")
    print(f"페이지 목록: {sorted(pages)}\n")

    run_start = time.monotonic()

    for i, page in enumerate(pages, 1):
        if total_found >= TARGET_COUNT:
            print(f"목표 {TARGET_COUNT}명 달성 - 수집 종료")
            break

        page_start = time.monotonic()
        print(f"[{i}/{len(pages)}] 랭킹 page={page} | 현재 {total_found}명 수집됨")

        ranking_data = api_get("ranking/union", {"date": TARGET_DATE, "page": page})
        if not ranking_data or not ranking_data.get("ranking"):
            print(f"  page {page}: 빈 결과, 건너뜀\n")
            continue

        candidates = ranking_data["ranking"]
        new_rows = []

        # 페이지 내 100명 캐릭터를 CONCURRENCY 스레드로 병렬 처리
        with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
            futures = [
                executor.submit(process_candidate, entry, seen_ocids, lock)
                for entry in candidates
            ]
            for future in as_completed(futures):
                result = future.result()
                if result:
                    new_rows.append(result)
                    total_found += 1

        elapsed_page = time.monotonic() - page_start
        elapsed_total = time.monotonic() - run_start
        tps = total_found / elapsed_total if elapsed_total > 0 else 0
        print(f"  본캐 {len(new_rows)}/{len(candidates)}명 | "
              f"{elapsed_page:.1f}s/page | 누적 {total_found}명 | {tps:.1f}명/s\n")

        save_append(new_rows, OUTPUT_FILE)

    elapsed = time.monotonic() - run_start
    _, final_ocids = load_existing(OUTPUT_FILE)
    print(f"=== 완료 | 소요: {elapsed:.1f}초 | {OUTPUT_FILE} 누적: {len(final_ocids)}명 ===")


if __name__ == "__main__":
    collect()
