"""
collect_main_characters.py — 종합 랭킹 기반 본캐 수집 (270~290, 계열·직업·레벨 균형)

ranking/overall (레벨 내림차순) 에서 직업명별 이진 탐색으로 270~290 페이지 범위를 확정하고
5개 계열 × 400명을 수집한다. 계열 내에서도 직업(job)을 균형 있게 수집한다:
  - Phase 1(다양성): 가용 직업마다 base(=max(MIN_PER_CLASS, 400//직업수))명까지 우선 수집
  - Phase 2(채움): 목표 미달분을 MAX_PER_CLASS 한도 내 라운드로빈으로 채움
  → 단일 직업(예: 신규 클래스 렌)이 한 계열을 독점하는 문제 방지, 직업당 최소 MIN_PER_CLASS 보장

API 흐름: ranking/overall → id → character/basic(생성일 필터) → user/union-raider (본캐 판별)
본캐 판별: max(union_block.block_level) <= character_level
신규 캐릭 필터: character_date_create > CREATE_CUTOFF 이면 제외 (12개월 윈도우 관측 불가 캐릭 배제)

가설별 설계 근거:
  - 레벨 270~290 제한 (H1): 270~290 이 주차 후보 신호 구간; 저레벨대 정체 신호 + 고레벨(286~290) active 대조군
  - 5계열 × 400명 = 2,000명 (H3): 주차 후보 비율 ~10% 가정 시 ~200 minority class →
                                    5-fold CV per fold ~40 → RF/XGBoost cross-fold variance 안정,
                                    Precision 95% CI ±2~3% 로 수용 기준(>0.95) 판정 가능
  - 5계열 균등 (H2): cluster × class_group Chi-Square 셀별 기대 빈도 ≥ 5 확보
  - 3 level_bin × 균등 (H2): cluster × level_band Chi-Square 셀별 기대 빈도 확보
  - 본캐 필터 (H1/H3): 부캐 성장 정체는 본캐 활동의 부산물 → 노이즈 제거
  - world_type=0 (All): 리부트는 메소 거래 불가 → 주차 후보 인센티브 자체가 다름

재현성: RANDOM_SEED + TARGET_DATE 고정으로 동일 결과 보장.
"""

import os
import sys
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding="utf-8")
# .env는 프로젝트 루트 기준으로 명시 로드 (CWD 의존성 제거)
_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_ENV_PATH)

API_KEY  = os.getenv("MAPLE_API_KEY")
BASE_URL = "https://open.api.nexon.com/maplestory/v1"

_session = requests.Session()
_session.mount("https://", HTTPAdapter(pool_connections=60, pool_maxsize=60))
_session.headers.update({"x-nxopen-api-key": API_KEY})

# ── 설정 ─────────────────────────────────────────────────────────────────────
RANDOM_SEED      = 42   # 페이지 셔플 재현성 — 동일 TARGET_DATE + seed → 동일 샘플
TARGET_DATE      = "2026-05-16"
OUTPUT_FILE      = str(Path(__file__).resolve().parent.parent / "data" / "main_characters.csv")
LEVEL_MIN        = 270
LEVEL_MAX        = 290
GROUPS           = ["전사", "마법사", "궁수", "도적", "해적"]
TARGET_PER_GROUP = 400   # 5계열 × 400 = 2,000명 (H3 supervised CV/Rule threshold 통계 안정성 확보)
# hi 는 exclusive: (270,280)=270~279, (280,286)=280~285, (286,291)=286~290
LEVEL_BINS       = [(270, 280), (280, 286), (286, 291)]
BIN_LABELS       = ["270~279", "280~285", "286~290"]
_PER_BIN         = TARGET_PER_GROUP // len(LEVEL_BINS)              # 133
BIN_TARGETS      = [_PER_BIN, _PER_BIN, _PER_BIN + (TARGET_PER_GROUP - _PER_BIN * len(LEVEL_BINS))]  # [133,133,134]
CONCURRENCY      = 30
MAX_RPS          = 400
MAX_POOL_PASSES  = 5    # 페이지 풀 최대 순환 횟수 (무한루프 방지)

# 신규 캐릭 필터: 생성일이 이 날짜 이후면 제외 (12개월 윈도우 관측 불가 → delta 신뢰 불가)
#   2025-06-30 = 관측 윈도우 시작월(2025-06) 말. 6월 생성(신규 클래스 렌 포함)까지는 허용.
CREATE_CUTOFF    = "2025-06-30"
# 계열 내 직업(job) 균형: 한 직업이 계열을 독점하지 않도록
MIN_PER_CLASS    = 10    # Phase 1 — 가용 직업당 최소 수집 목표
MAX_PER_CLASS    = 100   # Phase 2 — 직업당 상한 (단일 직업 독점 방지)
# ─────────────────────────────────────────────────────────────────────────────

CLASS_GROUP_MAP = {
    # 전사 (13)
    "히어로": "전사", "팔라딘": "전사", "다크나이트": "전사",
    "소울마스터": "전사", "미하일": "전사", "블래스터": "전사",
    "데몬슬레이어": "전사", "데몬어벤져": "전사", "아란": "전사",
    "카이저": "전사", "아델": "전사", "렌": "전사", "제로": "전사",
    # 마법사 (10)
    "아크메이지(불,독)": "마법사", "아크메이지(썬,콜)": "마법사", "비숍": "마법사",
    "플레임위자드": "마법사", "배틀메이지": "마법사", "에반": "마법사",
    "루미너스": "마법사", "일리움": "마법사", "라라": "마법사", "키네시스": "마법사",
    # 궁수 (7)
    "보우마스터": "궁수", "신궁": "궁수", "패스파인더": "궁수",
    "윈드브레이커": "궁수", "와일드헌터": "궁수", "메르세데스": "궁수", "카인": "궁수",
    # 도적 (9) — 제논은 도적/해적 하이브리드. Nexon 원분류(2013) + LUK 메인스탯 기준으로
    # 도적 단일 분류. H2 Chi-Square 독립성 가정 충족 위해 중복 분류 회피.
    # 향후 sensitivity 분석으로 제논만 해적 재분류 후 H2 재실행 가능.
    "나이트로드": "도적", "섀도어": "도적", "듀얼블레이더": "도적",
    "나이트워커": "도적", "팬텀": "도적", "카데나": "도적",
    "칼리": "도적", "호영": "도적", "제논": "도적",
    # 해적 (8)
    "바이퍼": "해적", "캡틴": "해적", "캐논마스터": "해적",
    "스트라이커": "해적", "메카닉": "해적", "은월": "해적",
    "엔젤릭버스터": "해적", "아크": "해적",
}

GROUP_TO_CLASSES = {
    grp: [c for c, g in CLASS_GROUP_MAP.items() if g == grp]
    for grp in GROUPS
}

# Nexon API `class` 파라미터 매핑 — "{소속}-{4차직업}" 또는 "{직업}-전체 전직"
# (1·2차 및 3차 전직 제외; 영웅·신규 직업은 "-전체 전직" 단일 항목)
CLASS_API_PARAM = {
    # 전사
    "히어로":             "전사-히어로",
    "팔라딘":             "전사-팔라딘",
    "다크나이트":         "전사-다크나이트",
    "소울마스터":         "기사단-소울마스터",
    "블래스터":           "레지스탕스-블래스터",
    "데몬슬레이어":       "레지스탕스-데몬슬레이어",
    "데몬어벤져":         "레지스탕스-데몬어벤져",
    "아란":               "아란-전체 전직",
    "카이저":             "카이저-전체 전직",
    "제로":               "초월자-제로",
    "아델":               "아델-전체 전직",
    "호영":               "호영-전체 전직",
    # 마법사
    "아크메이지(불,독)":  "마법사-아크메이지(불,독)",
    "아크메이지(썬,콜)":  "마법사-아크메이지(썬,콜)",
    "비숍":               "마법사-비숍",
    "배틀메이지":         "레지스탕스-배틀메이지",
    "에반":               "에반-전체 전직",
    "루미너스":           "루미너스-전체 전직",
    "플레임위자드":       "기사단-플레임위자드",
    "키네시스":           "프렌즈 월드-키네시스",
    "일리움":             "일리움-전체 전직",
    "라라":               "라라-전체 전직",
    "칼리":               "칼리-전체 전직",
    # 궁수
    "보우마스터":         "궁수-보우마스터",
    "신궁":               "궁수-신궁",
    "패스파인더":         "궁수-패스파인더",
    "윈드브레이커":       "기사단-윈드브레이커",
    "와일드헌터":         "레지스탕스-와일드헌터",
    "메르세데스":         "메르세데스-전체 전직",
    "카인":               "카인-전체 전직",
    "카데나":             "카데나-전체 전직",
    # 도적
    "나이트로드":         "도적-나이트로드",
    "섀도어":             "도적-섀도어",
    "듀얼블레이더":       "도적-듀얼블레이더",
    "나이트워커":         "기사단-나이트워커",
    "제논":               "레지스탕스-제논",
    "은월":               "은월-전체 전직",
    "팬텀":               "팬텀-전체 전직",
    "미하일":             "기사단-미하일",
    # 해적
    "바이퍼":             "해적-바이퍼",
    "캡틴":               "해적-캡틴",
    "캐논마스터":         "해적-캐논마스터",
    "스트라이커":         "기사단-스트라이커",
    "메카닉":             "레지스탕스-메카닉",
    "엔젤릭버스터":       "엔젤릭버스터-전체 전직",
    "아크":               "아크-전체 전직",
    "렌":                 "렌-전체 전직",
}

# 두 매핑의 키 일치 확인 (모듈 로드 시점)
_diff = set(CLASS_GROUP_MAP) ^ set(CLASS_API_PARAM)
assert not _diff, f"CLASS_GROUP_MAP ↔ CLASS_API_PARAM 키 불일치: {_diff}"


# ── 인프라 ────────────────────────────────────────────────────────────────────

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

# 엔드포인트별 첫 비-200 응답만 출력 (호출 폭주 시 로그 노이즈 방지)
_api_err_logged = {}


def api_get(endpoint, params):
    _limiter.acquire()
    try:
        r = _session.get(f"{BASE_URL}/{endpoint}", params=params, timeout=10)
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
        return None
    # 그 외 비-200: 엔드포인트당 최초 1회만 본문 일부 출력
    if not _api_err_logged.get(endpoint):
        body = (r.text or "")[:200].replace("\n", " ")
        print(f"  [API {r.status_code}] {endpoint} params={params}")
        print(f"           body={body}")
        _api_err_logged[endpoint] = True
    return None


def smoke_test():
    """수집 시작 전 API 키 + URL + 파라미터 정상성 사전 확인."""
    print("=== API 연결 사전 확인 ===")
    if not API_KEY:
        print(f"  !! API_KEY 미설정 — {_ENV_PATH} 의 MAPLE_API_KEY 확인 필요")
        return False
    print(f"  API_KEY: 설정됨 ({API_KEY[:8]}...{API_KEY[-4:]})")
    print(f"  .env  : {_ENV_PATH}")
    print(f"  date  : {TARGET_DATE}")

    test = api_get("ranking/overall", {
        "date": TARGET_DATE,
        "world_type": 0,
        "class": CLASS_API_PARAM["히어로"],   # "전사-히어로"
        "page": 1,
    })
    if test and test.get("ranking"):
        top = test["ranking"][0]
        print(f"  ranking/overall(전사-히어로, p1) OK — 1위 Lv{top.get('character_level')} {top.get('character_name')}\n")
        return True
    print(f"  !! ranking/overall(전사-히어로, p1) 실패 — 위 [API ...] 메시지 확인\n")
    return False


# ── 페이지 탐색 ───────────────────────────────────────────────────────────────

def get_page_top_level(class_name, page):
    """해당 페이지 첫 캐릭터 레벨 반환. 빈 페이지면 None."""
    data = api_get("ranking/overall", {
        "date": TARGET_DATE,
        "world_type": 0,
        "class": CLASS_API_PARAM[class_name],
        "page": page,
    })
    if not data or not data.get("ranking"):
        return None
    return int(data["ranking"][0].get("character_level", 0))


def find_last_page(class_name):
    """지수 탐색 + 이진 탐색으로 마지막 유효 페이지 번호 반환."""
    page = 1
    while True:
        lv = get_page_top_level(class_name, page)
        if lv is None:
            break
        if page >= 30_000:
            return page
        page *= 2
    lo, hi = max(1, page // 2), page
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if get_page_top_level(class_name, mid) is None:
            hi = mid - 1
        else:
            lo = mid
    return lo


def find_page_range(class_name):
    """
    이진 탐색으로 (start_page, end_page) 반환.
      start_page: top_level ≤ LEVEL_MAX 인 첫 페이지
      end_page  : top_level ≥ LEVEL_MIN 인 마지막 페이지
    범위 없으면 (None, None).
    """
    last = find_last_page(class_name)
    if last == 0:
        return None, None

    # ── start_page 탐색 ───────────────────────────────────────
    first_lv = get_page_top_level(class_name, 1)
    if first_lv is None:
        return None, None
    if first_lv <= LEVEL_MAX:
        start_page = 1
    else:
        lo, hi = 1, last
        while lo < hi:
            mid = (lo + hi) // 2
            lv = get_page_top_level(class_name, mid)
            if lv is None or lv < LEVEL_MIN:
                hi = mid           # 너무 뒤 (빈 페이지 or 범위 하한 미달)
            elif lv > LEVEL_MAX:
                lo = mid + 1       # 아직 290 초과 구간
            else:
                hi = mid           # ≤ LEVEL_MAX 진입, 더 앞 탐색
        start_page = lo

    sp_lv = get_page_top_level(class_name, start_page)
    if sp_lv is None or sp_lv < LEVEL_MIN:
        return None, None          # 270~290 캐릭터 없음

    # ── end_page 탐색 ────────────────────────────────────────
    lo, hi = start_page, last
    while lo < hi:
        mid = (lo + hi + 1) // 2
        lv = get_page_top_level(class_name, mid)
        if lv is None or lv < LEVEL_MIN:
            hi = mid - 1
        else:
            lo = mid
    end_page = lo

    if end_page < start_page:
        return None, None
    return start_page, end_page


# ── 후보 처리 ─────────────────────────────────────────────────────────────────

def fetch_ranking_page(class_name, page):
    """한 페이지(최대 100명) 랭킹 항목 반환."""
    data = api_get("ranking/overall", {
        "date": TARGET_DATE,
        "world_type": 0,
        "class": CLASS_API_PARAM[class_name],
        "page": page,
    })
    if not data:
        return []
    return data.get("ranking", [])


def level_bin_idx(level):
    """레벨 → 빈 인덱스 (0~2). 범위 밖이면 -1."""
    if level < LEVEL_MIN or level > LEVEL_MAX:
        return -1
    for i, (lo, hi) in enumerate(LEVEL_BINS):
        if lo <= level < hi:
            return i
    return -1


def process_candidate(entry, class_name, seen_ocids, lock):
    """
    랭킹 항목 1개 처리.
    ranking/overall 이 level/world 포함 → character/basic 호출 생략.
    id → user/union-raider (2회 API 호출).
    본캐이면 dict, 부캐/실패이면 None.

    character_class 권위 값:
      API 응답의 character_class 는 CLASS_API_PARAM 의 "{소속}-{직업}" 중 소속(prefix)만
      돌려주는 경우가 많아 신뢰 불가 (예: '마법사-아크메이지(썬,콜)' → API='마법사',
      '프렌즈 월드-키네시스' → API='프렌즈 월드'). 우리가 이미 직업명을 지정해 조회
      하므로 class_name 을 권위 있는 값으로 사용해 character_class / class_group 을 채운다.
    """
    char_name  = entry.get("character_name", "")
    char_level = int(entry.get("character_level") or 0)
    world_name = entry.get("world_name", "")
    union_lv   = int(entry.get("union_level") or 0)

    if not char_name or char_level < LEVEL_MIN or char_level > LEVEL_MAX:
        return None

    id_data = api_get("id", {"character_name": char_name})
    if not id_data:
        return None
    ocid = id_data.get("ocid", "")
    if not ocid:
        return None

    with lock:
        if ocid in seen_ocids:
            return None
        seen_ocids.add(ocid)

    # 신규 캐릭 필터 (수집 단계): 생성일 > CREATE_CUTOFF 이면 제외
    basic = api_get("character/basic", {"ocid": ocid, "date": TARGET_DATE})
    date_create = basic.get("character_date_create") if basic else None
    if not date_create or date_create[:10] > CREATE_CUTOFF:
        return None

    raider = api_get("user/union-raider", {"ocid": ocid, "date": TARGET_DATE})
    if not raider:
        return None
    max_block_lv = max(
        (int(b.get("block_level") or 0) for b in raider.get("union_block", [])),
        default=0,
    )
    if max_block_lv > char_level:
        return None    # 더 높은 레벨 캐릭터 존재 → 부캐

    return {
        "character_name":       char_name,
        "ocid":                 ocid,
        "level":                char_level,
        "character_class":      class_name,
        "world_name":           world_name,
        "union_level":          union_lv,
        "class_group":          CLASS_GROUP_MAP[class_name],
        "character_date_create": date_create[:10],
    }


# ── 저장 ─────────────────────────────────────────────────────────────────────

def save_append(rows, filepath):
    if not rows:
        return
    new_df = pd.DataFrame(rows)
    if os.path.exists(filepath):
        existing = pd.read_csv(filepath, encoding="utf-8-sig")
        new_df   = pd.concat([existing, new_df], ignore_index=True)
        new_df   = new_df.drop_duplicates(subset="ocid", keep="first")
    new_df.to_csv(filepath, index=False, encoding="utf-8-sig")
    print(f"  [저장] {filepath} — 누적 {len(new_df)}행")


# ── 계열별 수집 ───────────────────────────────────────────────────────────────

def collect_group(group_name, seen_ocids, lock):
    """
    단일 계열 TARGET_PER_GROUP 명 수집 — 계열 내 직업(job) 균형 보장.
      Phase 1(다양성): 가용 직업마다 base(=max(MIN_PER_CLASS, 400//직업수))명까지 우선 수집
      Phase 2(채움):  목표 미달분을 MAX_PER_CLASS 한도 내 라운드로빈으로 채움
    레벨 구간(빈)은 BIN_TARGETS 소프트 캡으로 균형 유지. 생성일 필터는 process_candidate 내.
    """
    classes     = GROUP_TO_CLASSES[group_name]
    group_start = time.monotonic()

    # ── 직업별 페이지 범위 → 셔플된 페이지 리스트 + 커서 ──────────────
    print(f"\n[{group_name}] 페이지 범위 탐색 중 ({len(classes)}개 직업)...")
    job_pages, cursor = {}, {}
    for cls in classes:
        sp, ep = find_page_range(cls)
        if sp is None:
            print(f"  {cls}: 범위 없음")
            continue
        pages = list(range(sp, ep + 1))
        random.shuffle(pages)
        job_pages[cls] = pages
        cursor[cls]    = 0
        print(f"  {cls}: p{sp}~p{ep} ({len(pages)}페이지)")

    available = list(job_pages.keys())
    if not available:
        print(f"[{group_name}] 수집 가능 페이지 없음 — 건너뜀")
        return []

    base = min(MAX_PER_CLASS, max(MIN_PER_CLASS, TARGET_PER_GROUP // len(available)))
    print(f"  → 직업 {len(available)}개 | Phase1 직업당 {base}명 | 직업당 상한 {MAX_PER_CLASS}명")

    class_counts = {c: 0 for c in available}
    bin_counts   = [0] * len(LEVEL_BINS)
    group_rows   = []
    total        = [0]
    last_print   = [0.0]

    def process_page(cls, ceiling):
        """cls 의 다음 페이지 1장 처리. 직업 ceiling·빈·총량 캡 적용. 페이지 소비 여부 반환."""
        if cursor[cls] >= len(job_pages[cls]):
            return False
        page = job_pages[cls][cursor[cls]]
        cursor[cls] += 1
        candidates = fetch_ranking_page(cls, page)
        to_process = [
            e for e in candidates
            if LEVEL_MIN <= int(e.get("character_level") or 0) <= LEVEL_MAX
        ]
        if not to_process:
            return True
        with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
            futures = [
                executor.submit(process_candidate, entry, cls, seen_ocids, lock)
                for entry in to_process
            ]
            # 카운터/append 는 호출 스레드(단일)에서만 수행 — 별도 락 불필요
            for future in as_completed(futures):
                r = future.result()
                if r is None or total[0] >= TARGET_PER_GROUP:
                    continue
                bi = level_bin_idx(r["level"])
                if class_counts[cls] >= ceiling or bi < 0 or bin_counts[bi] >= BIN_TARGETS[bi]:
                    continue
                class_counts[cls] += 1
                bin_counts[bi]    += 1
                total[0]          += 1
                group_rows.append(r)
        return True

    def maybe_print():
        now = time.monotonic()
        if now - last_print[0] >= 60 or total[0] >= TARGET_PER_GROUP:
            elapsed = now - group_start
            bins    = " | ".join(f"{BIN_LABELS[i]}:{bin_counts[i]}" for i in range(len(LEVEL_BINS)))
            active  = sum(1 for c in available if class_counts[c] > 0)
            print(f"  [{group_name}] {total[0]}/{TARGET_PER_GROUP}명 | 직업 {active}/{len(available)} 활성 | {bins} | {elapsed:.0f}s")
            last_print[0] = now

    # ── Phase 1: 직업 다양성 (각 직업 base 까지) ──────────────────────
    progress = True
    while progress and total[0] < TARGET_PER_GROUP:
        progress = False
        for cls in available:
            if total[0] >= TARGET_PER_GROUP:
                break
            if class_counts[cls] >= base or cursor[cls] >= len(job_pages[cls]):
                continue
            if process_page(cls, base):
                progress = True
            maybe_print()

    # ── Phase 2: 목표 채움 (MAX_PER_CLASS 한도, 라운드로빈) ────────────
    progress = True
    while progress and total[0] < TARGET_PER_GROUP:
        progress = False
        for cls in available:
            if total[0] >= TARGET_PER_GROUP:
                break
            if class_counts[cls] >= MAX_PER_CLASS or cursor[cls] >= len(job_pages[cls]):
                continue
            if process_page(cls, MAX_PER_CLASS):
                progress = True
            maybe_print()

    elapsed = time.monotonic() - group_start
    nonzero = {c: class_counts[c] for c in available if class_counts[c] > 0}
    print(f"  [{group_name}] 완료 {total[0]}명 | {elapsed/60:.1f}분 | 직업 {len(nonzero)}/{len(available)}개")
    print(f"     직업별: " + ", ".join(f"{c}:{n}" for c, n in sorted(nonzero.items(), key=lambda x: -x[1])))
    return group_rows


# ── 메인 ─────────────────────────────────────────────────────────────────────

def collect():
    if not smoke_test():
        print("[중단] API 호출이 실패합니다. 출력된 status code/body 와 .env 경로를 확인하세요.")
        return

    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)
        print(f"기존 파일 삭제: {OUTPUT_FILE}")

    random.seed(RANDOM_SEED)   # 페이지 셔플 재현성 확보 (H1/H2/H3 결과 재현 가능)

    seen_ocids = set()
    lock       = threading.Lock()
    run_start  = time.monotonic()

    total_target = len(GROUPS) * TARGET_PER_GROUP
    print(f"=== 종합 랭킹 기반 본캐 수집 ===")
    print(f"타겟: {LEVEL_MIN}~{LEVEL_MAX} | {len(GROUPS)}계열 × {TARGET_PER_GROUP}명 = {total_target}명")
    print(f"빈 목표(계열별): {dict(zip(BIN_LABELS, BIN_TARGETS))}")
    print(f"날짜: {TARGET_DATE} | seed: {RANDOM_SEED}\n")

    all_rows = []
    for group in GROUPS:
        rows = collect_group(group, seen_ocids, lock)
        all_rows.extend(rows)
        save_append(rows, OUTPUT_FILE)
        print(f"  → [{group}] 저장 완료 | 누적 {len(all_rows)}명\n")

    elapsed = time.monotonic() - run_start

    if not os.path.exists(OUTPUT_FILE):
        print(f"=== 완료 | {elapsed/60:.1f}분 | 출력 없음 ===")
        return

    df = pd.read_csv(OUTPUT_FILE, encoding="utf-8-sig")
    print(f"=== 완료 | {len(df)}명 | {elapsed/60:.1f}분 ===")
    print("\n[검증]")
    print(f"  행 수: {len(df)} (목표 {total_target})")
    print(f"  OCID 중복: {df['ocid'].duplicated().sum()}")
    print(f"  레벨 범위 위반: {((df['level'] < LEVEL_MIN) | (df['level'] > LEVEL_MAX)).sum()}")

    print("\n계열별 분포:")
    print(df["class_group"].value_counts().to_string())
    print("\n레벨 구간 분포:")
    df["_bin"] = pd.cut(
        df["level"],
        bins=[lo for lo, _ in LEVEL_BINS] + [LEVEL_BINS[-1][1]],
        labels=BIN_LABELS,
        right=False,
    )
    print(df["_bin"].value_counts().sort_index().to_string())

    print("\n계열 내 직업(job) 분포:")
    for grp in GROUPS:
        sub = df[df["class_group"] == grp]["character_class"].value_counts()
        if len(sub) == 0:
            continue
        min_n = int(sub.min())
        flag = "" if min_n >= MIN_PER_CLASS else f"  ⚠ 최소직업 {min_n}<{MIN_PER_CLASS}"
        print(f"  [{grp}] {len(sub)}개 직업 (min {min_n}, max {int(sub.max())}){flag}")
        print("     " + ", ".join(f"{c}:{n}" for c, n in sub.items()))

    # H2 검정 가능성 사전 확인: cluster × class_group × level_band 셀별 충분성
    print("\n[H2 사전 확인] cluster_label 추가 시 셀 크기 (주차 후보 비율 10% 가정):")
    print(f"  cluster × class_group  ({2 * len(GROUPS)}셀): 평균 {len(df) * 0.10 / len(GROUPS):.1f}/셀 (주차 후보)")
    print(f"  cluster × level_band   ({2 * len(LEVEL_BINS)}셀): 평균 {len(df) * 0.10 / len(LEVEL_BINS):.1f}/셀 (주차 후보)")


if __name__ == "__main__":
    collect()
