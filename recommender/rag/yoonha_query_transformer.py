"""
query_transformer.py
그룹 조건(group) + 자연어 입력(user_text) → 검색용 쿼리 변환

출력:
    query_text    : BM25용 자연어 쿼리
    query_filter  : hard_filter 조건 dict
    emotion_tags  : tag_filter용 감정 태그 리스트
    anchor_titles : dense embedding용 앵커 타이틀 리스트
"""

from __future__ import annotations

# -------------------------
# 상수
# -------------------------

# weight_pref → weight_max 매핑
WEIGHT_MAX_MAP = {
    "light": 2.5,
    "medium": 3.5,
    # heavy는 weight_max 없음 (하한선만 적용)
}

# weight_pref → BM25 쿼리 키워드
WEIGHT_TEXT_MAP = {
    "light":  "가벼운 입문 간단",
    "medium": "중간 보통 중급",
    "heavy":  "무거운 복잡 전략 고급",
}

# category → BM25 쿼리 키워드
CATEGORY_TEXT_MAP = {
    "Strategy":    "전략 strategy",
    "Economic":    "경제 economic",
    "Party":       "파티 party",
    "War":         "전쟁 war",
    "Family":      "가족 family",
    "Abstract":    "추상 abstract",
    "Cooperative": "협력 cooperative",
    "Deduction":   "추리 deduction",
    "Card Game":   "카드게임 card",
    "Thematic":    "테마 thematic",
}

# mechanism → BM25 쿼리 키워드
MECHANISM_TEXT_MAP = {
    "Worker Placement": "일꾼배치 worker placement",
    "Deck Building":    "덱빌딩 deck building",
    "Engine Building":  "엔진빌딩 engine building",
    "Area Control":     "지역장악 area control",
    "Market":           "시장 market",
    "Drafting":         "드래프팅 drafting",
    "Cooperative Game": "협력 cooperative",
}

# horror_tolerance → 감정 태그
HORROR_EMOTION_MAP = {
    0: ["공포없음"],
    1: [],
    2: [],
}

# weight_pref → 감정 태그
WEIGHT_EMOTION_MAP = {
    "light":  ["입문용", "가볍게즐길수있음"],
    "medium": [],
    "heavy":  [],
}

# 보드게임 카테고리별 앵커 타이틀 (dense embedding 기준점)
BOARDGAME_ANCHORS = {
    "Strategy":    ["Brass: Birmingham", "Twilight Struggle", "Terra Mystica"],
    "Economic":    ["Brass: Birmingham", "Ark Nova", "Terraforming Mars"],
    "Party":       ["Codenames", "Dixit", "Wavelength"],
    "Cooperative": ["Pandemic", "Spirit Island", "Arkham Horror"],
    "Deduction":   ["Mysterium", "Codenames", "Sherlock Holmes"],
    "Family":      ["Ticket to Ride", "Carcassonne", "Wingspan"],
    "War":         ["Twilight Struggle", "War of the Ring", "Memoir 44"],
}

# 머더미스터리 앵커 타이틀
MM_ANCHORS = {
    "default": ["구두룡 저택의 살인"],
}


# -------------------------
# 내부 헬퍼
# -------------------------
def _build_query_text(user_text: str, group: dict, category: str) -> str:
    """BM25용 자연어 쿼리 생성."""
    parts = [user_text] if user_text else []

    # 인원
    headcount = group.get("headcount")
    if headcount:
        parts.append(f"{headcount}인")

    # 난이도
    weight_pref = group.get("weight_pref")
    if weight_pref and weight_pref in WEIGHT_TEXT_MAP:
        parts.append(WEIGHT_TEXT_MAP[weight_pref])

    # 카테고리 (보드게임)
    if category == "boardgame":
        cat = group.get("category")
        if cat and cat in CATEGORY_TEXT_MAP:
            parts.append(CATEGORY_TEXT_MAP[cat])

        mech = group.get("mechanism")
        if mech and mech in MECHANISM_TEXT_MAP:
            parts.append(MECHANISM_TEXT_MAP[mech])

    # 머더미스터리 키워드
    if category == "murdermystery":
        parts.append("추리 크라임씬 머더미스터리")
        if group.get("horror_tolerance", 2) == 0:
            parts.append("공포없음 가벼운")

    return " ".join(parts)


def _build_query_filter(group: dict, category: str) -> dict:
    """hard_filter 조건 dict 생성."""
    query_filter = {}

    headcount = group.get("headcount")
    if headcount:
        query_filter["players"] = headcount

    play_time = group.get("play_time")
    if play_time:
        if category == "boardgame":
            query_filter["playing_time"] = play_time
        else:
            query_filter["max_time"] = play_time

    weight_pref = group.get("weight_pref")
    if weight_pref:
        query_filter["weight_pref"] = weight_pref
        if weight_pref in WEIGHT_MAX_MAP:
            query_filter["weight_max"] = WEIGHT_MAX_MAP[weight_pref]

    # 카테고리/메커니즘 (메타 가중치용)
    if category == "boardgame":
        if group.get("category"):
            query_filter["category"] = group["category"]
        if group.get("mechanism"):
            query_filter["mechanism"] = group["mechanism"]

    # 지역 (머더미스터리)
    if category == "murdermystery" and group.get("area"):
        query_filter["area"] = group["area"]

    return query_filter


def _build_emotion_tags(group: dict) -> list[str]:
    """tag_filter용 감정 태그 리스트 생성."""
    tags = []

    # 공포 수용도
    horror_tolerance = group.get("horror_tolerance", 2)
    tags.extend(HORROR_EMOTION_MAP.get(horror_tolerance, []))

    # 난이도 선호
    weight_pref = group.get("weight_pref")
    if weight_pref:
        tags.extend(WEIGHT_EMOTION_MAP.get(weight_pref, []))

    # 관계 유형
    relation = group.get("relation")
    if relation == "first_meeting":
        tags.extend(["처음만나는사이추천", "분위기좋음", "대화유도"])
    elif relation == "couple":
        tags.extend(["데이트추천", "분위기좋음"])
    elif relation == "friend":
        tags.extend(["웃음", "친목용"])
    elif relation == "coworker":
        tags.extend(["입문용", "가볍게즐길수있음"])

    return list(dict.fromkeys(tags))  # 중복 제거, 순서 유지


def _build_anchor_titles(group: dict, category: str) -> list[str]:
    """dense embedding 기준점 앵커 타이틀 리스트 생성."""
    if category == "boardgame":
        cat = group.get("category")
        if cat and cat in BOARDGAME_ANCHORS:
            return BOARDGAME_ANCHORS[cat]
        return []
    else:
        return MM_ANCHORS.get("default", [])


# -------------------------
# 공개 인터페이스
# -------------------------
def transform(
    user_text: str,
    group: dict,
    category: str,
) -> dict:
    """
    그룹 조건 + 자연어 입력 → 검색 쿼리 변환.

    Args:
        user_text : 사용자 원본 자연어 입력
                    예) "4명이서 할 보드게임"
        group     : 그룹 조건 dict
            headcount (int)         : 인원 수 (필수)
            horror_tolerance (int)  : 공포 수용도 0~2 (기본값 2)
            play_time (int)         : 최대 플레이 시간 분 단위
            weight_pref (str)       : "light" | "medium" | "heavy"
            category (str)          : 보드게임 카테고리 (보드게임만)
            mechanism (str)         : 메커니즘 (보드게임만)
            area (str)              : 지역 (머더미스터리만)
            relation (str)          : "first_meeting" | "couple" | "friend" | "coworker"
        category  : "boardgame" | "murdermystery"

    Returns:
        {
            "query_text":     str,        # BM25용
            "query_filter":   dict,       # hard_filter용
            "emotion_tags":   list[str],  # tag_filter용
            "anchor_titles":  list[str],  # dense embedding용
        }
    """
    if category not in ("boardgame", "murdermystery"):
        raise ValueError(f"알 수 없는 category: {category!r}")

    return {
        "query_text":    _build_query_text(user_text, group, category),
        "query_filter":  _build_query_filter(group, category),
        "emotion_tags":  _build_emotion_tags(group),
        "anchor_titles": _build_anchor_titles(group, category),
    }