"""
recommender/yoonha_graph.py

LangGraph 기반 RAG 추천 파이프라인.

외부 호출 예시:
    from recommender.graph import graph

    result = graph.invoke({
        "query": "4명이서 할 보드게임",
        "category": "boardgame"
    })

출력:
    {
        "answer": "추천 텍스트",
        "games": [...],
        "next_question": "..."
    }

내부 흐름:
    normalize_input
    → check_sufficiency
        ├─ 부족함 → clarify
        └─ 충분함 → query_transform → retrieve → tag_filter → generate
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Literal, TypedDict

from langgraph.graph import END, StateGraph


# ---------------------------------------------------------------------
# 직접 실행 / 패키지 import 모두 지원하기 위한 path 보정
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------
# Type 정의
# ---------------------------------------------------------------------
Category = Literal["boardgame", "murdermystery"]


class GraphInput(TypedDict, total=False):
    """
    외부에서 graph.invoke()로 넣는 입력 스펙.

    필수:
        query: 사용자 자연어 요청
        category: "boardgame" | "murdermystery"

    선택:
        group: 이미 파싱된 그룹 조건
        use_api: OpenAI API 사용 여부
    """

    query: str
    user_text: str
    category: str
    group: dict[str, Any]
    use_api: bool


class GraphOutput(TypedDict):
    """
    외부로 반환되는 최종 출력 스펙.
    """

    answer: str
    games: list[dict[str, Any]]
    next_question: str


class PipelineState(TypedDict, total=False):
    """
    LangGraph 내부 state.
    """

    # 외부 입력
    query: str
    user_text: str
    category: str
    group: dict[str, Any]
    use_api: bool

    # 조건 판단
    is_sufficient: bool
    missing_fields: list[str]

    # RAG 중간 결과
    query_text: str
    query_filter: dict[str, Any]
    emotion_tags: list[str]
    anchor_titles: list[str]
    retrieved_items: list[dict[str, Any]]
    filtered_items: list[dict[str, Any]]

    # 에러/진단
    retrieve_error: str
    generate_error: str

    # 최종 출력
    result: dict[str, Any]
    answer: str
    games: list[dict[str, Any]]
    next_question: str


# ---------------------------------------------------------------------
# 입력 파싱 유틸
# ---------------------------------------------------------------------
SUPPORTED_CATEGORIES = {"boardgame", "murdermystery"}

CATEGORY_ALIASES = {
    "boardgame": "boardgame",
    "boardgames": "boardgame",
    "board_game": "boardgame",
    "board-game": "boardgame",
    "board": "boardgame",
    "보드게임": "boardgame",
    "보드": "boardgame",
    "murdermystery": "murdermystery",
    "murder_mystery": "murdermystery",
    "murder-mystery": "murdermystery",
    "murder": "murdermystery",
    "mm": "murdermystery",
    "crime": "murdermystery",
    "crimescene": "murdermystery",
    "crime_scene": "murdermystery",
    "머더미스터리": "murdermystery",
    "머더": "murdermystery",
    "크라임씬": "murdermystery",
    "크라임": "murdermystery",
}

KOREAN_NUMBER_MAP = {
    "한": 1,
    "하나": 1,
    "둘": 2,
    "두": 2,
    "셋": 3,
    "세": 3,
    "넷": 4,
    "네": 4,
    "다섯": 5,
    "여섯": 6,
    "일곱": 7,
    "여덟": 8,
}

BOARDGAME_CATEGORY_KEYWORDS = {
    "전략": "Strategy",
    "경제": "Economic",
    "파티": "Party",
    "전쟁": "War",
    "가족": "Family",
    "패밀리": "Family",
    "추상": "Abstract",
    "협력": "Cooperative",
    "협동": "Cooperative",
    "추리": "Deduction",
    "카드": "Card Game",
    "테마": "Thematic",
}

BOARDGAME_MECHANISM_KEYWORDS = {
    "일꾼": "Worker Placement",
    "워커": "Worker Placement",
    "덱빌딩": "Deck Building",
    "덱 빌딩": "Deck Building",
    "엔진": "Engine Building",
    "엔진빌딩": "Engine Building",
    "지역장악": "Area Control",
    "영역": "Area Control",
    "마켓": "Market",
    "시장": "Market",
    "드래프팅": "Drafting",
    "드래프트": "Drafting",
    "협력형": "Cooperative Game",
}


def _normalize_category(raw_category: str | None, query: str) -> str:
    """
    외부 category 값을 내부 표준 category로 정규화.
    category가 없으면 query에서 추론한다.
    """

    value = (raw_category or "").strip().lower()

    if value in CATEGORY_ALIASES:
        return CATEGORY_ALIASES[value]

    query_lower = query.lower()

    if any(keyword in query_lower for keyword in ["머더", "크라임", "murder", "crime"]):
        return "murdermystery"

    if any(keyword in query_lower for keyword in ["보드", "board"]):
        return "boardgame"

    return value


def _extract_headcount(query: str) -> int | None:
    """
    query에서 인원 수를 추출한다.

    예:
        "4명이서" → 4
        "4인 보드게임" → 4
        "네 명이서" → 4
        "넷이서" → 4
    """

    if not query:
        return None

    digit_match = re.search(r"(\d{1,2})\s*(명|인)", query)
    if digit_match:
        return int(digit_match.group(1))

    for word, number in KOREAN_NUMBER_MAP.items():
        patterns = [
            rf"{word}\s*명",
            rf"{word}\s*인",
            rf"{word}이서",
            rf"{word}명이서",
        ]
        if any(re.search(pattern, query) for pattern in patterns):
            return number

    return None


def _extract_play_time(query: str) -> int | None:
    """
    query에서 최대 플레이 시간을 분 단위로 추출한다.

    예:
        "2시간 안에" → 120
        "90분 이하" → 90
        "1시간 반" → 90
    """

    if not query:
        return None

    hour_match = re.search(r"(\d{1,2})\s*시간", query)
    minute_match = re.search(r"(\d{1,3})\s*분", query)

    total_minutes = 0

    if hour_match:
        total_minutes += int(hour_match.group(1)) * 60

    if "반" in query and hour_match:
        total_minutes += 30

    if minute_match:
        total_minutes += int(minute_match.group(1))

    return total_minutes or None


def _extract_weight_pref(query: str) -> str | None:
    """
    보드게임 난이도 선호 추출.

    반환:
        "light" | "medium" | "heavy" | None
    """

    if not query:
        return None

    light_keywords = [
        "쉬운",
        "쉽게",
        "가벼운",
        "가볍게",
        "간단",
        "입문",
        "초보",
        "룰 쉬운",
        "룰이 쉬운",
        "부담없는",
        "부담 없는",
    ]

    medium_keywords = [
        "보통",
        "중간",
        "중급",
        "적당",
        "무난",
    ]

    heavy_keywords = [
        "어려운",
        "어렵",
        "고난도",
        "고난이도",
        "헤비",
        "무거운",
        "복잡",
        "빡센",
        "깊이있는",
        "깊이 있는",
    ]

    if any(keyword in query for keyword in light_keywords):
        return "light"

    if any(keyword in query for keyword in heavy_keywords):
        return "heavy"

    if any(keyword in query for keyword in medium_keywords):
        return "medium"

    return None


def _extract_horror_tolerance(query: str) -> int | None:
    """
    공포 수용도 추출.

    반환:
        0: 공포 불가
        1: 약간 가능
        2: 가능
    """

    if not query:
        return None

    horror_no_keywords = [
        "공포 싫",
        "공포는 싫",
        "공포 못",
        "공포 불가",
        "무서운 거 싫",
        "무서운건 싫",
        "무서운 것 싫",
        "겁 많",
        "겁많",
        "안 무서운",
        "안무서운",
        "공포 없는",
        "공포없",
    ]

    horror_low_keywords = [
        "약간 무서운",
        "살짝 무서운",
        "공포 조금",
        "조금 무서운",
    ]

    horror_ok_keywords = [
        "공포 괜찮",
        "무서운 거 괜찮",
        "호러 괜찮",
        "무서워도 괜찮",
    ]

    if any(keyword in query for keyword in horror_no_keywords):
        return 0

    if any(keyword in query for keyword in horror_low_keywords):
        return 1

    if any(keyword in query for keyword in horror_ok_keywords):
        return 2

    return None


def _extract_relation(query: str) -> str | None:
    """
    그룹 관계 유형 추출.

    반환:
        "first_meeting" | "couple" | "friend" | "coworker" | None
    """

    if not query:
        return None

    if any(keyword in query for keyword in ["처음", "첫만남", "첫 만남", "소개팅", "어색"]):
        return "first_meeting"

    if any(keyword in query for keyword in ["데이트", "커플", "연인", "남자친구", "여자친구"]):
        return "couple"

    if any(keyword in query for keyword in ["친구", "동창", "동기", "모임"]):
        return "friend"

    if any(keyword in query for keyword in ["회식", "직장", "회사", "동료", "팀빌딩", "워크샵"]):
        return "coworker"

    return None


def _extract_boardgame_category(query: str) -> str | None:
    """
    보드게임 BGG category 추출.
    """

    if not query:
        return None

    for keyword, category in BOARDGAME_CATEGORY_KEYWORDS.items():
        if keyword in query:
            return category

    return None


def _extract_boardgame_mechanism(query: str) -> str | None:
    """
    보드게임 mechanism 추출.
    """

    if not query:
        return None

    for keyword, mechanism in BOARDGAME_MECHANISM_KEYWORDS.items():
        if keyword in query:
            return mechanism

    return None


def _merge_group_from_query(
    query: str,
    category: str,
    group: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    외부 group과 query에서 파싱한 조건을 병합한다.
    명시적으로 들어온 group 값을 우선한다.
    """

    merged: dict[str, Any] = dict(group or {})

    if "headcount" not in merged:
        headcount = _extract_headcount(query)
        if headcount is not None:
            merged["headcount"] = headcount

    if "play_time" not in merged:
        play_time = _extract_play_time(query)
        if play_time is not None:
            merged["play_time"] = play_time

    if "weight_pref" not in merged:
        weight_pref = _extract_weight_pref(query)
        if weight_pref is not None:
            merged["weight_pref"] = weight_pref

    if "horror_tolerance" not in merged:
        horror_tolerance = _extract_horror_tolerance(query)
        if horror_tolerance is not None:
            merged["horror_tolerance"] = horror_tolerance

    if "relation" not in merged:
        relation = _extract_relation(query)
        if relation is not None:
            merged["relation"] = relation

    if category == "boardgame":
        if "category" not in merged:
            boardgame_category = _extract_boardgame_category(query)
            if boardgame_category is not None:
                merged["category"] = boardgame_category

        if "mechanism" not in merged:
            mechanism = _extract_boardgame_mechanism(query)
            if mechanism is not None:
                merged["mechanism"] = mechanism

    return merged


def _build_next_question(
    category: str,
    group: dict[str, Any],
    missing_fields: list[str] | None = None,
) -> str:
    """
    부족한 조건에 따라 다음 역질문을 생성한다.
    """

    missing_fields = missing_fields or []

    if "query" in missing_fields:
        return "어떤 활동을 찾고 계신지 알려주세요. 예: 4명이서 할 보드게임"

    if "category" in missing_fields:
        return "보드게임과 머더미스터리 중 어떤 활동을 추천받고 싶으신가요?"

    if "headcount" in missing_fields or not group.get("headcount"):
        return "몇 명이서 함께할 예정인가요?"

    if category == "boardgame":
        if not group.get("weight_pref"):
            return "게임 난이도는 어느 정도가 좋으세요? 가벼운 입문용, 보통, 어려운 전략 게임 중에서 골라주세요."

        if not group.get("play_time"):
            return "플레이 시간은 어느 정도가 적당한가요? 예: 30분, 1시간, 2시간"

        if not group.get("relation"):
            return "함께하는 분들과의 관계가 어떻게 되나요? 친구, 연인, 직장동료, 첫 만남 중에 알려주세요."

    if category == "murdermystery":
        if group.get("horror_tolerance") is None:
            return "공포 요소는 괜찮으신가요? 공포 불가, 약간 가능, 괜찮음 중에서 알려주세요."

        if not group.get("play_time"):
            return "플레이 시간은 어느 정도가 적당한가요? 예: 2시간, 3시간"

        if not group.get("relation"):
            return "함께하는 분들과의 관계가 어떻게 되나요? 친구, 연인, 직장동료, 첫 만남 중에 알려주세요."

    return "추가로 피하고 싶은 요소나 선호하는 분위기가 있나요?"


def _normalize_result(result: dict[str, Any] | None) -> dict[str, Any]:
    """
    어떤 generator 결과가 오더라도 최종 출력 스펙으로 맞춘다.
    """

    result = result or {}

    return {
        "answer": result.get("answer", "") or "",
        "games": result.get("games", []) or [],
        "next_question": result.get("next_question", "") or "",
    }


# ---------------------------------------------------------------------
# 룰 기반 generator fallback
# ---------------------------------------------------------------------
def _generate_without_api_local(
    items: list[dict[str, Any]],
    group: dict[str, Any],
    category: str,
    emotion_tags: list[str] | None = None,
    max_items: int = 5,
    retrieve_error: str = "",
) -> dict[str, Any]:
    """
    OpenAI API 없이 동작하는 로컬 fallback generator.
    yoonha_generator.py import가 실패해도 graph.invoke()가 깨지지 않도록 한다.
    """

    emotion_tags = emotion_tags or []
    games: list[dict[str, Any]] = []

    if not items:
        if retrieve_error:
            answer = (
                "검색 데이터 또는 FAISS 인덱스가 현재 실행 환경에 연결되어 있지 않아 "
                "추천 후보를 조회하지 못했습니다."
            )
        else:
            answer = "조건에 맞는 추천 결과를 찾지 못했습니다."

        return {
            "answer": answer,
            "games": [],
            "next_question": _build_next_question(category, group),
        }

    for item in items[:max_items]:
        title = item.get("title") or item.get("name") or "제목 없음"
        reasons: list[str] = []

        headcount = group.get("headcount")
        min_players = item.get("min_players")
        max_players = item.get("max_players")

        if headcount and isinstance(min_players, (int, float)) and isinstance(max_players, (int, float)):
            reasons.append(f"{headcount}명이 플레이하기에 적합한 인원 범위입니다.")

        rating = item.get("avg_rating") or item.get("rating")
        if isinstance(rating, (int, float)):
            reasons.append(f"평점 지표가 {rating}로 확인됩니다.")

        weight_pref = group.get("weight_pref")
        weight = item.get("weight")
        if category == "boardgame" and isinstance(weight, (int, float)) and weight_pref:
            if weight_pref == "light" and weight < 2.5:
                reasons.append("입문자도 부담 없이 시작할 수 있는 난이도입니다.")
            elif weight_pref == "medium" and 2.5 <= weight <= 3.5:
                reasons.append("너무 가볍지도 무겁지도 않은 중간 난이도입니다.")
            elif weight_pref == "heavy" and weight > 3.5:
                reasons.append("전략적 깊이가 있는 고난도 게임입니다.")

        item_tags = set(item.get("emotion_tags", []) or [])
        query_tags = set(emotion_tags)
        matched_tags = sorted(item_tags & query_tags)

        if matched_tags:
            reasons.append(f"{', '.join(matched_tags)} 태그가 그룹 조건과 맞습니다.")

        if not reasons:
            reasons.append("검색 조건과 메타데이터 기준으로 상위에 노출된 추천 후보입니다.")

        games.append(
            {
                "title": title,
                "reason": " ".join(reasons),
                "matched_tags": matched_tags,
                "final_score": item.get("final_score") or item.get("total_score"),
                "emotion_tags": item.get("emotion_tags", []) or [],
                "source": item.get("source"),
                "avg_rating": item.get("avg_rating") or item.get("rating"),
                "min_players": item.get("min_players"),
                "max_players": item.get("max_players"),
                "image": item.get("image"),
            }
        )

    headcount_text = f"{group.get('headcount')}명" if group.get("headcount") else "요청하신 조건"
    category_label = "보드게임" if category == "boardgame" else "머더미스터리"
    top_title = games[0]["title"] if games else ""

    answer = (
        f"{headcount_text} 그룹에 맞는 {category_label} 추천 결과입니다. "
        f"가장 우선 추천할 후보는 '{top_title}'입니다. "
        "인원 조건, 난이도, 태그 매칭, 평점 정보를 함께 고려했습니다."
    )

    return {
        "answer": answer,
        "games": games,
        "next_question": _build_next_question(category, group),
    }


# ---------------------------------------------------------------------
# LangGraph Node 함수
# ---------------------------------------------------------------------
def node_normalize_input(state: PipelineState) -> dict[str, Any]:
    """
    외부 입력 query/category를 내부 파이프라인 state로 변환한다.
    """

    query = state.get("query") or state.get("user_text") or ""
    query = str(query).strip()

    raw_category = state.get("category")
    category = _normalize_category(raw_category, query)

    group = _merge_group_from_query(
        query=query,
        category=category,
        group=state.get("group") or {},
    )

    return {
        "query": query,
        "user_text": query,
        "category": category,
        "group": group,
        "use_api": bool(state.get("use_api", True)),
        "query_text": "",
        "query_filter": {},
        "emotion_tags": [],
        "anchor_titles": [],
        "retrieved_items": [],
        "filtered_items": [],
        "retrieve_error": "",
        "generate_error": "",
        "result": {},
        "answer": "",
        "games": [],
        "next_question": "",
    }


def node_check_sufficiency(state: PipelineState) -> dict[str, Any]:
    """
    RAG 검색을 실행하기 위한 최소 조건이 충분한지 판단한다.

    현재 최소 조건:
        - query 존재
        - category가 boardgame 또는 murdermystery
        - headcount 존재

    나머지 조건은 recommendation quality를 높이는 선택 조건으로 보고,
    검색은 진행하되 generator에서 next_question으로 보완한다.
    """

    missing_fields: list[str] = []

    query = state.get("user_text") or state.get("query") or ""
    category = state.get("category") or ""
    group = state.get("group") or {}

    if not query:
        missing_fields.append("query")

    if category not in SUPPORTED_CATEGORIES:
        missing_fields.append("category")

    if not group.get("headcount"):
        missing_fields.append("headcount")

    return {
        "is_sufficient": len(missing_fields) == 0,
        "missing_fields": missing_fields,
    }


def route_after_sufficiency(state: PipelineState) -> str:
    """
    조건 충분 여부에 따라 다음 노드 결정.
    """

    if state.get("is_sufficient"):
        return "query_transform"

    return "clarify"


def node_clarify(state: PipelineState) -> dict[str, Any]:
    """
    조건이 부족한 경우 RAG 검색 없이 역질문을 반환한다.
    """

    category = state.get("category") or ""
    group = state.get("group") or {}
    missing_fields = state.get("missing_fields") or []

    next_question = _build_next_question(
        category=category,
        group=group,
        missing_fields=missing_fields,
    )

    result = {
        "answer": "추천을 정확히 하기 위해 조건이 조금 더 필요합니다.",
        "games": [],
        "next_question": next_question,
    }

    return {
        "result": result,
        "answer": result["answer"],
        "games": result["games"],
        "next_question": result["next_question"],
    }


def node_query_transform(state: PipelineState) -> dict[str, Any]:
    """
    그룹 조건 + 자연어 요청을 BM25/FAISS 검색용 쿼리로 변환한다.
    """

    from recommender.rag.yoonha_query_transformer import transform as query_transform

    transformed = query_transform(
        user_text=state.get("user_text", ""),
        group=state.get("group", {}),
        category=state.get("category", "boardgame"),
    )

    return {
        "query_text": transformed.get("query_text", ""),
        "query_filter": transformed.get("query_filter", {}),
        "emotion_tags": transformed.get("emotion_tags", []),
        "anchor_titles": transformed.get("anchor_titles", []),
    }


def node_retrieve(state: PipelineState) -> dict[str, Any]:
    """
    BM25 + FAISS 하이브리드 검색을 실행한다.

    데이터/FAISS index가 로컬에 없으면 예외를 잡고 빈 결과를 반환한다.
    실제 제출/실행 환경에서 data가 연결되어 있으면 정상 검색된다.
    """

    try:
        from recommender.rag.yoonha_hybrid_retriever import get_embedding, retrieve

        query_vector = get_embedding(
            state.get("anchor_titles", []),
            state.get("category", "boardgame"),
        )

        items = retrieve(
            query_text=state.get("query_text", ""),
            query_filter=state.get("query_filter", {}),
            query_vector=query_vector,
            category=state.get("category", "boardgame"),
            topk=50,
        )

        return {
            "retrieved_items": items,
            "retrieve_error": "",
        }

    except Exception as exc:
        return {
            "retrieved_items": [],
            "retrieve_error": str(exc),
        }


def node_tag_filter(state: PipelineState) -> dict[str, Any]:
    """
    감정 태그 기반 필터링 및 점수 조정.
    """

    items = state.get("retrieved_items", []) or []
    if not items:
        return {"filtered_items": []}

    from recommender.rag.yoonha_tag_filter import filter_and_score

    group = state.get("group", {})
    horror_tolerance = group.get("horror_tolerance", 2)

    filtered = filter_and_score(
        items=items,
        emotion_tags=state.get("emotion_tags", []),
        horror_tolerance=horror_tolerance,
    )

    return {"filtered_items": filtered}


def node_generate(state: PipelineState) -> dict[str, Any]:
    """
    추천 텍스트, 추천 게임 리스트, 역질문을 생성한다.

    use_api=True:
        yoonha_generator.generate 사용 시도.
        실패하면 로컬 fallback 사용.

    use_api=False:
        로컬 fallback 사용.
    """

    items = state.get("filtered_items", []) or []
    group = state.get("group", {}) or {}
    category = state.get("category", "boardgame")
    emotion_tags = state.get("emotion_tags", []) or []
    retrieve_error = state.get("retrieve_error", "")

    use_api = bool(state.get("use_api", True))

    if use_api:
        try:
            from recommender.rag.yoonha_generator import generate

            result = generate(
                items=items,
                group=group,
                category=category,
                emotion_tags=emotion_tags,
            )

        except Exception as exc:
            result = _generate_without_api_local(
                items=items,
                group=group,
                category=category,
                emotion_tags=emotion_tags,
                retrieve_error=retrieve_error,
            )
            result["answer"] = result["answer"] + f" API 생성은 실패하여 룰 기반 결과로 대체했습니다."

            return {
                "result": _normalize_result(result),
                "answer": result.get("answer", ""),
                "games": result.get("games", []),
                "next_question": result.get("next_question", ""),
                "generate_error": str(exc),
            }

    else:
        result = _generate_without_api_local(
            items=items,
            group=group,
            category=category,
            emotion_tags=emotion_tags,
            retrieve_error=retrieve_error,
        )

    formatted = _normalize_result(result)

    return {
        "result": formatted,
        "answer": formatted["answer"],
        "games": formatted["games"],
        "next_question": formatted["next_question"],
        "generate_error": "",
    }


# ---------------------------------------------------------------------
# Graph build
# ---------------------------------------------------------------------
def build_graph():
    """
    LangGraph 노드·엣지 구성 후 compiled graph 반환.
    """

    workflow = StateGraph(
        PipelineState,
        input_schema=GraphInput,
        output_schema=GraphOutput,
    )

    workflow.add_node("normalize_input", node_normalize_input)
    workflow.add_node("check_sufficiency", node_check_sufficiency)
    workflow.add_node("clarify", node_clarify)
    workflow.add_node("query_transform", node_query_transform)
    workflow.add_node("retrieve", node_retrieve)
    workflow.add_node("tag_filter", node_tag_filter)
    workflow.add_node("generate", node_generate)

    workflow.set_entry_point("normalize_input")

    workflow.add_edge("normalize_input", "check_sufficiency")

    workflow.add_conditional_edges(
        "check_sufficiency",
        route_after_sufficiency,
        {
            "clarify": "clarify",
            "query_transform": "query_transform",
        },
    )

    workflow.add_edge("clarify", END)

    workflow.add_edge("query_transform", "retrieve")
    workflow.add_edge("retrieve", "tag_filter")
    workflow.add_edge("tag_filter", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()


# ---------------------------------------------------------------------
# 공개 인터페이스
# ---------------------------------------------------------------------
graph = build_graph()


def run_pipeline(
    user_text: str,
    group: dict[str, Any] | None = None,
    category: str = "boardgame",
    use_api: bool = True,
) -> dict[str, Any]:
    """
    기존 yoonha_graph.py 방식과의 호환용 함수.

    기존 호출:
        run_pipeline(
            user_text="4명이서 할 전략 보드게임 추천해줘",
            group={"headcount": 4},
            category="boardgame",
            use_api=False,
        )

    신규 호출:
        graph.invoke({
            "query": "4명이서 할 보드게임",
            "category": "boardgame"
        })
    """

    return graph.invoke(
        {
            "query": user_text,
            "category": category,
            "group": group or {},
            "use_api": use_api,
        }
    )


# ---------------------------------------------------------------------
# 로컬 테스트
# ---------------------------------------------------------------------
if __name__ == "__main__":
    result = graph.invoke(
        {
            "query": "4명이서 할 보드게임",
            "category": "boardgame",
        }
    )

    print(result)