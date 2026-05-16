"""
yoonha_yu_eval_pipeline.py

Nolit 용국 RAG 파이프라인 통합 평가 스크립트 (yoonha_eval_pipeline 기준 동일 구조)

평가 항목:
    1. extract_query_conditions 조건 추출 확인
    2. retriever 검색 결과 비교
       - RRF (keyword + condition 융합)
       - Keyword-only
       - Condition-only
    3. Title Ground Truth Precision@5
       - 미리 지정한 정답 제목이 top5에 포함되는지 확인
    4. Condition@5
       - top5 결과가 query의 hard condition을 만족하는지 확인
       - 방탈출(bbabang): 지역 + 인원 + 가격 + 시간
       - 통합(total): 카테고리 + 인원
    5. emotion_tag (감정 태그) 필터링 확인
    6. prompt 생성 확인 (추천 이유 + 역질문)
    7. 종합 평가표
"""

from __future__ import annotations

import sys
import time
import json
import re
from pathlib import Path


# =========================================================
# Import path 설정
# - python -m recommender.eval.yonguk_eval_pipeline
# - python recommender/eval/yonguk_eval_pipeline.py
# 둘 다 대응
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAG_DIR = PROJECT_ROOT / "recommender" / "rag"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(RAG_DIR))


from recommender.rag.yonguk_bbabang_emo_tag import sentiment_tag_bonus
from recommender.rag.yonguk_bbabang_prompts import (
    generate_recommend_reason,
    generate_followup_question,
)
from recommender.rag.yonguk_total_emo_tag import (
    tag_item,
    filter_and_score as total_filter_and_score,
    is_horror_blocked,
)
from recommender.rag.yonguk_total_prompts import (
    generate_question_style_followups,
    print_question_style_followups,
)

# yonguk_eval_bbabang에서 사용 가능한 함수들 직접 임포트
# faiss / sentence_transformers가 없을 수 있으므로 조건부 임포트
_BBABANG_AVAILABLE = False
try:
    from recommender.eval.yonguk_eval_bbabang import (
        extract_query_conditions,
        passes_hard_filter,
        build_bm25,
        bm25_search,
        make_doc_text,
        rrf_fusion,
        normalize_score,
        first_weight_bonus,
        second_weight_rating_bonus,
        second_weight_rank_bonus,
        add_weight_bonus,
        filter_ranked_docs,
        build_stats_lookup,
        get_stats_item,
    )
    _BBABANG_AVAILABLE = True
except ImportError as e:
    print(f"  ⚠️ bbabang 모듈 임포트 실패 (faiss/sentence_transformers 필요): {e}")

    # extract_query_conditions만 로컬 대체 구현
    def extract_query_conditions(query):
        """bbabang 모듈 없을 때의 경량 대체 구현"""
        filters = {}
        prefs = {}

        if "서울" in query:
            filters["area"] = "서울"
        elif "경기" in query:
            filters["area"] = "경기"
        elif "강원" in query:
            filters["area"] = "강원"

        if "원주" in query:
            filters["location"] = "원주시"
        elif "강릉" in query:
            filters["location"] = "강릉시"
        elif "강남구" in query:
            filters["location"] = "강남구"
        elif "마포구" in query:
            filters["location"] = "마포구"
        elif "부천" in query:
            filters["location"] = "부천시"
        elif "수원" in query:
            filters["location"] = "수원시"
        elif "인천" in query:
            filters["location"] = "인천"

        m = re.search(r"(\d+)\s*(명|인)", query)
        if m:
            filters["max_players"] = int(m.group(1))

        m = re.search(r"(\d+)\s*만\s*원\s*이하", query)
        if m:
            filters["price"] = int(m.group(1)) * 10000

        m = re.search(r"(\d+)\s*원\s*이하", query)
        if m:
            filters["price"] = int(m.group(1))

        m = re.search(r"(\d+)\s*시간\s*이내", query)
        if m:
            filters["playing_time"] = int(m.group(1)) * 60

        m = re.search(r"(\d+)\s*분\s*이내", query)
        if m:
            filters["playing_time"] = int(m.group(1))

        if any(w in query for w in ["안 무서운", "안무서운", "무섭지 않은", "공포 없는", "쫄보"]):
            prefs["horror"] = "low"
        elif any(w in query for w in ["무서운", "공포", "호러"]):
            prefs["horror"] = "high"

        if any(w in query for w in ["쉬운", "입문", "초보", "방린이"]):
            prefs["difficulty"] = "low"
        elif any(w in query for w in ["어려운", "고난도", "하드", "빡센"]):
            prefs["difficulty"] = "high"

        if any(w in query for w in ["퍼즐", "문제"]):
            prefs["puzzle"] = True
        if any(w in query for w in ["스토리", "서사", "몰입"]):
            prefs["story"] = True
        if any(w in query for w in ["인테리어", "예쁜", "잘 꾸민"]):
            prefs["interior"] = True
        if any(w in query for w in ["연출", "장치", "퀄리티"]):
            prefs["production"] = True

        prefs["satisfaction"] = True
        return filters, prefs


# yonguk_eval_total에서 사용 가능한 함수들 임포트
# 이 모듈은 하드코딩된 Windows 경로를 사용하므로, 데이터 파일이 없으면
# 모듈 레벨 코드에서 에러가 날 수 있다. 함수만 개별 임포트한다.
_TOTAL_AVAILABLE = False
try:
    from recommender.eval.yonguk_eval_total import (
        simple_keyword_score,
        first_weight_score,
        second_weight_score,
        condition_score,
        keyword_retrieve,
        condition_retrieve,
        rrf_fusion as total_rrf_fusion,
        grade_score,
        get_title,
        get_number,
        extract_player_count,
        CONDITION_KEYWORDS,
    )
    _TOTAL_AVAILABLE = True
except Exception as e:
    print(f"  ⚠️ total 모듈 임포트 실패: {e}")
    print("  → total 관련 함수를 로컬에 재정의합니다.")

    # total eval의 핵심 함수들을 로컬에 재정의
    CONDITION_KEYWORDS = {
        "boardgame": ["보드게임", "보드", "게임", "board game", "boardgame", "tabletop"],
        "escape": ["방탈출", "테마", "탈출", "escape room", "escape", "theme"],
        "murder": ["머더", "머더미스터리", "미스터리", "추리", "murder", "mystery", "detective", "deduction"],
        "party": ["파티", "단체", "여럿", "웃긴", "친구", "모임", "party", "social", "group", "multiplayer", "casual", "family", "friends", "humor"],
        "easy": ["쉬운", "쉽", "입문", "초보", "간단", "easy", "simple", "beginner", "gateway", "light", "casual"],
        "popular": ["인기", "추천", "유명", "베스트", "popular", "recommended", "top rated", "best", "famous", "classic"],
        "many_players": ["단체", "여럿", "다인원", "6명", "7명", "8명", "6 players", "7 players", "8 players", "large group", "multiplayer", "group play", "many players"],
        "player_7": ["7명", "7인", "최대 7", "최대7", "7 players", "supports 7", "up to 7", "7 player"],
        "fun": ["재밌", "존잼", "꿀잼", "웃긴", "만족", "fun", "exciting", "enjoyable", "entertaining", "hilarious"],
        "story": ["스토리", "서사", "몰입", "연출", "story", "narrative", "theme", "immersive", "atmosphere"],
        "device": ["장치", "인테리어", "연출", "device", "mechanism", "production", "special effects"],
        "mystery": ["추리", "범인", "사건", "미스터리", "mystery", "deduction", "detective", "crime", "culprit"],
        "immersion": ["몰입", "분위기", "역할", "캐릭터", "immersion", "immersive", "roleplay", "character", "atmosphere"],
    }

    def get_title(item):
        for key in ["title", "name", "game_name", "theme_name", "primary_name", "kor_title", "eng_title"]:
            if key in item and item[key]:
                return str(item[key])
        return str(item)

    def get_number(item, keys, default=None):
        for key in keys:
            if key in item and item[key] not in [None, ""]:
                try:
                    return float(item[key])
                except Exception:
                    pass
        return default

    def extract_player_count(query):
        for pattern in [r"(\d+)\s*명", r"(\d+)\s*인", r"인원\s*(\d+)"]:
            match = re.search(pattern, query)
            if match:
                return int(match.group(1))
        return None

    def count_keywords(text, keywords):
        return sum(1 for kw in keywords if kw.lower() in text)

    def simple_keyword_score(query, item):
        text = json.dumps(item, ensure_ascii=False).lower()
        score = 0
        if "보드게임" in query:
            score += 3
        if "방탈출" in query:
            score += count_keywords(text, CONDITION_KEYWORDS["escape"]) * 3
        if "머더" in query or "머더미스터리" in query:
            score += count_keywords(text, CONDITION_KEYWORDS["murder"]) * 3
        if "파티" in query:
            score += count_keywords(text, CONDITION_KEYWORDS["party"]) * 4
        if "재밌" in query or "추천" in query:
            score += count_keywords(text, CONDITION_KEYWORDS["fun"]) * 2
        return score

    def first_weight_score(query, item, category):
        score = 0
        text = json.dumps(item, ensure_ascii=False).lower()
        min_players = get_number(item, ["min_players", "minplayers", "min_player", "minimum_players"])
        max_players = get_number(item, ["max_players", "maxplayers", "max_player", "maximum_players"])
        rating = get_number(item, ["avg_rating", "average_rating", "rating", "satisfaction", "bayesaverage", "bayes_average"])
        weight = get_number(item, ["weight", "average_weight", "complexity", "avg_weight"])
        difficulty = get_number(item, ["difficulty"])
        horror = get_number(item, ["horror"])
        query_player = extract_player_count(query)

        if query_player:
            if min_players and max_players:
                if min_players <= query_player <= max_players:
                    score += 20
                else:
                    score -= 100
            elif max_players:
                if query_player <= max_players:
                    score += 12
                else:
                    score -= 100

        if category == "boardgame":
            if "파티" in query:
                score += count_keywords(text, CONDITION_KEYWORDS["party"]) * 8
            if rating:
                score += rating * 2
            if weight:
                if any(k in query for k in ["쉬운", "입문", "초보", "간단"]):
                    score += max(0, 5 - weight) * 4
                if any(k in query for k in ["전략", "어려운", "헤비", "깊은"]):
                    score += weight * 4
        elif category == "escape":
            if horror is not None:
                if any(k in query for k in ["안 무서운", "공포 없음", "무섭지 않은"]):
                    score += max(0, 5 - horror) * 5
                if any(k in query for k in ["무서운", "공포", "호러"]):
                    score += horror * 5
            satisfaction = get_number(item, ["satisfaction"])
            if satisfaction is not None:
                score += satisfaction * 4
        elif category == "murder":
            if rating:
                score += rating * 5

        return score

    def second_weight_score(query, item, category):
        score = 0
        rating = get_number(item, ["avg_rating", "average_rating", "rating", "bayesaverage", "bayes_average", "satisfaction"])
        category_rank = get_number(item, ["category_rank", "overall_rank", "rank"])
        review_count = get_number(item, ["review_count", "num_reviews", "users_rated", "rating_count", "voters"])

        if rating is not None:
            if category == "boardgame":
                if rating >= 8: score += 20
                elif rating >= 7.5: score += 15
                elif rating >= 7: score += 10
                elif rating >= 6.5: score += 5
            elif category in ["escape", "murder"]:
                if rating >= 4.8: score += 20
                elif rating >= 4.5: score += 15
                elif rating >= 4.0: score += 10
                elif rating >= 3.5: score += 5

        if category_rank is not None:
            if category_rank <= 10: score += 25
            elif category_rank <= 50: score += 20
            elif category_rank <= 100: score += 15
            elif category_rank <= 300: score += 10
            elif category_rank <= 1000: score += 5

        if review_count is not None:
            if review_count >= 10000: score += 15
            elif review_count >= 3000: score += 12
            elif review_count >= 1000: score += 10
            elif review_count >= 300: score += 6
            elif review_count >= 100: score += 3

        return score

    def condition_score(item, conditions):
        text = json.dumps(item, ensure_ascii=False).lower()
        total_possible = sum(conditions.values())
        earned = 0
        detail = {}

        min_players = get_number(item, ["min_players", "minplayers", "min_player", "minimum_players"])
        max_players = get_number(item, ["max_players", "maxplayers", "max_player", "maximum_players"])
        weight_value = get_number(item, ["weight", "average_weight", "complexity", "avg_weight"])
        rating = get_number(item, ["rating", "average_rating", "avg_rating", "bayesaverage", "bayes_average"])

        for condition, weight in conditions.items():
            keywords = CONDITION_KEYWORDS.get(condition, [])
            matched = [kw for kw in keywords if kw.lower() in text]

            if condition == "boardgame":
                matched.append("category=boardgame")
            if condition == "popular" and rating and rating >= 7:
                matched.append(f"rating {rating}")
            if condition == "player_7":
                if min_players and max_players and min_players <= 7 <= max_players:
                    matched.append(f"player range {int(min_players)}-{int(max_players)}")
                elif max_players and max_players >= 7:
                    matched.append(f"max_players {int(max_players)}")
            if condition == "many_players" and max_players and max_players >= 6:
                matched.append(f"max_players {int(max_players)}")
            if condition == "easy" and weight_value and weight_value <= 2.5:
                matched.append(f"weight {weight_value}")

            if matched:
                earned += weight
                detail[condition] = {"score": weight, "matched": matched}
            else:
                detail[condition] = {"score": 0, "matched": []}

        if total_possible == 0:
            return 0, detail
        return earned / total_possible, detail

    def keyword_retrieve(query, data, category, top_k=50):
        scored = []
        for item in data:
            base = simple_keyword_score(query, item)
            first = first_weight_score(query, item, category)
            second = second_weight_score(query, item, category)
            final = base + first + second
            if final <= -50:
                continue
            scored.append({
                "title": get_title(item),
                "score": final,
                "base_score": base,
                "first_weight_score": first,
                "second_weight_score": second,
                "item": item,
            })
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def condition_retrieve(data, conditions, top_k=50):
        scored = []
        for item in data:
            score, detail = condition_score(item, conditions)
            scored.append({"title": get_title(item), "score": score, "detail": detail, "item": item})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def _rrf_score(rank, k=60):
        return 1 / (k + rank)

    def total_rrf_fusion(result_lists, k=60, top_k=5):
        fused = {}
        for result_list in result_lists:
            for rank, result in enumerate(result_list, start=1):
                title = result["title"]
                if title not in fused:
                    fused[title] = {"title": title, "rrf_score": 0, "item": result["item"], "debug": []}
                fused[title]["rrf_score"] += _rrf_score(rank, k)
                fused[title]["debug"].append({
                    "rank": rank, "score": result.get("score", 0),
                    "base_score": result.get("base_score"),
                    "first_weight_score": result.get("first_weight_score"),
                    "second_weight_score": result.get("second_weight_score"),
                })
        final = list(fused.values())
        final.sort(key=lambda x: x["rrf_score"], reverse=True)
        return final[:top_k]

    def grade_score(score):
        if score >= 0.8: return "매우 적합"
        elif score >= 0.6: return "적합"
        elif score >= 0.4: return "보통"
        elif score >= 0.2: return "낮음"
        else: return "부적합"


# =========================================================
# 평가 쿼리 정의 — 방탈출(bbabang)
# =========================================================

BBABANG_QUERIES = [
    {
        "name": "원주 스토리 좋은 방탈출",
        "user_text": "원주에서 스토리 좋고 만족도 높은 방탈출 추천",
        "expected_filters": {"area": None, "location": "원주시"},
        "expected_prefs": {"story": True, "satisfaction": True},
        "ground_truth": [
            "경성",
        ],
    },
    {
        "name": "원주 3인 스토리 방탈출",
        "user_text": "원주에서 3명이 할 수 있는 스토리 좋고 만족도 높은 방탈출 추천",
        "expected_filters": {"location": "원주시", "max_players": 3},
        "expected_prefs": {"story": True, "satisfaction": True},
        "ground_truth": [],
    },
    {
        "name": "원주 2인 안무서운 입문용",
        "user_text": "원주에서 2명이 할 수 있는 안 무서운 입문용 방탈출 추천",
        "expected_filters": {"location": "원주시", "max_players": 2},
        "expected_prefs": {"horror": "low", "difficulty": "low", "satisfaction": True},
        "ground_truth": [],
    },
    {
        "name": "강릉 커플 인테리어",
        "user_text": "강릉에서 커플이 하기 좋은 인테리어 예쁜 방탈출 추천",
        "expected_filters": {"location": "강릉시"},
        "expected_prefs": {"interior": True, "satisfaction": True},
        "ground_truth": [],
    },
    {
        "name": "원주 고난도 퍼즐",
        "user_text": "원주에서 퍼즐 잘 만들고 어려운 고난도 방탈출 추천",
        "expected_filters": {"location": "원주시"},
        "expected_prefs": {"puzzle": True, "difficulty": "high", "satisfaction": True},
        "ground_truth": [],
    },
]


# =========================================================
# 평가 쿼리 정의 — 통합(total: 보드게임/방탈출/머더미스터리)
# =========================================================

TOTAL_QUERIES = [
    {
        "name": "보드게임 일반 추천",
        "user_text": "보드게임 하나 추천해줘",
        "category": "boardgame",
        "conditions": {
            "boardgame": 2,
            "popular": 2,
            "fun": 1,
        },
        "ground_truth": [],
    },
    {
        "name": "재밌는 방탈출 추천",
        "user_text": "재밌는 방탈출 하나 추천해줘",
        "category": "escape",
        "conditions": {
            "escape": 2,
            "fun": 2,
            "story": 1,
            "device": 1,
        },
        "ground_truth": [],
    },
    {
        "name": "머더미스터리 일반 추천",
        "user_text": "머더미스터리 하나 추천해줘",
        "category": "murder",
        "conditions": {
            "murder": 2,
            "mystery": 2,
            "story": 1,
            "immersion": 1,
        },
        "ground_truth": [],
    },
    {
        "name": "파티게임 추천",
        "user_text": "나 요새 그냥 파티게임 하나 해보고 싶어",
        "category": "boardgame",
        "conditions": {
            "party": 3,
            "many_players": 2,
            "easy": 1,
            "fun": 1,
        },
        "ground_truth": [
            "Codenames",
            "코드네임",
            "Dixit",
            "딕싯",
        ],
    },
    {
        "name": "7인 보드게임",
        "user_text": "인원 7명 정도 되는데 뭐가 좋을까?",
        "category": "boardgame",
        "conditions": {
            "player_7": 3,
            "many_players": 2,
            "party": 2,
            "easy": 1,
        },
        "ground_truth": [],
    },
]


# =========================================================
# 기본 출력 유틸
# =========================================================

def print_header(text: str):
    print(f"\n{'=' * 70}")
    print(f"  {text}")
    print(f"{'=' * 70}")


def print_subheader(text: str):
    print(f"\n--- {text} ---")


def print_items(items, max_show=5, mode="total"):
    """
    mode="total" → total eval 결과 출력 (title, rrf_score 등)
    mode="bbabang" → bbabang eval 결과 출력 (doc_id, final_score 등)
    """
    for i, item in enumerate(items[:max_show], 1):
        if mode == "bbabang":
            doc_id, final_score, rrf_score, *_ = item
            print(f"  {i}. doc_id={doc_id} (final={final_score:.4f}, rrf={rrf_score:.4f})")
        else:
            title = item.get("title", "?")
            rrf = item.get("rrf_score", item.get("score", "?"))
            if isinstance(rrf, float):
                rrf = f"{rrf:.6f}"
            print(f"  {i}. {title} (RRF: {rrf})")


# =========================================================
# 평가 유틸: Title Ground Truth Precision@K
# =========================================================

def precision_at_k(items, ground_truth, k=5, title_key="title"):
    """
    Title Ground Truth Precision@K

    ground_truth에 정의된 정답 제목이 top-k 결과에 포함되는지 확인한다.
    ground_truth가 비어 있으면 None을 반환하고, 출력에서는 N/A로 표시한다.
    """
    if not ground_truth:
        return None

    pred_titles = []
    for item in items[:k]:
        if isinstance(item, dict):
            title = item.get("title", item.get("name", ""))
        elif isinstance(item, (list, tuple)):
            # bbabang 결과: (doc_id, final_score, ...)
            title = str(item[0])
        else:
            title = str(item)
        pred_titles.append(title)

    hits = 0

    for gt in ground_truth:
        for title in pred_titles:
            if gt and gt in title:
                hits += 1
                break

    return hits / k


# =========================================================
# 평가 유틸: Condition@K (bbabang 방탈출용)
# =========================================================

def _num(value, default=None):
    """숫자/문자열/NaN 값을 안전하게 숫자로 변환한다."""
    if value is None:
        return default

    if isinstance(value, (int, float)):
        try:
            if value != value:  # NaN
                return default
        except Exception:
            return default
        return float(value)

    try:
        text = str(value).strip().replace(",", "")
        if not text or text.lower() in {"none", "null", "nan", "na", "n/a", "-", "?"}:
            return default
        m = re.search(r"-?\d+(?:\.\d+)?", text)
        if not m:
            return default
        num = float(m.group(0))
        if num != num:
            return default
        return num
    except Exception:
        return default


def bbabang_condition_pass(item, filters):
    """
    방탈출(bbabang) Condition Pass 여부.

    Hard filter 조건만 평가한다: 지역, 인원, 가격, 플레이시간
    """
    if not filters:
        return True

    # 지역 조건
    if filters.get("area"):
        if item.get("area") != filters["area"]:
            return False

    if filters.get("location"):
        if item.get("location") != filters["location"]:
            return False

    # 인원 조건
    if filters.get("max_players"):
        max_players = _num(item.get("max_players"), None)
        if max_players is None:
            return False
        if filters["max_players"] > max_players:
            return False

    # 가격 조건
    if filters.get("price"):
        price = _num(item.get("price"), None)
        if price is None:
            return False
        if price > filters["price"]:
            return False

    # 시간 조건
    if filters.get("playing_time"):
        playing_time = _num(item.get("playing_time"), None)
        if playing_time is None:
            return False
        if playing_time > filters["playing_time"]:
            return False

    return True


def total_condition_pass(item, conditions, category):
    """
    통합(total) Condition Pass 여부.

    조건 충족도(condition_score)를 기준으로 평가한다.
    """
    score, _ = condition_score(item, conditions)
    return score >= 0.4  # "보통" 이상이면 pass


def condition_pass_at_k_bbabang(items, filters, review_metadata, stats_lookup, k=5):
    """
    bbabang Condition@K

    top-k 결과 중 query의 hard condition을 만족하는 결과의 비율.
    items는 (doc_id, final_score, ...) 튜플 리스트.
    """
    topk = items[:k]
    if not topk:
        return 0.0

    passed = 0
    for item_tuple in topk:
        doc_id = item_tuple[0]
        if doc_id < 0 or doc_id >= len(review_metadata):
            continue
        review_item = review_metadata[doc_id]
        stats_item = get_stats_item(review_item, stats_lookup)
        if bbabang_condition_pass(stats_item, filters):
            passed += 1

    return passed / len(topk)


def condition_pass_at_k_total(items, conditions, k=5):
    """
    total Condition@K

    top-k 결과 중 조건 충족도가 "보통"(0.4) 이상인 결과의 비율.
    """
    topk = items[:k]
    if not topk:
        return 0.0

    passed = 0
    for item in topk:
        raw_item = item.get("item", item)
        score, _ = condition_score(raw_item, conditions)
        if score >= 0.4:
            passed += 1

    return passed / len(topk)


# =========================================================
# 1. extract_query_conditions 테스트 (bbabang)
# =========================================================

def test_extract_query_conditions():
    print_header("1. extract_query_conditions 테스트 (bbabang)")

    for q in BBABANG_QUERIES:
        filters, prefs = extract_query_conditions(q["user_text"])

        print_subheader(q["name"])
        print(f"  query:    {q['user_text']}")
        print(f"  filters:  {filters}")
        print(f"  prefs:    {prefs}")
        print(f"  expected_filters: {q['expected_filters']}")
        print(f"  expected_prefs:   {q['expected_prefs']}")

        # 필터 일치 여부 확인
        match_count = 0
        total_expected = len(q["expected_filters"])

        for key, expected_val in q["expected_filters"].items():
            if expected_val is None:
                continue
            if filters.get(key) == expected_val:
                match_count += 1
            else:
                print(f"  ⚠️ 불일치: {key} = {filters.get(key)} (expected: {expected_val})")

        non_null_expected = sum(1 for v in q["expected_filters"].values() if v is not None)
        if non_null_expected > 0:
            print(f"  필터 일치율: {match_count}/{non_null_expected}")


# =========================================================
# 2. retriever 비교 테스트 (total)
# =========================================================

def test_total_retriever_comparison(data_by_category):
    print_header("2. retriever 비교 — total (keyword + condition + RRF)")

    results_table = []

    for q in TOTAL_QUERIES:
        category = q["category"]
        conditions = q["conditions"]
        query = q["user_text"]

        data = data_by_category.get(category, [])
        if not data:
            print_subheader(f"{q['name']} — ⚠️ {category} 데이터 없음, 건너뜀")
            continue

        print_subheader(q["name"])

        # Keyword retrieve
        t0 = time.time()
        keyword_results = keyword_retrieve(
            query=query,
            data=data,
            category=category,
            top_k=50,
        )
        keyword_time = time.time() - t0
        keyword_prec = precision_at_k(keyword_results, q["ground_truth"], k=5)
        keyword_cond = condition_pass_at_k_total(keyword_results, conditions, k=5)

        # Condition retrieve
        t0 = time.time()
        condition_results = condition_retrieve(
            data=data,
            conditions=conditions,
            top_k=50,
        )
        condition_time = time.time() - t0
        condition_prec = precision_at_k(condition_results, q["ground_truth"], k=5)
        condition_cond = condition_pass_at_k_total(condition_results, conditions, k=5)

        # RRF fusion
        t0 = time.time()
        rrf_results = total_rrf_fusion(
            [keyword_results, condition_results],
            k=60,
            top_k=5,
        )
        rrf_time = time.time() - t0
        rrf_prec = precision_at_k(rrf_results, q["ground_truth"], k=5)
        rrf_cond = condition_pass_at_k_total(rrf_results, conditions, k=5)

        print("\n  Title Ground Truth Precision@5")
        print(f"  {'방식':<14} {'P@5':>8} {'건수':>6} {'시간':>8}")
        print(f"  {'-' * 40}")

        for name, prec, items, elapsed in [
            ("RRF", rrf_prec, rrf_results, rrf_time),
            ("Keyword", keyword_prec, keyword_results, keyword_time),
            ("Condition", condition_prec, condition_results, condition_time),
        ]:
            prec_str = f"{prec:.3f}" if prec is not None else "N/A"
            print(f"  {name:<14} {prec_str:>8} {len(items):>6} {elapsed:>7.2f}s")

        print("\n  Condition@5 — 조건 충족도 ≥ 0.4 비율")
        print(f"  {'방식':<14} {'Cond@5':>8}")
        print(f"  {'-' * 26}")

        for name, cond in [
            ("RRF", rrf_cond),
            ("Keyword", keyword_cond),
            ("Condition", condition_cond),
        ]:
            print(f"  {name:<14} {cond:>8.3f}")

        print("\n  RRF 상위 5개:")
        print_items(rrf_results, 5, mode="total")

        # 조건 상세
        for idx, result in enumerate(rrf_results[:5], 1):
            score, detail = condition_score(result["item"], conditions)
            print(f"\n  [{idx}] {result['title']} — 조건 충족: {score:.4f} ({grade_score(score)})")
            for cond_name, cond_detail in detail.items():
                matched = cond_detail["matched"]
                cond_score = cond_detail["score"]
                matched_str = ", ".join(matched[:3]) if matched else "없음"
                print(f"      {cond_name}: {cond_score}점 / 매칭: {matched_str}")

        results_table.append({
            "category": category,
            "query": q["name"],

            "rrf_prec": rrf_prec,
            "keyword_prec": keyword_prec,
            "condition_prec": condition_prec,

            "rrf_cond": rrf_cond,
            "keyword_cond": keyword_cond,
            "condition_cond": condition_cond,
        })

    return results_table


# =========================================================
# 3. bbabang 단독 retriever 테스트 (FAISS/BM25 필요)
# =========================================================

def test_bbabang_retriever(review_metadata, stats_metadata):
    """
    방탈출(bbabang) retriever 테스트.

    FAISS index와 SentenceTransformer가 필요하므로,
    데이터 파일이 없으면 건너뛴다.
    """
    print_header("3. bbabang retriever 테스트 (BM25 + hard filter)")

    if not _BBABANG_AVAILABLE:
        print("  ⚠️ bbabang 모듈을 사용할 수 없습니다 (faiss/sentence_transformers 필요).")
        print("     pip install faiss-cpu sentence-transformers rank-bm25")
        return []

    if not review_metadata:
        print("  ⚠️ bbabang review metadata가 없어 건너뜁니다.")
        return []

    stats_lookup = build_stats_lookup(stats_metadata)
    bm25 = build_bm25(review_metadata)

    results_table = []

    for q in BBABANG_QUERIES:
        query = q["user_text"]
        filters, prefs = extract_query_conditions(query)

        print_subheader(q["name"])
        print(f"  query:   {query}")
        print(f"  filters: {filters}")
        print(f"  prefs:   {prefs}")

        # BM25 검색
        t0 = time.time()
        bm25_results = bm25_search(query, bm25, top_n=1000)
        bm25_time = time.time() - t0

        # Hard filter 적용
        bm25_filtered = filter_ranked_docs(
            ranked_docs=bm25_results,
            review_metadata=review_metadata,
            stats_lookup=stats_lookup,
            filters=filters,
        )

        print(f"\n  BM25 전체: {len(bm25_results)}건 → 필터 후: {len(bm25_filtered)}건 ({bm25_time:.2f}s)")

        # RRF (BM25 단독 → 자기 자신만으로 RRF → weight bonus)
        rrf_scores = rrf_fusion(
            [bm25_filtered],
            rrf_k=60,
        )

        final_results = add_weight_bonus(
            rrf_scores=rrf_scores,
            review_metadata=review_metadata,
            stats_lookup=stats_lookup,
            prefs=prefs,
            use_rank_bonus=True,
        )

        # Precision@5
        # bbabang에서는 doc_id로 제목을 찾아야 함
        final_with_titles = []
        for item_tuple in final_results[:5]:
            doc_id = item_tuple[0]
            review_item = review_metadata[doc_id]
            stats_item = get_stats_item(review_item, stats_lookup)
            final_with_titles.append({
                "title": review_item.get("title", "?"),
                "store_name": review_item.get("store_name", "?"),
                "doc_id": doc_id,
                "final_score": item_tuple[1],
                "item": stats_item,
            })

        prec = precision_at_k(final_with_titles, q["ground_truth"], k=5)
        cond = condition_pass_at_k_bbabang(
            final_results, filters, review_metadata, stats_lookup, k=5
        )

        prec_str = f"{prec:.3f}" if prec is not None else "N/A"
        print(f"\n  Precision@5:  {prec_str}")
        print(f"  Condition@5:  {cond:.3f}")

        print("\n  상위 5개 결과:")
        for i, item_info in enumerate(final_with_titles[:5], 1):
            doc_id = item_info["doc_id"]
            review_item = review_metadata[doc_id]
            stats_item = item_info["item"]

            print(
                f"  {i}. {review_item.get('title')} | {review_item.get('store_name')} | "
                f"area={stats_item.get('area')} | location={stats_item.get('location')} | "
                f"satisfaction={stats_item.get('satisfaction')} | "
                f"final={item_info['final_score']:.4f}"
            )

        results_table.append({
            "query": q["name"],
            "prec": prec,
            "cond": cond,
            "n_filtered": len(bm25_filtered),
        })

    return results_table


# =========================================================
# 4. emotion_tag (감정 태그) 테스트
# =========================================================

def test_emotion_tags():
    print_header("4. emotion_tag (감정 태그) 테스트")

    # 4-1. bbabang sentiment_tag_bonus 테스트
    print_subheader("4-1. bbabang sentiment_tag_bonus")

    test_reviews = [
        {
            "title": "긍정 리뷰 테스트",
            "store_name": "테스트 매장",
            "document": "진짜 꽃길이었습니다. 스토리도 좋고 몰입감 최고. 완전 추천합니다.",
        },
        {
            "title": "부정 리뷰 테스트",
            "store_name": "테스트 매장",
            "document": "별로였습니다. 장치 오류도 있었고 전반적으로 실망했습니다. 비추입니다.",
        },
        {
            "title": "중립 리뷰 테스트",
            "store_name": "테스트 매장",
            "document": "보통이었습니다. 나쁘지는 않았지만 특별한 점은 없었어요.",
        },
    ]

    for review in test_reviews:
        bonus, pos_tags, neg_tags = sentiment_tag_bonus(review)
        print(f"\n  제목: {review['title']}")
        print(f"  긍정 태그: {pos_tags}")
        print(f"  부정 태그: {neg_tags}")
        print(f"  보너스 점수: {bonus:+.4f}")

    # 4-2. total tag_item 테스트
    print_subheader("4-2. total tag_item (감정 태그 추출)")

    test_items = [
        {
            "title": "이스케이프룸: 시간의 방",
            "source": "escape",
            "document": "스토리도 좋고 연출도 좋아서 몰입감이 좋았습니다. "
                        "장치도 깔끔하게 작동했고 추천합니다.",
            "total_score": 100,
        },
        {
            "title": "코드네임",
            "source": "boardgame",
            "document": "친구들이랑 했는데 쉬운 입문용 게임이라 부담 없이 즐겼습니다. "
                        "대화도 많아지고 만족스러웠어요.",
            "total_score": 100,
        },
        {
            "title": "부정 테스트",
            "source": "murdermystery",
            "document": "기대하고 갔는데 전체적으로 아쉬웠습니다. "
                        "별로였고 실망스러워서 비추천합니다.",
            "total_score": 100,
        },
    ]

    for item in test_items:
        tags = tag_item(item)
        print(f"\n  제목: {item['title']} ({item['source']})")
        print(f"  추출된 태그: {tags}")

    # 4-3. total filter_and_score 테스트
    print_subheader("4-3. total filter_and_score (필터링 + 점수 재계산)")

    emotion_tags = [
        "재밌음", "만족도높음", "강추", "스토리좋음",
        "연출좋음", "몰입감좋음", "입문용", "친목용",
    ]

    filtered = total_filter_and_score(
        items=test_items,
        emotion_tags=emotion_tags,
        horror_tolerance=2,
        emotion_weight=5.0,
    )

    for item in filtered:
        base_score = item.get("total_score", 0)
        final_score = item.get("final_score", 0)
        diff = final_score - base_score
        sign = "+" if diff >= 0 else ""

        print(f"\n  제목: {item['title']}")
        print(f"  감정 태그: {item.get('emotion_tags', [])}")
        print(f"  기본 점수: {base_score} → 최종: {final_score} ({sign}{diff})")
        print(f"  긍정 매칭: {item.get('emotion_match_score', 0)}")
        print(f"  부정 점수: {item.get('negative_score', 0)}")


# =========================================================
# 5. prompt 생성 테스트 (추천 이유 + 역질문)
# =========================================================

def test_prompts():
    print_header("5. prompt 생성 테스트 (추천 이유 + 역질문)")

    # 5-1. bbabang 추천 이유
    print_subheader("5-1. bbabang 추천 이유 생성")

    review_item = {
        "title": "경성",
        "store_name": "셜록홈즈 원주점",
        "document": "진짜 꽃길이었습니다. 스토리도 좋고 몰입감이 좋았습니다. 추천합니다.",
    }

    stats_item = {
        "location": "원주시",
        "satisfaction": 3.66,
        "story": 3.8,
        "puzzle": 3.4,
        "interior": 3.2,
        "production": 3.5,
        "horror": 0.36,
        "difficulty": 3.59,
    }

    score_info = {
        "rrf_score": 0.0164,
        "first_bonus": 0.2196,
        "second_bonus": 0.1162,
        "sentiment_bonus": 0.0800,
        "pos_tags": ["꽃길", "추천", "몰입"],
        "neg_tags": [],
    }

    reason = generate_recommend_reason(review_item, stats_item, score_info)
    print(f"\n  {reason}")

    # 5-2. bbabang 역질문
    print_subheader("5-2. bbabang 역질문 생성")

    filters = {"location": "원주시"}
    prefs = {"story": True, "satisfaction": True}

    followup = generate_followup_question(filters, prefs)
    print(f"\n  {followup}")

    # 5-3. total 역질문 생성
    print_subheader("5-3. total 역질문 생성")

    test_cases = [
        ("보드게임 추천해줘", "boardgame", {}),
        ("방탈출 하고 싶어", "escape", {}),
        ("머더미스터리 하나 추천", "murder", {}),
        ("4명이서 할 파티게임", "boardgame", {"headcount": 4}),
    ]

    for query, category, group in test_cases:
        questions = generate_question_style_followups(
            query=query,
            category=category,
            group=group,
            top_result=None,
            max_questions=3,
        )

        print(f"\n  쿼리: '{query}' ({category})")
        for i, question in enumerate(questions, 1):
            print(f"    Q{i}. {question}")


# =========================================================
# 6. 종합 평가표
# =========================================================

def _avg(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return sum(vals) / len(vals) if vals else 0.0


def print_summary(total_results, bbabang_results):
    print_header("6. 종합 평가 점수 비교표")

    # 6-1. Total retriever 비교
    if total_results:
        print("\n  6-1. Total Retriever — Title Ground Truth Precision@5")
        print(f"\n  {'쿼리':<28} {'RRF':>8} {'Keyword':>8} {'Condition':>10}")
        print(f"  {'-' * 58}")

        for r in total_results:
            rrf = f"{r['rrf_prec']:.3f}" if r["rrf_prec"] is not None else "N/A"
            kw = f"{r['keyword_prec']:.3f}" if r["keyword_prec"] is not None else "N/A"
            cond = f"{r['condition_prec']:.3f}" if r["condition_prec"] is not None else "N/A"
            print(f"  {r['query']:<28} {rrf:>8} {kw:>8} {cond:>10}")

        print(f"  {'-' * 58}")
        print(
            f"  {'전체 평균':<28} "
            f"{_avg(total_results, 'rrf_prec'):>8.3f} "
            f"{_avg(total_results, 'keyword_prec'):>8.3f} "
            f"{_avg(total_results, 'condition_prec'):>10.3f}"
        )

        print("\n  6-2. Total Retriever — Condition@5 (조건 충족도 ≥ 0.4)")
        print(f"\n  {'쿼리':<28} {'RRF':>8} {'Keyword':>8} {'Condition':>10}")
        print(f"  {'-' * 58}")

        for r in total_results:
            print(
                f"  {r['query']:<28} "
                f"{r['rrf_cond']:>8.3f} "
                f"{r['keyword_cond']:>8.3f} "
                f"{r['condition_cond']:>10.3f}"
            )

        print(f"  {'-' * 58}")
        print(
            f"  {'전체 평균':<28} "
            f"{_avg(total_results, 'rrf_cond'):>8.3f} "
            f"{_avg(total_results, 'keyword_cond'):>8.3f} "
            f"{_avg(total_results, 'condition_cond'):>10.3f}"
        )

    # 6-3. Bbabang retriever 결과
    if bbabang_results:
        print("\n  6-3. Bbabang Retriever — Precision@5 & Condition@5")
        print(f"\n  {'쿼리':<28} {'P@5':>8} {'Cond@5':>8} {'필터 후':>8}")
        print(f"  {'-' * 56}")

        for r in bbabang_results:
            prec = f"{r['prec']:.3f}" if r["prec"] is not None else "N/A"
            print(
                f"  {r['query']:<28} "
                f"{prec:>8} "
                f"{r['cond']:>8.3f} "
                f"{r['n_filtered']:>8}"
            )

        print(f"  {'-' * 56}")
        print(
            f"  {'전체 평균':<28} "
            f"{_avg(bbabang_results, 'prec'):>8.3f} "
            f"{_avg(bbabang_results, 'cond'):>8.3f} "
            f"{'':>8}"
        )

    # 설명
    print("\n  참고:")
    print("  - Precision@5은 ground_truth 제목이 top5에 포함되었는지 보는 지표입니다.")
    print("  - Condition@5은 top5 결과가 hard condition을 만족하는지 보는 지표입니다.")
    print("  - Total의 Condition@5은 조건 충족도 ≥ 0.4 ('보통' 이상) 비율입니다.")
    print("  - Bbabang의 Condition@5은 지역/인원/가격/시간 hard filter 기준입니다.")
    print("  - ground_truth가 비어 있는 쿼리는 Precision@5에서 N/A로 표시됩니다.")


# =========================================================
# 데이터 로드 헬퍼
# =========================================================

def _try_load_json(path):
    """JSON 파일을 안전하게 로드한다."""
    if path is None:
        return []

    path = Path(path)

    if not path.exists():
        print(f"  [파일 없음] {path}")
        return []

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"  [로드 실패] {path}: {e}")
        return []


def _find_meta_files(base_dir):
    """
    프로젝트 디렉토리에서 메타데이터 파일을 자동 탐색한다.

    탐색 경로:
    1. base_dir (프로젝트 루트)
    2. base_dir / data
    3. 현재 작업 디렉토리
    """
    search_dirs = [
        Path(base_dir),
        Path(base_dir) / "data",
        Path.cwd(),
        Path.cwd() / "data",
    ]

    meta_files = {
        "boardgame": None,
        "escape_reviews": None,
        "escape_stats": None,
        "murder": None,
    }

    keywords = {
        "boardgame": ["bgg_stats", "boardgame_meta"],
        "escape_reviews": ["bbabang_reviews_meta", "bbabang_review"],
        "escape_stats": ["bbabang_stats_meta", "bbabang_stats"],
        "murder": ["murdermysterylog", "murder_meta"],
    }

    for search_dir in search_dirs:
        if not search_dir.exists():
            continue

        for file in search_dir.glob("*.json"):
            name = file.name.lower()

            for key, kw_list in keywords.items():
                if meta_files[key] is None:
                    for kw in kw_list:
                        if kw.lower() in name:
                            meta_files[key] = file
                            break

    return meta_files


# =========================================================
# 메인
# =========================================================

if __name__ == "__main__":
    print("\n🚀 Nolit 용국 RAG 파이프라인 통합 평가 시작\n")
    total_start = time.time()

    # ─── 데이터 파일 탐색 ────────────────────────────────────
    print_header("0. 데이터 파일 탐색")

    meta_files = _find_meta_files(PROJECT_ROOT)

    print(f"  boardgame:      {meta_files['boardgame']}")
    print(f"  escape_reviews: {meta_files['escape_reviews']}")
    print(f"  escape_stats:   {meta_files['escape_stats']}")
    print(f"  murder:         {meta_files['murder']}")

    # 데이터 로드
    boardgame_meta = _try_load_json(meta_files["boardgame"])
    escape_reviews_meta = _try_load_json(meta_files["escape_reviews"])
    escape_stats_meta = _try_load_json(meta_files["escape_stats"])
    murder_meta = _try_load_json(meta_files["murder"])

    print(f"\n  boardgame_meta:      {len(boardgame_meta)}건")
    print(f"  escape_reviews_meta: {len(escape_reviews_meta)}건")
    print(f"  escape_stats_meta:   {len(escape_stats_meta)}건")
    print(f"  murder_meta:         {len(murder_meta)}건")

    data_by_category = {
        "boardgame": boardgame_meta,
        "escape": escape_stats_meta if escape_stats_meta else escape_reviews_meta,
        "murder": murder_meta,
    }

    # 1. extract_query_conditions
    test_extract_query_conditions()

    # 2. total retriever 비교
    total_results = test_total_retriever_comparison(data_by_category)

    # 3. bbabang retriever 테스트
    bbabang_results = test_bbabang_retriever(escape_reviews_meta, escape_stats_meta)

    # 4. emotion_tag
    test_emotion_tags()

    # 5. prompt 생성
    test_prompts()

    # 6. 종합 비교
    print_summary(total_results, bbabang_results)

    total_elapsed = time.time() - total_start
    print(f"\n✅ 전체 평가 완료 ({total_elapsed:.1f}s)")