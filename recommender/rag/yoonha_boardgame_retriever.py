"""
boardgame_retriever.py
BGG + 보드라이프 통합 BM25 + FAISS RRF 하이브리드 검색
"""

import json
import faiss
import numpy as np
from pathlib import Path
from rank_bm25 import BM25Okapi

# -------------------------
# 데이터 로드
# -------------------------
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"

bgg_index = faiss.read_index(str(DATA_DIR / "faiss_bgg_stats.index"))
with open(DATA_DIR / "faiss_bgg_stats_meta.json", "r", encoding="utf-8") as f:
    bgg_stats = json.load(f)

bl_index = faiss.read_index(str(DATA_DIR / "faiss_boardlife_stats.index"))
with open(DATA_DIR / "faiss_boardlife_stats_meta.json", "r", encoding="utf-8") as f:
    bl_stats = json.load(f)

for item in bgg_stats:
    item["source"] = "bgg"
for item in bl_stats:
    item["source"] = "boardlife"

all_items = bgg_stats + bl_stats

print(f"[boardgame_retriever] BGG: {len(bgg_stats)}개 (dim={bgg_index.d}), 보드라이프: {len(bl_stats)}개 (dim={bl_index.d})")

# -------------------------
# BM25 준비
# -------------------------
CATEGORY_KO = {
    "Strategy": "전략", "Economic": "경제", "Party": "파티", "War": "전쟁",
    "Family": "가족", "Abstract": "추상", "Thematic": "테마", "Adventure": "어드벤처",
    "Fantasy": "판타지", "Horror": "공포", "Science Fiction": "SF",
    "Deduction": "추리", "Negotiation": "협상", "Cooperative": "협력",
    "Card Game": "카드게임", "Dice": "주사위", "Puzzle": "퍼즐",
}
MECHANISM_KO = {
    "Worker Placement": "일꾼배치", "Deck Building": "덱빌딩", "Engine Building": "엔진빌딩",
    "Area Control": "지역장악", "Cooperative Game": "협력", "Auction": "경매",
    "Market": "시장", "Hand Management": "패관리", "Tile Placement": "타일배치",
    "Route Building": "루트빌딩", "Push Your Luck": "운빨", "Voting": "투표",
    "Drafting": "드래프팅", "Roll and Write": "롤앤라이트",
}
BM25_STOPWORDS = {
    "one","two","three","four","five","six","seven","eight","nine","ten",
    "a","an","the","of","in","at","to","for","and","or","is","it",
    "city","cities","world","age","game","games","edition","player","players"
}


def _translate_tags(tags_raw, mapping):
    if isinstance(tags_raw, list):
        tags = tags_raw
    elif isinstance(tags_raw, str) and tags_raw:
        tags = tags_raw.split("|")
    else:
        return []
    result = list(tags)
    for tag in tags:
        ko = mapping.get(tag.strip())
        if ko:
            result.append(ko)
    return result


def _make_searchable_text(item):
    parts = [str(item.get("title", ""))]
    if item.get("title_eng"):
        parts.append(str(item["title_eng"]))
    parts.extend(_translate_tags(item.get("category", ""), CATEGORY_KO))
    parts.extend(_translate_tags(item.get("mechanism", ""), MECHANISM_KO))
    t = item.get("type", "")
    if isinstance(t, list):
        parts.extend(t)
    elif isinstance(t, str) and t:
        parts.append(t)
    des = item.get("designer", "")
    if isinstance(des, list):
        parts.extend(des)
    elif isinstance(des, str) and des:
        parts.extend(des.split("|"))
    for tag in item.get("emotion_tags", []):
        parts.append(str(tag))
    return " ".join([str(p) for p in parts if p])


_corpus = [_make_searchable_text(s) for s in all_items]
_tokenized_corpus = [c.lower().split() for c in _corpus]
_bm25 = BM25Okapi(_tokenized_corpus)
print(f"[boardgame_retriever] BM25 corpus 준비 완료: {len(_corpus)}개")


# -------------------------
# 하드 필터
# -------------------------
def hard_filter(item: dict, query_filter: dict) -> bool:
    """
    조건 불만족 아이템 제거. True = 통과.

    query_filter 지원 키:
        players (int)       : 플레이어 수
        playing_time (int)  : 최대 플레이 시간 (분)
        weight_max (float)  : 최대 난이도
        weight_pref (str)   : "light" | "medium" | "heavy"
    """
    if "players" in query_filter:
        max_p = item.get("max_players") or 999
        min_p = item.get("min_players") or 0
        if not isinstance(max_p, (int, float)): max_p = 999
        if not isinstance(min_p, (int, float)): min_p = 0
        if query_filter["players"] > max_p or query_filter["players"] < min_p:
            return False

    if "playing_time" in query_filter:
        if item.get("source") == "boardlife":
            max_t = item.get("max_time") or 0
            if isinstance(max_t, (int, float)) and max_t > 0 and max_t > query_filter["playing_time"]:
                return False
        else:
            pt = item.get("playing_time") or 0
            if isinstance(pt, (int, float)) and pt > 0 and pt > query_filter["playing_time"]:
                return False

    if "weight_max" in query_filter:
        w = item.get("weight")
        if isinstance(w, (int, float)) and w > query_filter["weight_max"]:
            return False

    if query_filter.get("weight_pref") == "heavy":
        w = item.get("weight")
        if isinstance(w, (int, float)) and w < 3.5:
            return False

    return True


# -------------------------
# 메타데이터 가중치
# -------------------------
def _metadata_weight(item: dict, query_filter: dict) -> float:
    score = 0.0

    rating = item.get("avg_rating")
    if isinstance(rating, (int, float)) and rating > 0:
        if item.get("source") == "bgg":
            score += (rating / 10) * 15
        else:
            score += (rating / 5) * 15

    if query_filter.get("category"):
        cat = item.get("category", "")
        cat_str = "|".join(cat) if isinstance(cat, list) else str(cat) if cat else ""
        if query_filter["category"].lower() in cat_str.lower():
            score += 10

    if query_filter.get("mechanism"):
        mech = item.get("mechanism", "")
        mech_str = "|".join(mech) if isinstance(mech, list) else str(mech) if mech else ""
        if query_filter["mechanism"].lower() in mech_str.lower():
            score += 8

    if query_filter.get("players"):
        rec = item.get("recommended_players") or item.get("best_players")
        if isinstance(rec, (int, float)) and rec == query_filter["players"]:
            score += 5

    if query_filter.get("weight_pref"):
        w = item.get("weight")
        if isinstance(w, (int, float)):
            if query_filter["weight_pref"] == "light" and w < 2.5:
                score += 5
            elif query_filter["weight_pref"] == "medium" and 2.5 <= w <= 3.5:
                score += 5
            elif query_filter["weight_pref"] == "heavy" and w > 3.5:
                score += 5

    cr = item.get("category_rank")
    if cr:
        if isinstance(cr, dict):
            overall = cr.get("Overall")
        elif isinstance(cr, str):
            try:
                cr_dict = json.loads(cr.replace("'", '"'))
                overall = cr_dict.get("Overall") or cr_dict.get("전략") or cr_dict.get("가족")
            except Exception:
                overall = None
        else:
            overall = None
        if isinstance(overall, (int, float)) and overall > 0:
            score += max(0, 10 - overall * 0.01)

    if item.get("source") == "boardlife":
        score *= 1.5

    return score


# -------------------------
# 검색 내부 함수
# -------------------------
def _bm25_search(query_text: str, query_filter: dict, topk: int = 200) -> dict:
    tokens = [t for t in query_text.lower().split() if t not in BM25_STOPWORDS]
    scores = _bm25.get_scores(tokens)
    top_idx = np.argsort(scores)[::-1]
    results = {}
    rank = 0
    for idx in top_idx:
        if idx >= len(all_items): continue
        item = all_items[idx]
        if not hard_filter(item, query_filter): continue
        key = f"{item['source']}::{item.get('title', '')}"
        if key not in results:
            rank += 1
            results[key] = {"item": item, "rank": rank, "bm25_score": float(scores[idx])}
            if rank >= topk: break
    return results


def _dense_search(query_vector: np.ndarray, query_filter: dict, topk: int = 200) -> dict:
    """BGG + 보드라이프 각각 검색 후 l2_dist 기준 합산 정렬."""
    results = {}
    for index, meta in [(bgg_index, bgg_stats), (bl_index, bl_stats)]:
        if query_vector.shape[1] != index.d:
            continue
        D, I = index.search(query_vector, topk * 3)
        for i, idx in enumerate(I[0]):
            if idx < 0 or idx >= len(meta): continue
            item = meta[idx]
            if not hard_filter(item, query_filter): continue
            key = f"{item['source']}::{item.get('title', '')}"
            if key not in results:
                results[key] = {"item": item, "l2_dist": float(D[0][i])}

    # l2_dist 기준 재정렬 후 rank 부여
    sorted_items = sorted(results.items(), key=lambda x: x[1]["l2_dist"])
    ranked = {}
    for rank, (key, data) in enumerate(sorted_items[:topk], 1):
        ranked[key] = {**data, "rank": rank}
    return ranked


def _rrf_fuse(
    bm25_results: dict,
    dense_results: dict,
    query_filter: dict,
    topk: int,
    k: int = 60,
) -> list[dict]:
    all_keys = set(list(bm25_results.keys()) + list(dense_results.keys()))
    scored = []
    for key in all_keys:
        bm25_data = bm25_results.get(key)
        dense_data = dense_results.get(key)
        bm25_rank = bm25_data["rank"] if bm25_data else 999
        dense_rank = dense_data["rank"] if dense_data else 999
        if bm25_rank == 999 and dense_rank == 999:
            continue
        rrf_score = 1 / (k + bm25_rank) + 1 / (k + dense_rank)
        if dense_rank == 999:
            rrf_score *= 0.7  # dense 없을 때 BM25 단독 패널티
        item = (bm25_data or dense_data)["item"]
        meta_score = _metadata_weight(item, query_filter)
        total_score = rrf_score * 3000 + meta_score
        item_copy = item.copy()
        item_copy["rrf_score"] = round(rrf_score, 6)
        item_copy["meta_score"] = round(meta_score, 2)
        item_copy["total_score"] = round(total_score, 2)
        item_copy["bm25_rank"] = bm25_rank
        item_copy["dense_rank"] = dense_rank
        scored.append(item_copy)
    scored.sort(key=lambda x: x["total_score"], reverse=True)
    return scored[:topk]


# -------------------------
# 앵커 임베딩 유틸
# -------------------------
def get_embedding(titles: list[str]) -> np.ndarray:
    """
    타이틀 리스트 → 평균 임베딩 벡터 반환 (shape: (1, dim)).
    타이틀을 찾지 못하면 bgg 인덱스 0번 벡터 사용.
    """
    embeddings = []
    for title in titles:
        for index, meta in [(bgg_index, bgg_stats), (bl_index, bl_stats)]:
            for i, s in enumerate(meta):
                if s.get("title") == title:
                    embeddings.append(index.reconstruct(i))
                    break
    if embeddings:
        return np.mean(embeddings, axis=0).reshape(1, -1).astype(np.float32)
    return bgg_index.reconstruct(0).reshape(1, -1).astype(np.float32)


# -------------------------
# 공개 인터페이스
# -------------------------
def retrieve(
    query_text: str,
    query_filter: dict,
    query_vector: np.ndarray,
    topk: int = 50,
) -> list[dict]:
    """RRF 하이브리드 검색 (BM25 + FAISS 융합)."""
    bm25_res = _bm25_search(query_text, query_filter, topk=200)
    dense_res = _dense_search(query_vector, query_filter, topk=200)
    return _rrf_fuse(bm25_res, dense_res, query_filter, topk=topk)


def retrieve_bm25(
    query_text: str,
    query_filter: dict,
    topk: int = 50,
) -> list[dict]:
    """BM25 단독 검색."""
    results = _bm25_search(query_text, query_filter, topk=topk)
    items = sorted(results.values(), key=lambda x: x["rank"])
    return [d["item"] for d in items]


def retrieve_dense(
    query_vector: np.ndarray,
    query_filter: dict,
    topk: int = 50,
) -> list[dict]:
    """Dense 단독 검색."""
    results = _dense_search(query_vector, query_filter, topk=topk)
    items = sorted(results.values(), key=lambda x: x["rank"])
    return [d["item"] for d in items]


def retrieve_vanilla(
    query_filter: dict,
    topk: int = 50,
) -> list[dict]:
    """Vanilla 검색 — 필터 통과 후 평점 내림차순."""
    results = [item.copy() for item in all_items if hard_filter(item, query_filter)]
    results.sort(key=lambda x: x.get("avg_rating") or 0, reverse=True)
    return results[:topk]