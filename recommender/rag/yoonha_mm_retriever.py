"""
mm_retriever.py
머더미스터리로그 BM25 + FAISS RRF 하이브리드 검색
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

_index = faiss.read_index(str(DATA_DIR / "faiss_murdermysterylog.index"))
with open(DATA_DIR / "faiss_murdermysterylog_meta.json", "r", encoding="utf-8") as f:
    all_items = json.load(f)

for item in all_items:
    item["source"] = "murdermysterylog"
    if "name" in item and "title" not in item:
        item["title"] = item["name"]

print(f"[mm_retriever] 머더미스터리: {len(all_items)}개 (dim={_index.d})")

# -------------------------
# BM25 준비
# -------------------------
def _make_searchable_text(item: dict) -> str:
    parts = [str(item.get("title", item.get("name", "")))]
    if item.get("description"):
        parts.append(str(item["description"])[:500])
    if item.get("시리즈"):
        parts.append(str(item["시리즈"]))
    if item.get("제작"):
        parts.append(str(item["제작"]))
    if item.get("reviews"):
        parts.append(str(item["reviews"]))
    for tag in item.get("emotion_tags", []):
        parts.append(str(tag))
    return " ".join(parts)


_corpus = [_make_searchable_text(s) for s in all_items]
_tokenized_corpus = [c.split() for c in _corpus]
_bm25 = BM25Okapi(_tokenized_corpus)
print(f"[mm_retriever] BM25 corpus 준비 완료: {len(_corpus)}개")


# -------------------------
# 하드 필터
# -------------------------
def hard_filter(item: dict, query_filter: dict) -> bool:
    """
    조건 불만족 아이템 제거. True = 통과.

    query_filter 지원 키:
        players (int)   : 플레이어 수
        max_time (int)  : 최대 플레이 시간 (분)
        area (str)      : 지역 (예: "경기", "서울") — 향후 확장용
    """
    if "players" in query_filter:
        max_p = item.get("max_players") or 999
        min_p = item.get("min_players") or 0
        if not isinstance(max_p, (int, float)): max_p = 999
        if not isinstance(min_p, (int, float)): min_p = 0
        if query_filter["players"] > max_p or query_filter["players"] < min_p:
            return False

    if "max_time" in query_filter:
        pt = item.get("play_time") or 0
        if isinstance(pt, (int, float)) and pt > 0 and pt > query_filter["max_time"]:
            return False

    if "area" in query_filter:
        item_area = item.get("area") or item.get("지역") or ""
        if query_filter["area"] not in str(item_area):
            return False

    return True


# -------------------------
# 메타데이터 가중치
# -------------------------
def _metadata_weight(item: dict, query_filter: dict) -> float:
    score = 0.0
    rating = item.get("rating")
    if isinstance(rating, (int, float)) and rating > 0:
        score += rating * 2
    return score


# -------------------------
# 검색 내부 함수
# -------------------------
def _bm25_search(query_text: str, query_filter: dict, topk: int = 200) -> dict:
    tokens = query_text.split()
    scores = _bm25.get_scores(tokens)
    top_idx = np.argsort(scores)[::-1]
    results = {}
    rank = 0
    for idx in top_idx:
        if idx >= len(all_items): continue
        item = all_items[idx]
        if not hard_filter(item, query_filter): continue
        title = item.get("title", item.get("name", ""))
        if title not in results:
            rank += 1
            results[title] = {"item": item, "rank": rank, "bm25_score": float(scores[idx])}
            if rank >= topk: break
    return results


def _dense_search(query_vector: np.ndarray, query_filter: dict, topk: int = 200) -> dict:
    if query_vector.shape[1] != _index.d:
        raise ValueError(f"쿼리 벡터 dim 불일치: {query_vector.shape[1]} != {_index.d}")
    D, I = _index.search(query_vector, topk * 3)
    results = {}
    rank = 0
    for i, idx in enumerate(I[0]):
        if idx < 0 or idx >= len(all_items): continue
        item = all_items[idx]
        if not hard_filter(item, query_filter): continue
        title = item.get("title", item.get("name", ""))
        if title not in results:
            rank += 1
            results[title] = {"item": item, "rank": rank, "l2_dist": float(D[0][i])}
            if rank >= topk: break
    return results


def _rrf_fuse(
    bm25_results: dict,
    dense_results: dict,
    query_filter: dict,
    topk: int,
    k: int = 60,
) -> list[dict]:
    all_titles = set(list(bm25_results.keys()) + list(dense_results.keys()))
    scored = []
    for title in all_titles:
        bm25_data = bm25_results.get(title)
        dense_data = dense_results.get(title)
        bm25_rank = bm25_data["rank"] if bm25_data else 999
        dense_rank = dense_data["rank"] if dense_data else 999
        if bm25_rank == 999 and dense_rank == 999:
            continue
        rrf_score = 1 / (k + bm25_rank) + 1 / (k + dense_rank)
        item = (bm25_data or dense_data)["item"]
        meta_score = _metadata_weight(item, query_filter)
        total_score = rrf_score * 1000 + meta_score
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
    타이틀을 찾지 못하면 인덱스 0번 벡터 사용.
    """
    embeddings = []
    for title in titles:
        for i, s in enumerate(all_items):
            if s.get("title") == title or s.get("name") == title:
                embeddings.append(_index.reconstruct(i))
                break
    if embeddings:
        return np.mean(embeddings, axis=0).reshape(1, -1).astype(np.float32)
    return _index.reconstruct(0).reshape(1, -1).astype(np.float32)


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
    results.sort(key=lambda x: x.get("rating") or 0, reverse=True)
    return results[:topk]