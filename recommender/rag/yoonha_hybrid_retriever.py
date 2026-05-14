"""
hybrid_retriever.py
category 기반 라우터 — yoonha_graph.py 에서 이 파일만 호출하면 됨.

내부적으로:
  "boardgame"     → boardgame_retriever.retrieve()
  "murdermystery" → mm_retriever.retrieve()
"""

import numpy as np
from pathlib import Path
import sys

# 직접 실행(python hybrid_retriever.py)과 패키지 임포트 둘 다 지원
sys.path.insert(0, str(Path(__file__).resolve().parent))
from yoonha_boardgame_retriever import (
    retrieve as _bg_retrieve,
    retrieve_bm25 as _bg_bm25,
    retrieve_dense as _bg_dense,
    retrieve_vanilla as _bg_vanilla,
    get_embedding as _bg_embedding,
)
from yoonha_mm_retriever import (
    retrieve as _mm_retrieve,
    retrieve_bm25 as _mm_bm25,
    retrieve_dense as _mm_dense,
    retrieve_vanilla as _mm_vanilla,
    get_embedding as _mm_embedding,
)


def retrieve(
    query_text: str,
    query_filter: dict,
    query_vector: np.ndarray,
    category: str,
    topk: int = 50,
) -> list[dict]:
    """
    카테고리에 맞는 retriever로 라우팅.

    Args:
        query_text:    BM25용 자연어 쿼리
        query_filter:  hard_filter 조건 dict
                       보드게임: players, playing_time, weight_max, weight_pref, category, mechanism
                       머더미스터리: players, max_time, area
        query_vector:  FAISS dense 검색용 벡터 (1, dim)
        category:      "boardgame" | "murdermystery"
        topk:          반환할 최대 아이템 수

    Returns:
        아이템 리스트 (total_score 내림차순)

    Raises:
        ValueError: category가 올바르지 않을 때
    """
    if category == "boardgame":
        return _bg_retrieve(query_text, query_filter, query_vector, topk)
    elif category == "murdermystery":
        return _mm_retrieve(query_text, query_filter, query_vector, topk)
    else:
        raise ValueError(f"알 수 없는 category: {category!r}. 'boardgame' 또는 'murdermystery' 사용.")


def retrieve_bm25(
    query_text: str,
    query_filter: dict,
    category: str,
    topk: int = 50,
) -> list[dict]:
    """BM25 단독 검색 라우터."""
    if category == "boardgame":
        return _bg_bm25(query_text, query_filter, topk)
    elif category == "murdermystery":
        return _mm_bm25(query_text, query_filter, topk)
    else:
        raise ValueError(f"알 수 없는 category: {category!r}.")


def retrieve_dense(
    query_vector: np.ndarray,
    query_filter: dict,
    category: str,
    topk: int = 50,
) -> list[dict]:
    """Dense 단독 검색 라우터."""
    if category == "boardgame":
        return _bg_dense(query_vector, query_filter, topk)
    elif category == "murdermystery":
        return _mm_dense(query_vector, query_filter, topk)
    else:
        raise ValueError(f"알 수 없는 category: {category!r}.")


def retrieve_vanilla(
    query_filter: dict,
    category: str,
    topk: int = 50,
) -> list[dict]:
    """Vanilla 검색 라우터 — 필터 통과 후 평점 내림차순."""
    if category == "boardgame":
        return _bg_vanilla(query_filter, topk)
    elif category == "murdermystery":
        return _mm_vanilla(query_filter, topk)
    else:
        raise ValueError(f"알 수 없는 category: {category!r}.")


def get_embedding(titles: list[str], category: str) -> np.ndarray:
    """
    앵커 타이틀 리스트 → 평균 임베딩 벡터 반환 (shape: (1, dim)).

    Args:
        titles:   임베딩 기준 타이틀 리스트 (없으면 빈 리스트 [] 전달)
        category: "boardgame" | "murdermystery"
    """
    if category == "boardgame":
        return _bg_embedding(titles)
    elif category == "murdermystery":
        return _mm_embedding(titles)
    else:
        raise ValueError(f"알 수 없는 category: {category!r}.")