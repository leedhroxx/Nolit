"""
yoonha_eval_pipeline.py
전체 RAG 파이프라인 통합 평가

테스트 항목:
    1. query_transformer 변환 확인
    2. retriever 검색 결과 확인 (RRF / BM25 / Dense / Vanilla)
    3. tag_filter 필터링 확인
    4. generator 생성 확인 (룰 기반)
    5. graph 파이프라인 E2E 확인
    6. Precision@K 비교 (RRF vs BM25 vs Dense vs Vanilla)
"""

import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "rag"))

from yoonha_query_transformer import transform as query_transform
from yoonha_hybrid_retriever import retrieve, retrieve_bm25, retrieve_dense, retrieve_vanilla, get_embedding
from yoonha_tag_filter import filter_and_score
from yoonha_generator import generate_without_api
from yoonha_graph import run_pipeline


# =========================================================
# 평가 쿼리 정의
# =========================================================

BOARDGAME_QUERIES = [
    {
        "name": "4인 전략 보드게임 (무거운)",
        "user_text": "4명이서 할 전략 보드게임",
        "group": {
            "headcount": 4,
            "play_time": 120,
            "weight_pref": "heavy",
            "category": "Strategy",
            "horror_tolerance": 2,
            "relation": "friend",
        },
        "ground_truth": ["Brass: Birmingham", "브라스: 버밍엄", "Twilight Struggle"],
    },
    {
        "name": "2인 가벼운 파티게임",
        "user_text": "2명이서 가볍게 할 게임",
        "group": {
            "headcount": 2,
            "play_time": 60,
            "weight_pref": "light",
            "category": "Party",
            "horror_tolerance": 2,
            "relation": "couple",
        },
        "ground_truth": ["Codenames", "Dixit", "코드네임"],
    },
    {
        "name": "3인 협력 게임",
        "user_text": "3명이서 협력하는 보드게임",
        "group": {
            "headcount": 3,
            "play_time": 120,
            "weight_pref": "medium",
            "category": "Cooperative",
            "horror_tolerance": 2,
            "relation": "friend",
        },
        "ground_truth": ["Pandemic", "Spirit Island", "팬데믹"],
    },
]

MURDER_QUERIES = [
    {
        "name": "6인 쉬운 입문 머더미스터리",
        "user_text": "6명이서 할 쉬운 머더미스터리",
        "group": {
            "headcount": 6,
            "play_time": 180,
            "horror_tolerance": 0,
            "relation": "first_meeting",
        },
        "ground_truth": ["구두룡 저택의 살인", "몇 번이고 푸른 달에 불을 붙였다"],
    },
    {
        "name": "4인 머더미스터리",
        "user_text": "4명이서 할 머더미스터리",
        "group": {
            "headcount": 4,
            "play_time": 240,
            "horror_tolerance": 2,
            "relation": "friend",
        },
        "ground_truth": [],
    },
    {
        "name": "8인 대규모 파티",
        "user_text": "8명이서 할 파티 머더미스터리",
        "group": {
            "headcount": 8,
            "play_time": 300,
            "horror_tolerance": 1,
            "relation": "friend",
        },
        "ground_truth": ["구두룡 저택의 살인"],
    },
]


# =========================================================
# 유틸
# =========================================================

def precision_at_k(items, ground_truth, k=10):
    if not ground_truth:
        return None
    pred_titles = []
    for item in items[:k]:
        pred_titles.append(item.get("title", item.get("name", "")))
    hits = sum(1 for gt in ground_truth if gt in pred_titles)
    return hits / k


def print_header(text):
    print(f"\n{'='*70}")
    print(f"  {text}")
    print(f"{'='*70}")


def print_subheader(text):
    print(f"\n--- {text} ---")


def print_items(items, max_show=5):
    for i, item in enumerate(items[:max_show], 1):
        title = item.get("title", item.get("name", "?"))
        score = item.get("final_score") or item.get("total_score") or item.get("avg_rating") or item.get("rating") or "?"
        source = item.get("source", "?")
        print(f"  {i}. {title} (점수: {score}, 소스: {source})")


# =========================================================
# 1. query_transformer 테스트
# =========================================================

def test_query_transformer():
    print_header("1. query_transformer 테스트")

    for q in BOARDGAME_QUERIES[:1] + MURDER_QUERIES[:1]:
        category = "boardgame" if q in BOARDGAME_QUERIES else "murdermystery"
        result = query_transform(q["user_text"], q["group"], category)
        print_subheader(f"{q['name']} ({category})")
        print(f"  query_text:    {result['query_text']}")
        print(f"  query_filter:  {result['query_filter']}")
        print(f"  emotion_tags:  {result['emotion_tags']}")
        print(f"  anchor_titles: {result['anchor_titles']}")


# =========================================================
# 2. retriever 비교 테스트 (RRF vs BM25 vs Dense vs Vanilla)
# =========================================================

def test_retriever_comparison(queries, category):
    print_header(f"2. retriever 비교 — {category}")

    results_table = []

    for q in queries:
        transformed = query_transform(q["user_text"], q["group"], category)
        query_vector = get_embedding(transformed["anchor_titles"], category)

        print_subheader(q["name"])

        # RRF
        t0 = time.time()
        rrf_items = retrieve(
            transformed["query_text"], transformed["query_filter"],
            query_vector, category, topk=50,
        )
        rrf_time = time.time() - t0
        rrf_prec = precision_at_k(rrf_items, q["ground_truth"], k=10)

        # BM25
        t0 = time.time()
        bm25_items = retrieve_bm25(
            transformed["query_text"], transformed["query_filter"],
            category, topk=50,
        )
        bm25_time = time.time() - t0
        bm25_prec = precision_at_k(bm25_items, q["ground_truth"], k=10)

        # Dense
        t0 = time.time()
        dense_items = retrieve_dense(
            query_vector, transformed["query_filter"],
            category, topk=50,
        )
        dense_time = time.time() - t0
        dense_prec = precision_at_k(dense_items, q["ground_truth"], k=10)

        # Vanilla
        t0 = time.time()
        vanilla_items = retrieve_vanilla(
            transformed["query_filter"], category, topk=50,
        )
        vanilla_time = time.time() - t0
        vanilla_prec = precision_at_k(vanilla_items, q["ground_truth"], k=10)

        # 결과 출력
        print(f"\n  {'방식':<12} {'P@10':>8} {'건수':>6} {'시간':>8}")
        print(f"  {'-'*38}")

        for name, prec, items, elapsed in [
            ("RRF", rrf_prec, rrf_items, rrf_time),
            ("BM25", bm25_prec, bm25_items, bm25_time),
            ("Dense", dense_prec, dense_items, dense_time),
            ("Vanilla", vanilla_prec, vanilla_items, vanilla_time),
        ]:
            prec_str = f"{prec:.3f}" if prec is not None else "N/A"
            print(f"  {name:<12} {prec_str:>8} {len(items):>6} {elapsed:>7.2f}s")

        # RRF 상위 5개 출력
        print(f"\n  RRF 상위 5개:")
        print_items(rrf_items, 5)

        results_table.append({
            "query": q["name"],
            "rrf_prec": rrf_prec,
            "bm25_prec": bm25_prec,
            "dense_prec": dense_prec,
            "vanilla_prec": vanilla_prec,
        })

    return results_table


# =========================================================
# 3. tag_filter 테스트
# =========================================================

def test_tag_filter():
    print_header("3. tag_filter 테스트")

    # 보드게임 — 공포 불가 + 입문용 태그
    q = BOARDGAME_QUERIES[1]  # 2인 가벼운 파티
    transformed = query_transform(q["user_text"], q["group"], "boardgame")
    query_vector = get_embedding(transformed["anchor_titles"], "boardgame")
    items = retrieve(
        transformed["query_text"], transformed["query_filter"],
        query_vector, "boardgame", topk=20,
    )

    print_subheader(f"필터 전: {len(items)}개")
    print_items(items, 3)

    filtered = filter_and_score(
        items,
        emotion_tags=transformed["emotion_tags"],
        horror_tolerance=q["group"].get("horror_tolerance", 2),
    )

    print_subheader(f"필터 후: {len(filtered)}개")
    print_items(filtered, 3)

    # 머더미스터리 — 공포 불가
    q2 = MURDER_QUERIES[0]  # 6인 쉬운 입문
    transformed2 = query_transform(q2["user_text"], q2["group"], "murdermystery")
    query_vector2 = get_embedding(transformed2["anchor_titles"], "murdermystery")
    items2 = retrieve(
        transformed2["query_text"], transformed2["query_filter"],
        query_vector2, "murdermystery", topk=20,
    )

    print_subheader(f"머더미스터리 필터 전: {len(items2)}개")
    print_items(items2, 3)

    filtered2 = filter_and_score(
        items2,
        emotion_tags=transformed2["emotion_tags"],
        horror_tolerance=q2["group"].get("horror_tolerance", 2),
    )

    print_subheader(f"머더미스터리 필터 후: {len(filtered2)}개")
    print_items(filtered2, 3)


# =========================================================
# 4. generator 테스트 (룰 기반)
# =========================================================

def test_generator():
    print_header("4. generator 테스트 (룰 기반)")

    # 보드게임
    q = BOARDGAME_QUERIES[0]
    transformed = query_transform(q["user_text"], q["group"], "boardgame")
    query_vector = get_embedding(transformed["anchor_titles"], "boardgame")
    items = retrieve(
        transformed["query_text"], transformed["query_filter"],
        query_vector, "boardgame", topk=10,
    )
    filtered = filter_and_score(items, transformed["emotion_tags"])

    result = generate_without_api(filtered, q["group"], "boardgame", transformed["emotion_tags"])

    print_subheader("보드게임 추천")
    for i, rec in enumerate(result["recommendations"], 1):
        print(f"  {i}. {rec['title']}")
        print(f"     {rec['reason']}")
    print(f"\n  ❓ 역질문: {result['follow_up_question']}")

    # 머더미스터리
    q2 = MURDER_QUERIES[0]
    transformed2 = query_transform(q2["user_text"], q2["group"], "murdermystery")
    query_vector2 = get_embedding(transformed2["anchor_titles"], "murdermystery")
    items2 = retrieve(
        transformed2["query_text"], transformed2["query_filter"],
        query_vector2, "murdermystery", topk=10,
    )
    filtered2 = filter_and_score(items2, transformed2["emotion_tags"], horror_tolerance=0)

    result2 = generate_without_api(filtered2, q2["group"], "murdermystery", transformed2["emotion_tags"])

    print_subheader("머더미스터리 추천")
    for i, rec in enumerate(result2["recommendations"], 1):
        print(f"  {i}. {rec['title']}")
        print(f"     {rec['reason']}")
    print(f"\n  ❓ 역질문: {result2['follow_up_question']}")


# =========================================================
# 5. graph E2E 테스트
# =========================================================

def test_graph_e2e():
    print_header("5. graph 파이프라인 E2E 테스트")

    test_cases = [
        ("boardgame", "4명이서 할 전략 보드게임", BOARDGAME_QUERIES[0]["group"]),
        ("murdermystery", "6명이서 할 쉬운 머더미스터리", MURDER_QUERIES[0]["group"]),
    ]

    for category, user_text, group in test_cases:
        print_subheader(f"{category}: \"{user_text}\"")
        t0 = time.time()
        result = run_pipeline(user_text, group, category, use_api=False)
        elapsed = time.time() - t0

        recs = result.get("recommendations", [])
        print(f"  추천 {len(recs)}개 생성 ({elapsed:.2f}s)")
        for i, rec in enumerate(recs[:3], 1):
            print(f"  {i}. {rec['title']}: {rec['reason'][:60]}...")
        print(f"  ❓ {result.get('follow_up_question', '')}")


# =========================================================
# 6. 종합 Precision 비교표
# =========================================================

def print_summary(bg_results, mm_results):
    print_header("6. 종합 Precision@10 비교표")

    print(f"\n  {'쿼리':<30} {'RRF':>8} {'BM25':>8} {'Dense':>8} {'Vanilla':>8}")
    print(f"  {'-'*66}")

    all_results = bg_results + mm_results
    for r in all_results:
        rrf = f"{r['rrf_prec']:.3f}" if r['rrf_prec'] is not None else "N/A"
        bm25 = f"{r['bm25_prec']:.3f}" if r['bm25_prec'] is not None else "N/A"
        dense = f"{r['dense_prec']:.3f}" if r['dense_prec'] is not None else "N/A"
        vanilla = f"{r['vanilla_prec']:.3f}" if r['vanilla_prec'] is not None else "N/A"
        print(f"  {r['query']:<30} {rrf:>8} {bm25:>8} {dense:>8} {vanilla:>8}")

    # 평균 (None 제외)
    def avg(key):
        vals = [r[key] for r in all_results if r[key] is not None]
        return sum(vals) / len(vals) if vals else 0

    print(f"  {'-'*66}")
    print(f"  {'평균':<30} {avg('rrf_prec'):>8.3f} {avg('bm25_prec'):>8.3f} {avg('dense_prec'):>8.3f} {avg('vanilla_prec'):>8.3f}")


# =========================================================
# 메인
# =========================================================

if __name__ == "__main__":
    print("\n🚀 Nolit RAG 파이프라인 통합 평가 시작\n")
    total_start = time.time()

    # 1. query_transformer
    test_query_transformer()

    # 2. retriever 비교
    bg_results = test_retriever_comparison(BOARDGAME_QUERIES, "boardgame")
    mm_results = test_retriever_comparison(MURDER_QUERIES, "murdermystery")

    # 3. tag_filter
    test_tag_filter()

    # 4. generator
    test_generator()

    # 5. graph E2E
    test_graph_e2e()

    # 6. 종합 비교
    print_summary(bg_results, mm_results)

    total_elapsed = time.time() - total_start
    print(f"\n✅ 전체 평가 완료 ({total_elapsed:.1f}s)")