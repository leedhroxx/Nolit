"""
yoonha_test_graph.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent.parent)
)

from recommender.yoonha_graph import graph, run_pipeline

# =========================================================
# 유틸
# =========================================================

def print_header(text):
    print(f"\n{'='*70}")
    print(f"  {text}")
    print(f"{'='*70}")


def print_subheader(text):
    print(f"\n--- {text} ---")


# =========================================================
# 1. graph.invoke 테스트
# =========================================================

def test_graph_invoke():
    print_header("1. graph.invoke 테스트")

    payload = {
        "query": "4명이서 할 보드게임",
        "category": "boardgame",
        "use_api": True,
    }

    print_subheader("입력")
    print(payload)

    start = time.time()

    result = graph.invoke(payload)

    elapsed = time.time() - start

    print_subheader("출력")
    print(result)

    assert isinstance(result, dict)

    assert "answer" in result
    assert "games" in result
    assert "next_question" in result

    assert isinstance(result["answer"], str)
    assert isinstance(result["games"], list)
    assert isinstance(result["next_question"], str)

    print(f"\n✅ graph.invoke 테스트 통과 ({elapsed:.2f}s)")


# =========================================================
# 2. run_pipeline 호환성 테스트
# =========================================================

def test_run_pipeline():
    print_header("2. run_pipeline 호환성 테스트")

    start = time.time()

    result = run_pipeline(
        user_text="4명이서 할 전략 보드게임 추천해줘",
        group={
            "headcount": 4,
            "weight_pref": "heavy",
            "play_time": 120,
            "relation": "friend",
        },
        category="boardgame",
        use_api=True,
    )

    elapsed = time.time() - start

    print_subheader("출력")
    print(result)

    assert isinstance(result, dict)

    assert "answer" in result
    assert "games" in result
    assert "next_question" in result

    print(f"\n✅ run_pipeline 테스트 통과 ({elapsed:.2f}s)")


# =========================================================
# 3. Clarifying Question 테스트
# =========================================================

def test_clarifying_question():
    print_header("3. Clarifying Question 테스트")

    payload = {
        "query": "보드게임 추천해줘",
        "category": "boardgame",
        "use_api": True,
    }

    print_subheader("입력")
    print(payload)

    start = time.time()

    result = graph.invoke(payload)

    elapsed = time.time() - start

    print_subheader("출력")
    print(result)

    assert isinstance(result, dict)

    assert "next_question" in result
    assert result["next_question"] != ""

    print(f"\n✅ Clarifying Question 테스트 통과 ({elapsed:.2f}s)")


# =========================================================
# 4. Murder Mystery 테스트
# =========================================================

def test_murder_mystery():
    print_header("4. 머더미스터리 추천 테스트")

    payload = {
        "query": "6명이서 할 쉬운 머더미스터리",
        "category": "murdermystery",
        "use_api": True,
    }

    print_subheader("입력")
    print(payload)

    start = time.time()

    result = graph.invoke(payload)

    elapsed = time.time() - start

    print_subheader("출력")
    print(result)

    assert isinstance(result, dict)

    assert "answer" in result
    assert "games" in result
    assert "next_question" in result

    print(f"\n✅ 머더미스터리 테스트 통과 ({elapsed:.2f}s)")


# =========================================================
# 메인
# =========================================================

if __name__ == "__main__":
    print("\n🚀 yoonha_test_graph 시작\n")

    total_start = time.time()

    test_graph_invoke()
    test_run_pipeline()
    test_clarifying_question()
    test_murder_mystery()

    total_elapsed = time.time() - total_start

    print(f"\n🎉 전체 테스트 완료 ({total_elapsed:.2f}s)")