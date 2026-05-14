"""
eval_query_transformer_test.py
query_transformer.py 직접 테스트 스크립트
실행: python recommender/eval/eval_query_transformer_test.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "rag"))
from yoonha_query_transformer import transform

def print_result(label, result):
    print(f"\n{'=' * 55}")
    print(f"  {label}")
    print(f"{'=' * 55}")
    print(f"  query_text   : {result['query_text']}")
    print(f"  query_filter : {result['query_filter']}")
    print(f"  emotion_tags : {result['emotion_tags']}")
    print(f"  anchor_titles: {result['anchor_titles']}")


# -------------------------
# 테스트 케이스
# -------------------------

# 1. 4인 처음 만나는 가벼운 보드게임
print_result("4인 처음 만나는 가벼운 보드게임", transform(
    user_text="4명이서 할 보드게임",
    group={
        "headcount": 4,
        "horror_tolerance": 2,
        "play_time": 60,
        "weight_pref": "light",
        "category": "Party",
        "relation": "first_meeting",
    },
    category="boardgame",
))

# 2. 2인 공포 불가 데이트 머더미스터리
print_result("2인 공포 불가 데이트 머더미스터리", transform(
    user_text="둘이서 할 수 있는 머더미스터리",
    group={
        "headcount": 2,
        "horror_tolerance": 0,
        "play_time": 90,
        "area": "서울",
        "relation": "couple",
    },
    category="murdermystery",
))

# 3. 6인 무거운 전략 보드게임
print_result("6인 무거운 전략 보드게임", transform(
    user_text="전략적인 보드게임 추천",
    group={
        "headcount": 6,
        "horror_tolerance": 2,
        "play_time": 180,
        "weight_pref": "heavy",
        "category": "Strategy",
        "relation": "friend",
    },
    category="boardgame",
))

# 4. 조건 최소 입력
print_result("최소 조건 (인원만)", transform(
    user_text="보드게임 추천해줘",
    group={"headcount": 3},
    category="boardgame",
))

print("\n✅ 테스트 완료\n")