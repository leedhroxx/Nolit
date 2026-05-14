"""
tag_filter_test.py
tag_filter.py 직접 테스트 스크립트
실행: python recommender/eval/tag_filter_test.py
"""

import sys
from pathlib import Path

RAG_DIR = Path(__file__).resolve().parent.parent / "rag"
sys.path.insert(0, str(RAG_DIR))
print(f"[경로 확인] rag 디렉토리: {RAG_DIR}")
from yoonha_tag_filter import filter_and_score, get_matched_tags

# -------------------------
# 테스트 아이템 — 실제 데이터로 교체 가능
# -------------------------
items = [
    {
        "title": "코드네임",
        "total_score": 100.0,
        "emotion_tags": ["입문용", "분위기좋음", "공포없음", "처음만나는사이추천"],
    },
    {
        "title": "공포방탈출A",
        "total_score": 120.0,
        "emotion_tags": ["공포있음", "무서움", "고난이도"],
    },
    {
        "title": "극공포방탈출B",
        "total_score": 115.0,
        "emotion_tags": ["극공포", "점프스케어"],
    },
    {
        "title": "중간난이도게임",
        "total_score": 90.0,
        "emotion_tags": ["협력", "대화유도", "복잡함"],
    },
    {
        "title": "가벼운파티게임",
        "total_score": 85.0,
        "emotion_tags": ["가볍게즐길수있음", "웃음", "데이트추천"],
    },
]

# -------------------------
# 테스트 조건 — 여기서 수정
# -------------------------
emotion_tags = ["입문용", "분위기좋음", "처음만나는사이추천"]
horror_tolerance = 0   # 0=불가, 1=약간, 2=가능
emotion_weight = 5.0   # 감정 태그 가중치


# -------------------------
# 실행
# -------------------------
def run(label, horror_tolerance):
    print(f"\n{'=' * 55}")
    print(f"  {label}  (horror_tolerance={horror_tolerance})")
    print(f"{'=' * 55}")
    result = filter_and_score(items, emotion_tags, horror_tolerance=horror_tolerance, emotion_weight=emotion_weight)
    if not result:
        print("  결과 없음 (모두 필터링됨)")
        return
    print(f"  {'제목':<20} {'total':>7} {'emotion':>8} {'neg':>6} {'final':>8}")
    print(f"  {'-' * 52}")
    for item in result:
        print(
            f"  {item['title']:<20}"
            f"  {item.get('total_score', 0):>7.1f}"
            f"  {item['emotion_match_score']:>7.1f}"
            f"  {item['negative_score']:>5.1f}"
            f"  {item['final_score']:>8.1f}"
        )

    print(f"\n  [get_matched_tags 확인 — 상위 1개]")
    matched = get_matched_tags(result[0], emotion_tags)
    print(f"  {result[0]['title']} → 매칭 태그: {matched}")


run("공포 불가", horror_tolerance=0)
run("약한 공포 가능", horror_tolerance=1)
run("공포 가능", horror_tolerance=2)

print("\n✅ 테스트 완료\n")