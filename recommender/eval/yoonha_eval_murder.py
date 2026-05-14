"""
머더미스터리로그 RRF 검색 + 4가지 방식 비교 평가
- RRF / BM25 / Dense / Vanilla(필터+평점순)
- Ground Truth: hard_filter 통과한 전체 아이템
- 지표: Recall@K, Filter Pass Rate@K
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

index = faiss.read_index(str(DATA_DIR / "faiss_murdermysterylog.index"))
with open(DATA_DIR / "faiss_murdermysterylog_meta.json", "r", encoding="utf-8") as f:
    meta = json.load(f)

for item in meta:
    item["source"] = "murdermysterylog"
    if "name" in item and "title" not in item:
        item["title"] = item["name"]

print(f"[로드] 머더미스터리로그: {len(meta)}개 (dim={index.d})")

# -------------------------
# BM25 준비
# -------------------------
def make_searchable_text(item):
    parts = [str(item.get("title", item.get("name", "")))]
    if item.get("description"):
        parts.append(str(item["description"])[:500])
    if item.get("시리즈"):
        parts.append(str(item["시리즈"]))
    if item.get("제작"):
        parts.append(str(item["제작"]))
    if item.get("reviews"):
        parts.append(str(item["reviews"]))
    return " ".join(parts)

corpus = [make_searchable_text(item) for item in meta]
tokenized_corpus = [c.split() for c in corpus]
bm25 = BM25Okapi(tokenized_corpus)
print(f"[BM25] corpus 준비 완료: {len(corpus)}개")

# -------------------------
# 하드 필터
# -------------------------
def hard_filter(item, query):
    if "players" in query:
        max_p = item.get("max_players") or 999
        min_p = item.get("min_players") or 0
        if not isinstance(max_p, (int, float)): max_p = 999
        if not isinstance(min_p, (int, float)): min_p = 0
        if query["players"] > max_p or query["players"] < min_p:
            return False
    if "max_time" in query:
        mt = item.get("play_time") or 0
        if isinstance(mt, (int, float)) and mt > 0 and mt > query["max_time"]:
            return False
    return True

# -------------------------
# 메타데이터 가중치
# -------------------------
def metadata_weight(item, query):
    score = 0.0
    rating = item.get("rating")
    if isinstance(rating, (int, float)) and rating > 0:
        score += rating * 2
    return score

# -------------------------
# 검색 함수
# -------------------------
def bm25_search(query_text, query_filter, topk=200):
    tokenized_query = query_text.split()
    scores = bm25.get_scores(tokenized_query)
    top_idx = np.argsort(scores)[::-1]
    results = {}
    rank = 0
    for idx in top_idx:
        if idx >= len(meta): continue
        item = meta[idx]
        if not hard_filter(item, query_filter): continue
        title = item.get("title", item.get("name", ""))
        if title not in results:
            rank += 1
            results[title] = {"item": item, "rank": rank, "bm25_score": float(scores[idx])}
            if rank >= topk: break
    return results

def dense_search(query_vector, query_filter, topk=200):
    if query_vector.shape[1] != index.d:
        raise ValueError(f"쿼리 벡터 dim 불일치: {query_vector.shape[1]} != {index.d}")
    D, I = index.search(query_vector, topk * 3)
    results = {}
    rank = 0
    for i, idx in enumerate(I[0]):
        if idx < 0 or idx >= len(meta): continue
        item = meta[idx]
        if not hard_filter(item, query_filter): continue
        title = item.get("title", item.get("name", ""))
        if title not in results:
            rank += 1
            results[title] = {"item": item, "rank": rank, "l2_dist": float(D[0][i])}
            if rank >= topk: break
    return results

def rrf_search(query_vector, query_text, query_filter, topk=50, k=60):
    bm25_results = bm25_search(query_text, query_filter, topk=200)
    dense_results = dense_search(query_vector, query_filter, topk=200)
    all_titles = set(list(bm25_results.keys()) + list(dense_results.keys()))
    scored = []
    for title in all_titles:
        bm25_data = bm25_results.get(title)
        dense_data = dense_results.get(title)
        bm25_rank = bm25_data["rank"] if bm25_data else 999
        dense_rank = dense_data["rank"] if dense_data else 999
        rrf_score = 1 / (k + bm25_rank) + 1 / (k + dense_rank)
        item = (bm25_data or dense_data)["item"]
        meta_score = metadata_weight(item, query_filter)
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

def get_average_embedding(titles):
    embeddings = []
    for t in titles:
        for idx, s in enumerate(meta):
            if s.get("title") == t or s.get("name") == t:
                embeddings.append(index.reconstruct(idx))
                break
    if embeddings:
        return np.mean(embeddings, axis=0).reshape(1, -1)
    else:
        return index.reconstruct(0).reshape(1, -1)

# -------------------------
# Vanilla 검색 (필터 + 평점순)
# -------------------------
def vanilla_search(query_filter, topk=50):
    results = [item.copy() for item in meta if hard_filter(item, query_filter)]
    results.sort(key=lambda x: x.get("rating") or 0, reverse=True)
    return results[:topk]

# -------------------------
# BM25 / Dense 결과 → 리스트 변환
# -------------------------
def bm25_search_list(query_text, query_filter, topk=50):
    results = bm25_search(query_text, query_filter, topk=topk)
    items = sorted(results.values(), key=lambda x: x["rank"])
    return [d["item"] for d in items]

def dense_search_list(query_vector, query_filter, topk=50):
    results = dense_search(query_vector, query_filter, topk=topk)
    items = sorted(results.values(), key=lambda x: x["rank"])
    return [d["item"] for d in items]

# -------------------------
# 평가 함수
# -------------------------
def make_ground_truth(query_filter):
    return [
        item.get("title", item.get("name", ""))
        for item in meta
        if hard_filter(item, query_filter)
    ]

def recall_at_k(pred, ground_truth, k=50):
    if not ground_truth: return None
    pred_topk = [p.get("title", p.get("name", "")) for p in pred[:k]]
    hits = sum(1 for t in ground_truth if t in pred_topk)
    return hits / len(ground_truth)

def filter_pass_rate_at_k(pred, query_filter, k=50):
    topk = pred[:k]
    if not topk: return 0.0
    passed = sum(1 for item in topk if hard_filter(item, query_filter))
    return passed / len(topk)

# -------------------------
# 4가지 방식 비교 평가
# -------------------------
def evaluate_all(queries, k=50):
    all_summary = []

    for q in queries:
        print(f"\n{'='*60}")
        print(f"🔍 쿼리: {q['name']}")
        print(f"{'='*60}")

        gt = make_ground_truth(q["query_filter"])
        print(f"  조건 만족 아이템 수 (GT): {len(gt)}개\n")

        methods = {
            "RRF":     rrf_search(q["query_vector"], q["query_text"], q["query_filter"], topk=k),
            "BM25":    bm25_search_list(q["query_text"], q["query_filter"], topk=k),
            "Dense":   dense_search_list(q["query_vector"], q["query_filter"], topk=k),
            "Vanilla": vanilla_search(q["query_filter"], topk=k),
        }

        for method_name, results in methods.items():
            recall = recall_at_k(results, gt, k)
            fpr = filter_pass_rate_at_k(results, q["query_filter"], k)
            recall_str = f"{recall:.3f}" if recall is not None else "N/A"
            fpr_icon = "✅" if fpr == 1.0 else "⚠️"
            print(f"  [{method_name:<7}] Recall@{k}={recall_str}  FPR={fpr:.3f} {fpr_icon}")
            all_summary.append({
                "query_name": q["name"],
                "method": method_name,
                "gt_count": len(gt),
                "recall_at_k": recall,
                "filter_pass_rate": fpr,
            })

        # 상위 5개 비교
        print(f"\n  [상위 5개 비교]")
        for i in range(5):
            row = f"  {i+1}. "
            for method_name, results in methods.items():
                title = results[i].get("title", results[i].get("name", "?"))[:15] if i < len(results) else "-"
                row += f"{method_name}:{title:<17} "
            print(row)

    # 전체 요약
    print(f"\n\n{'='*60}")
    print("📋 전체 비교 요약")
    print(f"{'='*60}")
    print(f"  {'쿼리':<25} {'방식':<8} {'Recall@'+str(k):<12} FPR")
    print(f"  {'-'*55}")
    for m in all_summary:
        recall_str = f"{m['recall_at_k']:.3f}" if m['recall_at_k'] is not None else "N/A "
        print(f"  {m['query_name'][:24]:<25} {m['method']:<8} {recall_str:<12} {m['filter_pass_rate']:.3f}")

    return all_summary

# -------------------------
# 평가 쿼리
# -------------------------
queries = [
    {
        "name": "4인 추리 중심",
        "query_filter": {"players": 4, "max_time": 240},
        "query_text": "추리 단서 범인 논리 증거 밀담 어려운",
        "query_vector": get_average_embedding([]),
    },
    {
        "name": "8인 대규모 파티",
        "query_filter": {"players": 8, "max_time": 300},
        "query_text": "대규모 파티 다인원 역할 연기 몰입 8인",
        "query_vector": get_average_embedding(["구두룡 저택의 살인"]),
    },
    {
        "name": "2인 짧은 게임",
        "query_filter": {"players": 2, "max_time": 90},
        "query_text": "2인 짧은 간단 가벼운 데이트 입문",
        "query_vector": get_average_embedding([]),
    },
    {
        "name": "6인 고난이도 추리",
        "query_filter": {"players": 6, "max_time": 360},
        "query_text": "고난이도 어려운 추리 논리 밀도 복잡",
        "query_vector": get_average_embedding([]),
    },
]

# -------------------------
# 실행
# -------------------------
if __name__ == "__main__":
    evaluate_all(queries, k=50)