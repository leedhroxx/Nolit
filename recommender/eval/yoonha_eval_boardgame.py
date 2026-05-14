"""
보드게임 통합 RRF 검색 + 4가지 방식 비교 평가
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

bgg_stats_index = faiss.read_index(str(DATA_DIR / "faiss_bgg_stats.index"))
with open(DATA_DIR / "faiss_bgg_stats_meta.json", "r", encoding="utf-8") as f:
    bgg_stats = json.load(f)

bl_stats_index = faiss.read_index(str(DATA_DIR / "faiss_boardlife_stats.index"))
with open(DATA_DIR / "faiss_boardlife_stats_meta.json", "r", encoding="utf-8") as f:
    bl_stats = json.load(f)

for item in bgg_stats:
    item["source"] = "bgg"
for item in bl_stats:
    item["source"] = "boardlife"

print(f"[로드] BGG: {len(bgg_stats)}개 (dim={bgg_stats_index.d}), 보드라이프: {len(bl_stats)}개 (dim={bl_stats_index.d})")

all_stats = bgg_stats + bl_stats

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

def translate_tags(tags_raw, mapping):
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

def make_searchable_text(item):
    parts = [str(item.get("title", ""))]
    if item.get("title_eng"):
        parts.append(str(item["title_eng"]))
    parts.extend(translate_tags(item.get("category", ""), CATEGORY_KO))
    parts.extend(translate_tags(item.get("mechanism", ""), MECHANISM_KO))
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
    return " ".join([str(p) for p in parts if p])

corpus = [make_searchable_text(s) for s in all_stats]
tokenized_corpus = [c.lower().split() for c in corpus]
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
    if "playing_time" in query:
        if item.get("source") == "boardlife":
            max_t = item.get("max_time") or 0
            if isinstance(max_t, (int, float)) and max_t > 0 and max_t > query["playing_time"]:
                return False
        else:
            pt = item.get("playing_time") or 0
            if isinstance(pt, (int, float)) and pt > 0 and pt > query["playing_time"]:
                return False
    if "weight_max" in query:
        w = item.get("weight")
        if isinstance(w, (int, float)) and w > query["weight_max"]:
            return False
    if query.get("weight_pref") == "heavy":
        w = item.get("weight")
        if isinstance(w, (int, float)) and w < 3.5:
            return False
    return True

# -------------------------
# 메타데이터 가중치
# -------------------------
def metadata_weight(item, query):
    score = 0.0
    rating = item.get("avg_rating")
    if isinstance(rating, (int, float)) and rating > 0:
        if item.get("source") == "bgg":
            score += (rating / 10) * 15
        else:
            score += (rating / 5) * 15
    if query.get("category"):
        cat = item.get("category", "")
        cat_str = "|".join(cat) if isinstance(cat, list) else str(cat) if cat else ""
        if query["category"].lower() in cat_str.lower():
            score += 10
    if query.get("mechanism"):
        mech = item.get("mechanism", "")
        mech_str = "|".join(mech) if isinstance(mech, list) else str(mech) if mech else ""
        if query["mechanism"].lower() in mech_str.lower():
            score += 8
    if query.get("type") and isinstance(item.get("type"), str):
        if query["type"] in item["type"]:
            score += 10
    if query.get("players"):
        rec = item.get("recommended_players") or item.get("best_players")
        if isinstance(rec, (int, float)) and rec == query["players"]:
            score += 5
    if query.get("weight_pref"):
        w = item.get("weight")
        if isinstance(w, (int, float)):
            if query["weight_pref"] == "light" and w < 2.5: score += 5
            elif query["weight_pref"] == "medium" and 2.5 <= w <= 3.5: score += 5
            elif query["weight_pref"] == "heavy" and w > 3.5: score += 5
    cr = item.get("category_rank")
    if cr:
        if isinstance(cr, dict):
            overall = cr.get("Overall")
        elif isinstance(cr, str):
            try:
                cr_dict = json.loads(cr.replace("'", '"'))
                overall = cr_dict.get("Overall") or cr_dict.get("전략") or cr_dict.get("가족")
            except:
                overall = None
        else:
            overall = None
        if isinstance(overall, (int, float)) and overall > 0:
            score += max(0, 10 - overall * 0.01)
    if item.get("source") == "boardlife":
        score *= 1.5
    return score

BM25_STOPWORDS = {
    "one","two","three","four","five","six","seven","eight","nine","ten",
    "a","an","the","of","in","at","to","for","and","or","is","it",
    "city","cities","world","age","game","games","edition","player","players"
}

def preprocess_query(text):
    tokens = text.lower().split()
    return " ".join([t for t in tokens if t not in BM25_STOPWORDS])

# -------------------------
# 검색 함수
# -------------------------
def bm25_search(query_text, query_filter, topk=200):
    tokenized_query = preprocess_query(query_text).split()
    scores = bm25.get_scores(tokenized_query)
    top_idx = np.argsort(scores)[::-1]
    results = {}
    rank = 0
    for idx in top_idx:
        if idx >= len(all_stats): continue
        item = all_stats[idx]
        if not hard_filter(item, query_filter): continue
        key = f"{item['source']}::{item.get('title', '')}"
        if key not in results:
            rank += 1
            results[key] = {"item": item, "rank": rank, "bm25_score": float(scores[idx])}
            if rank >= topk: break
    return results

def dense_search_single(index, meta, query_vector, query_filter, topk):
    if query_vector.shape[1] != index.d: return {}
    D, I = index.search(query_vector, topk * 3)
    results = {}
    rank = 0
    for i, idx in enumerate(I[0]):
        if idx < 0 or idx >= len(meta): continue
        item = meta[idx]
        if not hard_filter(item, query_filter): continue
        key = f"{item['source']}::{item.get('title', '')}"
        if key not in results:
            rank += 1
            results[key] = {"item": item, "rank": rank, "l2_dist": float(D[0][i])}
            if rank >= topk: break
    return results

def dense_search(query_vector, query_filter, topk=200):
    bgg_results = dense_search_single(bgg_stats_index, bgg_stats, query_vector, query_filter, topk)
    bl_results = dense_search_single(bl_stats_index, bl_stats, query_vector, query_filter, topk)
    combined = list({**bgg_results, **bl_results}.items())
    combined.sort(key=lambda x: x[1]["l2_dist"])
    results = {}
    for rank, (key, data) in enumerate(combined, 1):
        results[key] = data
        results[key]["rank"] = rank
        if rank >= topk: break
    return results

def rrf_search(query_vector, query_text, query_filter, topk=50, k=60):
    bm25_results = bm25_search(query_text, query_filter, topk=200)
    dense_results = dense_search(query_vector, query_filter, topk=200)
    all_keys = set(list(bm25_results.keys()) + list(dense_results.keys()))
    scored = []
    for key in all_keys:
        bm25_data = bm25_results.get(key)
        dense_data = dense_results.get(key)
        bm25_rank = bm25_data["rank"] if bm25_data else 999
        dense_rank = dense_data["rank"] if dense_data else 999
        if bm25_rank == 999 and dense_rank == 999: continue
        rrf_score = 1 / (k + bm25_rank) + 1 / (k + dense_rank)
        if dense_rank == 999 and bm25_rank != 999:
            rrf_score *= 0.7
        item = (bm25_data or dense_data)["item"]
        meta_score = metadata_weight(item, query_filter)
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

def get_average_embedding(titles, index=bgg_stats_index, meta=bgg_stats):
    embeddings = []
    for t in titles:
        for idx, s in enumerate(meta):
            if s["title"] == t:
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
    results = [item.copy() for item in all_stats if hard_filter(item, query_filter)]
    results.sort(key=lambda x: x.get("avg_rating") or 0, reverse=True)
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
        for item in all_stats
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
                title = results[i].get("title", "?")[:15] if i < len(results) else "-"
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
        "name": "2인 전략 보드게임 (무거운)",
        "query_filter": {"players": 2, "playing_time": 120, "category": "Strategy", "weight_pref": "heavy"},
        "query_text": "전략 strategy heavy economic two player 2인",
        "query_vector": get_average_embedding(["Brass: Birmingham", "Twilight Struggle", "7 Wonders Duel"]),
    },
    {
        "name": "4인 가벼운 파티게임",
        "query_filter": {"players": 4, "playing_time": 60, "category": "Party", "weight_pref": "light", "weight_max": 2.5},
        "query_text": "파티 party light fun word guessing 가벼운 파티게임",
        "query_vector": get_average_embedding(["Codenames", "Dixit", "Wavelength"]),
    },
    {
        "name": "2인 경제 게임",
        "query_filter": {"players": 2, "playing_time": 180, "category": "Economic", "mechanism": "Market", "weight_pref": "heavy"},
        "query_text": "경제 economic market engine building 엔진빌딩",
        "query_vector": get_average_embedding(["Brass: Birmingham", "Ark Nova", "Terraforming Mars"]),
    },
]

# -------------------------
# 실행
# -------------------------
if __name__ == "__main__":
    evaluate_all(queries, k=50)