"""
분석 스크립트: classifier=none baseline (MAP 0.8318) CSV의 rerank top-1 score 분포를 보고
classifier=none + score threshold 필터링의 최적 τ 후보 탐색.

목적: "사후 필터"로 분류기를 대체할 때, score < τ 인 경우 topk=[] 처리.
진짜 잡담(261, 276 등)은 낮은 score를 가지므로 자연스럽게 걸러짐.

Usage:
    python rag_system/analyze_score_threshold.py
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CSV_PATH = HERE / "logs" / "reference" / "output_08318.csv"


def load(path: Path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def classify_row(r):
    topk = r.get("topk", [])
    refs = r.get("references", [])
    top1_score = refs[0]["score"] if refs else None
    answer = r.get("answer", "")
    unanswerable = "제공된 정보로는 답변할 수 없습니다" in answer
    return {
        "eval_id": r["eval_id"],
        "has_topk": bool(topk),
        "top1_score": top1_score,
        "unanswerable": unanswerable,
        "query": r.get("standalone_query", "")[:40],
    }


def main():
    rows = load(CSV_PATH)
    print(f"총 {len(rows)}개 질의")

    analyzed = [classify_row(r) for r in rows]

    filled = [a for a in analyzed if a["has_topk"]]
    empty = [a for a in analyzed if not a["has_topk"]]
    print(f"  topk 채워짐: {len(filled)}, topk 빈 것: {len(empty)}")

    scored = [a for a in filled if a["top1_score"] is not None]
    scores = sorted([a["top1_score"] for a in scored])

    buckets = [(0, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)]
    print("\n[top-1 score 분포]")
    for lo, hi in buckets:
        n = sum(1 for s in scores if lo <= s < hi)
        bar = "#" * n
        print(f"  {lo:.2f}~{hi:.2f}: {n:3d} {bar}")

    print("\n[τ 후보별 topk=[] 처리될 질의 수]")
    for tau in [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]:
        cut = [a for a in scored if a["top1_score"] < tau]
        print(f"  τ={tau}: {len(cut):3d}개 cut")

    print("\n[경계 구간 상세: top-1 score < 0.3 인 질의 전체]")
    borderline = sorted([a for a in scored if a["top1_score"] < 0.3], key=lambda x: x["top1_score"])
    for a in borderline:
        marker = "❓" if a["unanswerable"] else "  "
        print(f"  {marker} id={a['eval_id']:3d}  score={a['top1_score']:.4f}  {a['query']}")

    unans = [a for a in filled if a["unanswerable"]]
    print(f"\n[답변 불가 케이스 ({len(unans)}개)] - retrieval 실패. threshold로 잘못 자르면 손해")
    for a in unans:
        s = a["top1_score"]
        print(f"  id={a['eval_id']:3d}  top1_score={s:.4f}  {a['query']}")


if __name__ == "__main__":
    main()
