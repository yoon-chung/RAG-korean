"""
Submission ensemble via Reciprocal Rank Fusion (RRF)

여러 submission CSV의 top-k를 RRF로 통합해 새 submission 생성.
multi-query 비결정성 평균화 + 서로 다른 setup의 강점 결합.

## 사용법
# 기본 (균등 가중치, 과반수가 empty면 empty)
python ensemble_submissions.py \
    --inputs logs/EXP-14/submission.csv logs/EXP-17/submission.csv logs/EXP-18/submission.csv \
    --output logs/ENSEMBLE-001/submission.csv

# 특정 submission에 가중치 (EXP-14 더 신뢰)
python ensemble_submissions.py \
    --inputs A.csv B.csv C.csv \
    --weights 1.5 1.0 1.0 \
    --output ENS.csv

# Chitchat vote threshold 직접 지정 (예: 3개 중 1개라도 empty면 chitchat)
python ensemble_submissions.py \
    --inputs A.csv B.csv C.csv \
    --chitchat-vote 1 \
    --output ENS.csv
"""

import argparse
import json
import datetime as _dt
from pathlib import Path
from collections import defaultdict
from typing import List, Dict


def load_submission(path: str) -> Dict[int, dict]:
    """Returns dict[eval_id] = full record dict."""
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            out[d["eval_id"]] = d
    return out


def ensemble(
    submissions: List[Dict[int, dict]],
    weights: List[float],
    rrf_k: int,
    top_k: int,
    chitchat_vote: int,
    rank1_plurality: bool = False,
) -> List[dict]:
    """RRF fusion across submissions.

    chitchat_vote: empty topk를 가진 submission이 이 수 이상이면 출력도 empty.
                   (e.g., 3개 submission, vote=2면 2개 이상이 empty여야 empty 출력)
    rank1_plurality: True면 top-1은 "rank-1 위치 가중 plurality vote", rank 2~3은 RRF.
                     소수 의견이 정답인 케이스(예: 한 sub만 GT를 rank 1에 둠)를 보존.
                     tie면 RRF top-1로 fallback.
    """
    all_eids = set()
    for s in submissions:
        all_eids.update(s.keys())

    out_records = []
    for eid in sorted(all_eids):
        records = [s.get(eid) for s in submissions]
        present = [(i, r) for i, r in enumerate(records) if r is not None]
        if not present:
            continue

        empty_count = sum(1 for _, r in present if not r.get("topk"))
        is_chitchat = empty_count >= chitchat_vote

        # 메타데이터는 첫 번째 submission 것을 따름 (sq, answer 등)
        base = present[0][1]
        new_record = {
            "eval_id": eid,
            "standalone_query": base.get("standalone_query", ""),
            "topk": [],
            "references": base.get("references", []),
            "answer": base.get("answer", ""),
        }

        if is_chitchat:
            new_record["topk"] = []
            new_record["references"] = []
        else:
            doc_scores = defaultdict(float)
            for i, r in present:
                topk = r.get("topk") or []
                if not topk:
                    continue
                w = weights[i] if i < len(weights) else 1.0
                for rank, docid in enumerate(topk):
                    doc_scores[docid] += w / (rrf_k + rank + 1)
            ranked = sorted(doc_scores.items(), key=lambda x: -x[1])

            if rank1_plurality:
                # rank-1 위치 가중 plurality
                r1_votes = defaultdict(float)
                for i, r in present:
                    topk = r.get("topk") or []
                    if not topk:
                        continue
                    w = weights[i] if i < len(weights) else 1.0
                    r1_votes[topk[0]] += w
                if r1_votes:
                    sorted_r1 = sorted(r1_votes.items(), key=lambda x: -x[1])
                    top_vote = sorted_r1[0][1]
                    tied = [d for d, v in sorted_r1 if v == top_vote]
                    if len(tied) == 1:
                        top1_doc = tied[0]
                    else:
                        # tie면 RRF top-1으로 결정
                        top1_doc = next((d for d, _ in ranked if d in tied), tied[0])
                    # rank 2~3은 RRF top-K에서 top1_doc 제외
                    rest = [d for d, _ in ranked if d != top1_doc][: top_k - 1]
                    new_record["topk"] = [top1_doc] + rest
                else:
                    new_record["topk"] = [d for d, _ in ranked[:top_k]]
            else:
                new_record["topk"] = [d for d, _ in ranked[:top_k]]

        out_records.append(new_record)

    return out_records


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inputs", nargs="+", required=True, help="submission CSV 경로들 (jsonl 포맷)")
    p.add_argument("--output", required=True, help="출력 CSV 경로")
    p.add_argument("--weights", nargs="+", type=float, default=None,
                   help="submission별 가중치 (default: 모두 1.0)")
    p.add_argument("--rrf-k", type=int, default=60, help="RRF k 상수 (default 60)")
    p.add_argument("--top-k", type=int, default=3, help="출력 top-k (default 3)")
    p.add_argument("--chitchat-vote", type=int, default=None,
                   help="empty 표시 vote 임계값 (default: strict majority = N//2 + 1)")
    p.add_argument("--rank1-plurality", action="store_true",
                   help="top-1은 rank-1 plurality vote (소수 의견 보존), rank 2-3은 RRF")
    args = p.parse_args()

    n = len(args.inputs)
    weights = args.weights or [1.0] * n
    if len(weights) != n:
        raise ValueError(f"weights 길이 {len(weights)} != inputs 길이 {n}")

    # strict majority: N=2면 2(both), N=3면 2, N=5면 3
    chitchat_vote = args.chitchat_vote if args.chitchat_vote is not None else (n // 2 + 1)

    submissions = [load_submission(p) for p in args.inputs]
    print(f"Loaded {n} submissions (chitchat_vote={chitchat_vote}, rrf_k={args.rrf_k}):")
    for i, (path, sub) in enumerate(zip(args.inputs, submissions)):
        empty = sum(1 for r in sub.values() if not r.get("topk"))
        print(f"  [{i}] w={weights[i]}  {path}  ({len(sub)} ids, {empty} empty)")

    out = ensemble(submissions, weights, args.rrf_k, args.top_k, chitchat_vote, args.rank1_plurality)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    empty_out = sum(1 for r in out if not r["topk"])
    print(f"\nOutput: {out_path}")
    print(f"  total: {len(out)} / empty: {empty_out} / non-empty: {len(out) - empty_out}")

    # Config dump — 재현성 및 추적용 (rag.py 실험 구조와 동일하게)
    config = {
        "mode": "ensemble",
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "output": str(out_path),
        "rrf_k": args.rrf_k,
        "top_k": args.top_k,
        "chitchat_vote": chitchat_vote,
        "rank1_plurality": args.rank1_plurality,
        "n_inputs": n,
        "inputs": [
            {
                "index": i,
                "path": args.inputs[i],
                "basename": Path(args.inputs[i]).name,
                "weight": weights[i],
                "total": len(submissions[i]),
                "empty": sum(1 for r in submissions[i].values() if not r.get("topk")),
            }
            for i in range(n)
        ],
        "output_stats": {
            "total": len(out),
            "empty": empty_out,
            "non_empty": len(out) - empty_out,
        },
    }
    config_path = out_path.parent / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print(f"  config: {config_path}")

    # 입력 대비 변화 요약 (첫 번째 submission 기준)
    if submissions:
        base = submissions[0]
        identical = 0
        diff_set = 0
        empty_changed = 0
        for r in out:
            b = base.get(r["eval_id"])
            if b is None:
                continue
            base_topk = b.get("topk") or []
            new_topk = r["topk"]
            if base_topk == new_topk:
                identical += 1
            elif (not base_topk) != (not new_topk):
                empty_changed += 1
            else:
                diff_set += 1
        print(f"\n  vs [{0}] {args.inputs[0]}:")
        print(f"    identical: {identical}")
        print(f"    diff topk: {diff_set}")
        print(f"    empty/non-empty changed: {empty_changed}")


if __name__ == "__main__":
    main()
