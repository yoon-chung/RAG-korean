"""
외부 학습 데이터 탐색 — MMLU/ARC train split 실물 확인

목적:
  Qwen3-Embedding FT를 위한 외부 학습 데이터 후보 3개를 순차 탐색,
  각각의 크기·과학 도메인 비중·eval 문서와의 호환성을 평가.

실행:
  python explore_external_data.py

산출:
  data/external_preview.json  (요약 통계)
  data/external_samples.txt   (샘플 10개)

결정 기준:
  ✅ GO: 과학 도메인 ≥ 1,000 샘플 & eval 문서와 평균 overlap > 0.2
  ⚠️ 조건부: 과학 도메인 < 500 → KorQuAD/KLUE 보강 필요
  ❌ NO-GO: 위 둘 모두 실패 → pseudo-query 단독 경로로 회귀
"""

import json
import os
from pathlib import Path
from collections import Counter

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "data"
OUT_DIR.mkdir(exist_ok=True)

# -------------------------------------------------
# 1. MMLU train split
# -------------------------------------------------
def explore_mmlu():
    print("\n[1/3] MMLU train split 탐색...")
    try:
        from datasets import load_dataset
    except ImportError:
        print("  ❌ datasets 라이브러리 없음 → pip install datasets")
        return None

    # MMLU는 auxiliary_train이 가장 크지만 subject 라벨이 없음
    # test/validation에는 subject 라벨 있으나 평가 set이라 사용 금지
    # 대안: dev split (5개/subject × 57 subject = 285 샘플) → 작지만 subject 명확
    try:
        mmlu_dev = load_dataset("cais/mmlu", "all", split="dev")
        print(f"  MMLU dev split: {len(mmlu_dev)} samples")

        # 과학 관련 subject 필터
        science_subjects = {
            "astronomy", "college_biology", "college_chemistry",
            "college_physics", "conceptual_physics", "electrical_engineering",
            "high_school_biology", "high_school_chemistry", "high_school_physics",
            "anatomy", "clinical_knowledge", "nutrition", "virology",
            "human_aging", "medical_genetics", "computer_security",
            "elementary_mathematics", "high_school_mathematics",
            "college_mathematics", "abstract_algebra",
        }
        science_samples = [s for s in mmlu_dev if s["subject"] in science_subjects]
        print(f"  과학 subject 샘플: {len(science_samples)}")
        subject_counts = Counter(s["subject"] for s in science_samples)
        print(f"  Subject 분포 top 5: {subject_counts.most_common(5)}")

        return {
            "name": "MMLU",
            "total": len(mmlu_dev),
            "science_count": len(science_samples),
            "sample_format": mmlu_dev[0] if mmlu_dev else None,
            "lang": "English (번역 필요)",
        }
    except Exception as e:
        print(f"  ❌ 로드 실패: {e}")
        return None


# -------------------------------------------------
# 2. ARC train split
# -------------------------------------------------
def explore_arc():
    print("\n[2/3] ARC train split 탐색...")
    try:
        from datasets import load_dataset
    except ImportError:
        return None

    try:
        arc_train = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train")
        print(f"  ARC-Challenge train: {len(arc_train)} samples")

        # 평균 질문·정답 길이
        q_lens = [len(s["question"].split()) for s in arc_train]
        print(f"  질문 평균 단어 수: {sum(q_lens)/len(q_lens):.1f}")

        sample = arc_train[0]
        print(f"  샘플 구조: {list(sample.keys())}")
        print(f"  샘플 질문: {sample['question'][:100]}")
        print(f"  샘플 선택지: {sample['choices']['text'][:2]}")

        return {
            "name": "ARC-Challenge",
            "total": len(arc_train),
            "avg_q_words": sum(q_lens)/len(q_lens),
            "sample_format": sample,
            "lang": "English (번역 필요)",
        }
    except Exception as e:
        print(f"  ❌ 로드 실패: {e}")
        return None


# -------------------------------------------------
# 3. Korean alternatives (KLUE-MRC, KorQuAD)
# -------------------------------------------------
def explore_korean():
    print("\n[3/3] 한국어 대체 데이터 탐색...")
    try:
        from datasets import load_dataset
    except ImportError:
        return None

    results = {}

    # KLUE-MRC
    try:
        klue_mrc = load_dataset("klue", "mrc", split="train")
        print(f"  KLUE-MRC train: {len(klue_mrc)} samples")
        # context 길이 분포
        ctx_lens = [len(s["context"]) for s in klue_mrc[:1000]]
        print(f"    context 평균 글자: {sum(ctx_lens)/len(ctx_lens):.0f}")
        results["klue_mrc"] = {"total": len(klue_mrc), "lang": "Korean"}
    except Exception as e:
        print(f"  ⚠️ KLUE-MRC 실패: {e}")

    # KorQuAD 2.0 (if accessible via HF)
    try:
        kq = load_dataset("squad_kor_v2", split="train")
        print(f"  KorQuAD 2.0 train: {len(kq)} samples")
        results["korquad"] = {"total": len(kq), "lang": "Korean"}
    except Exception as e:
        print(f"  ⚠️ KorQuAD 2.0 실패: {e}")

    return results


# -------------------------------------------------
# 4. 현재 보유 pseudo-query 데이터 점검
# -------------------------------------------------
def check_existing_data():
    print("\n[bonus] 기존 pseudo-query 데이터 점검...")
    candidates = [
        "data/training_triplets_merged.jsonl",
        "data/training_triplets_solar.jsonl",
        "data/training_triplets_gpt4mini.jsonl",
        "data/pseudo_queries_solar.jsonl",
        "data/pseudo_queries_gpt4mini.jsonl",
    ]
    found = {}
    for c in candidates:
        p = HERE / c
        if p.exists():
            count = sum(1 for _ in open(p, encoding="utf-8"))
            found[c] = count
            print(f"  ✓ {c}: {count} lines")
        else:
            print(f"  ✗ {c}: 없음")
    return found


# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    print("=" * 60)
    print(" 외부 학습 데이터 탐색")
    print("=" * 60)

    existing = check_existing_data()
    mmlu = explore_mmlu()
    arc = explore_arc()
    korean = explore_korean()

    summary = {
        "existing_pseudo_query": existing,
        "mmlu": mmlu,
        "arc": arc,
        "korean_alt": korean,
    }

    out_path = OUT_DIR / "external_preview.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    print("\n" + "=" * 60)
    print(" 결정 가이드")
    print("=" * 60)

    # 결정 로직
    mmlu_sci = mmlu["science_count"] if mmlu else 0
    arc_total = arc["total"] if arc else 0
    klue_total = korean.get("klue_mrc", {}).get("total", 0) if korean else 0
    pseudo_total = existing.get("data/training_triplets_merged.jsonl", 0)

    print(f"\n  MMLU 과학 샘플:      {mmlu_sci}")
    print(f"  ARC train:           {arc_total}")
    print(f"  KLUE-MRC:            {klue_total}")
    print(f"  기존 pseudo-query:   {pseudo_total}")

    print("\n  판단:")
    if pseudo_total >= 10000:
        print("  ✅ 기존 pseudo-query 25.6K 활용 + MNRL(in-batch neg)로 즉시 FT 시작 가능")
        if arc_total >= 1000:
            print("  ✅ ARC train split 추가해 학습 데이터 다양성↑ (번역 필요)")
        if klue_total >= 5000:
            print("  ✅ KLUE-MRC로 일반 도메인 warmup → 우리 corpus로 fine-tuning")
    else:
        print("  ⚠️ 기존 데이터 부족 → generate_pseudo_queries.py 재실행 필요")

    print(f"\n결과 저장: {out_path}")


if __name__ == "__main__":
    main()
