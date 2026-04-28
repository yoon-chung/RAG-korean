"""
외부 GT 데이터 수집 — ARC + MMLU validation

목적:
  pseudo-query가 아닌 "사람이 직접 만든 Q-A 쌍" 확보.
  corpus (ko_mmlu + ko_ai2_arc)와 같은 출처의 train/validation split 활용.

출력:
  ../data/external_gt_en.jsonl
    {"source": "arc"/"mmlu", "id": ..., "subject": ...,
     "question": ..., "choices": [...], "correct_choice": "..."}

주의:
  - MMLU는 validation(1,531) + dev(285) 활용. test는 우리 eval corpus 출처이므로 절대 사용 금지.
  - ARC는 train(1,119) 활용. test는 eval corpus 출처.
  - 과학 subject로 필터하지 않음. 우리 corpus가 "과학 + 사회·역사·인물·생활상식" 혼합이기 때문.
  - 명백히 우리 corpus에 없는 "미국 실무" 2개만 제외.
  - 나머지는 ft03_match_gt_to_corpus 단계의 rerank score로 자연스럽게 필터링.
"""

import json
from pathlib import Path
from datasets import load_dataset
from collections import Counter

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# MMLU에서 명백히 우리 corpus에 없는 "미국 실무" subject만 최소한으로 제외
# 나머지는 matching score로 자연 필터 (과도한 사전 제외는 있을 수도 있는 매칭을 잃음)
MMLU_EXCLUDE = {
    "professional_accounting",  # 미국 회계 실무
    "professional_law",         # 미국 법률 실무
}


def collect_arc():
    print("[ARC] Downloading ARC-Challenge train...")
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train")
    items = []
    for row in ds:
        # answerKey → correct choice text 매핑
        labels = row["choices"]["label"]   # ['A','B','C','D'] or ['1','2','3','4']
        texts = row["choices"]["text"]
        try:
            idx = labels.index(row["answerKey"])
            correct_text = texts[idx]
        except (ValueError, IndexError):
            continue

        items.append({
            "source": "arc",
            "id": row["id"],
            "subject": "science_reasoning",
            "question": row["question"],
            "choices": texts,
            "correct_choice": correct_text,
        })
    print(f"[ARC] collected {len(items)} items")
    return items


def collect_mmlu():
    print("[MMLU] Downloading validation + dev split...")
    items = []
    for split in ["validation", "dev"]:
        ds = load_dataset("cais/mmlu", "all", split=split)
        for i, row in enumerate(ds):
            subject = row["subject"]
            if subject in MMLU_EXCLUDE:
                continue

            # MMLU: choices는 list 4개, answer는 int (0-3)
            choices = row["choices"]
            ans_idx = row["answer"]
            if not (0 <= ans_idx < len(choices)):
                continue

            items.append({
                "source": "mmlu",
                "id": f"mmlu_{split}_{i}",
                "subject": subject,
                "question": row["question"],
                "choices": choices,
                "correct_choice": choices[ans_idx],
            })
    print(f"[MMLU] collected {len(items)} items (exclude {len(MMLU_EXCLUDE)} subjects)")
    # Subject 분포
    subj_count = Counter(item["subject"] for item in items)
    print(f"[MMLU] top 10 subjects: {subj_count.most_common(10)}")
    return items


def main():
    print("=" * 60)
    print(" 외부 GT 수집: ARC + MMLU")
    print("=" * 60)

    arc = collect_arc()
    mmlu = collect_mmlu()
    all_items = arc + mmlu

    # 저장
    out_path = DATA_DIR / "external_gt_en.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for item in all_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("\n" + "=" * 60)
    print(" 수집 완료")
    print("=" * 60)
    print(f"  ARC   : {len(arc)}")
    print(f"  MMLU  : {len(mmlu)}")
    print(f"  총     : {len(all_items)}")
    print(f"  저장    : {out_path}")
    print(f"\n다음 단계: python ft02_translate_gt_to_korean.py")


if __name__ == "__main__":
    main()
