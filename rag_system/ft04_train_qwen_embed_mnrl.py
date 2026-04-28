"""
Qwen3-Embedding-0.6B FT — MultipleNegativesRankingLoss (MNRL)

목적:
  외부 GT(ARC/MMLU) + 기존 pseudo-query 을 함께 써서 한국어 과학·지식 도메인에
  특화된 embedding 학습. In-batch negative 방식이라 hard negative mining 불필요.

데이터 구성:
  Primary: external_gt_matched.jsonl (진짜 GT → corpus 매칭, ~2,000~2,500 pairs)
  Optional: training_triplets_merged.jsonl (pseudo-query, 25.6K)

학습:
  - sentence-transformers SentenceTransformer
  - Loss: MultipleNegativesRankingLoss (in-batch neg + hard neg 옵션)
  - Model: Qwen/Qwen3-Embedding-0.6B (full FT, 3090 24GB fit)
  - Hyperparameter:
      batch_size=64, epochs=3, lr=2e-5, warmup 10%

출력:
  ../models/qwen3-embed-ft/ (저장된 SentenceTransformer 모델)

nohup 실행:
  nohup python train_qwen_embed_mnrl.py > logs/train_qwen_embed.log 2>&1 &
"""

import os
import sys
import json
import random
import argparse
from pathlib import Path
from datetime import datetime

# nohup redirect 상황에서 stdout 즉시 flush (버퍼링 방지)
sys.stdout.reconfigure(line_buffering=True)

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "data"
MODELS_DIR = HERE.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR = HERE / "logs" / "finetune_qwen_embed"
LOGS_DIR.mkdir(parents=True, exist_ok=True)


def load_jsonl(path):
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))
    return items


def build_training_examples(include_pseudo=True):
    """InputExample 리스트 생성.

    MNRL은 positive-only pair만 필요 (InputExample(texts=[q, pos]))
    In-batch negative로 다른 샘플을 자동으로 negative 취급
    """
    from sentence_transformers import InputExample

    examples = []

    # 1. External GT matched pairs (고품질)
    matched_path = DATA_DIR / "external_gt_matched.jsonl"
    if matched_path.exists():
        matched = load_jsonl(matched_path)
        for m in matched:
            examples.append(InputExample(texts=[m["query"], m["positive_content"]]))
        print(f"  External GT matched: {len(matched)} pairs")
    else:
        print(f"  ⚠️ external_gt_matched.jsonl 없음 (match_gt_to_corpus.py 먼저 실행)")

    # 2. Pseudo-query triplets (bulk)
    if include_pseudo:
        pseudo_path = DATA_DIR / "training_triplets_merged.jsonl"
        if pseudo_path.exists():
            pseudo = load_jsonl(pseudo_path)
            pseudo_count = 0
            for p in pseudo:
                query = p.get("query", "").strip()
                pos = (p.get("positive_content") or "").strip()
                if query and pos and len(query) >= 5 and len(pos) >= 20:
                    examples.append(InputExample(texts=[query, pos]))
                    pseudo_count += 1
            print(f"  Pseudo-query pairs: {pseudo_count}")

    random.shuffle(examples)
    return examples


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default="Qwen/Qwen3-Embedding-0.6B")
    p.add_argument("--output-dir", default=str(MODELS_DIR / "qwen3-embed-ft"))
    p.add_argument("--epochs", type=int, default=3)
    # batch 16 + seq 256: Qwen3-0.6B full FT in 3090 24GB safe (~6GB usage)
    # batch 64 / seq 512는 OOM 위험. 1,297 pair에 batch 16도 충분 (MNRL 작동)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--max-seq-length", type=int, default=256)
    p.add_argument("--no-pseudo", action="store_true", help="pseudo-query 데이터 제외")
    p.add_argument("--val-ratio", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)

    print("=" * 60)
    print(" Qwen3-Embedding-0.6B FT (MNRL)")
    print("=" * 60)

    # --- 데이터 준비 ---
    print("\n[1] 학습 데이터 준비")
    all_examples = build_training_examples(include_pseudo=not args.no_pseudo)
    print(f"  총 pair     : {len(all_examples)}")

    val_size = int(len(all_examples) * args.val_ratio)
    train_examples = all_examples[val_size:]
    val_examples = all_examples[:val_size]
    print(f"  train       : {len(train_examples)}")
    print(f"  val         : {len(val_examples)}")

    # --- 모델 로드 ---
    print(f"\n[2] Base model 로드: {args.base_model}")
    from sentence_transformers import SentenceTransformer, losses
    from torch.utils.data import DataLoader

    model = SentenceTransformer(args.base_model)
    model.max_seq_length = args.max_seq_length
    print(f"  max_seq_length: {model.max_seq_length}")

    # --- Training setup ---
    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=args.batch_size,
    )
    total_steps = len(train_dataloader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    loss = losses.MultipleNegativesRankingLoss(model)

    print(f"\n[3] Training config")
    print(f"  epochs      : {args.epochs}")
    print(f"  batch_size  : {args.batch_size}")
    print(f"  lr          : {args.lr}")
    print(f"  total_steps : {total_steps}")
    print(f"  warmup_steps: {warmup_steps}")
    print(f"  output_dir  : {args.output_dir}")

    # --- Evaluator (validation 추적) ---
    evaluator = None
    if val_examples:
        from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
        # MNRL 학습에는 similarity evaluator가 직접 맞진 않지만,
        # "query ↔ positive"의 유사도가 얼마나 높아지는지 측정
        val_queries = [e.texts[0] for e in val_examples]
        val_positives = [e.texts[1] for e in val_examples]
        val_labels = [1.0] * len(val_examples)
        evaluator = EmbeddingSimilarityEvaluator(
            sentences1=val_queries,
            sentences2=val_positives,
            scores=val_labels,
            name="val_qp",
            show_progress_bar=False,
        )

    # --- Train ---
    print(f"\n[4] Training 시작 — {datetime.now()}")
    start = datetime.now()

    model.fit(
        train_objectives=[(train_dataloader, loss)],
        evaluator=evaluator,
        epochs=args.epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": args.lr},
        output_path=args.output_dir,
        show_progress_bar=True,
        evaluation_steps=1000,
        save_best_model=True,
    )

    duration = (datetime.now() - start).total_seconds()
    print(f"\n[5] 완료 — {duration/60:.1f}분 소요")

    # --- Metadata 저장 ---
    meta = {
        "base_model": args.base_model,
        "train_pairs": len(train_examples),
        "val_pairs": len(val_examples),
        "include_pseudo": not args.no_pseudo,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "warmup_ratio": args.warmup_ratio,
        "duration_sec": round(duration, 2),
        "timestamp": datetime.now().isoformat(),
    }
    meta_path = Path(args.output_dir) / "ft_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\n  모델: {args.output_dir}")
    print(f"  메타: {meta_path}")
    print(f"\n다음 단계: rag_exp-09-001 실행으로 로컬 평가")
    print(f"  python rag.py eval --exp EXP-09-001-qwen3-embed-ft \\")
    print(f"      --extra-embed-model {args.output_dir} \\")
    print(f"      --rrf-weight-extra 1.0 ...")


if __name__ == "__main__":
    main()
