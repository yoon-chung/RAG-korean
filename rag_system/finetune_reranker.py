"""
Reranker Fine-tuning - bge-reranker-v2-m3 도메인 적응

사용법:
    # 기본 실행 (bge-reranker-v2-m3 fine-tuning)
    python finetune_reranker.py

    # 커스텀 설정
    python finetune_reranker.py \
        --base-model BAAI/bge-reranker-v2-m3 \
        --epochs 3 \
        --batch-size 16 \
        --lr 2e-5 \
        --val-ratio 0.1

    # 학습 후 평가
    python rag.py eval --exp EXP-05-FT-reranker \
        --reranker-model models/finetuned-reranker \
        --classifier none --score-threshold 0.05 --hyde \
        --rrf-weight-sparse 0.3 --rrf-weight-dense 0.7

산출물:
    models/finetuned-reranker/     (fine-tuned CrossEncoder 모델)
    logs/finetune_reranker/        (학습 로그)
"""

import os
import json
import random
import argparse
from pathlib import Path
from datetime import datetime

from sentence_transformers import CrossEncoder, InputExample
from torch.utils.data import DataLoader


HERE = Path(__file__).resolve().parent


def load_triplets(triplets_path: str, max_negatives: int = 3):
    """training_triplets.jsonl → InputExample 리스트"""
    examples = []
    with open(triplets_path) as f:
        for line in f:
            row = json.loads(line)
            query = row["query"]
            pos_content = row["positive_content"]

            # positive pair
            examples.append(InputExample(texts=[query, pos_content], label=1.0))

            # hard negative pairs
            neg_contents = row.get("hard_negative_contents", [])
            for neg in neg_contents[:max_negatives]:
                examples.append(InputExample(texts=[query, neg], label=0.0))

    return examples


def main():
    p = argparse.ArgumentParser(description="Fine-tune CrossEncoder Reranker")
    p.add_argument("--base-model", default="BAAI/bge-reranker-v2-m3")
    p.add_argument("--triplets-path", default="../data/training_triplets.jsonl")
    p.add_argument("--output-dir", default="models/finetuned-reranker")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--val-ratio", type=float, default=0.1,
                   help="Validation set ratio (default: 0.1)")
    p.add_argument("--max-negatives", type=int, default=3,
                   help="Max hard negatives per query (default: 3)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)

    # --- Load data ---
    print(f"[INFO] Loading triplets from {args.triplets_path}")
    all_examples = load_triplets(args.triplets_path, args.max_negatives)
    random.shuffle(all_examples)

    val_size = int(len(all_examples) * args.val_ratio)
    train_examples = all_examples[val_size:]
    val_examples = all_examples[:val_size]

    pos_count = sum(1 for e in train_examples if e.label == 1.0)
    neg_count = len(train_examples) - pos_count

    print(f"[INFO] Train: {len(train_examples)} examples (pos={pos_count}, neg={neg_count})")
    print(f"[INFO] Val:   {len(val_examples)} examples")

    # --- Load model ---
    print(f"\n[INFO] Loading base model: {args.base_model}")
    model = CrossEncoder(
        args.base_model,
        max_length=args.max_length,
        num_labels=1,
    )

    # --- Train ---
    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=args.batch_size,
    )

    total_steps = len(train_dataloader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    print(f"\n[INFO] Training config:")
    print(f"  Base model:    {args.base_model}")
    print(f"  Epochs:        {args.epochs}")
    print(f"  Batch size:    {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Total steps:   {total_steps}")
    print(f"  Warmup steps:  {warmup_steps}")
    print(f"  Max length:    {args.max_length}")
    print(f"  Output:        {args.output_dir}")

    log_dir = HERE / "logs" / "finetune_reranker"
    log_dir.mkdir(parents=True, exist_ok=True)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print(f"\n[INFO] Starting training...")
    start_time = datetime.now()

    model.fit(
        train_dataloader=train_dataloader,
        epochs=args.epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": args.lr},
        output_path=args.output_dir,
        show_progress_bar=True,
    )

    duration = (datetime.now() - start_time).total_seconds()
    print(f"\n[INFO] Training completed in {duration:.1f}s ({duration/60:.1f}min)")

    # 명시적 모델 저장
    model.save(args.output_dir)
    print(f"[INFO] Model saved to {args.output_dir}")

    # --- Validate ---
    if val_examples:
        print(f"\n[INFO] Evaluating on validation set ({len(val_examples)} examples)...")
        val_pairs = [e.texts for e in val_examples]
        val_labels = [e.label for e in val_examples]
        val_scores = model.predict(val_pairs, show_progress_bar=True)

        # Binary accuracy
        correct = sum(
            1 for score, label in zip(val_scores, val_labels)
            if (score > 0.5) == (label > 0.5)
        )
        accuracy = correct / len(val_labels)

        # Positive/negative score distribution
        pos_scores = [s for s, l in zip(val_scores, val_labels) if l > 0.5]
        neg_scores = [s for s, l in zip(val_scores, val_labels) if l <= 0.5]

        print(f"\n[VALIDATION RESULTS]")
        print(f"  Accuracy:      {accuracy:.4f} ({correct}/{len(val_labels)})")
        print(f"  Pos avg score: {sum(pos_scores)/len(pos_scores):.4f}" if pos_scores else "")
        print(f"  Neg avg score: {sum(neg_scores)/len(neg_scores):.4f}" if neg_scores else "")
        print(f"  Score gap:     {sum(pos_scores)/len(pos_scores) - sum(neg_scores)/len(neg_scores):.4f}"
              if pos_scores and neg_scores else "")

    # --- Save metadata ---
    meta = {
        "base_model": args.base_model,
        "triplets_path": args.triplets_path,
        "train_size": len(train_examples),
        "val_size": len(val_examples),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "max_length": args.max_length,
        "max_negatives": args.max_negatives,
        "duration_sec": round(duration, 2),
        "timestamp": datetime.now().isoformat(),
    }
    if val_examples:
        meta["val_accuracy"] = round(accuracy, 4)

    meta_path = Path(args.output_dir) / "finetune_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\n[INFO] Model saved to: {args.output_dir}")
    print(f"[INFO] Metadata saved to: {meta_path}")
    print(f"\n[NEXT STEP] Evaluate with fine-tuned reranker:")
    print(f"  python rag.py eval --exp EXP-05-FT-reranker \\")
    print(f"      --reranker-model {args.output_dir} \\")
    print(f"      --classifier none --score-threshold 0.05 --hyde \\")
    print(f"      --rrf-weight-sparse 0.3 --rrf-weight-dense 0.7")


if __name__ == "__main__":
    main()
