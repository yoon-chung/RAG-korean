"""
EXP-09-001 — Qwen3-Embedding FT + 3-way RRF Ensemble

실험 가설:
  외부 GT (ARC train + MMLU validation) 로 FT한 Qwen3-Embedding-0.6B를
  extra dense retriever로 탑재하면, 기존 bge-m3 단독 대비 ensemble 다양성이
  증가해 LB가 개선될 것.

파이프라인 구성:
  Sparse: BM25 (Elasticsearch + Nori)
  Dense : bge-m3 (ES KNN, 1024 dim)          ← 기존 primary 유지
  Extra : Qwen3-Embedding-0.6B FT (FAISS, 1024 dim)  ← 신규 추가
  RRF fuse (k=60, sparse 0.3 / dense 0.7 / extra 1.0)
  → top 20 → bge-reranker-v2-m3 → top 3

선행 조건:
  1. ft_run_pipeline.sh 완료 → models/qwen3-embed-ft/ 생성
  2. 본 스크립트 실행 전 FAISS 인덱스 구축:
        python rag.py index \\
            --embed-model models/qwen3-embed-ft \\
            --use-faiss \\
            --exp EXP-09-001-build-faiss

실행 (평가만):
  python rag_exp-09-001-qwen3-embed-ft.py eval \\
      --exp EXP-09-001-qwen3-embed-ft \\
      --extra-embed-model models/qwen3-embed-ft

비교 baseline:
  현재 최고점 0.9379 (4-way ensemble + id=104 후처리).
  단독 pipeline 기준으로는 tau002 0.9258, EXP-14 0.9348 비교.

기본 config 고정 (CLI로 override 가능):
  classifier none | score-threshold 0.02 | hyde on
  rrf-weight-sparse 0.3 | rrf-weight-dense 0.7 | rrf-weight-extra 1.0
  multi-query n=3 | llm-cache on | force-empty-sq-pattern on
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


# EXP-09-001 고정 default config
# CLI에 명시하지 않았을 때만 보강
DEFAULTS = [
    ("--classifier", "none"),
    ("--score-threshold", "0.02"),
    ("--hyde", None),
    ("--rrf-weight-sparse", "0.3"),
    ("--rrf-weight-dense", "0.7"),
    ("--rrf-weight-extra", "1.0"),
    ("--multi-query", None),
    ("--multi-query-n", "3"),
    ("--llm-cache", None),
    ("--force-empty-sq-pattern", None),
]


def inject_defaults(argv):
    """CLI에 없는 DEFAULTS만 보강. value 있는 flag와 boolean flag 모두 처리."""
    existing = {arg for arg in argv if arg.startswith("--")}
    out = list(argv)
    for flag, val in DEFAULTS:
        if flag in existing:
            continue
        out.append(flag)
        if val is not None:
            out.append(val)
    return out


def sanity_check():
    """EXP-09-001 실행 전 필수 조건 확인"""
    ft_model_path = HERE.parent / "models" / "qwen3-embed-ft"
    if not ft_model_path.exists():
        print(f"❌ FT 모델 없음: {ft_model_path}")
        print("   ft_run_pipeline.sh 또는 ft_run_step34.sh 먼저 실행 필요")
        sys.exit(1)

    # FAISS 인덱스 존재 여부 (extra model 경로 기반)
    faiss_dir = HERE / "faiss_indices"
    expected_pattern = "*qwen3_embed_ft*.faiss"
    faiss_files = list(faiss_dir.glob(expected_pattern)) if faiss_dir.exists() else []
    if not faiss_files:
        print(f"⚠️ FAISS 인덱스가 없습니다.")
        print(f"   먼저 다음 명령으로 인덱싱하세요:")
        print(f"   python rag.py index \\")
        print(f"       --embed-model models/qwen3-embed-ft \\")
        print(f"       --use-faiss \\")
        print(f"       --exp EXP-09-001-build-faiss")
        sys.exit(1)

    print(f"  ✓ FT 모델 확인 : {ft_model_path}")
    print(f"  ✓ FAISS 인덱스 : {faiss_files[0].name}")


if __name__ == "__main__":
    print("=" * 60)
    print(" EXP-09-001 — Qwen3-Embedding FT + 3-way Ensemble")
    print("=" * 60)

    # 사전 조건 확인 (eval/all 모드만)
    if len(sys.argv) > 1 and sys.argv[1] in ("eval", "all"):
        sanity_check()

    # Default config 주입
    original_argv = sys.argv[1:]
    sys.argv = [sys.argv[0]] + inject_defaults(original_argv)

    print(f"\n  최종 실행 config:")
    print(f"    {' '.join(sys.argv[1:])}\n")

    # rag.py의 main() 호출
    from rag import main
    main()
