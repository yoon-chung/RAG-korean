#!/bin/bash
# Step 3 (매칭) + Step 4 (FT 학습) 연속 실행
#
# 사용법:
#   cd rag_system
#   chmod +x ft_run_step34.sh
#   nohup ./ft_run_step34.sh > logs/ft_step34.log 2>&1 &
#   echo $! > logs/ft_step34.pid   # 프로세스 ID 저장 (나중에 kill용)
#   tail -f logs/ft_step34.log
#
# 예상 소요: ~60분 (Step 3 ~30분 + Step 4 ~30분, GT-only 기준)
# 학습 전략: External GT matched pairs만 사용 (--no-pseudo)
#   - pseudo-query 25.6K는 제외 (v1~v3 실패 이력 + real signal 드라운 방지)
#   - 2K real GT로 MNRL (in-batch negative)

set -e

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

mkdir -p logs

DATA_DIR="$HERE/../data"

echo "============================================================"
echo " Step 3+4 파이프라인 시작 — $(date)"
echo "============================================================"

# ========== 사전 체크 ==========
if [ ! -f "$DATA_DIR/external_gt_ko.jsonl" ]; then
    echo "❌ external_gt_ko.jsonl 없음. Step 2 먼저 실행 필요."
    exit 1
fi

KO_COUNT=$(wc -l < "$DATA_DIR/external_gt_ko.jsonl")
echo "  external_gt_ko.jsonl: $KO_COUNT items"

if [ "$KO_COUNT" -lt 500 ]; then
    echo "⚠️ 번역 결과가 500건 미만. Step 2 재확인 필요."
    exit 1
fi

# GPU 가용성 체크
if ! command -v nvidia-smi &> /dev/null; then
    echo "⚠️ nvidia-smi 없음. GPU 환경 확인 필요."
    exit 1
fi
nvidia-smi --query-gpu=name,memory.free,memory.total --format=csv,noheader

# ========== Step 3: Corpus 매칭 ==========
echo ""
echo "============================================================"
echo "[Step 3/4] Corpus 매칭 (bge retrieval + rerank)"
echo " 시작 — $(date)"
echo "============================================================"

python ft03_match_gt_to_corpus.py
if [ $? -ne 0 ]; then
    echo "❌ Step 3 실패"
    exit 1
fi

# Step 3 결과 검증
if [ ! -f "$DATA_DIR/external_gt_matched.jsonl" ]; then
    echo "❌ external_gt_matched.jsonl 생성 실패"
    exit 1
fi

MATCHED_COUNT=$(wc -l < "$DATA_DIR/external_gt_matched.jsonl")
echo ""
echo "  매칭 성공 pair: $MATCHED_COUNT"

if [ "$MATCHED_COUNT" -lt 300 ]; then
    echo "❌ 매칭 pair $MATCHED_COUNT < 300. Step 4 학습에 부적합."
    echo "   rerank_score threshold(0.2) 조정 후 재실행 검토 필요."
    exit 1
fi

# ========== Step 4: FT 학습 ==========
echo ""
echo "============================================================"
echo "[Step 4/4] Qwen3-Embedding FT (MNRL)"
echo " 시작 — $(date)"
echo "============================================================"

python ft04_train_qwen_embed_mnrl.py --no-pseudo
if [ $? -ne 0 ]; then
    echo "❌ Step 4 실패"
    exit 1
fi

# ========== 완료 ==========
echo ""
echo "============================================================"
echo " 전체 완료 — $(date)"
echo "============================================================"
echo ""
echo "산출물:"
echo "  - data/external_gt_matched.jsonl ($MATCHED_COUNT pairs)"
echo "  - models/qwen3-embed-ft/"
echo ""
echo "다음 단계:"
echo "  rag_exp-09-001-qwen3-embed-ft.py 작성 후 로컬 평가"
