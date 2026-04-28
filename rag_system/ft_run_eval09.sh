#!/bin/bash
# EXP-09-001 평가 파이프라인
#
# 전제:
#   ft_run_pipeline.sh 또는 ft_run_step34.sh 완료 상태
#   → models/qwen3-embed-ft/ 생성돼 있어야 함
#
# 실행:
#   cd rag_system
#   chmod +x ft_run_eval09.sh
#   nohup ./ft_run_eval09.sh > logs/exp09_eval.log 2>&1 &
#   tail -f logs/exp09_eval.log
#
# 2단계:
#   Step A: FT 모델로 FAISS 인덱스 구축 (5~10분)
#   Step B: 3-way ensemble 평가 (15~20분)

set -e

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

mkdir -p logs

FT_MODEL="models/qwen3-embed-ft"
ABS_FT_MODEL="$HERE/../$FT_MODEL"

echo "============================================================"
echo " EXP-09-001 평가 파이프라인 시작 — $(date)"
echo "============================================================"

# 사전 체크
if [ ! -d "$ABS_FT_MODEL" ]; then
    echo "❌ FT 모델 없음: $ABS_FT_MODEL"
    echo "   ft_run_step34.sh 먼저 실행 필요"
    exit 1
fi
echo "  ✓ FT 모델 확인: $ABS_FT_MODEL"

# ========== Step A: FAISS 인덱스 구축 ==========
echo ""
echo "[Step A] FAISS 인덱스 구축 (Qwen3-Embedding FT)"
echo " 시작 — $(date)"
echo ""

python rag.py index \
    --embed-model "$ABS_FT_MODEL" \
    --use-faiss \
    --exp EXP-09-001-build-faiss

if [ $? -ne 0 ]; then
    echo "❌ Step A (FAISS 인덱스) 실패"
    exit 1
fi

# FAISS 인덱스 파일 생성 확인
FAISS_FILES=$(ls faiss_indices/*qwen3_embed_ft*.faiss 2>/dev/null | wc -l)
if [ "$FAISS_FILES" -eq 0 ]; then
    echo "❌ FAISS 인덱스 파일 생성 실패"
    exit 1
fi
echo "  ✓ FAISS 인덱스 생성: $(ls faiss_indices/*qwen3_embed_ft*.faiss)"

# ========== Step B: 3-way ensemble 평가 ==========
echo ""
echo "[Step B] 3-way ensemble 평가"
echo " 시작 — $(date)"
echo ""

python rag_exp-09-001-qwen3-embed-ft.py eval \
    --exp EXP-09-001-qwen3-embed-ft \
    --extra-embed-model "$ABS_FT_MODEL"

if [ $? -ne 0 ]; then
    echo "❌ Step B (평가) 실패"
    exit 1
fi

# ========== 완료 ==========
echo ""
echo "============================================================"
echo " EXP-09-001 평가 완료 — $(date)"
echo "============================================================"
echo ""
echo "산출물:"
echo "  - logs/EXP-09-001-qwen3-embed-ft/submission.csv"
echo "  - logs/EXP-09-001-qwen3-embed-ft/config.json"
echo "  - logs/EXP-09-001-qwen3-embed-ft/meta.json"
echo ""
echo "다음 판단:"
echo "  - 로컬 MAP 확인 (meta.json의 map 필드)"
echo "  - tau002 (0.9258) 대비 +0.003 이상이면 LB 제출 고려"
echo "  - 4-way ensemble에 추가해 5-way로 확장 실험 검토"
