#!/bin/bash
# 3-way classifier ablation — EXP-09-001 (Qwen3-Embedding FT 3-way ensemble)
#
# 가설:
#   eval-overfit한 FORCE_EMPTY regex 후처리 대신, LLM 분류기로 chitchat을
#   사전 차단하는 원리적 접근이 LB 회복에 도움될 것.
#
# 비교 variant:
#   A: classifier=none      — 후처리 regex만 (현재 default, 218/276 통과되어 검색됨)
#   B: classifier=json + V4 — 기존 V4 프롬프트 (이전 EXP-17에서 -0.0106 후퇴 이력)
#   C: classifier=json + V5 — recall-bias + corpus scope (신규)
#
# 선행 조건:
#   - models/qwen3-embed-ft/ 존재
#   - faiss_indices/*qwen3_embed_ft*.faiss 존재 (없으면 자동 빌드)
#
# 사용:
#   cd rag_system
#   chmod +x ft_run_3way_classifier_compare.sh
#   nohup ./ft_run_3way_classifier_compare.sh > logs/3way_compare_$(date +%Y%m%d_%H%M).log 2>&1 &
#   echo $! > logs/3way_compare.pid
#
# 진행 확인:
#   tail -f logs/3way_compare_*.log
#
# 완료 후:
#   logs/EXP-09-001-A-cls-none/submission.csv
#   logs/EXP-09-001-B-cls-v4/submission.csv
#   logs/EXP-09-001-C-cls-v5/submission.csv
#   각 meta.json 의 map 비교

set -e

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
mkdir -p logs

ABS_FT_MODEL="$HERE/../models/qwen3-embed-ft"
START_TIME=$(date +%s)

echo "════════════════════════════════════════════════════════════"
echo " 3-way classifier ablation — $(date)"
echo "════════════════════════════════════════════════════════════"
echo ""

# 사전 체크
if [ ! -d "$ABS_FT_MODEL" ]; then
    echo "❌ FT 모델 없음: $ABS_FT_MODEL"
    exit 1
fi
echo "  ✓ FT 모델 확인: $ABS_FT_MODEL"

# FAISS 인덱스 존재 여부 (없으면 빌드)
FAISS_FILES=$(ls faiss_indices/*qwen3_embed_ft*.faiss 2>/dev/null | wc -l)
if [ "$FAISS_FILES" -eq 0 ]; then
    echo "  · FAISS 인덱스 없음 — 빌드 중..."
    python rag.py index \
        --embed-model "$ABS_FT_MODEL" \
        --use-faiss \
        --exp EXP-09-001-build-faiss
    echo "  ✓ FAISS 인덱스 빌드 완료"
else
    echo "  ✓ FAISS 인덱스 존재: $(ls faiss_indices/*qwen3_embed_ft*.faiss | head -1)"
fi
echo ""

# ========== Variant A: classifier=none (현재 baseline) ==========
echo "████████████████████████████████████████████████████████████"
echo " [1/3] Variant A: classifier=none — $(date +%H:%M:%S)"
echo "████████████████████████████████████████████████████████████"

python rag_exp-09-001-qwen3-embed-ft.py eval \
    --exp EXP-09-001-A-cls-none \
    --extra-embed-model "$ABS_FT_MODEL"

if [ $? -ne 0 ]; then
    echo "❌ Variant A 실패"
    exit 1
fi
echo "  ✓ Variant A 완료"
echo ""

# ========== Variant B: classifier=json + V4 ==========
echo "████████████████████████████████████████████████████████████"
echo " [2/3] Variant B: classifier=json + V4 — $(date +%H:%M:%S)"
echo "████████████████████████████████████████████████████████████"

python rag_exp-09-001-qwen3-embed-ft.py eval \
    --exp EXP-09-001-B-cls-v4 \
    --extra-embed-model "$ABS_FT_MODEL" \
    --classifier json \
    --classify-prompt v4

if [ $? -ne 0 ]; then
    echo "❌ Variant B 실패"
    exit 1
fi
echo "  ✓ Variant B 완료"
echo ""

# ========== Variant C: classifier=json + V5 (recall-bias) ==========
echo "████████████████████████████████████████████████████████████"
echo " [3/3] Variant C: classifier=json + V5 — $(date +%H:%M:%S)"
echo "████████████████████████████████████████████████████████████"

python rag_exp-09-001-qwen3-embed-ft.py eval \
    --exp EXP-09-001-C-cls-v5 \
    --extra-embed-model "$ABS_FT_MODEL" \
    --classifier json \
    --classify-prompt v5

if [ $? -ne 0 ]; then
    echo "❌ Variant C 실패"
    exit 1
fi
echo "  ✓ Variant C 완료"
echo ""

# ========== 결과 요약 ==========
TOTAL_ELAPSED=$(( ($(date +%s) - START_TIME) / 60 ))

echo "════════════════════════════════════════════════════════════"
echo " 3-way 비교 완료 — $(date)"
echo "════════════════════════════════════════════════════════════"
echo "   ⏱️  총 경과: ${TOTAL_ELAPSED}분"
echo ""
echo " 📊 submission 비교:"
for v in A-cls-none B-cls-v4 C-cls-v5; do
    sub="logs/EXP-09-001-$v/submission.csv"
    meta="logs/EXP-09-001-$v/meta.json"
    if [ -f "$sub" ]; then
        empties=$(grep -c '"topk": \[\]' "$sub")
        printf "   %-20s : empty=%d" "$v" "$empties"
        if [ -f "$meta" ]; then
            map=$(python -c "import json; m=json.load(open('$meta')); print(m.get('map', m.get('MAP', 'N/A')))" 2>/dev/null || echo "N/A")
            printf "  | local map=%s" "$map"
        fi
        echo ""
    fi
done

echo ""
echo " 🎯 다음 액션:"
echo "   - 가장 높은 variant 의 submission.csv 를 LB 제출"
echo "   - 또는 로컬에서 GT 보유 시 calc_map 로 검증 후 결정"
echo "════════════════════════════════════════════════════════════"
