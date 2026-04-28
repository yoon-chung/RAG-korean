#!/bin/bash
# Qwen3-Embedding FT 전체 파이프라인 실행 (nohup 용)
#
# 사용법:
#   cd rag_system
#   chmod +x ft_run_pipeline.sh
#   nohup ./ft_run_pipeline.sh > logs/ft_pipeline.log 2>&1 &
#   tail -f logs/ft_pipeline.log
#
# 4단계 순차 실행 + 단계별 실패 시 중단

set -e

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

mkdir -p logs

echo "============================================================"
echo " Qwen3-Embedding FT 파이프라인 시작 — $(date)"
echo "============================================================"

echo ""
echo "[Step 1/4] 외부 GT 수집 (ARC + MMLU)"
python ft01_collect_external_gt.py
if [ $? -ne 0 ]; then echo "❌ Step 1 실패"; exit 1; fi

echo ""
echo "[Step 2/4] Solar Pro 한국어 번역"
python ft02_translate_gt_to_korean.py
if [ $? -ne 0 ]; then echo "❌ Step 2 실패"; exit 1; fi

echo ""
echo "[Step 3/4] Corpus 매칭 (bge retrieval)"
python ft03_match_gt_to_corpus.py
if [ $? -ne 0 ]; then echo "❌ Step 3 실패"; exit 1; fi

echo ""
echo "[Step 4/4] Qwen3-Embedding FT (MNRL)"
python ft04_train_qwen_embed_mnrl.py
if [ $? -ne 0 ]; then echo "❌ Step 4 실패"; exit 1; fi

echo ""
echo "============================================================"
echo " 파이프라인 전체 완료 — $(date)"
echo "============================================================"
echo ""
echo "다음 단계: rag_exp-09-001-qwen3-embed-ft.py 로 로컬 평가"
