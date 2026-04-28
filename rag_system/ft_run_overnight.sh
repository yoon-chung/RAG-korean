#!/bin/bash
# 🌙 야간 풀 파이프라인 — Step 3 → 4 → FAISS 인덱스 → 평가
#
# 전제:
#   Step 2 (번역) 완료 상태 (data/external_gt_ko.jsonl 존재, 2,700건 내외)
#
# 실행 순서:
#   Phase 1 — ft_run_step34.sh
#     Step 3: GT → Corpus 매칭 (bge retrieval + rerank, ~25min)
#     Step 4: Qwen3-Embedding-0.6B FT (MNRL, GT-only, ~30min)
#   Phase 2 — ft_run_eval09.sh
#     Step A: FT 모델 기반 FAISS 인덱스 구축 (~10min)
#     Step B: 3-way ensemble 평가 (~20min)
#
# 예상 총 소요: ~85분 (여유분 포함 최대 ~120분)
#
# 사용법:
#   cd rag_system
#   chmod +x ft_run_overnight.sh
#   nohup ./ft_run_overnight.sh > logs/ft_overnight_$(date +%Y%m%d_%H%M).log 2>&1 &
#   echo $! > logs/ft_overnight.pid
#   echo "PID: $(cat logs/ft_overnight.pid) — sleep well 🌙"
#
# 아침 확인:
#   tail -80 logs/ft_overnight_*.log
#   cat logs/EXP-09-001-qwen3-embed-ft/meta.json | python -m json.tool
#
# 비상 종료:
#   PID=$(cat logs/ft_overnight.pid)
#   pkill -P $PID           # 하위 프로세스 종료
#   kill $PID               # 자신 종료

set -e

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

mkdir -p logs

START_TIME=$(date +%s)

echo "════════════════════════════════════════════════════════════"
echo " 🌙 야간 풀 파이프라인 시작"
echo "   시작 시각: $(date)"
echo "════════════════════════════════════════════════════════════"
echo ""
echo " 📋 실행 예정:"
echo "   Phase 1 — Step 3 (매칭) + Step 4 (FT 학습)"
echo "   Phase 2 — Step A (FAISS 인덱스) + Step B (평가)"
echo ""
echo " 📂 작업 디렉토리: $HERE"
echo " 💾 산출물 예정:"
echo "     ../data/external_gt_matched.jsonl"
echo "     models/qwen3-embed-ft/"
echo "     faiss_indices/science_kb_qwen3_embed_ft*.faiss"
echo "     logs/EXP-09-001-qwen3-embed-ft/submission.csv"
echo ""

# ========== 사전 체크 ==========
echo "🔍 사전 체크..."

if [ ! -f "../data/external_gt_ko.jsonl" ]; then
    echo "❌ external_gt_ko.jsonl 없음. Step 2 먼저 실행 필요."
    exit 1
fi

KO_COUNT=$(wc -l < ../data/external_gt_ko.jsonl)
echo "   ✓ 번역 데이터: $KO_COUNT items"

if [ "$KO_COUNT" -lt 2000 ]; then
    echo "⚠️ 번역 데이터가 2,000건 미만. 경고하고 계속 진행합니다."
fi

# GPU
if ! command -v nvidia-smi &>/dev/null; then
    echo "❌ nvidia-smi 없음 (GPU 필수)"
    exit 1
fi
GPU_FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
echo "   ✓ GPU free memory: ${GPU_FREE} MiB"

# ES 체크 (Step 3 와 Step B 에서 사용)
if [ -z "${ES_PASSWORD:-}" ]; then
    # .env 로드 시도
    if [ -f ".env" ]; then
        export $(grep -v '^#' .env | xargs)
    fi
fi

echo ""
START_P1=$(date +%s)

# ========== Phase 1: Step 3 + Step 4 ==========
echo "████████████████████████████████████████████████████████████"
echo " 🔧 Phase 1/2: 매칭 + FT 학습"
echo " 시작: $(date +%H:%M:%S)"
echo "████████████████████████████████████████████████████████████"
echo ""

./ft_run_step34.sh
if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Phase 1 실패 — 파이프라인 중단"
    echo "   경과 시간: $(( ($(date +%s) - START_TIME) / 60 ))분"
    exit 1
fi

P1_ELAPSED=$(( ($(date +%s) - START_P1) / 60 ))
echo ""
echo "✅ Phase 1 완료 (경과 ${P1_ELAPSED}분)"
echo ""

# Phase 1 산출물 검증
if [ ! -d "../models/qwen3-embed-ft" ] && [ ! -d "models/qwen3-embed-ft" ]; then
    echo "❌ FT 모델 생성 안 됨 — Phase 2 진행 불가"
    exit 1
fi

START_P2=$(date +%s)

# ========== Phase 2: Step A + Step B ==========
echo "████████████████████████████████████████████████████████████"
echo " 🧪 Phase 2/2: FAISS 인덱스 + 평가"
echo " 시작: $(date +%H:%M:%S)"
echo "████████████████████████████████████████████████████████████"
echo ""

./ft_run_eval09.sh
if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Phase 2 실패"
    echo "   FT 모델은 만들어졌으니 수동으로 평가 재시도 가능:"
    echo "   - python rag.py index --embed-model models/qwen3-embed-ft --use-faiss --exp EXP-09-001-build-faiss"
    echo "   - python rag_exp-09-001-qwen3-embed-ft.py eval --exp EXP-09-001-qwen3-embed-ft --extra-embed-model models/qwen3-embed-ft"
    exit 1
fi

P2_ELAPSED=$(( ($(date +%s) - START_P2) / 60 ))
TOTAL_ELAPSED=$(( ($(date +%s) - START_TIME) / 60 ))

# ========== 🎉 최종 요약 ==========
echo ""
echo "════════════════════════════════════════════════════════════"
echo " 🎉 전체 파이프라인 완료!"
echo "    종료 시각: $(date)"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "   ⏱️  경과 시간:"
echo "      Phase 1 (매칭+FT)    : ${P1_ELAPSED}분"
echo "      Phase 2 (인덱스+평가): ${P2_ELAPSED}분"
echo "      총                   : ${TOTAL_ELAPSED}분"
echo ""

# 산출물 요약
echo "   📦 산출물 확인:"
MATCHED_COUNT=$(wc -l < ../data/external_gt_matched.jsonl 2>/dev/null || echo "N/A")
echo "      매칭 pair         : $MATCHED_COUNT"

FT_META="models/qwen3-embed-ft/ft_meta.json"
[ ! -f "$FT_META" ] && FT_META="../models/qwen3-embed-ft/ft_meta.json"
if [ -f "$FT_META" ]; then
    TRAIN_PAIRS=$(python -c "import json; print(json.load(open('$FT_META')).get('train_pairs', 'N/A'))" 2>/dev/null || echo "N/A")
    FT_DURATION=$(python -c "import json; print(f\"{json.load(open('$FT_META')).get('duration_sec', 0)/60:.1f}min\")" 2>/dev/null || echo "N/A")
    echo "      FT 학습 pair      : $TRAIN_PAIRS"
    echo "      FT 학습 시간       : $FT_DURATION"
fi

# 로컬 MAP 추출
META_PATH="logs/EXP-09-001-qwen3-embed-ft/meta.json"
if [ -f "$META_PATH" ]; then
    echo ""
    echo "   📊 로컬 평가 결과:"
    python << PYEOF 2>/dev/null || echo "      (meta.json 파싱 실패 — 수동 확인 필요)"
import json
with open('$META_PATH') as f:
    meta = json.load(f)
# 주요 metric 출력 (키명은 rag.py 버전에 따라 다를 수 있음)
for key in ['map', 'MAP', 'mrr', 'MRR', 'map_score', 'mrr_score']:
    if key in meta:
        print(f"      {key:20s}: {meta[key]}")
# 다른 유의미한 key들도 표시
for key in ['timestamp', 'num_queries', 'chitchat_filtered']:
    if key in meta:
        print(f"      {key:20s}: {meta[key]}")
PYEOF
fi

echo ""
echo "   🎯 다음 액션 가이드:"
echo "      [로컬 MAP 해석]"
echo "      ≥ 0.9260 → tau002(0.9258) 대비 개선, 5-way ensemble 추가 가치"
echo "      ≈ 0.9200 → plateau, 기존 0.9379 유지"
echo "      < 0.9150 → FT 부작용, 원인 분석 필요"
echo ""
echo "      [확인 명령]"
echo "      cat $META_PATH | python -m json.tool"
echo "      head logs/EXP-09-001-qwen3-embed-ft/submission.csv"
echo ""
echo "════════════════════════════════════════════════════════════"
