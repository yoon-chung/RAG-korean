"""
외부 GT 한국어 번역 — Solar Pro

목적:
  external_gt_en.jsonl의 영문 문항을 자연스러운 한국어로 번역.
  우리 corpus(ko_mmlu + ko_ai2_arc)와 같은 톤·문체로 맞춤.

특징:
  - Batch 10개씩 처리 (비용·속도 최적화)
  - 증분 저장 (중단/재실행 시 이어서)
  - JSON mode로 파싱 안정화
  - 프롬프트는 "지식 콘텐츠 번역가" (과학 한정 X)

비용 추산:
  ~2,650 items × ~250 tokens/item = ~660K tokens
  Solar Pro: ~$0.3~0.6 (잔여 크레딧 $17 대비 안전)

출력:
  ../data/external_gt_ko.jsonl
    {"source": ..., "id": ..., "subject": ...,
     "question_ko": ..., "choices_ko": [...], "correct_choice_ko": ...}
"""

import os
import json
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "data"
load_dotenv(HERE / ".env")

INPUT_PATH = DATA_DIR / "external_gt_en.jsonl"
OUTPUT_PATH = DATA_DIR / "external_gt_ko.jsonl"

BATCH_SIZE = 10

SYSTEM_PROMPT = """너는 지식 콘텐츠 전문 번역가다.
영문 Q&A 문항을 자연스러운 한국어로 번역한다.

## 다루는 주제
과학·기술·의학·수학만이 아니라 사회·역사·인물·생활상식·예술 등 전 범위.
주제에 따라 적절한 한국어 톤 선택 (교과서적, 실용적, 학술적 등).

## 번역 규칙
1) 전문 용어는 국내 표준 표기를 따른다 (광합성, 자기장, 대류 등)
2) 고유명사는 한국 발음 표기 (예: Einstein → 아인슈타인, Darwin → 다윈)
3) 선택지 간 어미·길이 일관성 유지
4) 원문 의미를 왜곡하지 않고 자연스럽게
5) 단위·수식은 한국어 관례 유지 (°C, m/s, 2+3=5 그대로)
6) 불필요한 의역·설명 금지

## 입력 형식
JSON 배열. 각 item = {id, question, choices, correct_choice}

## 출력 형식 (JSON)
{"translations": [
  {"id": "...", "question_ko": "...", "choices_ko": ["...", "...", ...], "correct_choice_ko": "..."},
  ...
]}"""


def build_batch_prompt(batch):
    payload = [
        {
            "id": item["id"],
            "question": item["question"],
            "choices": item["choices"],
            "correct_choice": item["correct_choice"],
        }
        for item in batch
    ]
    return json.dumps(payload, ensure_ascii=False, indent=2)


def translate_batch(client, batch, max_retries=3):
    user_msg = build_batch_prompt(batch)

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model="solar-pro",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0,
                seed=1,
                response_format={"type": "json_object"},
                timeout=60,
            )
            result = json.loads(resp.choices[0].message.content)
            translations = result.get("translations", [])
            if len(translations) != len(batch):
                print(f"    ⚠️ 응답 개수 mismatch: expected {len(batch)}, got {len(translations)}")
                if attempt < max_retries - 1:
                    continue
            return translations
        except Exception as e:
            print(f"    ⚠️ attempt {attempt+1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)

    return []


def load_already_translated():
    """이미 번역된 id set 반환 (재실행 시 이어서 처리)"""
    done = set()
    if OUTPUT_PATH.exists():
        with open(OUTPUT_PATH, encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                    done.add(row["id"])
                except Exception:
                    continue
    return done


def main():
    print("=" * 60)
    print(" Solar Pro 한국어 번역")
    print("=" * 60)

    api_key = os.environ.get("UPSTAGE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("UPSTAGE_API_KEY not set in .env")

    client = OpenAI(api_key=api_key, base_url="https://api.upstage.ai/v1")

    # 입력 로드
    all_items = []
    with open(INPUT_PATH, encoding="utf-8") as f:
        for line in f:
            all_items.append(json.loads(line))
    print(f"  입력 : {len(all_items)} items")

    # 이미 번역된 건 제외
    done_ids = load_already_translated()
    pending = [item for item in all_items if item["id"] not in done_ids]
    print(f"  기존 번역: {len(done_ids)} (skip)")
    print(f"  처리 대상: {len(pending)} items")

    # 배치 단위 처리
    out_mode = "a" if done_ids else "w"
    with open(OUTPUT_PATH, out_mode, encoding="utf-8") as fout:
        for batch_start in range(0, len(pending), BATCH_SIZE):
            batch = pending[batch_start:batch_start + BATCH_SIZE]
            print(f"  [{batch_start+1:4d}/{len(pending)}] 번역 중...", end=" ", flush=True)

            translations = translate_batch(client, batch)
            if not translations:
                print("❌ skip")
                continue

            # id 기반 매칭으로 원본 정보와 결합
            id_to_trans = {t["id"]: t for t in translations}
            saved = 0
            for orig in batch:
                tr = id_to_trans.get(orig["id"])
                if not tr:
                    continue
                merged = {
                    "source": orig["source"],
                    "id": orig["id"],
                    "subject": orig["subject"],
                    "question_ko": tr.get("question_ko", "").strip(),
                    "choices_ko": tr.get("choices_ko", []),
                    "correct_choice_ko": tr.get("correct_choice_ko", "").strip(),
                }
                # 품질 체크 — 빈 번역 skip
                if not merged["question_ko"] or not merged["correct_choice_ko"]:
                    continue
                fout.write(json.dumps(merged, ensure_ascii=False) + "\n")
                saved += 1
            fout.flush()
            print(f"✓ saved {saved}/{len(batch)}")

            # Rate limit 고려 (Solar Pro는 정확한 한도 불명, 안전하게)
            time.sleep(0.5)

    # 결과 요약
    total = sum(1 for _ in open(OUTPUT_PATH, encoding="utf-8"))
    print("\n" + "=" * 60)
    print(" 번역 완료")
    print("=" * 60)
    print(f"  총 번역 건수: {total}")
    print(f"  저장        : {OUTPUT_PATH}")
    print(f"\n다음 단계: python ft03_match_gt_to_corpus.py")


if __name__ == "__main__":
    main()
