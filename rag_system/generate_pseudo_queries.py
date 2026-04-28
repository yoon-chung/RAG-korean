"""
Pseudo Query 생성기 — Reranker/Embedding Fine-tuning용 학습 데이터

사용법:
    # Solar Pro로 생성 (기본)
    python generate_pseudo_queries.py --llm solar

    # gpt-4o-mini로 생성
    python generate_pseudo_queries.py --llm gpt4mini

    # hard negative 포함 (ES 인덱스 필요)
    python generate_pseudo_queries.py --llm solar --hard-negatives

산출물:
    data/pseudo_queries_solar.jsonl        (Solar Pro 생성)
    data/pseudo_queries_gpt4mini.jsonl     (gpt-4o-mini 생성)
    data/training_triplets.jsonl           (--hard-negatives일 때)
"""

import os
import json
import time
import argparse
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
load_dotenv(HERE / ".env")

from openai import OpenAI


PSEUDO_QUERY_SYSTEM = """너는 과학 지식 문서를 읽고, 이 문서를 검색해서 찾을 수 있는
자연스러운 한국어 질문을 생성하는 도구다.

## 생성 규칙
1) 검색 엔진에 입력할 법한 자연스러운 구어체 질문을 3개 생성하라.
2) 3개의 질문은 서로 다른 유형이어야 한다:
   - 하나는 핵심 개념을 직접 묻는 질문 (예: "~은 무엇인가?", "~에 대해 알려줘")
   - 하나는 원인/이유/과정을 묻는 질문 (예: "왜 ~할까?", "~하는 이유는?")
   - 하나는 일상적/응용적 관점의 질문 (예: "~하면 어떻게 되나?", "~의 영향은?")
3) 질문은 1~2문장, 한국어로 작성한다.
4) 문서의 정답을 그대로 포함하지 않는다 (키워드 일부만 자연스럽게 포함).
5) 실제 사람이 궁금해서 검색할 법한 표현을 사용한다.

## 나쁜 예시 (피할 것)
- "다음 중 올바른 것은?" (시험 문제 스타일)
- 영어 질문
- 문서 내용을 그대로 복사한 질문
- 너무 구체적이어서 오직 이 문서만 답이 되는 질문

## 출력 형식
반드시 아래 JSON만 출력한다 (다른 설명 금지):
{"questions": ["질문1", "질문2", "질문3"]}"""


def create_client(llm_type: str) -> tuple:
    """(client, model_name) 반환"""
    if llm_type == "solar":
        api_key = os.environ.get("UPSTAGE_API_KEY")
        if not api_key:
            raise RuntimeError("UPSTAGE_API_KEY not set in .env")
        client = OpenAI(api_key=api_key, base_url="https://api.upstage.ai/v1")
        return client, "solar-pro"
    else:
        return OpenAI(), "gpt-4o-mini"


def generate_queries_for_doc(client: OpenAI, model: str, doc_content: str,
                             max_retries: int = 3) -> Optional[List[str]]:
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": PSEUDO_QUERY_SYSTEM},
                    {"role": "user", "content": doc_content},
                ],
                temperature=0.7,
                response_format={"type": "json_object"},
                timeout=30,
            )
            result = json.loads(resp.choices[0].message.content)
            questions = result.get("questions", [])
            if len(questions) >= 3:
                return questions[:3]
            print(f"  [WARN] Got {len(questions)} questions, retrying...")
        except Exception as e:
            print(f"  [WARN] Attempt {attempt+1} failed: {e}")
            time.sleep(2 ** attempt)
    return None


def generate_all_queries(docs: list, client: OpenAI, model: str,
                         output_path: Path, delay: float = 0.1):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 이어서 생성 가능 (resume)
    existing = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                existing.add(json.loads(line)["docid"])
        print(f"[INFO] Resuming: {len(existing)} docs already done")

    total = len(docs)
    success = len(existing)
    fail = 0

    with open(output_path, "a") as of:
        for i, doc in enumerate(docs):
            if doc["docid"] in existing:
                continue

            questions = generate_queries_for_doc(client, model, doc["content"])
            if questions:
                row = {
                    "docid": doc["docid"],
                    "src": doc.get("src", ""),
                    "content": doc["content"],
                    "questions": questions,
                }
                of.write(json.dumps(row, ensure_ascii=False) + "\n")
                of.flush()
                success += 1
            else:
                fail += 1
                print(f"  [ERROR] Failed for docid={doc['docid']}")

            if (i + 1) % 100 == 0:
                print(f"[PROGRESS] {i+1}/{total} (success={success}, fail={fail})")

            time.sleep(delay)

    print(f"\n[DONE] Total={total}, Success={success}, Fail={fail}")
    print(f"[DONE] Output: {output_path}")


def dense_search_negatives(es, index_name: str, query: str, embed_model,
                           docid: str, top_k: int = 10, max_negatives: int = 5):
    """Dense 검색으로 hard negative 수집 (BM25와 다른 관점의 오답)"""
    try:
        import numpy as np
        q_emb = embed_model.encode([query], normalize_embeddings=True)[0].tolist()
        hits = es.search(
            index=index_name,
            knn={
                "field": "embeddings",
                "query_vector": q_emb,
                "k": top_k,
                "num_candidates": max(top_k * 4, 100),
            },
            size=top_k,
        )["hits"]["hits"]
        neg_ids, neg_contents = [], []
        for h in hits:
            neg_id = h["_source"]["docid"]
            if neg_id != docid:
                neg_ids.append(neg_id)
                neg_contents.append(h["_source"]["content"])
            if len(neg_ids) >= max_negatives:
                break
        return neg_ids, neg_contents
    except Exception as e:
        return [], []


def generate_hard_negatives(queries_path: Path, output_path: Path,
                            es_index: str = None,
                            top_k: int = 10, max_negatives: int = 5,
                            use_dense: bool = False):
    """BM25 + (선택) Dense 검색으로 hard negative 수집
    --use-dense: dense 검색 실패 케이스도 negative로 추가 (멘토 조언)
    """
    from elasticsearch import Elasticsearch

    es = Elasticsearch(
        [os.environ.get("ES_HOST", "https://localhost:9200")],
        basic_auth=(os.environ["ES_USERNAME"], os.environ["ES_PASSWORD"]),
        ca_certs=os.environ.get("ES_CA_CERTS",
                                "../baseline/elasticsearch-8.8.0/config/certs/http_ca.crt"),
    )

    if es_index:
        index_name = es_index
    else:
        all_indices = list(es.indices.get(index="science_kb_*").keys())
        if all_indices:
            index_name = all_indices[0]
            print(f"[INFO] Auto-detected index: {index_name}")
        else:
            index_name = os.environ.get("ES_INDEX", "science_kb") + "_bge_m3"

    if not es.indices.exists(index=index_name):
        print(f"[ERROR] Index '{index_name}' not found. Run `python rag.py index` first.")
        return

    # Dense negative용 embedding 모델 로드
    embed_model = None
    if use_dense:
        try:
            from sentence_transformers import SentenceTransformer
            embed_model = SentenceTransformer("BAAI/bge-m3")
            print(f"[INFO] Loaded embedding model for dense negatives")
        except Exception as e:
            print(f"[WARN] Failed to load embedding model: {e}. Dense negatives disabled.")
            use_dense = False

    with open(queries_path) as f:
        docs = [json.loads(line) for line in f]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_triplets = 0

    with open(output_path, "w") as of:
        for i, doc in enumerate(docs):
            docid = doc["docid"]

            for query in doc["questions"]:
                # BM25 hard negatives
                try:
                    hits = es.search(
                        index=index_name,
                        query={"match": {"content": {"query": query}}},
                        size=top_k,
                    )["hits"]["hits"]
                except Exception as e:
                    print(f"  [WARN] BM25 search failed: {e}")
                    continue

                hard_neg_ids = []
                hard_neg_contents = []
                for h in hits:
                    neg_id = h["_source"]["docid"]
                    if neg_id != docid:
                        hard_neg_ids.append(neg_id)
                        hard_neg_contents.append(h["_source"]["content"])
                    if len(hard_neg_ids) >= max_negatives:
                        break

                # Dense hard negatives (BM25와 다른 관점)
                if use_dense and embed_model:
                    dense_neg_ids, dense_neg_contents = dense_search_negatives(
                        es, index_name, query, embed_model, docid,
                        top_k=top_k, max_negatives=max_negatives,
                    )
                    # BM25에 없는 dense negative만 추가 (중복 제거)
                    existing = set(hard_neg_ids)
                    for nid, ncont in zip(dense_neg_ids, dense_neg_contents):
                        if nid not in existing:
                            hard_neg_ids.append(nid)
                            hard_neg_contents.append(ncont)
                            existing.add(nid)

                triplet = {
                    "query": query,
                    "positive_docid": docid,
                    "positive_content": doc["content"],
                    "hard_negative_docids": hard_neg_ids,
                    "hard_negative_contents": hard_neg_contents,
                }
                of.write(json.dumps(triplet, ensure_ascii=False) + "\n")
                of.flush()
                total_triplets += 1

            if (i + 1) % 200 == 0:
                print(f"[PROGRESS] {i+1}/{len(docs)} docs, {total_triplets} triplets")

    print(f"\n[DONE] {total_triplets} training triplets → {output_path}")


def main():
    p = argparse.ArgumentParser(description="Pseudo Query Generator for Fine-tuning")
    p.add_argument("--llm", choices=["gpt4mini", "solar"], default="solar",
                   help="Which LLM to use (default: solar)")
    p.add_argument("--docs-path", default="../data/documents.jsonl")
    p.add_argument("--output-dir", default="../data")
    p.add_argument("--hard-negatives", action="store_true",
                   help="Generate hard negatives via BM25 (requires ES index)")
    p.add_argument("--use-dense", action="store_true",
                   help="Add dense search negatives (BM25+dense, requires embeddings in ES)")
    p.add_argument("--es-index", default=None,
                   help="ES index name for hard negatives (auto-detect if not set)")
    p.add_argument("--delay", type=float, default=0.1,
                   help="Delay between API calls in seconds (default: 0.1)")
    args = p.parse_args()

    # Load documents
    with open(args.docs_path) as f:
        docs = [json.loads(line) for line in f]
    print(f"[INFO] Loaded {len(docs)} documents from {args.docs_path}")

    output_dir = Path(args.output_dir)
    output_file = output_dir / f"pseudo_queries_{args.llm}.jsonl"

    # Generate
    client, model = create_client(args.llm)
    print(f"[INFO] Using LLM: {model}")
    print(f"[INFO] Output: {output_file}")
    print(f"[INFO] Estimated cost: ~${1.50 if args.llm == 'solar' else 0.55}")
    generate_all_queries(docs, client, model, output_file, args.delay)

    # Hard negatives
    if args.hard_negatives:
        triplets_path = output_dir / f"training_triplets_{args.llm}.jsonl"
        print(f"\n[INFO] Generating hard negatives → {triplets_path}")
        generate_hard_negatives(output_file, triplets_path,
                               es_index=args.es_index, use_dense=args.use_dense)

        # 합본 생성 (solar + gpt4mini 모두 있으면)
        solar_path = output_dir / "training_triplets_solar.jsonl"
        gpt_path = output_dir / "training_triplets_gpt4mini.jsonl"
        merged_path = output_dir / "training_triplets_merged.jsonl"
        if solar_path.exists() and gpt_path.exists():
            print(f"\n[INFO] Merging solar + gpt4mini triplets...")
            with open(merged_path, "w") as out:
                for src in [solar_path, gpt_path]:
                    with open(src) as f:
                        for line in f:
                            out.write(line)
            count = sum(1 for _ in open(merged_path))
            print(f"[INFO] Merged: {count} triplets → {merged_path}")

    # Summary
    print("\n" + "=" * 60)
    print("[SUMMARY]")
    for name in [f"pseudo_queries_{args.llm}.jsonl",
                 f"training_triplets_{args.llm}.jsonl",
                 "training_triplets_merged.jsonl"]:
        check_path = output_dir / name
        if check_path.exists():
            count = sum(1 for _ in open(check_path))
            print(f"  {name}: {count} entries")
    print("=" * 60)


if __name__ == "__main__":
    main()
