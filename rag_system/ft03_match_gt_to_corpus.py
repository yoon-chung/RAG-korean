"""
번역된 GT 질의를 우리 corpus에 매칭

목적:
  ARC/MMLU의 question을 우리 4,272 corpus의 문서에 매칭해
  (query, positive doc) 학습 쌍을 생성.

  이는 "우리 eval이 test split에서 유도된 것"과 동일한 구조이므로
  FT 학습 신호가 eval 상황과 자연스럽게 일치함.

파이프라인:
  1. 번역된 question_ko 로드
  2. 기존 BM25 + bge-m3 hybrid + bge-reranker 로 corpus 검색 (top-3)
  3. top-1 rerank score가 임계값 이상인 것만 채택 (저신뢰 pair 제외)

출력:
  ../data/external_gt_matched.jsonl
    {"query": ..., "positive_docid": ..., "positive_content": ..., "rerank_score": ...}

주의:
  - bge-reranker score를 기준으로 pair 품질 필터링
  - score < 0.2인 pair는 제외 (매칭 실패 가능성)
  - 이 과정은 rag.py의 retrieval + rerank 로직을 재사용
"""

import os
import re
import sys
import json
import time
from pathlib import Path

# nohup redirect 상황에서 stdout 즉시 flush (버퍼링 방지)
sys.stdout.reconfigure(line_buffering=True)

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "data"

INPUT_PATH = DATA_DIR / "external_gt_ko.jsonl"
OUTPUT_PATH = DATA_DIR / "external_gt_matched.jsonl"
DOCUMENTS_PATH = DATA_DIR / "documents.jsonl"

# 매칭 품질 임계값
RERANK_SCORE_THRESHOLD = 0.2   # 이하 pair는 매칭 실패로 간주


def slugify_model(name: str) -> str:
    """rag.py와 동일. BAAI/bge-m3 -> bge_m3"""
    tail = name.split("/")[-1].lower()
    tail = re.sub(r"[^a-z0-9]+", "_", tail).strip("_")
    return tail[:40]


def load_corpus():
    docs = []
    with open(DOCUMENTS_PATH, encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line))
    return docs, {d["docid"]: d for d in docs}


def main():
    print("=" * 60)
    print(" GT → Corpus 매칭")
    print("=" * 60)

    # 의존 패키지 로드
    try:
        from elasticsearch import Elasticsearch
        from sentence_transformers import SentenceTransformer, CrossEncoder
        import numpy as np
    except ImportError as e:
        raise RuntimeError(f"필수 패키지 누락: {e}")

    # 번역 결과 로드
    queries = []
    with open(INPUT_PATH, encoding="utf-8") as f:
        for line in f:
            queries.append(json.loads(line))
    print(f"  입력 query: {len(queries)}")

    # Corpus 로드
    docs, doc_map = load_corpus()
    print(f"  Corpus    : {len(docs)} docs")

    # ES + bge-m3 + bge-reranker 로드 (rag.py와 동일한 연결 방식)
    from dotenv import load_dotenv
    load_dotenv(HERE / ".env")

    # .strip() 으로 CRLF 줄바꿈 문자 제거 (.env가 Windows 형식일 때 방어)
    es_host = os.environ.get("ES_HOST", "https://localhost:9200").strip()
    es_username = os.environ.get("ES_USERNAME", "elastic").strip()
    es_password = os.environ.get("ES_PASSWORD", "").strip()
    es_ca_certs = os.environ.get("ES_CA_CERTS", "").strip()

    if not es_password:
        raise RuntimeError("ES_PASSWORD not set in .env")

    es_kwargs = {
        "basic_auth": (es_username, es_password),
        "verify_certs": False,
        "ssl_show_warn": False,
    }
    # CA 인증서 있으면 사용
    if es_ca_certs and Path(es_ca_certs).exists():
        es_kwargs["ca_certs"] = es_ca_certs
        es_kwargs["verify_certs"] = True
        es_kwargs.pop("ssl_show_warn", None)

    es = Elasticsearch(es_host, **es_kwargs)
    if not es.ping():
        raise RuntimeError(
            f"ES 연결 실패: {es_host}\n"
            f"  .env 확인: ES_HOST={es_host}, ES_USERNAME={es_username}\n"
            f"  ES 프로세스 확인: ps aux | grep elasticsearch"
        )
    print(f"  ES        : connected to {es_host}")

    print(f"  임베딩 모델 로드 중...")
    embed_model = SentenceTransformer("BAAI/bge-m3", device="cuda")
    print(f"  Reranker 로드 중...")
    reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", device="cuda", max_length=512)

    # 기존 ES 인덱스 사용 (rag.py와 동일한 자동 이름 규칙)
    # BAAI/bge-m3 → science_kb_bge_m3
    es_index_base = os.environ.get("ES_INDEX", "science_kb").strip()
    INDEX_NAME = f"{es_index_base}_{slugify_model('BAAI/bge-m3')}"
    print(f"  ES 인덱스   : {INDEX_NAME}")

    if not es.indices.exists(index=INDEX_NAME):
        # 대안 인덱스명도 시도 (기존 convention 다를 경우)
        alternatives = [
            "science_kb_bge_m3",
            "documents_v1",
            f"science_kb_{slugify_model('BAAI/bge-m3')}",
        ]
        found = None
        for alt in alternatives:
            if es.indices.exists(index=alt):
                found = alt
                break
        if found:
            INDEX_NAME = found
            print(f"  ES 인덱스 (자동 감지): {INDEX_NAME}")
        else:
            # 모든 인덱스 리스트 출력
            try:
                all_indices = list(es.indices.get_alias(index="*").keys())
            except Exception:
                all_indices = []
            raise RuntimeError(
                f"ES 인덱스 찾을 수 없음. 시도한 이름: {alternatives}\n"
                f"  현재 ES 인덱스 목록: {all_indices}\n"
                f"  해결: python rag.py index --exp EXP-XX"
            )

    def retrieve(query: str, topk: int = 20):
        # Sparse (BM25)
        sparse_resp = es.search(
            index=INDEX_NAME,
            query={"match": {"content": query}},
            size=50,
        )
        sparse_hits = sparse_resp["hits"]["hits"]

        # Dense (bge-m3)
        q_vec = embed_model.encode(query, normalize_embeddings=True)
        dense_resp = es.search(
            index=INDEX_NAME,
            knn={
                "field": "embeddings",
                "query_vector": q_vec.tolist(),
                "k": 50,
                "num_candidates": 100,
            },
            size=50,
        )
        dense_hits = dense_resp["hits"]["hits"]

        # RRF fuse (k=60, weighted 0.3/0.7)
        scores = {}
        srcs = {}
        for hits, w in [(sparse_hits, 0.3), (dense_hits, 0.7)]:
            for rank, h in enumerate(hits, start=1):
                docid = h["_source"]["docid"]
                scores[docid] = scores.get(docid, 0.0) + w / (60 + rank)
                srcs[docid] = h["_source"]
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:topk]

        # Rerank
        pairs = [(query, srcs[docid]["content"]) for docid, _ in ranked]
        rerank_scores = reranker.predict(pairs, show_progress_bar=False)
        reranked = sorted(
            zip([docid for docid, _ in ranked], rerank_scores),
            key=lambda x: x[1],
            reverse=True,
        )
        return reranked[:3]   # top-3

    # 매칭
    matched = 0
    rejected = 0
    t0 = time.time()

    with open(OUTPUT_PATH, "w", encoding="utf-8") as fout:
        for i, q in enumerate(queries):
            query = q["question_ko"]
            if not query:
                rejected += 1
                continue

            top_results = retrieve(query, topk=20)
            if not top_results:
                rejected += 1
                continue

            top1_docid, top1_score = top_results[0]

            # 임계값 체크
            if top1_score < RERANK_SCORE_THRESHOLD:
                rejected += 1
                if (i+1) % 100 == 0:
                    print(f"  [{i+1}/{len(queries)}] matched={matched} rejected={rejected} | "
                          f"elapsed {(time.time()-t0)/60:.1f}min")
                continue

            # 매칭 저장
            doc = doc_map[top1_docid]
            row = {
                "source": q["source"],
                "orig_id": q["id"],
                "subject": q["subject"],
                "query": query,
                "positive_docid": top1_docid,
                "positive_content": doc["content"],
                "rerank_score": float(top1_score),
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            matched += 1

            if (i+1) % 100 == 0:
                fout.flush()
                rate = (i+1) / (time.time() - t0)
                eta = (len(queries) - (i+1)) / rate / 60
                print(f"  [{i+1:4d}/{len(queries)}] matched={matched} rejected={rejected} | "
                      f"{rate:.1f}q/s | ETA {eta:.1f}min")

    # 요약
    total = matched + rejected
    print("\n" + "=" * 60)
    print(" 매칭 완료")
    print("=" * 60)
    print(f"  총 처리     : {total}")
    print(f"  매칭 성공    : {matched} ({matched/total*100:.1f}%)")
    print(f"  매칭 실패    : {rejected} ({rejected/total*100:.1f}%)")
    print(f"  저장        : {OUTPUT_PATH}")
    print(f"\n다음 단계: python ft04_train_qwen_embed_mnrl.py")


if __name__ == "__main__":
    main()
