"""
EXP-02-003-03 최고점 재현용 RAG 파이프라인 (MAP 0.9152, MRR 0.9242)
기반: rag.py (dev/yoon) — 최고점 재현에 불필요한 코드 제거, 기본값을 최적 설정으로 고정

최고점 재현 명령어:
    python rag_exp-02-003-03.py eval --exp EXP-02-003-03 \
        --classifier none --score-threshold 0.05 --hyde \
        --rrf-weight-sparse 0.3 --rrf-weight-dense 0.7

위 명령은 기본값으로 설정되어 있으므로 아래 단축 명령으로도 동일 결과:
    python rag_exp-02-003-03.py eval --exp EXP-02-003-03

구성 요소:
    - Embedding: BAAI/bge-m3 (hybrid: BM25 sparse + dense)
    - Reranker: BAAI/bge-reranker-v2-m3 (cross-encoder, top-20 → top-3)
    - 의도분류: classifier=none (분류기 없이 전체 검색, rerank score로 사후 필터)
    - Score threshold: τ=0.05 (rerank top-1 < 0.05 → topk=[], 비과학 잡담 필터)
    - HyDE: dense 쿼리를 LLM 생성 가상 답변 문서로 대체 (sparse는 원본 유지)
    - Weighted RRF: sparse 0.3 / dense 0.7 가중치 융합
    - LLM: gpt-4o-mini

실험 결과는 logs/<exp>/ 아래에 submission.csv / config.json / meta.json 으로 저장된다.
"""

import os
import re
import sys
import json
import time
import argparse
import traceback
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List, Tuple, Optional

from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
load_dotenv(HERE / ".env")

from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer
from openai import OpenAI


# ============================================================
# Config — 기본값이 EXP-02-003-03 최적 설정
# ============================================================
@dataclass
class Config:
    # Mode
    mode: str = "eval"
    exp_name: str = "default"

    # Retrieval
    retrieval_mode: str = "hybrid"
    use_reranker: bool = True
    classifier_mode: str = "none"

    # Models
    embed_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    llm_model: str = "gpt-4o-mini"

    # Retrieval params
    topk_sparse: int = 50
    topk_dense: int = 50
    topk_rerank: int = 20
    topk_final: int = 3
    rrf_k: int = 60
    score_threshold: float = 0.05
    use_hyde: bool = True
    rrf_weight_sparse: float = 0.3
    rrf_weight_dense: float = 0.7

    # Paths
    docs_path: str = "../data/documents.jsonl"
    eval_path: str = "../data/eval.jsonl"
    index_name: str = ""

    # Elasticsearch
    es_username: str = "elastic"
    es_password: str = ""
    es_host: str = "https://localhost:9200"
    es_ca_certs: str = "../baseline/elasticsearch-8.8.0/config/certs/http_ca.crt"

    def output_dir(self) -> Path:
        return HERE / "logs" / self.exp_name

    def to_dict(self) -> dict:
        return asdict(self)


def slugify_model(name: str) -> str:
    tail = name.split("/")[-1].lower()
    tail = re.sub(r"[^a-z0-9]+", "_", tail).strip("_")
    return tail[:40]


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="EXP-02-003-03 RAG pipeline (MAP 0.9152)")
    p.add_argument("mode", choices=["index", "eval", "all"])
    p.add_argument("--exp", dest="exp_name", default="default",
                   help="Experiment name (used for output dir)")

    # Retrieval
    p.add_argument("--retrieval-mode", choices=["sparse", "dense", "hybrid"], default="hybrid")
    p.add_argument("--no-reranker", action="store_true", help="Disable cross-encoder reranker")
    p.add_argument("--classifier", choices=["json", "none"], default="none")

    # Models
    p.add_argument("--embed-model", default=None)
    p.add_argument("--reranker-model", default=None)
    p.add_argument("--llm-model", default=None)

    # Params
    p.add_argument("--topk-sparse", type=int, default=50)
    p.add_argument("--topk-dense", type=int, default=50)
    p.add_argument("--topk-rerank", type=int, default=20)
    p.add_argument("--topk-final", type=int, default=3)
    p.add_argument("--rrf-k", type=int, default=60)
    p.add_argument("--score-threshold", type=float, default=0.05)
    p.add_argument("--hyde", action="store_true", default=True)
    p.add_argument("--no-hyde", action="store_true", help="Disable HyDE")
    p.add_argument("--rrf-weight-sparse", type=float, default=0.3)
    p.add_argument("--rrf-weight-dense", type=float, default=0.7)

    # Paths
    p.add_argument("--index-name", default=None, help="Override auto-derived index name")
    p.add_argument("--docs-path", default=None)
    p.add_argument("--eval-path", default=None)

    args = p.parse_args()

    cfg = Config()
    cfg.mode = args.mode
    cfg.exp_name = args.exp_name
    cfg.retrieval_mode = args.retrieval_mode
    cfg.use_reranker = not args.no_reranker
    cfg.classifier_mode = args.classifier

    cfg.embed_model = args.embed_model or os.environ.get("EMBED_MODEL", cfg.embed_model)
    cfg.reranker_model = args.reranker_model or os.environ.get("RERANKER_MODEL", cfg.reranker_model)
    cfg.llm_model = args.llm_model or os.environ.get("LLM_MODEL", cfg.llm_model)

    cfg.topk_sparse = args.topk_sparse
    cfg.topk_dense = args.topk_dense
    cfg.topk_rerank = args.topk_rerank
    cfg.topk_final = args.topk_final
    cfg.rrf_k = args.rrf_k
    cfg.score_threshold = args.score_threshold
    cfg.use_hyde = not args.no_hyde
    cfg.rrf_weight_sparse = args.rrf_weight_sparse
    cfg.rrf_weight_dense = args.rrf_weight_dense

    cfg.docs_path = args.docs_path or os.environ.get("DOCS_PATH", cfg.docs_path)
    cfg.eval_path = args.eval_path or os.environ.get("EVAL_PATH", cfg.eval_path)

    if args.index_name:
        cfg.index_name = args.index_name
    else:
        base = os.environ.get("ES_INDEX", "science_kb")
        cfg.index_name = f"{base}_{slugify_model(cfg.embed_model)}"

    cfg.es_username = os.environ["ES_USERNAME"]
    cfg.es_password = os.environ["ES_PASSWORD"]
    cfg.es_host = os.environ.get("ES_HOST", cfg.es_host)
    cfg.es_ca_certs = os.environ.get("ES_CA_CERTS", cfg.es_ca_certs)

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set in .env")

    return cfg


# ============================================================
# Runtime state
# ============================================================
class Runtime:
    def __init__(self, cfg: Config):
        self.cfg = cfg

        # Embedding
        self.embed_model_name = cfg.embed_model
        self.embed_loaded = False
        try:
            self.embed_model = SentenceTransformer(cfg.embed_model)
            self.embed_loaded = True
            print(f"[INFO] Loaded embedding model: {cfg.embed_model}")
        except Exception as e:
            fallback = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"
            print(f"[WARN] Failed to load {cfg.embed_model}: {e}")
            print(f"[INFO] Falling back to {fallback}")
            self.embed_model_name = fallback
            self.embed_model = SentenceTransformer(fallback)
            cfg.index_name = f"{cfg.index_name.rsplit('_', 1)[0]}_{slugify_model(fallback)}"
        self.embed_dim = self.embed_model.get_sentence_embedding_dimension()
        print(f"[INFO] Embedding dim: {self.embed_dim}")

        # Reranker
        self.reranker = None
        self.reranker_loaded = False
        if cfg.use_reranker:
            try:
                from sentence_transformers import CrossEncoder
                self.reranker = CrossEncoder(cfg.reranker_model)
                self.reranker_loaded = True
                print(f"[INFO] Loaded reranker: {cfg.reranker_model}")
            except Exception as e:
                print(f"[WARN] Reranker unavailable ({e}); rerank stage will be skipped.")
                self.reranker = None
        else:
            print("[INFO] Reranker disabled by --no-reranker.")

        # Elasticsearch
        self.es = Elasticsearch(
            [cfg.es_host],
            basic_auth=(cfg.es_username, cfg.es_password),
            ca_certs=cfg.es_ca_certs,
        )

        # OpenAI
        self.client = OpenAI()

    def embed(self, sentences):
        return self.embed_model.encode(sentences, normalize_embeddings=True)


# ============================================================
# Elasticsearch schema
# ============================================================
def build_settings() -> dict:
    return {
        "analysis": {
            "analyzer": {
                "nori": {
                    "type": "custom",
                    "tokenizer": "nori_tokenizer",
                    "decompound_mode": "mixed",
                    "filter": ["nori_posfilter", "lowercase"],
                }
            },
            "filter": {
                "nori_posfilter": {
                    "type": "nori_part_of_speech",
                    "stoptags": ["E", "J", "SC", "SE", "SF", "VCN", "VCP", "VX"],
                }
            },
        }
    }


def build_mappings(embed_dim: int) -> dict:
    return {
        "properties": {
            "docid": {"type": "keyword"},
            "src": {"type": "keyword"},
            "content": {"type": "text", "analyzer": "nori"},
            "embeddings": {
                "type": "dense_vector",
                "dims": embed_dim,
                "index": True,
                "similarity": "cosine",
            },
        }
    }


# ============================================================
# Indexing
# ============================================================
def index_documents(rt: Runtime):
    cfg = rt.cfg
    print(f"[INFO] Creating index '{cfg.index_name}'...")
    if rt.es.indices.exists(index=cfg.index_name):
        rt.es.indices.delete(index=cfg.index_name)
    rt.es.indices.create(
        index=cfg.index_name,
        settings=build_settings(),
        mappings=build_mappings(rt.embed_dim),
    )

    print(f"[INFO] Loading documents from {cfg.docs_path}")
    with open(cfg.docs_path) as f:
        docs = [json.loads(line) for line in f]
    print(f"[INFO] {len(docs)} documents loaded")

    print("[INFO] Generating embeddings...")
    for i in range(0, len(docs), 64):
        batch = docs[i:i + 64]
        contents = [d["content"] for d in batch]
        embs = rt.embed(contents).tolist()
        for d, e in zip(batch, embs):
            d["embeddings"] = e
        print(f"  embedded {i + len(batch)}/{len(docs)}")

    print("[INFO] Bulk indexing...")
    actions = [{"_index": cfg.index_name, "_source": d} for d in docs]
    ret = helpers.bulk(rt.es, actions)
    print(f"[INFO] Bulk result: {ret}")


# ============================================================
# Retrieval
# ============================================================
def sparse_search(rt: Runtime, query: str, size: int):
    return rt.es.search(
        index=rt.cfg.index_name,
        query={"match": {"content": {"query": query}}},
        size=size,
    )["hits"]["hits"]


def dense_search(rt: Runtime, query: str, size: int):
    q_emb = rt.embed([query])[0].tolist()
    return rt.es.search(
        index=rt.cfg.index_name,
        knn={
            "field": "embeddings",
            "query_vector": q_emb,
            "k": size,
            "num_candidates": max(size * 4, 100),
        },
        size=size,
    )["hits"]["hits"]


def rrf_fuse(result_lists: List[list], k: int = 60, weights: Optional[list] = None):
    scores, sources = {}, {}
    for idx, hits in enumerate(result_lists):
        w = weights[idx] if weights is not None else 1.0
        for rank, h in enumerate(hits, start=1):
            docid = h["_source"]["docid"]
            scores[docid] = scores.get(docid, 0.0) + w * 1.0 / (k + rank)
            if docid not in sources:
                sources[docid] = h["_source"]
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(docid, scores[docid], sources[docid]) for docid, _ in ranked]


def rerank_candidates(rt: Runtime, query: str, candidates, top_n: int):
    if rt.reranker is None or not candidates:
        return [(c[0], c[1], c[2]) for c in candidates[:top_n]]
    pairs = [(query, c[2]["content"]) for c in candidates]
    scores = rt.reranker.predict(pairs)
    rescored = [(c[0], float(s), c[2]) for c, s in zip(candidates, scores)]
    rescored.sort(key=lambda x: x[1], reverse=True)
    return rescored[:top_n]


def retrieve(rt: Runtime, query: str) -> List[Tuple[str, float, dict]]:
    cfg = rt.cfg
    mode = cfg.retrieval_mode

    dense_query = query
    if cfg.use_hyde:
        dense_query = generate_hyde_doc(rt, query)

    if mode == "sparse":
        pool_size = cfg.topk_rerank if cfg.use_reranker else cfg.topk_final
        hits = sparse_search(rt, query, pool_size)
        candidates = [(h["_source"]["docid"], h["_score"], h["_source"]) for h in hits]
    elif mode == "dense":
        pool_size = cfg.topk_rerank if cfg.use_reranker else cfg.topk_final
        hits = dense_search(rt, dense_query, pool_size)
        candidates = [(h["_source"]["docid"], h["_score"], h["_source"]) for h in hits]
    else:  # hybrid
        sparse_hits = sparse_search(rt, query, cfg.topk_sparse)
        dense_hits = dense_search(rt, dense_query, cfg.topk_dense)
        fused = rrf_fuse(
            [sparse_hits, dense_hits],
            k=cfg.rrf_k,
            weights=[cfg.rrf_weight_sparse, cfg.rrf_weight_dense],
        )
        candidates = fused[:cfg.topk_rerank] if cfg.use_reranker else fused[:cfg.topk_final]

    if cfg.use_reranker and rt.reranker is not None:
        reranked = rerank_candidates(rt, query, candidates, cfg.topk_final)
    else:
        reranked = candidates[:cfg.topk_final]

    if cfg.score_threshold > 0 and reranked and reranked[0][1] < cfg.score_threshold:
        return []
    return reranked


# ============================================================
# LLM prompts
# ============================================================
CLASSIFY_SYSTEM = """너는 과학 지식 질의응답 시스템의 질의 분석기다.

## 판단 규칙
1) 입력된 대화 메시지(멀티턴 가능)가 과학/자연/공학/CS 지식을 묻는 질문인지 분류하라.
   - 과학 범주:
     * 자연과학: 물리, 화학, 생물, 지구과학, 천문, 생태, 환경
     * 의학/보건: 의학, 약학, 생리, 질병, 치료법, 피임·건강 관리 방법
     * 공학/기술: 공학, 기술, 재료, 에너지, 수학 기반 자연과학
     * **컴퓨터 과학 전반**: 알고리즘, 자료구조, 암호학·해시, 정보이론, 계산 복잡도, 오토마타, 컴퓨터 보안 이론
     * **프로그래밍·코딩 질문**: 알고리즘 구현(예: merge sort, 정렬·탐색), 자료구조 사용(list, dict), 언어 문법·기능(lambda, 클래스, 예외 처리), 코드 작성 요청, 소프트웨어 공학(개발 방법론, 편향, 테스트)
     * **과학 연구 방법론**: 실험 설계, 관찰, 기록, 재현성, 과학적 사고방식
     * **과학자·연구자 관련 질문** (예: "Dmitri Ivanovsky가 누구야?", "뉴턴의 업적은?")
     * 과학적 원리·현상을 묻는 모든 질문
   - 비과학 범주:
     * 일상 잡담, 감정 표현, 인사, 감탄, 칭찬, 위로 요청
     * 사회/정치/경제/교육 정책, 일반 역사(과학자·과학사 외)
     * 취향·의견 질문, 연애·인간관계 조언
     * 특정 제품·서비스 추천, 여행·요리 등 생활 정보
2) 과학 질문이라면 대화 이력 전체 맥락을 반영해 **독립형 질의(standalone query)** 를 한국어로 재작성하라.
   - "그 현상", "이 사건", "그거" 등 지시대명사는 이전 발화를 해석해 구체 명사로 치환할 것
   - 검색 엔진이 이해할 수 있도록 핵심 키워드를 포함한 단일 문장으로 작성할 것
   - 프로그래밍 질문은 코드 자체가 아니라 **개념·목적·원리**를 키워드로 뽑을 것
     (예: "lambda 함수 언제 써?" → "Python lambda 함수의 용도와 사용 시점")
3) 과학 질문이 아니라면 standalone_query는 빈 문자열("")로 둔다.

## 출력 형식
반드시 아래 JSON 스키마만 출력한다 (다른 설명 금지):
{"is_science": true|false, "standalone_query": "..."}
"""

REWRITE_SYSTEM = """너는 멀티턴 대화의 마지막 질문을 검색엔진에 넣을 수 있는 독립형 한국어 질의로 재작성하는 도구다.
대화 맥락을 반영해 지시대명사를 구체 명사로 치환하고, 핵심 키워드를 포함한 한 문장으로 출력하라.
설명 없이 질의 문자열만 출력한다."""

ANSWER_SYSTEM = """## Role: 과학 상식 전문가

## Instructions
- 반드시 아래 Reference 정보 안에서만 답변을 구성한다.
- Reference에 근거 정보가 없다면 "제공된 정보로는 답변할 수 없습니다"라고 답한다.
- 외부 지식이나 추측으로 답을 만들어내지 않는다 (환각 금지).
- 한국어로 간결하고 정확하게 답변한다.
"""

CHITCHAT_SYSTEM = """너는 사용자와 자연스럽게 대화하는 한국어 어시스턴트다.
상대방의 감정, 상황에 공감하며 짧고 따뜻한 답변을 제공하라."""

HYDE_SYSTEM = """너는 과학 지식 질문에 대해 가상의 답변 문서를 생성하는 도구다.
입력된 질문에 대해 실제 사실 여부와 무관하게, 해당 질문의 답변이 될 만한 3~4 문장의 한국어 지식 문서를 작성하라.
- 문서 스타일로 작성 (설명문, "~이다", "~한다" 체)
- 질문의 핵심 키워드를 자연스럽게 포함
- 답변 문서만 출력, 다른 설명 금지"""


def classify_and_rewrite(rt: Runtime, messages):
    resp = rt.client.chat.completions.create(
        model=rt.cfg.llm_model,
        messages=[{"role": "system", "content": CLASSIFY_SYSTEM}] + messages,
        temperature=0,
        seed=1,
        response_format={"type": "json_object"},
        timeout=15,
    )
    return json.loads(resp.choices[0].message.content)


def rewrite_only(rt: Runtime, messages) -> str:
    resp = rt.client.chat.completions.create(
        model=rt.cfg.llm_model,
        messages=[{"role": "system", "content": REWRITE_SYSTEM}] + messages,
        temperature=0,
        seed=1,
        timeout=15,
    )
    return resp.choices[0].message.content.strip()


def generate_hyde_doc(rt: Runtime, query: str) -> str:
    try:
        resp = rt.client.chat.completions.create(
            model=rt.cfg.llm_model,
            messages=[
                {"role": "system", "content": HYDE_SYSTEM},
                {"role": "user", "content": query},
            ],
            temperature=0,
            seed=1,
            timeout=15,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"[WARN] HyDE 생성 실패 ({e}), 원본 쿼리 사용")
        return query


def generate_answer_with_context(rt: Runtime, messages, refs):
    ctx = "\n\n".join([f"[문서 {i+1}] {r['content']}" for i, r in enumerate(refs)])
    sys_prompt = ANSWER_SYSTEM + "\n\n## Reference\n" + ctx
    resp = rt.client.chat.completions.create(
        model=rt.cfg.llm_model,
        messages=[{"role": "system", "content": sys_prompt}] + messages,
        temperature=0,
        seed=1,
        timeout=30,
    )
    return resp.choices[0].message.content


def generate_chitchat(rt: Runtime, messages):
    resp = rt.client.chat.completions.create(
        model=rt.cfg.llm_model,
        messages=[{"role": "system", "content": CHITCHAT_SYSTEM}] + messages,
        temperature=0.3,
        seed=1,
        timeout=20,
    )
    return resp.choices[0].message.content


# ============================================================
# End-to-end answering
# ============================================================
def answer_question(rt: Runtime, messages):
    response = {"standalone_query": "", "topk": [], "references": [], "answer": ""}

    if rt.cfg.classifier_mode == "json":
        try:
            cls = classify_and_rewrite(rt, messages)
        except Exception as e:
            traceback.print_exc()
            response["answer"] = f"[ERROR-classify] {type(e).__name__}: {e}"
            return response

        if not cls.get("is_science"):
            try:
                response["answer"] = generate_chitchat(rt, messages)
            except Exception as e:
                traceback.print_exc()
                response["answer"] = f"[ERROR-chitchat] {type(e).__name__}: {e}"
            return response

        sq = (cls.get("standalone_query") or "").strip()
    else:
        try:
            sq = rewrite_only(rt, messages)
        except Exception as e:
            traceback.print_exc()
            response["answer"] = f"[ERROR-rewrite] {type(e).__name__}: {e}"
            return response

    response["standalone_query"] = sq
    if not sq:
        response["answer"] = "[ERROR] empty standalone_query"
        return response

    try:
        final = retrieve(rt, sq)
    except Exception as e:
        traceback.print_exc()
        response["answer"] = f"[ERROR-retrieve] {type(e).__name__}: {e}"
        return response

    refs = []
    for docid, score, src in final:
        response["topk"].append(docid)
        refs.append({"score": score, "content": src["content"]})
    response["references"] = refs

    try:
        response["answer"] = generate_answer_with_context(rt, messages, refs)
    except Exception as e:
        traceback.print_exc()
        response["answer"] = f"[ERROR-answer] {type(e).__name__}: {e}"

    return response


# ============================================================
# Eval loop
# ============================================================
def eval_rag(rt: Runtime):
    cfg = rt.cfg
    out_dir = cfg.output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    submission_path = out_dir / "submission.csv"

    started = time.time()
    with open(cfg.eval_path) as f, open(submission_path, "w") as of:
        for idx, line in enumerate(f):
            j = json.loads(line)
            print(f"\n=== Test {idx} (eval_id={j['eval_id']}) ===")
            print(f"Question: {j['msg']}")
            resp = answer_question(rt, j["msg"])
            print(f"SQ     : {resp['standalone_query']}")
            print(f"TopK   : {resp['topk']}")
            print(f"Answer : {resp['answer'][:120]}")

            row = {
                "eval_id": j["eval_id"],
                "standalone_query": resp["standalone_query"],
                "topk": resp["topk"],
                "answer": resp["answer"],
                "references": resp["references"],
            }
            of.write(json.dumps(row, ensure_ascii=False) + "\n")
            of.flush()
    return time.time() - started, submission_path


def write_artifacts(rt: Runtime, duration: float, submission_path: Path):
    out_dir = rt.cfg.output_dir()

    config_path = out_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(rt.cfg.to_dict(), f, ensure_ascii=False, indent=2)

    meta = {
        "exp_name": rt.cfg.exp_name,
        "duration_sec": round(duration, 2),
        "embed_model_loaded": rt.embed_model_name,
        "embed_dim": rt.embed_dim,
        "reranker_requested": rt.cfg.use_reranker,
        "reranker_loaded": rt.reranker_loaded,
        "reranker_model": rt.cfg.reranker_model if rt.reranker_loaded else None,
        "index_name": rt.cfg.index_name,
        "submission_path": str(submission_path.relative_to(HERE)),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\n[INFO] Artifacts written to {out_dir}/")
    print(f"  - submission.csv")
    print(f"  - config.json")
    print(f"  - meta.json")


# ============================================================
# Entrypoint
# ============================================================
def main():
    cfg = parse_args()
    print(f"[INFO] Experiment: {cfg.exp_name}")
    print(f"[INFO] Config: {json.dumps(cfg.to_dict(), ensure_ascii=False, indent=2)}")

    rt = Runtime(cfg)
    print(f"[INFO] ES info: {rt.es.info()}")

    cfg.output_dir().mkdir(parents=True, exist_ok=True)

    if cfg.mode in ("index", "all"):
        t0 = time.time()
        index_documents(rt)
        print(f"[INFO] Indexing took {time.time() - t0:.1f}s")

    if cfg.mode in ("eval", "all"):
        duration, submission_path = eval_rag(rt)
        write_artifacts(rt, duration, submission_path)
        print(f"\n[INFO] Eval took {duration:.1f}s")
        print(f"[INFO] Submission: {submission_path}")


if __name__ == "__main__":
    main()