"""
RAG pipeline (EXP-17, 최고점 setup baked-in)

Pipeline (--classifier=gate, default):
    1) LLM classifier로 의도 사전 분류 → 비과학이면 retrieve 스킵 + topk=[]
    2) rewrite_only로 SQ 생성
    3) Multi-query 확장 (n=3): SQ → paraphrase 생성 → 각각 retrieve → RRF fuse
    4) Hybrid retrieval (BM25 + bge-m3 dense) + RRF fuse
    5) bge-reranker-v2-m3 cross-encoder 리랭킹 → topk 3건 (score-threshold 비활성, default 0.0)

사용법:
    # 1) ES 인덱싱 (bge-m3 임베딩, 1회만 실행)
    python rag.py index --exp EXP-INDEX-001

    # 2) 평가 (default = 최고점 setup. answer LLM provider만 명시)
    python rag.py eval --exp EXP-02-003-17 --answer-llm-provider solar

    # 실험 ablation
    #   --classifier off    의도 분류 끄기 (baseline)
    #   --classifier full   classifier가 SQ까지 생성
    #   --no-multi-query    multi-query 끄기
"""

import os
import re
import sys
import json
import time
import hashlib
import argparse
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple, Optional

from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
load_dotenv(HERE / ".env")

from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# ============================================================
# Config
# ============================================================
@dataclass
class Config:
    # Mode
    mode: str = "eval"                    # index | eval | all
    exp_name: str = "default"

    # Retrieval
    retrieval_mode: str = "hybrid"        # sparse | dense | hybrid
    use_reranker: bool = True
    # off:  intent gate 없음, rewrite_only가 SQ 생성 (baseline ablation)
    # gate: intent gate 있음, rewrite_only가 SQ 생성 (최고점 setup, default)
    # full: intent gate 있음, classifier가 SQ 생성
    classifier_mode: str = "gate"

    # Models
    embed_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    llm_model: str = "gpt-4o-mini"

    # Retrieval params (최고점 setup 기본값)
    topk_sparse: int = 50
    topk_dense: int = 50
    topk_rerank: int = 20
    topk_final: int = 3
    rrf_k: int = 60
    score_threshold: float = 0.0          # 0.0 = 비활성. classifier가 chitchat 단일 소스, 과학 질의는 무조건 topk 3건
    rrf_weight_sparse: float = 1.0
    rrf_weight_dense: float = 1.0

    multi_query: bool = True
    multi_query_n: int = 3
    multi_query_include_original: bool = True

    # HyDE: 가상 답변 문서를 dense 검색 쿼리로 사용 (sparse는 원본 유지)
    # 도메인 중립 사전·백과사전 문체 → corpus의 역사·사회·지리 문서와 의미 거리 좁힘
    # mq3-tau002에서 +0.009 기여 추정 (212, 305 케이스)
    use_hyde: bool = False

    # LLM listwise rerank (bge-reranker 이후 2차 리랭크)
    # bge top-K → LLM이 listwise로 재정렬 → final top-3
    # solar-pro 기본(한국어 최적화 + 비용 ↓). gpt-4o-mini / gpt-4o도 선택 가능
    llm_rerank: bool = False
    llm_rerank_topk: int = 10            # bge-reranker에서 LLM으로 넘길 후보 수
    llm_rerank_provider: str = "solar"   # openai | solar
    llm_rerank_model: str = ""           # 빈값이면 provider별 default (solar→solar-pro, openai→gpt-4o-mini)

    # Answer/chitchat 생성용 LLM (retrieval-path LLM과 분리, MAP 무관)
    answer_llm_provider: str = "openai"   # openai | solar
    answer_llm_model: str = ""            # 빈 값이면 llm_model 공유

    # LLM 응답 디스크 캐시 (재현성용, 기본 off)
    # ON: rag_system/.llm_cache/<sha256>.json — 같은 입력이면 항상 같은 응답.
    # OpenAI seed가 best-effort라 multi-query/SQ 생성에 비결정성이 들어옴.
    # 캐시를 켜면 첫 실행에서 채우고, 이후 실행은 100% 동일 결과.
    llm_cache: bool = False

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
    """BAAI/bge-m3 -> bge_m3"""
    tail = name.split("/")[-1].lower()
    tail = re.sub(r"[^a-z0-9]+", "_", tail).strip("_")
    return tail[:40]


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="RAG pipeline (EXP-17, 최고점 setup 기본값)")
    p.add_argument("mode", choices=["index", "eval", "all"])
    p.add_argument("--exp", dest="exp_name", default="default",
                   help="Experiment name (used for output dir)")

    # Retrieval (defaults는 최고점 setup)
    p.add_argument("--retrieval-mode", choices=["sparse", "dense", "hybrid"], default="hybrid")
    p.add_argument("--no-reranker", action="store_true", help="Disable cross-encoder reranker")
    p.add_argument("--classifier", choices=["off", "gate", "full"], default="gate",
                   help="off: 분류 없음 / gate(default): 분류+rewrite_only SQ / full: 분류+classifier SQ")

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
    p.add_argument("--score-threshold", type=float, default=0.0,
                   help="rerank top-1 score < 이 값이면 topk=[] (default 0.0 = 비활성, 분류는 classifier 단일)")
    p.add_argument("--rrf-weight-sparse", type=float, default=1.0)
    p.add_argument("--rrf-weight-dense", type=float, default=1.0)

    # Multi-query (default ON, n=3)
    p.add_argument("--no-multi-query", action="store_true",
                   help="기본 ON. multi-query paraphrase fusion 비활성화 시 사용")
    p.add_argument("--multi-query-n", type=int, default=3)
    p.add_argument("--no-multi-query-original", action="store_true")

    # HyDE (default OFF — 부활 시도용)
    p.add_argument("--hyde", action="store_true",
                   help="HyDE: dense 검색 쿼리에 가상 답변 문서 사용 (sparse는 원본 유지)")

    # LLM listwise rerank (bge-reranker 이후 2차)
    p.add_argument("--llm-rerank", action="store_true",
                   help="bge-reranker 결과를 LLM으로 한 번 더 listwise rerank")
    p.add_argument("--llm-rerank-topk", type=int, default=10,
                   help="bge에서 LLM으로 넘길 후보 수 (default 10)")
    p.add_argument("--llm-rerank-provider", choices=["openai", "solar"], default="solar",
                   help="LLM rerank provider (default: solar)")
    p.add_argument("--llm-rerank-model", default=None,
                   help="빈값이면 solar→solar-pro, openai→gpt-4o-mini")

    # Answer LLM (MAP 무관, 비용 절감용 분리)
    p.add_argument("--answer-llm-provider", choices=["openai", "solar"], default="openai")
    p.add_argument("--answer-llm-model", default=None)

    # LLM 응답 디스크 캐시 (재현성용)
    p.add_argument("--llm-cache", action="store_true",
                   help="LLM 응답을 .llm_cache/에 디스크 캐시 (결정성 확보, 기본 off)")

    # Paths
    p.add_argument("--index-name", default=None)
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
    cfg.rrf_weight_sparse = args.rrf_weight_sparse
    cfg.rrf_weight_dense = args.rrf_weight_dense
    cfg.multi_query = not args.no_multi_query
    cfg.multi_query_n = args.multi_query_n
    cfg.multi_query_include_original = not args.no_multi_query_original
    cfg.use_hyde = args.hyde
    cfg.llm_rerank = args.llm_rerank
    cfg.llm_rerank_topk = args.llm_rerank_topk
    cfg.llm_rerank_provider = args.llm_rerank_provider
    if args.llm_rerank_model is not None:
        cfg.llm_rerank_model = args.llm_rerank_model
    cfg.answer_llm_provider = args.answer_llm_provider
    if args.answer_llm_model is not None:
        cfg.answer_llm_model = args.answer_llm_model
    cfg.llm_cache = args.llm_cache

    cfg.docs_path = args.docs_path or os.environ.get("DOCS_PATH", cfg.docs_path)
    cfg.eval_path = args.eval_path or os.environ.get("EVAL_PATH", cfg.eval_path)

    if args.index_name:
        cfg.index_name = args.index_name
    else:
        base = os.environ.get("ES_INDEX", "science_kb")
        cfg.index_name = f"{base}_{slugify_model(cfg.embed_model)}"

    # ES credentials from env
    cfg.es_username = os.environ["ES_USERNAME"]
    cfg.es_password = os.environ["ES_PASSWORD"]
    cfg.es_host = os.environ.get("ES_HOST", cfg.es_host)
    cfg.es_ca_certs = os.environ.get("ES_CA_CERTS", cfg.es_ca_certs)

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set in .env")

    return cfg


# ============================================================
# Runtime state (populated based on config)
# ============================================================
class Runtime:
    def __init__(self, cfg: Config):
        self.cfg = cfg

        # Embedding (bge-m3 등 SentenceTransformer 모델, ES dense_vector 사용)
        self.embed_model_name = cfg.embed_model
        self.embed_model = SentenceTransformer(cfg.embed_model)
        self.embed_dim = self.embed_model.get_sentence_embedding_dimension()
        self.embed_loaded = True
        print(f"[INFO] Loaded embedding model: {cfg.embed_model} (dim={self.embed_dim})")

        # Reranker (Cross-encoder)
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
        else:
            print("[INFO] Reranker disabled by --no-reranker.")

        # Elasticsearch
        self.es = Elasticsearch(
            [cfg.es_host],
            basic_auth=(cfg.es_username, cfg.es_password),
            ca_certs=cfg.es_ca_certs,
        )

        # OpenAI (retrieval-path: classify, SQ rewrite, multi-query variants)
        self.client = OpenAI()

        # Answer/chitchat 생성용 client+model — provider별 분기 (MAP 무관)
        self.answer_client = self.client
        self.answer_model = cfg.answer_llm_model or cfg.llm_model
        if cfg.answer_llm_provider == "solar":
            api_key = os.environ.get("UPSTAGE_API_KEY")
            if not api_key:
                raise RuntimeError("UPSTAGE_API_KEY not set (required for answer_llm_provider=solar)")
            self.answer_client = OpenAI(api_key=api_key, base_url="https://api.upstage.ai/v1")
            self.answer_model = cfg.answer_llm_model or "solar-pro"
            print(f"[INFO] Answer LLM routed to Solar: model={self.answer_model}")

        # LLM rerank용 client+model — provider별 분기 (bge 이후 2차 리랭크)
        self.rerank_client = self.client
        self.rerank_model = cfg.llm_rerank_model or "gpt-4o-mini"
        if cfg.llm_rerank:
            if cfg.llm_rerank_provider == "solar":
                # answer_client가 이미 solar면 재사용, 아니면 새로 생성
                if cfg.answer_llm_provider == "solar":
                    self.rerank_client = self.answer_client
                else:
                    api_key = os.environ.get("UPSTAGE_API_KEY")
                    if not api_key:
                        raise RuntimeError("UPSTAGE_API_KEY not set (required for llm_rerank_provider=solar)")
                    self.rerank_client = OpenAI(api_key=api_key, base_url="https://api.upstage.ai/v1")
                self.rerank_model = cfg.llm_rerank_model or "solar-pro"
            else:
                self.rerank_client = self.client
                self.rerank_model = cfg.llm_rerank_model or "gpt-4o-mini"
            print(f"[INFO] LLM rerank: provider={cfg.llm_rerank_provider}, model={self.rerank_model}, topk={cfg.llm_rerank_topk}")

    def embed(self, sentences, model_type: str = "passage"):
        """sentences: List[str], returns: ndarray (N, dim), L2-normalized"""
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


def build_mappings(embed_dim: int, include_dense: bool = True) -> dict:
    props = {
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
    return {"properties": props}


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
        mappings=build_mappings(rt.embed_dim, include_dense=True),
    )

    print(f"[INFO] Loading documents from {cfg.docs_path}")
    with open(cfg.docs_path) as f:
        docs = [json.loads(line) for line in f]
    print(f"[INFO] {len(docs)} documents loaded")

    print("[INFO] Generating embeddings...")
    batch_size = 64
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        contents = [d["content"] for d in batch]
        embs = rt.embed(contents, model_type="passage")
        for d, e in zip(batch, embs.tolist()):
            d["embeddings"] = e
        print(f"  embedded {i + len(batch)}/{len(docs)}")

    print("[INFO] Bulk indexing to ES...")
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
    """bge-m3 ES KNN dense retrieval"""
    q_emb = rt.embed([query], model_type="query")[0].tolist()
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
    """Weighted Reciprocal Rank Fusion"""
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
    if not candidates or not rt.reranker_loaded:
        return [(c[0], c[1], c[2]) for c in candidates[:top_n]]
    pairs = [(query, c[2]["content"]) for c in candidates]
    scores = rt.reranker.predict(pairs)
    rescored = [(c[0], float(s), c[2]) for c, s in zip(candidates, scores)]
    rescored.sort(key=lambda x: x[1], reverse=True)
    return rescored[:top_n]


def retrieve(rt: Runtime, query: str) -> List[Tuple[str, float, dict]]:
    cfg = rt.cfg
    mode = cfg.retrieval_mode

    # Multi-query: SQ 1개 → N개 paraphrase 생성 → 각각 retrieve 후 RRF fuse
    # SQ 한 번의 운에 의존하지 않고 reranker 입력 후보 안정화
    # HyDE on이면 각 쿼리마다 dense 경로에서 가상 답변 문서로 대체 (sparse는 원본 유지)
    if cfg.multi_query and mode == "hybrid":
        variants = generate_query_variants(rt, query, cfg.multi_query_n)
        queries = ([query] + variants) if cfg.multi_query_include_original else variants
        if not queries:
            queries = [query]
        all_lists = []
        weights = []
        for q in queries:
            dq = generate_hyde_doc(rt, q) if cfg.use_hyde else q
            sparse_hits = sparse_search(rt, q, cfg.topk_sparse)
            dense_hits = dense_search(rt, dq, cfg.topk_dense)
            all_lists.append(sparse_hits)
            all_lists.append(dense_hits)
            weights.append(cfg.rrf_weight_sparse)
            weights.append(cfg.rrf_weight_dense)
        fused = rrf_fuse(all_lists, k=cfg.rrf_k, weights=weights)
        candidates = fused[:cfg.topk_rerank] if cfg.use_reranker else fused[:cfg.topk_final]
    elif mode == "sparse":
        pool_size = cfg.topk_rerank if cfg.use_reranker else cfg.topk_final
        hits = sparse_search(rt, query, pool_size)
        candidates = [(h["_source"]["docid"], h["_score"], h["_source"]) for h in hits]
    elif mode == "dense":
        pool_size = cfg.topk_rerank if cfg.use_reranker else cfg.topk_final
        dense_query = generate_hyde_doc(rt, query) if cfg.use_hyde else query
        hits = dense_search(rt, dense_query, pool_size)
        candidates = [(h["_source"]["docid"], h["_score"], h["_source"]) for h in hits]
    else:  # hybrid (no multi-query)
        dense_query = generate_hyde_doc(rt, query) if cfg.use_hyde else query
        sparse_hits = sparse_search(rt, query, cfg.topk_sparse)
        dense_hits = dense_search(rt, dense_query, cfg.topk_dense)
        fused = rrf_fuse(
            [sparse_hits, dense_hits],
            k=cfg.rrf_k,
            weights=[cfg.rrf_weight_sparse, cfg.rrf_weight_dense],
        )
        candidates = fused[:cfg.topk_rerank] if cfg.use_reranker else fused[:cfg.topk_final]

    # 리랭킹 + score threshold (top-1 score < τ → 관련 문서 없음 = empty topk)
    if cfg.use_reranker and rt.reranker_loaded:
        # LLM 2차 rerank 켜져 있으면 bge는 topk_final 대신 더 넓게 (llm_rerank_topk)
        bge_top = cfg.llm_rerank_topk if cfg.llm_rerank else cfg.topk_final
        reranked = rerank_candidates(rt, query, candidates, bge_top)
        if cfg.score_threshold > 0 and reranked and reranked[0][1] < cfg.score_threshold:
            return []
        if cfg.llm_rerank and reranked:
            reranked = llm_rerank(rt, query, reranked, cfg.topk_final)
        else:
            reranked = reranked[:cfg.topk_final]
    else:
        reranked = candidates[:cfg.topk_final]
        if cfg.score_threshold > 0 and reranked and reranked[0][1] < cfg.score_threshold:
            return []
    return reranked


# ============================================================
# LLM prompts
# ============================================================
CLASSIFY_SYSTEM = """너는 광범위 지식 RAG 시스템의 질의 분석기다.

## 핵심 원칙 (가장 중요)
**기본값은 "is_science=true"다.** 입력이 객관적 정보를 묻거나 사실 지식을 요구하면 무조건 과학으로 분류한다.
주제 분야는 무관하다 — 자연과학·인문·역사·정치·경제·사회·법·철학·예술·생활상식·특정 인물·특정 사건 모두 과학.
비과학(is_science=false)은 아래 3종 패턴에 **명확히 해당될 때만** 분류한다:

## 비과학 판정 기준 (3종 한정)
A. **1인칭 감정 토로** — 화자가 자신의 현재 기분·상태를 표현하는 발화에만 한정
   - 비과학 예시 (1인칭, 자기 상태 표현):
     "요새 너무 힘들다", "우울해", "외로워", "지쳤어", "오늘 너무 즐거웠어", "기분이 좋아"
   - **과학 예시 (3자 객관 지식 질문, "어떤·무엇·원인·형태·분류"류):**
     "사회적 지원의 형태는?" → 과학 (사회학)
     "감정 노동이란 무엇인가?" → 과학 (심리학)
     "스트레스 호르몬의 작용 원리" → 과학 (생리학)
     "수면이 건강에 좋은 이유" → 과학 (의학)
     "우울증의 증상은?" → 과학 (정신의학)
   - 핵심 규칙: "**나/내가**"가 주어이고 본인 감정을 진술하는 발화만 비과학.
     "어떤·무엇·왜·어떻게"로 객관 정보를 묻는 질문은 주제가 감정·심리여도 과학.

B. **AI 챗봇 메타 발화** — 주체가 "너/당신/AI 챗봇" 인 발화에 한정
   - AI 자기지칭 질문: "너는 누구야?", "너 잘하는게 뭐야?", "너 모르는 것도 있니?", "너의 특기는?"
   - AI 칭찬·평가: "너 정말 똑똑하다", "대답 잘해줘서 기분이 좋아", "너 대단해"
   - ※ "Dmitri Ivanovsky가 누구야?", "벽돌공 일대기"처럼 **특정 사람·인물**을 묻는 건 과학

C. **단순 인사·종료 발화**
   - 인사: "안녕", "안녕 반가워", "잘 가"
   - 종료: "이제 그만 얘기하자", "그만 얘기해!"

## 과학 분류 보호 예시 (이런 류는 절대 비과학으로 분류 금지)
- "이란-콘트라 사건이 뭐야?" → 과학 (역사·정치 사건 = 객관 지식)
- "통학 버스의 가치는?" → 과학 (사회·생활 지식)
- "남미의 라틴 역사" → 과학 (역사)
- "프랑스 혁명의 원인" → 과학 (역사)
- "민주주의의 정의는?" → 과학 (정치학)
- "벽돌공 일대기" → 과학 (인물 정보)
- "선한 영향력을 준 인물의 일대기" → 과학 (motivational/감동 수식어 있어도 인물 정보 질문)
- "위인의 생애를 알려줘" → 과학 (위인전·전기 정보)
- "Dmitri Ivanovsky가 누구야?" → 과학 (인물)
- "확률 계산 예시 알려줘" → 과학 (수학·통계, "예시 알려줘" 캐주얼 톤이어도 객관 지식)
- "두 소스에서 발생한 사건의 기인 확률" → 과학 (확률·통계)
- "잠을 잘 잤을 때 이로운 점을 나열해줘" → 과학 (수면 생리·건강, "이로운 점 나열" 가이드 톤이어도 객관 지식)
- "운동의 장점은?" → 과학 (건강·운동생리학)

## 톤·문체 함정 주의
"~알려줘", "~나열해줘", "~예시", "~방법", "~장점/단점" 같은 캐주얼·가이드 문체가 들어가도
객관 지식·사실을 묻는 질문이면 **모두 과학**. 친근한 말투로 헷갈리지 말 것.

## 작업
1) 위 기준으로 is_science 판정.
2) 과학이면 대화 이력 맥락을 반영해 standalone_query를 한국어 한 문장으로 재작성.
   - 지시대명사(그것, 이거, 그 사건)는 이전 발화를 해석해 구체 명사로 치환.
   - **의문문 형태 강제**: 끝맺음은 "?", "~인가?", "~무엇인가?", "~어떻게?", "~왜?" 등.
   - 답을 SQ에 박지 말 것. 평서문("~이다")·요청문("~알려주세요") 금지.
3) 비과학이면 standalone_query는 "" (빈 문자열).

## 출력 형식
JSON만 출력 (다른 설명 금지):
{"is_science": true|false, "standalone_query": "..."}
"""

REWRITE_SYSTEM = """너는 멀티턴 대화의 마지막 질문을 검색엔진에 넣을 수 있는 독립형 한국어 질의로 재작성하는 도구다.
대화 맥락을 반영해 지시대명사를 구체 명사로 치환하고, 핵심 키워드를 포함한 한 문장으로 출력하라.
설명 없이 질의 문자열만 출력한다.

## 출력 형식 규칙 (필수)
- 반드시 의문문 형태로 작성한다. 끝맺음은 "?", "~인가?", "~무엇인가?", "~어떻게?", "~왜?" 등.
- 답을 미리 SQ에 박지 말 것. (X) "X는 Y이다" / (O) "X는 무엇인가?"
- 평서문(~이다/~입니다/~된다)·요청문(~알려주세요/~설명을 요청합니다) 금지."""

ANSWER_SYSTEM = """## Role: 과학 상식 전문가

## Instructions
- 반드시 아래 Reference 정보 안에서만 답변을 구성한다.
- Reference에 근거 정보가 없다면 "제공된 정보로는 답변할 수 없습니다"라고 답한다.
- 외부 지식이나 추측으로 답을 만들어내지 않는다 (환각 금지).
- 한국어로 간결하고 정확하게 답변한다.
"""

CHITCHAT_SYSTEM = """너는 사용자와 자연스럽게 대화하는 한국어 어시스턴트다.
1문장으로 응답하라. 인사·공감·짧은 위로만 한다. 부연 설명·예시·길게 설명 금지."""


# --- LLM listwise rerank ---
# bge-reranker의 top-K(기본 10) 결과를 LLM에 한 번에 주고 관련성 순으로 재정렬.
# cross-encoder의 의미 유사도 점수만으로 잡기 어려운 주제·의도 부합도 판정 강화.
LLM_RERANK_SYSTEM = """너는 검색 결과 재정렬기다. 쿼리에 대해 각 문서의 관련성을 평가하고 가장 관련 있는 순서로 재정렬하라.

## 평가 기준
1) 쿼리의 핵심 정보·사실을 직접 담은 문서를 우선
2) 단순 키워드 일치가 아닌 의미적 관련성 우선
3) 쿼리 의도에 답이 되는 문서 > 주제만 스쳐가는 문서

## 입력 형식
- 쿼리 1개
- 번호가 매겨진 문서 N개 (1부터 시작)

## 출력 형식 (JSON만, 다른 설명 금지)
{"ranking": [가장 관련 있는 문서부터 순서대로 N개 정수]}
예: {"ranking": [3, 1, 5, 2, 4, ...]}"""


# --- HyDE ---
# 도메인 중립 + 사전·백과사전 문체로 corpus 문서 스타일과 정렬.
# 역사·사회·지리 등 비STEM 질의에서 dense embedding이 GT 문서와 거리 좁히도록 유도.
HYDE_SYSTEM = """너는 입력된 한국어 질문에 대해 가상의 답변 문서를 생성하는 도구다.
질문의 답변이 될 만한 3~4 문장의 한국어 사전·백과사전 문체 문서를 작성하라.
- 문서 스타일로 작성 (설명문, "~이다", "~한다" 체)
- 질문의 핵심 키워드를 자연스럽게 포함
- 사실 여부와 무관하게 그럴듯한 답변 형태로 작성
- 답변 문서만 출력, 다른 설명 금지"""


# --- Multi-query ---
MULTI_QUERY_SYSTEM = """너는 검색 엔진 질의 다양화기다. 입력된 한국어 질의의 의미를 그대로 보존하면서 표현만 달리한 N개의 paraphrase를 생성하라.

## 생성 규칙
1) 각 paraphrase는 같은 정보를 묻되, 어휘·문체·문장구조를 다르게 해야 한다
2) 핵심 키워드(고유명사, 전문 용어)는 유지하되 일반 표현은 동의어 활용 가능
3) 너무 짧거나 너무 길지 않게 (원본과 비슷한 길이)
4) 질문형/평서형/키워드 나열형 등을 섞어서 검색 다양성 확보

## 출력 형식
반드시 아래 JSON 스키마만 출력 (다른 설명 금지):
{"variants": ["문장1", "문장2", ...]}"""


def llm_rerank(rt: Runtime, query: str, candidates: List[Tuple[str, float, dict]], top_k: int) -> List[Tuple[str, float, dict]]:
    """bge-reranker top-K 결과를 LLM에 listwise로 재정렬.

    candidates: [(docid, bge_score, src_dict)] 리스트 (bge-reranker가 이미 정렬)
    실패 시 bge 원순서 유지 (top_k로 잘라서 반환).
    """
    if not candidates:
        return []
    docs_text = "\n".join(
        f"{i+1}. {c[2].get('content', '')[:600]}" for i, c in enumerate(candidates)
    )
    user_msg = f"[쿼리]\n{query}\n\n[문서 목록]\n{docs_text}"
    try:
        content = chat_completion(
            rt,
            messages=[
                {"role": "system", "content": LLM_RERANK_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            temperature=0,
            seed=1,
            response_format={"type": "json_object"},
            timeout=30,
            client=rt.rerank_client,
            model=rt.rerank_model,
        )
        ranking = json.loads(content).get("ranking", [])
        new_order = []
        seen = set()
        for rank_idx in ranking:
            idx = int(rank_idx) - 1  # 1-indexed → 0-indexed
            if 0 <= idx < len(candidates) and idx not in seen:
                new_order.append(candidates[idx])
                seen.add(idx)
        # LLM이 누락한 후보는 bge 원순서대로 꼬리에 추가 (안전망)
        for i, c in enumerate(candidates):
            if i not in seen:
                new_order.append(c)
        return new_order[:top_k]
    except Exception as e:
        print(f"[WARN] LLM rerank 실패 ({e}), bge 순서 유지")
        return candidates[:top_k]


def generate_hyde_doc(rt: Runtime, query: str) -> str:
    """HyDE: 질의에 대한 가상 답변 문서 생성. dense retrieval 쿼리로 사용. 실패 시 원본 반환."""
    try:
        content = chat_completion(
            rt,
            messages=[
                {"role": "system", "content": HYDE_SYSTEM},
                {"role": "user", "content": query},
            ],
            temperature=0,
            seed=1,
            timeout=15,
        )
        return content.strip()
    except Exception as e:
        print(f"[WARN] HyDE 생성 실패 ({e}), 원본 쿼리 사용")
        return query


def generate_query_variants(rt: Runtime, sq: str, n: int) -> List[str]:
    """SQ에서 n개 paraphrase 생성. 실패 시 원본만 반환."""
    user_msg = f'원본 질의: "{sq}"\n\n위 질의의 paraphrase를 정확히 {n}개 생성해줘.'
    try:
        content = chat_completion(
            rt,
            messages=[
                {"role": "system", "content": MULTI_QUERY_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            temperature=0,
            seed=1,
            response_format={"type": "json_object"},
            timeout=15,
        )
        obj = json.loads(content)
        variants = [v.strip() for v in (obj.get("variants") or []) if v and v.strip()]
        return variants[:n]
    except Exception as e:
        print(f"[WARN] multi-query 생성 실패 ({e}), 원본만 사용")
        return []


_LLM_CACHE_DIR = HERE / ".llm_cache"


def _llm_cache_key(model: str, messages, temperature: float, seed: int, response_format) -> str:
    payload = json.dumps(
        {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "seed": seed,
            "response_format": response_format,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def chat_completion(
    rt: Runtime,
    *,
    messages,
    temperature: float = 0,
    seed: int = 1,
    response_format=None,
    timeout: int = 30,
    client=None,
    model=None,
) -> str:
    """LLM 호출 래퍼. choices 파싱까지 처리.

    Args:
        client: 미지정 시 rt.client (retrieval-path OpenAI). answer/chitchat은 rt.answer_client.
        model:  미지정 시 rt.cfg.llm_model. answer 경로는 rt.answer_model.

    cfg.llm_cache=True이면 (model+messages+temp+seed+format) 해시로 디스크 캐시.
    """
    if model is None:
        model = rt.cfg.llm_model
    if client is None:
        client = rt.client

    cache_path = None
    if rt.cfg.llm_cache:
        _LLM_CACHE_DIR.mkdir(exist_ok=True)
        key = _llm_cache_key(model, messages, temperature, seed, response_format)
        cache_path = _LLM_CACHE_DIR / f"{key}.json"
        if cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))["content"]

    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "seed": seed,
        "timeout": timeout,
    }
    if response_format is not None:
        kwargs["response_format"] = response_format
    resp = client.chat.completions.create(**kwargs)
    content = resp.choices[0].message.content

    if cache_path is not None:
        cache_path.write_text(
            json.dumps({"content": content}, ensure_ascii=False),
            encoding="utf-8",
        )
    return content


def classify_and_rewrite(rt: Runtime, messages):
    content = chat_completion(
        rt,
        messages=[{"role": "system", "content": CLASSIFY_SYSTEM}] + messages,
        temperature=0,
        seed=1,
        response_format={"type": "json_object"},
        timeout=15,
    )
    return json.loads(content)


def rewrite_only(rt: Runtime, messages) -> str:
    """classifier_mode이 off/gate일 때 사용. 분류 없이 standalone query만 생성."""
    content = chat_completion(
        rt,
        messages=[{"role": "system", "content": REWRITE_SYSTEM}] + messages,
        temperature=0,
        seed=1,
        timeout=15,
    )
    return content.strip()


def generate_answer_with_context(rt: Runtime, messages, refs):
    ctx = "\n\n".join([f"[문서 {i+1}] {r['content']}" for i, r in enumerate(refs)])
    sys_prompt = ANSWER_SYSTEM + "\n\n## Reference\n" + ctx
    return chat_completion(
        rt,
        messages=[{"role": "system", "content": sys_prompt}] + messages,
        temperature=0,
        seed=1,
        timeout=30,
        client=rt.answer_client,
        model=rt.answer_model,
    )


def generate_chitchat(rt: Runtime, messages):
    return chat_completion(
        rt,
        messages=[{"role": "system", "content": CHITCHAT_SYSTEM}] + messages,
        temperature=0.3,
        seed=1,
        timeout=20,
        client=rt.answer_client,
        model=rt.answer_model,
    )


# ============================================================
# End-to-end answering
# ============================================================
def answer_question(rt: Runtime, messages):
    response = {"standalone_query": "", "topk": [], "references": [], "answer": ""}
    mode = rt.cfg.classifier_mode

    cls = None
    if mode in ("gate", "full"):
        try:
            cls = classify_and_rewrite(rt, messages)
        except Exception as e:
            traceback.print_exc()
            response["answer"] = f"[ERROR-classify] {type(e).__name__}: {e}"
            return response

        if not cls.get("is_science"):
            msg_preview = messages[-1]["content"][:50] if messages else ""
            print(f"[INTENT-CHITCHAT] msg='{msg_preview}' → 비과학 판정, retrieve 스킵")
            try:
                response["answer"] = generate_chitchat(rt, messages)
            except Exception as e:
                traceback.print_exc()
                response["answer"] = f"[ERROR-chitchat] {type(e).__name__}: {e}"
            return response

    if mode == "full":
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

    # 3) 검색
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

    # 4) 답변 생성
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
            eval_id = j["eval_id"]
            print(f"\n=== Test {idx} (eval_id={eval_id}) ===")
            print(f"Question: {j['msg']}")

            resp = answer_question(rt, j["msg"])
            print(f"SQ     : {resp['standalone_query']}")
            print(f"TopK   : {resp['topk']}")
            print(f"Answer : {resp['answer'][:120]}")

            row = {
                "eval_id": eval_id,
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
