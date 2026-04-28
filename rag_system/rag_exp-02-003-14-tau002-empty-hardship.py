"""
고도화된 과학 지식 RAG 시스템

사용법:
    python rag.py index --exp EXP-00-002
    python rag.py eval  --exp EXP-01-001 --retrieval-mode sparse --no-reranker
    python rag.py all   --exp EXP-02-001 --embed-model snunlp/KR-SBERT-V40K-klueNLI-augSTS

실험 별 결과는 logs/<exp>/ 아래에 submission.csv / config.json / meta.json 으로 격리 저장된다.
"""

import os
import re
import sys
import json
import time
import hashlib
import argparse
import traceback
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
load_dotenv(HERE / ".env")

from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# FAISS는 Solar 4096d 풀 차원 사용을 위한 인메모리 dense store
# ES 8.8.0의 dense_vector 최대 2048d 제약을 우회. faiss-cpu 설치 필요.
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


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
    classifier_mode: str = "json"         # json | none

    # Models
    embed_provider: str = "local"         # local (SentenceTransformer) | solar (Upstage API)
    embed_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    llm_model: str = "gpt-4o-mini"
    use_faiss_dense: bool = False         # ES dense_vector 대신 FAISS 사용 (Solar는 항상 True, Qwen3-8B 4096d 등 고차원 필요 시 수동 on)

    # Retrieval params
    topk_sparse: int = 50
    topk_dense: int = 50
    topk_rerank: int = 20
    topk_final: int = 3
    rrf_k: int = 60
    score_threshold: float = 0.0          # rerank top-1 score < 이 값이면 topk=[]
    use_hyde: bool = False                # HyDE: 가상 답변 문서를 dense 쿼리로 사용
    rrf_weight_sparse: float = 1.0        # sparse RRF 가중치
    rrf_weight_dense: float = 1.0         # dense RRF 가중치
    extra_embed_model: str = ""           # 2차 dense retriever (ensemble, FAISS 전용). 비우면 off
    rrf_weight_extra: float = 1.0         # extra retriever RRF 가중치
    rescue_reranker: str = ""             # empty 케이스 2차 검증용 FT reranker 경로
    rescue_threshold: float = 0.1         # rescue reranker score > 이 값이면 topk 복구
    base_submission: str = ""             # 기존 submission.csv 경로. non-empty topk는 보존, empty만 재처리
    rescue_filter_classifier: bool = False  # empty 재처리 전 classifier 실행. is_science=false면 empty 유지
    rescue_filter_sq_pattern: bool = False  # base empty의 SQ가 BIOGRAPHY_SQ_PATTERNS 매치 시 empty 유지
    force_empty_sq_pattern: bool = False  # base non-empty의 SQ가 FORCE_EMPTY_SQ_PATTERNS 매치 시 강제 empty

    # Multi-query inference: SQ 1개 → N개 paraphrase 생성 → 각각 retrieve → RRF fuse
    # SQ 한 번의 운에 의존하지 않고 reranker score 안정화
    multi_query: bool = False
    multi_query_n: int = 3                # 생성할 paraphrase 개수 (원본 포함 시 총 N+1 query)
    multi_query_include_original: bool = True

    # Answer/chitchat 생성용 LLM (retrieval-path LLM과 분리)
    # answer/chitchat은 MAP 평가 무관 → 비용 절감용 Solar 라우팅 가능
    # retrieval-path (SQ/HyDE/variants)는 항상 llm_model(OpenAI) 사용
    answer_llm_provider: str = "openai"   # openai | solar
    answer_llm_model: str = ""            # 빈 값이면 llm_model 공유

    # LLM 결정성 확보용 디스크 캐시 (classify/rewrite/HyDE/answer/chitchat)
    # 같은 입력이면 API 재호출 없이 캐시된 응답 반환 → 재현 가능한 실험
    llm_cache: bool = False
    llm_cache_dir: str = ".llm_cache"

    # Paths
    docs_path: str = "../data/documents.jsonl"
    eval_path: str = "../data/eval.jsonl"
    index_name: str = ""                   # auto-derived from embed_model if empty

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
    p = argparse.ArgumentParser(description="RAG pipeline for Scientific Knowledge QA")
    p.add_argument("mode", choices=["index", "eval", "all"])
    p.add_argument("--exp", dest="exp_name", default="default",
                   help="Experiment name (used for output dir)")

    # Retrieval
    p.add_argument("--retrieval-mode", choices=["sparse", "dense", "hybrid"], default="hybrid")
    p.add_argument("--no-reranker", action="store_true", help="Disable cross-encoder reranker")
    p.add_argument("--classifier", choices=["json", "none"], default="json",
                   help="json: LLM intent classifier / none: skip classifier, always retrieve")

    # Models
    p.add_argument("--embed-provider", choices=["local", "solar"], default=None,
                   help="local (SentenceTransformer) | solar (Upstage API)")
    p.add_argument("--embed-model", default=None)
    p.add_argument("--reranker-model", default=None)
    p.add_argument("--llm-model", default=None)
    p.add_argument("--use-faiss", action="store_true",
                   help="ES dense_vector 대신 FAISS 사용 (local 4096d 등 ES 한도 초과 모델용)")

    # Params
    p.add_argument("--topk-sparse", type=int, default=50)
    p.add_argument("--topk-dense", type=int, default=50)
    p.add_argument("--topk-rerank", type=int, default=20)
    p.add_argument("--topk-final", type=int, default=3)
    p.add_argument("--rrf-k", type=int, default=60)
    p.add_argument("--score-threshold", type=float, default=0.0,
                   help="rerank top-1 score < 이 값이면 해당 질의 topk=[]")
    p.add_argument("--hyde", action="store_true",
                   help="HyDE: LLM이 생성한 가상 답변 문서를 dense 쿼리로 사용")
    p.add_argument("--rrf-weight-sparse", type=float, default=1.0,
                   help="sparse RRF 가중치")
    p.add_argument("--rrf-weight-dense", type=float, default=1.0,
                   help="dense RRF 가중치")
    p.add_argument("--extra-embed-model", default=None,
                   help="2차 dense retriever (ensemble, FAISS 전용, 사전 인덱싱 필요)")
    p.add_argument("--rrf-weight-extra", type=float, default=1.0,
                   help="extra retriever RRF 가중치")
    p.add_argument("--rescue-reranker", default="",
                   help="empty 케이스 2차 검증용 FT reranker 경로")
    p.add_argument("--rescue-threshold", type=float, default=0.1,
                   help="rescue reranker score > 이 값이면 topk 복구 (default: 0.1)")
    p.add_argument("--base-submission", default="",
                   help="기존 submission.csv 경로. non-empty topk는 그대로 보존, empty만 재처리")
    p.add_argument("--rescue-filter-classifier", action="store_true",
                   help="base empty 재처리 전 classifier 실행. is_science=false면 empty 유지 (chitchat 차단)")
    p.add_argument("--rescue-filter-sq-pattern", action="store_true",
                   help="base empty의 SQ가 BIOGRAPHY_SQ_PATTERNS 매치 시 empty 유지 (일대기/전기/위인 차단)")
    p.add_argument("--force-empty-sq-pattern", action="store_true",
                   help="base non-empty의 SQ가 FORCE_EMPTY_SQ_PATTERNS 매치 시 topk 강제 비움 (저신뢰 감정 쿼리 제거)")
    p.add_argument("--llm-cache", action="store_true",
                   help="LLM 호출 결과를 디스크 캐시 (재현 가능한 실험용)")
    p.add_argument("--llm-cache-dir", default=None,
                   help="LLM 캐시 디렉터리 (기본: rag_system/.llm_cache)")
    p.add_argument("--multi-query", action="store_true",
                   help="Multi-query: SQ에서 N개 paraphrase 생성 → 각각 retrieve → RRF fuse")
    p.add_argument("--multi-query-n", type=int, default=3,
                   help="multi-query paraphrase 개수 (default: 3)")
    p.add_argument("--no-multi-query-original", action="store_true",
                   help="multi-query에서 원본 SQ 제외 (variants만 사용)")
    p.add_argument("--answer-llm-provider", choices=["openai", "solar"], default="openai",
                   help="answer/chitchat 생성용 provider. retrieval-path는 항상 openai")
    p.add_argument("--answer-llm-model", default=None,
                   help="answer/chitchat 모델명 (기본: --llm-model 공유. solar면 'solar-pro' 권장)")

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

    # Models: CLI > env > default
    cfg.embed_provider = args.embed_provider or os.environ.get("EMBED_PROVIDER", cfg.embed_provider)
    # Solar provider면 CLI로 명시하지 않는 한 Solar 모델명으로 강제 (env의 EMBED_MODEL 무시)
    if cfg.embed_provider == "solar" and not args.embed_model:
        cfg.embed_model = "solar-embedding-1-large"
    else:
        cfg.embed_model = args.embed_model or os.environ.get("EMBED_MODEL", cfg.embed_model)
    cfg.reranker_model = args.reranker_model or os.environ.get("RERANKER_MODEL", cfg.reranker_model)
    cfg.llm_model = args.llm_model or os.environ.get("LLM_MODEL", cfg.llm_model)
    # Solar는 항상 FAISS, local에서도 --use-faiss로 강제 가능 (Qwen3-8B 4096d 등)
    cfg.use_faiss_dense = args.use_faiss or (cfg.embed_provider == "solar")

    cfg.topk_sparse = args.topk_sparse
    cfg.topk_dense = args.topk_dense
    cfg.topk_rerank = args.topk_rerank
    cfg.topk_final = args.topk_final
    cfg.rrf_k = args.rrf_k
    cfg.score_threshold = args.score_threshold
    cfg.use_hyde = args.hyde
    cfg.rrf_weight_sparse = args.rrf_weight_sparse
    cfg.rrf_weight_dense = args.rrf_weight_dense
    cfg.extra_embed_model = args.extra_embed_model or cfg.extra_embed_model
    cfg.rrf_weight_extra = args.rrf_weight_extra
    cfg.rescue_reranker = args.rescue_reranker
    cfg.rescue_threshold = args.rescue_threshold
    cfg.base_submission = args.base_submission
    cfg.rescue_filter_classifier = args.rescue_filter_classifier
    cfg.rescue_filter_sq_pattern = args.rescue_filter_sq_pattern
    cfg.force_empty_sq_pattern = args.force_empty_sq_pattern
    cfg.llm_cache = args.llm_cache
    if args.llm_cache_dir:
        cfg.llm_cache_dir = args.llm_cache_dir
    cfg.multi_query = args.multi_query
    cfg.multi_query_n = args.multi_query_n
    cfg.multi_query_include_original = not args.no_multi_query_original
    cfg.answer_llm_provider = args.answer_llm_provider
    if args.answer_llm_model is not None:
        cfg.answer_llm_model = args.answer_llm_model

    cfg.docs_path = args.docs_path or os.environ.get("DOCS_PATH", cfg.docs_path)
    cfg.eval_path = args.eval_path or os.environ.get("EVAL_PATH", cfg.eval_path)

    # Auto-derive index name per embed model
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

        # Embedding
        self.embed_model_name = cfg.embed_model
        self.embed_loaded = False
        self.use_faiss_dense = cfg.use_faiss_dense
        if cfg.embed_provider == "solar":
            # Upstage Solar embedding API + FAISS 인메모리 dense store
            # ES 8.8.0의 dense_vector 한계(2048d)를 우회하기 위해 FAISS로 4096d 풀 사용
            api_key = os.environ.get("UPSTAGE_API_KEY")
            if not api_key:
                raise RuntimeError("UPSTAGE_API_KEY not set in .env (required for embed_provider=solar)")
            self.solar_client = OpenAI(
                api_key=api_key,
                base_url="https://api.upstage.ai/v1",
            )
            self.embed_model = None
            self.embed_loaded = True
            self.embed_dim = 4096  # Solar 풀 차원 사용 (FAISS가 처리)
            print(f"[INFO] Solar embedding API: {cfg.embed_model} (full dim={self.embed_dim}, FAISS dense store)")
        else:
            try:
                # Qwen3-Embedding-8B 등 대형 모델은 fp16 강제 (3090 24GB fp32 로드 불가)
                model_kwargs = {}
                if "Qwen3-Embedding-8B" in cfg.embed_model:
                    model_kwargs["torch_dtype"] = "float16"
                    print(f"[INFO] Loading {cfg.embed_model} with fp16 (memory constraint)")
                self.embed_model = SentenceTransformer(cfg.embed_model, model_kwargs=model_kwargs)
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

        # FAISS 인덱스 (Solar 또는 --use-faiss 플래그 시)
        if self.use_faiss_dense:
            if not FAISS_AVAILABLE:
                raise RuntimeError("faiss not installed. Run: pip install faiss-cpu")
            self.faiss_dir = HERE / "faiss_indices"
            self.faiss_dir.mkdir(exist_ok=True)
            self.faiss_path = self.faiss_dir / f"{cfg.index_name}.faiss"
            self.faiss_docids_path = self.faiss_dir / f"{cfg.index_name}.docids.json"
            self.faiss_index = None
            self.faiss_docids = []
            self.faiss_doc_sources = {}  # docid -> {docid, src, content} for ES-like return

            # 기존 FAISS 인덱스가 있으면 로드 (eval 모드용)
            if self.faiss_path.exists():
                print(f"[INFO] Loading FAISS index from {self.faiss_path}")
                self.faiss_index = faiss.read_index(str(self.faiss_path))
                with open(self.faiss_docids_path) as f:
                    self.faiss_docids = json.load(f)
                # 문서 source는 documents.jsonl에서 재구성 (메모리 ~2MB)
                with open(cfg.docs_path) as f:
                    for line in f:
                        d = json.loads(line)
                        self.faiss_doc_sources[d["docid"]] = {
                            "docid": d["docid"],
                            "src": d.get("src", ""),
                            "content": d["content"],
                        }
                print(f"[INFO] FAISS index loaded: {self.faiss_index.ntotal} vectors, dim={self.faiss_index.d}")

        # Extra (2nd) dense retriever for ensemble - always FAISS
        # Primary(ES 또는 FAISS) + extra(FAISS) 3-way RRF 지원
        self.extra_embed_model = None
        self.extra_faiss_index = None
        self.extra_faiss_docids = None
        if cfg.extra_embed_model:
            if not FAISS_AVAILABLE:
                raise RuntimeError("faiss not installed. Run: pip install faiss-cpu")
            # faiss_dir 보장 (primary가 ES일 때도 필요)
            if not hasattr(self, "faiss_dir"):
                self.faiss_dir = HERE / "faiss_indices"

            # Load extra embedding model
            extra_kwargs = {}
            if "Qwen3-Embedding-8B" in cfg.extra_embed_model:
                extra_kwargs["torch_dtype"] = "float16"
                print(f"[INFO] Loading extra {cfg.extra_embed_model} with fp16")
            self.extra_embed_model = SentenceTransformer(cfg.extra_embed_model, model_kwargs=extra_kwargs)
            self.extra_embed_dim = self.extra_embed_model.get_sentence_embedding_dimension()
            print(f"[INFO] Loaded extra embedding model: {cfg.extra_embed_model} (dim={self.extra_embed_dim})")

            # Load extra FAISS index (사전 인덱싱 필요)
            extra_base = os.environ.get("ES_INDEX", "science_kb")
            extra_index_name = f"{extra_base}_{slugify_model(cfg.extra_embed_model)}"
            extra_faiss_path = self.faiss_dir / f"{extra_index_name}.faiss"
            extra_docids_path = self.faiss_dir / f"{extra_index_name}.docids.json"
            if not extra_faiss_path.exists():
                raise RuntimeError(
                    f"Extra FAISS index not found: {extra_faiss_path}\n"
                    f"먼저 인덱싱: python rag.py index --embed-model {cfg.extra_embed_model} --use-faiss"
                )
            self.extra_faiss_index = faiss.read_index(str(extra_faiss_path))
            with open(extra_docids_path) as f:
                self.extra_faiss_docids = json.load(f)
            print(f"[INFO] Extra FAISS loaded: {self.extra_faiss_index.ntotal} vectors")

            # doc sources (primary가 ES면 아직 안 만들어진 상태 → 여기서 생성)
            if not hasattr(self, "faiss_doc_sources") or not self.faiss_doc_sources:
                self.faiss_doc_sources = {}
                with open(cfg.docs_path) as f:
                    for line in f:
                        d = json.loads(line)
                        self.faiss_doc_sources[d["docid"]] = {
                            "docid": d["docid"],
                            "src": d.get("src", ""),
                            "content": d["content"],
                        }

        # Reranker (optional) - Cross-encoder 또는 Qwen3-Reranker (causal LM) 분기
        self.reranker = None
        self.reranker_loaded = False
        self.reranker_kind = None  # "cross-encoder" | "qwen3"
        self.qwen3_rr_tokenizer = None
        self.qwen3_rr_model = None
        self.qwen3_rr_yes_id = None
        self.qwen3_rr_no_id = None
        if cfg.use_reranker:
            if cfg.reranker_model.startswith("Qwen/Qwen3-Reranker"):
                # Qwen3-Reranker는 causal LM의 yes/no 토큰 확률로 점수 산출
                # 공식 모델카드 예제에 맞춰 prefix/suffix 토큰 사전 계산 + content 별도 truncation
                try:
                    from transformers import AutoTokenizer, AutoModelForCausalLM
                    import torch
                    print(f"[INFO] Loading Qwen3-Reranker {cfg.reranker_model} with fp16")
                    self.qwen3_rr_tokenizer = AutoTokenizer.from_pretrained(
                        cfg.reranker_model, padding_side="left"
                    )
                    self.qwen3_rr_model = AutoModelForCausalLM.from_pretrained(
                        cfg.reranker_model,
                        torch_dtype=torch.float16,
                        device_map="auto",
                    ).eval()
                    # yes/no 토큰 id - convert_tokens_to_ids 공식 방식
                    self.qwen3_rr_yes_id = self.qwen3_rr_tokenizer.convert_tokens_to_ids("yes")
                    self.qwen3_rr_no_id = self.qwen3_rr_tokenizer.convert_tokens_to_ids("no")
                    # Prefix/suffix 토큰 사전 계산 (공식 예제 방식)
                    prefix = (
                        "<|im_start|>system\n"
                        "Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
                        "Note that the answer can only be \"yes\" or \"no\"."
                        "<|im_end|>\n<|im_start|>user\n"
                    )
                    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
                    self.qwen3_rr_prefix_ids = self.qwen3_rr_tokenizer.encode(prefix, add_special_tokens=False)
                    self.qwen3_rr_suffix_ids = self.qwen3_rr_tokenizer.encode(suffix, add_special_tokens=False)
                    self.qwen3_rr_max_length = 8192
                    self.reranker_kind = "qwen3"
                    self.reranker_loaded = True
                    print(f"[INFO] Loaded Qwen3-Reranker (yes_id={self.qwen3_rr_yes_id}, no_id={self.qwen3_rr_no_id}, prefix_len={len(self.qwen3_rr_prefix_ids)}, suffix_len={len(self.qwen3_rr_suffix_ids)})")
                except Exception as e:
                    print(f"[WARN] Qwen3-Reranker 로드 실패 ({e}); rerank 스킵됨.")
            else:
                try:
                    from sentence_transformers import CrossEncoder
                    self.reranker = CrossEncoder(cfg.reranker_model)
                    self.reranker_kind = "cross-encoder"
                    self.reranker_loaded = True
                    print(f"[INFO] Loaded reranker: {cfg.reranker_model}")
                except Exception as e:
                    print(f"[WARN] Reranker unavailable ({e}); rerank stage will be skipped.")
                    self.reranker = None
        else:
            print("[INFO] Reranker disabled by --no-reranker.")

        # Rescue reranker (empty 케이스 2차 검증용) - Cross-encoder 또는 Qwen3-Reranker
        self.rescue_reranker = None
        self.rescue_reranker_kind = None  # "cross-encoder" | "qwen3"
        self.qwen3_rescue_tokenizer = None
        self.qwen3_rescue_model = None
        self.qwen3_rescue_yes_id = None
        self.qwen3_rescue_no_id = None
        if cfg.rescue_reranker:
            if cfg.rescue_reranker.startswith("Qwen/Qwen3-Reranker"):
                # Qwen3-Reranker를 rescue로 사용
                try:
                    from transformers import AutoTokenizer, AutoModelForCausalLM
                    import torch
                    print(f"[INFO] Loading Qwen3-Reranker rescue {cfg.rescue_reranker} with fp16")
                    self.qwen3_rescue_tokenizer = AutoTokenizer.from_pretrained(
                        cfg.rescue_reranker, padding_side="left"
                    )
                    self.qwen3_rescue_model = AutoModelForCausalLM.from_pretrained(
                        cfg.rescue_reranker,
                        torch_dtype=torch.float16,
                        device_map="auto",
                    ).eval()
                    self.qwen3_rescue_yes_id = self.qwen3_rescue_tokenizer.convert_tokens_to_ids("yes")
                    self.qwen3_rescue_no_id = self.qwen3_rescue_tokenizer.convert_tokens_to_ids("no")
                    prefix = (
                        "<|im_start|>system\n"
                        "Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
                        "Note that the answer can only be \"yes\" or \"no\"."
                        "<|im_end|>\n<|im_start|>user\n"
                    )
                    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
                    self.qwen3_rescue_prefix_ids = self.qwen3_rescue_tokenizer.encode(prefix, add_special_tokens=False)
                    self.qwen3_rescue_suffix_ids = self.qwen3_rescue_tokenizer.encode(suffix, add_special_tokens=False)
                    self.qwen3_rescue_max_length = 8192
                    self.rescue_reranker = "qwen3-loaded"  # truthy sentinel (실제 model은 별도 속성)
                    self.rescue_reranker_kind = "qwen3"
                    print(f"[INFO] Loaded Qwen3-Reranker rescue (yes_id={self.qwen3_rescue_yes_id}, no_id={self.qwen3_rescue_no_id})")
                except Exception as e:
                    print(f"[WARN] Qwen3-Reranker rescue 로드 실패 ({e})")
            else:
                try:
                    from sentence_transformers import CrossEncoder
                    self.rescue_reranker = CrossEncoder(cfg.rescue_reranker)
                    self.rescue_reranker_kind = "cross-encoder"
                    print(f"[INFO] Loaded rescue reranker: {cfg.rescue_reranker}")
                except Exception as e:
                    print(f"[WARN] Rescue reranker unavailable ({e})")

        # Base submission: non-empty topk는 그대로 보존, empty만 재처리
        self.base_submission = {}
        if cfg.base_submission:
            with open(cfg.base_submission) as _bf:
                for _line in _bf:
                    _r = json.loads(_line)
                    self.base_submission[_r["eval_id"]] = _r
            _n_nonempty = sum(1 for r in self.base_submission.values() if r.get("topk"))
            _n_empty = len(self.base_submission) - _n_nonempty
            print(f"[INFO] Loaded base submission: {len(self.base_submission)} rows ({_n_nonempty} non-empty, {_n_empty} empty)")

        # Elasticsearch
        self.es = Elasticsearch(
            [cfg.es_host],
            basic_auth=(cfg.es_username, cfg.es_password),
            ca_certs=cfg.es_ca_certs,
        )

        # OpenAI (retrieval-path: SQ rewrite, HyDE, multi-query variants)
        self.client = OpenAI()

        # Answer/chitchat 생성용 client+model - provider별 분기
        # MAP 무관 호출만 별도 라우팅 (비용 최적화)
        self.answer_client = self.client
        self.answer_model = cfg.answer_llm_model or cfg.llm_model
        if cfg.answer_llm_provider == "solar":
            api_key = os.environ.get("UPSTAGE_API_KEY")
            if not api_key:
                raise RuntimeError("UPSTAGE_API_KEY not set (required for answer_llm_provider=solar)")
            self.answer_client = OpenAI(api_key=api_key, base_url="https://api.upstage.ai/v1")
            self.answer_model = cfg.answer_llm_model or "solar-pro"
            print(f"[INFO] Answer LLM routed to Solar: model={self.answer_model}")

    def embed(self, sentences, model_type: str = "passage", use_extra: bool = False):
        """
        sentences: List[str]
        model_type: "query" | "passage"  (Solar/Qwen3 instruction 분기용)
        use_extra: True면 extra_embed_model 사용 (ensemble 2nd retriever, FAISS 전용)
        returns: np.ndarray (N, dim), L2-normalized
        """
        if use_extra:
            model = self.extra_embed_model
            model_name = self.cfg.extra_embed_model
        else:
            model = self.embed_model
            model_name = self.cfg.embed_model

        # Qwen3-Embedding instruction prefix
        if model_name.startswith("Qwen/Qwen3") and model_type == "query":
            instruction = "Given a Korean science question, retrieve relevant passages that answer it"
            sentences = [f"Instruct: {instruction}\nQuery: {s}" for s in sentences]

        # Solar API (primary 전용)
        if not use_extra and self.cfg.embed_provider == "solar":
            # Solar는 query/passage 모델이 분리됨
            # FAISS 사용으로 truncation 없이 4096d 풀 차원 그대로 사용
            solar_model = f"solar-embedding-1-large-{model_type}"
            resp = self.solar_client.embeddings.create(
                model=solar_model,
                input=sentences,
            )
            embs = np.array([d.embedding for d in resp.data], dtype=np.float32)
            # L2 정규화 (FAISS IndexFlatIP에서 cosine 유사도 구현용)
            norms = np.linalg.norm(embs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return embs / norms
        return model.encode(sentences, normalize_embeddings=True)


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
    """include_dense=False일 때 ES dense_vector 필드 생략 (FAISS로 분리 시)"""
    props = {
        "docid": {"type": "keyword"},
        "src": {"type": "keyword"},
        "content": {"type": "text", "analyzer": "nori"},
    }
    if include_dense:
        props["embeddings"] = {
            "type": "dense_vector",
            "dims": embed_dim,
            "index": True,
            "similarity": "cosine",
        }
    return {"properties": props}


# ============================================================
# Indexing
# ============================================================
def index_documents(rt: Runtime):
    cfg = rt.cfg
    use_faiss_dense = rt.use_faiss_dense  # Solar 또는 --use-faiss 시 ES 대신 FAISS

    print(f"[INFO] Creating index '{cfg.index_name}'...")
    if rt.es.indices.exists(index=cfg.index_name):
        rt.es.indices.delete(index=cfg.index_name)
    rt.es.indices.create(
        index=cfg.index_name,
        settings=build_settings(),
        mappings=build_mappings(rt.embed_dim, include_dense=not use_faiss_dense),
    )

    print(f"[INFO] Loading documents from {cfg.docs_path}")
    with open(cfg.docs_path) as f:
        docs = [json.loads(line) for line in f]
    print(f"[INFO] {len(docs)} documents loaded")

    print("[INFO] Generating embeddings...")
    # Solar API: request당 토큰 제한 / Qwen3-8B: 3090 24GB fp16 제약으로 8 / 기타: 64
    if cfg.embed_provider == "solar":
        batch_size = 16
    elif "Qwen3-Embedding-8B" in cfg.embed_model:
        batch_size = 8
    else:
        batch_size = 64
    print(f"[INFO] Embedding batch size: {batch_size}")
    all_embs = [] if use_faiss_dense else None
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        contents = [d["content"] for d in batch]
        embs = rt.embed(contents, model_type="passage")
        if use_faiss_dense:
            all_embs.append(embs)
        else:
            for d, e in zip(batch, embs.tolist()):
                d["embeddings"] = e
        print(f"  embedded {i + len(batch)}/{len(docs)}")

    # Solar: FAISS 인덱스 빌드 + 디스크 저장
    if use_faiss_dense:
        all_embs = np.vstack(all_embs).astype(np.float32)
        print(f"[INFO] Building FAISS index: shape={all_embs.shape}")
        rt.faiss_index = faiss.IndexFlatIP(rt.embed_dim)  # cosine via IP (정규화된 벡터)
        rt.faiss_index.add(all_embs)
        rt.faiss_docids = [d["docid"] for d in docs]
        rt.faiss_doc_sources = {
            d["docid"]: {"docid": d["docid"], "src": d.get("src", ""), "content": d["content"]}
            for d in docs
        }
        faiss.write_index(rt.faiss_index, str(rt.faiss_path))
        with open(rt.faiss_docids_path, "w") as f:
            json.dump(rt.faiss_docids, f, ensure_ascii=False)
        print(f"[INFO] FAISS index saved: {rt.faiss_path} ({rt.faiss_index.ntotal} vectors)")

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
    # FAISS 경로 (Solar 또는 --use-faiss 시)
    if rt.use_faiss_dense:
        if rt.faiss_index is None:
            raise RuntimeError("FAISS index not loaded. Run `index` mode first to build.")
        q_emb = rt.embed([query], model_type="query").astype(np.float32)
        scores, indices = rt.faiss_index.search(q_emb, size)
        # ES hits 포맷으로 변환 (다운스트림 호환)
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < 0:
                continue  # FAISS는 빈 슬롯에 -1 반환
            docid = rt.faiss_docids[idx]
            src = rt.faiss_doc_sources[docid]
            results.append({"_source": src, "_score": float(score)})
        return results

    # bge-m3 등 로컬 모델: ES KNN
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


def extra_dense_search(rt: Runtime, query: str, size: int):
    """2차 dense retriever (ensemble용, FAISS 전용)"""
    if rt.extra_faiss_index is None:
        raise RuntimeError("extra_faiss_index not loaded - check --extra-embed-model + 사전 인덱싱")
    q_emb = rt.embed([query], model_type="query", use_extra=True).astype(np.float32)
    scores, indices = rt.extra_faiss_index.search(q_emb, size)
    results = []
    for idx, score in zip(indices[0], scores[0]):
        if idx < 0:
            continue
        docid = rt.extra_faiss_docids[idx]
        src = rt.faiss_doc_sources[docid]
        results.append({"_source": src, "_score": float(score)})
    return results


def rrf_fuse(result_lists: List[list], k: int = 60, weights: Optional[list] = None):
    # weights=None → 모든 리스트 동일 가중치
    # weights=[w_s, w_d] → 각 리스트별 가중치 적용
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


QWEN3_RR_INSTRUCTION = "Given a Korean science question, determine if the document contains the answer to the question."


def qwen3_rerank_batch(rt: Runtime, query: str, docs: List[str], batch_size: int = 4) -> List[float]:
    """Qwen3-Reranker score: P(yes) / (P(yes) + P(no))
    공식 모델카드 방식: prefix + content + suffix 를 개별 토큰화 후 결합.
    content는 max_length에서 prefix/suffix 제외한 크기로 longest_first truncation.
    """
    import torch
    contents = [
        f"<Instruct>: {QWEN3_RR_INSTRUCTION}\n<Query>: {query}\n<Document>: {doc}"
        for doc in docs
    ]
    scores: List[float] = []
    max_content_len = rt.qwen3_rr_max_length - len(rt.qwen3_rr_prefix_ids) - len(rt.qwen3_rr_suffix_ids)
    with torch.no_grad():
        for i in range(0, len(contents), batch_size):
            batch = contents[i:i + batch_size]
            # Content 만 먼저 truncation
            enc = rt.qwen3_rr_tokenizer(
                batch,
                padding=False,
                truncation="longest_first",
                return_attention_mask=False,
                max_length=max_content_len,
            )
            # prefix + content + suffix 결합
            for j in range(len(enc["input_ids"])):
                enc["input_ids"][j] = (
                    rt.qwen3_rr_prefix_ids + enc["input_ids"][j] + rt.qwen3_rr_suffix_ids
                )
            inputs = rt.qwen3_rr_tokenizer.pad(
                enc, padding=True, return_tensors="pt", max_length=rt.qwen3_rr_max_length,
            ).to(rt.qwen3_rr_model.device)
            logits = rt.qwen3_rr_model(**inputs).logits[:, -1, :]  # (B, vocab)
            yes_l = logits[:, rt.qwen3_rr_yes_id]
            no_l = logits[:, rt.qwen3_rr_no_id]
            # 공식: log_softmax 후 exp - numerical stability
            stacked = torch.stack([no_l, yes_l], dim=1)
            log_probs = torch.nn.functional.log_softmax(stacked, dim=1)
            scores.extend(log_probs[:, 1].exp().cpu().float().tolist())
    return scores


def qwen3_rescue_rerank_batch(rt: Runtime, query: str, docs: List[str], batch_size: int = 4) -> List[float]:
    """Rescue 경로용 Qwen3-Reranker. qwen3_rerank_batch와 동일한 로직, rescue 전용 속성 참조"""
    import torch
    contents = [
        f"<Instruct>: {QWEN3_RR_INSTRUCTION}\n<Query>: {query}\n<Document>: {doc}"
        for doc in docs
    ]
    scores: List[float] = []
    max_content_len = rt.qwen3_rescue_max_length - len(rt.qwen3_rescue_prefix_ids) - len(rt.qwen3_rescue_suffix_ids)
    with torch.no_grad():
        for i in range(0, len(contents), batch_size):
            batch = contents[i:i + batch_size]
            enc = rt.qwen3_rescue_tokenizer(
                batch,
                padding=False,
                truncation="longest_first",
                return_attention_mask=False,
                max_length=max_content_len,
            )
            for j in range(len(enc["input_ids"])):
                enc["input_ids"][j] = (
                    rt.qwen3_rescue_prefix_ids + enc["input_ids"][j] + rt.qwen3_rescue_suffix_ids
                )
            inputs = rt.qwen3_rescue_tokenizer.pad(
                enc, padding=True, return_tensors="pt", max_length=rt.qwen3_rescue_max_length,
            ).to(rt.qwen3_rescue_model.device)
            logits = rt.qwen3_rescue_model(**inputs).logits[:, -1, :]
            yes_l = logits[:, rt.qwen3_rescue_yes_id]
            no_l = logits[:, rt.qwen3_rescue_no_id]
            stacked = torch.stack([no_l, yes_l], dim=1)
            log_probs = torch.nn.functional.log_softmax(stacked, dim=1)
            scores.extend(log_probs[:, 1].exp().cpu().float().tolist())
    return scores


def rerank_candidates(rt: Runtime, query: str, candidates, top_n: int):
    if not candidates or rt.reranker_kind is None:
        return [(c[0], c[1], c[2]) for c in candidates[:top_n]]
    if rt.reranker_kind == "qwen3":
        docs = [c[2]["content"] for c in candidates]
        scores = qwen3_rerank_batch(rt, query, docs)
    else:
        pairs = [(query, c[2]["content"]) for c in candidates]
        scores = rt.reranker.predict(pairs)
    rescored = [(c[0], float(s), c[2]) for c, s in zip(candidates, scores)]
    rescored.sort(key=lambda x: x[1], reverse=True)
    return rescored[:top_n]


def retrieve(rt: Runtime, query: str) -> List[Tuple[str, float, dict]]:
    cfg = rt.cfg
    mode = cfg.retrieval_mode

    # --- Multi-query: SQ → N개 paraphrase 생성, 각 query로 retrieve 후 RRF fuse ---
    # SQ 한 번의 운에 의존하지 않고 reranker 입력 후보 안정화
    if cfg.multi_query and mode == "hybrid":
        variants = generate_query_variants(rt, query, cfg.multi_query_n)
        queries = ([query] + variants) if cfg.multi_query_include_original else variants
        if not queries:
            queries = [query]  # 안전망

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
            # 3-way ensemble: extra dense retriever
            if rt.extra_embed_model is not None:
                extra_hits = extra_dense_search(rt, dq, cfg.topk_dense)
                all_lists.append(extra_hits)
                weights.append(cfg.rrf_weight_extra)
        fused = rrf_fuse(all_lists, k=cfg.rrf_k, weights=weights)
        candidates = fused[:cfg.topk_rerank] if cfg.use_reranker else fused[:cfg.topk_final]

    else:
        # --- HyDE: dense 검색용 가상 답변 문서 생성 ---
        # sparse는 원본 질의 유지 (BM25 키워드 매칭), dense만 HyDE 문서로 대체
        dense_query = query
        if cfg.use_hyde:
            dense_query = generate_hyde_doc(rt, query)

        # 1차 후보 수집
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
            lists = [sparse_hits, dense_hits]
            weights = [cfg.rrf_weight_sparse, cfg.rrf_weight_dense]
            # 3-way ensemble: extra dense retriever
            if rt.extra_embed_model is not None:
                extra_hits = extra_dense_search(rt, dense_query, cfg.topk_dense)
                lists.append(extra_hits)
                weights.append(cfg.rrf_weight_extra)
            fused = rrf_fuse(lists, k=cfg.rrf_k, weights=weights)
            candidates = fused[:cfg.topk_rerank] if cfg.use_reranker else fused[:cfg.topk_final]

    # 2차 리랭킹 + score threshold 필터 (top-1 < threshold → topk=[])
    if cfg.use_reranker and rt.reranker_loaded:
        reranked = rerank_candidates(rt, query, candidates, cfg.topk_final)
    else:
        reranked = candidates[:cfg.topk_final]
    # top-1 score가 임계값 미만이면 관련 문서 없음으로 간주 → 빈 topk 반환
    if cfg.score_threshold > 0 and reranked and reranked[0][1] < cfg.score_threshold:
        # --- Rescue reranker: empty 판정된 케이스를 2차 reranker로 검증 ---
        # Cross-encoder (FT) 또는 Qwen3-Reranker 둘 다 지원
        if rt.rescue_reranker is not None and candidates:
            rescue_candidates = candidates[:cfg.topk_rerank]
            if rt.rescue_reranker_kind == "qwen3":
                rescue_scores = qwen3_rescue_rerank_batch(
                    rt, query, [c[2]["content"] for c in rescue_candidates]
                )
            else:
                rescue_pairs = [(query, c[2]["content"]) for c in rescue_candidates]
                rescue_scores = rt.rescue_reranker.predict(rescue_pairs)
            rescue_top = max(rescue_scores)
            if rescue_top > cfg.rescue_threshold:
                rescored = [(c[0], float(s), c[2]) for c, s in zip(rescue_candidates, rescue_scores)]
                rescored.sort(key=lambda x: x[1], reverse=True)
                print(f"  [RESCUE] query='{query[:40]}' rescued with score={rescue_top:.4f} ({rt.rescue_reranker_kind})")
                return rescored[:cfg.topk_final]
        return []
    return reranked


# ============================================================
# LLM prompts
# ============================================================
CLASSIFY_SYSTEM_V1 = """너는 과학 지식 질의응답 시스템의 질의 분석기다.

## 판단 규칙
1) 입력된 대화 메시지(멀티턴 가능)가 과학/자연/공학 지식을 묻는 질문인지 분류하라.
   - 과학 범주: 물리, 화학, 생물, 지구과학, 천문, 의학/생리, 생태, 환경, 공학, 기술, 수학 기반 자연과학, 재료, 에너지
   - 비과학 범주: 일상 잡담, 감정 표현, 인사, 감탄, 칭찬, 사회/정치/경제/교육 정책, 역사, 인물 상식, 취향 질문, 조언 요청
2) 과학 질문이라면 대화 이력 전체 맥락을 반영해 **독립형 질의(standalone query)** 를 한국어로 재작성하라.
   - "그 현상", "이 사건", "그거" 등 지시대명사는 이전 발화를 해석해 구체 명사로 치환할 것
   - 검색 엔진이 이해할 수 있도록 핵심 키워드를 포함한 단일 문장으로 작성할 것
3) 과학 질문이 아니라면 standalone_query는 빈 문자열("")로 둔다.

## 출력 형식
반드시 아래 JSON 스키마만 출력한다 (다른 설명 금지):
{"is_science": true|false, "standalone_query": "..."}
"""

CLASSIFY_SYSTEM_V2 = """너는 과학 지식 질의응답 시스템의 질의 분석기다.

## 판단 규칙
1) 입력된 대화 메시지(멀티턴 가능)가 과학/자연/공학 지식을 묻는 질문인지 분류하라.
   - 과학 범주:
     * 자연과학: 물리, 화학, 생물, 지구과학, 천문, 생태, 환경
     * 의학/보건: 의학, 약학, 생리, 질병, 치료법, 피임·건강 관리 방법
     * 공학/기술: 공학, 기술, 재료, 에너지, 수학 기반 자연과학
     * **과학자·연구자 관련 질문** (예: "Dmitri Ivanovsky가 누구야?", "뉴턴의 업적은?") → 과학으로 분류
     * 과학적 원리·현상을 묻는 모든 질문
   - 비과학 범주:
     * 일상 잡담, 감정 표현, 인사, 감탄, 칭찬, 위로 요청
     * 사회/정치/경제/교육 정책, 일반 역사(과학자 외)
     * 프로그래밍·코딩 질문
     * 취향·의견 질문, 연애·인간관계 조언
2) 과학 질문이라면 대화 이력 전체 맥락을 반영해 **독립형 질의(standalone query)** 를 한국어로 재작성하라.
   - "그 현상", "이 사건", "그거" 등 지시대명사는 이전 발화를 해석해 구체 명사로 치환할 것
   - 검색 엔진이 이해할 수 있도록 핵심 키워드를 포함한 단일 문장으로 작성할 것
3) 과학 질문이 아니라면 standalone_query는 빈 문자열("")로 둔다.

## 출력 형식
반드시 아래 JSON 스키마만 출력한다 (다른 설명 금지):
{"is_science": true|false, "standalone_query": "..."}
"""

CLASSIFY_SYSTEM_V3 = """너는 과학 지식 질의응답 시스템의 질의 분석기다.

## 판단 규칙
1) 입력된 대화 메시지(멀티턴 가능)가 과학/자연/공학 지식을 묻는 질문인지 분류하라.
   - 과학 범주:
     * 자연과학: 물리, 화학, 생물, 지구과학, 천문, 생태, 환경
     * 의학/보건: 의학, 약학, 생리, 질병, 치료법, 피임·건강 관리 방법
     * 공학/기술: 공학, 기술, 재료, 에너지, 수학 기반 자연과학
     * **컴퓨터 과학 이론**: 알고리즘, 자료구조, 암호학·해시 함수, 정보이론, 계산 복잡도, 오토마타 이론, 컴퓨터 보안 이론
     * **과학자·연구자 관련 질문** (예: "Dmitri Ivanovsky가 누구야?", "뉴턴의 업적은?") → 과학으로 분류
     * 과학적 원리·현상을 묻는 모든 질문
   - 비과학 범주:
     * 일상 잡담, 감정 표현, 인사, 감탄, 칭찬, 위로 요청
     * 사회/정치/경제/교육 정책, 일반 역사(과학자 외)
     * **특정 프로그래밍 언어의 문법·사용법 질문** (예: "Python에서 lambda 함수 언제 써?", "class 정의 방법 알려줘")
     * 코드 작성 요청
     * 취향·의견 질문, 연애·인간관계 조언
2) 과학 질문이라면 대화 이력 전체 맥락을 반영해 **독립형 질의(standalone query)** 를 한국어로 재작성하라.
   - "그 현상", "이 사건", "그거" 등 지시대명사는 이전 발화를 해석해 구체 명사로 치환할 것
   - 검색 엔진이 이해할 수 있도록 핵심 키워드를 포함한 단일 문장으로 작성할 것
3) 과학 질문이 아니라면 standalone_query는 빈 문자열("")로 둔다.

## 출력 형식
반드시 아래 JSON 스키마만 출력한다 (다른 설명 금지):
{"is_science": true|false, "standalone_query": "..."}
"""

CLASSIFY_SYSTEM_V4 = """너는 과학 지식 질의응답 시스템의 질의 분석기다.

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

CLASSIFY_SYSTEM = CLASSIFY_SYSTEM_V4  # V4: 프로그래밍/CS/연구방법론 과학 범주 편입

# AI 자기지칭 chitchat SQ 패턴 (classifier=none + τ 하향 시 chitchat이 retrieval 통과하는 문제 보정)
# rewrite_only는 모든 질의에 SQ 생성하므로 후처리 필터 필요
CHITCHAT_SQ_PATTERNS = [
    re.compile(r"^너는\s*누구"),
    re.compile(r"^너의\s*(잘|특기|장점)"),
    re.compile(r"^너가?\s*(모르|잘하|뭘)"),
]

# 인물·전기류 SQ 패턴 - rescue 후보에서 제외
# tau002 empty set의 12 "벽돌공의 일대기"를 차단하면서 37/93(과학 질의) rescue 허용
# "전기"는 제외 (電氣: 전기공학/전자 과학 문서에 다수 등장, 오차단 위험)
BIOGRAPHY_SQ_PATTERNS = [
    re.compile(r"일대기|위인|생애|전기문|전기작가"),
]

# 감정/고통 발화 SQ 패턴 - base non-empty 저신뢰 케이스 강제 empty
# tau002 "요즘 힘든 상황을 극복하는 방법..." (218/276, bge top-1 score 0.084) 타겟
# 과학 문서 corpus에 감정 상태 쿼리는 매칭 품질 낮음 → topk 비우는 게 안전
FORCE_EMPTY_SQ_PATTERNS = [
    re.compile(r"힘든\s*상황|힘들다"),
]

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


# --- HyDE ---
# 기존: "과학 지식 질문" 명시 → 역사/사회/일반상식 질의에서 dense embedding이
# GT(역사·지리·사회 문서)와 의미 거리 벌어지는 문제 (e.g. id=212 "남미 라틴 역사",
# id=81 "통학버스" 등 corpus의 global_facts/human_aging 범주 문서들).
# 변경: 도메인 중립 + 사전·백과사전 문체 명시로 corpus 문서 스타일과 정렬.
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


def generate_query_variants(rt: Runtime, sq: str, n: int) -> List[str]:
    """SQ에서 n개 paraphrase 생성. 실패 시 원본만 반환."""
    user_msg = f'원본 질의: "{sq}"\n\n위 질의의 paraphrase를 정확히 {n}개 생성해줘.'
    try:
        content = cached_chat_completion(
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


def cached_chat_completion(
    rt: Runtime,
    *,
    messages,
    temperature: float = 0,
    seed: int = 1,
    response_format=None,
    timeout: int = 30,
    client=None,
    model=None,
    provider: str = "openai",
) -> str:
    """LLM 호출 래퍼. rt.cfg.llm_cache가 켜지면 입력 해시로 디스크 캐시.

    Args:
        client: 미지정 시 rt.client (retrieval-path OpenAI). answer/chitchat은 rt.answer_client.
        model:  미지정 시 rt.cfg.llm_model. answer 경로는 rt.answer_model.
        provider: 캐시 키 분리용 (openai | solar). 같은 model 이름이라도 provider 다르면 다른 응답.

    캐시 키 = SHA256(provider + model + messages + temperature + seed + response_format).
    반환값은 메시지 content 문자열 (choices 파싱까지 처리).
    """
    if model is None:
        model = rt.cfg.llm_model
    if client is None:
        client = rt.client

    cache_path = None
    if rt.cfg.llm_cache:
        key_payload = {
            "provider": provider,
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "seed": seed,
            "response_format": response_format,
        }
        key = hashlib.sha256(
            json.dumps(key_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        cache_dir = Path(rt.cfg.llm_cache_dir)
        if not cache_dir.is_absolute():
            cache_dir = HERE / cache_dir
        cache_path = cache_dir / f"{key}.json"
        if cache_path.exists():
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)["content"]

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
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump({"content": content}, f, ensure_ascii=False)

    return content


def classify_and_rewrite(rt: Runtime, messages):
    content = cached_chat_completion(
        rt,
        messages=[{"role": "system", "content": CLASSIFY_SYSTEM}] + messages,
        temperature=0,
        seed=1,
        response_format={"type": "json_object"},
        timeout=15,
    )
    return json.loads(content)


def rewrite_only(rt: Runtime, messages) -> str:
    """classifier_mode=none 일 때 사용. 분류 없이 standalone query만 생성."""
    content = cached_chat_completion(
        rt,
        messages=[{"role": "system", "content": REWRITE_SYSTEM}] + messages,
        temperature=0,
        seed=1,
        timeout=15,
    )
    return content.strip()


def generate_hyde_doc(rt: Runtime, query: str) -> str:
    """HyDE: 질의에 대한 가상 답변 문서 생성. dense retrieval 쿼리로 사용됨."""
    try:
        content = cached_chat_completion(
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


def generate_answer_with_context(rt: Runtime, messages, refs):
    ctx = "\n\n".join([f"[문서 {i+1}] {r['content']}" for i, r in enumerate(refs)])
    sys_prompt = ANSWER_SYSTEM + "\n\n## Reference\n" + ctx
    return cached_chat_completion(
        rt,
        messages=[{"role": "system", "content": sys_prompt}] + messages,
        temperature=0,
        seed=1,
        timeout=30,
        client=rt.answer_client,
        model=rt.answer_model,
        provider=rt.cfg.answer_llm_provider,
    )


def generate_chitchat(rt: Runtime, messages):
    return cached_chat_completion(
        rt,
        messages=[{"role": "system", "content": CHITCHAT_SYSTEM}] + messages,
        temperature=0.3,
        seed=1,
        timeout=20,
        client=rt.answer_client,
        model=rt.answer_model,
        provider=rt.cfg.answer_llm_provider,
    )


# ============================================================
# End-to-end answering
# ============================================================
def answer_question(rt: Runtime, messages):
    response = {"standalone_query": "", "topk": [], "references": [], "answer": ""}

    # 1) 의도 분류 분기
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
        # classifier_mode == "none": 모든 질문에 대해 검색
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

    # 2) 검색
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

    # 3) 답변 생성
    try:
        response["answer"] = generate_answer_with_context(rt, messages, refs)
    except Exception as e:
        traceback.print_exc()
        response["answer"] = f"[ERROR-answer] {type(e).__name__}: {e}"

    # 4) Chitchat SQ 패턴 후처리 필터 - AI 자기지칭 질의는 topk 강제 비움
    # classifier=none + τ 하향 조합에서 220/229 같은 chitchat이 retrieval 통과하는 문제 보정
    if any(p.search(sq) for p in CHITCHAT_SQ_PATTERNS):
        response["topk"] = []
        response["references"] = []

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

            # Base submission 분기:
            #  - non-empty topk: 그대로 보존 (retrieve/LLM 호출 생략)
            #  - empty topk + --rescue-filter-classifier: classifier로 science 판정
            #      비과학이면 empty 유지, 과학이면 일반 path로 fall-through (rescue reranker 작동)
            if eval_id in rt.base_submission:
                base = rt.base_submission[eval_id]
                if base.get("topk"):
                    # Force-empty: base non-empty이지만 SQ가 감정 발화 패턴이면 topk 비움
                    if cfg.force_empty_sq_pattern:
                        base_sq = base.get("standalone_query", "") or ""
                        if any(p.search(base_sq) for p in FORCE_EMPTY_SQ_PATTERNS):
                            empty_base = {**base, "topk": [], "references": []}
                            print(f"[BASE-FORCE-EMPTY] SQ='{base_sq[:50]}' → topk 강제 비움")
                            of.write(json.dumps(empty_base, ensure_ascii=False) + "\n")
                            of.flush()
                            continue
                    print(f"[BASE-PRESERVE] topk={base['topk']}")
                    of.write(json.dumps(base, ensure_ascii=False) + "\n")
                    of.flush()
                    continue
                # SQ 패턴 게이트 (LLM 호출 0): base에 저장된 SQ로 즉시 판정
                if cfg.rescue_filter_sq_pattern:
                    base_sq = base.get("standalone_query", "") or ""
                    if any(p.search(base_sq) for p in BIOGRAPHY_SQ_PATTERNS):
                        print(f"[BASE-EMPTY-BIOGRAPHY] SQ='{base_sq[:50]}' → preserve empty")
                        of.write(json.dumps(base, ensure_ascii=False) + "\n")
                        of.flush()
                        continue
                    if any(p.search(base_sq) for p in CHITCHAT_SQ_PATTERNS):
                        print(f"[BASE-EMPTY-CHITCHAT-SQ] → preserve empty")
                        of.write(json.dumps(base, ensure_ascii=False) + "\n")
                        of.flush()
                        continue
                    if not base_sq:
                        print(f"[BASE-EMPTY-NO-SQ] → preserve empty (no SQ to rescue)")
                        of.write(json.dumps(base, ensure_ascii=False) + "\n")
                        of.flush()
                        continue
                    print(f"[BASE-EMPTY-SCIENCE-SQ] SQ='{base_sq[:50]}' → rescue path")
                elif cfg.rescue_filter_classifier:
                    try:
                        cls = classify_and_rewrite(rt, j["msg"])
                    except Exception as e:
                        print(f"[WARN] classifier 실패 ({e}) - empty 유지")
                        cls = {"is_science": False}
                    if not cls.get("is_science"):
                        print(f"[BASE-EMPTY-CHITCHAT] classifier rejects → preserve empty")
                        of.write(json.dumps(base, ensure_ascii=False) + "\n")
                        of.flush()
                        continue
                    print(f"[BASE-EMPTY-SCIENCE] classifier accepts → rescue path")

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
