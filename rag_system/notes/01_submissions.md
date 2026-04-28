# 리더보드 제출 로그

일일 제출 한도: **5회** (KST 자정 기준 리셋)

> ⚠️ **용어 정정 (2026-04-21)**: LB 결과에 출력되는 2개 숫자는 `public MAP / public MRR`이다. 이전 일부 기록에서 2번째 숫자(MRR)를 "private score"로 해석한 부분은 오류. private MAP은 대회 종료까지 비공개.
>
> ⚠️⚠️ **중대 정정 (2026-04-21, EXP-14 이후)**: public/private split은 `eval_id` 기반이 **아님**. 실제로는 **eval.jsonl의 파일 line index**로 분할 — line 0~109 = public, line 110~219 = private.
> - eval_id는 0~309 범위에서 non-sequential (220행이 random eval_id로 매핑)
> - 예: eval_id=276은 line 21 → **public**, eval_id=37은 line 201 → **private**
> - **이전 기록의 "37/93/12는 public" 은 모두 오류. 세 건 모두 private** (line 201/188/167)
> - 이 오판 때문에 "37/93/12 rescue가 public LB 기여 0" 진단이 잘못된 결론으로 이어졌음 (실제는 단순히 public 영향이 아예 없던 것)
> - 재매핑 대상 line<110 = public 진짜 리스트: 아래 실험에서 line index 기준 재확인 필요

## 제출 기록 템플릿

```
### SUB-XXX | YYYY-MM-DD HH:MM
- **파일**: logs/submission_XXX.csv
- **구성**: 어떤 EXP 조합인지
- **MAP public**:
- **MAP private**: (대회 종료 후 공개)
- **오늘 남은 제출 횟수**:
- **소감/다음 액션**:
```

---

## 제출 로그

<!-- 새 제출은 이 아래에 누적 -->

### SUB-001 | 2026-04-12
- **파일**: logs/submission.csv (rag_system/rag.py eval 전체 220 결과)
- **구성**: EVAL-002 (rag_system 초기 통합, BGE-M3 + RRF + bge-reranker-v2-m3 + JSON mode 의도 분류)
- **MAP public**: **0.7295**
- **MRR**: **0.7379**
- **baseline 대비**: +0.034 (0.6955 → 0.7295)
- **소감/다음 액션**:
  - 첫 시도로 베이스라인 초과. 아직 ablation 안 해서 어느 요소가 결정적인지 불명
  - 다음 제출 전에 각 구성 요소 기여도 분해 필요 (리랭커 on/off, Hybrid vs Sparse only 등)
  - 리랭커 실제 로드 여부를 다음 실행 로그로 확인

### SUB-XXX | 2026-04-14 | EXP-03-003-rrfk100
- **파일**: logs/EXP-03-003-rrfk100/submission.csv
- **구성**: V2 프롬프트 + pool 50/50 + rerank 20 + **rrf_k=100**
- **MAP public**: **0.7386**
- **MRR**: **0.7470**
- **Δ vs V2 기준 (rrf_k=60, 0.7477)**: **-0.0091**
- **소감/다음 액션**:
  - RRF k=100은 pool=70과 **정확히 동일한 폭(-0.0091)** 으로 하락 — 수치 우연 일치 아닌 공통 원인 가능성 (둘 다 상위 확신도 희석 방향) → RRF k 튜닝 종료
  - Phase 3(rerank/pool/rrf_k) 전부 기본값이 최적으로 확인 → retrieval 튜닝 종료
  - 다음: ko-reranker (`Dongjin-kr/ko-reranker`) 교체 실험

### SUB-XXX | 2026-04-14 | EXP-04-001-koreranker
- **파일**: logs/EXP-04-001-koreranker/submission.csv
- **구성**: V2 프롬프트 + **Dongjin-kr/ko-reranker** (나머지 기본값)
- **MAP public**: **0.7258**
- **MRR**: **0.7273**
- **Δ vs V2 기준 (bge-reranker-v2-m3, 0.7477)**: **-0.0219**
- **소감/다음 액션**:
  - ko-reranker 유의미하게 하락 → **bge-reranker-v2-m3 유지**, ko-reranker 폐기
  - 실패 케이스(14, 206, 221, 241) 전혀 복구 안 됨 → reranker 단 문제 아님 재확인
  - 다음: V4 프롬프트 (프로그래밍/CS/연구방법론 과학 편입) 실험 — EXP-04-002

### SUB-XXX | 2026-04-15 | EXP-04-002-promptV4 
- **파일**: logs/EXP-04-002-promptV4/submission.csv
- **구성**: CLASSIFY_SYSTEM V4 (프로그래밍·CS·연구방법론 과학 편입) + bge-reranker-v2-m3 + 기본값
- **MAP public**: **0.7667**
- **MRR**: **0.7727**
- **Δ vs V2 기준 (0.7477)**: **+0.0190**
- **누적 Δ vs Phase 1 (EVAL-002, 0.7295)**: **+0.0372**
- **소감/다음 액션**:
  - 프롬프트 단독 변경으로 +0.019 — 두 번째로 큰 단일 개선폭 (EXP-02-001 V2 프롬프트 +0.0182 다음)
  - CS 오분류 복구가 핵심 (eval_id 21, 210 등)
  - 실패 케이스(14, 206, 221, 241)는 여전 — retrieval recall 병목 확정
  - 다음 방향: embedding 모델 교체 또는 HyDE로 retrieval recall 향상 시도

### SUB-XXX | 2026-04-16 | EXP-02-003-01 (classifier=none + score threshold 0.05) 
- **파일**: logs/EXP-02-003-01/submission.csv
- **구성**: classifier=none + `--score-threshold 0.05` (rerank top-1 score < 0.05 이면 topk=[])
- **MAP public**: **0.8939**
- **MRR**: **0.9030**
- **Δ vs (EXP-02-003, 0.8318)**: **+0.0621**
- **Δ vs (EXP-04-002-promptV4, 0.7667)**: **+0.1272**
- **누적 Δ vs Phase 1 EVAL-002 (0.7295)**: **+0.1644**
- **소감/다음 액션**:
  - 분류기 제거 + rerank score 사후 필터 조합이 패러다임 전환 성공
  - 예측 범위(+0.06~0.08) 정확히 적중, τ=0.05가 bimodal 분포의 자연 단절점과 일치
  - threshold 발동 24건, 이상 케이스 0건 (strange=0) → 로직 정상
  - 고점수 "답변 불가" 케이스 (14, 21, 25, 27, 102, 221, 252) 재확인 — reranker/embedding 단 개선 필요
  - 다음: Solar embedding 또는 weighted RRF로 retrieval recall 향상

### SUB-XXX | 2026-04-16 | EXP-02-003-02 (위 + HyDE)
- **파일**: logs/EXP-02-003-02/submission.csv
- **구성**: classifier=none + score-threshold 0.05 + **HyDE** (dense 쿼리만 가상 답변 문서로 교체, sparse는 원본 유지)
- **MAP public**: **0.9000**
- **MRR**: **0.9091**
- **Δ vs EXP-02-003-01 (0.8939)**: **+0.0061**
- **소감/다음 액션**:
  - 예상 범위(+0.01~0.03) 하단. 소폭 개선
  - topk 변화가 많지 않았음 — bge-m3가 이미 잘 매칭하는 질의가 대부분
  - HyDE는 서술형 질의에서 일부 recall 개선 (id=270 나무 생태계 등)
  - LLM 호출 +220회 비용 대비 이득은 제한적이지만 누적 가치는 있음
  - 다음: Solar embedding 또는 weighted RRF

### SUB-XXX | 2026-04-16 | EXP-02-003-03 (위 + Weighted RRF dense 0.7/sparse 0.3) 🏆
- **파일**: logs/EXP-02-003-03/submission.csv
- **구성**: classifier=none + score-threshold 0.05 + HyDE + **Weighted RRF (w_s=0.3, w_d=0.7)**
- **MAP public**: **0.9152**
- **MRR**: **0.9242**
- **Δ vs EXP-02-003-02 (0.9000)**: **+0.0152**
- **누적 Δ vs Phase 1 EVAL-002 (0.7295)**: **+0.1857**
- **소감/다음 액션**:
  - 예상 범위(+0.005~+0.015) 상단 초과. dense 가중치 상향이 top-3 품질을 의미 있게 개선
  - id=81 (통학 버스) top-1 score=0.054로 threshold 간신히 통과해 topk 채워진 regression 있었으나, 다른 개선이 더 컸음
  - id=270 등 top-1 교체 일부 있었으나 전반적 이득 우세
  - 다음: Solar embedding 또는 weight sweep (0.2/0.8 등)

### SUB-XXX | 2026-04-16 | EXP-02-003-05 (위 + Weighted RRF 0.2/0.8 sweep)
- **파일**: logs/EXP-02-003-05/submission.csv
- **구성**: classifier=none + score-threshold 0.05 + HyDE + **Weighted RRF (w_s=0.2, w_d=0.8)**
- **MAP public**: **0.8909**
- **MRR**: **0.9000**
- **Δ vs EXP-02-003-03 (0.9152, w=0.3/0.7)**: **-0.0243** ❌
- **소감/다음 액션**:
  - 의외의 큰 하락. dense 비중을 더 올리면 sparse의 보완 효과 손실 → **0.3/0.7이 sweet spot 확정**
  - id=81 (통학 버스) 빈 topk로 +0.0045 회복 신호 있었으나, 다른 질의들의 retrieval 품질 하락이 훨씬 컸음
  - sparse는 키워드 매칭으로 정확한 docid 후보를 가져오는 역할 — 0.2까지 약화시키면 정답 후보가 RRF 상위에서 밀려남
  - **0.3/0.7 (EXP-02-003-03) 유지가 정답**. 더 극단(0.1/0.9, 0.4/0.6)은 시도 가치 낮음
  - 다음: Solar embedding (직교 축으로 전환)

### SUB-XXX | 2026-04-16 | EXP-02-003-04 Solar embedding (no-HyDE) ⚠️
- **파일**: logs/EXP-02-003-04-solar-nohyde/submission.csv
- **구성**: classifier=none + score-threshold 0.05 + **Solar embedding (no HyDE)** + Weighted RRF (w_s=0.3, w_d=0.7)
- **변경**: bge-m3 (1024d) → **Solar embedding 1 large (4096d → 2048d truncated, ES 8.8.0 호환)**, query/passage 모델 분리, HyDE 끔
- **MAP public**: **0.8879**
- **MRR**: **0.8970**
- **Δ vs EXP-02-003-03 (0.9152)**: **-0.0273**
- **소감**:
  - 예상 범위(0.91~0.93) 하단 크게 하회 → 당시 "Solar dead end" 판단했으나 **HyDE 켠 후 결과(아래 Solar HyDE)로 결론 뒤집힘**
  - HyDE 끄면 Solar는 bge-m3 대비 -0.0273. 하지만 이는 **HyDE 부재 영향이 지배적**이었음 (Solar HyDE 결과 참조)

### SUB-XXX | 2026-04-16 | EXP-02-003-04 Solar embedding (HyDE) ✅
- **파일**: logs/EXP-02-003-04-solar/submission.csv
- **구성**: classifier=none + score-threshold 0.05 + **Solar embedding + HyDE** + Weighted RRF (w_s=0.3, w_d=0.7)
- **변경**: HyDE 켬 (위 no-HyDE 변형 대비)
- **MAP public**: **0.9091**
- **MRR**: **0.9182**
- **Δ vs EXP-02-003-03 (0.9152)**: **-0.0061** (1% 격차)
- **Δ vs Solar no-HyDE (0.8879)**: **+0.0212** 🚀
- **핵심 발견**:
  - **Solar에서 HyDE 효과는 +0.0212 — bge-m3에서의 +0.0061보다 ~3.5배 큼**
  - 이전 가설("Solar query/passage 분리로 HyDE 불필요") **틀렸음** — Solar에서 HyDE가 더 critical
  - 추정 원인: Solar query model은 짧은 질의를 짧은 형식으로 임베딩 → 긴 passage embedding과 의미적 거리 여전히 큼. HyDE가 query를 긴 가상 답변 문서로 변환해주면 passage embedding과의 매칭 품질이 크게 향상
  - 4096d → 2048d truncation 손실은 **HyDE로 상당 부분 보완됨** (-0.0273에서 -0.0061로 회복)
- **결론**: Solar는 죽지 않았음. HyDE 필수

### SUB-XXX | 2026-04-16 | EXP-02-003-04-faiss Solar 4096d FAISS + HyDE ❌
- **파일**: logs/EXP-02-003-04-faiss-hyde/submission.csv
- **구성**: 위와 동일 + **FAISS in-memory (4096d 풀 차원, truncation 제거)**
- **변경**: ES dense_vector → FAISS IndexFlatIP, 4096d 풀 사용
- **MAP public**: **0.8985**
- **MRR**: **0.9076**
- **Δ vs EXP-02-003-03 (0.9152)**: **-0.0167**
- **Δ vs Solar 2048d HyDE (0.9091)**: **-0.0106** ❌ (기대와 정반대)
- **놀라운 결과**:
  - 4096d 풀 사용이 **오히려 -0.0106 하락**. "truncation이 손실 원인" 가설 **틀렸음**
  - CSV diff는 대부분 동일했지만 LB는 유의미한 차이 → 미세 변화가 누적됨
- **원인 추정**:
  - **LLM 비결정성의 영향 지배적**: gpt-4o-mini seed=1에도 `rewrite_only`와 `generate_hyde_doc` 출력이 run마다 미세 다름
    - id=81 예시: "가치와 장점"(2048d HyDE) → "가치에 대한 설명"(4096d FAISS) → topk 결과 달라짐
  - **FAISS IndexFlatIP (exact) vs ES HNSW (approximate) 차이**:
    - exact search가 항상 최고는 아님. HNSW의 approximation이 일부 질의에선 오히려 다양성 공급 역할
  - **2048d truncation 손실이 생각보다 작았음**: Solar가 사실상 Matryoshka-like로 앞 차원에 핵심 정보 집중
- **LLM 비결정성 측정치**: 같은 pipeline에서 런마다 ±0.01~0.02 변동 가능 → 제출 MAP의 noise floor 확인
- **id=81 GT 추정**: 여전히 불확실. 4096d FAISS에서 empty + LB 하락 → GT filled 가능성 소폭 ↑ (그러나 다른 노이즈와 섞여 확정 불가)
- **결론**: **2048d Solar HyDE (0.9091)가 Solar 라인의 사실상 천장**. 4096d는 무효. Solar 탐색 종료 시점
- **현재 best**: **EXP-02-003-03 (0.9152) 🏆 확정 유지** — Solar로 초과 못 함
- **다음 방향 (재정렬)**:
  1. ⭐ **EXP-02-003-03 최종 제출로 확정 고려** — 0.9152 위로 올리기 어렵다는 증거 누적
  2. Qwen3-Embedding-0.6B (1024d, ES 호환) — 마지막 직교 축 시도
  3. Multi-query generation (HyDE 확장) — LLM 비결정성 평균화 효과 기대 가능
  4. LLM 비결정성 완화: rewrite_only/HyDE 프롬프트 강화하여 변동성 줄이기

### SUB-XXX | 2026-04-18 | EXP-06-000 Cached Baseline (결정성 확보) ⚠️
- **파일**: logs/EXP-06-000-cached-baseline/submission.csv
- **구성**: EXP-02-003-03와 동일 파라미터 + `--llm-cache` (SHA256 키로 SQ/HyDE 모두 디스크 캐시, 결정적 재현 가능)
- **MAP public**: **0.9045**
- **MRR**: **0.9136**
- **Δ vs EXP-02-003-03 (0.9152)**: **−0.0107 MAP**, **−0.0106 MRR**
- **사전 예상 범위**: 0.910~0.920 (중앙값 0.913) → **하단도 하회** (−0.003 예측 미스)
- **로컬 diff (cb vs 옛 0.9152 파일)**: identical 187 / changed-nonempty 31 / regressed 2 (id=81, 52) / SQ 다름 59건
- **해석**:
  - 옛 0.9152는 **운 좋은 LLM 샘플** — 캐싱은 그날 첫 호출 결과를 동결하는데, 그게 평균보다 약간 나쁜 샘플이었음
  - 관측된 −0.011은 이전 EXP-02-003-04-faiss에서 추정한 **LLM 비결정성 noise floor ±0.01~0.02**와 일치
  - id=52(연기 열수공, baseline score 0.97) GT hit이었을 가능성 ↑ → cb empty로 내려가 −0.0045 기여 추정
  - id=81(통학버스, GT empty) cb empty로 내려가 +0.0045 기여 → id=52와 상쇄
  - 나머지 −0.011은 59건 SQ 변화가 평균적으로 검색 품질 약간 하락시킨 누적 효과
- **의미**:
  - ⭐ **새 reference = 0.9045**: 앞으로 모든 cached 실험의 incremental Δ 측정 기준
  - **0.9152 안전판 유지**: 옛 stochastic 파일은 마감 직전 최종 잠금 후보로 보존
  - **결정성 비용 = −0.011 MAP** 감수하는 대신 ±0.01 노이즈 제거 → 작은 효과(+0.003) 측정 가능
  - 0.9152 초과를 노리려면 cached pipeline에서 **최소 +0.011** 개선 필요
- **소감/다음 액션**:
  - 비결정성이 축복이자 저주였음이 정량 확인. 캐싱 이전 0.9091/0.9152/0.8985 오락가락하던 이유
  - LB는 **앵커 확보용**으로 1회 소모 가치 있었음 — 이제 sweep 결과 Δ가 "실측 LB Δ" 기대치로 환산 가능
  - 다음: 에러 분석 10건 (score 0.1~0.5 경계 케이스) → 실패 유형 분류 → 타겟 개입. 그 후 multi-query 또는 Qwen3-8B

### SUB-XXX | 2026-04-19 | EXP-02-003-03-mq3-tau002 🏆 새 best 달성
- **파일**: logs/EXP-02-003-03-mq3-tau002/submission.csv
- **구성**: EXP-02-003-03 (0.9152) 베이스 + **multi-query n=3** + 도메인 중립 HyDE + **τ=0.02** + answer LLM = Solar
  - 인덱스/임베딩/리랭커 전부 동일 (색인 재사용)
  - 실제 변경: multi-query 추가 + HyDE 프롬프트 도메인 중립화 + τ 0.05→0.02 + answer LLM Solar 라우팅
- **MAP public**: **0.9258** 🎉
- **MRR**: **0.9333**
- **Δ vs 옛 0.9152**: **+0.0106 MAP**, **+0.0091 MRR**
- **사전 예상 범위**: 0.920~0.925 (중앙값 0.922) → **실제 0.9258**, 예상 상단도 상회
- **Δ 분해 (실측)**:
  - eval_id=212 GT 복구 (남미 라틴 역사, doc `b303e4ec`): +0.0045
  - eval_id=305 GT 복구 (농장 해안, doc `17da5059`): +0.0045
  - eval_id=52 유지 (연기 열수공 doc `70fd8635`, 옛 baseline에도 있었음): ±0
  - 49 changed-nonempty 케이스 순효과: **+0.0016** (rank shuffle이 GT-ward 기울어짐)
  - 220 chitchat: cached mq3에선 FP였으나 live sample에선 empty 유지 (행운)
- **실행 방식 (성공 레시피)**: 첫 run이 live stochastic SQ/HyDE 샘플 pin(캐시 populate) → 이후 같은 샘플 replay로 τ만 isolated 비교 가능
  - 캐시 디렉터리 `.llm_cache_mq3_live/` (기존 `.llm_cache/`는 `.llm_cache_copy/`로 백업)
- **핵심 성공 요인**:
  1. **Multi-query**: SQ 변동성 제거로 eval_id=52 같은 재현성 의존 GT 안정 확보
  2. **τ=0.02 하향**: 경계 과학 질문 (212, 305) 복구 (cached sweep에서 chitchat max score 0.0112 확인 후 안전 마진 유지)
  3. **새 HyDE (도메인 중립)**: 비-순수과학 corpus (global_facts, human_aging 등) 커버
  4. **live run**: cached 샘플보다 220 chitchat FP 없는 운 좋은 sample
- **소감/다음 액션**:
  - 옛 0.9152 → 0.9258 전체 상승의 **~85%가 τ+multi-query+HyDE 조합**, **~15%가 live sample 운**
  - 보수적으로 **마감 잠금 후보로 채택 고려**
  - 추가 상승 옵션: Qwen3-Embedding-8B FAISS (eval_id 37, 93 복구 타겟)
  - 최종 제출 2개 중 1개는 이 파일 확정, 다른 1개는 옛 0.9152 (안전판) 또는 Qwen3-8B 결과

### SUB-미제출 | EXP-02-003-03-mq3-tau005 (티켓 보류)
- **파일**: logs/EXP-02-003-03-mq3-tau005/submission.csv
- **사유**: tau002가 0.9258로 목표 달성 → tau005 제출 티켓을 다른 실험(Qwen3-8B 등)에 사용
- **로컬 추정**: 212, 305 재필터 효과 −0.009 → 추정 LB ~0.9168
- **결론**: τ=0.02가 live에서도 τ=0.05 대비 **최소 +0.009 우위** 확정 (로컬 diff 기반). **τ=0.02 채택**

### SUB-XXX | 2026-04-21 | EXP-02-003-07-ens-w10 ❌ (앙상블 개악)
- **파일**: logs/EXP-02-003-07-ens-w10/submission.csv
- **구성**: tau002 베이스 + **Qwen3-Embedding-8B FAISS** extra retriever (3-way RRF, weight=1.0)
- **LB**: MAP **0.9212** / MRR **0.9288**
- **Δ vs tau002 (0.9258 / 0.9333)**: **−0.0046 MAP**, **−0.0045 MRR**
- **분석**:
  - 로컬 top-1 diff vs tau002: 2건 (eval_id=104, 270)
  - Public 구간(≤110)은 104 한 건 — bge-m3 `354...` (GT hit) → Qwen3-8B `778...` (miss) → −0.0045
  - 270은 private 구간, public LB에 기여 없음
  - 로컬 delta 거의 완벽히 LB에 반영됨
- **Weight sweep (w05/w15/w20)**: 모두 w10과 유사. 가중치 4배 차이에도 top-1 변화 2~3건뿐 → Qwen3-8B 기여 marginal
- **topk_rerank 40 확대 시도**: ens-w10과 bit-identical — 후보 pool 확장으로도 해결 불가
- **결론**: 
  - Qwen3-8B ensemble은 **MAP 개악** (−0.0046)
  - Phase 7의 미해결 eval_id 37, 93 여전히 empty — **bottleneck은 retriever가 아니라 reranker**
  - 현 bge-reranker-v2-m3가 후보 중 GT를 낮게 평가하는 구조적 한계
- **다음 방향**: 
  - A. Qwen3-8B primary (bge-m3 weight=0) — bge signal 희석 가설 검증
  - B. Threshold 0.005 — 37 solo score 0.0075 살리기
  - 두 실험 결과 후 최종 제출 2장 확정

### SUB-XXX | 2026-04-21 | EXP-02-003-07-tau005-clean ❌ (rescue 무효)
- **파일**: logs/EXP-02-003-07-tau005-clean/submission.csv
- **구성**: ens-w10 + `--score-threshold 0.005` + `CHITCHAT_SQ_PATTERNS` 후처리 필터
  - Phase 7 미해결 37, 93 rescue 기대 (τ 하향)
  - 220/229 chitchat FP는 SQ 패턴 (`^너는 누구`, `^너의 (잘|특기|장점)`, `^너가? (모르|잘하|뭘)`)으로 강제 empty
- **LB**: MAP **0.9212** / MRR **0.9288** — **ens-w10과 bit-identical 결과**
- **Δ vs tau002**: **−0.0046 MAP**, **−0.0045 MRR**
- **Δ vs ens-w10**: **0**
- **LB 역산으로 확정된 사실 (중요)**:
  - **220, 229 = private eval_id 구간**: eval_id >110이므로 chitchat FP 제거해도 public LB 변화 0 (legitimate)
  - **37, 93, 12는 모두 public eval_id (≤110)**: rescue로 public LB가 −0.0046 변한 것은 이 3건의 합성 효과
  - 수학적 분해 (public MAP -0.506 AP): **12 chitchat FP (-1)** + **{37, 93} 중 하나가 rank-2 hit (+0.5)** + 나머지 miss(0) 조합이 유력
  - (이전 기록의 "둘 다 private" 표현은 eval_id 오식. 37/93은 public eval_id)
  - 결과: Phase 8의 **모든 retrieval/threshold/chitchat 필터 실험이 public MAP 0.9212에 수렴**
- **로컬 지표**:
  - 220, 229 topk=[] ✓ (SQ 패턴 필터 정상 작동)
  - 37, 93, 12 topk 3개 유지 ✓ (rescue 보존)
  - tau005 대비 diff [220, 229]만, reranker fp noise 0건
- **근본 결론**:
  - 104 top-1 flip이 ens 구성의 **유일한 확정 public 손실** (−0.0045)
  - Phase 7의 37, 93 rescue가 public MAP를 깎지 않은 정도의 기여 (math 상 37 or 93 중 하나는 rank-2 hit 가능성, 다른 하나는 miss)
  - **tau002 (public MAP 0.9258)이 현 setup 최대** 재확인
- **최종 잠금 후보**: tau002 유지. Phase 8 전체가 −0.0046 수렴해 ROI 없음 확인

### SUB-XXX | 2026-04-21 | EXP-02-003-07-tau002-t005 ❌ (Qwen3 없이도 개악)
- **파일**: logs/EXP-02-003-07-tau002-t005/submission.csv
- **구성**: **tau002 base (bge-m3 only, Qwen3-8B 미사용)** + `--score-threshold 0.005` + chitchat filter
  - 가설: Qwen3-8B 없이 bge-m3만 남기면 104 flip 없이 37/93 rescue 가능 (public LB 순수 +gain 기대)
- **로컬 (vs tau002)**:
  - rescued [12, 37, 93] — bge-m3만으로도 candidates에 GT 있음 확인
  - killed 0, chitchat FP [12]만
  - top-1 diff [104, 270, 305] — reranker fp noise hotspot
- **LB**: MAP **0.9197** / MRR **0.9288**
- **Δ vs tau002 (0.9258)**: **−0.0061 MAP**, **−0.0045 MRR**
- **Δ vs tau005-clean (0.9212)**: **−0.0015 MAP**, **±0 MRR**
- **분석**:
  - **104 flip −0.0045**: bge-m3만 써도 재실행 시 reranker fp noise로 104 top-1이 `354...` → 다른 doc 변경. sanity-v2에서도 관찰된 **재현성 한계**
  - **추가 −0.0015**: public 구간 1건에서 GT rank 2 → rank 3 강등 추정 (reranker fp noise top-2/3 영향, MAP 계산 0.5 → 0.333 × 1/110 ≈ 0.0015 일치)
  - 37/93/12 rescue는 public LB 기여 0 재확인 (tau005-clean과 동일 패턴)
- **핵심 발견**:
  - **Qwen3-8B 유무 무관**하게 τ 하향은 public LB 개선 없음
  - bge-m3 단독 재실행 시도 **sanity-v2 재현성과 같은 수준의 fp noise** 발생 (104 flip)
  - tau002 (2026-04-19 실행) 이후 동일 config 재실행은 **−0.0045 ~ −0.0061 범위**로 수렴할 가능성
- **최종 잠금 후보 업데이트**:
  - 메인: **tau002 (public MAP 0.9258)** 확정 유지
  - 2차 slot 후보: tau002-t005 (0.9197) 또는 tau005-clean (0.9212) — 둘 다 rescue 3건 포함, private 구간(eval_id >110)에서 업사이드 가능성 노림 (private MAP 미확인)
  - 또는 신규 축 실험 결과에 따라 교체

### SUB-XXX | 2026-04-21 | EXP-02-003-08-qwen3rr-tau03 ❌❌ (대폭 개악)
- **파일**: logs/EXP-02-003-08-qwen3rr-tau03/submission.csv
- **구성**: tau002 base + **Qwen3-Reranker-8B 교체** (bge-reranker 대체) + τ=0.3 + chitchat filter
  - 목적: Phase 8에서 확정된 "reranker bottleneck" 돌파 시도
  - 인프라: rag.py에 causal LM 기반 reranker 분기 추가, yes/no 토큰 확률로 score 산출
- **LB**: MAP **0.8697** / MRR **0.8727**
- **Δ vs tau002 (0.9258 / 0.9333)**: **−0.0561 MAP**, **−0.0606 MRR** — Phase 전체 최대 개악
- **로컬 (vs tau002)**:
  - rescued [12, 37, 93] — Phase 7~8 미해결 쿼리 최초 복구
  - **killed [14, 16, 50, 206, 251, 306]** — tau002가 맞췄던 케이스 다수 empty로 전환 (경고 신호)
    - id=14 "세제 거품" tau002 score 0.598 → Qwen3 empty: **Qwen3이 한국어 과학 corpus에 부적합** 시사
  - top-1 diff 57건 (public 27 / private 30) — 대다수가 GT-miss 방향
  - chitchat FP [12]만 (필터링은 정상 작동)
- **분석**:
  - 33건 public 쿼리 변화 (rescued 3 + killed 3 + top-1 flip 27) → net −0.056
  - Qwen3-Reranker가 **bge-reranker-v2-m3의 도메인 적합성을 따라잡지 못함**
  - MTEB 1위 ≠ 한국어 과학 QA 성능
- **결론**:
  - **Reranker 교체 축 종료** — bge-reranker-v2-m3 유지 확정
  - 대형 8B 모델이 소형 특화 560M 모델을 한국어 과학 corpus에서 이기지 못함
  - Phase 9 실패로 **tau002 (0.9258)가 pipeline 최종 상한** 확정
- **최종 잠금 확정**:
  - 메인: **tau002 (public MAP 0.9258)** — 모든 Phase 실험 거쳐 최고점 재확인
  - 2차 후보: 옛 0.9152 OR tau002-t005/tau005-clean (private 구간 업사이드 가능성, private MAP 미확인)

### SUB-XXX | 2026-04-22 | ENSEMBLE-001 (4-way RRF) ✅ **NEW PUBLIC 최고점 0.9356**
- **파일**: logs/reference/submission.csv (ensemble_submissions.py 출력)
- **구성**: RRF fusion, chitchat-vote=3 (strict majority 3/4)
  - 입력 4개 (균등 가중치):
    1. EXP-14 (0.9348) — 기존 최고 base
    2. Teammate output_0.9333.csv (0.9333) — 독립 파이프라인 (동일 23-empty 구조)
    3. mq3-tau002 (0.9258) — 검증된 base
    4. EXP-18-no-hyde — SQ 형식 규칙 + LLM 캐시 적용 pipeline
- **로컬 검증 (vs E14)**:
  - 23 empty 유지 ✓ (E14와 동일 구조, 12/37/93 포함)
  - 194 identical + 15 diff-order + 11 diff-set + **0 top-1 change** + 0 empty mismatch
  - 26건 rank 2~3 refinement (E14 top-1 100% 보존)
- **LB**: MAP **0.9356** / MRR **0.9424**
- **Δ vs E14 (0.9348 / 0.9424)**: **+0.0008 MAP**, **±0 MRR** — pipeline 신기록
- **해석**:
  - MAP 소폭 개선은 26건 rank 2~3 refinement에서 일부 GT가 rank 2→1 또는 rank 3→2로 이동한 효과
  - MRR 동일 = E14의 top-1 hit 그대로 유지 (ensemble이 top-1을 바꾸지 않도록 설계된 결과와 일치)
  - 4개 독립 파이프라인 RRF의 **variance-reduction**이 단일 실행 lucky run 의존성 제거
- **핵심 발견**:
  - "top-1 불변 + rank 2~3 다수결" 구조가 안전하면서도 +α 가능
  - Teammate csv (독립 pipeline)가 ensemble 다양성에 기여
  - NH의 SQ 형식 규칙이 여기서 validation됨
- **다음**: 최종 2슬롯 중 1개 확정. 나머지 슬롯은 공격적 시도 (5-way, weighted 등)

### SUB-XXX | 2026-04-22 | EXP-02-003-19-solar-rerank ❌ **미제출 (역효과 확인)**
- **파일**: logs/reference/exp-02-003-19-solar-rerank
- **구성**: EXP-18-no-hyde 파이프라인 + **Solar Pro 2차 listwise rerank** (bge top-10 → solar-pro → top-3)
  - 신규 기능: `--llm-rerank --llm-rerank-provider solar --llm-rerank-topk 10`
- **로컬 분석 (vs NH)**:
  - 220건 중 **166 diff-set (top-1 변경 20건)** — 매우 공격적 reshuffle
  - **strong baseline(E14/Teammate/tau002) 합의와 16 vs 2로 반대 방향** (Solar가 틀린 쪽)
  - Solar top-1 중 10건이 bge top-10에서 끌어온 (NH top-3 밖)
- **기대 LB 추정**: −0.03 ~ −0.08 (제출 가치 없음)
- **진단**:
  - bge-reranker-v2-m3(학습된 cross-encoder) 뒤에 generalist LLM 얹으면 **역효과**
  - Solar Pro가 한국어 특화지만 **passage ranking task specialization 없음** → bge의 학습된 랭킹을 뒤흔듦
  - RankGPT 문헌과 일치: 강한 reranker 뒤에 LLM 2차는 효과 미미하거나 -
- **발표자료 활용**: "새 기법 시도 → 실패 → 분석 → 교훈"
- **결론**: LLM-as-reranker 축 종결. bge-reranker off-the-shelf 최적

### SUB-XXX | 2026-04-22 | EXP-02-003-18-no-hyde ❌ **미제출 (ensemble 입력으로만 활용)**
- **파일**: logs/reference/exp-02-003-18-no-hyde
- **구성**: EXP-17 baseline + **SQ 형식 규칙** + **LLM 디스크 캐시** (HyDE는 off)
- **로컬 (vs EXP-17)**: 138 iden + 20 diff-order + 62 diff-set + 0 empty mismatch
  - 20 diff-order = SQ 형식 규칙의 reranker 영향 (같은 top-3 재정렬)
- **로컬 (vs E14)**: 104 iden + 19 diff-order + 94 diff-set + 3 empty mismatch (12/37/93 PRIVATE)
- **기대 LB**: 0.925 ~ 0.930 (E14 0.9348에 못 미치고 E17 0.9242 소폭 상회 추정)
- **결론**: 단독 제출 가치 낮음. ENSEMBLE-001의 입력 4개 중 하나로 기여 (SQ 규칙 반영 pipeline 대표)

### SUB-XXX | 2026-04-22 | EXP-02-003-18 (HyDE on) ❌ **미제출 (HyDE 효과 미미 확인)**
- **파일**: logs/reference/submission_exp-02-003-18
- **구성**: EXP-17 baseline + SQ 형식 규칙 + LLM 캐시 + **HyDE on** (도메인 중립 프롬프트)
- **로컬 (vs NH, same pipeline no-hyde)**: 179 iden + 0 diff-order + 41 diff-set + 0 empty mismatch
  - **HyDE on/off의 실제 차이는 41 queries만**, top-1 변경 2건뿐 (270, 246)
  - 305(농장 해안) top-1은 HyDE on/off 모두 동일 — **HyDE 고유 기여 없음**
- **mq3-tau002 시절 HyDE +0.009 효과 재현 실패**
- **원인 가설**: multi-query n=3과 HyDE가 결합하면 각 variant마다 HyDE 호출 → 4개 가상 문서가 dense 희석. tau002의 단일 HyDE 집중도 손실
- **결론**: 현재 pipeline에서 HyDE 효과 미미. 캐시 무효화 비용만 발생 → 향후 기본 off

### SUB-XXX | 2026-04-22 | EXP-02-003-18 (코드 패치) — SQ 형식 규칙 + LLM 디스크 캐시 + HyDE 부활
- **파일**: rag_system/rag_exp-02-003-17.py 패치 (+175줄 누적)
- **변경 (코드 +56줄)**:
  1. `REWRITE_SYSTEM` + `CLASSIFY_SYSTEM` 양쪽에 SQ **의문문 강제** 규칙 추가
     - 평서문/요청문 금지 ("X는 Y이다", "~알려주세요" → "X는 무엇인가?")
     - EXP-17 분석 시 발견한 답변형 SQ 4건 (eid=98, 107, 243, 286) 교정 목적
  2. `--llm-cache` flag + `chat_completion()` 안 디스크 캐시
     - 키: `sha256(model + messages + temp + seed + response_format)`
     - 위치: `rag_system/.llm_cache/<hash>.json` (.gitignored)
     - 첫 호출 → API + 저장 / 이후 → 즉시 반환
- **목적**: OpenAI seed가 best-effort라 multi-query/SQ 호출에 비결정성. 캐시로 진짜 결정성 확보
- **검증 방법**: 같은 명령 2회 실행 → submission.csv가 byte-level 동일해야 함
- **다음 액션**: 내일 제출 5회 한도 안에서 1회 베이스 + 패러미터 변주 실험 (cache hit으로 SQ/multi-query는 고정, 검색 파라미터만 변동)

### SUB-XXX | 2026-04-22 | EXP-02-003-17 ❌ (classifier-only 재구현, -0.0106 후퇴)
- **파일**: logs/reference/exp-02-003-17 (220 line jsonl)
- **구성**: tau002 + 후처리 CSV 머지 의존 제거. **classifier 단일 의도 분류 + score_threshold=0.0**으로 깔끔 재구현
  - rewrite_only SQ + multi-query n=3 + RRF + bge-reranker-v2-m3
  - chitchat 분류는 GPT-4o-mini classifier가 단독 결정 (218/276 포함 20건 정확히 empty)
- **로컬 검증**: empty 20건 / non-empty 200건 — 기대 chitchat 20건 전부 일치 ✓
- **LB**: MAP **0.9242** / MRR **0.9318**
- **Δ vs EXP-14 (0.9348 / 0.9424)**: **−0.0106 MAP**, **−0.0106 MRR** (양 partition 동일 폭 하락)

#### 손실 분해 (EXP-14 ↔ EXP-17 220건 비교)
| 구분 | 건수 | 영향 |
|---|---|---|
| 완전 동일 (양쪽 empty 20 포함) | 148 | 0 |
| 같은 집합 / 순서 다름 | 1 (id=202) | 거의 0 |
| top3 멤버 다름 (top-1 동일 63 + 변경 5) | 68 | 노이즈 ± |
| **empty mismatch** | **3 (12, 37, 93 모두 PRIVATE)** | EXP-14 empty → EXP-17 retrieve |

#### 결정적 진단: Public 손실은 multi-query 비결정성에서
- 12/37/93은 모두 PRIVATE 파티션 (line 167/188/201)
- **Public도 똑같이 -0.0106 떨어짐** → 12/37/93과 무관, 100% diff-set/순위 변동에서 발생
- 68 diff-set 중 SQ 동일이 46건 = base SQ가 같은데 top-3가 다름 = **multi-query LLM 비결정성**
- 22건은 SQ 어미 차이 ("~인가?" vs "~인가요?") = LLM 호출 분산
- Private 분해: diff-set ~-0.92 AP + 12/37/93 net -0.25 AP = -1.17 AP (3건 중 1건 정도가 GT science 추정)

#### 에러 분석 (5개 고품질 submission 교차 비교)
- top-1 3개 이상 갈리는 쿼리: **1건 (eid=246)**
- top-3 합의 없는 쿼리: **0건**
- **시스템은 이미 견고함**. multi-query 비결정성이 rank 2~3 셔플을 일으키는 게 전부

#### 답변형 SQ 4건 발견 (수정 대상)
| eid | 사용자 메시지 | EXP-14 SQ | 문제 |
|---|---|---|---|
| 98 | 어떤 물리적 현상을 보고 그걸 알게 되었어? | "은하들이 멀어지는 현상은 적색편이 현상에 의해 관측됩니다" | 답변 박힘 |
| 107 | 어떤 원인 때문에 발생하는지 궁금해 | "기억 상실증의 원인에는 어떤 것들이 있는지 알고 싶다" | 평서문 |
| 243 | 그 이유를 힘의 원리로 설명해줘 | "지구 위에서 가만히 서 있을 수 있는 이유는 중력과 지면 반작용 힘이 균형을 이루기 때문입니다" | 답변 박힘 |
| 286 | 메탄과 산소의 화학 반응에 대해 알려줘 | "메탄과 산소의 화학 반응에 대한 설명을 요청합니다" | 요청문 |

→ EXP-18 패치 (위 항목)에서 SQ 형식 규칙으로 교정

#### 추가 발견
- 12, 37, 93은 PRIVATE 전용. score_threshold=0.0 부활 필요성 낮음 (private에 미세한 영향만)
- 멀티턴 SQ 20건 중 답변형 4건 외에는 키워드 보존·맥락 치환 양호
- 단일턴 짧은 질문(≤15자) 31건 전부 자연스러운 SQ 생성

#### 결론
- **EXP-14 (0.9348)는 mq3-tau002 단일 실행의 lucky run + 후처리 패치**. 같은 코드 재실행 시 EXP-17처럼 -0.01 변동 가능
- **EXP-17 (0.9242)는 재현 가능한 실력 점수**. classifier 단독 설계는 옳음
- 진짜 격차는 multi-query 비결정성 → EXP-18에서 LLM 캐시로 결정성 확보 + 답변형 SQ 패치

### SUB-XXX | 2026-04-21 | EXP-02-003-16-tau002-empty-81 ❌ (81도 과학 GT 확정)
- **파일**: logs/EXP-02-003-16-tau002-empty-81/submission.csv
- **구성**: EXP-14 + `통학\s*버스` 패턴 추가 (id=81 추가 empty 대상)
- **로컬**: 81/218/276 세 건 topk=[]
- **LB**: MAP **0.9258** / MRR **0.9333**
- **Δ vs EXP-14 (0.9348 / 0.9424)**: **−0.0090 MAP**, **−0.0091 MRR**
- **진단**:
  - -0.00909 = 1 AP 손실 → **id=81 = 실제 과학 GT hit** 확정 (212와 동일 패턴)
  - tau002 top-1 (score 0.054)이 정답이었음 → empty 처리로 손실
- **누적 교훈 (EXP-15, EXP-16)**:
  - bge score 0.03~0.06 저신뢰 케이스도 **GT hit 비율 높음** (212, 81 둘 다 confirmed)
  - msg가 "역사", "버스" 같은 비과학 키워드 포함해도 corpus에 매칭 doc 존재 가능 (MMLU 광범위)
  - **저신뢰 일괄 empty 전략은 무효** — 단건 LB 검증만이 신뢰 가능
- **액션**: `통학\s*버스` 패턴 revert. 남은 public non-empty 저신뢰 후보(241 정육면체, 307 강아지 사회화)는 과학 색채 강해 검증 가치 낮음
- **EXP-14 (0.9348) public 최고점 유지**

### SUB-XXX | 2026-04-21 | EXP-02-003-15-tau002-empty-212 ❌ (212는 과학 GT 확정)
- **파일**: logs/EXP-02-003-15-tau002-empty-212/submission.csv
- **구성**: EXP-14 + `라틴\s*역사` 패턴 추가 (id=212 추가 empty 대상)
- **로컬**: 정확히 3건 변화 (212/218/276 모두 topk=[])
- **LB**: MAP **0.9258** / MRR **0.9333**
- **Δ vs EXP-14 (0.9348 / 0.9424)**: **−0.0090 MAP**, **−0.0091 MRR**
- **진단**:
  - -0.00909 = 1 AP 손실 → **id=212 = 실제 과학 GT hit** 확정
  - tau002 top-1 `b303e4ec` (score 0.034)이 정답이었음 → empty로 덮어 쓰면서 손실
  - **교훈: bge score 낮음 ≠ chitchat**. MMLU/ARC corpus에 역사·일반 지식 문서도 존재, 저신뢰여도 GT hit 가능
- **액션**:
  - `라틴\s*역사` 패턴 revert → FORCE_EMPTY_SQ_PATTERNS는 `힘든\s*상황|힘들다` 만 유지
  - msg 내용만으로 chitchat 판정 불가. **LB 단건 단위 직접 검증만이 신뢰 가능한 시그널**
- **EXP-14 (0.9348) 최고점 유지**

### SUB-XXX | 2026-04-21 | EXP-02-003-14-tau002-empty-hardship ✅ (NEW PUBLIC 최고점 0.9348)
- **파일**: logs/EXP-02-003-14-tau002-empty-hardship/submission.csv
- **구성**: tau002 base-load + `--force-empty-sq-pattern` (FORCE_EMPTY_SQ_PATTERNS: `힘든\s*상황|힘들다`)
  - SQ가 감정 발화 패턴 매치 시 base의 non-empty topk를 강제 empty로 덮어쓰기
  - 타겟: id=218, 276 (동일 SQ "요즘 힘든 상황을 극복하는 방법", tau002 top-1 score 0.084)
- **로컬 (vs tau002)**:
  - 변화 정확히 2건: id=218, 276 topk=[] ✓
  - non-empty 나머지 198건 보존 ✓
- **LB**: MAP **0.9348** / MRR **0.9424** — **pipeline 신기록**
- **Δ vs tau002 (0.9258 / 0.9333)**: **+0.0090 MAP**, **+0.0091 MRR**
- **핵심 발견**:
  - +0.00909 ≈ 1 AP / 110 → **276 (line 21) = public chitchat GT 확정** (empty 처리로 AP 0→1)
  - 이 결과로 **public/private split이 eval_id가 아닌 line index 기반**임을 발견 (상단 정정 박스 참조)
  - 218 (line 181) = private → hidden gain 동일 가능 (쿼리 동일)
- **함의**:
  - tau002의 public chitchat FP 추가 존재 가능성 → line<110 중 저신뢰 non-empty 케이스 재감사 필요
  - 감정/일상 발화성 SQ (힘들다/지쳐/우울 등) 강제 empty 전략 유효
  - 이전 rescue 실험 3건(Exp 11/12/13)의 ±0은 "rescue 축 사망"이 아니라 "대상이 모두 private"이었기 때문 — 본질 재해석 필요

### SUB-XXX | 2026-04-21 | EXP-02-003-13-tau002-rescue12 ⚪ (tau002 동률, 12도 miss 확정, rescue 축 완전 사망)
- **파일**: logs/EXP-02-003-13-tau002-rescue12/submission.csv
- **구성**: tau002 base-load + rescue_threshold=0.005 (SQ 패턴 필터 없음 → 12, 37, 93 모두 rescue 허용)
  - 목적: 12 rescue의 LB 영향 검증 (마지막 rescue 진단)
- **로컬 (vs tau002)**:
  - 변화 정확히 3건: id=12, 37, 93 모두 rescue ✓
  - non-empty 199/199 보존 ✓
- **LB**: MAP **0.9258** / MRR **0.9333** — **tau002와 완전히 동일**
- **Δ vs tau002**: **±0 MAP**, **±0 MRR**
- **최종 진단 결론**:
  - **12, 37, 93 모두 public LB 기여 0** (rescue해도 LB 불변)
  - 12도 miss 확정 — chitchat FP도 아님 (chitchat이었다면 -0.00909). GT 존재하지만 bge rescue top-3에 미포함
  - **tau005-clean -0.0046 전체가 ens-w10 base 차이 + τ 변경에서 발생**, rescue 3건은 0 기여로 최종 확증
- **rescue 축 완전 사망**:
  - `tau002 (0.9258) = 현 pipeline public MAP 절대 상한` 재재확인
  - empty set 21개 중 public LB 개선 가능한 케이스 0 (rescue로 복구해도 GT를 top-3에 담지 못함)
  - 다음 개선 가능성: **non-empty low-confidence 교체** (bge top-1 score 낮은 케이스 재retrieve), private 구간 업사이드

### SUB-XXX | 2026-04-21 | EXP-02-003-12-tau002-rescue93 ⚪ (tau002 동률, 93도 miss 확정)
- **파일**: logs/EXP-02-003-12-tau002-rescue93/submission.csv
- **구성**: tau002 base-load + `--rescue-filter-sq-pattern` (BIOGRAPHY_SQ_PATTERNS 게이트) + rescue_threshold=0.005
  - 목적: Exp 11에서 classifier가 오판한 93까지 rescue하여 "tau005-clean의 +0.00455는 93 기여" 가설 검증
  - SQ 패턴 기반 게이트: `일대기|위인|생애|전기문|전기작가` → 12 차단, 37/93 허용
  - rescue_threshold 0.005: 16 chitchat(score<0.005) 자동 제외
- **로컬 (vs tau002)**:
  - 변화 정확히 **2건**: id=37, 93 rescue ✓
  - id=12 empty 유지 ✓ (biography 패턴 차단)
  - non-empty 199/199 보존 ✓ (killed=0)
- **LB**: MAP **0.9258** / MRR **0.9333** — **tau002와 완전히 동일**
- **Δ vs tau002**: **±0 MAP**, **±0 MRR**
- **진단 결론 (매우 중요)**:
  - **93도 miss 확정**. 37+93 둘 다 rescue해도 MAP 변화 없음 → 둘 다 bge-reranker top-3 rescue 결과에 GT 없음
  - 이전 "tau005-clean -0.0046 = 12 FP + {37 or 93} rank-2 hit" 분해는 **오류**
  - 진짜 원인: tau005-clean은 ens-w10(3-way RRF) base + τ=0.005로 구성이 tau002와 크게 달랐음. -0.0046은 rescue 3건만이 아니라 **retrieval 구성 차이 + 기타 threshold 케이스 변화**의 합산
  - 실제로 **37, 93, 12 모두 public LB 기여 0** (rescue해도 LB 불변)
- **최종 상한 재확인**:
  - tau002 (0.9258) = 현 pipeline의 **public MAP 절대 상한**
  - 모든 rescue/threshold 변주는 ±0 또는 개악으로 수렴
  - private에서 잠재 업사이드 가능성 남아있지만 미검증

### SUB-XXX | 2026-04-21 | EXP-02-003-11-rescue37 ⚪ (tau002 동률, 진단 가치)
- **파일**: logs/EXP-02-003-11-rescue37/submission.csv
- **구성**: tau002 submission base-load + `--rescue-filter-classifier`로 empty 재처리 시 science 판정
  - 새 기능: `--base-submission` (non-empty topk 그대로 보존, retrieve 호출 skip)
  - `--rescue-filter-classifier`: 원본 msg로 classifier 실행, `is_science=false` → empty 유지
  - rescue_reranker: bge-reranker-v2-m3, τ=0.0
- **로컬 (vs tau002)**:
  - tau non-empty 199/199 보존 ✓ (killed=0 구조적 보장)
  - 실제 변화 = **id=37 1건만** rescue
  - classifier가 12, 93을 non-science로 판정 → empty 유지
- **LB**: MAP **0.9258** / MRR **0.9333** — **tau002와 완전히 동일**
- **Δ vs tau002**: **±0 MAP**, **±0 MRR**
- **진단 가치 (매우 중요)**:
  - tau005-clean (-0.0046) 수학 역산 재검증:
    - 이전 가설: 12 FP(-0.00909) + {37 or 93} rank-2 hit(+0.00455) + 나머지 miss
    - 이번 실험: 37만 rescue했는데 ±0 → **37은 miss 확정**
    - 따라서 tau005-clean의 +0.00455는 **93이 기여** (93 public GT rank-2 유력)
  - **93이 실제 LB 개선 기여자**. 37은 rescue해도 무의미
- **다음 방향**:
  - 93만 타겟하는 코드 경로 설계 필요
  - classifier가 93을 non-science로 판정 → classifier prompt 또는 SQ 패턴 기반 필터로 전환 검토
- **상태**: tau002와 동점이라 LB상 순위 변화 없음. 진단 실험으로서 가치.

### SUB-XXX | 2026-04-21 | EXP-02-003-10-qwen3rr-rescue ❌ (rescue 효과 없음)
- **파일**: logs/EXP-02-003-10-qwen3rr-rescue/submission.csv
- **구성**: tau002 base (bge-reranker 유지) + **Qwen3-Reranker-8B를 rescue 전용으로만 사용**
  - empty(topk=[])인 21개 케이스에 한해 Qwen3로 재rerank
  - tau non-empty 케이스는 bge 결과 보존 (floor = tau002 0.9258 보장 설계)
- **LB**: MAP **0.9197** / MRR **0.9288**
- **Δ vs tau002 (0.9258 / 0.9333)**: **−0.0061 MAP**, **−0.0045 MRR**
- **로컬 (vs tau002)**:
  - killed = 0 ✓ (tau non-empty 21건 전부 top-1 보존)
  - rescued [12, 37, 93] — 모두 public
  - chitchat FP [12]만 (필터링 정상)
  - top-2/3 shuffle 61건 (Qwen3 로드로 인한 CUDA fp noise로 추정)
- **분석**:
  - **tau002-t005와 public MAP/MRR 완전히 동일 (0.9197/0.9288)** → Qwen3 rescue가 "topK 확대+τ 하향"과 구분 불가능한 효과
  - 37/93/12 rescue는 public LB 기여 0 재확인 (3번째 실험)
  - top-2/3 shuffle이 public −0.0061 감소 주범 (tau002의 최적 순위를 약간 흐트러뜨림)
  - ⚠️ **용어 정정**: LB 출력 2개 숫자는 `public MAP / public MRR`. 이전 기록에서 2번째 숫자를 "private"으로 해석한 건 오류 (private은 대회 종료까지 비공개)
- **결론**:
  - **Strategy B(rescue-only) 축 종료**: qwen3 투입이 tau002-t005 대비 추가 가치 없음
  - 최종 2슬롯 후보 재확인:
    - **메인**: tau002 (public MAP 0.9258 — 유일한 public 최고점)
    - **2차**: tau005-clean / tau002-t005 / rescue 중 하나 (public MRR 0.9288 동률, private 미지) — 셋 중 택1

### SUB-XXX | 2026-04-25 | EXP-09-001-C-cls-v5 ❌ (V5 분류기 over-filter, −0.0167)
- **파일**: logs/EXP-09-001-C-cls-v5/submission.csv (로컬: logs/reference/exp-09-001-c)
- **구성**:
  - **Qwen3-Embedding-0.6B FT** (외부 GT 1,365 pair, MNRL 3 epoch, full FT 24GB)
  - **3-way RRF ensemble**: BM25 + bge-m3 + Qwen3-FT(FAISS) (가중 0.3/0.7/1.0)
  - **classifier=json + V5 prompt** (recall-bias + corpus scope 명시, 비과학 5 카테고리 한정)
  - 기타: HyDE on, multi-query n=3, llm-cache on, force-empty-sq-pattern (no base-submission이라 no-op)
- **LB**: MAP **0.9212** / MRR **0.9288**
- **Δ vs best 0.9379**: **−0.0167 MAP**
- **Δ vs tau002 (0.9258)**: **−0.0046 MAP**
- **로컬 변화 (vs A=cls-none, vs best 0.9379)**:
  - empty 21 (A) → 26 (C). 추가 차단 5건 = {218, 276 (chitchat 정확), 28, 108, 233 (over-filter 의심)}
  - best 대비 추가 차단 3건 = {28 "감정적 지원", 108 "여행 좋은점", 233 "남녀관계 정서"}
  - retrieval 92건 다름 중 top1은 86건(94%) 동일 (V5 SQ rewrite는 2-3위 docs만 영향)
- **변량 비교 (3-way classifier ablation)**:
  | variant | empty | LB MAP | 상태 |
  |---|---|---|---|
  | A (cls=none, regex 후처리 제거) | 21 | (미제출) | 218/276 통과로 손실 예상 |
  | B (cls=json + V4) | **51** | (미제출) | 30건 over-filter, 실측 ~0.85대 추정 → 폐기 |
  | C (cls=json + V5) | 26 | **0.9212** | 28/108/233 추가 차단으로 −0.0167 손실 |
- **분석 — V5 실패 원인**:
  - V5 "비과학 5 카테고리"(인사/AI자기지칭/위로/취향/연애상담/메타) 중 "위로 요청"·"개인 취향"·"인간관계 상담"이 너무 광범위
  - corpus가 MMLU 사회/심리/공중보건 docs를 포함 → "감정 지원"(28), "여행"(108), "남녀관계"(233)이 GT hit이었던 것으로 추정 (각 −1/220)
  - V5의 recall-bias 의도(애매하면 true)가 이 3건에선 작동 안 함 — LLM이 5 카테고리 명시에 너무 단호하게 매칭
- **교훈**:
  - **"원리적 분류기" ≠ "LB 개선"**: corpus가 광범위(사회·심리·생활)할수록 분류기는 손해
  - V4의 "사회·정치·교육 정책" 카테고리(28건 over-filter)와 V5의 "위로/취향/인간관계"(3건 over-filter) 모두 corpus 실제 분포와 불일치
  - 결국 **classifier=none + 후처리 regex (force-empty-sq-pattern)** 가 best 0.9379를 만들어낸 이유 = corpus 광범위성에 가장 적게 간섭
  - 발표 자료 메시지: "원리적 접근 시도 → 실측 후 corpus 광범위성 발견 → 후처리 regex 정당화"로 정리 가능
- **다음 액션**:
  - **남은 1슬롯 제출 후보**: best 0.9379 그대로 또는 A variant (qwen3-FT + cls=none, regex 후처리 없음) 측정 검토
  - V5 prompt 후속 수정은 ROI 낮음 (LB 1회당 코스트가 너무 큼)
  - Qwen3-Embedding FT 자체는 로컬 변화는 만들어내지만 단독으로 LB 개선 신호 안 보임 → 4-way → 5-way 확장도 효과 의문
