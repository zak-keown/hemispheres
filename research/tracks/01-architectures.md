# Hemispheres literature survey — Track: architectural prior art for separable / swappable knowledge

Compiled 2026-09-22. Sources: arXiv abstracts/HTML, ar5iv, official blogs/READMEs, fetched live during this session.
Confidence tags: **[V]** = numbers checked against the arXiv abstract/HTML or official page during this session; **[M]** = from my prior knowledge of the paper, not re-checked this session (IDs/dates believed correct); **[?]** = could not verify / secondary source only / treat with caution.

---

## 0. Executive summary

- The idea "put knowledge in a separate component" has been tried at every interface point of a transformer: the **input context** (REALM, RAG, Atlas, Self-RAG, LMLM), **attention KV** (RETRO, Memorizing Transformers, Memory³, KBLaM, Cartridges, MemoryLLM/M+), the **FFN/residual stream** (PKM, PEER, UltraMem, Memory Layers at Scale, Engram, Apple hierarchical memories, K-Adapter, LoRA knowledge modules, PRAG/DyPRAG, Doc-to-LoRA), the **output distribution** (kNN-LM, Memory Decoder, MLP Memory), and **fast weights updated at test time** (Titans, ATLAS, Larimar).
- **2025–2026 produced the strongest evidence yet that knowledge is separable in practice**: DeepSeek's Engram (Jan 2026) shows that when the hashed-n-gram memory is switched off, TriviaQA retains only 29% of performance while reading comprehension retains 81–93% [V]; Apple's hierarchical memories (ICLR 2026) show a 160M "anchor" + 18M fetched memory ≈ a model >2× larger, and memory blocks can be blocked/deleted/replaced without retraining the anchor [V]; LMLM (Cornell, 2025) masks factual values out of the pretraining loss so the model learns to *look them up* in a triple DB, enabling unlearning by DB deletion [V].
- **Closest prior art, mid-2026:**
  - *Cornell's LMLM → Co-LMLM* (arXiv:2505.15962, 2607.07707): the core is trained with facts loss-masked. It emits a hidden-state query, and free-text facts come back from a 2.2B-entry KB. A 360M model reaches SimpleQA 21.7, and deleting KB entries passes TOFU unlearning. It stays tiny, and the key space is tied to the core's hidden states.
  - *Tsinghua's DMoE* (arXiv:2606.14243): per-passage LoRA experts on the last FFN of a *frozen, knowledge-heavy* base, fired only when the base is uncertain. This is augmentation, not a split.
  - *Negative results on weight-writes:* Baseten/O'Neill (arXiv:2607.11020) finds that sequentially weight-written facts decay (1% retained after 20 writes for bare statements) and that "the reliable channel is context rather than the weights." Hypernetwork writers (Doc-to-LoRA; scaling laws in arXiv:2607.19604) make weight-writes gradient-free but need a writer about as big as the target.
- **But no system yet delivers all of: (a) a knowledge-light core that genuinely lacks the externalized facts, (b) a cheap, gradient-free, per-fact write/delete op in a format the core natively consumes, (c) edits that propagate through multi-hop reasoning, (d) at non-toy scale.** Every separable system leaks: blocking Apple's memory drops atomic-number accuracy 70%→20% (not 0); Engram-off TriviaQA is 29% (not 0); LMLM concedes "some factual knowledge remains" in parameters. Parametric memories (memory layers, Engram, Apple) lack a write op that isn't gradient descent. Write-op systems (KBLaM, Memory³, LMLM) are retrofitted, small, or restricted to entity triples.
- The dominant practical incumbent remains **RAG over a knowledge-heavy frozen LLM**, whose key failure for Hemispheres' goals is **prior dominance**: the frozen core overrides contradicting retrieved facts (vanilla RAG counterfactual compliance only 20–52% in TokenMem's measurements [V]). A knowledge-light core is the principled fix, and that is the gap.

---

## 1. Retrieval-integrated LMs (knowledge as text / nearest neighbours)

### 1.1 REALM — Guu et al., Google. arXiv:2002.08909 (Feb 2020) [M]
- **Mechanism:** Masked LM whose prediction is marginalized over documents retrieved by a learned dense retriever (MIPS over Wikipedia). Retriever trained end-to-end through the MLM likelihood (with asynchronous index refresh).
- **Where knowledge lives:** Wikipedia text corpus + dense index; the encoder still stores a lot parametrically.
- **Updates:** Swap/re-embed the corpus; LM not retrained (though index must be re-embedded).
- **Results:** NQ-Open ~40.4 EM; +4–16 points absolute over prior open-QA SOTA at the time.
- **Limits:** Expensive index refresh during pretraining; encoder-only (extractive QA).

### 1.2 RAG — Lewis et al., FAIR. arXiv:2005.11401 (May 2020) [V]
- **Mechanism:** BART generator conditioned on top-k passages from DPR over a Wikipedia dense index; RAG-Sequence / RAG-Token marginalization.
- **Where knowledge lives:** Non-parametric text index + parametric generator.
- **Updates:** **Index hot-swapping** (Sec. 4.5): 70% accuracy on 2016 world leaders with 2016 index; 68% on 2018 leaders with 2018 index; mismatched 12% / 4% [V via ar5iv]. Canonical demonstration that swapping the store updates behaviour without retraining.
- **Results:** NQ 44.5 EM (RAG-Seq), 44.1 (RAG-Tok) [V].
- **Limits:** Generator still carries parametric knowledge that can conflict with retrieval; context-length cost; retrieval errors propagate.

### 1.3 RETRO — Borgeaud et al., DeepMind. arXiv:2112.04426 (Dec 2021) [V]
- **Mechanism:** Chunked cross-attention: input split into 64-token chunks; for each, neighbours retrieved (frozen BERT embeddings, kNN) from a **2T-token** database, encoded by a small encoder, and cross-attended in decoder layers.
- **Where knowledge lives:** Retrieval DB (text chunks + continuations).
- **Updates:** Change DB without retraining (frozen retriever).
- **Results:** 7.5B RETRO ≈ GPT-3/Jurassic-1 on the Pile with **25× fewer params**; gains roughly **constant from 150M to 7B** (the paper's own claim) [V]; RETROfitting a pretrained model with ~3% of pretraining data nearly matches from-scratch RETRO.
- **Limits / skeptical notes:** Test-set leakage — gains partly from overlap between eval text and the DB; RETRO still beats baselines at leakage α≥12.5% [V]. Norlund et al. (arXiv:2305.16243, ACL 2023) [V]: replacing dense retrieval with BM25 yields 13.6% lower perplexity, and RETRO's loss reduction "almost exclusively" stems from surface token overlap rather than semantic use.

### 1.4 RETRO follow-ups (NVIDIA)
- **"Shall We Pretrain Autoregressive LMs with Retrieval?"** Wang et al., arXiv:2304.06762 (Apr 2023, EMNLP 2023) [V]: Reproduced RETRO 148M–9.5B, 330B-token DB. RETRO "largely outperforms GPT on knowledge-intensive tasks, but is on par with GPT on other tasks"; original RETRO was weak at open-domain QA, needed RETRO++ (feed top evidence into the decoder context; +8.6 EM NQ).
- **InstructRetro / Retro 48B**, Wang et al., arXiv:2310.07713 (Oct 2023) [V]: continued-pretrain 43B GPT on +100B tokens retrieving from 1.2T tokens; better perplexity at +2.58% GPU hours; after instruction tuning +7% short-form QA/RC, +10% long-form QA, +16% summarization vs GPT-43B. **Key finding: the retrieval encoder can be ablated at inference ("comparable results")** — i.e., after instruction tuning, the chunk cross-attention pathway is not what matters; the benefit looks like a better decoder plus context-RAG.
- **On "retrieval gains shrink at scale" (asked to verify):** *Mixed / not cleanly supported.* RETRO itself reports constant gains to 7B. Evidence that is supportive: Mallen et al. (arXiv:2212.10511, ACL 2023) [V] — scaling improves popular-entity knowledge, "scaling fails to appreciably improve memorization of factual knowledge in the long tail", and unassisted large LMs remain competitive with retrieval for popular entities (so retrieval's *marginal* value concentrates on the long tail). Singh et al. "To Memorize or to Retrieve" (arXiv:2604.00715, Apr 2026) [V]: retrieval gains depend on capacity and pretraining exposure and are front-loaded (median 91% of max gain at ~1 retrieval token per parameter); smaller models gain more in perplexity, larger models more in accuracy. The InstructRetro encoder ablation is the strongest hint that *architectural* retrieval pathways (as opposed to context RAG) lose relevance at scale. I found no clean paper titled/claiming "RETRO gains vanish at scale" — flag **[?]**.
- **Samuel et al., "More Room for Language"** arXiv:2404.10939 (NAACL 2024) [V]: retrieval-augmented pretraining makes models "save substantially less world knowledge in their weights" — direct evidence that *training with a knowledge store produces a more knowledge-light core*, but also found weaker global-context understanding. Important for Hemispheres.
- **MassiveDS**, Shao et al. arXiv:2407.12854 (Jul 2024) [V]: 1.4T-token datastore; LM performance improves monotonically with datastore size; small model + big datastore can beat larger LM-only models on knowledge-intensive tasks; (I recall smaller benefits on reasoning-heavy tasks — [M]).

### 1.5 Atlas — Izacard et al., Meta. arXiv:2208.03299 (Aug 2022; JMLR 2023) [V]
- **Mechanism:** Contriever retriever + T5 Fusion-in-Decoder, jointly pretrained; several retriever-training losses.
- **Updates:** TempLAMA: trained with 2017 index → 57.7% on 2017 answers, 1.5% on 2020; **swap to 2020 index zero-shot → 53.1%**; closed-book T5 only 3.5% in the reverse setting [V]. Index PQ-compressed 49GB→4GB with negligible loss [V].
- **Results:** Atlas-11B: >42% NQ and 84.7% TriviaQA with 64 examples, ~3 points above PaLM-540B (50× more compute) [V].
- **Limits:** Encoder–decoder, short outputs; retrieval cost; reasoning beyond QA not demonstrated.

### 1.6 kNN-LM — Khandelwal et al. arXiv:1911.00172 (Nov 2019, ICLR 2020) [M]
- **Mechanism:** Interpolate the LM's next-token distribution with a distribution from kNN search over a datastore of (context-hidden-state → next token) pairs. **Output-distribution interface.**
- **Updates:** Datastore swap gives domain adaptation with no training.
- **Results:** WikiText-103 ppl 15.79 (−2.9) [M].
- **Limits:** Datastore is huge (one entry per training token); Geng, Zhao, Rush "Great Memory, Shallow Reasoning" arXiv:2408.11815 (NAACL 2025) [V]: kNN-LMs help memory-intensive tasks but fail at reasoning that integrates multiple facts, **even with oracle retrieval** — an interface-location lesson: injecting knowledge only at the output distribution does not let the core reason over it. Also Wang et al. 2023 (arXiv:2305.14625) [M]: kNN-LM does not improve open-ended generation quality.

### 1.7 Memorizing Transformers — Wu et al., Google. arXiv:2203.08913 (Mar 2022) [M]
- kNN attention in one layer over a non-differentiable external memory of past (key, value) pairs (up to 262K tokens). Gains on long documents/code/math proofs. It is a *context* memory, not a curated knowledge store; can ingest new material at test time by just reading it.

### 1.8 Self-RAG — Asai et al. arXiv:2310.11511 (Oct 2023, ICLR 2024) [V]
- LM trained with reflection tokens to decide when to retrieve, and to critique relevance/support. 7B/13B beat ChatGPT and RAG-Llama2-chat on open QA, fact verification; better citation accuracy. Knowledge lives in the corpus; the base still carries parametric knowledge. Relevant as a *control policy* for when a core should consult the store.

### 1.9 Parametric RAG (PRAG) — Su et al. arXiv:2501.15915 (Jan 2025, SIGIR 2025) [V]; DyPRAG — Tan et al. arXiv:2503.23895 (Mar 2025) [V]
- **PRAG mechanism:** Offline, each document is converted (via augmentation + LoRA training) into a per-document LoRA; at inference, LoRAs of retrieved docs are merged into the FFN. **Knowledge format: weights; interface: FFN.**
- **DyPRAG:** A small hypernetwork ("parameter translator") maps a document to LoRA parameters at test time — no per-document training/storage.
- **Updates:** Add a document → train a LoRA (PRAG) or run hypernetwork (DyPRAG). Base frozen.
- **Limits:** Tang et al., "Understanding Parametric Knowledge Injection in RAG," arXiv:2510.12668 (Oct 2025) [V]: parametric representations capture document-level semantics and mostly affect deeper FFNs, providing "high-level guidance but limited evidence consolidation"; P-RAG alone does not consistently beat token RAG; best in combination. i.e., **LoRA-from-document does not reliably carry precise facts**.

### 1.10 DMoE — "Decoupled Mixture-of-Experts for Parametric Knowledge Injection." Yue, Su, Ai, Tang, Wang, Kang, Zhan, Liu (Tsinghua; same group as PRAG/DyPRAG). arXiv:2606.14243 (12 Jun 2026) [V, full HTML read]
- **Mechanism:** A PRAG-style bank of **one LoRA expert per passage** (27,613 Wikipedia passages → 27,613 experts). Each expert: LoRA r=4, α=16, ~123K params (~481 KiB), trained for 1 epoch on the passage + one paraphrase + three generated QA pairs, base frozen (~10 s per expert on an A100; 76.7 A100-GPU-hours for the bank; 13.08 GiB on disk). **Experts attach only to the final-layer FFN.** A token-level **uncertainty-aware trigger** fires when next-token entropy TU_t > τ (default 2.0); when fired, a **BM25 lexical router** over each expert's text surrogate (passage + augmentations) picks top-k=3 experts, whose LoRA deltas are summed onto the last FFN (θ' = θ + ΣΔθ_i).
- **Why last layer:** Mid-network experts would change hidden states and invalidate cached KV for later tokens; last-layer placement provably leaves all KV intact. Ablation: putting experts at 25/50/75/100% of FFN layers degrades accuracy sharply; last-only is best. KV reuse gives 1.3–5.1× latency and 1.2–2.5× GPU-memory reduction.
- **Results (Llama-3.2-1B-Instruct; also Qwen2.5-1.5B):** best or tied-best on 11/14 metrics. E.g., HotpotQA EM 0.180 vs PRAG 0.073 / SFT-LoRA 0.077 / Basic-RAG 0.170; Quasar-T EM 0.313 vs 0.220 / 0.213 / 0.280; CWQ EM 0.247 vs PRAG 0.250; StrategyQA 0.567 vs 0.560 / 0.553 / 0.433. Latency 2.67 s/sample vs PRAG 1.36, FLARE 9.26; 7.24 GB vs PRAG 4.83 GB. Versus a coupled MoE (OLMoE-1B-7B): 7.5× faster, 3.6× less GPU memory.
- **Updates:** Add = train one ~10 s LoRA + insert into BM25 index. Delete = remove adapter directory and index entry. No backbone, router, or other-expert retraining.
- **How it differs from a reasoning/knowledge split (the key point):** DMoE is **knowledge *augmentation* of an unchanged, knowledge-heavy base**, not a split. (i) The base is frozen and "remains unchanged"; its parametric knowledge is fully retained, and the paper makes no attempt to make it knowledge-light. So **deleting an expert removes only the injected delta, not the base's own copy of the fact**, and a changed fact must beat the base's prior. (ii) The router consults experts *only when the base is uncertain* (entropy > τ). So the base's confident priors (including stale or wrong ones) are never checked against the store, which is the opposite of a design in which the reasoner always defers to the knowledge component. (iii) Knowledge enters only as a rank-4 delta on the final FFN, i.e., at the output end after all reasoning layers. That is structurally close to the output-level interfaces (kNN-LM, Memory Decoder), which have known "shallow reasoning" limits. Multi-hop gains (HotpotQA) come from retrieval-triggered steering of the final projection, not from the core reasoning over the retrieved knowledge. (iv) Granularity is passage-level weights, with gradient-trained writes (~10 s each), no gradient-free compile. (v) Scale is 1–1.5B with ~27K passages. Conflict, counterfactual and deletion-leakage evaluations are absent.
- **Limits (stated or implied):** routing quality depends on the lexical query; entropy is an imperfect proxy for knowledge gaps; bank size vs disk trade-off; no conflict/OOD analysis.

---

## 2. Memory layers / sparse lookup (knowledge as trainable key–value slots in the FFN position)

### 2.1 Product Key Memory (PKM) — Lample et al., FAIR. arXiv:1907.05242 (Jul 2019, NeurIPS 2019) [M]
- Replace some FFNs with a key–value memory addressed by product keys (two half-keys → √N×√N Cartesian product) enabling exact top-k over ~1M slots cheaply. 12-layer + memory beat 24-layer baseline, ~2× faster inference. Knowledge in value slots; update requires training.

### 2.2 PEER / Mixture of a Million Experts — Xu Owen He, Google DeepMind. arXiv:2407.04153 (Jul 2024) [V]
- Product-key retrieval over >1M single-neuron experts; better performance–compute trade-off than dense FFN and coarse MoE on LM tasks. No knowledge-editing study. [Exact numbers not re-checked.]

### 2.3 UltraMem — Huang et al., ByteDance Seed. arXiv:2411.12364 (Nov 2024, ICLR 2025) [V]; UltraMemV2 arXiv:2508.18756 (Aug 2025) [V]
- Ultra-sparse memory layer (Tucker-decomposed product-key retrieval etc.), up to 20M slots; vs MoE at equal params/activations: better quality, 2–6× faster inference, up to 83% lower inference cost (vendor blog).
- V2: memory layers in every block, FFN-based values; 120B total / 2.5B active; parity with MoE; +1.6 long-context memorization, +6.2 multi-round memorization, +7.9 in-context learning; "activation density has greater impact on performance than total sparse parameter count." No editing/unlearning evaluation.

### 2.4 Memory Layers at Scale — Berges et al., Meta FAIR. arXiv:2412.09764 (Dec 2024, ICLR 2025) [V]
- **Mechanism:** Product-key memory replacing ~3 FFNs, memory parameters **shared across memory layers**, SiLU-gated output, custom CUDA kernels (memory-bandwidth bound: 3TB/s vs PyTorch 400GB/s).
- **Results:** 1.3B base + 64M keys (≈128B memory params): NQ 20.78 vs 7.76 dense (Llama2-7B 25.10); TriviaQA F1 62.14 vs 32.64 (Llama2-7B 64.00) [V]. 8B base + memory, 1T tokens, approaches Llama3.1-8B (15T tokens). Gains "especially pronounced for factual tasks"; degrades if >3 FFNs replaced (dense compute still needed).
- **Where knowledge lives:** Value slots (factual associations disproportionately).
- **Updates:** Gradient training only; no separable write op. But sparse by design → basis for 2.8.

### 2.5 Mixture of Lookup Experts (MoLE) — arXiv:2503.15798 (Mar 2025, ICML 2025) [V]
- Experts take the embedding-layer output (token-id only), so they are re-parameterized post-training into per-token lookup tables offloaded to storage. Systems precursor to Engram-style static lookup. Gemma 3n's **Per-Layer Embeddings** (Google, Jun 2025) [V, dev blog] likewise put ~3B/4B of 5B/8B params into per-token, per-layer embedding tables streamed from CPU — a deployed "lookup memory," though not framed as knowledge.

### 2.6 DeepSeek Engram — "Conditional Memory via Scalable Lookup: A New Axis of Sparsity for LLMs." Cheng et al. (DeepSeek + PKU), arXiv:2601.07372 (v1 12 Jan 2026; v2 12 Jul 2026) [V]
- **Mechanism:** Modernized **hashed N-gram embeddings**: suffix 2–3-grams of the (compressed) token context hashed via multi-head multiplicative-XOR hashing into huge embedding tables (O(1) lookup); retrieved vector gated by a context-aware scalar σ(RMSNorm(h)·RMSNorm(k)/√d) and added to the residual stream. Production config: modules at layers 2 and 15. Deterministic addressing → prefetch from host DRAM; a 100B-param table offloaded with ~2.8% throughput penalty on an 8B backbone [V].
- **Sparsity allocation law:** U-shaped; best with ~20–25% of the sparse budget in Engram, rest in MoE (val loss 1.7248→1.7109) [V].
- **Results:** Engram-27B vs iso-param/iso-FLOP MoE: MMLU +3.4, CMMLU +4.0, BBH +5.0, ARC-C +3.7, HumanEval +3.0, MATH +2.4; multi-query NIAH 84.2→97.0 [V]. Interpretation: offloading static local pattern reconstruction frees early layers/attention.
- **Separability evidence (most relevant finding of 2026):** suppressing Engram at inference → factual benchmarks retain only **29–44%** (TriviaQA 29%) while reading comprehension retains **81–93%** (C3 93%) [V]. Knowledge demonstrably concentrates in the lookup tables; comprehension/reasoning in the backbone.
- **Updates:** **None discussed** — tables are learned in pretraining; keys are surface n-grams, so a fact isn't a single addressable row (spread over many n-gram rows and hash collisions). Follow-ups: TF-Engram (arXiv:2607.07388, Jul 2026) [V] builds phrase memory tables offline from corpora and injects via hidden-state injection with no training (Qwen3-0.6B avg 57.6→59.4 — small gain, small model); "User as Engram" (arXiv:2606.19172, Jun 2026, single author) [V] stores per-user facts as local edits to a hash-keyed table, claims 5.6× indirect-reasoning accuracy vs baselines and lossless multi-user composition vs per-user LoRA — **unrefereed, treat cautiously**; hot-tier/CXL systems papers (2601.16531, 2603.10087).
- **Adoption caveat [V-ish]:** Despite heavy speculation, the DeepSeek-V4 release (HF blog 24 Apr 2026; tech report ~May 2026) describes CSA/HCA hybrid attention, DeepSeekMoE and mHC — **no Engram**; secondary sources state explicitly "Engram is absent from V4." So Engram is not yet validated at frontier scale.

### 2.7 Apple — "Pretraining with Hierarchical Memories: Separating Long-Tail and Common Knowledge." Pouransari et al., arXiv:2510.02375 (Oct 2025; ICLR 2026) [V]
- **Mechanism:** Small "anchor" LM + large bank of FFN memory blocks. Pretraining docs are hierarchically clustered (Sentence-BERT embeddings, 4-level k-means tree, branching 16); at train and inference time the context is routed down the tree and the fetched memory parameters are **concatenated into the SwiGLU FFN inner dimension**. Works added during pretraining or post-hoc; banks scaled to >21B.
- **Results:** 160M anchor + 18M fetched from a 4.6B bank ≈ a regular model with >2× params; +4.1 on specific-knowledge tasks (410M, ~10% extra memory); atomic-number long-tail accuracy 17%→83% for a 1.4B model [V].
- **Updates:** Memory banks can be **blocked, deleted, or replaced without retraining the anchor**; new memories can be trained on new data with the anchor frozen [V]. Blocking the relevant memory: atomic-number accuracy 70%→20% [V] (i.e., leakage remains in the anchor).
- **Limits:** Granularity is *cluster* (document-topic), not fact; writes still require gradient training of the memory block; routing via an external sentence encoder; scaling laws for memories "unexplored."

### 2.8 Meta — "Continual Learning via Sparse Memory Finetuning." Lin et al., arXiv:2510.15103 (Oct 2025) [V]
- **Mechanism:** Starting from a memory-layer model (1.3B; middle FFN layer 12/22 replaced by memory with 1M slots, top-k=32, 4 heads), finetune **only the memory slots whose access frequency on the new data is high relative to a background pretraining corpus (TF-IDF ranking)**; t=500 slots for fact learning, 10k for document QA.
- **Results:** Learning 1K TriviaQA facts: held-out NaturalQuestions F1 drops **89% (full FT), 71% (LoRA), 11% (sparse memory FT)** at matched acquisition [V]. Also SimpleQA-grounded document QA with Active Reading augmentations.
- **Limits (authors):** only fact learning; "RAG is a natural present-day solution" for these; unclear for reasoning/coding; not tested at larger scale; optimizer-sensitive. Follow-up: "Improving Sparse Memory Finetuning" arXiv:2604.05248 (Apr 2026) [V] — KL-divergence-based slot selection for "surprising" tokens; retrofits Qwen-2.5-0.5B.
- **Relevance:** Best current evidence that memory layers give *low-interference* writes, but writes are still gradient steps, deletion is not addressed, and the core is not knowledge-light.

### 2.9 ExplicitLM — Yu et al., arXiv:2511.01581 (Nov 2025) [V abstract only]
- Million-scale memory bank of **human-readable token sequences**, retrieved by product-key coarse filter + Gumbel-softmax fine match, end-to-end differentiable; 20% frozen explicit facts / 80% learnable implicit entries updated by EMA. Claims up to 43.67% gain on knowledge-intensive tasks, 3.62× in low-data. **Unrefereed; small-scale; claims not independently checked [?].** Same group: TokenMem (2.13).

### 2.10 Other 2025–2026 parametric-memory items
- **MLP Memory** — Wei et al., arXiv:2508.01832 (Aug 2025; rev. Feb 2026) [V]: an MLP pretrained to imitate a kNN retriever's distribution over the whole pretraining corpus; interpolated with the decoder's output. +12.3% relative QA across five benchmarks, up to 10 pts on HaluEval, 2.5× faster than RAG. Output-distribution interface; updates need retraining.
- **Lamini Memory Tuning / MoME** — arXiv:2406.17642 (Jun 2024) [V]: "millions of memory experts" to drive loss to ~0 on key facts. Thin technical detail; commercial; **[?]**.
- **MeMo** (correlation-matrix memories; workshop, arXiv:2606.24040, Jun 2026) [V abstract]: version-aware replace/obsolete/rollback operations over explicit CMM memory — conceptually on-target (transactional edits) but workshop-level, no scale evidence.
- **MUNKEY** — arXiv:2603.15033 (Mar 2026) [V]: "unlearning by design" via key deletion in a memory-augmented transformer — but evaluated on image classification, not LMs.

---

## 3. Explicit / plug-in knowledge (store in a format the frozen or co-trained model consumes directly)

### 3.1 Memory³ (Memory-cubed) — Yang et al., IAAR Shanghai. arXiv:2407.01178 (Jul 2024) [V]
- **Mechanism:** "Explicit memory" = **sparsified attention KV** of reference chunks, pre-computed by the model itself: first half of layers are memory layers; each 128-token reference keeps 8 tokens per head (selected by attention received), vector-quantized; 5 memories (640 ref tokens) retrieved per 64-token chunk via BGE-M3. Bank of 1.1×10^8 references: 7.17 PB raw → 45.9 TB sparsified → 4.02 TB compressed [V].
- **Theory:** "memory circuitry" argument that knowledge can be externalized so the model only needs parameters for "abstract knowledge." Two-stage pretraining (warm-up without memory, then with).
- **Results:** 2.4B model beats larger LLMs and RAG baselines and decodes faster than RAG (figure-level claim) [V]; authors concede results "may not be comparable to SOTA."
- **Updates:** Not experimentally addressed. *In principle* a new fact = run the model on new text and store its sparse KV (no gradient). **This is architecturally the closest precedent to a Hemispheres "compiled write op"**, but no editing/deletion or knowledge-lightness evaluation exists.

### 3.2 KBLaM — Wang et al., Microsoft Research. arXiv:2410.10450 (Oct 2024; ICLR 2025) [V]
- **Mechanism:** Each KB triple (name, property, value) → key from sentence-encoder(name+property), value from encoder(value), via learned linear adapters into each layer's KV space ("knowledge tokens"). **Rectangular attention:** prompt tokens attend to all knowledge tokens; knowledge tokens don't attend to each other → linear cost in KB size. Adapters instruction-tuned on **synthetic** data; base LLM (Llama-3-8B) frozen.
- **Updates:** Add/remove/modify a triple = add/remove/replace one vector pair; no retraining [V].
- **Results:** >10K triples (~200K text tokens) into an 8B model with 8K context on one A100; 512 triples ≈ memory of RAG with 5; above ~200 triples refuses better than in-context baseline [V blog].
- **Limits:** README: when KB differs from training distribution, answers are incomplete, reworded or "entirely incorrect"; "research project, not production"; trained on factual Q&A only, "further research needed … complex reasoning" [V]. Fixed-length vector per triple loses information. Follow-up **AtlasKV** (arXiv:2510.17934, ICLR 2026) [V]: KG2KV + hierarchical KV pruning to ~1B-triple KGs in <20GB VRAM, "no retraining when adapting to new knowledge."
- **Relevance:** Exactly the right *interface* idea (fact-granular, gradient-free write into attention), but retrofitted onto a knowledge-heavy base (prior dominance unaddressed) and poor OOD generalization of the encoder→KV mapping.

### 3.3 Cartridges — Eyuboglu et al., Stanford Hazy Research. arXiv:2506.06266 (Jun 2025) [V]
- **Mechanism:** Per-corpus **trained KV cache** (prefix-tuning-like) optimized offline by "self-study": generate synthetic conversations about the corpus, train the cartridge by context distillation (naive next-token training on the corpus is not competitive).
- **Results:** Matches ICL quality with 38.6× less memory, 26.4× throughput; effective context 128k→484k on MTOB; cartridges can be **composed at inference without retraining** [V].
- **Updates:** New/changed corpus → re-run self-study (GPU-hours-scale, amortized). Base frozen.
- **Limits:** Per-corpus training cost; edits to one fact require retraining that cartridge; composition quality not guaranteed at many cartridges.

### 3.4 Memory Decoder — Cao et al., SJTU/Shanghai AI Lab. arXiv:2508.09874 (Aug 2025; NeurIPS 2025) [V]
- **Mechanism:** Small transformer decoder (e.g., 0.5B) pretrained to imitate kNN-LM distributions over a domain datastore (KL + CE); at inference, output distributions interpolated with any same-tokenizer LLM: p = α·p_mem + (1−α)·p_LLM.
- **Results:** Avg −6.17 perplexity across biomed/finance/law on Qwen/Llama 0.5B–72B; 1.28× overhead vs 2.17× kNN-LM, 1.51× in-context RAG; cross-tokenizer transfer at ~10% of training budget [V].
- **Limits:** Knowledge updates require retraining the decoder; output-level interface inherits kNN-LM's "shallow reasoning" concern.

### 3.5 Knowledge Card — Feng et al. arXiv:2305.09955 (May 2023; ICLR 2024 oral) [V]
- Domain-specialized small LMs ("cards") generate background text that is filtered (relevance, brevity, factuality) and fed to a black-box LLM. Knowledge in small LMs' weights → text interface. Update by training/adding a card. Modular and community-contributable, but cards hallucinate and the base LLM's priors remain.

### 3.6 K-Adapter — Wang et al. arXiv:2002.01808 (Feb 2020) [M]
- Frozen RoBERTa + separately trained factual (Wikidata-aligned) and linguistic adapters as side networks; outputs concatenated. Early "keep base frozen, knowledge in plug-ins" example; encoder-era.

### 3.7 LoRA-as-knowledge-module
- **Knowledge Modules with Deep Context Distillation** — Caccia et al. (Microsoft Research Montréal), arXiv:2503.08727 (Mar 2025) [V]: document-level LoRA KMs; next-token prediction on the document is a poor objective; train KMs to match hidden states and logits of a teacher that has the document in context (DCD). Beats alternatives on QuALITY/NarrativeQA with Phi-3 3B and Llama-3.1 8B, synergistic with RAG.
- **Understanding LoRA as Knowledge Memory** — Back et al., arXiv:2603.01097 (ICML 2026) [V abstract]: empirical capacity, internalization, multi-module composition study; positions LoRA as a third memory axis beside RAG/ICL. (Specific capacity numbers not retrieved **[?]**.)
- **Doc-to-LoRA (D2L)**, Charakorn, Cetin, Uesaka, Lange (Sakana AI), arXiv:2602.15902 (13 Feb 2026) [V, full HTML read]. Text-to-LoRA is arXiv:2506.06105 (2025) [M].
  - *Mechanism:* a 309M-param Perceiver hypernetwork (8 cross-attention blocks) reads the base model's activations on a document and emits a rank-8 LoRA for the **MLP down-projection of every layer**. It is meta-trained with query-independent context distillation, minimizing KL(p(y|x,c) ‖ p_{θ+H(c)}(y|x)), on about 3.2M FineWeb-Edu and QA contexts. Long docs are chunked and the per-chunk LoRAs are concatenated along the rank dimension.
  - *Results:* primary target Gemma-2-2B-IT (8K context); also Mistral-7B, Qwen3-4B. Internalization takes ~0.21 s versus ~40 s for oracle context distillation (CD). Relative to ICL, normalized QA scores are SQuAD 0.81, DROP 0.66, ROPES 0.91, and it beats CD at <2 GB versus >40 GB VRAM. On 2WikiMultihopQA it scores 0.857 versus 0.901 for oracle CD. NIAH is near-perfect to 40K tokens (5× the training max), using <50–100 MB versus 11–12 GB for ICL at long length.
  - *Limits (stated):* a gap to ICL remains. The hypernetwork must be retrained for each target LLM (meta-training took 5 days on 8 H200s). Multi-document composition is not demonstrated. Internalized knowledge hurts unrelated queries (0.096 vs 0.201 base). Only LoRA parameterization is studied.
  - *Difference from the others:* D2L is a gradient-free, amortized write into weights (one forward pass) for a frozen, knowledge-heavy base. It is per-context or session memory rather than a persistent editable store: it has no addressing, no delete op beyond dropping the adapter, and no knowledge-light core.
- **Hypernetwork knowledge-injection scaling laws**, Dhankhar, Baha, Saparov, arXiv:2607.19604 (21 Jul 2026) [V, full HTML read].
  - *Mechanism:* a hypernetwork maps a verbalized triple (s, r, o), or a noisy context containing it, to LoRA factors for the later ⌊L/2⌋ layers of a frozen target. The design deliberately decouples injection capacity (the hypernetwork) from target capability.
  - *Data and scaling:* the MegaWikiQA dataset has tens of millions of single- and multi-hop (≤4-hop) QA across 39 domains. Loss follows power laws along hypernetwork width (exponent ≈ −0.10 ID), depth (≈ −0.09) and **target-model size (≈ −0.23 ID, −0.18 OOD)**, so scaling the target helps most.
  - *Comparison with finetuning:* versus LoRA-FT and full FT, the ID exponents are slightly worse (−0.226 vs −0.250/−0.249) but the OOD exponents are steeper (e.g., rephrased −0.107 vs LoRA −0.083; MCQ −0.171 vs full-FT −0.101). The OOD advantage widens with target size.
  - *Limits:* the hypernetwork needed about 2.5B params for a 1.5B target. Rephrasing robustness scales poorly (−0.036 to −0.042), multi-hop is only shallow, and targets go up to 14B.
  - *Difference from D2L:* this work injects at training time and studies scaling. D2L does instant test-time internalization of arbitrary documents.
  - *Hemispheres relevance:* this is the only quantitative scaling evidence for a learned write operator. Its catch is that the writer must be about as large as the model it writes into.
- Related: Generative Adapter (arXiv:2411.05877) [M], DyPRAG (1.9), DMoE (1.10).
- **"Can a Language Model Learn Facts Continually in Its Weights?"** by Charles O'Neill (Baseten), arXiv:2607.11020 (13 Jul 2026) [V, full HTML read]. This is the "Baseten cortex" paper. **"cortex" is only the name of the code repository (footnote); the paper proposes no architecture.**
  - *Setup:* 247 invented facts written into Qwen3-4B (checked on 8B) via LoRA r16/r4 and full FT, with SFT on bare statements versus diverse "study" data, offline/online context distillation, and batch versus sequential merging, over 20–100 sequential writes.
  - *Results:* diverse restatements shrink the recitation-to-use (entailment) gap from 27.4 to 5.4 points. After 20 sequential writes, bare-statement facts retain 1% accuracy versus 46% for study-trained facts, and 70% of wrong answers contain the most recently written fact. Forgotten facts keep 57–67% of their log-prob lift, so they are "behaviorally inaccessible" but still stored. Supplying them in context recovers 77–80%. Later writes interfere with earlier ones whatever the storage method, and weight-written facts become "question-keyed."
  - *Conclusion:* "when facts must be composed or survive later writes, the reliable channel is context rather than the weights."
  - *Difference from the others:* this is a negative, diagnostic result about dense or LoRA weight-writes into a monolithic model, not a split architecture. It strongly supports designs where updated knowledge enters through the context or attention channel and the core is not rewritten.
- **Cautions:** Tang et al. 2510.12668 (parametric injection weak on precise facts); Gekhman et al. arXiv:2405.05904 [M] (fine-tuning on new facts is slow and increases hallucination); Ovadia et al. arXiv:2312.05934 [M] (RAG beats unsupervised FT for knowledge injection). LoRA composition across many modules degrades (interference) [M].

### 3.8 MemoryLLM — Wang et al. arXiv:2402.04624 (Feb 2024; ICML 2024) [M]; M+ — arXiv:2502.00592 (Feb 2025) [V]
- **Mechanism:** Llama-2-7B + ~1B-parameter latent memory pool (memory tokens in every layer). "Self-update": new text is processed and a random subset of memory tokens is overwritten → no gradient step needed at deployment. M+ adds a long-term memory of dropped tokens and a co-trained retriever over hidden states, extending retention from <20k to >160k tokens [V].
- **Limits:** Designed for context/knowledge *absorption* rather than addressable fact storage; overwrites are random (no targeted delete); forgetting is gradual and uncontrolled. Base carries its own parametric knowledge.

### 3.9 Larimar — Das et al., IBM. arXiv:2403.11901 (Mar 2024; ICML 2024) [V abstract]
- Kanerva-machine-style distributed episodic memory matrix: encoder writes fact encodings into memory via closed-form (least-squares) update; decoder LLM conditioned on read-out. One-shot edits without gradient training, 8–10× faster than baseline editors, supports selective forgetting and leakage prevention [V]. Tested at GPT-2/GPT-J scale on CounterFact/ZsRE-style edit benchmarks [M]. Closest to "gradient-free write + delete," but small scale and single-fact edit benchmarks.

### 3.10 LM2 — Kang et al., Convergence Labs. arXiv:2502.06049 (Feb 2025) [V]
- Decoder + auxiliary memory bank with cross-attention and gated updates; +37.1% over RMT, +86.3% over Llama-3.2 on BABILong. **This is a long-context working memory, not a swappable knowledge store** — include as a design point, not as a Hemispheres competitor.

### 3.11 Titans — Behrouz et al., Google. arXiv:2501.00663 (Dec 31 2024) [V]; ATLAS — arXiv:2505.23735 (May 2025) [V]; Nested Learning / HOPE — arXiv:2512.24695 (NeurIPS 2025) [V]
- **Mechanism:** Neural long-term memory (an MLP) whose weights are updated *during the forward pass* by gradient of an associative-memory loss with "surprise," momentum and forgetting; combined with attention as short-term memory. Titans scales to >2M-token contexts; ATLAS optimizes memory over a window (Omega rule), +80% accuracy at 10M context on BABILong over Titans [V]. HOPE: self-modifying module + "continuum memory system" with multi-rate updates, reports better knowledge incorporation / continual learning.
- **Relevance:** Test-time memorization of the *context*; persistent knowledge store semantics (addressable, deletable) absent. Independent reproductions of Titans' headline gains were not verified by me **[?]**.

### 3.12 LMLM — "Pre-training Limited Memory Language Models with Internal and External Knowledge." Zhao et al. (Cornell), arXiv:2505.15962 (May 2025; v2 Oct 2025) [V]
- **Mechanism:** During pretraining, text is annotated with lookup calls `<|db_start|> Entity <|sep|> Relation <|db_value|> Value <|db_end|>`; the retrieved **values are masked out of the training loss**, so the model learns to emit the query instead of memorizing the value. DB: 54.6M Wikipedia triples (9.5M entities); fuzzy match via MiniLM embeddings, cos≥0.6.
- **Results:** FactScore: LMLM-LLaMA2-382M 31.9% vs 14.0% standard (LLaMA2-7B 34.0%); T-REx EM 58.1 vs 52.0 (7B 60.5) [V]. **Unlearning on TOFU by deleting DB entries: ideal forgetting (p>0.05) with no utility loss; beats NPO** [V].
- **Limits:** entity-level triples only; ≤382M params; no multi-hop; fuzzy-match errors; "some factual knowledge remains unremoved from model parameters."
- **Relevance:** The most direct precedent for Hemispheres' *training objective* (a core trained not to memorize). Gap: scale, non-triple knowledge, multi-hop, and a learned (not symbolic) store.

### 3.12b Co-LMLM — "Continuous-Query Limited Memory Language Models." Feldman, Zhao, Godey, Go, Hua, Weinberger, Sun, Artzi (Cornell). arXiv:2607.07707 (8 Jul 2026) [V, full HTML read]
- **Mechanism:** LMLM with a learned retrieval interface.
  - *Query:* when the model emits `<FACT>`, its **last-layer hidden state at that position is the query vector**. There is no textual query and no triple.
  - *Store:* a KB of (dense key → **free-text fact span**) pairs. The top-1 span is spliced into the context and closed with `</FACT>`, then decoding continues.
  - *Loss:* the next-token loss is **masked on the retrieved span** (inclusive of `</FACT>`, excluding `<FACT>`), so facts are not memorized. A bidirectional InfoNCE loss (λ=0.25) aligns query states with keys.
  - *Annotation:* Gemini 3.1 Pro annotated about 60K seed documents, then ModernBERT (span detection) and Qwen2.5-1.5B (question generation) annotated the full corpus.
  - *KB size:* about 240M items from Wikipedia and **2.2B items** with FineWeb-Edu, versus about 145M for the relational LMLM ("Rel-LMLM").
- **Results:** SmolLM2-style 135M/360M models, trained on about 3B Wikipedia tokens or 90B FineWeb-Edu tokens.
  - Co-LMLM-360M perplexity is 10.5 versus 14.0 for the standard model, and lower than models trained on 40× more data.
  - SimpleQA is 21.2 (21.7 with FineWeb), which the authors put "in line with gpt-4o-mini and higher than Claude Sonnet 4.5" per public leaderboards. Treat that comparison with caution: it pits a retrieval-equipped model against closed-book frontier models.
  - FactScore 34.2 (+24.2 over standard; Rel-LMLM-360M 22.9), TriviaQA 31.4/36.9, PopQA 46.5/50.6 (+30.3).
  - Retrieval takes about 2.2 ms per call versus about 28 ms for a text-query "LMLM-Asker" (13×).
  - TOFU unlearning by deleting KB entries gives forget quality p>0.05 with utility preserved. NPO, a gradient-based unlearning method, loses utility.
- **Limits (stated):** modest scale (≤360M), Wikipedia-style knowledge, one-time index construction cost, and an open problem of "how to adapt such an index once a model is fine-tuned" (post-training and continual learning shift the query space).
- **How LMLM / Co-LMLM differ from the rest:** they are the **only systems that train the core to *not* memorize the externalized facts** (loss masking). Their store is text read through the context channel, so the core reasons over retrieved facts in-context, and editing or deletion is a store operation. Compared with DMoE, the base is knowledge-light by construction and consults the store on its own initiative (the `<FACT>` token), not only when its entropy is high. Compared with Engram and Apple's memories, the store is addressable at fact granularity with free-text values and has gradient-free writes. The weaknesses Hemispheres could target:
  - *Scale:* ≤382M (LMLM) and ≤360M (Co-LMLM).
  - *Coupled key space:* keys are the model's own hidden states, so the index must be rebuilt or re-aligned when the core is fine-tuned.
  - *Deletion leakage:* measured only on TOFU's synthetic authors.
  - *Reasoning:* no evaluation of whether the knowledge-light core reasons as well as a matched standard model, or of multi-hop edit propagation.

### 3.13 Conflict-handling components (relevant to "change a fact")
- **TokenMem** — Yu et al., arXiv:2607.22625 (Jun/Jul 2026) [V]: parallel gated knowledge channel (~3–7M params) outside the residual stream for frozen LLMs, two-phase curriculum incl. counterfactual adherence; Knowledge Compliance ~69–70% vs 20–52% for vanilla RAG across Qwen3/Llama-3.1-8B/OLMo-3-7B; without phase 2 compliance collapses to ~0 [V].
- **"Quantifying Prior Dominance in RAG Systems"** arXiv:2606.23695 (Apr 2026, single author) [V]: a commercial API overrode explicit evidence in nearly half of adversarial conflicts.
- **SCR** — He et al., arXiv:2503.05212 (Mar 2025) [V]: evaluates 10 model-editing methods, finds all have significant shortcomings; selective in-context reasoning works better.
- **ERASE** — Li et al., arXiv:2406.11830 (Jun 2024) [V]: when adding a doc, delete/rewrite stale KB entries; +6–13% accuracy over RAG on news/conversation streams. (Store-maintenance matters, not just read path.)
- Knowledge-editing background [M]: ROME (2202.05262), MEMIT (2210.07229) collapse under many sequential edits (Gupta et al. 2401.07453); MQuAKE (2305.14795) and RippleEdits (2307.12976): edited facts rarely propagate to multi-hop/entailed facts; WISE (2405.14768) and GRACE (2211.11031) use side memories/codebooks.

---

## 4. "Cognitive core" and explicit decoupling of knowledge from reasoning

- **Karpathy, "cognitive core"** (X post, ~27 Jun 2025, status 1938626382248149433) [V]: race for "a few billion param model that maximally sacrifices encyclopedic knowledge for capability," always-on, tool-using, reasoning-with-a-dial, on-device finetuning. Later interviews (2025) argue pretraining knowledge "holds back" intelligence and a ~1B core may suffice [secondary sources, **[?]** on exact quotes]. A vision statement, not an artifact.
- **Han, Pari, Gershman, Agrawal, "Position: General Intelligence Requires Reward-based Pretraining"** arXiv:2502.19402 (ICML 2025) [V]: hypothesizes coupling of reasoning and knowledge limits transfer; proposes RL-based pretraining on synthetic tasks, and decoupling knowledge into an external memory bank with a small context window. Position paper.
- **Guo & Chen, "Decoupling Knowledge and Reasoning in Transformers: A Modular Architecture with Generalized Cross-Attention"** arXiv:2501.00823 (Jan 2025) [V]: shows the FFN is a special case of cross-attention to an implicit KB; proposes cross-attention to a shared KB. **Theoretical; no experiments.**
- **"Decoupling Knowledge and Reasoning in LLMs: an exploration using dual-system theory"** arXiv:2507.18178 (Jul 2025) [V]: an *analysis* (fast vs slow thinking attribution across 15 LLMs) finding knowledge retrieval concentrated in lower layers, reasoning adjustments in higher layers. Not an architecture.
- **SynthWorlds** — Gu et al., arXiv:2510.24427 (ICLR 2026) [V]: parallel real/synthetic-entity corpora; measures a "knowledge advantage gap"; knowledge-integration mechanisms reduce but do not close it. **Useful evaluation methodology for Hemispheres.**
- **Procedural Pretraining** — Jiang et al., arXiv:2601.21725 (ICML 2026) [V]: 0.1–0.3% algorithmic/formal-language data up front gives same performance with 55–86% of the data; argues for separating knowledge acquisition from reasoning. Evidence for a "reasoning-first" core curriculum.
- **Phi-4** tech report arXiv:2412.08905 [V]: small model with strong reasoning but explicitly "fundamentally limited by its size" on factual knowledge (hallucinated biographies; post-trained to decline SimpleQA). Empirical existence proof of a reasoning-heavy/knowledge-light small core — but without any knowledge store to compensate.
- **Pleias-RAG** (arXiv:2504.18225) [V title only]: small reasoners trained specifically for grounded, cited RAG.
- **Unable to find:** any 2025–2026 paper that trains, at ≥1B scale, a core whose factual knowledge is *systematically excluded* (as in LMLM) and pairs it with a learned, editable store, evaluated on add/change/remove with multi-hop propagation. A news item about a "first cognitive model" 4B on-device matching GPT-5.4 surfaced in search but the page was empty and I could not identify the model **[?]**.

---

## 5. Synthesis

### 5.1 Taxonomy of design choices

| Interface location | Knowledge format | Addressing | Examples | Write/update cost | Delete semantics |
|---|---|---|---|---|---|
| **Input/context** | Raw text | Dense/BM25 retriever | REALM, RAG, Atlas, Self-RAG, Knowledge Card (text generated by plug-in LMs), ERASE | ~0 (index insert); re-embed on retriever change | Easy in store; **core still knows** (prior dominance) |
| Input/context via tool call | Structured triples (LMLM) / free-text spans with learned dense keys (Co-LMLM) | Model-emitted text query + fuzzy match (LMLM); `<FACT>` hidden state as query vector (Co-LMLM) | LMLM, Co-LMLM | ~0 (DB row; Co-LMLM needs key encoding in core's space) | Clean in store (TOFU p>0.05); residual param leakage small but nonzero |
| Final-layer FFN, entropy-gated | Per-passage LoRA weights | BM25 over passage surrogates, fired when next-token entropy > τ | DMoE | ~10 s LoRA training per passage | Delete adapter + index entry; base's own knowledge untouched |
| **Attention (KV)** | Encoded KV (per chunk) | Retriever or all-to-all | RETRO (chunk cross-attn), Memorizing Transformers, Memory³ (sparse self-KV), KBLaM/AtlasKV (encoder→KV), MemoryLLM/M+ (latent tokens), LM2 | Forward pass only (Memory³, KBLaM, MemoryLLM); or trained (Cartridges: self-study per corpus) | Per-entry removal possible (KBLaM, Memory³); random overwrite (MemoryLLM) |
| **FFN / residual** | Weights: value slots, hashed n-gram embeddings, memory FFN blocks, LoRAs | Product keys, hashing, clustering tree, retrieval of LoRAs, hypernetworks | PKM, PEER, UltraMem, Memory Layers, MoLE, Gemma 3n PLE, Engram, Apple hierarchical memories, K-Adapter, KMs/PRAG/DyPRAG/Doc-to-LoRA, TokenMem, ExplicitLM | Gradient training (pretrain/SMF/LoRA) or hypernetwork forward pass | Block/replace block (Apple); switch module off (Engram); no per-fact delete |
| **Output distribution** | Datastore / small LM | kNN or learned imitator | kNN-LM, Memory Decoder, MLP Memory | Datastore insert (kNN-LM) or retrain (MemDec, MLP Mem) | kNN-LM easy; others need retrain |
| **Fast weights at test time** | Memory MLP / matrix | Associative recall | Titans, ATLAS, HOPE, Larimar | Online gradient / closed-form write | Larimar supports selective forgetting; Titans uncontrolled decay |

Orthogonal axes worth naming explicitly in the Hemispheres design doc:
1. **Was the core trained to be knowledge-light?** (LMLM loss masking; Memory³ / Apple / Engram / retrieval-pretraining partially induce it — cf. "More Room for Language") vs **retrofitted to a knowledge-heavy base** (RAG, KBLaM, Cartridges, Memory Decoder, TokenMem). Retrofitted systems fight prior dominance; co-trained ones are small-scale.
2. **Granularity of addressing:** fact (KBLaM, LMLM, Larimar) vs chunk (RETRO, Memory³, RAG) vs topic cluster (Apple) vs surface n-gram (Engram) vs slot (memory layers, distributed across many facts).
3. **Write operator:** symbolic insert (text/triple) → forward-pass compile (Memory³, KBLaM, MemoryLLM, DyPRAG, Doc-to-LoRA) → sparse gradient (SMF) → dense gradient (LoRA/full FT).
4. **Read locus & depth:** early-layer lookups (Engram at layers 2/15; Memory³ first half) vs all layers (KBLaM) vs output only (kNN-LM). kNN-LM's oracle failure suggests knowledge must enter *before* the reasoning layers to be reasoned over.

### 5.2 What the evidence says (claims Hemispheres can lean on)
1. Knowledge *can* be concentrated in a separable component at scale, with comprehension largely intact when it is removed (Engram: 29–44% vs 81–93% retained).
2. Small cores + big memories ≈ 2× larger dense models on knowledge (Apple, Memory Layers: 1.3B+memory ≈ approaching 7B on NQ/TQA; LMLM-382M ≈ approaching 7B on FactScore).
3. Swapping a text store updates answers (RAG 70%/68% vs 12%/4%; Atlas 2017→2020 index 1.5%→53.1%).
4. Training with an external store reduces what the weights memorize (Samuel et al.; LMLM loss masking).
5. Sparse slot updates dramatically reduce forgetting vs LoRA/FT (SMF: 11% vs 71%/89%).

### 5.3 Open problems (where every existing system falls short)
1. **Residual leakage / incomplete externalization.** No system shows the core truly *lacks* removed facts: Apple 70%→20%, Engram TriviaQA 29% retained, LMLM concedes residual memorization. For "remove a fact" (privacy, retraction), leakage is a correctness failure, not a nuisance. No standard metric for "core knowledge leakage."
2. **Prior dominance on change.** When the core is knowledge-heavy, changed facts are overridden (vanilla RAG 20–52% compliance on counterfactuals; commercial API ~half of adversarial conflicts). Fixes so far are bolt-on (TokenMem curriculum, SCR).
3. **No cheap, fact-granular write op for parametric memories.** Memory layers/Engram/Apple memories are written by gradient descent; SMF is the best low-interference option but is still training and untested beyond fact QA. Hash/n-gram keys (Engram) don't align with facts, so editing a fact touches many rows and collides with others.
4. **Edit propagation / multi-hop.** Parametric edits don't propagate (MQuAKE, RippleEdits); kNN-LM fails multi-fact reasoning even with oracle retrieval; LMLM does not handle multi-hop; KBLaM trained only on single-fact QA. Context-based stores propagate better because the core reasons over text — which argues for the store feeding *attention/context*, not just FFN or output.
5. **Retrofitted encoders generalize poorly.** KBLaM's encoder→KV adapters fail OOD; Memory Decoder needs retraining per domain; Cartridges need per-corpus self-study.
6. **Non-entity knowledge.** Almost all evaluations are entity/attribute facts. Where do procedural, commonsense, linguistic, and "how-to" knowledge live, and should they be externalized? Engram's gains on BBH/MATH suggest static pattern memory helps reasoning too — so the knowledge/reasoning boundary is not a clean line.
7. **Scaling laws for the split.** Engram's U-curve (≈20–25% of sparse budget to memory) and Singh et al.'s retrieval-token rule are the only quantitative guides; there is no law for "minimum core size at fixed reasoning quality when knowledge is fully externalized."
8. **Frontier validation is absent.** Engram did not ship in DeepSeek-V4; Memory Layers/UltraMem ship as capacity mechanisms, not editable stores. Hardware (memory-bandwidth-bound lookups) is a recurring cost.
9. **Evaluation.** Need a benchmark covering add / change / delete × single-hop / multi-hop / entailed facts × leakage × reasoning retention (SynthWorlds' real-vs-synthetic-entity trick is a good building block; TOFU for deletion; MQuAKE/RippleEdits for propagation).

### 5.4 Where the genuine gap is (candidate novel contribution)
**Positioning vs the four nearest works:**
- *Co-LMLM* already has (A) and a text-valued, fact-addressable store with gradient-free insert/delete. Hemispheres must differ on at least one of these:
  - scale (≥1B, with a reasoning-retention comparison against a matched standard model);
  - a store whose keys are **decoupled from the core's hidden-state space**, so the core can be fine-tuned without re-indexing, which Co-LMLM names as an open problem;
  - reading knowledge into attention/KV rather than splicing text;
  - edit-propagation, counterfactual and leakage evaluation.
- *DMoE* is augmentation of a knowledge-heavy base: output-end, entropy-gated, gradient-written. Hemispheres should present it as a baseline showing why a split is needed, e.g. confident stale priors never trigger routing.
- *Doc-to-LoRA / hypernetwork scaling* is the best "learned writer" evidence, but the writer is about as large as the target and writes are per-context rather than persistently addressable.
- *O'Neill (Baseten)* is independent evidence that weight-writes into a monolithic model do not compose or persist, which motivates keeping updates out of core weights.

The intersection nobody occupies:
- **(A) A core trained to be knowledge-light by construction** — generalize LMLM/Co-LMLM's value-masking (Co-LMLM already extends it to free-text spans at 360M) (e.g., mask spans that the store can supply; or SynthWorlds-style entity randomization during pretraining so the core cannot profit from memorizing entity facts), at ≥1B scale, so the core *must* read from the store and therefore has low prior dominance and low deletion leakage.
- **(B) A store with a compiled, gradient-free, fact-granular write/delete op in the core's native representational space** — Memory³-style self-encoded sparse KV or KBLaM-style knowledge tokens, but *co-trained with the core* (fixing KBLaM's OOD problem) and entry-addressable (fixing Engram's n-gram/hash misalignment and Apple's cluster granularity).
- **(C) Read-in at attention level early enough that the core reasons over retrieved knowledge** (avoiding kNN-LM/Memory-Decoder shallow-reasoning limits), with a learned "consult-the-store" policy (Self-RAG-like) for when the core should defer.
- **(D) Evaluation that the field lacks:** leakage after deletion, counterfactual compliance after change, multi-hop propagation after add/change, and reasoning retention vs a matched dense model and vs RAG over a strong frozen LLM (the real incumbent).

Strongest baselines to beat:
- RAG over a strong frozen LLM, with SCR/TokenMem-style conflict handling
- Co-LMLM (the most direct competitor)
- DMoE and PRAG/DyPRAG
- Doc-to-LoRA
- Memory Layers + sparse memory finetuning
- Apple hierarchical memories
- KBLaM/AtlasKV

Risks to flag honestly: (i) Engram/Memory-Layer results suggest a lot of "knowledge" is useful for reasoning (BBH/MATH gains), so an overly knowledge-starved core may lose reasoning; (ii) retrieval-pretrained models lose some global-context skill (Samuel et al.); (iii) RAG with long contexts keeps improving, so the architectural win must be on update *correctness* (compliance, leakage, propagation) and cost, not just QA accuracy.

---

## 6. Verification flags (what I could not confirm)
- RETRO "gains shrink at scale": not found as a clean published result; RETRO itself claims constant gains to 7B. Supporting evidence is indirect (InstructRetro encoder ablation; Mallen et al.; Singh et al. 2026).
- DeepSeek-V4 omitting Engram: from HF blog (no mention) + secondary blogs stating "Engram is absent"; I did not read the V4 technical report itself.
- Engram v2 (Jul 2026) changes not examined.
- ExplicitLM, TF-Engram, User-as-Engram, MeMo, Quantifying Prior Dominance, TokenMem: recent, largely unrefereed/small-scale; numbers are author claims.
- PEER, PKM, kNN-LM, REALM, K-Adapter, MemoryLLM, Larimar numeric details from memory [M].
- Titans/ATLAS headline results: author-reported; independent replication not checked.
- Karpathy's secondary-source quotes (e.g., "~1B core in 20 years") not verified against primary audio.
- "Understanding LoRA as Knowledge Memory" (ICML 2026) capacity numbers not retrieved.
- An unidentified 2026 "first cognitive model" (4B, on-device) news item could not be verified.
- Added on request and read in full via arXiv HTML: DMoE (2606.14243), Co-LMLM (2607.07707), O'Neill/Baseten (2607.11020), Doc-to-LoRA (2602.15902), hypernetwork scaling laws (2607.19604). I relied on a summarizing fetch of each HTML, so table values should be spot-checked before they are quoted in a paper.
  - Co-LMLM's "above Claude Sonnet 4.5 on SimpleQA" compares a retrieval-equipped model with closed-book models and should not be read as a like-for-like result.
  - The "Baseten cortex" paper does not propose a "cortex" architecture. "cortex" is only its code repository name.
  - None of these five is peer-reviewed as far as I could see; I did not check venues.

## 7. Key references (arXiv IDs)
REALM 2002.08909 · RAG 2005.11401 · RETRO 2112.04426 · RETRO reproduction 2304.06762 · InstructRetro 2310.07713 · Surface-based retrieval 2305.16243 · More Room for Language 2404.10939 · MassiveDS 2407.12854 · To Memorize or to Retrieve 2604.00715 · Mallen/PopQA 2212.10511 · Atlas 2208.03299 · kNN-LM 1911.00172 · kNN-LM limits 2408.11815 · Memorizing Transformers 2203.08913 · Self-RAG 2310.11511 · PRAG 2501.15915 · DyPRAG 2503.23895 · Understanding P-RAG 2510.12668 · PKM 1907.05242 · PEER 2407.04153 · UltraMem 2411.12364 · UltraMemV2 2508.18756 · Memory Layers at Scale 2412.09764 · MoLE 2503.15798 · Engram 2601.07372 · TF-Engram 2607.07388 · User as Engram 2606.19172 · Apple hierarchical memories 2510.02375 · Sparse memory finetuning 2510.15103 · Improving SMF 2604.05248 · ExplicitLM 2511.01581 · MLP Memory 2508.01832 · Lamini MoME 2406.17642 · Memory³ 2407.01178 · KBLaM 2410.10450 · AtlasKV 2510.17934 · Cartridges 2506.06266 · Memory Decoder 2508.09874 · Knowledge Card 2305.09955 · K-Adapter 2002.01808 · Knowledge Modules (DCD) 2503.08727 · LoRA as Knowledge Memory 2603.01097 · Doc-to-LoRA 2602.15902 · MemoryLLM 2402.04624 · M+ 2502.00592 · Larimar 2403.11901 · LM2 2502.06049 · Titans 2501.00663 · ATLAS 2505.23735 · Nested Learning 2512.24695 · LMLM 2505.15962 · Co-LMLM 2607.07707 · DMoE 2606.14243 · O'Neill/Baseten continual fact writes 2607.11020 · Hypernetwork injection scaling laws 2607.19604 · TokenMem 2607.22625 · Prior Dominance 2606.23695 · SCR 2503.05212 · ERASE 2406.11830 · Reward-based pretraining position 2502.19402 · Generalized cross-attention decoupling 2501.00823 · Dual-system decoupling analysis 2507.18178 · SynthWorlds 2510.24427 · Procedural pretraining 2601.21725 · Physics of LMs 3.3 (2 bits/param) 2404.05405 · Phi-4 2412.08905 · MUNKEY 2603.15033 · MeMo versioning 2606.24040.
