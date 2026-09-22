# Hemispheres: evaluation methods and small-scale feasibility

Track: evaluation methodology, and what a solo researcher can build or train on an Apple M5 Max (64 GB) laptop plus rented GPUs.
Date of research: 2026-09-22. Sources: Hugging Face Hub (MCP), HF Papers, web search and fetch.
Legend: **[V]** = verified this session (HF Hub, repo page, or paper page). **[M]** = from my prior knowledge, not re-checked this session. **[E]** = my own estimate. Treat [M] and [E] items as things to confirm.

---

## 0. TL;DR

- **Evaluation.** Hemispheres claims you can update knowledge without retraining the reasoner. The metric that tests that claim is the **propagation gap**: after an update, how well the model answers multi-hop and ripple questions compared with an oracle that gets the new fact in context. Most editing benchmarks mainly measure efficacy (recall of the edited fact itself), and editing methods already do well on efficacy. Propagation is where they fail. Two examples: PropMEND reaches 22.4% on non-verbatim RippleEdits questions, against 12.7% for the next-best method [V]. A July 2026 study of writing facts into Qwen3 weights reports "bare-statement facts retain 1% accuracy" after 20 sequential writes [V].
- **Datasets to use.** For controlled work, use synthetic worlds: bioS (PhysicsLM4), Grokked-Transformer KGs, and fictional TOFU-style entities. Real-data benchmarks come second: MQuAKE-Remastered (not the original MQuAKE), RippleEdits, WikiBigEdit, and zsRE/CounterFact only as sanity checks.
- **Laptop.** The laptop is well suited to synthetic-world experiments: models of 10–150M params trained from scratch in MLX or PyTorch-MPS, runs of hours to about 2 days. It can also attach small adapter or memory modules to a frozen 0.6–4B model. It is **not** the right place to pretrain a 1B model from scratch.
- **Cloud.** H100s cost roughly $2–3/GPU-hr in 2026 [V]. A Chinchilla-optimal 124M model costs a few dollars. 350M costs about $25–30. 1B at 20B tokens costs about $200–250 per run [E]. Budget 5–10x that for ablations.
- **Closest prior work with usable code:** LMLM / Co-LMLM (Cornell; externalizes facts to a KB during pretraining, weights and data on HF) [V], KBLaM (MIT license, supports Llama-3.2-1B) [V], Meta memory layers (CC-BY-NC, CUDA) [V], DeepSeek Engram (demo code only) [V], Cartridges (Apache-2.0, Qwen3-4B, needs GPUs) [V], and EasyEdit for editing baselines (updated 2026-07-08 for Transformers 5.x) [V].

---

## 1. Benchmarks and datasets

### 1a. Classic single-fact knowledge editing

| Benchmark | What it measures | Size | HF id / source | Known issues |
|---|---|---|---|---|
| **CounterFact** (ROME, Meng et al. 2022) | Efficacy, paraphrase generalization, neighborhood specificity, generation fluency and consistency for counterfactual edits ("X's native language is French → Dutch") | 21,919 records [V: NeelNanda card] | `azhx/counterfact`, `NeelNanda/counterfact-tracing` (community mirrors) [V]; original at rome.baulab.info [M] | Counterfactual edits fight the model's strong priors. Metrics are probability comparisons (P(new) > P(old)), which are easy to satisfy. Neighborhood prompts are shallow. CounterFact+ (Hoelscher-Obermaier 2023) adds harder specificity tests [M]. No multi-hop. |
| **zsRE** (Levy 2017; MEND/ROME split) | Edit QA success, paraphrase rephrase, locality (NQ questions) | ~10k–20k eval edits depending on split [M]; `gjyotin305/full_zsre` has 100k–1M rows [V] | `wangzn2001/zsre`, `McGill-NLP/zsre_qa`, KnowEdit bundle [V] | Measured token-by-token with teacher forcing. **"The Mirage of Model Editing" (arXiv 2502.11177) [V]** shows teacher-forced evaluation greatly overstates editing success compared with real autoregressive QA, and introduces QAEdit. Use generation-based exact match only. |
| **KnowEdit** (zjunlp) | Bundle: WikiBio, ZsRE, WikiData-Counterfact, WikiData-Recent, ConvSent, Sanitation | varies | `zjunlp/KnowEdit` (MIT) [V] | Inherits the issues of its components. Convenient with EasyEdit. |
| **WikiBigEdit** (Thede et al., ICML 2025, arXiv 2503.05683) | **Lifelong** editing with real Wikidata changes over time, 500k+ QA edits, includes personas and multi-hop checks | 100k–1M rows [V] | `lukasthede/WikiBigEdit` (Apache-2.0) [V] | Finding: editing methods collapse at scale, and plain RAG or continual fine-tuning often wins. Hemispheres should be compared against this. |

### 1b. Propagation and multi-hop after update (most important for Hemispheres)

| Benchmark | What it measures | Size | HF id / source | Known issues |
|---|---|---|---|---|
| **MQuAKE** (Zhong et al. 2023) | Multi-hop questions (2–4 hops) whose answer must change after one or more edits. CF (counterfactual) and T (temporal, real Wikidata changes) variants | MQuAKE-CF ~9.2k, CF-3k subset, MQuAKE-T ~1.8k [M] | `princeton-nlp/MQuAKE` GitHub [M]; `nbalepur/mquake` (small) [V] | **The original is flawed.** MQuAKE-Remastered found edit contamination (edits in one case silently change another case's answers), conflicting edits, missing information in question instructions, and duplicates. Do not report numbers on the original alone. |
| **MQuAKE-Remastered** (Zhong et al., ICLR 2025) | Cleaned MQuAKE. Adds CF-6334 with proper train/test splits for parameter-based methods | Splits CF3k, CF9k, CF6334, T; 10k–100k rows | `henryzhongsc/MQuAKE-Remastered` (CC-BY-4.0, updated 2026-06-27) [V] | **Use this.** CF-6334 is the right split if a trained knowledge module might have seen test relations. |
| MQuAKE-ST / THESEUS (arXiv 2609.14528, Sep 2026) | MQuAKE rebuilt over a fixed Wikidata KG with annotated evidence paths | 10k–100k | `HalcyonSolutions/MQuAKE-ST` [V] | Very new. Useful if Hemispheres has an explicit KG or path-based knowledge side, because it lets you score path faithfulness. |
| MQuAKE-uns-HPSE (arXiv 2608.11660) | Unstructured (free-text) multi-hop editing | 1k–10k | `lliutianc/mquake-uns-hpse` (MIT) [V] | Very new, unvetted. |
| **RippleEdits** (Cohen et al., TACL 2024) | Ripple effects of an edit on six criteria: Logical Generalization, Compositionality I & II, Subject Aliasing, Relation Specificity, Forgetfulness. Subsets RECENT / RANDOM / POPULAR | ~5k edits total [M]; many test queries per edit | GitHub `edenbiran/RippleEdits` (MIT, JSON) [V]; not on HF under that name [V] | Queries are templated. Some "ground truths" come from Wikidata and are noisy. PropMEND (arXiv 2506.08920) [V] adds **Controlled RippleEdit**, which tests propagation along relations and entities unseen during editor training. That is the right generalization test for a trained knowledge module. |
| MINTQA (arXiv 2412.17032) [V] | Multi-hop QA over new and long-tail knowledge | ~? | paper | Useful as a "new knowledge" multi-hop test. Size unverified. |
| CRAFT (arXiv 2508.01302) [V] | Evolving editing benchmark with composite reasoning and temporal consistency | ? | paper | New; check release. |
| Evaluating Deep Unlearning (arXiv 2410.15153) [V] | Deletion must also stop re-derivation of the fact through logical rules from retained facts | synthetic family KG (EDU-RELAT) [M] | paper | Closest thing to "ripple effects of deletion". |

### 1c. Temporal and streaming knowledge (continual updating)

| Benchmark | What it measures | Size | HF id | Known issues |
|---|---|---|---|---|
| **TemporalWiki** (Jang et al. 2022) | Continual pretraining on Wikipedia diffs between monthly snapshots (2021). Probes split into changed and unchanged facts | Snapshots are GBs; TWiki-Probes are ~10k-scale [M] | `seonghyeonye/TemporalWiki` [V] (card unverified) | Stale (2021). Probe templates are noisy. The community rebuild `saxenan3/temporalwiki-drift-cl` (Nov 2025 → Feb 2026 snapshots, 1k–10k QA, CC-BY-SA) [V] is small and unvetted but post-dates most base models' cutoffs. |
| **StreamingQA** (Liska et al., DeepMind 2022) | QA over a stream of WMT news (2007–2020), tested quarter by quarter | ~147k questions [M] | `pclucas14/streamingqa_sanitized`, `bg51717/streamingqa` [V]; original GitHub `deepmind/streamingqa` [M] | News documents must be rebuilt from WMT. Old events are already in modern models' pretraining data, so "new knowledge" is contaminated. |
| **EvolvingQA** (Kim et al. 2024) | Continual knowledge learning with monthly Wikipedia snapshots; separate updated and new knowledge | 1M–10M rows [V] | `kat-research/EvolvingQA` [V] | Generated by an LLM from diffs, so some noise. Mostly single-hop. |
| **RealTimeQA** (Kasai et al. 2022) | Weekly current-events multiple-choice questions | Weekly batches, a few thousand total [M] | Only community copies on HF (`bzz2/unnatural_realtimeqa_public_2023`) [V] | Update cadence after 2024 unclear [unverified]. |
| **FreshQA** (Vu et al. 2023, FreshLLMs) | 600 questions [M] labeled never / slow / fast-changing plus false-premise; answers updated over time | ~600 [M] | GitHub `freshllms/freshqa` is **still updated; latest listed version 2026-04-21** [V]; HF mirrors by date, e.g. `bojanbabic/freshqa_09302025` [V] | Too small for training. Best used as a sanity eval of whether a model with an updated module answers current facts. |
| **OAKS** (arXiv 2603.07392, Mar 2026) [V] | Online adaptation to streaming knowledge: the same questions asked at every interval while facts change several times. OAKS-BABI and OAKS-Novel | ? | GitHub `kaistAI/OAKS` [V] | New. Mostly in-context and agent-memory evaluation, but the protocol (repeated queries across update phases) is a good template for Hemispheres. |
| ChroKnowBench (2410.09870), TiEBe (2501.07482) [V] | Chronological and time-sensitive knowledge | – | papers | Supplementary. |
| EMERGE (2507.03617) [V] | Updating a KG from emerging text; 1.25M KG edits across 10 Wikidata snapshots from 2019–2025 | large | promised release | Relevant if the knowledge side is a KG populated from text. Check release status. |

### 1d. Deletion / unlearning

| Benchmark | What it measures | Size | HF id | Known issues |
|---|---|---|---|---|
| **TOFU** (Maini et al. 2024) | Unlearning facts about 200 fictitious authors (20 QA each = 4k QA). Forget splits 1/5/10%. Also retain, real-author, and world-facts sets | 4k QA plus perturbed variants | `locuslab/TOFU` (MIT, 88.6k downloads) [V]; maintained eval harness `locuslab/open-unlearning` [M] | Fictitious, so no pretraining leakage, which suits Hemispheres. Known issues: ROUGE and truth-ratio metrics can be gamed; relearning attacks recover "unlearned" facts; paraphrase leakage. For Hemispheres, deletion should be exact removal from the knowledge store. The test is whether the reasoner **still produces the fact** (residual leakage into reasoner weights). WaterDrum-TOFU (`Glow-AI/WaterDrum-TOFU`) adds watermark-based metrics [V]. |

### 1e. Multi-hop reasoning (static; checks the reasoner stays competent)

| Benchmark | Size | HF id | Notes |
|---|---|---|---|
| HotpotQA | 113k | `hotpotqa/hotpot_qa` (CC-BY-SA-4.0) [V] | Many questions can be answered with shortcuts from a single hop. |
| 2WikiMultihopQA | ~192k [M] | `framolfese/2WikiMultihopQA` (HotpotQA-format repack), `xanhho/2WikiMultihopQA` (Apache-2.0) [V] | Template-generated. Has evidence triples, which suits KG-style knowledge modules. |
| MuSiQue | ~25k answerable (plus unanswerable) [M] | `dgslibisey/MuSiQue` (19k downloads) [V] | Built to resist single-hop shortcuts. Hardest of the three. |

### 1f. Other 2025–2026 work to know (verified on HF Papers)

- **"Can a Language Model Learn Facts Continually in Its Weights?"** (arXiv 2607.11020, Baseten, Jul 2026). Writes invented facts into Qwen3 and tests 5 question types against an **in-context oracle**. Bare-statement facts keep 1% accuracy after 20 sequential writes; facts written with diverse "study" data keep 46%. A forgotten fact supplied in the prompt recovers to 77–80%. Code at `basetenlabs/cortex`, data at `baseten/cortex`. **Very close to Hemispheres' motivation. Its protocol (in-context oracle, sequential writes, recitation vs use) is a good one to reuse.**
- **Scaling Laws for Hypernetwork-Based Knowledge Injection** (arXiv 2607.19604, Jul 2026). Separates injection capacity from the target model's general capability. MegaWikiQA has tens of millions of multi-hop QA pairs from Wikidata5M; data in the HF collection `nace-ai/hypernetwork-datasets`.
- **PropMEND** (2506.08920). Meta-learned propagation; Controlled RippleEdit dataset.
- **Doc-to-LoRA** (2602.15902). A hypernetwork turns a document into a LoRA adapter, i.e. a knowledge-module-as-adapter baseline.
- **LMLM** (2505.15962) and **Co-LMLM** (2607.07707). Pretraining that offloads facts to an external KB. Co-LMLM at 360M reports SimpleQA-verified scores on par with gpt-4o-mini. An LMLM forgetting audit is at arXiv 2607.00605 [V].
- **Pretraining with hierarchical memories** (Apple, 2510.02375). Separates long-tail from common knowledge.
- **Memory Decoder** (2508.09874). A pretrained plug-and-play memory for a frozen LLM.

### 1g. Evaluation protocol recommendations for Hemispheres

1. **Always report an in-context oracle.** The same frozen reasoner with the updated fact(s) pasted into the prompt. The propagation gap is (accuracy with module update) / (accuracy with in-context oracle), measured separately for single-hop, 2-hop, 3-hop, aliasing, and reverse questions.
2. **Score by autoregressive generation with exact match or F1.** Never use teacher-forced probability comparisons (Mirage paper).
3. **Scale the number of updates**: 1 → 100 → 10k → 100k sequential updates (WikiBigEdit-style). Plot retention of *earlier* updates as well.
4. **Locality:** unrelated facts unchanged, and general reasoning unchanged (GSM8K / BBH / MuSiQue with gold context). If the reasoner is truly frozen and the module is gated, these should be exactly invariant. That is itself a selling point to show.
5. **Deletion:** remove from the store, then probe for leakage of the fact through reasoner weights (TOFU-style, plus relearning attacks).
6. **World swap (the key Hemispheres test):** train with knowledge module K_A (world A), then swap in K_B, an entirely new synthetic world the reasoner never saw. Multi-hop accuracy on B should match A. None of the standard benchmarks test this, but synthetic worlds make it easy.
7. **Cost per update:** wall-clock and FLOPs per fact, compared with MEMIT/AlphaEdit, LoRA fine-tuning, RAG indexing, and Doc-to-LoRA or hypernetwork injection.
8. **Baselines:** in-context oracle; BM25 or dense RAG; LoRA continual fine-tuning; MEMIT / AlphaEdit / WISE / GRACE via EasyEdit; MeLLo (the MQuAKE memory-based method); KBLaM; LMLM (at matched scale if pretraining from scratch).

---

## 2. Synthetic worlds for controlled experiments

| Setup | How it's built | Public code / data | Laptop fit |
|---|---|---|---|
| **Physics of LMs bioS / bioR** (Allen-Zhu & Li, Parts 3.1–3.3) | N synthetic people, each with 6 attributes (birth date, birth city, university, major, employer, employer city) sampled uniformly. **bioS** fills attributes into sentence templates (single or multi-template, optionally permuted and with full-name repetition). **bioR** uses LLM-written "close-to-real" bios. QA pairs cover half the people and are used for (mixed) training; the other half test **knowledge extraction**. Key finding: extraction to unseen people works only with **knowledge augmentation** (multiple paraphrases or permutations per person). Part 3.3 capacity result is about **2 bits/param**. | **Yes.** `facebookresearch/PhysicsLM4` includes the Capo dataset, which contains bioS and bioR generators under `/data-synthetic-pretrain/Capo-bioS-bioR` (Apache-2.0, plus BSD-3 for Lingua code). Released model weights are in a "Physics of Language Models" HF collection [V]. The FAQ says release of the full bioS data and bioR prompts was pending review [V], but the generator is in the repo. Training code uses Lingua (CUDA-oriented) [V]. | Excellent. Re-implement the generator in ~200 lines. With N = 10k–100k people and a GPT-2-small-or-smaller model, a run takes hours. |
| **Grokked Transformers** (Wang et al., NeurIPS 2024) | Random KG: [M] ~2,000 entities, 200 relations, ~20 outgoing edges per entity, giving atomic facts (h, r, t). **Inferred** 2-hop facts (h, r1, r2) → t. A subset of entities has *only* atomic facts in training, which gives the OOD test. The ratio φ = inferred/atomic controls the speed of grokking. Tasks: composition and comparison (plus a large-search-space "complex reasoning" task). Finding: implicit composition generalizes in-distribution only after grokking and **fails OOD**. Comparison generalizes OOD. | **Yes.** `OSU-NLP-Group/GrokkedTransformer`: data-generation notebooks (composition/comparison/complex_reasoning.ipynb), data via SharePoint, 8-layer GPT-2-style models, `max_steps 1.5M`, batch 512, torch 1.13/CUDA [V]. Follow-up "Grokking in the Wild" (2504.20752) [V] extends to real multi-hop data via augmentation. | Very good. Sequences are ~5–8 tokens, so each step is cheap. Full grokking to 1.5M steps could take ~1–3 days on an M5 Max [E]. Smaller KGs grok faster. **The OOD-composition failure is exactly the gap Hemispheres claims to close**: facts in a module, composition in a frozen reasoner. |
| **Two-Hop Curse** (Balesni, Korbak, Evans 2024/25; "Lessons from Studying Two-Hop Latent Reasoning", arXiv 2411.16353) | Synthetic facts about fictional entities as triplets, fine-tuned into Llama-3-8B-class models. Without chain-of-thought, models compose two synthetic facts at chance when they were learned in separate documents. They succeed when one fact is synthetic and the other natural, or with CoT. | Paper [V]. Public code and data **unverified**. The design is easy to replicate. | Good as a fine-tuning experiment on a frozen 1–4B model with LoRA. Directly tests "does a newly written fact propagate into latent reasoning". |
| **EntiGraph / Synthetic Continued Pretraining** (Yang et al. 2024, arXiv 2409.07431) | From a small corpus (QuALITY, ~1.3M tokens), extract entities and prompt an LLM to write about relations between entity pairs and triples. This yields a ~455M-token synthetic corpus [M] for continued pretraining of Llama-3-8B. Knowledge gain scales log-linearly with synthetic tokens. | Corpus `zitongyang/entigraph-quality-corpus` (Apache-2.0) [V]; QA SFT set `zitongyang/entigraph-qasft` [V]; model `zitongyang/llama-3-8b-entigraph-quality` [V]; code GitHub `ZitongYang/Synthetic_Continued_Pretraining` [M]. | You can reuse the released corpus. Regenerating it needs API spend. Useful as the "how to *write* a document into a knowledge module" recipe (diverse restatements); the cortex paper reaches the same conclusion. |
| TOFU fictitious authors | 200 GPT-4-generated author profiles plus 4k QA | `locuslab/TOFU` [V] | Ready-made fictional world with deletion splits. |
| KBLaM synthetic KB | LLM-generated entities with (name, description, objectives, purpose) triples plus QA | Generator script `dataset_generation/gen_synthetic_data.py` (uses Azure OpenAI) [V] | Reusable. |
| "Learning from Synthetic Data Improves Multi-hop Reasoning" (2603.02091) [V] | Rule-generated fictional-knowledge multi-hop data transfers to real benchmarks | paper | Supports training the reasoner on synthetic worlds only. |
| MegaWikiQA (2607.19604) | Tens of millions of multi-hop QA from Wikidata5M across 39 domains | `nace-ai/hypernetwork-datasets` collection [V] | Large. A good source of real-KG multi-hop at scale. |

**Recommended toy world for Hemispheres.** Combine bioS-style entity attributes with Grokked-style relational edges (entity → employer → city → country, and so on). Generate worlds A, B, C with *disjoint* entities but the *same schema*. That supports three tests:
- train the reasoner on A;
- swap in the knowledge for B and test;
- make counterfactual edits inside B and test 1-, 2-, and 3-hop propagation.

---

## 3. Small-scale training feasibility

### 3a. The hardware: Apple M5 Max
- M5 Pro and M5 Max were announced March 2026 [V: Apple newsroom]. Each GPU core has a **Neural Accelerator** (matrix engine) [V].
- Memory bandwidth is 460 GB/s on the 32-core GPU and 614 GB/s on the 40-core GPU [V: Wikipedia/Apple]. Which GPU bin the 64 GB configuration uses is **unverified**; check the machine.
- Apple claims more than 4x the peak GPU AI compute of M4 Max [V]. Third-party estimates put FP16 at ~70 TFLOPS [V as a claim; Apple doesn't publish TFLOPS].
- MLX needs **macOS 26.2+** to use the Neural Accelerators [V: Apple ML Research]. Apple reports ~3.5–4x faster time-to-first-token than M4 across Qwen 1.7B–30B, and 19–27% faster decode [V]. These are inference (compute-bound prefill) numbers; **no published M5 Max training throughput was found** [V: searched].

### 3b. Frameworks
- **MLX / mlx-lm** [V]: `mlx_lm.lora` supports `--fine-tune-type lora | dora | full`, QLoRA (train on a quantized model), and `--grad-checkpoint`. Model families include Llama, Qwen2/3, Gemma, OLMo, Mistral, Phi, MiniCPM, and InternLM2. Custom architectures are easy in raw MLX: NumPy-like API, `mlx.nn`, `mlx.optimizers`, autodiff with `value_and_grad`, and `mx.compile`. It is the best-performing option on Apple Silicon for most transformer workloads. mlx-lm docs focus on fine-tuning. For from-scratch pretraining, use community ports (nanochat-mlx: `scasella/nanochat-mlx`, `NeuroArchitect/nanochat-mlx` [V]) or write a ~300-line loop.
  - Anecdotal throughput [V, community]: GPT-2 124M pretraining at ~3.0k tok/s on a **base M4 Mac mini 16GB**; ~3.7k tok/s fp32 GPT-2 fine-tune on an M4 Pro. Old mlx-lm doc: ~250 tok/s LoRA on a 7B-class model on an M1 Max.
  - Hemispheres needs **Qwen3.5**, which is a hybrid architecture (Gated DeltaNet linear attention with full attention every 4th layer, attention output gate, multimodal wrapper) [V: config]. Check that mlx-lm supports training it before relying on it [unverified]. Plain-transformer Qwen3 / SmolLM3 / OLMo are safe.
- **PyTorch MPS** [V, mixed sources]: works for training and supports bf16 (macOS 14+). **`torch.compile` on MPS is not usable for training** per current reports. Some ops still fall back to CPU. Benchmarks conflict (MLX faster in most op-level tests; one training benchmark favored MPS). Most research code (EasyEdit, KBLaM, memory layers, Cartridges, GrokkedTransformer) assumes CUDA. It often runs on MPS with `device="mps"` patches, except custom CUDA kernels (Meta memory layers' embedding-bag kernels, FlexAttention, Triton).
- **Recommendation:** run synthetic-world experiments in **MLX** (fast, full control, no CUDA dependencies). Use **PyTorch-MPS or cloud CUDA** for anything built on HF-Transformers research code (EasyEdit baselines, KBLaM).

### 3c. Realistic laptop budgets on a 64 GB M5 Max [E, measure first]
- **From-scratch pretraining.** Achieved training throughput is maybe 15–25 TFLOPS in bf16 with MLX (optimistic; M-series training MFU is typically well below NVIDIA's). A 124M GPT at ~1 GFLOP/token gives **~10–30k tok/s**, so 1B tokens takes ~10–30 h.
  - 124M Chinchilla-optimal (2.5B tokens): ~1–3 days.
  - 350M: 7B tokens is impractical (1–3 weeks).
  - Synthetic-world models (10–100M params, 50M–1B tokens including repetitions): **hours to 1–2 days**. This is the sweet spot.
- **Memory.** 64 GB comfortably holds full fine-tuning of ≤1B models in bf16 with AdamW (≈16 bytes/param, i.e. 16 GB for 1B, plus activations), LoRA or adapters on 4–9B frozen models, and QLoRA on bigger ones. A frozen 4B reasoner in bf16 (~8 GB) plus a trainable knowledge module of 100M–1B params fits.
- **Action item:** run a 10-minute benchmark (MLX GPT-2-small forward/backward at seq 1024, bf16) before planning. It turns every [E] number here into a measured one.

### 3d. Cloud GPUs
- **H100 pricing, Sep 2026** [V]: $1.49–1.60/hr (Vast.ai marketplace), $1.99 (RunPod PCIe), $2.69 (RunPod SXM), $2.99 (Lambda SXM), up to $6.98 (Azure). Plan for **$2–3/GPU-hr**, or ~$16–24/hr per 8xH100 node.
- **Speedrun reference points** [V]:
  - modded-nanogpt 124M reaches val loss 3.28 in **1.126 min on 8xH100** (record 2026-08-06).
  - The 350M track (loss 2.92) record is **17.35 min** (2025-12-31).
  - nanochat: d20 (~561M) in about 1.5 h on 8xH100 (~$48; ~$15 spot). The "GPT-2 CORE" leaderboard best is **1.65 h** (2026-03-14).
  - These tuned codebases (Muon, FP8, FlexAttention) are the cheapest way to get a baseline dense model to compare a Hemispheres variant against.
- **Rough pretraining costs** at 6ND FLOPs, ~40% MFU on H100 bf16 (~400 TFLOPS effective), $2.5/GPU-hr [E]:

| Model | Tokens | GPU-hours | ~Cost/run |
|---|---|---|---|
| 124M | 2.5B (Chinchilla) | ~1.3 | ~$3–5 (speedrun code: < $1) |
| 350M | 7B | ~10 | ~$25–30 |
| 1B | 20B (Chinchilla) | ~85 | ~$200–250 |
| 1B | 100B (over-trained, "useful") | ~420 | ~$1,000–1,300 |

  For a paper you need a dense baseline, the Hemispheres variant, and 3–5 ablations, at 2–3 scales. A convincing 124M–1B scaling study costs **$2k–10k of compute** [E], not counting data prep and debugging.

### 3e. Small open base models to use as frozen reasoners (Sep 2026)

| Model | Sizes (base available) | License | Notes |
|---|---|---|---|
| **Qwen3.5** (Feb 2026) | 0.8B, 2B, 4B, 9B `-Base` (+27B, 35B-A3B, …) [V] | Apache-2.0 [V] | Current Qwen small line. Hybrid linear/full attention and multimodal: harder to hook memory layers into and to do interpretability on. Tooling support needs checking. |
| **Qwen3** (2025) | 0.6B, 1.7B, 4B, 8B Base [M] | Apache-2.0 [M] | **Plain transformer. Safest choice for inserting modules.** Used by Cartridges and the cortex paper. Well supported in MLX, EasyEdit (probably), and PEFT. |
| **SmolLM3-3B-Base** (Jul 2025) | 3B [V] | Apache-2.0 [V] | Fully open (data mix published; 11T tokens). Intermediate checkpoints in `SmolLM3-3B-checkpoints` [V]. SmolLM2 135M/360M/1.7B with intermediate checkpoints [V]: good tiny reasoners. |
| **OLMo 2 1B** (`allenai/OLMo-2-0425-1B`) [V]; OLMo 3 / 3.1 7B & 32B [V] | 1B, 7B, 32B | Apache-2.0 [V] | **Fully open pretraining data (Dolma), so you can check whether the reasoner has seen a fact.** That makes it the most useful choice for contamination-controlled knowledge experiments. No OLMo 3 at 1B found [V]. |
| **Gemma 4** (Mar–May 2026) | E2B, E4B, 12B, 26B-A4B, 31B [V] | **Apache-2.0** [V: HF tags] (a change from the custom Gemma license) | E2B/E4B use per-layer embeddings (an unusual architecture). Gemma 3 1B/4B remain under the Gemma Terms of Use [M]. |
| **Llama 3.2** 1B/3B | 1B, 3B | Llama 3.2 Community License (attribution, 700M-MAU clause) [M] | KBLaM supports Llama-3.2-1B-Instruct out of the box [V]. Old but widely used in the editing literature. |
| HF `nanowhale-100m-base` (Apr 2026) | 100M (DeepSeek-V4-style MoE) | Apache-2.0 [V] | Curiosity. Tiny MoE trained on FineWeb-edu. |

**Pick:** Qwen3-0.6B/1.7B-Base or OLMo-2-1B for the medium experiment (a plain transformer, where OLMo also gives known pretraining data). Then Qwen3-4B or Qwen3.5-4B for the headline result.

---

## 4. Open-source code for the closest architectures

| System | Repo | License | Status / usability |
|---|---|---|---|
| **Memory Layers at Scale** (Meta, 2412.09764) | `facebookresearch/memory` [V] | **CC-BY-NC** [V] | Product-key memory layers built on Meta Lingua. Custom CUDA kernels; configs for 373M and 7B; single-GPU launch possible [V]. Not Apple-friendly as-is. Product-key memory is ~100 lines to re-implement in MLX (use `mx.take` plus a top-k over sub-keys). The non-commercial license matters only if you reuse the code. |
| **DeepSeek Engram** (arXiv 2601.07372, Jan 2026) | `deepseek-ai/Engram` [V] | ? (check) | "Conditional memory": hashed N-gram embedding lookup, O(1), a separate sparsity axis from MoE. Reports gains over iso-parameter/iso-FLOP MoE at 27B [V]. **The code is a demo that mocks attention and MoE** [V]. Follow-ups include a hot-tier extension (2601.16531) and TF-Engram (2607.07388) [V]. Conceptually close to Hemispheres (a separable, lookup-based knowledge store); easy to re-implement. |
| **KBLaM** (Microsoft, ICLR 2025) | `microsoft/KBLaM` [V] | MIT [V] | KB triples → sentence-encoder embeddings → linear adapters → "knowledge tokens" that the LM attends to with rectangular attention. Base models: Llama-3-8B-Instruct, **Llama-3.2-1B-Instruct**, Phi-3-mini [V]. Synthetic KB generator included (needs an OpenAI/Azure endpoint) [V]. Research-grade; PyTorch. **1B variant is likely runnable on MPS or small cloud GPUs** [E]. Closest drop-in baseline for "attach a knowledge module to a frozen model". |
| **LMLM / Co-LMLM** (Cornell) | `kilian-group/LMLM` [V] | ? (check) | Pretraining recipe that masks the loss on facts retrieved from a DB, so the model learns to look facts up instead of memorizing them. HF: models `LMLM-llama2-176M`, `-382M`, an annotator model, pretraining data `LMLM-pretrain-dwiki6.1M(_v2/_cleaned)`, a database, and a **MQuAKE multi-hop SFT set** (`LMLM-multihop-mquake-sft`, Feb 2026) [V]. **The most directly comparable "separate knowledge from reasoning at pretraining" baseline, at a scale you can afford (176M–382M).** |
| **Memory³** (IAAR-Shanghai, 2024) | No official code or weights found on HF or GitHub this session [V: searched] | – | 2.4B model with sparse "explicit memory" KV caches. Treat as paper-only [unverified]. |
| **Cartridges** (Hazy Research, 2025) | `HazyResearch/cartridges` [V] | Apache-2.0 [V] | Trains a compact KV cache ("cartridge") per corpus via self-study (synthetic conversations plus context distillation). Supports Qwen3-4B. Needs GPUs and an inference server (Tokasaurus/SGLang) for synthesis [V]. A strong "knowledge-as-a-swappable-artifact" baseline; plan to run it in the cloud. |
| **RETRO** | `lucidrains/RETRO-pytorch` [V] (PyTorch, Faiss; community, likely unmaintained [M]); NVIDIA Megatron-LM `tools/retro` + InstructRetro branch [V] | MIT / Apache-ish [M] | Megatron is heavy multi-GPU infrastructure. The lucidrains version is hackable for small experiments. Retrieval is a baseline, not the main comparison. |
| **Memory Decoder** (2508.09874) | Paper [V]; code unverified | – | Small pretrained transformer that mimics kNN-LM distributions; plugged into frozen LLMs by interpolation. |
| **Doc-to-LoRA** (2602.15902), **hypernetwork injection** (2607.19604) | HF collection `nace-ai/hypernetwork-datasets` [V] | – | "Knowledge module = generated adapter" baselines. |
| **EasyEdit** (zjunlp) | `zjunlp/EasyEdit` [V] | MIT [M] | Methods: ROME, MEMIT, PMET, AlphaEdit, UltraEdit, R-ROME, EMMET, MEND, MALMEN, SERAC, IKE, GRACE, MELO, **WISE**, FT/AdaLoRA, UNKE, AnyEdit, CORE, NAMET [V]. Datasets: KnowEdit, ZsRE, CounterFact, WikiBio, WikiData-Recent, **WikiBigEdit**, AKEW, LEME, UNKE [V]. **2026-07-08: Transformers 5.x compatibility and better multi-GPU support** [V]. No MPS mention [V]. Qwen/Llama supported; Qwen3 or Qwen3.5 support not stated [V]. Use it for editing baselines, preferably on a rented GPU. |
| ROME / MEMIT originals | `kmeng01/rome`, `kmeng01/memit` [M] | MIT [M] | Superseded by the EasyEdit implementations. |
| MQuAKE / MeLLo | `princeton-nlp/MQuAKE` [M] | – | MeLLo is the memory-based (retrieve-edit-then-reason) baseline, conceptually close to Hemispheres with an external store. |
| Baseten cortex | `basetenlabs/cortex` + HF `baseten/cortex` [V] | ? | Protocol for sequential fact writes into Qwen3 with an in-context oracle. Reuse its evaluation. |

---

## 5. Recommended experimental ladder

### (a) Laptop toy: synthetic world with a swappable knowledge module (hours to days, MLX)
**Goal:** show that a reasoner trained against module K_A handles a *new* module K_B, and in-place edits to K_B, with multi-hop accuracy close to the in-context oracle. A monolithic transformer of equal parameter count should fail at this.

- **World generator:** bioS-style people with 6 attributes, plus relational edges (works_at → company, company → HQ city, city → country, person → mentor). Worlds A, B, C use disjoint entity names and the same schema. About 5k–20k entities per world.
- **Architectures** (all ~20–60M params, 6–8 layers, d=384–512, trained from scratch in MLX):
  1. Dense baseline: facts trained into the weights.
  2. Reasoner plus **product-key memory layer** (or Engram-style hashed lookup) inserted mid-stack. Only the memory values are world-specific. Train on world A with composition QA, then freeze the reasoner and write world B into a fresh memory. Writing can be a gradient step on memory values only, or a direct write via the key encoder.
  3. Reasoner plus **explicit KB lookup**, LMLM-style: emit a lookup token, KB returns the value, loss is masked on retrieved values.
  4. In-context oracle: the same reasoner with the relevant facts in context.
- **Metrics:** 1/2/3-hop exact-match accuracy on world B. Accuracy after k ∈ {1, 100, 1k} counterfactual edits, including multi-hop answers through edited facts. Locality. Parameters and time per update. Reuse the Grokked-Transformer ID/OOD split logic, since OOD composition is where dense models fail.
- **Compute [E]:** each run is ~50M–500M training tokens on a 20–60M model, so ~1–10 h on an M5 Max. A full ablation grid takes a few days.
- **Deliverable:** one figure of propagation gap vs number of edits for each architecture. That alone is a workshop-paper-quality result if the gap closes.

### (b) Medium: knowledge module on a frozen small open model (days to 2 weeks; laptop plus some cloud)
- **Reasoner:** Qwen3-1.7B-Base or OLMo-2-1B (frozen). Scale to Qwen3-4B or Qwen3.5-4B for the final result.
- **Module options:** (i) KBLaM-style knowledge tokens from an encoder over triples; (ii) memory layer(s) inserted with a gated residual, trained with the reasoner frozen; (iii) generated LoRA via a hypernetwork, Doc-to-LoRA style. Train the *interface* (adapters, gates, key encoder) once on a training world. After that, knowledge updates touch only the store.
- **Data:** synthetic worlds for interface training (step (a) generator, TOFU, KBLaM synthetic KB). Then test on MQuAKE-Remastered CF-6334 (test split), RippleEdits plus Controlled RippleEdit, WikiBigEdit (sequential scale), TOFU (deletion), and FreshQA or the 2025–26 TemporalWiki-drift set (real post-cutoff updates).
- **Baselines:** in-context oracle, RAG, LoRA continual fine-tuning, MEMIT/AlphaEdit/WISE/GRACE (EasyEdit), MeLLo, KBLaM, Cartridges (cloud).
- **Compute [E]:** interface training on a 1–2B frozen model fits in 64 GB with bf16 and gradient checkpointing. EasyEdit baselines are best run on a rented A100 or H100: ~20–60 GPU-hours, ~$50–150.

### (c) A convincing paper-level result
1. **The two-component claim, shown at more than one scale.** Pretrain from scratch a reasoner-plus-knowledge-store model and a matched dense baseline at ~125M, ~350M, and ~1B (cloud, using modded-nanogpt or nanochat code). Compare on standard LM evals, and show the Hemispheres model is not worse at iso-FLOPs and iso-active-params. Also compare against LMLM (176M/382M weights public) and memory-layer baselines. Cost ~$2k–10k [E].
2. **Knowledge update without touching the reasoner.** Large-scale sequential updates (WikiBigEdit, 10k–100k edits) with (i) edit accuracy, (ii) **multi-hop propagation on MQuAKE-Remastered and RippleEdits close to the in-context oracle**, (iii) locality and exact invariance of reasoning benchmarks, and (iv) deletion with zero leakage (TOFU plus relearning attack).
3. **World swap:** train the reasoner on one knowledge store, swap in a disjoint store (synthetic world, or a different domain KB), and show reasoning transfers with no reasoner training.
4. **Cost:** update cost per fact is orders of magnitude below fine-tuning and competitive with RAG, with better multi-hop than RAG.
5. **Mechanistic check:** probe that facts are *not* stored in the reasoner. Ablate the store and measure factual recall collapsing while held-out reasoning (on in-context facts) stays intact.

Reviewers will ask "why not just RAG or long context?" and "how is this different from LMLM, KBLaM, memory layers, and Engram?" The answer has to be multi-hop propagation and latent (non-CoT) composition over updated facts. On those, RAG and in-weights editing are weak (the two-hop curse, the cortex paper, PropMEND's numbers).

---

## 6. Unverified or open items
- M5 Max training throughput (no published numbers; all laptop tok/s figures are [E]). Also which GPU bin the 64 GB configuration uses.
- Whether mlx-lm and EasyEdit support training Qwen3.5 (hybrid linear attention).
- Exact sizes of MQuAKE-CF/T, RippleEdits, StreamingQA, 2Wiki, MuSiQue, and zsRE splits (from memory).
- Grokked-Transformer KG hyperparameters (2k entities / 200 relations) are from memory. The repo only confirmed notebooks, batch 512, and 1.5M steps.
- Public code or data for the Two-Hop Curse synthetic dataset.
- Memory³ code and weights availability (not found).
- Licenses for the Engram and LMLM repos.
- RealTimeQA update status after 2024.
- EntiGraph GitHub repo name (`ZitongYang/Synthetic_Continued_Pretraining`) is from memory; HF artifacts are verified.

## Sources (selected)
- HF datasets: henryzhongsc/MQuAKE-Remastered, lukasthede/WikiBigEdit, zjunlp/KnowEdit, locuslab/TOFU, kat-research/EvolvingQA, hotpotqa/hotpot_qa, dgslibisey/MuSiQue, framolfese/2WikiMultihopQA, zitongyang/entigraph-quality-corpus, kilian-group/LMLM-*, saxenan3/temporalwiki-drift-cl, HalcyonSolutions/MQuAKE-ST
- HF papers: 2502.11177 (Mirage), 2503.05683 (WikiBigEdit), 2506.08920 (PropMEND), 2603.07392 (OAKS), 2607.11020 (cortex), 2607.19604 (hypernetwork scaling), 2607.07707 (Co-LMLM), 2510.02375, 2508.09874, 2602.15902, 2412.09764, 2309.14316, 2404.05405, 2411.16353, 2409.07431, 2504.20752, 2603.02091
- [Apple ML Research: MLX on M5](https://machinelearning.apple.com/research/exploring-llms-mlx-m5); [Apple M5 Pro/Max newsroom](https://www.apple.com/newsroom/2026/03/apple-debuts-m5-pro-and-m5-max-to-supercharge-the-most-demanding-pro-workflows/); [Apple M5 (Wikipedia)](https://en.wikipedia.org/wiki/Apple_M5)
- [mlx-lm LORA.md](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md); [nanochat-mlx](https://github.com/scasella/nanochat-mlx); [Profiling Apple Silicon for ML training](https://arxiv.org/pdf/2501.14925)
- [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt); [nanochat](https://github.com/karpathy/nanochat)
- [H100 prices (IntuitionLabs)](https://intuitionlabs.ai/articles/h100-rental-prices-cloud-comparison); [RunPod vs Lambda vs Vast 2026](https://tech-insider.org/runpod-vs-lambda-vs-vast-ai-2026/)
- [facebookresearch/memory](https://github.com/facebookresearch/memory); [microsoft/KBLaM](https://github.com/microsoft/KBLaM); [HazyResearch/cartridges](https://github.com/HazyResearch/cartridges); [zjunlp/EasyEdit](https://github.com/zjunlp/EasyEdit); [deepseek-ai/Engram](https://github.com/deepseek-ai/Engram); [kilian-group/LMLM](https://github.com/kilian-group/LMLM); [facebookresearch/PhysicsLM4](https://github.com/facebookresearch/PhysicsLM4); [OSU-NLP-Group/GrokkedTransformer](https://github.com/OSU-NLP-Group/GrokkedTransformer); [edenbiran/RippleEdits](https://github.com/edenbiran/RippleEdits); [freshllms/freshqa](https://github.com/freshllms/freshqa); [lucidrains/RETRO-pytorch](https://github.com/lucidrains/RETRO-pytorch); [Megatron RETRO](https://github.com/NVIDIA/Megatron-LM/tree/InstructRetro/tools/retro)
