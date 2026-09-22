# Hemispheres: can knowledge and reasoning be separated inside neural LMs?

Literature survey, compiled 2026-09-22. Track: the scientific premise, meaning whether knowledge and reasoning can be separated at all, and what the interface between them has to do.

**Verification legend.** [V] means the arXiv ID, title, date and main claim were checked against arXiv (abstract page or API) during this survey. [V-web] means it was checked against the official page or a reliable secondary source. [M] means it comes from background knowledge and was not re-checked in this session. Treat [M] claims as needing a citation check before they go into a paper. Where the headline number is the authors' own claim and nobody has replicated it, the entry says so.

---

## 0. Bottom line

1. **Separation is partial and depends on the regime.** It is not all-or-nothing. The best-supported separation is between **long-tail, arbitrary, entity-attribute facts**, which are compressible, lookup-like and dependent on capacity, and **procedures and algorithms**, which depend on data and are shared across entities, languages and domains. There is growing architectural evidence (Engram, memory layers, hierarchical memories, LMLM) that moving the first kind out into a lookup store costs nothing, and can even *help* reasoning. Ablating the store wipes out factual recall while leaving reading comprehension, open-book QA and logic largely intact.
2. **What does not separate cleanly is "common" or conceptual knowledge.** This covers concepts, types, categories, relations, and the representations that facts are composed *through*. Every successful memory-offload architecture keeps a "backbone/anchor" that still holds common knowledge. None of them builds a reasoner that has no knowledge at all. The design question is **where to draw the line**, not whether a line exists.
3. **The hardest problem is composition, not storage.** Parametric facts compose poorly in a single forward pass: the two-hop curse, the reversal curse, and OOD composition failure in grokked transformers. Facts placed *in context* compose well. This argues strongly that the Hemispheres interface should hand retrieved knowledge back **in a form that attention can operate on**, whether tokens or KV/residual entries the reasoner attends to. It argues against merging knowledge into weights (FFN) and hoping the reasoner will compose it latently. The interface also has to be **re-entrant**, meaning callable again at any depth or reasoning step, because multi-hop needs the output of lookup 1 to become the key for lookup 2.
4. **The name is a metaphor.** Popular "left brain = logic, right brain = knowledge/creativity" is not supported (Nielsen et al. 2013). The defensible neuroscience analogies are complementary learning systems (hippocampus and neocortex), the semantic-memory vs. multiple-demand/executive dissociation, and the language vs. thought dissociation (Fedorenko et al. 2024). Each fits imperfectly, as discussed below.

---

## 1. Mechanistic evidence on where factual knowledge lives

### 1.1 FFNs as key-value memories and knowledge neurons
- **Geva, Schuster, Berant, "Transformer Feed-Forward Layers Are Key-Value Memories"**, arXiv 2012.14913 (2020-12-29; EMNLP 2021) [V]. FFN first-layer rows act as keys that detect textual patterns. Second-layer columns act as values that induce output-vocabulary distributions. Lower layers are shallow and pattern-like, upper layers semantic. This is the origin of the "FFN = memory" framing.
- **Dai et al., "Knowledge Neurons in Pretrained Transformers"**, arXiv 2104.08696 (2021-04-18; ACL 2022) [V]. Integrated-gradients attribution finds a few FFN neurons whose suppression or amplification changes a relational fact. *Caveat:* later work (e.g. Niu et al., "What does the Knowledge Neuron Thesis have to do with knowledge?", ICLR 2024 [M]) argues these neurons track token co-occurrence patterns rather than "knowledge" as such, and that editing them has limited generality.
- **Meng et al., ROME, "Locating and Editing Factual Associations in GPT"**, arXiv 2202.05262 (2022-02-10) [V]. Causal tracing localizes subject-attribute recall to mid-layer MLPs at the last subject token. The rank-one edit works for single facts. MEMIT (2210.07229) [M] scales this to thousands of edits.

### 1.2 Critiques: localization does not equal a separable module
- **Hase, Bansal, Kim, Ghandeharioun, "Does Localization Inform Editing?"**, arXiv 2301.04213 (2023-01-10; NeurIPS 2023 spotlight) [V]. Causal-tracing localization "does not provide any insight into which model MLP layer would be best to edit". Which layer gets edited predicts success far better than where tracing says the fact lives. *Implication:* a fact can be *written* in many places, and *where the model normally reads it from* is not a crisp address. Facts are not stored as isolated records.
- **Nanda, Rajamanoharan, Kramár, Shah, "Fact Finding" (Alignment Forum sequence, Dec 2023, not on arXiv)** [V-web]. In Pythia-2.8B, early MLPs act as a *lookup table in heavy superposition*. The authors *failed* to reverse-engineer the neuron-level lookup and falsified several simple hypotheses. The outputs (attribute directions) are interpretable, but the lookup itself is diffuse. They suggest treating fact recall as a black box that produces "multi-token embeddings".
- **Chughtai, Cooney, Nanda, "Summing Up the Facts: Additive Mechanisms Behind Factual Recall"**, arXiv 2402.07321 (2024-02-11) [V]. Several independent mechanisms (subject-driven, relation-driven, mixed) each contribute partial evidence, and the evidence is *summed*. Recall is redundant, not a single path.
- **Knowledge-editing ripple failures.** MQuAKE (Zhong et al., arXiv 2305.14795, 2023-05-24) [V] and RippleEdits (Cohen et al., arXiv 2307.12976, 2023-07-24) [V] show that edited facts often do not propagate to multi-hop questions or logical implications. **CaKE** (Yao et al., arXiv 2503.16356, EMNLP 2025) [V] attributes this to layer-local edits not being integrated into the *reasoning circuits* that consume them. *Implication for Hemispheres:* if you write knowledge into weights, the reasoner may not "see" it. This is the strongest practical argument for an explicit, attended interface rather than weight patches.

### 1.3 Dissecting the recall pipeline: attention does the extraction
- **Geva, Bastings, Filippova, Globerson, "Dissecting Recall of Factual Associations in Auto-Regressive LMs"**, arXiv 2304.14767 (2023-04-28) [V]. Recall happens in three steps: (1) early MLPs *enrich* the subject representation with many attributes; (2) relation information propagates to the last position; (3) upper-layer **attention heads extract** the specific attribute from the enriched subject representation. Several heads encode subject-attribute mappings in their parameters.
  - *This matters a great deal for Hemispheres.* In a vanilla transformer, knowledge is not a single FFN lookup. It is **FFN enrichment followed by attention-based querying**, so the "reasoner" (attention) and the "store" (MLP) already interact through a residual-stream interface. That interface is fairly well characterized.
- **Hernandez et al., "Linearity of Relation Decoding"**, arXiv 2308.09124 (2023-08-17) [V]. For many relations, attribute extraction from the subject representation is well approximated by a single affine map per relation. So the *key* is roughly "subject representation × relation-specific linear operator". That is a useful design prior for an interface.
- **Yu, Belinkov, Ananiadou, "Back Attention"**, arXiv 2502.10835 (2025-02-15) [V]. Logit-flow analysis finds four stages (entity enrichment, attribute extraction for entities, relation enrichment, attribute extraction for relations). Multi-hop failures cluster at the last stage. Letting lower layers attend to higher-layer representations helps.
- **Wong, "Paying Attention to Facts"**, arXiv 2502.05076 (2025-02-07) [V]. Theory plus toy experiments show that attention layers can themselves store facts, with capacity tied to tensor rank. *Implication:* "FFN = knowledge, attention = reasoning" is a tendency, not a law.
- **Zucchet et al., "How do language models learn facts? Dynamics, curricula and hallucinations"**, arXiv 2503.21676 (CoLM 2025) [V-web]. Fact learning shows a plateau during which **attention-based recall circuits** form. Hallucinations appear at the same time as knowledge. Fine-tuning new facts corrupts existing ones. So a *recall circuit* has to exist before facts can be stored usefully, which hints that the "reader" is a learned skill in its own right.

### 1.4 Capacity and storage format
- **Allen-Zhu & Li, Physics of LMs 3.1 (Knowledge Storage & Extraction)**, arXiv 2309.14316 (2023-09-25) [V]. Facts are *extractable* (usable in QA) only if pretraining included **knowledge augmentation**, meaning multiple paraphrases or permutations per entity. Otherwise the model memorizes the text without making the fact linearly available at the entity name. Knowledge storage format therefore depends on training data, and "stored" does not mean "retrievable".
- **Physics of LMs 3.2 (Knowledge Manipulation)**, arXiv 2309.14402 (2023-09-25) [V]. Models can retrieve facts but fail at simple *manipulation* without CoT: classification ("is X's birth year even?"), comparison, and especially **inverse search**, which relates to the reversal curse. CoT at inference helps classification and comparison but not inverse search.
- **Physics of LMs 3.3 (Knowledge Capacity Scaling Laws)**, arXiv 2404.05405 (2024-04-08) [V]. About **2 bits of knowledge per parameter**, which holds under int8 quantization. MoE has almost no capacity penalty. Junk data reduces capacity (mitigated by prepending domain tags). *Implication:* storing knowledge is expensive in parameters. A 1B "cognitive core" holds at most about 2 Gbit of facts. This is the quantitative reason to externalize.
- **Roberts, Chatterji, Narang, Lewis, Hupkes, "Compute Optimal Scaling of Skills: Knowledge vs Reasoning"**, arXiv 2503.10061 (ACL Findings 2025) [V-web]. Knowledge QA is **capacity-hungry**, preferring more parameters. Code/reasoning is **data-hungry**, preferring more tokens. The difference persists after controlling for datamix. This is independent scaling evidence that the two skills behave differently.

---

## 2. Where reasoning happens

### 2.1 Attention circuits for in-context computation
- **Olsson et al., "In-context Learning and Induction Heads"**, arXiv 2209.11895 (2022-09-24) [V]. Induction heads (prefix matching plus copying) are a general in-context mechanism that emerges in a phase change together with ICL ability.
- **Bietti et al., "Birth of a Transformer: A Memory Viewpoint"**, arXiv 2306.00802 (2023-06-01) [V]. On a toy task mixing global bigram statistics and in-context induction, **global (knowledge-like) bigrams are learned first and stored in weight matrices as associative memories**. Induction heads, which are in-context and reasoning-like, form later in attention.
- **Chen, Bruna, Bietti, "Distributional Associations vs In-Context Reasoning: A Study of Feed-forward and Attention Layers"**, arXiv 2406.03068 (ICLR 2025) [V]. In controlled settings **FFNs learn distributional associations** (e.g. bigrams) and **attention learns in-context reasoning**, and the paper explains the split through gradient noise. On Pythia, **truncating (low-rank) FFN components *improves* in-context reasoning** by removing the distributional "generic answer" prior [paper claim]. This links to LASER (Sharma et al., arXiv 2312.13558, 2023-12-21) [V], where rank reduction of late MLPs improves some reasoning benchmarks. This is the cleanest mechanistic evidence *for* a functional division, and it also shows that parametric knowledge can *interfere* with reasoning.
- **Nichani, Lee, Bietti, "Understanding Factual Recall in Transformers via Associative Memories"**, arXiv 2412.06538 (2024-12-09) [V]. Both self-attention and MLPs can act as associative memories, with storage capacity scaling linearly in parameter count. The paper also describes a sequential learning dynamic with a "hallucination" plateau.
- **Anthropic, "On the Biology of a Large Language Model"** (Lindsey et al., transformer-circuits.pub, 2025-03-27) [V-web]. Findings on Claude 3.5 Haiku:
  - *Multi-hop:* "the capital of the state containing Dallas" activates Dallas→**Texas**→Austin features in sequence, a real latent intermediate. **But a direct Dallas→"say Austin" shortcut edge runs in parallel.** Latent composition and memorized shortcut coexist and add together.
  - *Hallucination:* a default "can't answer" circuit is suppressed by "known entity" features. Hallucination happens when familiarity misfires. This is a mechanistic *interface* signal: the model has a (weak) internal "do I have this fact?" detector.
  - *Multilingual:* shared, language-independent concept features in the middle layers, with language-specific features at the edges. Concepts are shared across surface forms.
  - Caveat: attribution graphs cover only part of the computation (replacement-model error), so these are case studies, not proofs.

### 2.2 Disentangling knowledge from reasoning (2024–2026)
- **Merullo, Vatsavaya, Bushnaq, Lewis (Goodfire), "From Memorization to Reasoning in the Spectrum of Loss Curvature"**, arXiv 2510.24256 (2025-10-28) [V]. Weight components ordered by loss curvature separate memorization (sharp) from generalizing structure. Removing the low-curvature, idiosyncratic components suppresses memorized recitation, and **closed-book fact retrieval and arithmetic drop specifically and consistently, while *open-book* fact retrieval and general logical reasoning are preserved.** This is probably the **single most on-point 2025 result**. A weight-space decomposition exists in which "closed-book facts" can be removed while "use facts given in context" and "logic" survive. Caveat: arithmetic falls on the "memorized" side, which is a warning that some things we call reasoning are really lookup tables.
- **Ruis et al., "Procedural Knowledge in Pretraining Drives Reasoning in LLMs"**, arXiv 2411.12580 (2024-11-19; ICLR 2025) [V-web]. Influence functions over 5M pretraining docs (7B/35B Cohere models). Answers to *factual* questions depend on a few specific documents that contain the answer. *Reasoning* (simple math) outputs depend on many documents containing **procedural knowledge** (formulae, code), spread across queries. Facts and procedures therefore draw on data in different ways, which fits a split between an arbitrary-fact store and a procedure-bearing reasoner.
- **Hu et al., "LLMs Are Cross-Lingual Knowledge-Free Reasoners"**, arXiv 2406.16655 (2024-06-24) [V]. Knowledge-free reasoning transfers **almost perfectly across languages**, while knowledge retrieval transfers poorly. Hidden-state similarity and FFN-neuron overlap are higher for the reasoning tasks. The authors hypothesize that reasoning neurons are shared across languages while knowledge is stored per language.
- **Jin et al., "Disentangling Memory and Reasoning Ability in LLMs"**, arXiv 2411.13504 (2024-11-20; ACL 2025) [V]. Training with `<memory>`/`<reason>` special tokens that split CoT into recall and reasoning steps improves accuracy and error attribution. This is a *behavioural* decomposition, not a mechanistic one.
- **"Knowledge or Reasoning? A Close Look at How LLMs Think Across Domains"** (UCSC-VLAA), arXiv 2506.02126 (2025-06-02) [V]. Decomposes reasoning traces into knowledge correctness (Knowledge Index) and reasoning quality (InfoGain). R1-distilled general reasoning **does not transfer well to medicine**. SFT raises accuracy at the cost of reasoning quality. RL helps by *pruning incorrect knowledge* from reasoning paths. *Reading:* in knowledge-heavy domains the bottleneck is knowledge, and "reasoning skill" alone does not carry across domains.
- **Thapa et al., "Disentangling Reasoning and Knowledge in Medical LLMs"**, arXiv 2505.11462 (2025-05-16) [V]. Only 32.8% of biomedical QA items need complex reasoning. Models score consistently higher on knowledge subsets than on reasoning subsets. Useful as a benchmark-construction precedent.
- **Huan et al., "Does Math Reasoning Improve General LLM Capabilities?"**, arXiv 2507.00432 (2025-06-30) [V-web]. Across more than 20 reasoning-tuned models, math gains rarely transfer to other domains. RL-tuned models transfer better than SFT-tuned ones, because SFT causes representation drift.
- *Weaker or less-vetted 2025 claims (flag):*
  - Fartale et al., arXiv 2510.03366 (2025-10-03) [V]. Claims separable "recall circuits" and "reasoning circuits" in Qwen and LLaMA: ablating recall circuits cuts recall by up to 15% while leaving reasoning intact, and the reverse. Uses synthetic puzzles. Small effect sizes, and not widely replicated.
  - Yang, Gao, Wu, arXiv 2507.18178 (2025-07-24) [V]. Claims that "knowledge resides in lower layers, reasoning in higher layers", based on fast/slow prompting across 15 LLMs. The method is prompt-based, so the layer claim is indirect.
  - Shao & Wu, "Who Reasons in the LLMs?", arXiv 2505.20993 (2025-05-27) [V]. Claims reasoning is attributable mainly to the attention **o_proj**. Striking, but a single paper and unreplicated.

### 2.3 Architectural offloading experiments (the most direct evidence)
These are the experiments closest to Hemispheres. Each trains a backbone together with a separate, sparsely accessed knowledge store.

| Work | Store | Key (addressing) | Fusion | Knowledge-vs-reasoning evidence |
|---|---|---|---|---|
| **Engram, DeepSeek, "Conditional Memory via Scalable Lookup"**, arXiv 2601.07372 (2026-01-12; rev. 2026-07-12) [V] | Huge hashed N-gram embedding tables (27B-scale model) | **Hashed suffix N-grams of normalized token IDs** (surface form, not semantic) | Hidden state as query, retrieved vector as key/value, scalar gate plus conv. Inserted at layers 2 and 15 | Iso-param and iso-FLOP vs. MoE: MMLU +3.4, **BBH +5.0, ARC-C +3.7, HumanEval +3.0, MATH +2.4** (reasoning gains larger than knowledge gains). **Ablating Engram at inference: TriviaQA keeps ~29% of performance, factual benchmarks keep 29–44%, reading comprehension keeps 81–93%.** Mechanism: frees early layers from "static reconstruction", which effectively adds depth. U-shaped optimum at ~20–25% of sparse params in memory. |
| **Memory Layers at Scale**, Meta, arXiv 2412.09764 (2024-12-12) [V] | Product-key memory, up to 128B params | Learned product keys on hidden state | Replaces some FFNs | Beats dense models at 2× compute and matched MoE. Gains "especially pronounced for factual tasks". |
| **Pretraining with Hierarchical Memories**, Apple, arXiv 2510.02375 (v1 2025-09-29; ICLR 2026) [V] | 4.6B-param bank of FFN "memory blocks" | **Document-level**: Sentence-BERT embedding plus 4-level hierarchical k-means (external, frozen encoder) | Fetched block concatenated into SwiGLU FFN inner dim | 160M + 18M fetched is comparable to a model more than 2× larger. **Specific-knowledge tasks gain 4–5 pts; common-knowledge tasks gain little or lose slightly.** Blocking 1/16 of the bank drops atomic-number prediction from 70% to 20%. The "anchor" model explicitly keeps common knowledge plus reasoning. |
| **LMLM, "Pre-training Limited/Large Memory LMs with Internal and External Knowledge"**, Cornell, arXiv 2505.15962 (2025-05-21; v2 2025-07-02) [V] | External **symbolic** triple DB | Model-generated textual lookup (entity, relation) | Returned values inserted as tokens, **masked from loss** so they are not memorized | Small LMLMs competitive with larger LLMs. Facts become editable and verifiable, and unlearning becomes a DB delete. |
| **Co-LMLM**, arXiv 2607.07707 (2026-07-08) [V] | Text values with **continuous vector keys** | Model-generated vector query | Tokens | 360M model beats models trained on 40× more data on perplexity [paper claim]. |
| **MLP Memory**, arXiv 2508.01832 (2025-08-03) [V] | MLP pretrained to imitate kNN-LM retriever | Hidden state | Output-probability interpolation | +12.3% relative QA, fewer hallucinations, 2.5× faster than RAG. |
| **Memorization Sinks**, arXiv 2507.09937 (ICML 2025) [V-web] | Per-sequence neuron subsets | Sequence ID | Training-time routing | Memorization can be *isolated by design* and removed without hurting general ability. Gradient Routing (Cloud et al., arXiv 2410.04332) [V] is related. |

**Key reading.** Engram's ablation pattern is the clearest demonstration so far that, **when the architecture offers a lookup primitive, the network *chooses* to put factual knowledge there** and keeps comprehension and reasoning in the backbone. The *reasoning* benchmark gains suggest that knowledge storage was crowding out reasoning capacity. Caveat: Engram is keyed on local N-grams, so it stores "static patterns" (named entities, idioms, multi-token units), not relational facts about arbitrary entities in context. Whether that ablation pattern holds for a *semantically keyed* store is open.

---

## 3. Limits of latent composition

### 3.1 The evidence
- **Reversal curse**: Berglund et al., arXiv 2309.12288 (2023-09-21) [V]. A model trained on "A is B" does not learn "B is A". It does not happen in context.
- **Two-hop curse**: Balesni, Korbak, Evans, "Lessons from Studying Two-Hop Latent Reasoning", arXiv 2411.16353 (v1 2024-11-25 as "The Two-Hop Curse"; v4 2025-11-23) [V]. Llama-3-8B and GPT-4o fine-tuned on synthetic facts **fail to compose two synthetic facts without CoT**. They **succeed when one fact is synthetic and the other natural**, and also when both facts appear together in the same document or prompt. With CoT they succeed. The later versions re-frame the result: latent two-hop *exists*, but experimental artefacts can cause both spurious failures and spurious successes.
- **Yang et al., "Do LLMs Latently Perform Multi-Hop Reasoning?"**, arXiv 2402.16837 (2024-02-26) [V], and **"...without Exploiting Shortcuts?" (SOCRATES)**, arXiv 2411.16679 (ACL 2025) [V]. Once shortcut co-occurrence is removed, latent composition is highly relation-dependent: about 80% when the bridge entity is a country, about 5% when it is a year. A large gap remains between latent and CoT performance.
- **Biran et al., "Hopping Too Late"**, arXiv 2406.12775 (2024-06-18) [V]. The bridge entity is resolved in mid-to-late layers, so the layers needed to run the second hop are already used up. Back-patching later representations into earlier layers fixes a substantial fraction of failures.
- **Wang, Yue, Su, Sun, "Grokked Transformers are Implicit Reasoners"**, arXiv 2405.15071 (NeurIPS 2024) [V]. Implicit reasoning over parametric facts emerges only through **grokking**. **Comparison** generalizes OOD. **Composition does not**, because in the generalizing circuit atomic facts are stored at one depth and the second hop needs them at a later depth, so facts seen only as atomic facts never become available where the second hop runs. Parameter sharing across layers helps. They also show that a fully grokked small transformer beats GPT-4-Turbo and Gemini-1.5-Pro *with RAG / in-context facts* on a hard comparison task with a large search space. This is a caution that in-context reasoning is not automatically better for every kind of reasoning.
- **Karmim et al., "Multi-Hop Knowledge Composition is Bound by Pretraining Exposure"**, arXiv 2606.09338 (2026-06-08; EMNLP 2026) [V]. Composition fails **even at 97% one-hop accuracy**. Compositional training helps *only* entities that already appeared in compositional contexts during pretraining. So latent composability is a per-entity property learned from data, not a general skill.
- **Lin, Chen, Xu, "Identity Bridge"**, arXiv 2509.24653 (2025-09-29) [V]. Adding identity supervision on bridge tokens lets even a one-layer Emb-MLP model generalize two-hop OOD. The failure comes from missing supervision on the *bridge representation*, meaning the interface between hops.
- **Feng, Russell, Steinhardt, "Extractive Structures"**, arXiv 2412.04614 (2024-12-05) [V]. Out-of-context two-hop *does* work in pretrained models when **facts come before their implications in training**, through "extractive structures" (informative, upstream, downstream components). The effect depends on data ordering.
- **Treutlein et al., "Connecting the Dots"**, arXiv 2406.14546 (2024-06-20) [V]. LLMs can infer latent structure spread across training documents, a kind of out-of-context reasoning that retrieval of individual documents would not obviously reproduce. This points *against* complete externalization.
- **Lampinen et al., "On the generalization of LMs from in-context learning and finetuning"**, arXiv 2505.00661 (2025-05-01) [V]. In data-matched settings **ICL generalizes more flexibly than fine-tuning** (reversals, deductions). Adding in-context reasoning traces to fine-tuning data closes some of the gap.
- **Chaudhry, Thiagarajan, Lampinen, "Improving Latent Generalization Using Test-time Compute"**, arXiv 2604.01430 (2026-04-01) [V]. RL-trained "thinking" fixes many latent-generalization failures on in-weights knowledge and transfers to new knowledge. It does **not** enable direct reversal. Thinking models "remain well below" ICL on reversal.

### 3.2 Implications for Hemispheres
- **Composing parametric knowledge inside one forward pass is fragile, data-dependent and depth-limited.** Composing knowledge that is *present in context* (so attention can see both facts) is robust. So if knowledge is external, the reasoner *can* compose it, and probably composes it *better* than it composes its own parametric facts, **provided the retrieved knowledge appears as attended content (tokens or KV entries)** and multi-hop is done by iterating (CoT or recurrence).
- The "hopping too late" and grokked-composition results suggest a concrete architectural requirement: **the knowledge store must be accessible at every hop.** That means either re-entrant retrieval across reasoning steps (token level) or retrieval at many depths plus weight sharing or recurrence (representation level). Engram's fixed insertion at layers 2 and 15 would not solve two-hop by itself.
- There is a **cost** to externalizing. Latent single-pass shortcuts (Anthropic's Dallas→Austin edge) and "connecting the dots" generalization over the corpus may be lost, so more hops have to be explicit, which means more tokens and latency.

---

## 4. Knowledge-light reasoners

- **Karpathy's "cognitive core"** (X post, 2025-06-27 [V-web]: "a few billion param model that maximally sacrifices encyclopedic knowledge for capability"; repeated on the Dwarkesh podcast, Oct 2025 [V-web], where he speculated that about 1B params could suffice). This is an *opinion and forecast, not evidence*. Note that Karpathy also argues pretraining data would have to be cleaned or reshaped to get there, not just shrunk.
- **Phi line**: "Textbooks Are All You Need" (2306.11644) [M], Phi-4 (arXiv 2412.08905, Dec 2024) [V-web]. Strong reasoning for the size, but the tech report states the model is "fundamentally limited by its size" on factual knowledge. The base model answered about 90% of SimpleQA items *incorrectly*. Post-training shifted it toward "not attempted" (final incorrect rate about 15.8%). A clean existence proof that **reasoning scale and factual scale come apart**, and that the small reasoner then needs calibration ("I don't know") to avoid confabulating.
- **TinyStories** (arXiv 2305.07759, 2023-05-12) [V]. Models under 10M params trained on a narrow-vocabulary corpus write coherent stories with some reasoning. Evidence that linguistic and procedural competence does not require broad world knowledge, though the "reasoning" there is shallow.
- **Physics of LMs 2.1** (arXiv 2407.20311) [V]. GSM-style reasoning learned from purely synthetic data with no world knowledge.
- **Procedural / formal pre-pretraining**: Hu et al. arXiv 2502.19249 (2025-02-26) [V], where formal-language pre-pretraining gives the same loss with 33% fewer natural-language tokens and the resulting attention heads stay important. Jiang et al., "Procedural Pretraining", arXiv 2601.21725 (2026-01-29) [V], where 0.1–0.3% procedural data raises NIAH from 10% to 98% and cuts FLOPs, with **attention more important for structured domains and MLP for language**. Together these support the idea that reasoning-like inductive biases can be installed with little or no world knowledge, *mostly in attention*.
- **RL-trained search-while-reasoning**: Search-R1 (arXiv 2503.09516) [V-web], with +41% (Qwen2.5-7B) and +20% (3B) over RAG baselines; R1-Searcher (arXiv 2503.05592) [V-web]; ReSearch (arXiv 2503.19470, 2025-03-25) [V]; ReaLM-Retrieve (arXiv 2604.26649, 2026-04-29) [V], with step-level uncertainty-triggered retrieval, 71.2 F1 on MuSiQue at 1.8 calls per question. These show that **small reasoners learn a token-level retrieval interface through outcome-only RL** and do multi-hop by iterating. Caveats: benchmarks are Wikipedia multi-hop QA, and base models are ordinary knowledge-rich LMs, not knowledge-stripped ones. None of them tests a reasoner trained *without* the knowledge.
- **Reasoning models and hallucination**: "The Hallucination Tax of Reinforcement Finetuning" (arXiv 2505.13988; EMNLP Findings 2025) [V-web] finds RFT cuts refusal on unanswerable questions by more than 80%, and 10% unanswerable data restores it. Several 2025–26 works report lower factual accuracy for reasoning models than for their base models on fact-seeking tasks (cited in 2604.26649 [V]; primary sources not re-checked). *Implication:* a knowledge-light reasoner must be *trained* to notice missing knowledge and call the store. Anthropic's "known entity" circuit is the mechanism to preserve or supervise.
- **Transfer across knowledge domains**: mixed. Reasoning transfers across *languages* when knowledge-free (2406.16655). Reasoning transfers poorly from *math to medicine* (2506.02126, 2507.00432), probably because domain reasoning is shaped by domain knowledge (what counts as a relevant premise). So "domain-general reasoning skill" is real for formal and logical operations and weak for expert inference.

### Does reasoning need world knowledge to form concepts?
- **Evidence it does:** Ruis et al. (procedures are learned from documents, not abstractly). Karmim et al. 2026 (composability is per entity and depends on exposure). Anthropic's multilingual concept features (concepts are shared representational currency for both recall and reasoning). "Connecting the Dots" (inference over the whole corpus). Hierarchical-memories and Engram both *keep* common knowledge in the backbone, and no work shows a good reasoner with **zero** common knowledge.
- **Evidence it needs less than we store:** 2 bits/param makes long-tail facts expensive. Knowledge QA is capacity-hungry while code is data-hungry. Engram and hierarchical-memory ablations show long-tail facts leave while the backbone keeps working. Merullo et al. remove closed-book recall and keep open-book recall and logic.
- **Working synthesis:** there are three tiers. (a) *Conceptual/schematic knowledge* (types, relations, affordances, common facts) is entangled with reasoning and should stay in the reasoner. (b) *Long-tail, arbitrary instance facts* (entity attributes, dates, numbers) can be separated. (c) *Procedural knowledge* (how to do X) is mostly reasoner-side, though "instruction retrieval" (arXiv 2510.13935) and "Reasoning Memory" (arXiv 2604.01348) show some of it can also be retrieved [not deeply verified].

---

## 5. Evidence AGAINST clean separability (entanglement)

1. **Superposition and distributedness.** Fact lookup is diffuse and superposed (Nanda et al. 2023), additive and redundant (Chughtai et al. 2024), and localization does not predict editability (Hase et al. 2023). There is no "fact module" to cut out of a trained dense model.
2. **Attention stores facts too** (Wong 2025; Nichani et al. 2024; Geva 2023's attribute-extraction heads). Recall *uses* attention, and the recall circuit is learned alongside the facts (Zucchet et al. 2025).
3. **Reasoning uses knowledge-shaped representations.** Composability varies by relation and by entity (SOCRATES; Karmim 2026). Arithmetic behaves like memorized, idiosyncratic structure (Merullo 2025). Domain reasoning does not transfer (2506.02126).
4. **Unlearning knowledge often damages reasoning.** RMU and NPO unlearning degrades AIME, MATH-500 and GPQA scores in reasoning models, per Wang et al. arXiv 2506.12963 (2025-06-15) [V abstract; the specific degradation claim is from the paper body via search snippet, not re-checked]. This may reflect blunt unlearning methods more than intrinsic entanglement.
5. **Edits do not propagate** (MQuAKE, RippleEdits, CaKE). The reasoner consumes knowledge through learned pathways that a new weight-stored fact does not automatically join. This argues against *weight-level* hemispheres and is neutral or positive for *context-level* ones.
6. **Parametric knowledge supports inferences retrieval cannot easily reproduce**: "Connecting the Dots" style aggregation, and the grokked transformer beating RAG-equipped frontier LLMs on a large-search-space comparison task (Wang et al. 2024).
7. **Knowledge can hurt reasoning** (Chen/Bruna/Bietti; LASER). This supports separability, but it also shows that the backbone's *residual* knowledge will bias how it reads retrieved facts (knowledge-conflict problem). A Hemispheres reasoner needs to be trained to prefer the store.

---

## 6. Cognitive-science analogies (use carefully)

- **Complementary Learning Systems (CLS).** McClelland, McNaughton, O'Reilly, *Psychological Review* 102(3), 1995 [M]. Kumaran, Hassabis, McClelland, "What Learning Systems do Intelligent Agents Need? CLS Theory Updated", *Trends Cogn. Sci.* 2016, doi 10.1016/j.tics.2016.05.004 [V-web]. The hippocampus learns fast (episodic, pattern-separated). The neocortex learns slowly and in an interleaved way (structured, generalizing). Consolidation replays hippocampal content into cortex.
  - **Sun, Advani, Spruston, Saxe, Fitzgerald, "Organizing memories for generalization in CLS"**, *Nature Neuroscience* 2023 (doi 10.1038/s41593-023-01382-9) [V-web]. Unregulated consolidation causes overfitting, so **only the predictable components of memories should consolidate**. This is a principled answer to "what goes in the reasoner vs. the store": consolidate what generalizes (tier a/c), keep the unpredictable long tail (tier b) in the fast store.
  - *Mismatch:* in CLS the hippocampus is a *fast-learning episodic* store, not an encyclopedia. The Hemispheres store is closer to an externalized *semantic* memory. CLS supports the "fast store + slow generalizer + consolidation" pattern better than it supports "reasoner vs. facts".
- **Semantic vs. procedural vs. episodic memory** (Tulving; Squire) [M]. The Hemispheres split maps roughly onto semantic-fact memory (store) and procedural skill plus working memory (reasoner). In neuropsychology, semantic dementia (anterior temporal lobe "hub", Lambon Ralph et al., *Nat. Rev. Neurosci.* 2017 [M]) erodes concepts while some non-semantic reasoning is spared. The loss is of *concepts*, though, not just facts, which supports the view that conceptual knowledge is not a lookup table.
- **Language vs. thought dissociation**: Fedorenko, Piantadosi, Gibson, "Language is primarily a tool for communication rather than thought", *Nature* 630:575–586 (2024) [V-web]. People with global aphasia can still do algebra, chess and logic, and the **multiple-demand network** (fronto-parietal) supports fluid reasoning. Mahowald et al., "Dissociating language and thought in LLMs", arXiv 2301.06627 [M], applies this to LLMs. It is a good precedent for "a domain-general reasoning system dissociable from a knowledge/language system", but the dissociated pair is *language* vs. *reasoning*, not *knowledge* vs. *reasoning*.
- **Hemispheric lateralization.** Real lateralization exists (language is left-lateralized in most right-handers, some visuospatial and attentional functions lean right). But **"left-brained vs. right-brained" as a personality/cognitive-style split is not supported**: Nielsen et al., *PLoS ONE* 8(8): e71275 (2013) [V-web], found no whole-brain left or right dominance across 1,011 resting-state scans; lateralization is connection-specific. Popular "logic vs. knowledge/creativity" hemisphere stories come from over-extending split-brain findings (Sperry/Gazzaniga) [M]. McGilchrist's *The Master and His Emissary* (2009) [M] is a popular source to *avoid* citing as science. **Recommendation:** state in any write-up that "Hemispheres" is a metaphor for two cooperating, specialized systems joined by a high-bandwidth interface (the corpus callosum is a nice metaphor for the interface). It is not a claim about how brains divide knowledge from reasoning.

---

## 7. Synthesis

### 7.1 Strongest evidence FOR separability
1. **Engram ablation** (2601.07372). Removing the lookup memory drops factual benchmarks to 29–44% while reading comprehension keeps 81–93%. Adding it improves *reasoning* benchmarks more than knowledge benchmarks at iso-params and iso-FLOPs.
2. **Merullo et al.** (2510.24256). A curvature-based weight edit removes closed-book fact retrieval but keeps open-book retrieval and logical reasoning.
3. **Chen/Bruna/Bietti** (2406.03068), with Bietti 2023. FFN = distributional associations, attention = in-context reasoning, shown in controlled settings and in Pythia. Removing FFN associations can *help* reasoning.
4. **Scaling asymmetry**: knowledge is capacity-bound (2 bits/param, 2404.05405), knowledge QA is capacity-hungry while code is data-hungry (2503.10061), and Phi-4 reasons well while failing SimpleQA.
5. **Transfer asymmetry**: knowledge-free reasoning transfers across languages almost perfectly, knowledge does not (2406.16655). Procedural vs. factual influence patterns differ (2411.12580).
6. **Working memory-offload systems**: hierarchical memories (2510.02375), LMLM and Co-LMLM (2505.15962, 2607.07707), memory layers (2412.09764), Search-R1 family.

### 7.2 Strongest evidence AGAINST
1. Recall is superposed, redundant and attention-mediated. Localization does not predict editing (2301.04213, Nanda 2023, 2402.07321).
2. Every offload system keeps "common knowledge" in the backbone. Nobody has shown a competent reasoner with near-zero world knowledge. Conceptual knowledge *is* the reasoner's representational vocabulary (Anthropic shared concept features; semantic-dementia analogy).
3. Composability is per entity and exposure-dependent (2606.09338, 2411.16679). Reasoning over facts is shaped by how those facts were learned.
4. Domain reasoning does not transfer across knowledge domains (2506.02126, 2507.00432).
5. Unlearning knowledge damages reasoning (2506.12963, needs checking). Weight edits do not propagate into reasoning (MQuAKE, CaKE).
6. Parametric knowledge enables corpus-level inference (2406.14546) and some kinds of implicit reasoning that beat in-context facts (2405.15071).

### 7.3 What the interface must support
1. **Addressing, meaning what the lookup is keyed by.** Current systems use four options, in rising order of expressiveness:
   - (a) *surface N-grams* (Engram): cheap, deterministic, prefetchable, but cannot handle "the capital of the state containing Dallas";
   - (b) *document or context embedding from a frozen external encoder* (hierarchical memories): coarse, one fetch per document;
   - (c) *learned hidden-state keys* (memory layers, MLP memory, Co-LMLM vector queries);
   - (d) *model-generated symbolic queries* (LMLM triples, Search-R1 text queries): editable and auditable, but slow.
   The mechanistic literature (Geva 2023; Hernandez 2023) says the natural key is **(subject representation, relation)**, with relation acting roughly as a linear operator. So the interface should allow **relation-conditioned queries**, not only entity-conditioned ones.
2. **Composition across lookups, meaning re-entrancy.** The output of lookup *k* must be usable as the key of lookup *k+1*. Token-level loops (CoT plus tool) do this for free. Representation-level stores need either multiple insertion depths plus recurrence or weight sharing (grokked-composition and hopping-too-late results), or a latent-loop mechanism. The bridge-entity representation needs supervision (identity bridge, 2509.24653).
3. **Return format the reasoner can attend over.** In-context facts compose and reverse far better than in-weights facts (2505.00661, 2604.01430, the two-hop same-document result). Returning retrieved knowledge as *attendable* items (tokens or KV entries) is better supported than additive FFN injection. FFN injection (hierarchical memories, Engram gate) works for single-hop and local patterns but has no demonstrated multi-hop advantage.
4. **"Do I know this?" signal and conflict handling.** The reasoner needs a calibrated trigger for lookup (Anthropic known-entity circuit; step-level uncertainty in 2604.26649), and it has to prefer the store over residual parametric priors (Chen/Bietti interference; hallucination tax).
5. **Update semantics.** For the "update knowledge without retraining the reasoner" goal, (d)-type stores and frozen-key stores support clean edits. Learned-key stores (c) are coupled to the backbone's representation space, so the reasoner's representation drift after any fine-tune breaks the keys. Weight-level edits do not propagate (CaKE). The interface should be **key-stable**: keys defined in a frozen space, or a symbolic one.
6. **Granularity split.** Keep tier-a (conceptual/common) knowledge in the reasoner, and put tier-b (long-tail instance facts) in the store. Use the CLS "consolidate only what generalizes" criterion (Sun et al. 2023) to decide what migrates.

### 7.4 Falsifiable hypotheses for small-scale experiments
(Designed to run at roughly 10M–300M parameters on synthetic knowledge graphs plus small natural corpora.)

- **H1: Offloading frees reasoning capacity.** At fixed backbone params, a model trained with a separable fact store (e.g. an LMLM-style masked-value lookup over a synthetic biography KG, following Physics-of-LMs 3.1) scores **≥ the dense baseline** on held-out *knowledge-free* reasoning (synthetic logic/arithmetic, bAbI-style, Dyck), and the gap **widens as the number of facts grows past about 2 bits/param of backbone**. *Falsified if* the dense model matches or beats the separated one on reasoning once the fact load exceeds capacity.
- **H2: Double dissociation under ablation.** Zeroing the store drops closed-book fact QA to near chance while knowledge-free reasoning and open-book (in-context-fact) QA keep ≥ 90%. Conversely, ablating the reasoner's attention heads identified by path patching hurts open-book composition but not single-hop store lookup. *Falsified if* store ablation drops reasoning by more than about 10% (entanglement), or if no reasoner-side ablation spares single-hop lookup.
- **H3: Hot-swap without retraining.** Replace the store with a counterfactual KG (a new world with the same schema but different facts) *and no reasoner updates*. Accuracy on 1-hop and 2-hop questions over the new facts should match the original world within a few points. Compare against weight-edit baselines (ROME/MEMIT) on the dense model, where MQuAKE-style 2-hop propagation should fail. *Falsified if* hot-swap accuracy drops a lot, which would mean the reasoner learned facts-specific features (e.g. memorizing store outputs or key idiosyncrasies).
- **H4: Return format matters.** With the same store contents, a *token/KV-return* interface beats an *additive FFN-injection* interface on two-hop composition and reversal at equal compute, while the two are equal on single-hop. *Falsified if* FFN injection matches on 2-hop and reversal.
- **H5: Re-entrancy is necessary for multi-hop.** A single retrieval per forward pass, or per question, caps 2-hop accuracy near the product of two one-hop accuracies *only when hops are explicit*. Without iteration, 2-hop accuracy on novel compositions is near chance (replicating the two-hop curse). A re-entrant interface (retrieve → reason → retrieve) closes it. *Falsified if* single-shot latent 2-hop on OOD compositions exceeds about 50% without iteration.
- **H6: What migrates is predictable.** If the store is optional (a gate), then under training the reasoner keeps *high-frequency / predictable* facts internally and sends *long-tail / arbitrary* facts to the store (the Sun et al. 2023 prediction). Measure the internal-recall rate as a function of fact frequency and of "predictability" (derivable-from-schema vs. arbitrary). *Falsified if* the allocation is independent of frequency and predictability.
- **H7: Cross-domain transfer of the reasoner.** A reasoner trained with store A (domain A facts) and then paired with store B (a disjoint domain with the same relation types) generalizes, while a store B with *new relation types* fails. This separates "entity knowledge" (swappable) from "relational schema" (reasoner-side conceptual knowledge). *Falsified in the optimistic direction* if new relation types also work zero-shot, and in the pessimistic direction if even same-schema swaps fail.
- **H8: Knowledge-light reasoners hallucinate unless trained to abstain or call the store.** Measure the confabulation rate on unanswerable-in-store queries with and without training data that has missing entries (as in the hallucination-tax / SUM result). *Falsified if* confabulation stays high even with abstention training. That would suggest the "known entity" signal cannot be learned from the store interface.

**Minimum viable testbed.** Synthetic biography KG (Physics-of-LMs 3.x style, with augmentation), plus a synthetic reasoning suite (comparison, composition, inverse search), plus a small natural corpus (TinyStories or FineWeb-Edu subset) for language. Baselines: dense model, dense + RAG, LMLM-style symbolic store, memory-layer / hierarchical FFN store. All numbers reported both in-distribution and on OOD compositions (Wang et al. split).

---

## 8. Reference list (arXiv IDs and dates)

| ID | Date | Title (short) | Status |
|---|---|---|---|
| 2012.14913 | 2020-12-29 | FFN layers are key-value memories (Geva) | V |
| 2104.08696 | 2021-04-18 | Knowledge neurons (Dai) | V |
| 2202.05262 | 2022-02-10 | ROME (Meng) | V |
| 2209.11895 | 2022-09-24 | Induction heads (Olsson) | V |
| 2301.04213 | 2023-01-10 | Does localization inform editing? (Hase) | V |
| 2304.14767 | 2023-04-28 | Dissecting recall of factual associations (Geva) | V |
| 2305.07759 | 2023-05-12 | TinyStories | V |
| 2305.14795 | 2023-05-24 | MQuAKE | V |
| 2306.00802 | 2023-06-01 | Birth of a Transformer (Bietti) | V |
| 2307.12976 | 2023-07-24 | RippleEdits | V |
| 2308.09124 | 2023-08-17 | Linearity of relation decoding | V |
| 2309.12288 | 2023-09-21 | Reversal curse | V |
| 2309.14316 / 2309.14402 | 2023-09-25 | Physics of LMs 3.1 / 3.2 | V |
| 2312.13558 | 2023-12-21 | LASER | V |
| AF post | 2023-12 | Fact Finding (Nanda et al.) | V-web |
| 2402.07321 | 2024-02-11 | Summing up the facts | V |
| 2402.16837 | 2024-02-26 | Latent multi-hop reasoning (Yang) | V |
| 2404.05405 | 2024-04-08 | Physics of LMs 3.3, 2 bits/param | V |
| 2405.15071 | 2024-05-23 | Grokked transformers are implicit reasoners | V |
| 2406.03068 | 2024-06-05 | Distributional associations vs in-context reasoning | V |
| 2406.12775 | 2024-06-18 | Hopping too late | V |
| 2406.14546 | 2024-06-20 | Connecting the dots | V |
| 2406.16655 | 2024-06-24 | Cross-lingual knowledge-free reasoners | V |
| 2407.20311 | 2024-07-29 | Physics of LMs 2.1 | V |
| 2410.04332 | 2024-10-06 | Gradient routing | V |
| 2411.12580 | 2024-11-19 | Procedural knowledge drives reasoning | V-web |
| 2411.13504 | 2024-11-20 | Disentangling memory and reasoning (special tokens) | V |
| 2411.16353 | 2024-11-25 (v4 2025-11-23) | Two-hop curse / Lessons from two-hop latent reasoning | V |
| 2411.16679 | 2024-11-25 | SOCRATES latent multi-hop without shortcuts | V |
| 2412.04614 | 2024-12-05 | Extractive structures | V |
| 2412.06538 | 2024-12-09 | Factual recall via associative memories | V |
| 2412.08905 | 2024-12 | Phi-4 technical report | V-web |
| 2412.09764 | 2024-12-12 | Memory layers at scale | V |
| 2502.05076 | 2025-02-07 | Paying attention to facts | V |
| 2502.10835 | 2025-02-15 | Back attention | V |
| 2502.19249 | 2025-02-26 | Formal-language pre-pretraining | V |
| 2503.05592 / 2503.09516 / 2503.19470 | 2025-03 | R1-Searcher / Search-R1 / ReSearch | V-web / V-web / V |
| 2503.10061 | 2025-03 | Compute-optimal scaling: knowledge vs reasoning | V-web |
| 2503.16356 | 2025-03 | CaKE circuit-aware editing | V |
| 2503.21676 | 2025-03 | How LMs learn facts (Zucchet) | V-web |
| transformer-circuits | 2025-03-27 | Biology of an LLM (Anthropic) | V-web |
| 2505.00661 | 2025-05-01 | ICL vs finetuning generalization (Lampinen) | V |
| 2505.11462 | 2025-05-16 | Disentangling reasoning and knowledge in medical LLMs | V |
| 2505.13988 | 2025-05 | Hallucination tax of RFT | V-web |
| 2505.15962 | 2025-05-21 | LMLM | V |
| 2505.20993 | 2025-05-27 | Who reasons in LLMs? (o_proj) | V (flag) |
| 2506.02126 | 2025-06-02 | Knowledge or Reasoning? across domains | V |
| 2506.12963 | 2025-06-15 | Reasoning model unlearning | V (partial) |
| 2507.00432 | 2025-06-30 | Does math reasoning transfer? | V-web |
| 2507.09937 | 2025-07 | Memorization sinks | V-web |
| 2507.18178 | 2025-07-24 | Decoupling knowledge and reasoning (dual-system) | V (flag) |
| 2508.01832 | 2025-08-03 | MLP Memory | V |
| 2509.24653 | 2025-09-29 | Identity bridge two-hop | V |
| 2510.02375 | 2025-09-29 | Hierarchical memories (Apple; ICLR 2026) | V |
| 2510.03366 | 2025-10-03 | Recall vs reasoning circuits layer-wise | V (flag) |
| 2510.24256 | 2025-10-28 | Memorization to reasoning, loss curvature (Goodfire) | V |
| 2601.07372 | 2026-01-12 | Engram conditional memory (DeepSeek) | V |
| 2601.21725 | 2026-01-29 | Procedural pretraining | V |
| 2603.01097 | 2026-03-01 | LoRA as knowledge memory | V |
| 2604.01430 | 2026-04-01 | Latent generalization via test-time compute (Lampinen) | V |
| 2604.26649 | 2026-04-29 | ReaLM-Retrieve adaptive retrieval for LRMs | V |
| 2606.09338 | 2026-06-08 | Multi-hop composition bound by pretraining exposure | V |
| 2607.07707 | 2026-07-08 | Co-LMLM | V |
| Nature 630:575 | 2024 | Language primarily for communication (Fedorenko) | V-web |
| PLoS ONE e71275 | 2013 | Left vs right brain hypothesis (Nielsen) | V-web |
| TiCS 2016 | 2016-06 | CLS theory updated (Kumaran, Hassabis, McClelland) | V-web |
| Nat Neurosci 2023 | 2023-07-20 | Organizing memories for generalization in CLS (Sun, Saxe et al.) | V-web |
| Psych Rev 1995 | 1995 | CLS (McClelland, McNaughton, O'Reilly) | M |

**Known gaps and unverified items:** the Niu et al. ICLR 2024 knowledge-neuron critique; MEMIT (2210.07229); Lambon Ralph 2017; Mahowald 2301.06627; the primary sources for "reasoning models have lower factual accuracy"; and the specific unlearning-harms-reasoning numbers in 2506.12963. None of these were re-verified in this session.
