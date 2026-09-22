# Hemispheres literature survey: how knowledge is updated in monolithic LLMs today, and why it is hard

Survey date: 2026-09-22. Scope: the *baselines* Hemispheres must beat (parameter editing, fine-tuning/continued pretraining, continual learning, unlearning, and RAG).

**How the claims were checked**
- **[V]** means I checked the claim against the arXiv abstract, the HTML page or the proceedings page during this session.
- **[M]** means the claim comes from memory or secondary sources and was not re-checked this session. Treat these as "probably right, check before citing".
- **[U]** means the paper is recent, from a single or small group, and/or not peer reviewed. Its results have not been reproduced; use them as signals, not established facts.

---

## 0. TL;DR

1. **Parameter editing (ROME/MEMIT/AlphaEdit family) edits a string mapping, not "knowledge".**
   - It achieves ~95–100% "efficacy" on the edited prompt under the usual lenient evaluation.
   - It fails on:
     - multi-hop use of the edited fact (MQuAKE: MEMIT 96.2% edit-wise vs **7.0%** multi-hop);
     - the reverse direction (BAKE: ROME 99.7% forward vs **0.26%** reverse);
     - realistic autoregressive evaluation (Mirage: **96.8% → 38.5%**);
     - scale (collapse after hundreds to low thousands of sequential edits for classic methods; ~9% reliability at 1,000 sequential edits under WILD evaluation).
2. **2025–26 lifelong editors push sequential-edit counts much higher.** ENCORE reaches 10k edits; UltraEdit claims 1M–2M; NAS, StableEdit and BetaEdit add stabilizers. But:
   - these results are almost always measured on single-hop ZsRE/CounterFact-style recall;
   - independent reproductions and critiques show the protection is "bounded rather than unconditional" (AlphaEdit reproducibility study, 2026);
   - edits do not propagate into reasoning (ICML 2026 "theoretical limits" paper; NMI 2026 perspective).
3. **Edits and unlearning suppress; they do not erase.** The old fact stays recoverable:
   - by probing (up to 0.96 linear-probe accuracy after ROME, [U]);
   - by adversarial suffixes (85%+ in white-box settings, [U]);
   - by a few steps of benign fine-tuning (RMU on WMDP-Bio recovered to 62.4% with 10 unrelated samples; 88% of pre-unlearning accuracy recovered in Deeb & Roger).
   - "Remove a fact" is currently unsolved for any method that leaves the fact in shared weights.
4. **Fine-tuning new facts in is inefficient and harmful:**
   - it needs heavy paraphrase augmentation (Allen-Zhu 3.1; Ovadia 2023; EntiGraph; Active Reading);
   - it increases hallucination (Gekhman 2024);
   - it causes "priming" spillover (Sun et al. 2025);
   - it causes forgetting: NQ F1 drops 89% with full FT and 71% with LoRA, versus 11% with sparse memory FT (Meta 2025).
5. **RAG over a frozen model is the strongest practical baseline today.** WikiBigEdit: ~95% update accuracy for RAG versus collapse for ROME/MEMIT and 40–50% for WISE. SCR beats all parameter editors. But RAG has its own failure modes:
   - **conflicts.** Models override correct priors with bad context >60% of the time (ClashEval), yet stubbornly keep stale priors in code/API settings (only ~42–66% adoption of updated APIs);
   - **multi-hop and aggregation over retrieved facts.** Vanilla RAG manages 25% on multi-source temporal questions;
   - **long-context degradation** ("context rot");
   - **retrieval misses;**
   - **inference cost.** Inference time roughly doubled over WikiBigEdit;
   - **no true deletion.** The frozen model still knows the old fact.
6. **The open problem, stated precisely:** an update mechanism that is all of the following at once:
   - (a) **compositional.** The updated fact is used inside multi-hop and implicit reasoning, in both directions, without being in the prompt;
   - (b) **scalable** to ≥10^5–10^6 updates with bounded interference and cost;
   - (c) **truly subtractive**, so deletions survive probing, relearning and quantization;
   - (d) **authoritative over stale parametric knowledge**, but not credulous toward wrong inputs.
   Neither RAG nor any editor does all four today. Section 7 turns these into testable acceptance criteria for Hemispheres.

---

## 1. Parameter-level model editing

### 1.1 Classic methods (mechanism, headline numbers, arXiv)

| Method | arXiv / date / venue | Mechanism | Headline claim | Key known failure |
|---|---|---|---|---|
| **MEND** | 2110.11309, Oct 2021, ICLR'22 [M] | Hypernetwork transforms fine-tuning gradients into low-rank weight edits | Fast single edits on large models | Breaks down under many or sequential edits; needs meta-training |
| **ROME** | 2202.05262, Feb 2022, NeurIPS'22 [M] | Causal tracing locates mid-layer MLPs; rank-one closed-form update to the MLP "key→value" memory | ~100% efficacy on CounterFact single edits | Hase et al. 2301.04213: localization does *not* predict which layer is best to edit. Collapse under sequential edits (2401.07453). Reverse direction 0.26% (BAKE). |
| **SERAC** | 2206.06520, Jun 2022, ICML'22 [M] | Semi-parametric: explicit edit memory + scope classifier + counterfactual model; base model frozen | Strong on in-scope/out-of-scope separation | Needs a trained scope classifier and counterfactual model; reasoning over the edit happens in a small side model |
| **MEMIT** | 2210.07229, Oct 2022, ICLR'23 [M] | ROME generalized to batches, spread across several MLP layers via least squares | 10,000 *batched* edits on GPT-J | Batched ≠ sequential. MQuAKE multi-hop 7.0% vs 96.2% edit-wise [V]. Norm growth in sequential use (2502.01636 [V]). |
| **GRACE** | 2211.11031, Nov 2022, NeurIPS'23 [V] | Discrete key-value codebook at one layer with a deferral radius; weights untouched | "First method enabling thousands of sequential edits using only streaming errors" | Poor paraphrase generalization (codebook keys are local). The old fact is still linearly decodable from hidden states (0.79 probe accuracy, 2609.18985 [U]). ~9% reliability at 1k edits under WILD eval (2502.11177 [V]). |
| **PMET** | 2308.08742, Aug 2023, AAAI'24 [M] | Also optimizes attention (MHSA) hidden states; updates only the FFN | Improves over MEMIT on CounterFact/zsRE | Same locate-then-edit limitations |
| **WISE** | 2405.14768, May 2024, NeurIPS'24 [V] | Dual memory: main (pretrained) + side FFN memory; router; knowledge sharding then merging | Coins the "impossible triangle": reliability, generalization and locality can't all be achieved together | WikiBigEdit: 40–50% update accuracy, converging back to pre-update knowledge within 10k edits [V]. ~9% reliability at 1k edits under WILD eval [V]. |
| **AlphaEdit** | 2410.02355, Oct 2024, ICLR'25 Outstanding Paper [V] | Projects the MEMIT-style perturbation onto the null space of preserved-knowledge keys, so outputs on preserved keys are unchanged in theory | +36.7% average over locate-then-edit baselines with one line of code; stable to ~2–3k sequential edits in the paper [M for the exact count] | Reproducibility study 2606.26783 [V][U]: reproduces the original metrics. However, the benefit "does not generalize uniformly" to newer architectures, degrades at much higher edit counts, and harms downstream competence *and safety refusals*. Its protection is "bounded rather than unconditional". BetaEdit 2605.09285 [U]: the null space is only approximate, which leads to "knowledge leakage". |

Other reference points:
- **IKE**, in-context editing (2305.12740 [M]).
- **MeLLo**, memory plus iterative prompting (in the MQuAKE paper).
- **KnowEdit/EasyEdit**, the comprehensive study (2401.01286 [M]).
- Surveys: 2305.13172 (Yao et al.) [M] and 2310.16218 [V-search].
- Nature Machine Intelligence 2026 perspective by Zhang, Yao et al., "Towards principled knowledge editing methods for LLM reasoning" (s42256-026-01276-y) [V-abstract]. The same group maintains EasyEdit and here concedes that current KE "treat[s] LLMs as modular knowledge stores where facts can be edited independently, ignoring that knowledge forms an interconnected system".

### 1.2 2025–2026 lifelong/sequential editing successors

- **ENCORE**, "Lifelong Knowledge Editing requires Better Regularization". 2502.01636, Feb 2025, EMNLP'25 Findings [V].
  - Diagnosis:
    - locate-then-edit overfits the edited fact;
    - sequential edits cause **disproportionate norm growth** of the edited matrix.
  - Fix: most-probable early stopping plus a Frobenius-norm constraint.
  - Result: **10,000 sequential edits** without loss of downstream performance (Llama3-8B); 61% faster than MEMIT and 64% faster than AlphaEdit.
- **UltraEdit**. 2505.14679, May 2025, TMLR'26 [V].
  - Mechanism: training-, subject- and memory-free closed-form parameter shift with "lifelong normalization" of feature statistics.
  - Claims: **up to 1M edits** (the abstract variously says 1M and 2M) on a 7B model; 7× faster than prior SOTA; <1/3 the VRAM; runs on a 24GB GPU. Introduces UltraEditBench (>2M editing pairs).
  - *Skeptical note:* the evaluation is ZsRE/FEVER/WikiBigEdit-style single-hop recall. I found no evidence of multi-hop, reverse or free-form WILD evaluation at that scale.
- **RLEdit**, reinforced lifelong editing (hypernetwork trained with RL). 2502.05759 [V-search]. No numbers checked.
- **MEMOIR**. 2506.07899, NeurIPS'25 [V].
  - Mechanism: residual memory module; sample-dependent sparse masks (top-k activation "fingerprint") confine each edit to a distinct subset of columns; at inference, the query's mask is matched to stored masks.
  - Claims: "thousands of sequential edits with minimal forgetting".
  - *Relevance:* architecturally close to a separate knowledge store, but still keyed on activations of the frozen model.
- **RILKE**. 2511.20892, Nov 2025, ACL'26 oral [V-abstract].
  - Mechanism: representation-space interventions in low-dimensional subspaces plus a query-adaptive router.
  - Fact counts are not in the abstract.
- **Norm-Anchor Scaling (NAS)**. 2602.02543, Jan 2026 [V][U].
  - Formalizes sequential collapse as a **positive norm-feedback loop**: solved value vectors and edited MLP weights amplify each other, giving roughly exponential norm growth.
  - Rescaling value vectors to a reference norm extends the usable editing horizon by more than 4× and improves long-run performance by 72.2%.
- **StableEdit / "More Edits, More Stable"**. 2605.11836, May 2026, ICML'26 [V-abstract].
  - Lifelong normalization plus ridge regression gives asymptotically orthogonal, bounded-norm updates.
  - Earlier edits can *help* later ones.
- **BetaEdit**. 2605.09285, May 2026 [V-abstract][U]. History-aware null-space editing that addresses AlphaEdit's approximate-null-space leakage.
- **LightEdit**. 2604.19089, Apr 2026 [V-abstract][U]. Retrieval plus decoding-time suppression of the model's original-knowledge probabilities. Note that this is effectively a **RAG-plus-logit-steering** hybrid, not a weight edit.
- **Multi-hop-targeted editors**. All of these still evaluate on MQuAKE-like data:
  - **CaKE**, circuit-aware editing. 2503.16356, EMNLP'25 [V]. Layer-localized edits "inadequately integrate updated knowledge into reasoning pathways"; about 20% average improvement on MQuAKE.
  - **AcE**, attribution-controlled editing. 2510.07896, ICLR'26 [V-search]. Implicit subjects act as "query neurons" feeding "value neurons" across layers.
  - **Redundant Editing**. 2601.04600, Jan 2026 [V]. ROME's multi-hop failures are "hopping-too-late", generalization decay in deeper layers, and overfitting. Writing the same fact into several layers gives +15.5 pp (+96% relative) on 2-hop, at a cost to specificity.
  - **DecKER**. 2506.00536, ACL'25 Findings [V-abstract]. Decouples reasoning from knowledge injection *for in-context editing*. This is conceptually the closest prior art to Hemispheres' framing, at the prompt level.

**Pattern.** The 2025–26 advances mostly fix the *stability* axis (norm growth, interference, speed). They do not fix the *propagation* axis (multi-hop, reverse, implicit reasoning) or the *deletion* axis.

### 1.3 Known failure modes, with evidence

**(a) Ripple effects / logical consistency**
- **RippleEdits**. 2307.12976, TACL'24 [V].
  - Criteria: logical generalization, compositionality, subject aliasing, preservation and relation specificity.
  - Finding: "current methods fail to introduce consistent changes in the model's knowledge", and **a simple in-context editing baseline obtains the best scores**.

**(b) Multi-hop failure**
- **MQuAKE**. 2305.14795, EMNLP'23 [V]. Table on MQuAKE-CF-3k, GPT-J:

  | Method | Edit-wise | Multi-hop |
  |---|---|---|
  | FT | 46.9% | 1.5% |
  | MEND | 73.2% | 8.0% |
  | ROME | 88.3% | 7.4% |
  | MEMIT | 96.2% | 7.0% |

  - The unedited base model gets 40.5% multi-hop on the original facts.
  - MeLLo (memory plus prompting) reaches 14.2% on GPT-J and 30.9% on GPT-3.5.
- **MQuAKE-Remastered**. ICLR'25 spotlight [V]. **33–76% of MQuAKE's questions/labels were corrupted** (edit contamination, conflicting edits, duplicates). Many published multi-hop editing gains are therefore suspect. Use the Remastered version.
- **Shortcuts.** 2402.11900 [V-search]: models may answer multi-hop questions via memorized shortcuts, which editing leaves stale.

**(c) Reversal curse after editing**
- **BAKE / "Untying the Reversal Curse via Bidirectional LM Editing"**. 2310.10322 [V-search]. LLaMA-2 with ROME scores **99.70% forward vs 0.26% reverse**.
- **Bilinear representation**. 2509.21993, ICLR'26 [V].
  - Models trained on relational KGs with the right regularization develop bilinear relation structure (inverse = transpose, composition = product).
  - In such models, "updates to a single fact propagate correctly to its reverse and logically dependent relations".
  - Only shown in small models trained from scratch.
  - *Highly relevant design hint for a knowledge module:* make relational structure explicit.

**(d) Collapse under many edits / general-capability damage**
- Gupta et al., "Model Editing at Scale leads to Gradual and Catastrophic Forgetting". 2401.07453, ACL'24 Findings [V]. Two phases: gradual forgetting of earlier edits and abilities, then abrupt collapse.
- Yang et al., "Butterfly Effect of Model Editing". 2402.09656, ACL'24 Findings [V]. "Even a single edit can trigger model collapse". Proposes perplexity as a cheap collapse proxy; introduces HardEdit.
- Gu et al., "Model Editing Harms General Abilities of LLMs" (RECT). 2401.04700 [M].
- **"Revisiting Parameter-Based Knowledge Editing: Theoretical Limits and Empirical Evidence"**. 2606.00570, ICML'26 [V-abstract].
  - "Dimensional collapse hypothesis": localized edits propagate along fragile representation directions and cause **reasoning collapse**.
  - "Parameter-based editing methods consistently damage core LLM capabilities"; a simple retrieval baseline "substantially outperforms all parameter-editing methods".
- Pohl et al., "Towards a Principled Evaluation of Knowledge Editors". 2507.05937, ACL'25 workshop [V].
  - Editor rankings flip with metric and batch size.
  - String-match metrics produce false positives.
  - Damage to general capabilities is "a constant blind spot".

**(e) Evaluation over-optimism**
- **"The Mirage of Model Editing"**. 2502.11177, Feb 2025 (ACL'25) [V].
  - Introduces the QAEdit benchmark (NQ + TriviaQA + SimpleQA, 19,249 samples) and the WILD evaluation.
  - Single edits: **96.8% (standard eval) → 38.5% (WILD)**.
  - Sequential edits at 1,000 (batch size 1): **FT-M ~40.5%; ROME/MEMIT/GRACE/WISE average ~9.3%**.
  - Inflation causes:
    - teacher-forced decoding, which leaks the answer's content and length;
    - truncating output to the target length, which hides run-on errors;
    - context-free prompts identical to the edit prompt.
- He et al., "Benchmarking and Rethinking Knowledge Editing". 2505.18690, May 2025 [V].
  - Under realistic autoregressive evaluation, on instruction-tuned *and reasoning* models, "parameter-based editing methods perform poorly".
  - **SCR** (Selective Contextual Reasoning, 2503.05212 [V]: frozen model plus external text memory plus relevance selection) "consistently outperforms them across all settings".
- **HalluEditBench**. 2410.16251, ICLR'25 [V]. On >6,000 real hallucinations, editors show "limited performance" on generalization, portability and robustness.
- **Overfitting/EVOKE** ("Uncovering Overfitting in LLM Editing"). 2410.07819 [M]. Edited models over-predict the new object even in unrelated contexts.
- ThinkEval. 2506.01386 [V-search]. Measures knowledge leakage of edited facts in thought-based knowledge graphs; not read in detail.

**(f) Edits suppress rather than erase** (see also §4)
- "Exposing the Illusion of Erasure in Knowledge Editing". 2606.23276, Jun 2026 [V][U].
  - GCG-optimized suffixes recover pre-edit answers from ROME/MEMIT/MEND/FT-L-edited models.
  - Recovery rates: 85%+ context-guided in white-box settings; 15–48.5% blind; 75%+ cross-model transfer.
- "Suppressed, Not Erased". 2609.18985, Jul/Sep 2026 [V][U].
  - 50 CounterFact edits on GPT-2-XL, all 100% behaviorally successful.
  - A linear probe still decodes the *original* object: 0.96 after ROME, 0.86 after FT-L, 0.79 after GRACE.
  - *Caveats:* tiny scale. The GRACE result is partly expected, because layers below the codebook still compute the original fact. Even so, the point stands for Hemispheres: **if the reasoning module has computed the old fact, retrieval or override downstream does not delete it**.

**(g) Real-world, large-scale comparisons**
- **WikiBigEdit**. 2503.05683, ICML'25 [V].
  - Real Wikidata diffs, 500K+ QA pairs over 8 timesteps (Feb–Jul 2024).
  - Methods: ROME, R-ROME, MEMIT, WISE, RAG and LoRA+merging, on 5 models of ~7–8B.
  - ROME/R-ROME/MEMIT "rapidly degrade within the first few hundred updates, leading to model collapse". WISE falls back to pre-update knowledge within 10k edits.
  - **RAG reaches ~95% update accuracy**, 78.6% on rephrases and 66.2% on persona rewrites, with near-perfect locality. However:
    - RAG's inference time roughly doubles by the end;
    - RAG decays on older timesteps as memory clutters;
    - multi-hop gains are marginal (~3–5 pp).
  - LoRA plus model merging is competitive with specialized editors at zero inference overhead.
  - *Conclusion of the paper:* standard RAG and continual FT with merging beat specialized KE at scale.
- "RAG or Learning? … Continuous Knowledge Drift". 2604.05096, Apr 2026 [V][U].
  - 2024–25 event stream.
  - ROME scores **10.81% on historical questions**: updates overwrite both old knowledge and earlier updates.
  - Vanilla RAG manages 25.4% on multi-source temporal questions, versus 71.2% for a time-aware event-graph retrieval (Chronos) with GPT-4o.
  - ReAct-RAG reaches 93.4% on single-timestamp queries.

---

## 2. Knowledge injection by fine-tuning / continued pretraining vs RAG

- **Ovadia et al., "Fine-Tuning or Retrieval?"** 2312.05934, Dec 2023 (EMNLP'24) [V].
  - Unsupervised FT versus RAG on MMLU subsets and a new current-events set.
  - "RAG consistently outperforms [FT], both for existing knowledge … and entirely new knowledge."
  - LLMs struggle to learn new facts from unsupervised FT; exposure to **numerous paraphrases** helps.
- **Allen-Zhu & Li, Physics of LMs 3.1**. 2309.14316, Sep 2023 (ICML'24) [V].
  - On synthetic biographies, knowledge is *stored* but not *extractable* for QA (**~0% extraction accuracy**) unless pretraining data is augmented: multiple paraphrases, sentence shuffles, or QA mixed in during pretraining.
  - Once memorized without augmentation, instruction FT does not fix it.
  - Companion parts:
    - **3.2** (2309.14402 [M]): knowledge *manipulation*, e.g. inverse search and comparisons, fails without CoT. The reversal curse is structural.
    - **3.3** (2404.05405 [M]): capacity of ~2 bits/parameter for stored facts.
  - Implication: the capacity of a parametric knowledge store is bounded, which matters for "millions of facts".
- **Gekhman et al., "Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations?"** 2405.05904, EMNLP'24 [V].
  - Examples with new knowledge are learned "significantly slower" than known ones.
  - As they are eventually fit, **hallucination increases linearly**. Early stopping mitigates this.
  - The authors conclude that LLMs mostly acquire facts in pretraining and FT teaches them how to *use* those facts.
- **Mecklenburg et al., "Injecting New Knowledge into LLMs via SFT"**. 2404.00213 [V-search]. Fact-based synthetic QA generation beats token-based; needs many examples per fact.
- **Chang et al., "How Do LLMs Acquire Factual Knowledge During Pretraining?"** 2406.11813, NeurIPS'24 [V].
  - Each exposure raises a fact's probability a little, and later training dilutes the gain.
  - Forgetting follows a power law in steps.
  - Duplicated data leads to faster forgetting; larger batches make forgetting more robust.
- **Sun et al. (Google DeepMind), "How new data permeates LLM knowledge and how to dilute it"**. 2504.09522, 2025 (ICLR'26?) [V].
  - **"Priming"**: learning one new fact makes the model apply it in unrelated contexts.
  - Predictable from the keyword's pre-learning token probability, across PaLM-2, Gemma and Llama.
  - "Stepping-stone" augmentation and "ignore-k" update pruning cut priming by 50–95%.
  - This is the fine-tuning analogue of editing's locality failure.
- **Synthetic continued pretraining / EntiGraph**. Yang et al., 2409.07431, Sep 2024 (ICLR'25) [V].
  - A 1.3M-token corpus (QuALITY) is expanded via entity-relation synthesis into **455M synthetic tokens**. Llama-3-8B is then continually pretrained on it.
  - Closed-book QA accuracy scales **log-linearly** with synthetic tokens.
  - The gains compound with RAG.
  - Cost: roughly 350× token amplification to learn a small corpus.
- **Active Reading** (Meta FAIR). 2508.09494, Aug 2025 [V].
  - Model-generated study strategies produce the synthetic data.
  - Results: 66% on a Wikipedia-grounded SimpleQA subset (**+313% relative** over vanilla FT); 26% on FinanceBench (+160%).
  - **Meta WikiExpert-8B**, trained on **1T generated tokens**, beats 236B–405B models on factual QA.
  - Shows parametric injection *can* work at scale, but at pretraining-level compute. It is not a "cheap update".
- **SEAL, Self-Adapting LMs**. 2506.10943, Jun 2025 (MIT) [V].
  - The model RL-learns to write its own fine-tuning data ("implications") and then LoRA-updates on it.
  - SQuAD no-context QA, Qwen2.5-7B:

    | Setting | Base | Passage only | Base synthetic | GPT-4.1 synthetic | SEAL |
    |---|---|---|---|---|---|
    | Single passage | 32.7% | 33.5% | 39.7% | 46.3% | **47.0%** |
    | 200 passages (continued pretraining) | 32.7% | 32.2% | 41.0% | 39.4% | **43.8%** |

  - Limitations:
    - performance on earlier passages degrades under sequential self-edits (catastrophic forgetting);
    - ~30–45 s per self-edit evaluation.
  - *Skeptical note:* the absolute gains are modest (~+14 pp), and SQuAD is easy.
- **2026 successors**
  - **Self-Consolidating LMs (SCoL)**. 2605.07076, May 2026 [V-abstract][U]. Meta-RL chooses which layers to update, aligned with high-Fisher regions. Beats prompting, summarization, batch TTT and sequential FT on SQuAD and LongBench v2.
  - **GRIN**, mixed-policy RL for knowledge injection. 2608.25243, Aug 2026 [V-abstract][U]. "SFT memorizes injected facts but fails to generalize across paraphrasing and reasoning". RL with golden-answer injection generalizes better. Introduces the Blank and Counter (counterfactual-overwrite) benchmarks.
  - **Diffusion-inspired masked FT** for knowledge injection. 2510.09885 [V-search]. A demasking objective is more sample-efficient.
  - "From Style to Facts: Mapping the Boundaries of Knowledge Injection with Finetuning". 2503.05919 [V-search]. Not read in detail.
  - **Hypernetwork knowledge-injection scaling laws**. 2607.19604, Jul 2026 [V][U].
    - A hypernetwork generates LoRA adapters from facts for a frozen target (Wikidata5M, 1.25M examples).
    - Power laws hold. The fact-count exponent is only −0.080, so **adding facts yields weak returns**.
    - OOD generalization is steeper than with LoRA/full-FT.
  - **Decoupled MoE for Parametric Knowledge Injection (DMoE)**. 2606.14243, Jun 2026 [V-abstract][U].
    - Independently updatable expert modules at the final-layer FFN, plus an uncertainty-aware router decoupled from the base model.
    - Beats retrieval and adapter baselines. Deletion is not described.
    - **Close prior art to Hemispheres; flag for the architecture track.**
  - Adjacent prior art for the architecture track, not read in detail here:
    - Parametric RAG (2501.15915 [M]);
    - Dynamic/Parametric RAG (2506.06704 [V-search]);
    - Knowledge Modules via Deep Context Distillation (2503.08727 [M]);
    - Memory Decoder (2508.09874 [M]).

**Synthesis for §2.** Parametric injection needs roughly 10–1000× data amplification per fact to become *extractable* and *generalizable*. It raises hallucination and priming, forgets older material, and gains little per added fact. RAG wins on new facts per unit of cost, while parametric injection wins only when facts must be *used implicitly*: EntiGraph and RAG compound, and GRIN/CaKE report better reasoning use.

---

## 3. Continual learning and catastrophic forgetting

- **Surveys:**
  - Wu et al., "Continual Learning for LLMs: A Survey". 2402.01364 [V-search].
  - Shi/Wang et al., "Continual Learning of LLMs: A Comprehensive Survey". 2404.16789, ACM CSUR 2025 [V-search].
  - "LLM Evolution as an Industry-Scale Ecosystem: A Lifecycle Perspective on CL". 2606.24901 [V-search, not read].
  - Luo et al., empirical study of forgetting in continual instruction tuning. 2308.08747 [M].
- **LoRA Learns Less and Forgets Less**. Biderman et al., 2405.09673, TMLR'24 [V].
  - Code and math, instruction FT (~100K pairs) and continued pretraining (20B tokens).
  - LoRA substantially underperforms full FT on the target domain but forgets less than weight decay/dropout regularization.
  - Full-FT perturbations have **10–100× higher rank** than typical LoRA.
- **Sparse Memory Finetuning** (Meta FAIR). Lin et al., 2510.15103, Oct 2025 [V].
  - Uses memory-layer models (see Memory Layers at Scale, 2412.09764 [M]) and updates only memory slots that a new fact activates highly *relative to their pretraining usage* (TF-IDF-like).
  - After learning new facts, **NQ F1 drops 89% (full FT) and 71% (LoRA), versus 11% (sparse memory FT)**, at equal acquisition.
  - Follow-up: "Improving Sparse Memory Finetuning". 2604.05248, Apr 2026 [V][U]. Retrofits memory modules onto Qwen-2.5-0.5B with KL-based slot selection.
  - **This is the strongest existing evidence that a sparse, addressable knowledge store reduces interference. It is the single most important baseline for Hemispheres.**
- **RL's Razor**. Shenfeld, Pari & Agrawal, 2509.04259, Sep 2025 [V].
  - At matched new-task performance, on-policy RL forgets much less than SFT.
  - Forgetting is predicted by the KL(fine-tuned‖base) *on the new-task distribution*.
- **TiC-LM**. 2504.02107, Apr 2025 [V].
  - 114 Common Crawl dumps.
  - Continual pretraining with autoregressive meta-schedules plus fixed-ratio replay matches retraining from scratch at **2.6× less compute**.
  - Replay matters for general web data, less for specialized domains.
- **STOC**. "Towards Understanding Continual Factual Knowledge Acquisition". 2605.10640, ICML'26 [V].
  - Theory: regularization methods "merely adjust the convergence rate … without altering the inherent forgetting tendency", while replay changes the dynamics.
  - Proposes attention-guided generative replay.
- **Test-time training**
  - Theory and positive results exist, e.g. "Surprising Effectiveness of TTT for Few-Shot Learning" (2411.07279 [V-search]) and "In-Place TTT" (2604.06169 [V-search]).
  - "Beyond Perplexity". 2607.00368, Jul 2026 [V][U]. **One-step LoRA TTT lowers support/answer loss across three Qwen3 scales while free-form recall after removing the context stays at zero.** TTT "memory" claims evaluated by perplexity are overstated.
- **Knowledge-update benchmarks** (streams of changing facts):
  - CKL (2110.03215 [M]);
  - TemporalWiki (2204.14211 [V-search]; diffs of consecutive Wikipedia/Wikidata snapshots);
  - StreamingQA (2205.11388 [M]);
  - RealTimeQA (2207.13332 [M]);
  - EvolvingQA (2311.08106 [M]);
  - WikiBigEdit (2503.05683 [V]; auto-extensible, 500K+);
  - UltraEditBench (2505.14679 [V]; 2M pairs, synthetic-ish);
  - QAEdit (2502.11177 [V]);
  - evolveQA (2510.19172 [V-search]);
  - the continuous-drift benchmark of 2604.05096 [V];
  - TIDE (2608.08512 [V-search]);
  - Blank/Counter (2608.25243).

---

## 4. Deletion / unlearning, the "remove a fact" operation

**Benchmarks**
- **TOFU**. 2401.06121, Jan 2024 [V-search]. 200 fictitious author profiles × 20 QA pairs. Forget quality versus model utility.
- **WMDP**. 2403.03218 [M]. Hazardous bio/cyber/chem knowledge proxy; introduces RMU.
- **MUSE**. 2407.06460 [V-search]. Six-way evaluation on Harry Potter books and news (~6M tokens); verbatim versus knowledge sets.

**Evidence that unlearning is shallow**
- Łucki et al., "An Adversarial Perspective on Machine Unlearning for AI Safety". 2409.18025 [V].
  - Fine-tuning RMU on **10 unrelated samples** recovers WMDP-Bio accuracy to **62.4%**. Base Zephyr-7B scores ~64% and RMU ~30% [M for base and RMU numbers].
  - Orthogonalizing a single direction gives 64.7%.
  - Knowledge is obfuscated, not removed.
- Hu et al., "Unlearning or Obfuscating? … Benign Relearning". 2406.13356, ICLR'25 [V].
  - Relearning on public medical articles restores bioweapon knowledge.
  - Relearning on general Harry Potter wiki text restores verbatim text.
- Deeb & Roger, "Do Unlearning Methods Remove Information from LM Weights?" 2410.08827 [V]. Fine-tuning on *disjoint* facts from the same distribution recovers **88% of pre-unlearning accuracy**.
- Zhang et al., "Catastrophic Failure of LLM Unlearning via Quantization". 2410.16454 [V-search]. Low-bit quantization restores forgotten knowledge. Follow-ups: DurableUn 2605.02196 and quantization-permanent unlearning 2605.15138 [V-search].
- "Unlearning Isn't Deletion". 2505.16831, v3 May 2026 [V].
  - GA/NPO/RLabel forget-set drops of 60–80% often **fully reverse** after relearning.
  - Representation-level metrics (CKA, PCA shift, Fisher) show features are intact.
- "Unlearning vs. Obfuscation". 2505.02884 [V-search]. Many methods "remove" by *adding* distracting information.
- Lynch et al., "Eight Methods to Evaluate Robust Unlearning". 2402.16835 [M].
- Cooper et al., "Machine Unlearning Doesn't Do What You Think". 2412.06966 [M]. A policy- and legal-oriented critique.

**Partial remedy**
- **UNDO, "Distillation Robustifies Unlearning"**. 2506.06278, NeurIPS'25 [V].
  - Distill an unlearned model into a *noised or random-init* student. The behavior transfers but latent capabilities do not.
  - Matches retrain-from-scratch robustness at 60–80% of the compute.
  - Implication: **the only robust deletion known today is re-deriving the weights without the knowledge**.

**Editing as deletion**
- "Editing as Unlearning". 2505.19855 [V-search]. WISE and AlphaEdit are decent unlearning baselines for pretrained knowledge.
- But editing leaves recoverable traces: 2606.23276 and 2609.18985 (see §1.3f).

**Implication for Hemispheres.** Deletion is only verifiable if the fact provably lives *only* in the removable store. That requires showing the reasoning component cannot produce the fact:
- by probing;
- under few-shot relearning attacks (a Deeb–Roger style held-out-facts test);
- after quantization;
- with the store detached.

That is a strong and testable claim. Nobody has demonstrated it for a capable model.

---

## 5. RAG failure modes (the other baseline)

**Knowledge conflicts**
- Survey: Xu et al., "Knowledge Conflicts for LLMs: A Survey". 2403.08319, EMNLP'24 [V-search].
- Longpre et al., entity-based knowledge conflicts. 2102.08501 [M]. Early models over-relied on parametric memory.
- **Xie et al., "Adaptive Chameleon or Stubborn Sloth"**. 2305.13300, ICLR'24 spotlight [V].
  - LLMs *accept* coherent, convincing counter-memory evidence.
  - With mixed evidence they show **confirmation bias** toward their parametric memory.
- **ClashEval**. Wu, Wu & Zou, 2404.10198, NeurIPS'24 D&B [V].
  - 1,200+ questions over six domains with perturbed context.
  - Six top LLMs including GPT-4o **adopt incorrect retrieved content, overriding correct priors, >60% of the time**.
  - Adoption falls as the perturbation becomes more implausible, and rises as the model's prior confidence falls.
- **"When LLMs Lag Behind: Knowledge Conflicts from Evolving APIs"**. 2604.09515, Apr 2026 [V].
  - 270 real API updates across 8 Python libraries and 11 models.
  - Executable-code rate is 42.55% without full documentation and ~66% with structured documentation. Models "still struggle to override stale parametric knowledge".
  - This is the stubborn-sloth failure in a practical setting.
- "Three Regimes of Context-Parametric Conflict". 2605.11574, May 2026 [V][U; single author].
  - Context-following swings from ~100% down to 6–71% depending on task framing.
  - Parametric certainty predicts resistance.
- Mitigations: decoding-based (e.g. 2605.12185 [V-search]), explicit conflict resolution (2606.20245 [V-search]), and "situated faithfulness" (2410.14675 [V-search]).
- *Net:* RAG is **simultaneously too credulous (ClashEval) and too stubborn (API evolution, confirmation bias)**, and which one happens depends on prior strength and framing. That makes it an unreliable authority mechanism for updates.

**Sufficiency and abstention**
- Joren et al. (Google), "Sufficient Context". 2411.06037, ICLR'25 [V].
  - Frontier models answer well with sufficient context but **hallucinate rather than abstain when context is insufficient**.
  - Smaller models hallucinate or abstain even when context *is* sufficient.
  - Selective generation gains 2–10%.
- Cuconasu et al., "The Power of Noise". 2401.14887 [M]. Retrieval noise and distractors affect accuracy non-monotonically.

**Long-context degradation**
- Liu et al., "Lost in the Middle". 2307.03172 [M]. U-shaped accuracy by evidence position.
- RULER (2404.06654 [M]) and NoLiMa (2502.05167 [M]; large drops by 32K when lexical overlap is removed).
- Chroma, "Context Rot". July 2025 technical report (not arXiv) [V-search]. **All 18 frontier models tested** (GPT-4.1, Claude 4, Gemini 2.5, Qwen3, …) degrade as input length grows, even on trivial tasks and well below the window limit.

**Multi-hop and aggregation**
- WikiBigEdit: RAG gives only ~3–5 pp multi-hop gain, and first-hop retrieval accuracy is below 50%.
- 2604.05096: vanilla RAG manages 25% on multi-source temporal questions.
- MQuAKE MeLLo: 30.9% even with GPT-3.5.

**Scaling and cost**
- WikiBigEdit: RAG's inference time roughly doubles, and accuracy decays on older timesteps as the store grows (retrieval clutter).

**No deletion of parametric knowledge**
- RAG masks the stale fact only when retrieval fires *and* the model defers. The stale fact resurfaces when retrieval misses or the question is implicit or multi-hop.

---

## 6. What exactly is unsolved

1. **Propagation / compositional use of updates.**
   - Every parameter editor fails when the edited fact is an *intermediate* hop: MQuAKE 7%, Mirage, the ICML'26 theoretical limits paper, the NMI'26 perspective.
   - Reverse-direction use also fails: BAKE 0.26%.
   - Multi-hop-specific fixes (CaKE, AcE, Redundant Editing) give +15–20 pp on benchmarks that were themselves 33–76% corrupted.
   - RAG helps multi-hop only marginally, unless the model explicitly decomposes the question (MeLLo, DecKER, event graphs).
   - *Nobody has shown updates that are used implicitly, i.e. without verbalized retrieval, in multi-step reasoning at the level of the model's original knowledge.* The unedited base's multi-hop score on original facts (40.5% on MQuAKE with GPT-J) is the natural ceiling.
2. **Scaling to 10^5–10^6+ updates with bounded interference.**
   - Stability fixes reach 10k (ENCORE) and claim 1–2M (UltraEdit), but only on single-hop recall.
   - Under WILD evaluation, classic editors are ~9% at 1k.
   - RAG scales in storage but degrades with clutter and doubles inference cost.
   - Parametric capacity is bounded at ~2 bits/param (Physics 3.3) [M].
3. **Verifiable deletion.**
   - No method except retrain or distill-from-scratch survives relearning, probing and quantization attacks.
   - Editing and unlearning are suppression.
4. **Authority without credulity.**
   - The system must prefer the *updated* knowledge over stale parametric priors (fixing stubborn-sloth/API lag), without adopting arbitrary wrong context (fixing ClashEval credulity).
   - In a monolith this is an unresolved arbitration problem, because the stale fact and the new fact live in the same substrate with no provenance.
5. **Preserving general and reasoning capability under updates.**
   - Editing damages capabilities (Butterfly Effect; the ICML'26 theoretical limits paper; the AlphaEdit reproduction harming safety refusals).
   - FT forgets and primes.
   - Sparse memory FT is the best current evidence of low interference (11% NQ drop), but it measures held-out QA, not reasoning.
6. **Evaluation itself.**
   - Teacher forcing, truncation, string-match false positives and corrupted benchmarks (MQuAKE) inflated a decade-scale literature.
   - Any claim by Hemispheres must use WILD-style free generation with LLM or exact-match judging, MQuAKE-Remastered, and real update streams (WikiBigEdit).

## 7. Concrete properties Hemispheres must demonstrate

### (a) To beat RAG over a frozen model
The strongest practical baseline is SCR/ReAct-RAG with a strong retriever.

1. **Implicit multi-hop use of updated facts beats RAG at matched base capability.**
   - Benchmarks: MQuAKE-Remastered, the multi-hop portion of WikiBigEdit, and the multi-source temporal questions of 2604.05096.
   - Report the result with no retrieved text in the prompt, or at least with the same token budget.
   - *Target:* multi-hop accuracy on updated facts ≈ multi-hop accuracy on original facts. There should be no "update penalty".
2. **No conflict pathology.**
   - On a ClashEval-style set, the updated store should override stale priors ~100% of the time (the API-evolution setting is the key test).
   - At the same time, adversarial or irrelevant *prompt* text must not override the store beyond an agreed rate.
   - Needs explicit provenance and priority semantics, as opposed to soft attention competition.
3. **Flat cost and accuracy as the store grows.**
   - Inference latency and accuracy on old updates should stay flat from 10^3 to 10^6 facts. RAG, by contrast, roughly doubles latency and decays on old timesteps.
   - Report this over WikiBigEdit's temporal splits.
4. **Robustness to retrieval misses.** Paraphrase, persona and alias queries (RAG: 78.6% and 66.2% on WikiBigEdit) should reach at least edit-level accuracy.
5. **Real deletion.** RAG cannot remove what the frozen model already knows (see 7c).

### (b) To beat MEMIT/AlphaEdit-style editing (and ENCORE/UltraEdit/MEMOIR)
1. **Sequential scale with WILD evaluation.**
   - ≥10^5 sequential single edits (batch size 1) with QAEdit/WILD-style free generation.
   - Reliability and generalization should stay above the RAG baseline.
   - Compare with ~9% for classic editors at 1k and ~40% for FT-M.
2. **Zero drift in the reasoning module** (the core architectural claim).
   - General benchmarks (MMLU, GSM8K/MATH, code, instruction following, safety refusals) should be *bit-for-bit or statistically unchanged* after N updates.
   - Also report perplexity drift (the Butterfly Effect proxy).
   - Include the safety-refusal check, because the AlphaEdit reproduction found degradation there.
3. **Ripple and reverse consistency.**
   - RippleEdits scores on all 5 criteria; BAKE reverse accuracy well above the ~0% of editors.
   - Show that a single store update produces correct inverse and composed answers.
4. **Locality without the impossible triangle.** Simultaneously high reliability, generalization and locality under WISE's evaluation, *and* under free-form generation.
5. **Update cost.** Report wall-clock and memory per update against UltraEdit (7B on 24GB), ENCORE and sparse memory FT.

### (c) Shared deletion and faithfulness criteria

For the *remove* operation to count, it must pass all of these:
1. Linear-probe decodability of the deleted fact returns to chance (2609.18985 protocol).
2. GCG suffix extraction fails (2606.23276 protocol).
3. Relearning attack: fine-tuning on held-out same-distribution facts recovers ≤ the recovery rate of a never-trained control (Deeb–Roger protocol).
4. Quantization does not restore the fact.
5. With the knowledge store detached, the reasoning module's closed-book accuracy on the knowledge domain is near chance.
   - This is the hardest and most distinctive claim, because pretraining the reasoning module on natural text inevitably teaches facts.
   - How Hemispheres trains the reasoning module to be *knowledge-poor but reasoning-capable* is likely the crux. Allen-Zhu 3.1/3.2 suggest extraction and manipulation skills are learned *jointly* with the knowledge format.

### (d) Controls Hemispheres must include
- **Same-compute monolith.** A dense model of equal total parameters and FLOPs, to show that the separation doesn't cost capability.
- **Sparse memory FT** (2510.15103) on the same backbone. This is the closest "knowledge-in-addressable-memory" baseline.
- **DMoE** (2606.14243), MEMOIR (2506.07899) and SCR (2503.05212) as architectural and near-architectural comparators.
- Evaluation on **MQuAKE-Remastered, WikiBigEdit, QAEdit/WILD, RippleEdits, BAKE, ClashEval, TOFU/WMDP-style relearning**.

---

## 8. Things I could not verify, or that deserve skepticism
- **UltraEdit's "1M/2M edits"**: the abstract's wording varies between versions. No multi-hop or WILD evaluation was seen at that scale.
- **AlphaEdit's exact sequential-edit ceiling**: the original paper's count (~2–3k) is from memory. The 2026 reproduction (2606.26783) is a student-style reproducibility report, not peer-reviewed as far as I saw.
- **Single-author or small-scale 2026 preprints**: 2605.11574 (three regimes), 2609.18985 (probe traces; only 50 edits on GPT-2-XL), 2606.23276 (GCG extraction), 2604.05096 (small benchmark: 513 quadruples), 2607.00368 (TTT), 2607.19604 (hypernetwork scaling). All are directionally consistent with older, better-vetted results, but individually weak.
- **Łucki et al. baselines**: the base and RMU WMDP-Bio numbers (~64% and ~30%) are from memory.
- **Physics of LMs 3.2/3.3 specifics** (the ~2 bits/param figure, inverse search): from memory.
- **Unread papers**: I did not read DMoE's full results or deletion semantics (abstract only). They should be read by whoever owns the architecture track.
- **Venue labels**: some are from GitHub READMEs or search snippets rather than proceedings pages.
