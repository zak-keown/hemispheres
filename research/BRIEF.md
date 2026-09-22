# Hemispheres: Research Brief

*2026-09-22. This brief combines four literature tracks in `research/tracks/`. Those files have the full citations and mark every claim as checked this session, from memory, or unverified. Spot-check any number here against its track before quoting it.*

---

## 1. The idea, restated precisely

> **Hemispheres** is an LLM with two parts. The **reasoner** is a network that is light on knowledge and does composition, inference and language. The **store** holds facts and supports adding, changing and deleting individual facts without any gradient update to the reasoner. The reasoner uses facts from the store as fluently as a normal LLM uses facts it memorized, including in multi-step reasoning.

The last sentence carries the weight. Putting facts outside the weights is old; RAG has done it since 2020. The hard part is making the reasoner **use an updated fact as well as a memorized one**, especially when that fact is one step in a multi-step chain. The whole project turns on that.

## 2. Verdict

- **The general idea is not novel.** Almost every place a transformer can take in knowledge has been tried: the input text (RAG), attention (RETRO, KBLaM, Memory³), the feed-forward layers (memory layers, Engram, Apple's hierarchical memories), the output distribution (kNN-LM) and weights updated at test time (Titans).
- **The premise is now empirically credible for long-tail facts.** 2025–26 work shows it:
  - DeepSeek's Engram: switching its memory off leaves TriviaQA at 29% of its score, while reading comprehension keeps 81–93%.
  - Apple's hierarchical memories: a 160M model plus memory matches a dense model more than twice its size.
  - Cornell's LMLM and Co-LMLM: a core trained *not* to memorize facts, with deletion that passes the TOFU unlearning test.
- **Nobody has built the full combination.** No system combines:
  1. a core trained to lack the facts, at 1B+ scale, and shown to reason as well as a matched dense model;
  2. per-fact writes and deletes with no gradient step, whose keys stay valid if the core is fine-tuned;
  3. updated facts carrying through multi-step reasoning, near the level of putting the fact straight into the prompt;
  4. verified deletion with no leakage from the core.

  Each piece exists somewhere; nobody has all four together, and nobody has measured (3) properly. That is the gap.
- **The closest competitor is Co-LMLM** (arXiv 2607.07707, July 2026). The model emits a `<FACT>` token, its hidden state queries a store of 2.2B free-text facts, and the retrieved text is spliced into the input and excluded from the training loss, so the core never learns to memorize it. It works, but only at 360M parameters. It has no multi-hop or reasoning-retention evaluation. Its index is tied to the core's hidden states, so fine-tuning the core breaks it (the authors call this an open problem). Hemispheres must be clearly positioned against it.

## 3. What the evidence says about separability

**For separation** (details in track 03, §7.1)
- Engram's double dissociation: remove the store and fact recall collapses while reading comprehension mostly survives. At equal parameters, Engram also *improved reasoning* (BBH +5.0, MATH +2.4).
- Removing memorization-linked weight components hurts closed-book fact recall but spares logical reasoning and recall of facts given in the prompt (Merullo et al. / Goodfire, 2510.24256).
- In controlled settings, feed-forward layers learn associations and attention does in-context reasoning. Trimming the feed-forward part can *improve* reasoning (2406.03068).
- Scaling differs: knowledge grows with parameters (about 2 bits per parameter) and reasoning skill grows with data.
- Transfer differs: knowledge-free reasoning transfers across languages; memorized knowledge doesn't.

**Against separation** (track 03, §7.2)
- Fact recall inside trained models is spread out and redundant, and uses attention as well as feed-forward layers. Where interpretability tools say a fact lives doesn't predict where editing it works.
- **Every working offload system keeps common and conceptual knowledge in the core.** No capable reasoner with near-zero world knowledge exists. "Concepts are knowledge": relation types, schemas and common sense probably have to stay in the reasoner.
- Combining two facts in a single forward pass (no written-out steps) is unreliable even in monolithic models: about 80% for some relations, about 5% for others. It fails even at 97% single-fact accuracy (the "two-hop curse"). An external store doesn't fix this by itself.

**What follows:** separate by **type of knowledge**, not all-or-nothing. The reasoner keeps schemas, relation types and common concepts. The store holds arbitrary long-tail instance facts. The complementary-learning-systems rule applies: only knowledge that generalizes gets consolidated into the slow learner (Sun, Saxe et al. 2023).

## 4. The incumbents and why they fall short

| Approach | What works | What breaks (headline evidence) |
|---|---|---|
| **Weight editing** (ROME, MEMIT, AlphaEdit, …) | Recall of the edited fact itself | MEMIT gets 96.2% on the edited question but **7.0% on multi-hop** (MQuAKE). ROME gets 99.7% forward but 0.26% in reverse. Under realistic free generation, single-edit success drops from 96.8% to 38.5% ("Mirage of Model Editing"). Collapses within a few hundred updates at real-world scale (WikiBigEdit). |
| **Fine-tuning new facts in** | Works with heavy paraphrasing | Raises hallucination (Gekhman 2024). Leaks new facts into unrelated contexts. Very expensive: EntiGraph needed ~455M synthetic tokens to learn a 1.3M-token corpus. Plain-statement facts keep 1% accuracy after 20 sequential writes (Baseten 2607.11020). |
| **Unlearning** | Suppresses output | Knowledge is still recoverable: 10 unrelated fine-tuning samples bring WMDP back to 62%. Quantization and linear probes also recover it. Only rebuilding the weights actually removes it. |
| **RAG over a frozen model** | ~95% update accuracy, easy deletion from the store | Conflicts with what the model already believes: it follows changed facts only 20–52% of the time, yet trusts *wrong* retrieved text >60% of the time. Weak multi-hop gains (+3–5 points on WikiBigEdit). Quality degrades as input grows. **Cannot delete what the frozen model already knows.** |

RAG is the real incumbent. The editing literature is converging on "retrieval beats editing" (ICML'26, 2606.00570). So Hemispheres must beat **RAG on correctness**:
- facts carry through multi-step reasoning;
- changed facts actually win over old beliefs;
- deleted facts don't leak;
- accuracy holds as the store grows.

It should also be clearly competitive on cost. Beating it on QA accuracy alone is not enough.

## 5. The design space: five decisions

Each decision has a leaning from the literature and an open question.

1. **What the store is keyed by.** Options run from surface N-grams (Engram) to document embeddings (Apple), learned hidden states (Co-LMLM) and model-generated symbolic queries (LMLM, Search-R1).
   - *Leaning:* **subject + relation** keys (relation decoding is roughly a linear map on the subject representation), defined in a **frozen or symbolic space** so the core can be fine-tuned without re-indexing. This is Co-LMLM's open problem, and fixing it is a concrete contribution.
2. **How retrieved knowledge enters the reasoner.** Options are added into the feed-forward/residual path, placed in attention as key/value entries, or inserted as text into the input.
   - *Leaning:* **items the reasoner can attend over** (tokens or KV entries), read in **early**. Facts in context combine and reverse far better than facts in weights. Output-end designs (kNN-LM, Memory Decoder, DMoE) fail multi-fact reasoning even with perfect retrieval.
3. **Repeated lookup (re-entrancy).** The answer from lookup *k* must become the key for lookup *k+1*.
   - **This is the central fork**:
     - **(A) Token-level lookup:** the model writes out a lookup, like Co-LMLM or Search-R1. Proven to work, auditable and easy to edit. The novelty must then come from scale, stable keys, the knowledge-light core and evaluation.
     - **(B) Latent lookup:** repeated retrieval inside the forward pass, without writing out steps, e.g. a looped or recurrent block that re-queries the store. This is more novel and directly attacks the two-hop curse, but it's riskier.
   - The first experiment (§7) should test A against B head-to-head. It's cheap.
4. **Training the core to be knowledge-light.** LMLM-style loss masking on store-supplied values, or randomizing entity names during pretraining so memorizing entities doesn't pay. Without this the core keeps stale beliefs that override the store (RAG's conflict problem) and leaks deleted facts.
5. **"Do I know this?" and abstention.** A knowledge-light reasoner will make things up unless it's trained on queries the store can't answer. Anthropic found a "known entity" circuit, so this signal can be learned; it must be trained explicitly.

## 6. The strongest case against Hemispheres

- **"Just use RAG with a strong reasoning model."** Test-time reasoning plus search (Search-R1 and its descendants) already chains lookups in text, and long contexts keep improving. If Hemispheres only matches that, it's not worth it. **Answer:** win on update correctness (overriding old beliefs, leakage, reasoning over changed facts), on cost per query, and possibly on latent multi-hop.
- **Starving the core of knowledge may starve its reasoning.** Engram's gains on BBH and MATH suggest stored patterns help reasoning. Some "knowledge" is part of reasoning. The line between what stays and what moves has to be found experimentally (H6 below).
- **Latent composition might just be hard.** If the two-hop curse is architectural, a latent store won't fix it. Fork A (token-level lookup) is then the only route, and the contribution is more modest.
- **The name.** Left-brain/right-brain lateralization is largely a myth: no whole-brain left/right dominance was found in 1,011 scans. Keep "Hemispheres" as a metaphor, and cite complementary learning systems (hippocampus/neocortex) as the real analogy.

## 7. Recommended plan

### Step 0: throughput benchmark (10 minutes)
No published M5 Max training throughput exists; the estimate is 10–30k tokens/s for a 124M model in MLX. Measure it first; it sets the size of every later experiment.

### Step 1: synthetic-world toy (laptop, days, MLX)
- **World generator:** synthetic biographies (bioS-style) plus relational edges, e.g. person → employer → HQ city → country, plus a mentor link. Several worlds share a schema but use disjoint entities; about 5–20k entities each. The generator is simple to re-implement.
- **Arms**, each a ~20–60M-parameter reasoner trained from scratch on world A:
  1. **Dense:** facts in the weights; updates by MEMIT or fine-tuning.
  2. **Token-level lookup (fork A):** LMLM-style lookups, loss masked on retrieved values.
  3. **Latent store (fork B):** KV entries the reasoner attends to at an early layer, keyed by subject + relation, with a looped re-query block.
  4. **In-context oracle:** the same reasoner with the gold facts pasted in (the upper bound).
- **Test:** freeze the reasoner. Swap in world B's store, then apply 1 → 100 → 1,000 counterfactual edits. Measure 1/2/3-hop exact-match accuracy, using the Grokked-Transformer in-distribution vs. out-of-distribution composition split.
- **Headline metric:** the **propagation gap**, meaning multi-hop accuracy after an update divided by the in-context oracle's.
- **Deliverable:** one plot of propagation gap vs. number of edits for each arm. If an arm closes the gap where dense fails, that alone is a workshop-level result.

The toy tests these hypotheses, in priority order (full list in track 03, §7.4):
- **H3, hot-swap:** a new world with the same schema, no reasoner training, and 1-hop/2-hop accuracy within a few points of world A.
- **H5, re-entrancy:** without repeated lookup, 2-hop on new combinations stays near chance.
- **H4, format:** attendable KV/tokens beat feed-forward injection on 2-hop and reverse questions, and tie on 1-hop.
- **H2, double dissociation:** store off means closed-book QA collapses, while reasoning and open-book QA keep ≥90%.
- **H7, schema boundary:** new entities with old relation types work; new relation types fail.

### Step 2: interface on a frozen open model (1–2 weeks, laptop plus ~$50–150 of rented GPU)
- Frozen Qwen3-1.7B-Base or OLMo-2-1B, then 4B. OLMo is useful because its pretraining data is fully open, so contamination can be checked.
- Train only the interface; afterwards updates touch only the store.
- **Evaluate on:**
  - MQuAKE-Remastered (not the original, which has 33–76% corrupted items)
  - RippleEdits
  - WikiBigEdit (sequential updates at scale)
  - TOFU plus relearning attacks (deletion)
  - FreshQA
  - HotpotQA and MuSiQue (the reasoner stays competent)
- **Scoring:** always by generation with exact match; token-by-token scoring overstates editing success.
- **Baselines:** in-context oracle, RAG, LoRA fine-tuning, EasyEdit methods, KBLaM, Co-LMLM, DMoE, Doc-to-LoRA.

### Step 3: paper-level result (cloud, ~$2–10k)
- Pretrain Hemispheres and a matched dense model at 125M, 350M and 1B with the modded-nanogpt or nanochat code.
- Show:
  - no loss at equal compute;
  - 10k–100k sequential updates with multi-hop near the oracle;
  - reasoning benchmarks unchanged;
  - deletion that survives probes and relearning attacks;
  - a world swap with no reasoner training;
  - update cost orders of magnitude below fine-tuning.

## 8. Recent papers to read first

1. Co-LMLM (2607.07707) and LMLM (2505.15962): closest prior art.
2. DeepSeek Engram (2601.07372): the strongest separability evidence.
3. Apple, Pretraining with Hierarchical Memories (2510.02375).
4. Meta, Continual Learning via Sparse Memory Finetuning (2510.15103).
5. KBLaM (2410.10450) and Memory³ (2407.01178): the attention-level store.
6. MQuAKE (2305.14795) and "The Mirage of Model Editing" (2502.11177): the evaluation traps.
7. Two-hop curse (Balesni et al.) and Grokked Transformers (Wang et al. 2024): the latent-composition problem.
8. SynthWorlds (2510.24427): the real-vs-synthetic-entity evaluation method.

## 9. Caveats

- Several 2026 papers cited in the tracks are single-author or unrefereed preprints; the tracks flag them.
- All laptop throughput and cloud cost figures are estimates.
- Numbers for the papers added late (DMoE, Co-LMLM, Doc-to-LoRA, Baseten) come from summarized full-text reads. Spot-check them before relying on them.
