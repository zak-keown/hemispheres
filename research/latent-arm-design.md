# Latent-store arm: design options

*2026-09-22. Draft for choosing the fourth arm of the Step-1 toy.*

## What the smoke test changed

A tiny lookup-arm model, trained for 90 seconds, reached 93% on the world-B swap and 94% on ripple questions after 100 edits, with no retraining. Written-out lookups make propagation easy in this toy. So the lookup arm is the **ceiling**. The research question becomes:

> Can a reasoner match the lookup arm when each lookup happens **inside the forward pass**, with no lookup tokens written out?

To keep that comparison clean, the latent arm should differ from the lookup arm in one thing only: where the query and the returned value live. The store (the world's triples), the questions and the hop supervision stay the same.

## What every option shares

- **Store:** one entry per fact `(s, r, o)`. Writing world B means encoding B's triples with a forward pass only. Deleting a fact means deleting its entry.
- **Key:** `key = K(name(s), r)`. K is a small encoder that reads the subject's name tokens through the model's own token embeddings (mean-pool, plus a relation embedding, then an MLP). It is trained jointly on world A, then frozen. Because keys are computed from surface names, they stay valid across worlds and never depend on the reasoner's hidden states. That avoids Co-LMLM's problem of re-indexing the store whenever the model is fine-tuned.
- **Retrieval:** exact dot-product top-k (k = 4) over all ~62k entries, with the score matmul done without gradients. Gradients reach the query and the keys through the re-scored top-k entries only, as in memory layers. The compute cost is estimated at about 25% on top of the small model's step.
- **Hop supervision (optional, see Q2):** an InfoNCE loss that pushes the query at read layer *i*, at the `A:` position, toward the key of hop *i*'s fact. The lookup arm effectively gets this supervision through its written traces, so turning it on is the fair comparison. Turning it off is the ablation.

## Option 1: token-valued latent lookup (recommended first)

A **read layer** is inserted after blocks 2, 4 and 6 of the 8-block model. At every position *t*:

1. The query is `q_t = W_q h_t`, and the top-k entries by `q_t · key_j` are retrieved.
2. Each entry's value is the **object's name tokens**, embedded with the shared token embedding plus a within-name position embedding. That gives k × ≤6 vectors.
3. `h_t` cross-attends over those vectors. The retrieval score `q_t · key_j` is added to the attention logits, so selection is learned end to end (the KBLaM-style trick).

For multi-hop, read layer 2's query is formed after read layer 1's result has entered the residual stream, so the depth is unrolled at one hop per read layer (at most 3 hops). While decoding the answer, each position retrieves again and copies the next name token, much as the lookup arm copies from `[RESULT] … [END]`.

- **Why first:** the values are tokens the model can copy. This is the same copying mechanism the lookup arm already uses successfully, so any gap between the arms is attributable to lookups happening inside the forward pass.
- **Risk:** top-k retrieval can fail to get started early in training. Hop supervision is the mitigation.

## Option 2: vector-valued latent lookup (KBLaM-style)

The same, except each value is a single vector `V(name(o))` from a value encoder, not name tokens. The reasoner has to decode a multi-token name, including names from worlds it has never seen, from one vector.

- **Why:** this is the purer "knowledge in the model's native representation" design, and the next hop's query can be built directly from that vector.
- **Risk:** decoding unseen names from a vector requires the value encoder to generalize compositionally over syllables. If it doesn't, the world swap fails for reasons that have nothing to do with reasoning. That makes it a good second experiment, not a first one.

## Option 3: looped reasoner (can be combined with 1 or 2)

Replace the 3 distinct read layers with **one weight-shared block (read + attention + MLP) applied K times**, with K = 3, or adaptive.

- **Why:** the Grokked Transformers paper found that sharing weights across layers improved out-of-distribution composition. A shared "hop circuit" also directly tests H5 (re-entrancy), and K is an ablation knob: K = 1 should fail on 2- and 3-hop questions.
- **Risk:** looped transformers can be less stable to train. It's cleaner to add this after Option 1 works.

## A training-protocol choice that matters for every store arm

**Store randomization (multi-world training).** When trained on world A alone, a store arm could quietly memorize world A in its weights as well as using the store. That would inflate its world-A scores and hide problems until the swap. The alternative is to train on many generated worlds, sampling a different world and store for each example. Memorizing can't pay off, so the model *must* use the store. The generator makes this cheap: disjoint partitions, plus seeds.

- **For:** it directly enforces "the reasoner holds no facts", the core premise of Hemispheres. It also makes deletion leakage testable: with the store removed, facts should drop to chance.
- **Against:** it is a different protocol from "train on A, swap to B". Run it as a second condition rather than replacing the first.

## Recommendation

1. **Option 1** with hop supervision on, trained on world A only, to compare directly with the lookup arm.
2. Then ablate:
   - hop supervision off;
   - read layers → looped block (Option 3);
   - single-world → multi-world training.
3. **Option 2** comes last.

**Success criterion:** world-B swap and ripple accuracy within about 5 points of the lookup arm, at equal parameters and training tokens, with no lookup tokens written out.
