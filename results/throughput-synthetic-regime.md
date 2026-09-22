# MLX throughput — Apple M5 Max (40 GPU cores, 69 GB)

macOS 27.0 · MLX 0.32.2 · Now drawing from 'AC Power' · GPU 0% busy before start

## Peak matmul

| dtype | n | TFLOPS |
|---|---|---|
| float32 | 4096 | 32.4 |
| float32 | 8192 | 25.3 |
| float16 | 4096 | 44.2 |
| float16 | 8192 | 49.3 |
| bfloat16 | 4096 | 43.1 |
| bfloat16 | 8192 | 41.3 |

## Training (AdamW, causal LM, seq 256, random tokens)

| model | vocab | params (non-emb) | batch | dtype | compiled | tok/s | step s | TFLOPS | MFU* | peak mem GB | compile s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tiny | 8192 | 14M (11M) | 32 | bfloat16 | yes | 208,423 | 0.04 | 18.7 | 43% | 2.9 | 0.1 |
| tiny | 8192 | 14M (11M) | 64 | bfloat16 | yes | 184,591 | 0.09 | 16.5 | 38% | 5.5 | 0.1 |
| tiny | 8192 | 14M (11M) | 128 | bfloat16 | yes | 174,787 | 0.19 | 15.7 | 36% | 10.9 | 0.3 |
| small | 8192 | 29M (25M) | 32 | bfloat16 | yes | 95,076 | 0.09 | 17.9 | 42% | 4.0 | 0.1 |
| small | 8192 | 29M (25M) | 64 | bfloat16 | yes | 89,637 | 0.18 | 16.9 | 39% | 7.7 | 0.2 |
| small | 8192 | 29M (25M) | 128 | bfloat16 | yes | 88,084 | 0.37 | 16.6 | 39% | 15.1 | 0.4 |
| gpt2 | 8192 | 91M (85M) | 32 | bfloat16 | yes | 35,991 | 0.23 | 20.7 | 48% | 7.2 | 0.3 |
| gpt2 | 8192 | 91M (85M) | 64 | bfloat16 | yes | 39,322 | 0.42 | 22.6 | 53% | 13.8 | 0.5 |
| gpt2 | 8192 | 91M (85M) | 128 | bfloat16 | yes | 40,237 | 0.81 | 23.2 | 54% | 27.1 | 0.9 |

\*MFU is measured against this machine's own peak matmul for the same dtype, not a spec-sheet number.

## Planning (best bf16 config per size, at peak-burst throughput)

| model | best batch | tok/s | 1B tokens | 20 tok/param (Chinchilla) |
|---|---|---|---|---|
| tiny (14M, vocab 8192) | 32 | 208,423 | 1.3 h | 275M tok → 0.4 h |
| small (29M, vocab 8192) | 32 | 95,076 | 2.9 h | 587M tok → 1.7 h |
| gpt2 (91M, vocab 8192) | 128 | 40,237 | 6.9 h | 1.82B tok → 12.6 h |
