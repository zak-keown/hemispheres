# MLX throughput — Apple M5 Max (40 GPU cores, 69 GB)

macOS 27.0 · MLX 0.32.2 · Now drawing from 'AC Power' · GPU 0% busy before start

## Training (AdamW, causal LM, seq 256, random tokens)

| model | vocab | params (non-emb) | batch | dtype | compiled | tok/s | step s | TFLOPS | MFU* | peak mem GB | compile s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| small | 8192 | 29M (25M) | 128 | bfloat16 | yes | 105,175 | 0.31 | 19.9 | — | 15.1 | 0.4 |

\*MFU is measured against this machine's own peak matmul for the same dtype, not a spec-sheet number.

## Sustained: small batch 128 for 600s

Tokens/sec per 30s window: 107,195, 106,607, 106,758, 106,138, 105,514, 105,184, 104,883, 104,787, 104,408, 104,727, 104,553, 103,284, 103,491, 104,550, 103,793, 103,814, 103,954, 103,575, 104,157

Last window vs first: 97%

## Planning (best bf16 config per size, at peak-burst throughput)

| model | best batch | tok/s | 1B tokens | 20 tok/param (Chinchilla) |
|---|---|---|---|---|
| small (29M, vocab 8192) | 128 | 105,175 | 2.6 h | 587M tok → 1.6 h |
