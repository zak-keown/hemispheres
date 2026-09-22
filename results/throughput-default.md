# MLX throughput — Apple M5 Max (40 GPU cores, 69 GB)

macOS 27.0 · MLX 0.32.2 · Now drawing from 'AC Power' · GPU 1% busy before start

## Peak matmul

| dtype | n | TFLOPS |
|---|---|---|
| float32 | 4096 | 26.2 |
| float32 | 8192 | 24.5 |
| float16 | 4096 | 42.8 |
| float16 | 8192 | 39.9 |
| bfloat16 | 4096 | 43.1 |
| bfloat16 | 8192 | 40.7 |

## Training (AdamW, causal LM, seq 1024, random tokens)

| model | vocab | params (non-emb) | batch | dtype | compiled | tok/s | step s | TFLOPS | MFU* | peak mem GB | compile s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tiny | 50304 | 30M (11M) | 8 | bfloat16 | yes | 87,237 | 0.09 | 18.1 | 42% | 10.5 | 3.6 |
| tiny | 50304 | 30M (11M) | 16 | bfloat16 | yes | 87,496 | 0.19 | 18.2 | 42% | 20.7 | 0.4 |
| tiny | 50304 | 30M (11M) | 32 | bfloat16 | yes | 73,259 | 0.45 | 15.2 | 35% | 40.4 | 0.9 |
| small | 50304 | 51M (25M) | 8 | bfloat16 | yes | 52,193 | 0.16 | 18.6 | 43% | 12.1 | 0.3 |
| small | 50304 | 51M (25M) | 16 | bfloat16 | yes | 52,734 | 0.31 | 18.8 | 44% | 23.6 | 0.5 |
| small | 50304 | 51M (25M) | 32 | bfloat16 | yes | 49,911 | 0.66 | 17.8 | 41% | 46.0 | 1.1 |
| gpt2 | 50304 | 124M (85M) | 8 | bfloat16 | yes | 25,368 | 0.32 | 21.7 | 50% | 16.5 | 0.5 |
| gpt2 | 50304 | 124M (85M) | 16 | bfloat16 | yes | 25,040 | 0.65 | 21.4 | 50% | 32.1 | 0.8 |
| gpt2 | 50304 | 124M (85M) | 32 | bfloat16 | yes | 20,512 | 1.60 | 17.5 | 41% | 62.1 | 2.0 |
| medium | 50304 | 354M (302M) | 8 | bfloat16 | yes | 9,263 | 0.88 | 22.4 | 52% | 29.6 | 1.0 |
| medium | 50304 | 354M (302M) | 16 | bfloat16 | yes | 8,668 | 1.89 | 21.0 | 49% | 57.0 | 2.1 |
| medium | 50304 | — | 32 | bfloat16 | yes | exit -15:  | | | | | |
| gpt2 | 50304 | 124M (85M) | 16 | float32 | yes | 19,634 | 0.83 | 16.8 | 64% | 44.7 | 4.0 |
| gpt2 | 50304 | 124M (85M) | 16 | bfloat16 | no | 24,200 | 0.68 | 20.7 | 48% | 31.9 | 1.0 |
| tiny | 8192 | 14M (11M) | 32 | bfloat16 | yes | 165,929 | 0.20 | 18.4 | 43% | 12.7 | 0.3 |
| tiny | 8192 | 14M (11M) | 64 | bfloat16 | yes | 161,569 | 0.41 | 17.9 | 42% | 25.3 | 0.7 |
| small | 8192 | 29M (25M) | 32 | bfloat16 | yes | 92,920 | 0.35 | 21.0 | 49% | 18.3 | 0.5 |
| small | 8192 | 29M (25M) | 64 | bfloat16 | yes | 90,320 | 0.73 | 20.5 | 48% | 36.4 | 1.1 |

\*MFU is measured against this machine's own peak matmul for the same dtype, not a spec-sheet number.

## Planning (best bf16 config per size, at peak-burst throughput)

| model | best batch | tok/s | 1B tokens | 20 tok/param (Chinchilla) |
|---|---|---|---|---|
| tiny (30M, vocab 50304) | 16 | 87,496 | 3.2 h | 599M tok → 1.9 h |
| small (51M, vocab 50304) | 16 | 52,734 | 5.3 h | 1.02B tok → 5.4 h |
| gpt2 (124M, vocab 50304) | 8 | 25,368 | 11.0 h | 2.47B tok → 27.1 h |
| medium (354M, vocab 50304) | 8 | 9,263 | 30.0 h | 7.07B tok → 212.0 h |
| tiny (14M, vocab 8192) | 32 | 165,929 | 1.7 h | 275M tok → 0.5 h |
| small (29M, vocab 8192) | 32 | 92,920 | 3.0 h | 587M tok → 1.8 h |
