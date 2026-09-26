# soha/: Soha's pipeline (leaderboard 0.981)

## What it does
1. **Candidate generation:**
   - Blocking on 10 key types.
   - TF-IDF char-ngram retrieval (`TFIDF_TOP_N=30`).
   - A fine-tuned bi-encoder, `intfloat/multilingual-e5-small`, with MNRL loss. It is trained on the "embedding fold" of the training S1s (hash ≥ 500) and returns the top `EMB_TOP_N=20` neighbours.
2. **Cross-encoder:** e5-small with a 1-logit head and BCE loss, fine-tuned on the embedding fold (positives plus hard negatives, 1M pairs, `MAX_LEN=96`).
3. **Stage 1 (LightGBM):** string, number and address features, with out-of-fold 2-fold scoring. Pairs with p1 > PRUNE go on to stage 2.
4. **Stage 2 (LightGBM):** adds competition context (other candidates of the same S1 and the same S2/S3 record) and the cross-encoder probability `ce`.
5. **Stage 3 (LightGBM):** adds "sibling" features, i.e. how much a candidate resembles the confident (s2 > 0.5) matches of the same S1. Only pairs with s2 > 0.01 are re-scored.
6. **Decision:** threshold t3 = 0.80. Each S2/S3 record is assigned to at most one S1 (the best one).

The LightGBM stages are trained only on non-embedding-fold S1s (`--train-frac 0.08`), so there is no leakage from the encoders. Local validation F0.5: stage-3 0.9865. Unstop leaderboard: 0.981.

## Exact commands (Jarvislabs, 2× A30, 32 CPU)

**Original 0.981 run**, from a folder with `er_gpu_full.zip`, `stage3_patch.zip` and `run_local_heavy.sh`:
```bash
FAST=1 N_PROC=30 bash run_local_heavy.sh
# FAST=1 => TFIDF_TOP_N=30 EMB_TOP_N=20 EMB_PAIRS=1000000 CE_PAIRS=1000000 --train-frac 0.08
# i.e. python3 src/pipeline.py --data dataset --work work --out output --gpu --ce --train-frac 0.08
```

**Re-run of stages 2–3 (26 Sep).** This keeps the existing bi-encoder, cross-encoder and model1:
```bash
N_PROC=30 TOKENIZERS_PARALLELISM=false python3 src/pipeline.py --data dataset --work work --out output --gpu --ce --train-frac 0.08
```

**Probability export** (the files in this folder):
```bash
python3 src/export_probs.py work submissions
```

**Experiment in progress:** the cross-encoder upgraded to `intfloat/multilingual-e5-base` (CE_PAIRS=1.5M), written to `work_base/`:
```bash
CUDA_VISIBLE_DEVICES=1 CE_MODEL=intfloat/multilingual-e5-base python3 src/pipeline.py --data dataset --work work_base --out output_base --stage ce_train
CE_MODEL=intfloat/multilingual-e5-base N_PROC=30 python3 src/pipeline.py --data dataset --work work_base --out output_base --gpu --ce --train-frac 0.08
```
Stages run in this order: prep → block_train → block_test → embed_train → embed_test → ce_train → train → train3 → predict → rescore → rescore3 → write. Every stage is resumable; you can run a single one with `--stage NAME`.

## Probability files
| File | Contents |
|---|---|
| `soha_france_pairs.tsv.gz` | All France test S1s; every candidate pair with p ≥ 0.05 |
| `soha_us_india_20k_each_pairs.tsv.gz` | 20,000 random US S1s and 20,000 random India S1s (`RandomState(0)`); pairs with p ≥ 0.05 |
| `soha_us_india_sampled_s1.tsv.gz` | The 40k sampled S1 ids, so that S1s with no pair ≥ 0.05 are known |
| `soha_export_stats.txt` | Mean candidates per S1 scored by stage 2 and stage 3, overall and per country |

In these files, p is the stage-3 probability, taken before the 0.80 threshold and the one-to-one step. Pairs that stage 3 did not re-score (s2 ≤ 0.01) carry their stage-2 probability. Pairs that stage 1 pruned carry 0.

## Where the models are (Jarvis instance, `/home/er_gpu/`)
| Model | Path |
|---|---|
| Bi-encoder | `work/e5_finetuned/` (HF format) |
| Cross-encoder (e5-small) | `work/ce_model/` |
| Cross-encoder (e5-base, experiment) | `work_base/ce_model/` |
| LightGBM | `work/model1.txt`, `work/model2.txt`, `work/model3.txt` |
| Thresholds | `work/model_meta.json` |

The models are not in git because of their size; they can be copied on request.

## Code
The code is in `src/`. Main files:
- `pipeline.py`: orchestration
- `blocking.py`, `retrieval.py`, `embed.py`: candidate generation
- `crossenc.py`: cross-encoder
- `features.py`, `model.py`: stage 1–2 features and models
- `sibling.py`: stage 3
- `writer.py`: output
- `export_probs.py`: probability export

The environment is described in `requirements_gpu.txt`.
