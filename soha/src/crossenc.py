"""Stage C (GPU): fine-tuned cross-encoder.

A bi-encoder (embed.py) turns each record into one vector independently; a cross-encoder reads
the S1 record and the candidate TOGETHER ("[CLS] s1 name | s1 address [SEP] cand name | cand
address") so every token can attend to the other record - it sees that "Etrepndiels" is a typo
of "Enterprises" in this specific context, that the house numbers agree, etc.
Model: intfloat/multilingual-e5-small (MIT, 118M params) with a 1-logit classification head,
fine-tuned with binary cross-entropy on (S1, S2/S3) pairs of the TRAINING split, only for S1
entities of the embedding fold (never used by LightGBM -> no leakage).  Its probability is added
as feature "ce" to stage 2 / stage 3 for every pair that stage 1 finds plausible.

Scoring uses every GPU of the machine (one forked worker per GPU).
"""
import os
import subprocess
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

CE_MODEL = os.environ.get("CE_MODEL", "intfloat/multilingual-e5-small")
CE_PAIRS = int(os.environ.get("CE_PAIRS", 1_500_000))
MAX_LEN = 96
SH = np.int64(1 << 24)


def _log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def n_gpus():
    """Number of GPUs without initialising CUDA in this process (so we can fork later)."""
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    if vis is not None and vis.strip() != "":
        return max(1, len([v for v in vis.split(",") if v.strip() != ""]))
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout
        return max(1, sum(1 for l in out.splitlines() if l.startswith("GPU")))
    except Exception:
        return 1


def _texts(path):
    t = pq.read_table(path, columns=["name_raw", "addr_raw"])
    return [n + " | " + a for n, a in zip(t.column("name_raw").to_pylist(),
                                          t.column("addr_raw").to_pylist())]


def finetune(work, out_dir, seed=0):
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, \
        get_linear_schedule_with_warmup
    import embed
    d = f"{work}/train"
    rng = np.random.RandomState(seed)
    g = pd.read_parquet(f"{d}/gt_pairs.parquet")
    g = g[embed.emb_fold(g.r1.values)]
    cols = ["r1", "r2", "bscore"] + [c for c in ("tsim", "esim")
                                     if c in pq.ParquetFile(f"{d}/cands.parquet").schema_arrow.names]
    c = pd.read_parquet(f"{d}/cands.parquet", columns=cols)
    c = c[embed.emb_fold(c.r1.values)]
    gcode = np.sort(g.r1.values.astype(np.int64) * SH + g.r2.values)
    ccode = c.r1.values.astype(np.int64) * SH + c.r2.values
    pos = np.clip(np.searchsorted(gcode, ccode), 0, len(gcode) - 1)
    neg = c[gcode[pos] != ccode]
    del c
    # hard negatives: the top-2 non-matches of each S1 by every similarity signal
    parts = []
    for col in cols[2:]:
        parts.append(neg.sort_values(col, ascending=False).groupby("r1").head(2)[["r1", "r2"]])
    neg = pd.concat(parts).drop_duplicates()
    n_pos = min(len(g), CE_PAIRS // 3)
    n_neg = min(len(neg), CE_PAIRS - n_pos)
    P = pd.concat([g.iloc[rng.choice(len(g), n_pos, replace=False)].assign(y=1.0),
                   neg.iloc[rng.choice(len(neg), n_neg, replace=False)].assign(y=0.0)])
    P = P.iloc[rng.permutation(len(P))]
    s1 = _texts(f"{d}/s1.parquet")
    s23 = _texts(f"{d}/s2.parquet") + _texts(f"{d}/s3.parquet")
    A = [s1[i] for i in P.r1.values]; B = [s23[i] for i in P.r2.values]; Y = P.y.values
    del s1, s23
    _log(f"cross-encoder: fine-tuning on {len(P):,} pairs ({n_pos:,} positive)")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(CE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(CE_MODEL, num_labels=1).to(dev)
    bs = 128
    idx = np.arange(len(P))
    steps = len(idx) // bs
    opt = torch.optim.AdamW(model.parameters(), lr=4e-5, weight_decay=0.01)
    sch = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)
    scaler = torch.amp.GradScaler(enabled=dev == "cuda")
    lossf = torch.nn.BCEWithLogitsLoss()
    model.train()
    for k in range(steps):
        b = idx[k * bs:(k + 1) * bs]
        enc = tok([A[i] for i in b], [B[i] for i in b], truncation=True, max_length=MAX_LEN,
                  padding=True, return_tensors="pt").to(dev)
        y = torch.tensor(Y[b], dtype=torch.float32, device=dev)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=dev == "cuda"):
            logit = model(**enc).logits.squeeze(-1)
        loss = lossf(logit.float(), y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); sch.step()
        if k % 1000 == 0:
            _log(f"  step {k:,}/{steps:,} loss {loss.item():.4f}")
    model.save_pretrained(out_dir); tok.save_pretrained(out_dir)
    _log("saved cross-encoder to", out_dir)


def _score_worker(gpu, k, model_dir, out_path, d):
    """Runs in a fresh (spawned) process: forking after LightGBM/OpenMP threads can deadlock."""
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    torch.set_num_threads(2)
    dev = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(dev).eval()
    if dev != "cpu":
        model.half()
    A = _texts(f"{d}/s1.parquet")
    B = _texts(f"{d}/s2.parquet") + _texts(f"{d}/s3.parquet")
    z = np.load(f"{d}/ce_job.npz")
    r1, r2 = z["r1"], z["r2"]
    order = z["order"][gpu::k]           # every k-th pair of the length-sorted order
    out = np.load(out_path, mmap_mode="r+")
    bs = 512
    with torch.inference_mode():
        for a in range(0, len(order), bs):
            ii = order[a:a + bs]
            enc = tok([A[j] for j in r1[ii]], [B[j] for j in r2[ii]], truncation=True,
                      max_length=MAX_LEN, padding=True, return_tensors="pt").to(dev)
            out[ii] = torch.sigmoid(model(**enc).logits.squeeze(-1).float()).cpu().numpy()
    out.flush()


def score(work, split, r1, r2, model_dir=None):
    """Cross-encoder probability for pairs (r1, r2) of a split; one worker per GPU."""
    import multiprocessing as mp
    model_dir = os.path.abspath(model_dir or f"{work}/ce_model")
    d = os.path.abspath(f"{work}/{split}")
    r1 = np.asarray(r1); r2 = np.asarray(r2)
    n = len(r1)
    # sort pairs by text length so batches have similar lengths (much less padding)
    l1 = pc_len(f"{d}/s1.parquet")
    l23 = np.concatenate([pc_len(f"{d}/s2.parquet"), pc_len(f"{d}/s3.parquet")])
    order = np.argsort(l1[r1] + l23[r2], kind="stable")
    np.savez(f"{d}/ce_job.npz", r1=r1, r2=r2, order=order)
    out_path = f"{d}/ce_tmp.npy"
    np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float32, shape=(n,)).flush()
    k = n_gpus()
    t0 = time.time()
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=_score_worker, args=(i, k, model_dir, out_path, d)) for i in range(k)]
    for p in procs: p.start()
    for p in procs: p.join()
    if any(p.exitcode != 0 for p in procs):
        raise RuntimeError("cross-encoder scoring worker failed")
    res = np.load(out_path).astype(np.float32)
    os.remove(out_path); os.remove(f"{d}/ce_job.npz")
    _log(f"cross-encoder scored {n:,} {split} pairs on {k} GPU(s) in {time.time() - t0:.0f}s")
    return res


def pc_len(path):
    """Character length of 'name | address' for every record of a parquet file."""
    import pyarrow.compute as pc
    t = pq.read_table(path, columns=["name_raw", "addr_raw"])
    return (pc.utf8_length(t.column("name_raw")).to_numpy(zero_copy_only=False)
            + pc.utf8_length(t.column("addr_raw")).to_numpy(zero_copy_only=False) + 3).astype(np.int32)
