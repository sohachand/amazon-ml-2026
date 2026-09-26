"""Stage E (GPU): fine-tuned multilingual sentence embeddings.

Model: intfloat/multilingual-e5-small (MIT licence, 118M parameters) - understands Devanagari,
Tamil, Bengali, Gujarati, Kannada, French... natively, so it can learn that
"एसएस फूड प्राइवेट लिमिटेड | AF-0684, GHAZIABAD" and "Ss Food Private Limited | Af-684, Ghaziabad"
are the same business, and that "Payne Etrepndiels" is a typo of "Payne Enterprises".

1. finetune():  contrastive fine-tuning (MultipleNegativesRankingLoss with one hard negative per
   example) on (S1 record, matched S2/S3 record) pairs of the TRAINING split only, and only for
   S1 entities of the "embedding fold" (half of the train S1 entities, chosen by hash). The other
   half is what LightGBM trains on, so the embedding similarity is never evaluated on pairs the
   embedding model has seen (no leakage).
2. encode():    every record of a split -> L2-normalised 384-d float16 vector (npy, memory-mapped).
3. add_embedding_candidates(): per country, exact GPU top-k cosine search S1 -> S2/S3, union with
   the existing candidates, and attach esim (cosine) and erank (rank within the S1) to all pairs.
"""
import gc
import math
import os
import random
import time

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

MODEL_NAME = os.environ.get("EMB_MODEL", "intfloat/multilingual-e5-small")
MAX_LEN = 64
EMB_TOP_N = int(os.environ.get("EMB_TOP_N", 20))
SH = np.int64(1 << 24)


def _dev():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def emb_fold(r1):
    """True for S1 entities reserved for fine-tuning the embedding model (never used by LightGBM)."""
    return (np.asarray(r1).astype(np.int64) * 2654435761 % 1000) >= 500


def _texts(path):
    t = pq.read_table(path, columns=["name_raw", "addr_raw", "country"])
    return ["query: " + n + " | " + a + " | " + c
            for n, a, c in zip(t.column("name_raw").to_pylist(), t.column("addr_raw").to_pylist(),
                               t.column("country").to_pylist())]


def _log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def finetune(work, out_dir, n_pairs=int(os.environ.get("EMB_PAIRS", 1_200_000)), batch_size=256,
             epochs=1, seed=0):
    import torch
    from sentence_transformers import SentenceTransformer, InputExample, losses
    from torch.utils.data import DataLoader

    d = f"{work}/train"
    g = pd.read_parquet(f"{d}/gt_pairs.parquet")
    g = g[emb_fold(g.r1.values)]
    rng = np.random.RandomState(seed)
    if len(g) > n_pairs:
        g = g.iloc[rng.choice(len(g), n_pairs, replace=False)]
    # hard negatives: the best-ranked NON-matching candidate of the same S1 entity
    c = pd.read_parquet(f"{d}/cands.parquet", columns=["r1", "r2", "bscore"])
    c = c[emb_fold(c.r1.values)]
    ga = pd.read_parquet(f"{d}/gt_pairs.parquet")
    gcode = np.sort(ga.r1.values.astype(np.int64) * SH + ga.r2.values)
    del ga
    ccode = c.r1.values.astype(np.int64) * SH + c.r2.values
    pos = np.clip(np.searchsorted(gcode, ccode), 0, len(gcode) - 1)
    neg = c[gcode[pos] != ccode].sort_values("bscore", ascending=False).drop_duplicates("r1")
    hard = dict(zip(neg.r1.values, neg.r2.values))
    del c, neg
    s1 = _texts(f"{d}/s1.parquet")
    s23 = _texts(f"{d}/s2.parquet") + _texts(f"{d}/s3.parquet")
    ex = []
    for a, b in zip(g.r1.values, g.r2.values):
        h = hard.get(a)
        if h is None:
            h = rng.randint(len(s23))
        ex.append(InputExample(texts=[s1[a], s23[b], s23[h]]))
    del s1, s23
    gc.collect()
    _log(f"fine-tuning on {len(ex):,} (anchor, positive, hard negative) triplets")
    model = SentenceTransformer(MODEL_NAME, device=_dev())
    model.max_seq_length = MAX_LEN
    random.seed(seed)
    dl = DataLoader(ex, shuffle=True, batch_size=batch_size, drop_last=True)
    loss = losses.MultipleNegativesRankingLoss(model, scale=30.0)
    model.fit(train_objectives=[(dl, loss)], epochs=epochs,
              warmup_steps=int(0.05 * len(dl)), optimizer_params={"lr": 5e-5},
              use_amp=_dev() == "cuda", show_progress_bar=True)
    model.save(out_dir)
    _log("saved fine-tuned model to", out_dir)


def encode(work, split, model_dir, batch_size=1024):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_dir, device=_dev())
    model.max_seq_length = MAX_LEN
    if _dev() == "cuda":
        model.half()
    d = f"{work}/{split}"
    for k in (1, 2, 3):
        out = f"{d}/emb_s{k}.npy"
        if os.path.exists(out):
            continue
        texts = _texts(f"{d}/s{k}.parquet")
        # sort by length for speed, then restore order
        order = np.argsort([len(t) for t in texts])
        E = np.lib.format.open_memmap(out + ".tmp", mode="w+", dtype=np.float16,
                                      shape=(len(texts), model.get_sentence_embedding_dimension()))
        step = 200_000
        for a in range(0, len(texts), step):
            idx = order[a:a + step]
            v = model.encode([texts[i] for i in idx], batch_size=batch_size, convert_to_numpy=True,
                             normalize_embeddings=True, show_progress_bar=False)
            E[idx] = v.astype(np.float16)
            _log(f"  {split} s{k}: {min(a + step, len(texts)):,}/{len(texts):,}")
        E.flush()
        del E
        os.replace(out + ".tmp", out)


def _country_rows(path, country):
    col = pq.read_table(path, columns=["country"]).column(0)
    return np.flatnonzero(pc.equal(col, country).to_numpy(zero_copy_only=False)).astype(np.int64)


def add_embedding_candidates(work, split, top_n=EMB_TOP_N):
    """Union cands.parquet with GPU top-n cosine neighbours; add esim / erank to every pair."""
    import torch
    d = f"{work}/{split}"
    cands = pd.read_parquet(f"{d}/cands.parquet")
    n2 = pq.ParquetFile(f"{d}/s2.parquet").metadata.num_rows
    E1 = np.load(f"{d}/emb_s1.npy", mmap_mode="r")
    E2 = np.load(f"{d}/emb_s2.npy", mmap_mode="r")
    E3 = np.load(f"{d}/emb_s3.npy", mmap_mode="r")

    def E23(idx):
        """Rows of the stacked S2+S3 embedding matrix without loading it all into RAM."""
        idx = np.asarray(idx)
        out = np.empty((len(idx), E2.shape[1]), np.float16)
        m = idx < n2
        if m.any():
            o = np.argsort(idx[m]); sel = idx[m][o]
            tmp = np.empty((len(sel), E2.shape[1]), np.float16); tmp[o] = E2[sel]
            out[m] = tmp
        if (~m).any():
            o = np.argsort(idx[~m]); sel = idx[~m][o] - n2
            tmp = np.empty((len(sel), E2.shape[1]), np.float16); tmp[o] = E3[sel]
            out[~m] = tmp
        return out
    countries = pc.unique(pq.read_table(f"{d}/s1.parquet", columns=["country"]).column(0)).to_pylist()
    new_codes = []
    for c in countries:
        i1 = _country_rows(f"{d}/s1.parquet", c)
        i2 = np.concatenate([_country_rows(f"{d}/s2.parquet", c), _country_rows(f"{d}/s3.parquet", c) + n2])
        if not len(i1) or not len(i2):
            continue
        # memory-light exact search: S2/S3 vectors go to the GPU in slices of 500k rows and the
        # running top-n of every S1 record is merged slice by slice (fits GPUs with ~4 GB free)
        dev = _dev()
        best_s = torch.full((len(i1), top_n), -2.0, device=dev, dtype=torch.float16)
        best_i = torch.zeros((len(i1), top_n), device=dev, dtype=torch.int64)
        for b in range(0, len(i2), 500_000):
            D = torch.from_numpy(E23(i2[b:b + 500_000])).to(dev)
            for a in range(0, len(i1), 4096):
                q = torch.from_numpy(np.asarray(E1[i1[a:a + 4096]])).to(dev)
                s = q @ D.T
                v, ix = torch.topk(s, min(top_n, s.shape[1]), dim=1)
                cat_s = torch.cat([best_s[a:a + 4096], v], 1)
                cat_i = torch.cat([best_i[a:a + 4096], ix + b], 1)
                ns, sel = torch.topk(cat_s, top_n, dim=1)
                best_s[a:a + 4096] = ns
                best_i[a:a + 4096] = torch.gather(cat_i, 1, sel)
            del D
        bi = best_i.cpu().numpy(); bs_ = best_s.float().cpu().numpy()
        del best_s, best_i
        ok = bs_ > -1.5                                   # slots actually filled
        r1 = np.repeat(i1, top_n)[ok.ravel()]
        r2 = i2[bi.ravel()[ok.ravel()]]
        new_codes.append(r1.astype(np.int64) * SH + r2)
        if dev == "cuda":
            torch.cuda.empty_cache()
        _log(f"  embedding neighbours {split} {c!r} done")
    bcode = cands.r1.values.astype(np.int64) * SH + cands.r2.values
    extra = np.setdiff1d(np.unique(np.concatenate(new_codes)), bcode)
    add = pd.DataFrame({"r1": (extra // SH).astype(np.int32), "r2": (extra % SH).astype(np.int32)})
    for col, fill in [("bscore", 0), ("bits", 0), ("brank1", 999), ("brank2", 999), ("tsim", 0), ("trank", 999)]:
        add[col] = np.array(fill, dtype=cands[col].dtype)
    cands = pd.concat([cands, add], ignore_index=True)
    _log(f"  {split}: +{len(add):,} pairs from embeddings -> {len(cands):,}")
    # cosine for every pair
    r1 = cands.r1.values; r2 = cands.r2.values
    es = np.empty(len(cands), np.float32)
    for a in range(0, len(cands), 500_000):
        b = min(a + 500_000, len(cands))
        x = torch.from_numpy(np.asarray(E1[r1[a:b]])).to(_dev())
        y = torch.from_numpy(E23(r2[a:b])).to(_dev())
        es[a:b] = (x.float() * y.float()).sum(1).cpu().numpy()
        del x, y
    cands["esim"] = es
    o = np.lexsort((-es, r1))
    g = r1[o]
    start = np.r_[True, g[1:] != g[:-1]]
    idx = np.arange(len(g))
    rk = np.empty(len(g), np.int32)
    rk[o] = idx - np.maximum.accumulate(np.where(start, idx, 0))
    cands["erank"] = np.minimum(rk, 999).astype(np.int16)
    cands.to_parquet(f"{d}/cands.parquet", index=False)
