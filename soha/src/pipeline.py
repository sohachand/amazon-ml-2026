"""End-to-end pipeline: data -> normalise -> blocking -> features -> LightGBM -> outputs.

usage:
  python src/pipeline.py --data ../dataset --work ./work --out ./output [--train-frac 0.06]

Every stage caches its result in --work, so a re-run resumes where it stopped.
"""
import argparse, gc, json, os, sys, time

os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")  # macOS fork safety
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import lightgbm as lgb

import prep
import blocking
import retrieval
import features as F
import model as M
from idmap import gt_pairs
from writer import write_lists

T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)


def n_rows(path):
    return pq.ParquetFile(path).metadata.num_rows


def stage_prep(data, work):
    for split in ("train", "test"):
        d = f"{work}/{split}"
        if not os.path.exists(f"{d}/s3.parquet"):
            log("normalising", split)
            prep.main(f"{data}/{split}", split, d)


def stage_block(work, split):
    d = f"{work}/{split}"
    if not os.path.exists(f"{d}/cands.parquet"):
        log("blocking", split)
        c = blocking.generate_split(d)
        c.to_parquet(f"{d}/cands_block.parquet", index=False)
        log("tf-idf retrieval", split)
        c = retrieval.add_tfidf(d, c)
        c.to_parquet(f"{d}/cands.parquet", index=False)
        del c; gc.collect()
    log("blocking done", split)


def pair_features(d, pairs, nc1, nc2, s1t, s23t, nidf, aidf):
    F.setup(s1t, s23t, nidf, aidf)
    return F.compute(pairs, n_rows(f"{d}/s2.parquet"), nc1, nc2)


def _labels(g, r1, r2):
    SH = np.int64(1 << 24)
    gcode = np.sort(g.r1.values.astype(np.int64) * SH + g.r2.values)
    pcode = r1.astype(np.int64) * SH + r2
    pos = np.clip(np.searchsorted(gcode, pcode), 0, len(gcode) - 1)
    return (gcode[pos] == pcode).astype(np.int8)


def _tune(Pva, s, gva, va_s1, grid):
    res = []
    for t in grid:
        f05, pr, rc = M.macro_f05(M.decide(Pva, s, t), gva, va_s1)
        res.append((float(t), f05, pr, rc))
    return max(res, key=lambda r: r[1]), res



def use_ce(work):
    """Cross-encoder feature is used when its model was trained and was not disabled."""
    return os.path.exists(f"{work}/ce_model/config.json") and os.environ.get("NO_CE") != "1"


def ce_scores(work, split, r1, r2):
    import crossenc
    return crossenc.score(work, split, r1, r2)


def stage_ce(data, work):
    """GPU: fine-tune the cross-encoder on the embedding fold of the training split."""
    import crossenc
    d = f"{work}/train"
    if not os.path.exists(f"{d}/gt_pairs.parquet"):
        gt_pairs(f"{data}/train/train_ground_truth.tsv", d).to_parquet(f"{d}/gt_pairs.parquet")
    if not os.path.exists(f"{work}/ce_model/config.json"):
        log("fine-tuning cross-encoder")
        crossenc.finetune(work, f"{work}/ce_model")
    log("cross-encoder ready")

def stage_train(data, work, frac, seed=0):
    """Two-stage model. Stage 1 scores pairs from string/number features; stage 2 re-scores
    each pair given the stage-1 scores of competing pairs (same S2/S3 record, same S1)."""
    d = f"{work}/train"
    if os.path.exists(f"{work}/model2.txt"):
        log("model exists, skipping training")
        return
    if not os.path.exists(f"{d}/gt_pairs.parquet"):
        gt_pairs(f"{data}/train/train_ground_truth.tsv", d).to_parquet(f"{d}/gt_pairs.parquet")
    g = pd.read_parquet(f"{d}/gt_pairs.parquet")
    c = pd.read_parquet(f"{d}/cands.parquet")
    n1 = n_rows(f"{d}/s1.parquet")
    n23 = n_rows(f"{d}/s2.parquet") + n_rows(f"{d}/s3.parquet")
    rng = np.random.RandomState(seed)
    samp = rng.rand(n1) < frac
    if os.path.exists(f"{d}/emb_s1.npy"):
        import embed
        samp &= ~embed.emb_fold(np.arange(n1))  # never train LightGBM on the embedding fold
    core = samp[c.r1.values]
    r2m = np.zeros(n23, bool); r2m[c.r2.values[core]] = True
    # sampled S1 pairs + the strongest competitors (by blocking rank) of their S2/S3 records
    ext = core | (r2m[c.r2.values] & (c.brank2.values < 4))
    P = c[ext].reset_index(drop=True)
    is_core = samp[P.r1.values]
    y = _labels(g, P.r1.values, P.r2.values)
    log(f"train sample: {samp.sum():,} S1, {is_core.sum():,} core pairs, {len(P):,} with competitors")
    nc1 = np.bincount(c.r1.values, minlength=n1); nc2 = np.bincount(c.r2.values, minlength=n23)
    del c; gc.collect()
    s1t, s23t = F.load_tables(d)
    nidf = F.token_idf(s23t, "name_core"); aidf = F.token_idf(s23t, "addr_tok")
    X = pair_features(d, P, nc1, nc2, s1t, s23t, nidf, aidf)
    del s1t, s23t, nidf, aidf; gc.collect()
    log("train features", X.shape)
    # stage 1: out-of-fold scores with S1-grouped 2-fold split
    fold = (P.r1.values.astype(np.int64) * 2654435761 % 1000 < 500)
    p1 = np.zeros(len(P), np.float32)
    for k in (False, True):
        m = M.fit(X[fold != k], y[fold != k], M.P1, M.N1_ROUNDS)
        p1[fold == k] = m.predict(X[fold == k])
    # stage 2 only looks at pairs stage 1 finds plausible (p1 > PRUNE); others score 0
    kp = p1 > M.PRUNE
    K = np.flatnonzero(kp)
    C = M.context(P.r1.values[K], P.r2.values[K], p1[K])
    ceK = ce_scores(work, "train", P.r1.values[K], P.r2.values[K]) if use_ce(work) else None
    X2 = M.stage2_matrix(X[M.S2_RAW].values[K], C, ceK)
    yK = y[K]
    # stage 2 trained on core pairs only (their S1 and S2/S3 groups are complete)
    s1_samp = np.flatnonzero(samp)
    va_s1_mask = np.zeros(n1, bool); va_s1_mask[s1_samp[rng.rand(len(s1_samp)) < 0.3]] = True
    coreK = is_core[K]
    vaK = va_s1_mask[P.r1.values[K]] & coreK
    trK = (~va_s1_mask[P.r1.values[K]]) & coreK
    m2 = M.fit(X2[trK], yK[trK], M.P2, M.N2_ROUNDS)
    va = va_s1_mask[P.r1.values] & is_core
    s2full = np.zeros(len(P), np.float32)
    s2full[K[vaK]] = m2.predict(X2[vaK])
    s2 = s2full[va]
    Pva = P[va]; gva = g[va_s1_mask[g.r1.values]]; va_s1 = np.flatnonzero(va_s1_mask)
    grid = np.round(np.arange(0.30, 0.96, 0.025), 3)
    b1, _ = _tune(Pva, p1[va], gva, va_s1, grid)
    b2, res = _tune(Pva, s2, gva, va_s1, grid)
    log("stage-1 only : t=%.3f F0.5=%.4f P=%.4f R=%.4f" % b1)
    for r in res:
        log("  stage-2 t=%.3f F0.5=%.4f P=%.4f R=%.4f" % r)
    log("stage-2      : t=%.3f F0.5=%.4f P=%.4f R=%.4f" % b2)
    imp = pd.Series(m2.feature_importance("gain"), index=X2.columns).sort_values(ascending=False)
    log("stage-2 top features:\n" + imp.head(12).round(0).to_string())
    # final models
    m1 = M.fit(X, y, M.P1, M.N1_ROUNDS); m1.save_model(f"{work}/model1.txt")
    m2 = M.fit(X2[coreK], yK[coreK], M.P2, M.N2_ROUNDS); m2.save_model(f"{work}/model2.txt")
    json.dump({"threshold": b2[0], "val_f05": b2[1], "val_precision": b2[2], "val_recall": b2[3],
               "stage1_val_f05": b1[1], "train_frac": frac},
              open(f"{work}/model_meta.json", "w"), indent=1)


def stage_embed(data, work, split):
    """GPU: fine-tune the embedding model (train only), encode records, add embedding candidates."""
    import embed
    d = f"{work}/{split}"
    mdir = f"{work}/e5_finetuned"
    if split == "train" and not os.path.exists(f"{d}/gt_pairs.parquet"):
        # the fine-tuning needs the true pairs, which are otherwise only built by stage "train"
        gt_pairs(f"{data}/train/train_ground_truth.tsv", d).to_parquet(f"{d}/gt_pairs.parquet")
    if split == "train" and not os.path.exists(f"{mdir}/config.json"):
        log("fine-tuning embedding model")
        embed.finetune(work, mdir)
    log("encoding", split)
    embed.encode(work, split, mdir)
    if not os.path.exists(f"{d}/emb_done"):
        embed.add_embedding_candidates(work, split)
        open(f"{d}/emb_done", "w").write("ok")
    log("embeddings done", split)


def stage_predict(work, out, pairs_per_chunk=1_000_000):
    d = f"{work}/test"
    b1 = lgb.Booster(model_file=f"{work}/model1.txt")
    b2 = lgb.Booster(model_file=f"{work}/model2.txt")
    c = pd.read_parquet(f"{d}/cands.parquet")
    n1 = n_rows(f"{d}/s1.parquet")
    n23 = n_rows(f"{d}/s2.parquet") + n_rows(f"{d}/s3.parquet")
    nc1 = np.bincount(c.r1.values, minlength=n1); nc2 = np.bincount(c.r2.values, minlength=n23)
    s1t, s23t = F.load_tables(d)
    nidf = F.token_idf(s23t, "name_core"); aidf = F.token_idf(s23t, "addr_tok")
    log("tables loaded")
    # stage-1 scores and stage-2 raw features go to disk (resumable, low RAM)
    n_chunks = max(1, int(np.ceil(len(c) / pairs_per_chunk)))
    mode = "r+" if os.path.exists(f"{d}/p1.npy") else "w+"
    p1 = np.lib.format.open_memmap(f"{d}/p1.npy", mode=mode, dtype=np.float32, shape=(len(c),))
    raw = np.lib.format.open_memmap(f"{d}/s2raw.npy", mode=mode, dtype=np.float32,
                                    shape=(len(c), len(M.S2_RAW)))
    done_f = f"{d}/p1_done.txt"
    done = set(open(done_f).read().split()) if os.path.exists(done_f) else set()
    part = c.r1.values % n_chunks
    for k in range(n_chunks):
        if str(k) in done:
            continue
        idx = np.flatnonzero(part == k)
        X = pair_features(d, c.iloc[idx].reset_index(drop=True), nc1, nc2, s1t, s23t, nidf, aidf)
        p1[idx] = b1.predict(X)
        raw[idx] = X[M.S2_RAW].values
        p1.flush(); raw.flush()
        with open(done_f, "a") as f:
            f.write(f"{k}\n")
        del X; gc.collect()
        log(f"scored chunk {k + 1}/{n_chunks}")


def stage_rescore(work, out):
    """Stage 2 on test: competition context from all stage-1 scores, then re-score."""
    d = f"{work}/test"
    b2 = lgb.Booster(model_file=f"{work}/model2.txt")
    c = pd.read_parquet(f"{d}/cands.parquet", columns=["r1", "r2"])
    p1 = np.load(f"{d}/p1.npy")
    K = np.flatnonzero(p1 > M.PRUNE)          # only plausible pairs get stage 2 (low memory)
    C = M.context(c.r1.values[K], c.r2.values[K], p1[K])
    ce = None
    if "ce" in b2.feature_name():
        if not os.path.exists(f"{d}/ceK.npy"):
            np.save(f"{d}/ceK.npy", ce_scores(work, "test", c.r1.values[K], c.r2.values[K]))
        ce = np.load(f"{d}/ceK.npy")
    del c; gc.collect()
    raw = np.load(f"{d}/s2raw.npy", mmap_mode="r")
    scores = np.zeros(len(p1), np.float32)
    step = 2_000_000
    for a in range(0, len(K), step):
        idx = K[a:a + step]
        scores[idx] = b2.predict(M.stage2_matrix(np.asarray(raw[idx]), C[a:a + step],
                                                 None if ce is None else ce[a:a + step]))
    np.save(f"{d}/scores.npy", scores)
    log(f"stage-2 rescored {len(K):,} of {len(p1):,} pairs")
    log("stage-2 rescoring done")



# ------------------------------------------------------------------ stage 3 (sibling evidence)
def _x3(X2, s2, r1, r2, n_s2, s23t):
    """Stage-3 matrix: stage-2 inputs + stage-2 score + its competition context + sibling feats."""
    import sibling as SB
    Q = M.context(r1, r2, s2)
    Sb = SB.sib_features(r1, r2, s2, n_s2, s23t)
    cols = list(X2.columns) + ["s2"] + ["q_" + c for c in M.CTX_NAMES[1:]] + SB.SIB_NAMES
    return pd.DataFrame(np.hstack([X2.values.astype(np.float32), s2[:, None].astype(np.float32),
                                   Q[:, 1:], Sb]), columns=cols)


def stage_train3(data, work, frac, seed=0):
    """Re-creates the stage-2 training sample (same seed), gets out-of-fold stage-2 scores and
    trains stage 3 on top.  Keeps stage 3 only if it improves validation macro F0.5."""
    import sibling as SB
    d = f"{work}/train"
    if os.path.exists(f"{work}/model3.txt") or os.path.exists(f"{work}/model3_rejected"):
        log("stage-3 model exists, skipping"); return
    g = pd.read_parquet(f"{d}/gt_pairs.parquet")
    c = pd.read_parquet(f"{d}/cands.parquet")
    n1 = n_rows(f"{d}/s1.parquet"); n_s2 = n_rows(f"{d}/s2.parquet")
    n23 = n_s2 + n_rows(f"{d}/s3.parquet")
    rng = np.random.RandomState(seed)
    samp = rng.rand(n1) < frac
    if os.path.exists(f"{d}/emb_s1.npy"):
        import embed
        samp &= ~embed.emb_fold(np.arange(n1))
    core = samp[c.r1.values]
    r2m = np.zeros(n23, bool); r2m[c.r2.values[core]] = True
    ext = core | (r2m[c.r2.values] & (c.brank2.values < 4))
    P = c[ext].reset_index(drop=True)
    is_core = samp[P.r1.values]
    y = _labels(g, P.r1.values, P.r2.values)
    nc1 = np.bincount(c.r1.values, minlength=n1); nc2 = np.bincount(c.r2.values, minlength=n23)
    del c; gc.collect()
    s1t, s23t = F.load_tables(d)
    nidf = F.token_idf(s23t, "name_core"); aidf = F.token_idf(s23t, "addr_tok")
    X = pair_features(d, P, nc1, nc2, s1t, s23t, nidf, aidf)
    del s1t, nidf, aidf; gc.collect()
    log("stage-3: train features", X.shape)
    fold = (P.r1.values.astype(np.int64) * 2654435761 % 1000 < 500)
    p1 = np.zeros(len(P), np.float32)
    for k in (False, True):
        m = M.fit(X[fold != k], y[fold != k], M.P1, M.N1_ROUNDS)
        p1[fold == k] = m.predict(X[fold == k])
    K = np.flatnonzero((p1 > M.PRUNE) & is_core)   # core pairs only: complete S1 groups
    Kall = np.flatnonzero(p1 > M.PRUNE)
    C = M.context(P.r1.values[Kall], P.r2.values[Kall], p1[Kall])
    ceA = ce_scores(work, "train", P.r1.values[Kall], P.r2.values[Kall]) if use_ce(work) else None
    X2all = M.stage2_matrix(X[M.S2_RAW].values[Kall], C, ceA)
    sel = is_core[Kall]
    X2 = X2all[sel].reset_index(drop=True); del X2all, X; gc.collect()
    yK = y[K]; r1K = P.r1.values[K]; r2K = P.r2.values[K]
    # out-of-fold stage-2 scores for every core pair (2 folds by S1)
    f2 = (r1K.astype(np.int64) * 40503 % 1000 < 500)
    s2 = np.zeros(len(K), np.float32)
    for k in (False, True):
        m = M.fit(X2[f2 != k], yK[f2 != k], M.P2, M.N2_ROUNDS)
        s2[f2 == k] = m.predict(X2[f2 == k])
    k3 = s2 > SB.KEEP3
    X3 = _x3(X2[k3].reset_index(drop=True), s2[k3], r1K[k3], r2K[k3], n_s2, s23t)
    y3 = yK[k3]; r13 = r1K[k3]
    log("stage-3 matrix", X3.shape)
    # validation: 30% of sampled S1 entities
    s1_samp = np.flatnonzero(samp)
    vmask = np.zeros(n1, bool); vmask[s1_samp[rng.rand(len(s1_samp)) < 0.3]] = True
    va3 = vmask[r13]
    m3 = M.fit(X3[~va3], y3[~va3], M.P2, M.N2_ROUNDS)
    Pc = P[is_core].reset_index(drop=True)
    va_pairs = vmask[Pc.r1.values]
    Pva = Pc[va_pairs].reset_index(drop=True)
    gva = g[vmask[g.r1.values]]; va_s1 = np.flatnonzero(vmask)
    # scores aligned with Pva: map (r1,r2) codes
    SH = np.int64(1 << 24)
    code_va = Pva.r1.values.astype(np.int64) * SH + Pva.r2.values
    def align(codes, vals):
        o = np.argsort(codes); cs = codes[o]
        pos = np.clip(np.searchsorted(cs, code_va), 0, len(cs) - 1)
        return np.where(cs[pos] == code_va, vals[o][pos], 0).astype(np.float32)
    ck = r1K.astype(np.int64) * SH + r2K
    sc2 = align(ck[vmask[r1K]], s2[vmask[r1K]])
    s3v = np.zeros(len(K), np.float32); s3v[np.flatnonzero(k3)[va3]] = m3.predict(X3[va3])
    sc3 = align(ck[vmask[r1K]], s3v[vmask[r1K]])
    grid = np.round(np.arange(0.30, 0.96, 0.025), 3)
    b2, _ = _tune(Pva, sc2, gva, va_s1, grid)
    b3, res = _tune(Pva, sc3, gva, va_s1, grid)
    log("stage-2 (oof) : t=%.3f F0.5=%.4f P=%.4f R=%.4f" % b2)
    for r in res:
        log("  stage-3 t=%.3f F0.5=%.4f P=%.4f R=%.4f" % r)
    log("stage-3       : t=%.3f F0.5=%.4f P=%.4f R=%.4f" % b3)
    imp = pd.Series(m3.feature_importance("gain"), index=X3.columns).sort_values(ascending=False)
    log("stage-3 top features:\n" + imp.head(15).round(0).to_string())
    if b3[1] <= b2[1]:
        log("stage 3 does not help on validation -> not used")
        open(f"{work}/model3_rejected", "w").write("1"); return
    m3 = M.fit(X3, y3, M.P2, M.N2_ROUNDS); m3.save_model(f"{work}/model3.txt")
    meta = json.load(open(f"{work}/model_meta.json"))
    meta.update({"threshold3": b3[0], "val_f05_stage3": b3[1], "val_f05_stage2_oof": b2[1]})
    json.dump(meta, open(f"{work}/model_meta.json", "w"), indent=1)


def stage_rescore3(work):
    """Stage 3 on test, on top of the saved stage-1 / stage-2 scores."""
    import sibling as SB
    d = f"{work}/test"
    if not os.path.exists(f"{work}/model3.txt"):
        log("no stage-3 model; keeping stage-2 scores"); return
    b3 = lgb.Booster(model_file=f"{work}/model3.txt")
    c = pd.read_parquet(f"{d}/cands.parquet", columns=["r1", "r2"])
    p1 = np.load(f"{d}/p1.npy"); s2all = np.load(f"{d}/scores.npy")
    K = np.flatnonzero(p1 > M.PRUNE)
    C = M.context(c.r1.values[K], c.r2.values[K], p1[K])
    raw = np.load(f"{d}/s2raw.npy", mmap_mode="r")
    s2 = s2all[K]
    k3 = s2 > SB.KEEP3
    K3 = K[k3]
    ce = np.load(f"{d}/ceK.npy")[k3] if "ce" in b3.feature_name() else None
    X2 = M.stage2_matrix(np.asarray(raw[K3]), C[k3], ce); del C; gc.collect()
    _, s23t = F.load_tables(d)
    n_s2 = n_rows(f"{d}/s2.parquet")
    X3 = _x3(X2, s2[k3], c.r1.values[K3], c.r2.values[K3], n_s2, s23t)
    del X2, s23t; gc.collect()
    s3 = np.zeros(len(p1), np.float32)
    s3[K3] = b3.predict(X3)
    np.save(f"{d}/scores3.npy", s3)
    log(f"stage-3 rescored {len(K3):,} pairs")

def stage_write(work, out):
    """Apply the decision rule to saved scores and write both submission files."""
    d = f"{work}/test"
    meta = json.load(open(f"{work}/model_meta.json"))
    c = pd.read_parquet(f"{d}/cands.parquet", columns=["r1", "r2"])
    if os.path.exists(f"{d}/scores3.npy") and "threshold3" in meta:
        t = meta["threshold3"]; scores = np.load(f"{d}/scores3.npy"); log("using stage-3 scores")
    else:
        t = meta["threshold"]; scores = np.load(f"{d}/scores.npy")
    pred = M.decide(c, scores, t)
    os.makedirs(out, exist_ok=True)
    write_lists(f"{out}/matching_results.tsv", "matched_entity_ids", d,
                pred.r1.values, pred.r2.values)
    del pred, scores; gc.collect()
    write_lists(f"{out}/candidate_pairs.tsv", "candidate_entity_ids", d,
                c.r1.values, c.r2.values)
    log(f"wrote outputs to {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--work", default="work")
    ap.add_argument("--out", default="output")
    ap.add_argument("--train-frac", type=float, default=0.03)
    ap.add_argument("--stage", default="all")
    ap.add_argument("--gpu", action="store_true", help="add fine-tuned embedding stage (needs CUDA)")
    ap.add_argument("--ce", action="store_true", help="add fine-tuned cross-encoder feature (needs --gpu)")
    a = ap.parse_args()
    if a.stage == "all":
        # each stage in its own process so memory is fully returned between stages
        import subprocess
        stages = ["prep", "block_train", "block_test"]
        if a.gpu:
            stages += ["embed_train", "embed_test"]
            if a.ce:
                stages += ["ce_train"]
        stages += ["train", "train3", "predict", "rescore", "rescore3", "write"]
        env = dict(os.environ)
        for st in stages:
            cmd = [sys.executable, os.path.abspath(__file__), "--data", a.data, "--work", a.work,
                   "--out", a.out, "--train-frac", str(a.train_frac), "--stage", st]
            if st.startswith("embed") and env.get("NO_EMB") == "1":
                continue
            if st == "ce_train" and env.get("NO_CE") == "1":
                continue
            # GPU training steps use ONE GPU: multi-GPU DataParallel can deadlock on some cloud
            # machines (both GPUs at "100%" but no progress). Scoring still uses all GPUs.
            env_st = dict(env, CUDA_VISIBLE_DEVICES=env.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]) \
                if st in ("embed_train", "embed_test", "ce_train") else env
            r = subprocess.run(cmd, env=env_st)
            if r.returncode != 0 and env.get("NO_CE") != "1" and st in ("train", "rescore") \
                    and os.path.exists(f"{a.work}/ce_model/config.json"):
                # safety net for the cross-encoder: retrain / rescore without it
                log(f"stage {st} FAILED with cross-encoder; retrying WITHOUT cross-encoder")
                env["NO_CE"] = "1"
                for f in ("model2.txt", "model3.txt", "model3_rejected",
                          "test/scores.npy", "test/scores3.npy", "test/ceK.npy"):
                    if os.path.exists(f"{a.work}/{f}"):
                        os.remove(f"{a.work}/{f}")
                # (stage-1 scores of the test set do not depend on the cross-encoder: kept)
                for st2 in ["train", "train3"] + (["rescore"] if st == "rescore" else []):
                    cmd2 = cmd[:-1] + [st2]
                    r2 = subprocess.run(cmd2, env=env)
                    if r2.returncode != 0 and st2 not in ("train3", "rescore3"):
                        raise SystemExit(f"stage {st2} failed with exit code {r2.returncode}")
                continue
            if r.returncode != 0:
                if st == "ce_train":
                    log(f"stage ce_train FAILED (exit {r.returncode}); continuing WITHOUT cross-encoder")
                    env["NO_CE"] = "1"
                    continue
                if st in ("train3", "rescore3"):
                    # optional stage: on failure the stage-2 scores are used as before
                    log(f"stage {st} FAILED (exit {r.returncode}); using stage-2 scores")
                    for f in (f"{a.work}/model3.txt", f"{a.work}/test/scores3.npy"):
                        if os.path.exists(f):
                            os.remove(f)
                    continue
                if st.startswith("embed"):
                    # safety net: never lose the run because of the optional embedding stage -
                    # continue with the (already validated) non-embedding model instead
                    log(f"stage {st} FAILED (exit {r.returncode}); continuing WITHOUT embeddings")
                    env["NO_EMB"] = "1"
                    open(f"{a.work}/NO_EMB", "w").write("1")
                    continue
                raise SystemExit(f"stage {st} failed with exit code {r.returncode}")
        log("done")
    elif a.stage == "prep":
        stage_prep(a.data, a.work)
    elif a.stage == "block_train":
        stage_block(a.work, "train")
    elif a.stage == "train":
        stage_train(a.data, a.work, a.train_frac)
    elif a.stage == "block_test":
        stage_block(a.work, "test")
    elif a.stage == "predict":
        stage_predict(a.work, a.out)
    elif a.stage == "embed_train":
        stage_embed(a.data, a.work, "train")
    elif a.stage == "embed_test":
        stage_embed(a.data, a.work, "test")
    elif a.stage == "rescore":
        stage_rescore(a.work, a.out)
    elif a.stage == "ce_train":
        stage_ce(a.data, a.work)
    elif a.stage == "train3":
        stage_train3(a.data, a.work, a.train_frac)
    elif a.stage == "rescore3":
        stage_rescore3(a.work)
    elif a.stage == "write":
        stage_write(a.work, a.out)


if __name__ == "__main__":
    main()
