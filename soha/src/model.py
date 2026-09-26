"""Step 4: two-stage LightGBM matcher + decision rule + macro F0.5 evaluation."""
import numpy as np
import pandas as pd
import lightgbm as lgb

P1 = dict(objective="binary", learning_rate=0.08, num_leaves=127, min_data_in_leaf=100,
          feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
          verbose=-1, num_threads=int(__import__("os").environ.get("N_PROC", __import__("os").cpu_count())))
P2 = dict(P1, num_leaves=63)
N1_ROUNDS, N2_ROUNDS = 600, 300


def fit(X, y, params, rounds):
    return lgb.train(params, lgb.Dataset(X, y, free_raw_data=True), rounds)


def context_feats(pairs, p):
    """Stage-2 features: how a pair's stage-1 score compares with competing pairs."""
    d = pd.DataFrame({"r1": pairs["r1"].values, "r2": pairs["r2"].values, "p": p})
    g2 = d.groupby("r2")["p"]
    g1 = d.groupby("r1")["p"]
    out = pd.DataFrame(index=d.index)
    out["p"] = d.p
    mx2 = g2.transform("max")
    # best competing S1 for the same S2/S3 record (excluding this pair)
    d["_r"] = g2.rank(ascending=False, method="first")
    second2 = d.p.where(d._r == 2).groupby(d.r2).transform("max").fillna(0)
    out["p_rank_r2"] = d._r
    out["p_other_r2"] = np.where(d._r == 1, second2, mx2)
    out["p_gap_r2"] = out.p - out.p_other_r2
    out["p_rank_r1"] = g1.rank(ascending=False, method="first")
    out["p_max_r1"] = g1.transform("max")
    out["p_sum_r1"] = g1.transform("sum")
    out["n_hi_r1"] = (d.p > 0.5).groupby(d.r1).transform("sum")
    out["p_rel_r1"] = out.p / (out.p_max_r1 + 1e-6)
    return out


def decide(pairs, score, t):
    """Predicted pairs: score>t and pair is the best S1 for its S2/S3 record."""
    d = pd.DataFrame({"r1": pairs["r1"].values, "r2": pairs["r2"].values, "s": score})
    d = d[d.s > t]
    d = d.sort_values("s", ascending=False).drop_duplicates("r2")
    return d[["r1", "r2"]]


def macro_f05(pred, gt, s1_ids):
    """pred, gt: DataFrames (r1, r2); s1_ids: array of S1 rids in the evaluation set."""
    s1_ids = np.asarray(s1_ids)
    idx = pd.Index(s1_ids)
    pred = pred[pred.r1.isin(idx)]
    gt = gt[gt.r1.isin(idx)]
    tp = pred.merge(gt, on=["r1", "r2"]).groupby("r1").size()
    npred = pred.groupby("r1").size()
    ngt = gt.groupby("r1").size()
    df = pd.DataFrame(index=idx)
    df["tp"] = tp.reindex(idx).fillna(0).values
    df["np"] = npred.reindex(idx).fillna(0).values
    df["ng"] = ngt.reindex(idx).fillna(0).values
    P = np.where(df.np > 0, df.tp / np.maximum(df.np, 1), 0.0)
    R = np.where(df.ng > 0, df.tp / np.maximum(df.ng, 1), 0.0)
    f = np.where((P + R) > 0, 1.25 * P * R / np.maximum(0.25 * P + R, 1e-12), 0.0)
    f = np.where((df.np == 0) & (df.ng == 0), 1.0, f)
    return float(f.mean()), float(P[df.np > 0].mean() if (df.np > 0).any() else 0), \
        float(R[df.ng > 0].mean() if (df.ng > 0).any() else 0)


# ---------------------------------------------------------------- stage 2 (competition context)
S2_RAW = ["a_tset", "n_tsort", "nc_jacc", "num_subset", "bscore", "brank1", "brank2", "n_c1", "n_c2",
          "tsim", "trank", "esim", "erank"]
PRUNE = 0.03  # pairs with stage-1 score below this are dropped before stage 2 (saves memory)
CTX_NAMES = ["p1", "rank_r2", "p_other_r2", "gap_r2", "n_r2", "rank_r1", "pmax_r1", "rel_r1",
             "nhi_r1", "psum_r1", "second_r1"]


def _grp_stats(g, p):
    """For group ids g and scores p: rank within group (desc), best other score in the group,
    group size, group max, group sum, second best in group, count of p>0.5."""
    n = len(g)
    o = np.lexsort((-p, g))
    gs, ps = g[o], p[o]
    start = np.r_[True, gs[1:] != gs[:-1]]
    idx = np.arange(n)
    first = np.maximum.accumulate(np.where(start, idx, 0))
    rank = idx - first
    st = np.flatnonzero(start)
    size = np.diff(np.r_[st, n])
    gmax = ps[st]
    second = np.where(size > 1, ps[np.minimum(st + 1, n - 1)], 0.0)
    gsum = np.add.reduceat(ps, st)
    nhi = np.add.reduceat((ps > 0.5).astype(np.float32), st)
    gid = np.cumsum(start) - 1
    other = np.where(rank == 0, second[gid], gmax[gid])
    out = np.empty((n, 6), np.float32)
    out[o, 0] = rank; out[o, 1] = other; out[o, 2] = size[gid]
    out[o, 3] = gmax[gid]; out[o, 4] = gsum[gid]; out[o, 5] = nhi[gid]
    sec = np.empty(n, np.float32); sec[o] = second[gid]
    return out, sec


def context(r1, r2, p):
    p = p.astype(np.float32)
    a, _ = _grp_stats(r2.astype(np.int64), p)
    b, sec1 = _grp_stats(r1.astype(np.int64), p)
    C = np.empty((len(p), len(CTX_NAMES)), np.float32)
    C[:, 0] = p
    C[:, 1] = a[:, 0]; C[:, 2] = a[:, 1]; C[:, 3] = p - a[:, 1]; C[:, 4] = a[:, 2]
    C[:, 5] = b[:, 0]; C[:, 6] = b[:, 3]; C[:, 7] = p / (b[:, 3] + 1e-6)
    C[:, 8] = b[:, 5]; C[:, 9] = b[:, 4]; C[:, 10] = sec1
    return C


def stage2_matrix(raw, C, ce=None):
    """Stage-2 inputs; `ce` (cross-encoder probability) is appended when available."""
    if ce is None:
        return pd.DataFrame(np.hstack([raw.astype(np.float32), C]), columns=S2_RAW + CTX_NAMES)
    return pd.DataFrame(np.hstack([raw.astype(np.float32), C, np.asarray(ce, np.float32)[:, None]]),
                        columns=S2_RAW + CTX_NAMES + ["ce"])


def _grp_lean(g, p, out, cols):
    """Memory-lean group stats written into out[:, cols] (cols: rank, other, size, max, sum, nhi, second)."""
    n = len(g)
    o = np.lexsort((-p, g)).astype(np.int32)
    gs = g[o]
    start = np.empty(n, bool); start[0] = True; np.not_equal(gs[1:], gs[:-1], out=start[1:])
    del gs
    ps = p[o]
    idx = np.arange(n, dtype=np.int32)
    first = np.maximum.accumulate(np.where(start, idx, 0).astype(np.int32))
    rank = (idx - first).astype(np.float32)
    del idx
    st = np.flatnonzero(start).astype(np.int32)
    gid = (np.cumsum(start, dtype=np.int32) - 1)
    del start
    size = np.diff(np.r_[st, n]).astype(np.float32)
    gmax = ps[st]
    second = np.where(size > 1, ps[np.minimum(st + 1, n - 1)], 0.0).astype(np.float32)
    gsum = np.add.reduceat(ps, st).astype(np.float32)
    nhi = np.add.reduceat((ps > 0.5).astype(np.float32), st)
    del ps, st, first
    vals = {"rank": rank,
            "other": np.where(rank == 0, second[gid], gmax[gid]).astype(np.float32),
            "size": size[gid], "max": gmax[gid], "sum": gsum[gid], "nhi": nhi[gid],
            "second": second[gid]}
    for name, col in cols.items():
        tmp = np.empty(n, np.float32); tmp[o] = vals[name]; out[:, col] = tmp
        del tmp


def context_to(r1, r2, p, out):
    """Same columns as context(), written into a preallocated (possibly memmapped) array."""
    p = np.asarray(p, dtype=np.float32)
    out[:, 0] = p
    _grp_lean(r2, p, out, {"rank": 1, "other": 2, "size": 4})
    out[:, 3] = p - out[:, 2]
    _grp_lean(r1, p, out, {"rank": 5, "max": 6, "nhi": 8, "sum": 9, "second": 10})
    out[:, 7] = p / (out[:, 6] + 1e-6)
    return out
