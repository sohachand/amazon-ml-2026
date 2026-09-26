"""Stage 3: sibling evidence.

One Source-1 business usually has 3-4 true matches spread over Source 2 and Source 3, and those
records also resemble EACH OTHER.  After stage 2 the easy matches of an S1 entity are known with
high confidence; a hard candidate (heavy typo, transliteration, missing address) that closely
resembles one of those confident "siblings" is very likely a match too, while a false candidate
usually does not resemble them.  Stage 2 judged every (S1, candidate) pair on its own evidence
plus score competition; stage 3 adds this record-to-record evidence.

sib_features(): for each pair (r1, r2) with stage-2 score s, take the top-3 other candidates of the
same r1 with s > SIB_T ("siblings") and compare record r2 with each sibling (names, core names,
addresses, numbers).  Aggregates (max / score-weighted) become stage-3 features.
"""
import os
import multiprocessing as mp

import numpy as np
import pyarrow as pa
from rapidfuzz import fuzz

SIB_T = 0.5          # a sibling must have stage-2 score above this
TOP_SIB = 3          # compare with at most this many siblings
KEEP3 = 0.01         # pairs with stage-2 score below this are not re-scored by stage 3
SIB_NAMES = ["sib_n", "sib_nm", "sib_nc", "sib_ad", "sib_num", "sib_best_s", "sib_nmean",
             "n_conf", "n_conf_same", "self_is_sib"]
_G = {}


def _cmp(span):
    lo, hi = span
    a = pa.array(_G["a"][lo:hi]); b = pa.array(_G["b"][lo:hi])
    T = _G["T"]
    A = {c: T.column(c).take(a).to_pylist() for c in ("name_lat", "name_core", "addr_lat", "nums")}
    B = {c: T.column(c).take(b).to_pylist() for c in ("name_lat", "name_core", "addr_lat", "nums")}
    out = np.empty((hi - lo, 4), np.float32)
    for i in range(hi - lo):
        a1, a2 = A["addr_lat"][i], B["addr_lat"][i]
        n1, n2 = set(A["nums"][i].split()), set(B["nums"][i].split())
        out[i, 0] = fuzz.token_set_ratio(A["name_lat"][i], B["name_lat"][i])
        out[i, 1] = fuzz.ratio(A["name_core"][i], B["name_core"][i])
        out[i, 2] = fuzz.token_set_ratio(a1, a2) if a1 and a2 else -1
        u = n1 | n2
        out[i, 3] = len(n1 & n2) / len(u) if u else -1
    return out


def _record_sims(s23t, a, b, n_proc=None, chunk=50_000):
    """String similarities between S2/S3 records a[i] and b[i]."""
    _G["T"], _G["a"], _G["b"] = s23t, a, b
    n = len(a)
    out = np.empty((n, 4), np.float32)
    spans = [(i, min(i + chunk, n)) for i in range(0, n, chunk)]
    with mp.get_context("fork").Pool(n_proc or int(os.environ.get("N_PROC", os.cpu_count() or 2))) as pool:
        for (lo, hi), r in zip(spans, pool.imap(_cmp, spans, chunksize=1)):
            out[lo:hi] = r
    return out


def sib_features(r1, r2, s, n_s2, s23t):
    """r1, r2, s: arrays of the pairs to score (all candidates of each r1 should be present).
    Returns float32 array (n, len(SIB_NAMES))."""
    n = len(r1)
    r1 = r1.astype(np.int64); r2 = r2.astype(np.int64); s = s.astype(np.float32)
    o = np.lexsort((-s, r1))
    g = r1[o]
    start = np.r_[True, g[1:] != g[:-1]]
    idx = np.arange(n)
    rank = idx - np.maximum.accumulate(np.where(start, idx, 0))
    conf = s[o] > SIB_T
    # per r1 group: count of confident candidates overall and per source (S2 / S3)
    gid = np.cumsum(start) - 1
    is3 = (r2[o] >= n_s2)
    n_conf_g = np.bincount(gid, weights=conf, minlength=gid[-1] + 1 if n else 0)
    n_conf3_g = np.bincount(gid, weights=conf & is3, minlength=gid[-1] + 1 if n else 0)
    feats = np.zeros((n, len(SIB_NAMES)), np.float32)
    feats[:, 2:5] = -1
    self_conf = conf.astype(np.float32)
    nc = n_conf_g[gid] - self_conf
    nc_same = np.where(is3, n_conf3_g[gid], n_conf_g[gid] - n_conf3_g[gid]) - self_conf
    tmp = np.empty(n, np.float32)
    tmp[o] = nc; feats[:, 7] = tmp
    tmp[o] = nc_same; feats[:, 8] = tmp
    tmp[o] = ((rank < TOP_SIB) & conf).astype(np.float32); feats[:, 9] = tmp
    # sibling slots: the top-TOP_SIB (by s) confident candidates of each group
    first = np.flatnonzero(start)
    sizes = np.diff(np.r_[first, n])
    pa_list, pb_list, ps_list, own_list = [], [], [], []
    for k in range(TOP_SIB):
        has = sizes > k
        pos = first[has] + k                      # sorted position of the k-th sibling
        ok = conf[pos]
        pos = pos[ok]
        grp = np.flatnonzero(has)[ok]
        # every member of that group (except the sibling itself) is compared with it
        m = np.isin(gid, grp)
        mem = np.flatnonzero(m)
        sib_pos = np.empty(gid[-1] + 1 if n else 0, np.int64); sib_pos[grp] = pos
        sp = sib_pos[gid[mem]]
        keep = sp != mem
        mem, sp = mem[keep], sp[keep]
        pa_list.append(r2[o][mem]); pb_list.append(r2[o][sp]); ps_list.append(s[o][sp])
        own_list.append(mem)
    if not pa_list or not sum(len(x) for x in pa_list):
        return feats
    A = np.concatenate(pa_list); B = np.concatenate(pb_list); S = np.concatenate(ps_list)
    OWN = np.concatenate(own_list)
    sims = _record_sims(s23t, A, B)
    # aggregate per pair (sorted position OWN)
    cnt = np.bincount(OWN, minlength=n).astype(np.float32)
    agg = np.full((n, 4), -1, np.float32)
    for j in range(4):
        np.maximum.at(agg[:, j], OWN, sims[:, j])
    nmean = np.bincount(OWN, weights=sims[:, 0], minlength=n) / np.maximum(cnt, 1)
    # score of the sibling most similar by name
    best_s = np.zeros(n, np.float32)
    ordr = np.lexsort((sims[:, 0], OWN))           # last per OWN = highest name sim
    last = np.r_[OWN[ordr][1:] != OWN[ordr][:-1], True]
    best_s[OWN[ordr][last]] = S[ordr][last]
    sorted_feats = np.column_stack([cnt, agg[:, 0], agg[:, 1], agg[:, 2], agg[:, 3], best_s,
                                    np.where(cnt > 0, nmean, -1)]).astype(np.float32)
    out = np.empty((n, 7), np.float32); out[o] = sorted_feats
    feats[:, :7] = out
    return feats
