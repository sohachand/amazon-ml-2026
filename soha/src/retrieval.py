"""Stage 2b: TF-IDF nearest-neighbour candidate retrieval (complements key blocking).

Key blocking must drop or truncate very large blocks (common names such as "Sai Consulting",
common address words such as "Delhi"), so some true matches never make it into its top-K.
Here every record becomes a sparse TF-IDF vector over its core-name skeleton tokens,
address tokens and address numbers (fitted on S2+S3 of the same country; the 1% most frequent
tokens are ignored).  For every S1 record we retrieve the top-N most cosine-similar S2/S3
records with a multi-threaded sparse top-n matrix product (sparse_dot_topn).
The union with the blocking candidates is returned, and the cosine similarity and its rank
within the S1 record are attached to *every* candidate pair as features (tsim, trank).
Memory is kept low by working one country at a time on integer pair codes.
"""
import gc
import os

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

TOP_N = int(os.environ.get("TFIDF_TOP_N", 30))
MIN_SIM = 0.05
SH = np.int64(1 << 24)


def _docs(path, country):
    """Documents 'n<name tok> a<addr tok> #<number>' for the rows of one country (file order)."""
    t = pq.read_table(path, columns=["name_core", "addr_tok", "nums"],
                      filters=[("country", "=", country)])
    out = [" ".join(["n" + x for x in a.split()] + ["a" + x for x in b.split()]
                    + ["#" + x for x in c.split()])
           for a, b, c in zip(t.column("name_core").to_pylist(), t.column("addr_tok").to_pylist(),
                              t.column("nums").to_pylist())]
    return out


def _rowdot(Q, M, l1, l2, step=1_000_000):
    """Cosine similarity for explicit (row of Q, row of M) pairs, in chunks."""
    out = np.zeros(len(l1), np.float32)
    for a in range(0, len(l1), step):
        b = min(a + step, len(l1))
        out[a:b] = np.asarray(Q[l1[a:b]].multiply(M[l2[a:b]]).sum(axis=1)).ravel()
    return out


def _country_rows(path, country):
    col = pq.read_table(path, columns=["country"]).column(0)
    return np.flatnonzero(pc.equal(col, country).to_numpy(zero_copy_only=False)).astype(np.int32)


def add_tfidf(d, cands, n_threads=None, verbose=True):
    """cands: blocking candidates (r1, r2, bscore, bits, brank1, brank2). Returns the union with
    TF-IDF top-N neighbours plus columns tsim (cosine) and trank (rank of tsim within the S1)."""
    n_threads = n_threads or int(os.environ.get("N_PROC", os.cpu_count() or 2))
    n2 = pq.ParquetFile(f"{d}/s2.parquet").metadata.num_rows
    n1 = pq.ParquetFile(f"{d}/s1.parquet").metadata.num_rows
    countries = sorted(set(pc.unique(pq.read_table(f"{d}/s1.parquet", columns=["country"]).column(0)).to_pylist()))
    bcode = cands.r1.values.astype(np.int64) * SH + cands.r2.values
    codes_all, sims_all = [], []
    for c in countries:
        i1 = _country_rows(f"{d}/s1.parquet", c)
        i2 = np.concatenate([_country_rows(f"{d}/s2.parquet", c),
                             _country_rows(f"{d}/s3.parquet", c) + n2]).astype(np.int32)
        if not len(i1) or not len(i2):
            continue
        vec = TfidfVectorizer(token_pattern=r"\S+", lowercase=False, sublinear_tf=True,
                              max_df=0.01, dtype=np.float32)
        docs = _docs(f"{d}/s2.parquet", c) + _docs(f"{d}/s3.parquet", c)
        M = vec.fit_transform(docs).tocsr()
        del docs
        Q = vec.transform(_docs(f"{d}/s1.parquet", c)).tocsr()
        MT = M.T.tocsr()
        found = []
        for a in range(0, Q.shape[0], 50_000):
            R = sp_matmul_topn(Q[a:a + 50_000], MT, top_n=TOP_N, threshold=MIN_SIM,
                               n_threads=n_threads).tocoo()
            found.append(i1[R.row + a].astype(np.int64) * SH + i2[R.col])
        del MT
        m1 = np.zeros(n1, bool); m1[i1] = True
        bc = bcode[m1[cands.r1.values]]
        u = np.unique(np.concatenate(found + [bc]))
        del found, bc
        r1 = (u // SH).astype(np.int32); r2 = (u % SH).astype(np.int32)
        sim = _rowdot(Q, M, np.searchsorted(i1, r1), np.searchsorted(i2, r2))
        codes_all.append(u); sims_all.append(sim)
        if verbose:
            print(f"  tfidf {c!r}: union {len(u):,} pairs", flush=True)
        del M, Q, vec, u, r1, r2, sim
        gc.collect()
    u = np.concatenate(codes_all); sim = np.concatenate(sims_all)
    del codes_all, sims_all
    o = np.argsort(u); u, sim = u[o], sim[o]
    # attach blocking columns (pairs found only by tf-idf get neutral values)
    ob = np.argsort(bcode); bs = bcode[ob]
    pos = np.clip(np.searchsorted(bs, u), 0, len(bs) - 1)
    inb = bs[pos] == u
    src = ob[pos]
    out = pd.DataFrame({"r1": (u // SH).astype(np.int32), "r2": (u % SH).astype(np.int32)})
    for col, fill, dt in [("bscore", 0, np.float32), ("bits", 0, np.int16),
                          ("brank1", 999, np.int16), ("brank2", 999, np.int16)]:
        v = np.full(len(u), fill, dt)
        v[inb] = cands[col].values[src[inb]]
        out[col] = v
    out["tsim"] = sim.astype(np.float32)
    # rank of tsim within each S1 (u is sorted by r1 already)
    oo = np.lexsort((-sim, out.r1.values))
    g = out.r1.values[oo]
    start = np.r_[True, g[1:] != g[:-1]]
    idx = np.arange(len(g))
    rk = np.empty(len(g), np.int32)
    rk[oo] = idx - np.maximum.accumulate(np.where(start, idx, 0))
    out["trank"] = np.minimum(rk, 999).astype(np.int16)
    return out
