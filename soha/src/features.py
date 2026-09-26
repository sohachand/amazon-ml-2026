"""Step 3: pairwise features for (S1 record, S2/S3 record) candidate pairs.

Record tables are held as Arrow tables (compact); worker processes are forked and gather
only the rows of their chunk, so memory stays bounded even for tens of millions of pairs.
"""
import os
import multiprocessing as mp

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein

FCOLS = ["country", "name_lat", "name_full", "name_core", "addr_lat", "addr_tok", "nums",
         "nonlatin", "dom"]
_NONLATIN_RE = "[^\\x00-\\x{024F}\\s\\p{P}\\p{S}\\p{N}]"
_G = {}

FEAT_NAMES = [
    "n_ratio", "n_tsort", "n_tset", "n_partial", "nf_ratio", "nf_tset", "nc_ratio", "nc_tset",
    "nc_tsort", "nj_ratio", "nj_partial", "nj_jw", "nc_inter", "nc_union", "nc_jacc", "nc_len1",
    "nc_len2", "n_idf1", "n_idf2", "n_idfi", "has_addr", "a_tset", "a_ratio", "a_partial",
    "at_inter", "at_len1", "at_len2", "a_idf1", "a_idf2", "a_idfi", "num_inter", "num_len1",
    "num_len2", "num_first_eq", "num_subset", "num_disjoint", "num_maxlen_common", "b_nonlatin",
    "b_dom", "nj_lev",
]
BLOCK_FEATS = ["bscore", "bits", "brank1", "brank2", "is_s3", "n_c1", "n_c2", "tsim", "trank",
               "esim", "erank"]
ALL_FEATS = FEAT_NAMES + BLOCK_FEATS


def _read(path):
    cols = [c for c in FCOLS if c != "nonlatin"] + ["name_raw"]
    t = pq.read_table(path, columns=cols)
    nl = pc.cast(pc.match_substring_regex(t.column("name_raw"), _NONLATIN_RE), pa.int8())
    t = t.drop(["name_raw"]).append_column("nonlatin", nl)
    t = t.set_column(t.schema.get_field_index("country"), "country",
                     pc.cast(t.column("country"), pa.string()))
    return t.combine_chunks()


def load_tables(d):
    """Arrow tables for S1 and S2+S3 (S2 rows first, then S3) of a normalised split dir."""
    s1 = _read(f"{d}/s1.parquet")
    s23 = pa.concat_tables([_read(f"{d}/s{k}.parquet") for k in (2, 3)]).combine_chunks()
    return s1, s23


def token_idf(s23, col):
    """IDF of tokens in `col` per country, stored compactly as (sorted hash array, idf array).
    Tokens are already unique within a record, so value counts = document frequencies."""
    n = s23.num_rows
    hs, vs = [], []
    for c in pc.unique(s23.column("country")).to_pylist():
        m = pc.equal(s23.column("country"), c)
        toks = pc.list_flatten(pc.utf8_split_whitespace(pc.filter(s23.column(col), m)))
        vc = pc.value_counts(toks)
        vals = vc.field("values").to_pylist()
        hs.append(np.fromiter((hash(c + "|" + t) for t in vals), np.int64, len(vals)))
        vs.append(np.log(n / (1.0 + vc.field("counts").to_numpy())).astype(np.float32))
    h = np.concatenate(hs); v = np.concatenate(vs)
    o = np.argsort(h)
    return h[o], v[o]


def _local_idf(tok_lists, country, table):
    """Small per-chunk dict token->idf for all tokens appearing in the chunk."""
    h, v = table
    keys = {c + "|" + t for c, s in zip(country, tok_lists) for t in s.split()}
    keys = list(keys)
    if not keys:
        return {}
    q = np.fromiter((hash(k) for k in keys), np.int64, len(keys))
    pos = np.clip(np.searchsorted(h, q), 0, len(h) - 1)
    hit = h[pos] == q
    return {k: float(v[p]) for k, p, ok in zip(keys, pos, hit) if ok}


def _idf_overlap(a, b, c, idf, default):
    if not a or not b:
        return 0.0, 0.0, 0.0
    wa = {t: idf.get(c + "|" + t, default) for t in a}
    wb = {t: idf.get(c + "|" + t, default) for t in b}
    inter = sum(wa[t] for t in a & b)
    sa, sb = sum(wa.values()), sum(wb.values())
    return (inter / sa if sa else 0.0), (inter / sb if sb else 0.0), inter


def _nonlatin(s):
    for ch in s:
        if ord(ch) > 0x24F and ch.isalpha():
            return 1
    return 0


def _pair_feats(span):
    lo, hi = span
    r1 = pa.array(_G["r1"][lo:hi]); r2 = pa.array(_G["r2"][lo:hi])
    A = {c: _G["A"].column(c).take(r1).to_pylist() for c in FCOLS if c not in ("nonlatin", "dom")}
    B = {c: _G["B"].column(c).take(r2).to_pylist() for c in FCOLS}
    ND, AD = _G["nd"], _G["ad"]
    nidf = _local_idf(A["name_core"] + B["name_core"], B["country"] + B["country"], _G["nidf"])
    aidf = _local_idf(A["addr_tok"] + B["addr_tok"], B["country"] + B["country"], _G["aidf"])
    rows = []
    for i in range(hi - lo):
        c = A["country"][i]
        n1, n2 = A["name_lat"][i], B["name_lat"][i]
        f1, f2 = A["name_full"][i], B["name_full"][i]
        k1, k2 = A["name_core"][i], B["name_core"][i]
        ks1, ks2 = set(k1.split()), set(k2.split())
        a1, a2 = A["addr_lat"][i], B["addr_lat"][i]
        at1, at2 = set(A["addr_tok"][i].split()), set(B["addr_tok"][i].split())
        nm1, nm2 = A["nums"][i].split(), B["nums"][i].split()
        sn1, sn2 = set(nm1), set(nm2)
        io1, io2, ioi = _idf_overlap(ks1, ks2, c, nidf, ND)
        ao1, ao2, aoi = _idf_overlap(at1, at2, c, aidf, AD)
        j1, j2 = k1.replace(" ", ""), k2.replace(" ", "")
        inter_n = len(ks1 & ks2)
        un = len(ks1 | ks2)
        has_a = (1 if a1 else 0) + 2 * (1 if a2 else 0)
        both = has_a == 3
        rows.append((
            fuzz.ratio(n1, n2), fuzz.token_sort_ratio(n1, n2), fuzz.token_set_ratio(n1, n2),
            fuzz.partial_ratio(n1, n2) if n1 and n2 else 0,
            fuzz.ratio(f1, f2), fuzz.token_set_ratio(f1, f2),
            fuzz.ratio(k1, k2), fuzz.token_set_ratio(k1, k2), fuzz.token_sort_ratio(k1, k2),
            fuzz.ratio(j1, j2), fuzz.partial_ratio(j1, j2) if j1 and j2 else 0,
            JaroWinkler.similarity(j1, j2),
            inter_n, un, inter_n / un if un else 0.0, len(ks1), len(ks2),
            io1, io2, ioi,
            has_a,
            fuzz.token_set_ratio(a1, a2) if both else -1,
            fuzz.ratio(a1, a2) if both else -1,
            fuzz.partial_ratio(a1, a2) if both else -1,
            len(at1 & at2), len(at1), len(at2), ao1, ao2, aoi,
            len(sn1 & sn2), len(sn1), len(sn2),
            int(bool(nm1) and bool(nm2) and nm1[0] == nm2[0]),
            int(bool(sn2) and sn2 <= sn1), int(bool(sn1) and bool(sn2) and not (sn1 & sn2)),
            max((len(x) for x in sn1 & sn2), default=0),
            B["nonlatin"][i], int(B["dom"][i]),
            Levenshtein.distance(j1, j2),
        ))
    return np.asarray(rows, dtype=np.float32).reshape(-1, len(FEAT_NAMES))


def setup(s1, s23, nidf, aidf):
    _G["A"], _G["B"], _G["nidf"], _G["aidf"] = s1, s23, nidf, aidf
    _G["nd"] = float(nidf[1].max()) if len(nidf[1]) else 1.0
    _G["ad"] = float(aidf[1].max()) if len(aidf[1]) else 1.0


def compute(pairs, n_s2, n_c1_all, n_c2_all, n_proc=None, chunk=20_000):
    """pairs: DataFrame (r1, r2, bscore, bits, brank1, brank2). Returns float32 feature frame."""
    _G["r1"] = pairs["r1"].values; _G["r2"] = pairs["r2"].values
    n = len(pairs)
    spans = [(i, min(i + chunk, n)) for i in range(0, n, chunk)]
    n_proc = n_proc or int(os.environ.get("N_PROC", os.cpu_count() or 2))
    ctx = mp.get_context("fork")  # workers share the Arrow tables copy-on-write
    out = np.empty((n, len(ALL_FEATS)), np.float32)   # filled in place: no extra copies
    nf = len(FEAT_NAMES)
    with ctx.Pool(n_proc) as pool:
        for (lo, hi), r in zip(spans, pool.imap(_pair_feats, spans, chunksize=1)):
            out[lo:hi, :nf] = r
    for i, c in enumerate(["bscore", "bits", "brank1", "brank2"]):
        out[:, nf + i] = pairs[c].values
    out[:, nf + 4] = pairs["r2"].values >= n_s2
    out[:, nf + 5] = n_c1_all[pairs["r1"].values]
    out[:, nf + 6] = n_c2_all[pairs["r2"].values]
    out[:, nf + 7] = pairs["tsim"].values
    out[:, nf + 8] = pairs["trank"].values
    # embedding similarity (GPU pipeline); neutral values when embeddings were not computed
    use_emb = "esim" in pairs and os.environ.get("NO_EMB") != "1"
    out[:, nf + 9] = pairs["esim"].values if use_emb else 0.0
    out[:, nf + 10] = pairs["erank"].values if use_emb else 999
    return pd.DataFrame(out, columns=ALL_FEATS, copy=False)
