"""Step 2: candidate generation (blocking) - vectorised, memory-bounded.

Complementary blocking keys (all within the same country label):
  N2  pair of core-name skeleton tokens          (robust to one typo'd / missing token)
  N1  single core-name skeleton token            (only kept if the block is small)
  NN  core-name token + address number           (short/ambiguous names with an address)
  AN  address number + address token             (name missing / DBA / transliterated)
  NJ  concatenated core-name skeleton (+sorted)  (domain-style names: foobar.com, spacing)
Blocks larger than a cap (per side) are dropped: they carry little signal.
Every shared key adds weight w_type / log2(1 + block size) to the pair's blocking score.
Each S1 record keeps its top-K candidates and each S2/S3 record its top-M S1 records.
"""
import gc
import numpy as np
import pandas as pd

KEY_TYPES = ["N2", "N1", "NN", "AN", "NJ", "N2N", "ANN", "AT2", "NT", "NE"]
CAP = {"N2": 150, "N1": 40, "NN": 100, "AN": 100, "NJ": 100, "N2N": 60, "ANN": 80, "AT2": 60,
       "NT": 40, "NE": 300, "NX": 400}
TYPE_W = {"N2": 1.0, "N1": 0.8, "NN": 1.0, "AN": 1.2, "NJ": 1.0, "N2N": 1.5, "ANN": 1.2, "AT2": 1.0,
          "NT": 1.3, "NE": 0.6, "NX": 0.7}
TOP_K = 40
NPARTS = {"N2": 4, "NN": 4, "AN": 6, "N1": 2, "NJ": 1, "N2N": 4, "ANN": 2, "AT2": 6, "NT": 6, "NE": 2, "NX": 2}
BIT = {k: np.int16(1 << i) for i, k in enumerate(KEY_TYPES)}  # 10 types fit in int16
NT, NA, NN_ = 5, 5, 3  # max name tokens, address tokens, numbers used


def _iter(x, step=500_000):
    """Iterate python strings from a (possibly arrow-backed) Series/array/list in slices."""
    if isinstance(x, list):
        yield from x
        return
    x = pd.Series(x) if not isinstance(x, pd.Series) else x
    for a in range(0, len(x), step):
        yield from x.iloc[a:a + step].tolist()


class Vocab:
    def __init__(self):
        self.d = {}

    def ids(self, series, width, filt=None):
        """Series of space-joined tokens -> int64 matrix [n, width] of token ids (-1 pad)."""
        d = self.d
        out = np.full((len(series), width), -1, dtype=np.int32)
        for i, s in enumerate(_iter(series)):
            if not s:
                continue
            j = 0
            for t in s.split():
                if filt is not None and not filt(t):
                    continue
                v = d.get(t)
                if v is None:
                    v = len(d)
                    d[t] = v
                out[i, j] = v
                j += 1
                if j == width:
                    break
        return out

    def one(self, values):
        d = self.d
        out = np.full(len(values), -1, dtype=np.int32)
        for i, s in enumerate(_iter(values)):
            if s:
                v = d.get(s)
                if v is None:
                    v = len(d)
                    d[s] = v
                out[i] = v
        return out


def _encode(df, voc):
    name = voc.ids(df.name_core, NT, lambda t: not t.isdigit())
    nums = voc.ids(df.nums, NN_, lambda t: t != "0")
    addr = voc.ids(df.addr_tok, NA)
    nj1 = voc.one(["#" + s.replace(" ", "") if s else "" for s in _iter(df.name_core)])
    nj2 = voc.one(["#" + "".join(sorted(s.split())) if s.count(" ") else "" for s in _iter(df.name_core)])
    return name, nums, addr, nj1, nj2


def _keys(enc, kt, V, part=0, nparts=1, side=1):
    """(keys int64, rows int32) for one key type restricted to one hash partition."""
    name, nums, addr, nj1, nj2 = [np.asarray(x, dtype=np.int64) for x in enc]
    n = len(name)
    rows = np.arange(n, dtype=np.int32)
    ks, rs = [], []

    def add(k, m):
        if nparts > 1:
            m = m & ((k % nparts) == part)
        ks.append(k[m]); rs.append(rows[m])

    if kt == "N2":
        for a in range(NT):
            for b in range(a + 1, NT):
                x, y = name[:, a], name[:, b]
                m = (x >= 0) & (y >= 0) & (x != y)
                add(np.minimum(x, y) * V + np.maximum(x, y), m)
    elif kt == "N1":
        for a in range(NT):
            x = name[:, a]
            add(x, x >= 0)
    elif kt == "NN":
        for a in range(3):
            for b in range(NN_):
                x, y = name[:, a], nums[:, b]
                add(x * V + y, (x >= 0) & (y >= 0))
    elif kt == "AN":
        for a in range(NN_):
            for b in range(NA):
                x, y = nums[:, a], addr[:, b]
                add(x * V + y, (x >= 0) & (y >= 0))
    elif kt == "N2N":
        M = np.int64(1000003)
        for a in range(4):
            for b in range(a + 1, 4):
                x, y = name[:, a], name[:, b]
                base = np.minimum(x, y) * V + np.maximum(x, y)
                for n in range(2):
                    z = nums[:, n]
                    add(base * M + z, (x >= 0) & (y >= 0) & (x != y) & (z >= 0))
    elif kt == "ANN":
        for a in range(NN_):
            for b in range(a + 1, NN_):
                x, y = nums[:, a], nums[:, b]
                add(np.minimum(x, y) * V + np.maximum(x, y), (x >= 0) & (y >= 0) & (x != y))
    elif kt == "NE":
        # name-only keys with a much larger cap, emitted on the S2/S3 side only for records that
        # have no address at all (their name is the only evidence, so common names must be kept)
        noaddr = (addr[:, 0] < 0) & (nums[:, 0] < 0)
        sel = noaddr if side == 2 else np.ones(n, bool)
        for a in range(4):
            for b in range(a + 1, 4):
                x, y = name[:, a], name[:, b]
                add(np.minimum(x, y) * V + np.maximum(x, y) + 7, sel & (x >= 0) & (y >= 0) & (x != y))
        add(nj1 * 3 + 1, sel & (nj1 >= 0))
    elif kt == "NX":
        # exact multi-token core name (order-free): allowed to form big blocks, because business
        # names repeat a lot; the address keys then decide the ranking inside the block
        add(nj2, nj2 >= 0)
    elif kt == "NT":
        # core-name token x address token: e.g. "oncology" + "sebring"
        for a in range(3):
            for b in range(NA):
                x, y = name[:, a], addr[:, b]
                add(x * V + y, (x >= 0) & (y >= 0))
    elif kt == "AT2":
        for a in range(NA):
            for b in range(a + 1, NA):
                x, y = addr[:, a], addr[:, b]
                add(np.minimum(x, y) * V + np.maximum(x, y), (x >= 0) & (y >= 0) & (x != y))
    elif kt == "NJ":
        add(nj1, nj1 >= 0)
        add(nj2, nj2 >= 0)
    k = np.concatenate(ks); r = np.concatenate(rs)
    # dedupe (key,row) so a record counts once per block
    if len(k):
        o = np.lexsort((r, k)); k, r = k[o], r[o]
        keep = np.ones(len(k), bool); keep[1:] = (k[1:] != k[:-1]) | (r[1:] != r[:-1])
        k, r = k[keep], r[keep]
    return k, r


def _counts(k):
    # k sorted -> block sizes per element
    if not len(k):
        return np.zeros(0, np.int32)
    b = np.flatnonzero(np.r_[True, k[1:] != k[:-1], True])
    sz = np.diff(b)
    return np.repeat(sz, sz).astype(np.int32)


def _country_block(e1, e2, V, verbose, tmp="/tmp/er_block"):
    import os, shutil
    SH = np.int64(1 << 24)
    Q = 8
    os.makedirs(tmp, exist_ok=True)
    for kt in KEY_TYPES:
      P = NPARTS.get(kt, 1)
      acc_c, acc_w = [], []
      for part in range(P):
        k1, r1 = _keys(e1, kt, V, part, P)
        k2, r2 = _keys(e2, kt, V, part, P, side=2)
        c1, c2 = _counts(k1), _counts(k2)
        m1, m2 = c1 <= CAP[kt], c2 <= CAP[kt]
        a = pd.DataFrame({"k": k1[m1], "r1": r1[m1]})
        b = pd.DataFrame({"k": k2[m2], "r2": r2[m2], "c2": c2[m2]})
        del k1, r1, k2, r2, c1, c2, m1, m2
        j = a.merge(b, on="k", how="inner")
        del a, b
        code = j["r1"].values.astype(np.int64) * SH + j["r2"].values
        w = (TYPE_W[kt] / np.log2(1.0 + j["c2"].values)).astype(np.float32)
        del j
        code, w = _reduce(code, w, np.maximum)
        acc_c.append(code); acc_w.append(w)
        del code, w
        gc.collect()
      uc, w = _reduce(np.concatenate(acc_c), np.concatenate(acc_w), np.maximum)  # one vote per type
      del acc_c, acc_w
      if True:
        q_of = ((uc // SH) % Q).astype(np.int8)
        for q in range(Q):
            m = q_of == q
            np.save(f"{tmp}/{kt}_{q}_c.npy", uc[m]); np.save(f"{tmp}/{kt}_{q}_w.npy", w[m])
        del q_of, m
        if verbose:
            print(f"    {kt}: {len(uc):,} pairs", flush=True)
        del uc, w
        gc.collect()
    del e1, e2
    # aggregate across key types in r1-partitions (bounded memory); keep top-K per r1
    outs = []
    for q in range(Q):
        cs, ws, bs = [], [], []
        for kt in KEY_TYPES:
            cq = np.load(f"{tmp}/{kt}_{q}_c.npy"); cs.append(cq)
            ws.append(np.load(f"{tmp}/{kt}_{q}_w.npy"))
            bs.append(np.full(len(cq), BIT[kt], np.int16))
        code = np.concatenate(cs); w = np.concatenate(ws); bits = np.concatenate(bs)
        del cs, ws, bs
        o = np.argsort(code, kind="stable")
        code, w, bits = code[o], w[o], bits[o]
        del o
        st = np.flatnonzero(np.r_[True, code[1:] != code[:-1]])
        uc = code[st]
        r1 = (uc // SH).astype(np.int32); r2 = (uc % SH).astype(np.int32)
        bs_ = np.add.reduceat(w, st).astype(np.float32)
        bt = np.bitwise_or.reduceat(bits, st).astype(np.int16)
        del code, w, bits, uc, st
        o = np.lexsort((-bs_, r1))           # by r1, then score desc
        rk = _group_rank(r1[o])
        keep = o[rk < TOP_K]; rk = rk[rk < TOP_K]
        outs.append(pd.DataFrame({"r1": r1[keep], "r2": r2[keep], "bscore": bs_[keep],
                                  "bits": bt[keep], "brank1": rk.astype(np.int16)}))
        del r1, r2, bs_, bt, o, keep, rk
        gc.collect()
    shutil.rmtree(tmp, ignore_errors=True)
    return pd.concat(outs, ignore_index=True)


def _group_rank(g_sorted):
    """0-based position within runs of equal values of a sorted array."""
    n = len(g_sorted)
    if not n:
        return np.zeros(0, np.int32)
    start = np.r_[True, g_sorted[1:] != g_sorted[:-1]]
    idx = np.arange(n)
    first = np.maximum.accumulate(np.where(start, idx, 0))
    return (idx - first).astype(np.int32)


def _reduce(code, w, ufunc):
    o = np.argsort(code, kind="stable")
    code, w = code[o], w[o]
    st = np.flatnonzero(np.r_[True, code[1:] != code[:-1]]) if len(code) else np.zeros(0, int)
    if not len(code):
        return code, w
    return code[st], ufunc.reduceat(w, st)


def encode_to_disk(d, out):
    """Encode all countries of a normalised split directory into npy files (run in a child process)."""
    import os
    os.makedirs(out, exist_ok=True)
    cols = ["country", "name_core", "nums", "addr_tok"]
    s1 = pd.read_parquet(f"{d}/s1.parquet", columns=cols, dtype_backend="pyarrow")
    s23 = pd.concat([pd.read_parquet(f"{d}/s{k}.parquet", columns=cols, dtype_backend="pyarrow")
                     for k in (2, 3)], ignore_index=True)
    ctrs = sorted(set(s1.country.unique()) & set(s23.country.unique()))
    for ci, c in enumerate(ctrs):
        i1 = np.flatnonzero((s1.country == c).to_numpy(dtype=bool)).astype(np.int32)
        i2 = np.flatnonzero((s23.country == c).to_numpy(dtype=bool)).astype(np.int32)
        voc = Vocab()
        e1 = _encode(s1.iloc[i1], voc)
        e2 = _encode(s23.iloc[i2], voc)
        np.savez(f"{out}/c{ci}.npz", i1=i1, i2=i2, V=np.int64(len(voc.d) + 1),
                 **{f"a{j}": x for j, x in enumerate(e1)}, **{f"b{j}": x for j, x in enumerate(e2)})
        print(f"  encoded country={c!r}: S1={len(i1):,} S23={len(i2):,} V={len(voc.d):,}", flush=True)
        del voc, e1, e2
    return len(ctrs)


def block_country_file(f, out, top_k=40, top_m=12, verbose=True):
    z = np.load(f)
    i1, i2, V = z["i1"], z["i2"], z["V"]
    e1 = tuple(z[f"a{j}"] for j in range(5)); e2 = tuple(z[f"b{j}"] for j in range(5))
    o = _country_block(e1, e2, V, verbose, tmp=out + ".tmp")
    del e1, e2
    gc.collect()
    o["r1"] = i1[o["r1"].values]
    o["r2"] = i2[o["r2"].values]
    ordr = np.lexsort((-o["bscore"].values, o["r2"].values))
    rk2 = np.empty(len(o), np.int32)
    rk2[ordr] = _group_rank(o["r2"].values[ordr])
    o["brank2"] = rk2.astype(np.int16)
    o = o[rk2 < top_m].reset_index(drop=True)
    if verbose:
        print(f"    kept {len(o):,}", flush=True)
    o.to_parquet(out, index=False)


def _child(fn, *a):
    """Run fn(*a) in a fresh forked process (memory is fully released afterwards)."""
    import multiprocessing as mp
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    pr = ctx.Process(target=lambda: q.put(fn(*a)))
    pr.start()
    pr.join()
    if pr.exitcode != 0:
        raise RuntimeError(f"{fn.__name__} failed (exit code {pr.exitcode}; out of memory?)")
    return q.get()


def generate_split(d, verbose=True):
    """Full blocking for a split directory; each heavy stage runs in its own process so memory
    is returned to the OS between stages. Returns candidate DataFrame."""
    enc = f"{d}/enc"
    n = _child(encode_to_disk, d, enc)
    res = []
    for ci in range(n):
        _child(block_country_file, f"{enc}/c{ci}.npz", f"{enc}/cand{ci}.parquet")
        res.append(pd.read_parquet(f"{enc}/cand{ci}.parquet"))
    return pd.concat(res, ignore_index=True)
