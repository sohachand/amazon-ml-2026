import sys, numpy as np, pandas as pd, gc
from idmap import gt_pairs
from blocking import BIT
d = sys.argv[1]
g = gt_pairs(sys.argv[2], d); g.to_parquet(f"{d}/gt_pairs.parquet", index=False)
gc.collect()
c = pd.read_parquet(f"{d}/cands.parquet")
SH = np.int64(1 << 24)
cc = c.r1.values.astype(np.int64) * SH + c.r2.values
o = np.argsort(cc); cc = cc[o]
gc_ = g.r1.values.astype(np.int64) * SH + g.r2.values
pos = np.clip(np.searchsorted(cc, gc_), 0, len(cc) - 1)
hit = cc[pos] == gc_
row = o[pos]
print("pairs", len(c), "per S1", len(c) / 2206821, "pair recall", hit.mean())
b = np.where(hit, c.bits.values[row], 0)
for kt, v in BIT.items():
    print(kt, "has key", ((b & v) > 0).mean())
br1 = np.where(hit, c.brank1.values[row], 999)
for k in [5, 10, 20, 40]:
    print("recall@brank1<", k, (br1 < k).mean())
# recall by source / whether S2 has address
