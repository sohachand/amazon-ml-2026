"""Export per-pair final probabilities for teammates.

p = stage-3 probability where stage 3 scored the pair, else the stage-2 probability
(before the threshold and the one-to-one step).  Writes to <out_dir>:
  soha_france_pairs.tsv.gz            all France S1, pairs with p >= 0.05
  soha_us_india_20k_each_pairs.tsv.gz random 20k US S1 + 20k India S1, pairs with p >= 0.05
  soha_us_india_sampled_s1.tsv.gz     the sampled S1 ids (so S1 with no pair >= 0.05 are known)
and prints the mean number of candidates per S1 scored by stages 2 and 3.
usage: python3 src/export_probs.py work submissions
"""
import os
import sys

import numpy as np
import pandas as pd

import model as M
import sibling as SB

work = sys.argv[1] if len(sys.argv) > 1 else "work"
out = sys.argv[2] if len(sys.argv) > 2 else "submissions"
d = f"{work}/test"
os.makedirs(out, exist_ok=True)

c = pd.read_parquet(f"{d}/cands.parquet", columns=["r1", "r2"])
r1 = c.r1.values; r2 = c.r2.values; del c
p1 = np.load(f"{d}/p1.npy")
assert len(p1) == len(r1), "p1.npy does not match cands.parquet (stale files?)"
s2 = np.load(f"{d}/scores.npy")
s3 = np.load(f"{d}/scores3.npy") if os.path.exists(f"{d}/scores3.npy") else None
st2 = p1 > M.PRUNE
st3 = st2 & (s2 > SB.KEEP3)
p = np.where(st3, s3, s2) if s3 is not None else s2
p = p.astype(np.float32)

s1 = pd.read_parquet(f"{d}/s1.parquet", columns=["entity_id", "country"])
e23 = np.concatenate([pd.read_parquet(f"{d}/s2.parquet", columns=["entity_id"]).entity_id.values,
                      pd.read_parquet(f"{d}/s3.parquet", columns=["entity_id"]).entity_id.values])
country = s1.country.astype(str).str.strip().str.lower().values
n1 = len(s1)

print(f"test S1: {n1:,}   candidate pairs: {len(r1):,}")
print(f"stage-2 scored pairs: {st2.sum():,}  -> mean per S1 {st2.sum() / n1:.2f}")
print(f"stage-3 scored pairs: {st3.sum():,}  -> mean per S1 {st3.sum() / n1:.2f}")
for cn in sorted(set(country)):
    m = country[r1] == cn
    k = (country == cn).sum()
    print(f"  {cn:>8}: S1 {k:,}  all cands/S1 {m.sum() / k:.2f}  stage2/S1 {(st2 & m).sum() / k:.2f}"
          f"  stage3/S1 {(st3 & m).sum() / k:.2f}")


def dump(mask, path):
    k = mask & (p >= 0.05)
    df = pd.DataFrame({"source1_entity_id": s1.entity_id.values[r1[k]],
                       "entity_id": e23[r2[k]], "p": np.round(p[k], 5)})
    df.to_csv(path, sep="\t", index=False, compression="gzip")
    print(f"wrote {path}: {len(df):,} pairs, {os.path.getsize(path) / 1e6:.1f} MB")


is_fr = np.isin(country, ["france", "fr"])
dump(is_fr[r1], f"{out}/soha_france_pairs.tsv.gz")
rng = np.random.RandomState(0)
pick = []
for names in (["us", "usa", "united states"], ["india", "in"]):
    ids = np.flatnonzero(np.isin(country, names))
    pick.append(rng.choice(ids, min(20000, len(ids)), replace=False))
pick = np.concatenate(pick)
sel = np.zeros(n1, bool); sel[pick] = True
dump(sel[r1], f"{out}/soha_us_india_20k_each_pairs.tsv.gz")
pd.DataFrame({"source1_entity_id": s1.entity_id.values[pick], "country": s1.country.values[pick]}) \
    .to_csv(f"{out}/soha_us_india_sampled_s1.tsv.gz", sep="\t", index=False, compression="gzip")
print("done")
