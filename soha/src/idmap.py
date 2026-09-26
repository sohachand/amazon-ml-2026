"""Memory-light entity_id <-> row mapping (ids are 'S<k>-<digits>')."""
import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq


def id_nums(path):
    t = pq.read_table(path, columns=["entity_id"]).column(0)
    return np.asarray(pc.cast(pc.utf8_slice_codeunits(t, 3), "int64"))


def lookup(keys_sorted_order, nums, q):
    """rows of q (int64 ids) in nums; -1 if absent."""
    o = keys_sorted_order
    pos = np.searchsorted(nums[o], q)
    pos = np.clip(pos, 0, len(o) - 1)
    r = o[pos]
    return np.where(nums[r] == q, r, -1)


def gt_pairs(gt_path, d):
    n1 = id_nums(f"{d}/s1.parquet"); n2 = id_nums(f"{d}/s2.parquet"); n3 = id_nums(f"{d}/s3.parquet")
    gt = pd.read_csv(gt_path, sep="\t", dtype=str, keep_default_na=False)
    gt = gt[gt.matched_entity_ids != ""]
    e = gt.assign(m=gt.matched_entity_ids.str.split(",")).explode("m")
    a = e.source1_entity_id.str.slice(3).astype(np.int64).values
    src = e.m.str.slice(1, 2).values
    b = e.m.str.slice(3).astype(np.int64).values
    r1 = lookup(np.argsort(n1), n1, a)
    r2 = np.where(src == "2", lookup(np.argsort(n2), n2, b), -1)
    is3 = src == "3"
    r2[is3] = lookup(np.argsort(n3), n3, b[is3]) + len(n2)
    assert (r1 >= 0).all() and (r2 >= 0).all()
    return pd.DataFrame({"r1": r1.astype(np.int32), "r2": r2.astype(np.int32)})
