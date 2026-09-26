"""Memory-light writer for the two submission TSVs (streams in S1 chunks)."""
import numpy as np
import pyarrow.parquet as pq
from idmap import id_nums


def write_lists(path, col, d, r1, r2, chunk=200_000):
    s1 = pq.read_table(f"{d}/s1.parquet", columns=["entity_id"]).column(0).to_pylist()
    n2 = id_nums(f"{d}/s2.parquet"); n3 = id_nums(f"{d}/s3.parquet")
    ns2 = len(n2)
    num = np.concatenate([n2, n3])
    o = np.lexsort((r2, r1))
    r1, r2 = r1[o], r2[o]
    bounds = np.searchsorted(r1, np.arange(len(s1) + 1))
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"source1_entity_id\t{col}\n")
        for a in range(0, len(s1), chunk):
            b = min(a + chunk, len(s1))
            lo, hi = bounds[a], bounds[b]
            ids = [("S2-" if x < ns2 else "S3-") + str(num[x]) for x in r2[lo:hi].tolist()]
            lines = []
            for i in range(a, b):
                s, e = bounds[i] - lo, bounds[i + 1] - lo
                lines.append(s1[i] + "\t" + ",".join(ids[s:e]))
            f.write("\n".join(lines) + "\n")
