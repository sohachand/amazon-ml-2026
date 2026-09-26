"""Step 1: normalise every source file into a compact parquet table.

usage: python prep.py <data_dir> <split(train|test)> <out_dir>
"""
import sys, os
import pandas as pd
from multiprocessing import Pool
from normalize import norm_name, norm_addr

N_PROC = int(os.environ.get("N_PROC", os.cpu_count() or 2))


def _proc(df):
    out = {k: [] for k in ["name_lat", "name_core", "name_full", "dom", "addr_lat", "addr_tok", "nums"]}
    for n, a, c in zip(df.business_name.values, df.business_address.values, df.country.values):
        lat, core, full, dom = norm_name(n)
        al, at, nm = norm_addr(a, c)
        out["name_lat"].append(lat)
        out["name_core"].append(" ".join(core))
        out["name_full"].append(" ".join(full))
        out["dom"].append(dom)
        out["addr_lat"].append(al)
        out["addr_tok"].append(" ".join(at))
        out["nums"].append(" ".join(nm))
    res = pd.DataFrame(out)
    res.insert(0, "country", df.country.values)
    res.insert(0, "entity_id", df.entity_id.values)
    res["name_raw"] = df.business_name.values
    res["addr_raw"] = df.business_address.values
    return res


def main(data_dir, split, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for s in (1, 2, 3):
        f = os.path.join(data_dir, f"{split}_source{s}.tsv")
        chunks = pd.read_csv(f, sep="\t", dtype=str, keep_default_na=False, chunksize=100_000,
                             quoting=3)
        with Pool(N_PROC) as pool:
            parts = list(pool.imap(_proc, chunks, chunksize=1))
        df = pd.concat(parts, ignore_index=True)
        df["dom"] = df["dom"].astype("int8")
        df.to_parquet(os.path.join(out_dir, f"s{s}.parquet"), index=False)
        print(split, s, len(df), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:4])
