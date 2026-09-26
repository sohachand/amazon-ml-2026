"""Decompress the uploaded *.tsv.zst files into dataset/train and dataset/test."""
import glob, os, zstandard
os.makedirs("dataset/train", exist_ok=True); os.makedirs("dataset/test", exist_ok=True)
for f in sorted(glob.glob("data_upload/*.tsv.zst")):
    name = os.path.basename(f)[:-4]
    out = f"dataset/{'train' if name.startswith('train') else 'test'}/{name}"
    if os.path.exists(out):
        continue
    with open(f, "rb") as i, open(out, "wb") as o:
        zstandard.ZstdDecompressor().copy_stream(i, o)
    print("unpacked", out, os.path.getsize(out))
