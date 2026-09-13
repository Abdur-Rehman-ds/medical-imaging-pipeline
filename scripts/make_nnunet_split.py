"""Generate nnU-Net splits_final.json from the project fold manifest.

FR-3.6 (nnU-Net benchmark) + decision No.13: the benchmark must
validate on the same held-out patients as baseline fold 0, so we
override nnU-Net's own 5-fold split with a single-fold split derived
from configs/data/split_manifest.csv:
  val   = fold 0 (valid cases)
  train = folds 1-4 (valid cases)
Recorded as part of Appendix E decision No.17.
"""
import csv, json
from pathlib import Path

MANIFEST = Path("configs/data/split_manifest.csv")
OUT = Path("configs/data/nnunet_splits_final.json")

train, val = [], []
with MANIFEST.open() as f:
    for row in csv.DictReader(f):
        if row["status"] != "valid":
            continue
        (val if row["fold"] == "0" else train).append(row["case_id"])

train.sort()
val.sort()
assert len(val) == 74, f"expected 74 val cases, got {len(val)}"
assert len(train) == 294, f"expected 294 train cases, got {len(train)}"
assert not set(train) & set(val), "train/val overlap!"
assert "BraTS20_Training_355" not in train + val, "invalid case leaked in"

OUT.write_text(json.dumps([{"train": train, "val": val}], indent=1))
print(f"wrote {OUT}: 1 split, train={len(train)}, val={len(val)}")
print("val head:", val[:3], "... tail:", val[-3:])
