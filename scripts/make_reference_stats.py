"""FR-7.2 — build the training-set reference distribution for drift scoring.

Runs the SAME preprocessing the API applies (build_inference_transforms),
computes compute_input_drift_stats() per case, then aggregates
mean_across_cases / std_across_cases per (modality, stat). Output JSON is
what src/monitoring/logging.py reads (REFERENCE_STATS, default
configs/monitoring/reference_stats.json).

Intended run environment: Kaggle CPU session against the BraTS2020
dataset (no GPU quota needed):
    PYTHONPATH=. python scripts/make_reference_stats.py \
        --data-root /kaggle/input/... --out reference_stats.json
Then commit the JSON to configs/monitoring/. Until that is done, the
API logs drift_score=null (reason no-reference) by design.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch  # noqa: F401  (import order matters before MONAI on some setups)
from omegaconf import OmegaConf

from src.data.preprocessing import build_case_dict, build_inference_transforms
from src.monitoring.logging import MODALITY_ORDER, compute_input_drift_stats

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True,
                    help="directory of case subdirectories (BraTS layout)")
    ap.add_argument("--out", default="reference_stats.json")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap number of cases (0 = all)")
    args = ap.parse_args()

    inf_cfg = OmegaConf.load(REPO_ROOT / "configs/inference/default.yaml")
    transforms = build_inference_transforms(inf_cfg)

    case_dirs = sorted(p for p in Path(args.data_root).iterdir() if p.is_dir())
    if args.limit:
        case_dirs = case_dirs[: args.limit]
    print(f"computing stats over {len(case_dirs)} cases")

    per_case: list[dict] = []
    for i, cdir in enumerate(case_dirs):
        try:
            data = build_case_dict(cdir, cdir.name, include_label=False)
            image = transforms(data)["image"]
            per_case.append(compute_input_drift_stats(np.asarray(image)))
        except Exception as e:
            print(f"  skip {cdir.name}: {e}")
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(case_dirs)} done")

    if not per_case:
        raise SystemExit("no cases processed — check --data-root")

    ref: dict = {"n_cases": len(per_case)}
    stat_keys = ("mean", "std", "p1", "p50", "p99", "fg_fraction")
    for m in MODALITY_ORDER:
        ref[m] = {}
        for k in stat_keys:
            vals = np.array([c[m][k] for c in per_case if m in c])
            ref[m][k] = {
                "mean_across_cases": round(float(vals.mean()), 4),
                "std_across_cases": round(float(vals.std()), 4),
            }
    Path(args.out).write_text(json.dumps(ref, indent=2))
    print(f"wrote {args.out} ({len(per_case)} cases)")


if __name__ == "__main__":
    main()
