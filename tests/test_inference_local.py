"""Local CPU smoke test for src/inference/sliding_window.py (FR-4.1..4.5).

Runs a small synthetic volume through the full inference path with a
tiny untrained UNet. Verifies mechanics, not model quality:
  - output mask uses canonical BraTS labels only ({0,1,2,4}, never 3)
  - volumes/voxel counts/confidence computed for all 3 tumor labels
  - mask NIfTI + summary JSON written and readable
  - postprocess() respects min_component_voxels
Run:  .venv/bin/python tests/test_inference_local.py
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import torch
from monai.networks.nets import UNet
from omegaconf import OmegaConf

from src.inference.sliding_window import infer_case, postprocess

REPO = Path(__file__).resolve().parents[1]


def tiny_model() -> UNet:
    # Small untrained UNet — same interface as the real one (4 in, 4 out)
    return UNet(spatial_dims=3, in_channels=4, out_channels=4,
                channels=(4, 8, 16), strides=(2, 2), num_res_units=1)


def main() -> None:
    cfg = OmegaConf.load(REPO / "configs/inference/default.yaml")
    # Shrink for a fast CPU test — the code path is identical
    cfg.patch_size = [32, 32, 32]

    volume = torch.randn(4, 64, 64, 48)          # synthetic 4-modality volume
    affine = np.eye(4)

    with tempfile.TemporaryDirectory() as tmp:
        summary = infer_case(volume, affine, tiny_model(), cfg,
                             out_dir=Path(tmp), case_id="SYNTH_001",
                             model_version="test-untrained")

        # FR-4.4 — canonical labels only, never internal label 3
        import nibabel as nib
        mask = np.asarray(nib.load(Path(tmp) / summary["mask_file"]).dataobj)
        labels_found = set(np.unique(mask).tolist())
        assert labels_found <= {0, 1, 2, 4}, f"non-canonical labels: {labels_found}"
        assert 3 not in labels_found

        # FR-4.4 — all three tumor labels reported
        for name in ("NCR_NET", "edema", "enhancing_tumor"):
            assert name in summary["per_label_volumes_mm3"]
            assert name in summary["confidence_summary"]

        # FR-4.5 — summary JSON on disk, with disclaimer (Section 14)
        on_disk = json.loads((Path(tmp) / "SYNTH_001_summary.json").read_text())
        assert on_disk["case_id"] == "SYNTH_001"
        assert "disclaimer" in on_disk

        print("mask labels found:", sorted(labels_found))
        print("volumes mm3:", summary["per_label_volumes_mm3"])
        print("processing_time_s:", summary["processing_time_s"])

    # FR-4.3 — component filtering: a 2-voxel blob must be removed at
    # threshold 50, kept at threshold 1
    probs = torch.zeros(1, 4, 16, 16, 16)
    probs[0, 0] = 1.0                      # background everywhere...
    probs[0, 0, 5, 5, 5:7] = 0.0
    probs[0, 3, 5, 5, 5:7] = 1.0           # ...except a 2-voxel label-3 blob
    kept = postprocess(probs, OmegaConf.create(
        {"postprocess": {"min_component_voxels": 1, "fill_holes": True}}))
    removed = postprocess(probs, OmegaConf.create(
        {"postprocess": {"min_component_voxels": 50, "fill_holes": True}}))
    assert kept["per_label_voxels"]["enhancing_tumor"] == 2
    assert removed["per_label_voxels"]["enhancing_tumor"] == 0
    print("component filtering: 2-voxel blob kept at thr=1, removed at thr=50")

    print("ALL_INFERENCE_TESTS_PASSED")


if __name__ == "__main__":
    main()
