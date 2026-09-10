"""Unit tests for src/inference/sliding_window.py — Section 11.
Covers FR-4.1..4.5 on tiny synthetic volumes; CPU-only, no real data.
"""

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from src.inference.sliding_window import (
    _clean_binary_mask,
    infer_case,
    postprocess,
    run_inference,
    save_results,
)


def _cfg(min_vox=5, fill=True):
    return OmegaConf.create({
        "patch_size": [8, 8, 8], "sw_batch_size": 1, "overlap": 0.25,
        "blend_mode": "gaussian",
        "postprocess": {"min_component_voxels": min_vox, "fill_holes": fill},
    })


def _probs_from_internal(internal: np.ndarray) -> torch.Tensor:
    """One-hot-ish probabilities [1,4,H,W,D] with 0.9 on the given label."""
    h, w, d = internal.shape
    probs = np.full((4, h, w, d), 0.1 / 3, dtype=np.float32)
    for lab in range(4):
        probs[lab][internal == lab] = 0.9
    return torch.from_numpy(probs).unsqueeze(0)


def test_fr_4_3_small_component_removed_large_kept():
    binary = np.zeros((16, 16, 16), dtype=bool)
    binary[2:6, 2:6, 2:6] = True        # 64 voxels — keep
    binary[12, 12, 12] = True           # 1 voxel — drop (min 5)
    cleaned = _clean_binary_mask(binary, min_voxels=5, fill_holes=False)
    assert cleaned[3, 3, 3] and not cleaned[12, 12, 12]
    assert cleaned.sum() == 64


def test_fr_4_3_hole_filling():
    binary = np.zeros((12, 12, 12), dtype=bool)
    binary[2:9, 2:9, 2:9] = True
    binary[5, 5, 5] = False             # enclosed hole
    cleaned = _clean_binary_mask(binary, min_voxels=1, fill_holes=True)
    assert cleaned[5, 5, 5]


def test_fr_4_4_inverse_remap_and_counts():
    internal = np.zeros((16, 16, 16), dtype=np.int64)
    internal[2:6, 2:6, 2:6] = 1         # 64 vox NCR/NET
    internal[8:12, 8:12, 8:12] = 2      # 64 vox edema
    internal[2:5, 10:13, 2:5] = 3       # 27 vox enhancing (internal 3)
    result = postprocess(_probs_from_internal(internal), _cfg(min_vox=5), voxel_volume_mm3=1.0)
    mask = result["mask"]
    assert set(np.unique(mask).tolist()) <= {0, 1, 2, 4}   # canonical labels only
    assert 3 not in np.unique(mask)                        # internal 3 -> canonical 4
    assert result["per_label_voxels"]["NCR_NET"] == 64
    assert result["per_label_voxels"]["edema"] == 64
    assert result["per_label_voxels"]["enhancing_tumor"] == 27
    assert result["per_label_volumes_mm3"]["NCR_NET"] == 64.0


def test_fr_4_4_voxel_volume_scaling():
    internal = np.zeros((8, 8, 8), dtype=np.int64)
    internal[0:2, 0:2, 0:2] = 1         # 8 voxels
    result = postprocess(_probs_from_internal(internal), _cfg(min_vox=1), voxel_volume_mm3=2.5)
    assert result["per_label_volumes_mm3"]["NCR_NET"] == pytest.approx(20.0)


def test_fr_4_5_confidence_summary_present_and_absent_regions():
    internal = np.zeros((8, 8, 8), dtype=np.int64)
    internal[0:3, 0:3, 0:3] = 1         # only NCR/NET present
    result = postprocess(_probs_from_internal(internal), _cfg(min_vox=1))
    assert result["confidence_summary"]["NCR_NET"] == pytest.approx(0.9, abs=1e-3)
    assert result["confidence_summary"]["edema"] == 0.0          # absent -> 0
    assert result["confidence_summary"]["enhancing_tumor"] == 0.0


def test_fr_4_5_save_results_writes_mask_and_summary(tmp_path):
    internal = np.zeros((8, 8, 8), dtype=np.int64)
    internal[0:2, 0:2, 0:2] = 2
    result = postprocess(_probs_from_internal(internal), _cfg(min_vox=1))
    summary = save_results(result, np.eye(4), tmp_path, "case_abc", "vtest", 1.23)
    assert (tmp_path / "case_abc_seg.nii.gz").exists()
    assert (tmp_path / "case_abc_summary.json").exists()
    assert summary["model_version"] == "vtest"
    assert "disclaimer" in summary and "NOT" in summary["disclaimer"]  # FR-6.6/S14


class _FakeModel(torch.nn.Module):
    """Deterministic 4-class logits: strongly favors class 0 everywhere.
    Enough to exercise the sliding-window + softmax path on CPU."""
    def forward(self, x):
        b, _, h, w, d = x.shape
        logits = torch.zeros(b, 4, h, w, d)
        logits[:, 0] = 5.0
        return logits


def test_fr_4_1_4_2_run_inference_shapes_and_probs():
    vol = torch.zeros(4, 12, 12, 12)    # [C,H,W,D] — batch dim added inside
    probs = run_inference(vol, _FakeModel(), _cfg(), device=torch.device("cpu"))
    assert probs.shape == (1, 4, 12, 12, 12)
    s = probs.sum(dim=1)
    assert torch.allclose(s, torch.ones_like(s), atol=1e-4)  # softmax sums to 1


def test_fr_4_1_to_4_5_infer_case_end_to_end(tmp_path):
    vol = torch.zeros(4, 12, 12, 12)
    summary = infer_case(vol, np.eye(4), _FakeModel(), _cfg(), tmp_path,
                         "case_e2e", model_version="vtest", device=torch.device("cpu"))
    assert (tmp_path / "case_e2e_seg.nii.gz").exists()
    assert summary["per_label_voxels"]["edema"] == 0   # model predicts background
    assert summary["processing_time_s"] >= 0
