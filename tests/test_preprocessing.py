"""Unit tests for src/data/preprocessing.py — Section 11 (Testing Strategy).
Test names reference the FR-ID they cover (Appendix C traceability).
Uses tiny synthetic NIfTI volumes — no real data, no GPU.
"""
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from omegaconf import OmegaConf

from src.data.preprocessing import (
    ALL_KEYS,
    MODALITY_KEYS,
    build_case_dict,
    build_val_transforms,
)


@pytest.fixture
def synthetic_case(tmp_path):
    """A tiny 12x12x12 case: 4 modalities + a label map using canonical
    BraTS labels {0,1,2,4}, anisotropic 2mm spacing so Spacingd has
    real work to do (FR-2.2)."""
    case_id = "case_test0001"
    case_dir = tmp_path / case_id
    case_dir.mkdir()
    affine = np.diag([2.0, 2.0, 2.0, 1.0])  # 2mm iso -> resample doubles dims
    rng = np.random.default_rng(0)
    for key in MODALITY_KEYS:
        vol = rng.normal(100, 20, size=(12, 12, 12)).astype(np.float32)
        nib.save(nib.Nifti1Image(vol, affine), case_dir / f"{case_id}_{key}.nii")
    seg = np.zeros((12, 12, 12), dtype=np.float32)
    seg[2:5, 2:5, 2:5] = 1      # NCR/NET
    seg[6:9, 6:9, 6:9] = 2      # edema
    seg[2:4, 8:10, 2:4] = 4     # enhancing (canonical label 4)
    nib.save(nib.Nifti1Image(seg, affine), case_dir / f"{case_id}_seg.nii")
    return case_dir, case_id


def _val_cfg():
    return OmegaConf.create({"patch_size": [8, 8, 8], "positive_negative_ratio": 1.0})


def test_fr_1_1_build_case_dict_finds_nii_and_raises_on_missing(synthetic_case, tmp_path):
    case_dir, case_id = synthetic_case
    d = build_case_dict(case_dir, case_id)
    assert set(d.keys()) == set(ALL_KEYS)
    for path in d.values():
        assert Path(path).exists()
    with pytest.raises(FileNotFoundError):
        build_case_dict(tmp_path, "case_missing")


def test_fr_2_2_resample_preserves_label_integrity(synthetic_case):
    """Resampling the label map must use nearest-neighbor: after 2mm->1mm
    resampling (which interpolates), the label set must still be a subset
    of the internal set {0,1,2,3} (post-remap) with no fractional or
    invented values — linear interpolation would produce intermediates."""
    case_dir, case_id = synthetic_case
    tf = build_val_transforms(_val_cfg())
    out = tf(build_case_dict(case_dir, case_id))
    seg = np.asarray(out["seg"])
    # resampled from 2mm to 1mm: spatial dims should have grown
    assert seg.shape[-1] > 12
    values = set(np.unique(seg).tolist())
    assert values <= {0.0, 1.0, 2.0, 3.0}, f"invalid label values: {values}"
    # remap 4->3 happened: canonical 4 must be gone, internal 3 present
    assert 4.0 not in values and 3.0 in values


def test_fr_2_3_zscore_normalization_after_percentile_clip(synthetic_case):
    """After per-modality z-score normalization, each image channel must
    be ~zero-mean/unit-std (FR-2.3). Channels: 4 modalities concatenated."""
    case_dir, case_id = synthetic_case
    tf = build_val_transforms(_val_cfg())
    out = tf(build_case_dict(case_dir, case_id))
    img = np.asarray(out["image"])
    assert img.shape[0] == 4  # 4 modalities concatenated (FR-1.1)
    for c in range(4):
        assert abs(float(img[c].mean())) < 0.1
        assert abs(float(img[c].std()) - 1.0) < 0.15
