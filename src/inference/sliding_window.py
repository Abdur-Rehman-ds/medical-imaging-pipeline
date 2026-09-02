"""Full-volume inference — production path.

Implements:
  FR-4.1 — sliding-window inference, configurable patch size / overlap
  FR-4.2 — Gaussian-weighted blending of overlapping patches
  FR-4.3 — post-process: remove small connected components, fill holes
  FR-4.4 — re-map to 4 canonical BraTS labels, compute per-label
           voxel/volume counts (mm^3)
  FR-4.5 — return mask (NIfTI) + summary JSON + optional overlay preview

Uses monai.inferers.SlidingWindowInferer for FR-4.1/4.2 rather than a
hand-rolled tiler — it already implements Gaussian blending correctly.

NFR Performance (Section 5): full-volume inference must complete in
<=30s on RTX3090/4080-class GPU, <=4min CPU fallback, for a standard
240x240x155 volume. patch_size/overlap in configs/inference/default.yaml
directly trade accuracy for latency here.

LABEL SPACES (see src/data/preprocessing.py): the model operates in the
internal contiguous space {0,1,2,3} (BraTS 4 remapped to 3 for
training). This module maps predictions BACK to canonical BraTS labels
{0: background, 1: NCR/NET, 2: edema, 4: enhancing} for all outputs
(FR-4.4) — the inverse of the training-time remap.
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
from monai.inferers import SlidingWindowInferer

# Internal {0,1,2,3} -> canonical BraTS {0,1,2,4} (FR-4.4)
INTERNAL_TO_BRATS = {0: 0, 1: 1, 2: 2, 3: 4}
BRATS_LABEL_NAMES = {1: "NCR_NET", 2: "edema", 4: "enhancing_tumor"}


def run_inference(volume: torch.Tensor, model, cfg, device=None) -> torch.Tensor:
    """FR-4.1, FR-4.2. Returns softmax probabilities over the full volume.

    volume: [C,H,W,D] or [1,C,H,W,D] preprocessed tensor (output of
    build_inference_transforms). Returns probabilities [1,4,H,W,D] on CPU.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if volume.ndim == 4:
        volume = volume.unsqueeze(0)  # add batch dim

    inferer = SlidingWindowInferer(
        roi_size=tuple(cfg.patch_size),
        sw_batch_size=getattr(cfg, "sw_batch_size", 1),
        overlap=cfg.overlap,
        mode=cfg.blend_mode,  # "gaussian" per FR-4.2
    )
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        logits = inferer(volume.to(device), model)
        probs = torch.softmax(logits, dim=1)
    return probs.cpu()


def _clean_binary_mask(binary: np.ndarray, min_voxels: int, fill_holes: bool) -> np.ndarray:
    """FR-4.3 — drop connected components smaller than min_voxels, then
    (optionally) fill enclosed holes. Operates on one binary label mask.
    """
    from scipy import ndimage

    labeled, num = ndimage.label(binary)
    if num == 0:
        return binary
    counts = np.bincount(labeled.ravel())
    counts[0] = 0  # background component
    keep = np.flatnonzero(counts >= min_voxels)
    cleaned = np.isin(labeled, keep)
    if fill_holes:
        cleaned = ndimage.binary_fill_holes(cleaned)
    return cleaned


def postprocess(raw_prediction: torch.Tensor, cfg, voxel_volume_mm3: float = 1.0) -> dict:
    """FR-4.3, FR-4.4. raw_prediction: probabilities [1,4,H,W,D].

    Returns dict with keys:
      mask                  — np.ndarray [H,W,D], canonical BraTS labels {0,1,2,4}
      per_label_volumes_mm3 — {label_name: mm^3} for the 3 tumor labels
      per_label_voxels      — {label_name: voxel count}
      confidence_summary    — mean winning-class probability inside each
                              predicted label region (0 if label absent)

    voxel_volume_mm3: product of voxel spacing; 1.0 after the standard
    1mm isotropic resampling (FR-2.2).
    """
    probs = raw_prediction[0].numpy()          # [4,H,W,D]
    internal = probs.argmax(axis=0)            # {0,1,2,3}
    winning_prob = probs.max(axis=0)

    min_vox = cfg.postprocess.min_component_voxels
    fill = cfg.postprocess.fill_holes

    # FR-4.3 per tumor label (internal 1,2,3); background untouched
    cleaned = np.zeros_like(internal)
    for lab in (1, 2, 3):
        binary = internal == lab
        if binary.any():
            binary = _clean_binary_mask(binary, min_vox, fill)
        cleaned[binary] = lab

    # FR-4.4 — inverse remap to canonical BraTS labels
    mask = np.zeros_like(cleaned)
    for internal_lab, brats_lab in INTERNAL_TO_BRATS.items():
        mask[cleaned == internal_lab] = brats_lab

    volumes_mm3, voxels, confidence = {}, {}, {}
    for brats_lab, name in BRATS_LABEL_NAMES.items():
        region = mask == brats_lab
        count = int(region.sum())
        voxels[name] = count
        volumes_mm3[name] = round(count * voxel_volume_mm3, 1)
        confidence[name] = round(float(winning_prob[region].mean()), 4) if count else 0.0

    return {
        "mask": mask.astype(np.uint8),
        "per_label_volumes_mm3": volumes_mm3,
        "per_label_voxels": voxels,
        "confidence_summary": confidence,
    }


def save_results(result: dict, affine: np.ndarray, out_dir: Path, case_id: str,
                 model_version: str, processing_time_s: float) -> dict:
    """FR-4.5 — write mask as NIfTI + summary JSON. Returns summary dict."""
    import nibabel as nib

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mask_path = out_dir / f"{case_id}_seg.nii.gz"
    nib.save(nib.Nifti1Image(result["mask"], affine), str(mask_path))

    summary = {
        "case_id": case_id,
        "model_version": model_version,
        "processing_time_s": round(processing_time_s, 2),
        "per_label_volumes_mm3": result["per_label_volumes_mm3"],
        "per_label_voxels": result["per_label_voxels"],
        "confidence_summary": result["confidence_summary"],
        "mask_file": mask_path.name,
        # Section 14 — mandatory non-clinical-use disclaimer (FR-6.6 analog
        # for API-side outputs; also required in API responses per 14.1)
        "disclaimer": ("Research/educational output only. NOT a medical "
                       "device; NOT for clinical diagnosis or treatment."),
    }
    summary_path = out_dir / f"{case_id}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    return summary


def infer_case(volume: torch.Tensor, affine: np.ndarray, model, cfg,
               out_dir: Path, case_id: str, model_version: str = "unknown",
               voxel_volume_mm3: float = 1.0, device=None) -> dict:
    """End-to-end: FR-4.1..4.5 for one preprocessed volume.

    volume/affine come from build_inference_transforms (image tensor and
    its NIfTI affine). Returns the summary dict; writes mask + JSON to
    out_dir.
    """
    start = time.time()
    probs = run_inference(volume, model, cfg, device=device)
    result = postprocess(probs, cfg, voxel_volume_mm3=voxel_volume_mm3)
    elapsed = time.time() - start
    return save_results(result, affine, out_dir, case_id, model_version, elapsed)
