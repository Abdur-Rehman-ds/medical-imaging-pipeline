"""Preprocessing pipeline.

Implements:
  FR-2.1 — reorient to RAS
  FR-2.2 — resample to 1mm^3 isotropic (linear for intensities,
           nearest-neighbor for label maps)
  FR-2.3 — per-modality z-score normalization after clipping to
           0.5-99.5 percentile
  FR-2.4 — foreground-oversampled 3D patch extraction (training only)
  FR-2.5 — stochastic augmentations (training only): flip, rotation,
           elastic deformation, gamma jitter, Gaussian noise
  FR-2.6 — log exact preprocessing config alongside every processed
           dataset artifact (use configs/data/brats.yaml as the source
           of truth, persist a copy next to the cached tensors)

Use MONAI transforms (monai.transforms) for all of the above rather than
hand-rolled implementations — Section 2.4 / 6.3 specify MONAI as the
transform library.
"""

from monai.transforms import Compose


def build_train_transforms(cfg) -> Compose:
    """FR-2.1..2.5, training branch (augmentation included)."""
    raise NotImplementedError


def build_inference_transforms(cfg) -> Compose:
    """FR-2.1..2.3 only — no augmentation, no patch extraction
    (full-volume sliding-window handles patching at inference time,
    see src/inference/sliding_window.py).
    """
    raise NotImplementedError
