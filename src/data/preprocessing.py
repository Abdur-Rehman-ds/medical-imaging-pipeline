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
           dataset artifact

Uses MONAI dictionary transforms (monai.transforms, "...d" suffix) since
each data item is a dict of {"t1": path, "t1ce": path, "t2": path,
"flair": path, "seg": path} before loading, and {"image": tensor,
"label": tensor} after ConcatItemsd — matching build_case_dict() below.
"""

from pathlib import Path

from monai.transforms import (
    Compose,
    ConcatItemsd,
    DeleteItemsd,
    EnsureChannelFirstd,
    LoadImaged,
    MapLabelValued,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandGaussianNoised,
    RandRotated,
    RandScaleIntensityd,
    RandCropByPosNegLabeld,
    Rand3DElasticd,
    ScaleIntensityRangePercentilesd,
    Spacingd,
)

MODALITY_KEYS = ["t1", "t1ce", "t2", "flair"]
LABEL_KEY = "seg"
ALL_KEYS = MODALITY_KEYS + [LABEL_KEY]


def build_case_dict(case_dir: Path, case_id: str, include_label: bool = True) -> dict:
    """Build the {"t1": path, ...} dict LoadImaged expects for one case.

    Handles both .nii and .nii.gz (BraTS2020 Kaggle mirror uses .nii,
    other releases use .nii.gz — see configs/data/brats.yaml note).
    """

    def find(suffix: str) -> str:
        for ext in (".nii", ".nii.gz"):
            candidate = case_dir / f"{case_id}_{suffix}{ext}"
            if candidate.exists():
                return str(candidate)
        raise FileNotFoundError(f"Missing {suffix} for case {case_id} in {case_dir}")

    d = {key: find(key) for key in MODALITY_KEYS}
    if include_label:
        d[LABEL_KEY] = find(LABEL_KEY)
    return d


def _shared_load_and_normalize(keys: list[str]) -> list:
    """FR-2.1, FR-2.2, FR-2.3 — steps common to both train and inference."""
    return [
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        Orientationd(keys=keys, axcodes="RAS"),
        Spacingd(
            keys=keys,
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear",) * len(MODALITY_KEYS) + (("nearest",) if LABEL_KEY in keys else ()),
        ),
        ScaleIntensityRangePercentilesd(
            keys=MODALITY_KEYS, lower=0.5, upper=99.5, b_min=0.0, b_max=1.0, clip=True
        ),
        NormalizeIntensityd(keys=MODALITY_KEYS, channel_wise=True),
        ConcatItemsd(keys=MODALITY_KEYS, name="image"),
        DeleteItemsd(keys=MODALITY_KEYS),
    ]


def build_train_transforms(cfg) -> Compose:
    """FR-2.1..2.5, training branch (patch extraction + augmentation)."""
    patch_size = tuple(cfg.patch_size)
    pos_neg_ratio = cfg.positive_negative_ratio

    steps = _shared_load_and_normalize(ALL_KEYS)
    steps += [
        MapLabelValued(keys=[LABEL_KEY], orig_labels=[0, 1, 2, 4], target_labels=[0, 1, 2, 3]),
        RandCropByPosNegLabeld(
            keys=["image", LABEL_KEY],
            label_key=LABEL_KEY,
            spatial_size=patch_size,
            pos=pos_neg_ratio,
            neg=1.0,
            num_samples=1,
            image_key="image",
        ),
        RandFlipd(keys=["image", LABEL_KEY], spatial_axis=0, prob=0.5),
        RandFlipd(keys=["image", LABEL_KEY], spatial_axis=1, prob=0.5),
        RandFlipd(keys=["image", LABEL_KEY], spatial_axis=2, prob=0.5),
        RandRotated(
            keys=["image", LABEL_KEY],
            range_x=0.26, range_y=0.26, range_z=0.26,
            mode=("bilinear", "nearest"),
            prob=0.3,
        ),
        Rand3DElasticd(
            keys=["image", LABEL_KEY],
            sigma_range=(5, 8),
            magnitude_range=(50, 150),
            mode=("bilinear", "nearest"),
            prob=0.2,
        ),
        RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.3),
        RandGaussianNoised(keys=["image"], std=0.01, prob=0.2),
    ]
    return Compose(steps)


def build_inference_transforms(cfg) -> Compose:
    """FR-2.1..2.3 only — no augmentation, no patch extraction (full-volume
    sliding-window handles patching at inference time, see
    src/inference/sliding_window.py).
    """
    return Compose(_shared_load_and_normalize(MODALITY_KEYS))
