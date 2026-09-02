"""Data ingestion: upload validation and case registration.

Implements:
  FR-1.1 — accept NIfTI upload for T1/T1ce/T2/FLAIR
  FR-1.2 — validate modality completeness + co-registration (affine/shape)
  FR-1.3 — reject with specific error code on missing/mismatched/corrupt input
  FR-1.5 — assign case ID, persist raw upload before any processing begins

STORAGE (decision recorded 2026-09-02): raw uploads are persisted to a
local directory (default data/uploads/, override via CASE_STORAGE_DIR
env var) rather than S3/MinIO (SRS Section 6.3). Same external behavior;
storage backend is swappable later without changing callers. Mirrors
the "simplest option first" checkpoint-persistence decision.

CASE IDS (decision recorded 2026-09-02): "case_" + 12-char UUID hex —
SRS requires uniqueness (FR-1.5) but no specific format.
"""

import os
import shutil
import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np

REQUIRED_MODALITIES = ("t1", "t1ce", "t2", "flair")


class IngestionError(Enum):
    """FR-1.3 — specific, distinguishable error codes."""
    MISSING_MODALITY = "MISSING_MODALITY"
    DIMENSION_MISMATCH = "DIMENSION_MISMATCH"
    UNREADABLE_HEADER = "UNREADABLE_HEADER"


@dataclass
class CaseUpload:
    case_id: str
    modality_paths: dict[str, Path]  # keys: t1, t1ce, t2, flair


class IngestionFailure(Exception):
    """Raised by register_case when validation fails. Carries the
    specific error codes for FR-5.7 structured API responses."""

    def __init__(self, errors: list[IngestionError]):
        self.errors = errors
        super().__init__(", ".join(e.value for e in errors))


def validate_case(modality_paths: dict[str, Path]) -> list[IngestionError]:
    """FR-1.2 / FR-1.3. Returns empty list if valid.

    Checks, in order: all 4 modalities present -> every file loads as
    NIfTI -> all shapes and affines match (co-registration proxy).
    """
    import nibabel as nib

    errors: list[IngestionError] = []

    missing = [m for m in REQUIRED_MODALITIES if m not in modality_paths]
    if missing:
        return [IngestionError.MISSING_MODALITY]

    shapes, affines = [], []
    for m in REQUIRED_MODALITIES:
        try:
            img = nib.load(str(modality_paths[m]))
            shapes.append(img.shape)
            affines.append(img.affine)
        except Exception:
            errors.append(IngestionError.UNREADABLE_HEADER)
    if errors:
        return errors

    if len(set(shapes)) > 1 or any(
        not np.allclose(affines[0], a, atol=1e-3) for a in affines[1:]
    ):
        return [IngestionError.DIMENSION_MISMATCH]

    return []


def get_storage_dir() -> Path:
    return Path(os.environ.get("CASE_STORAGE_DIR", "data/uploads"))


def register_case(modality_paths: dict[str, Path]) -> CaseUpload:
    """FR-1.5. Validates (raising IngestionFailure with specific codes on
    any error), assigns a unique case ID, and copies the raw files into
    storage BEFORE any preprocessing runs.
    """
    errors = validate_case(modality_paths)
    if errors:
        raise IngestionFailure(errors)

    case_id = f"case_{uuid.uuid4().hex[:12]}"
    case_dir = get_storage_dir() / case_id
    case_dir.mkdir(parents=True, exist_ok=False)

    stored: dict[str, Path] = {}
    for m in REQUIRED_MODALITIES:
        src = Path(modality_paths[m])
        suffix = "".join(src.suffixes)  # .nii or .nii.gz
        dest = case_dir / f"{case_id}_{m}{suffix}"
        shutil.copy2(src, dest)
        stored[m] = dest

    return CaseUpload(case_id=case_id, modality_paths=stored)
