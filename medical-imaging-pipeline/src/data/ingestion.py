"""Data ingestion: upload validation and case registration.

Implements:
  FR-1.1 — accept NIfTI upload for T1/T1ce/T2/FLAIR
  FR-1.2 — validate modality completeness + co-registration (affine/shape)
  FR-1.3 — reject with specific error code on missing/mismatched/corrupt input
  FR-1.5 — assign case ID, persist raw upload to versioned object storage
            before any processing begins
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class IngestionError(Enum):
    """FR-1.3 — specific, distinguishable error codes."""

    MISSING_MODALITY = "MISSING_MODALITY"
    DIMENSION_MISMATCH = "DIMENSION_MISMATCH"
    UNREADABLE_HEADER = "UNREADABLE_HEADER"


@dataclass
class CaseUpload:
    case_id: str
    modality_paths: dict[str, Path]  # keys: t1, t1ce, t2, flair


def validate_case(modality_paths: dict[str, Path]) -> list[IngestionError]:
    """FR-1.2 / FR-1.3. Returns empty list if valid.

    TODO: load each volume's affine + shape via nibabel and compare;
    flag IngestionError.DIMENSION_MISMATCH on affine/shape mismatch,
    IngestionError.MISSING_MODALITY if any of the 4 required keys absent,
    IngestionError.UNREADABLE_HEADER on load failure.
    """
    raise NotImplementedError


def register_case(modality_paths: dict[str, Path]) -> CaseUpload:
    """FR-1.5. Assigns a unique case ID and persists to object storage
    BEFORE any preprocessing runs. Must call validate_case() first and
    raise on any IngestionError.
    """
    raise NotImplementedError
