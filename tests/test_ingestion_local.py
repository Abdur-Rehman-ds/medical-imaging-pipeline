"""Local test for src/data/ingestion.py (FR-1.1, FR-1.2, FR-1.3, FR-1.5).

Synthetic NIfTI files; no real data, no GPU.
Run:  PYTHONPATH=. .venv/bin/python tests/test_ingestion_local.py
"""

import os
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np

from src.data.ingestion import (
    IngestionError,
    IngestionFailure,
    register_case,
    validate_case,
)


def write_nifti(path: Path, shape=(24, 24, 16), affine=None) -> Path:
    affine = np.eye(4) if affine is None else affine
    nib.save(nib.Nifti1Image(np.random.rand(*shape).astype(np.float32), affine), str(path))
    return path


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        os.environ["CASE_STORAGE_DIR"] = str(tmp / "storage")

        # Valid case: 4 matching modalities
        good = {m: write_nifti(tmp / f"{m}.nii.gz") for m in ("t1", "t1ce", "t2", "flair")}
        assert validate_case(good) == []
        up = register_case(good)
        assert up.case_id.startswith("case_")
        for m, p in up.modality_paths.items():
            assert p.exists(), f"{m} not persisted"
        print("valid case registered:", up.case_id)

        # FR-1.3 — missing modality
        missing = {m: good[m] for m in ("t1", "t1ce", "t2")}
        assert validate_case(missing) == [IngestionError.MISSING_MODALITY]
        print("missing modality detected")

        # FR-1.2/1.3 — shape mismatch
        bad_shape = dict(good)
        bad_shape["flair"] = write_nifti(tmp / "flair_bad.nii.gz", shape=(20, 20, 12))
        assert validate_case(bad_shape) == [IngestionError.DIMENSION_MISMATCH]
        print("dimension mismatch detected")

        # FR-1.3 — corrupt file
        corrupt = dict(good)
        corrupt_path = tmp / "corrupt.nii.gz"
        corrupt_path.write_bytes(b"this is not a nifti file")
        corrupt["t2"] = corrupt_path
        assert IngestionError.UNREADABLE_HEADER in validate_case(corrupt)
        print("unreadable header detected")

        # register_case must raise, with codes attached, on invalid input
        try:
            register_case(missing)
            raise AssertionError("register_case accepted an invalid case")
        except IngestionFailure as e:
            assert IngestionError.MISSING_MODALITY in e.errors
        print("register_case rejects invalid input with specific codes")

    print("ALL_INGESTION_TESTS_PASSED")


if __name__ == "__main__":
    main()
