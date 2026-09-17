"""FR-1.4 — DICOM-to-NIfTI conversion stub (dcm2niix).

The pipeline's native input is NIfTI (FR-1.1). This module is the
documented conversion stub the SRS asks for: a caller with a DICOM
series converts it here, then feeds the resulting NIfTI through the
normal ingestion path (validate_case / register_case), which applies
the same FR-1.2/1.3 checks as any native upload. Wiring:

    nifti = convert_dicom_series(Path("series_dir/"), Path("out/"))
    register_case({"t1": nifti, ...})  # one conversion per modality

DESIGN (decision recorded 2026-09-20, Appendix E No.21): library-level
stub only — no API endpoint. FR-1.4 is an S-priority "conversion stub";
the REST surface stays NIfTI-only. Conversion requires the dcm2niix
binary (SRS names it explicitly) on PATH — installed in Dockerfile.api
and CI via apt; a missing binary raises DicomConversionError with a
clear message rather than failing obscurely.
"""

import shutil
import subprocess
from pathlib import Path


class DicomConversionError(Exception):
    """Maps to IngestionError.DICOM_CONVERSION_FAILED at the API layer."""


def convert_dicom_series(series_dir: Path, out_dir: Path) -> Path:
    """Convert one DICOM series directory to one compressed NIfTI.

    Returns the path of the produced .nii.gz. Raises
    DicomConversionError if dcm2niix is missing, exits non-zero,
    produces no NIfTI (e.g. empty/garbage input), or produces more
    than one (ambiguous multi-series input — caller must split).
    """
    if shutil.which("dcm2niix") is None:
        raise DicomConversionError(
            "dcm2niix binary not found on PATH (FR-1.4 requires it; "
            "see Dockerfile.api / apt install dcm2niix)"
        )
    if not series_dir.is_dir():
        raise DicomConversionError(f"not a directory: {series_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    before = set(out_dir.glob("*.nii.gz"))

    # -z y: gzip output; -f: deterministic filename pattern; -w 1:
    # overwrite rather than rename on name collision.
    proc = subprocess.run(
        ["dcm2niix", "-z", "y", "-w", "1",
         "-f", "%p_%s", "-o", str(out_dir), str(series_dir)],
        capture_output=True, text=True,
        check=False,  # non-zero exit handled below: dcm2niix can warn (exit!=0) yet still produce valid output
    )
    produced = sorted(set(out_dir.glob("*.nii.gz")) - before)

    if proc.returncode != 0 and not produced:
        raise DicomConversionError(
            f"dcm2niix failed (exit {proc.returncode}): "
            f"{proc.stderr.strip() or proc.stdout.strip()}"
        )
    if len(produced) == 0:
        raise DicomConversionError(
            f"dcm2niix produced no NIfTI output from {series_dir}"
        )
    if len(produced) > 1:
        raise DicomConversionError(
            f"dcm2niix produced {len(produced)} NIfTI files from "
            f"{series_dir}; expected exactly one series per call"
        )
    return produced[0]
