"""FR-1.4 — DICOM-to-NIfTI conversion stub tests.

Success path: build a tiny synthetic DICOM series with pydicom (8
slices, 32x32), convert with dcm2niix, assert exactly one .nii.gz that
nibabel can load with the expected geometry. Failure paths: missing
dir and garbage input raise DicomConversionError.

Requires the dcm2niix binary; skips (not fails) when absent so the
suite stays runnable on machines without it. CI installs it via apt.
"""

import shutil
from pathlib import Path

import pytest

from src.data.dicom_import import DicomConversionError, convert_dicom_series

needs_dcm2niix = pytest.mark.skipif(
    shutil.which("dcm2niix") is None,
    reason="dcm2niix binary not installed (FR-1.4 stub dependency)",
)


def _write_synthetic_series(series_dir: Path, n_slices: int = 8) -> None:
    import numpy as np
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    series_dir.mkdir(parents=True)
    study_uid, series_uid, for_uid = generate_uid(), generate_uid(), generate_uid()
    for i in range(n_slices):
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.4"  # MR
        meta.MediaStorageSOPInstanceUID = generate_uid()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian

        ds = Dataset()
        ds.file_meta = meta
        ds.SOPClassUID = meta.MediaStorageSOPClassUID
        ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
        ds.Modality = "MR"
        ds.PatientName = "SYNTHETIC^FR14"
        ds.PatientID = "FR14"
        ds.StudyInstanceUID = study_uid
        ds.SeriesInstanceUID = series_uid
        ds.FrameOfReferenceUID = for_uid
        ds.SeriesNumber = 1
        ds.InstanceNumber = i + 1
        ds.ImagePositionPatient = [0.0, 0.0, float(i)]
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.PixelSpacing = [1.0, 1.0]
        ds.SliceThickness = 1.0
        ds.Rows = ds.Columns = 32
        ds.BitsAllocated = ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        rng = np.random.default_rng(i)
        ds.PixelData = rng.integers(0, 1000, (32, 32), dtype=np.uint16).tobytes()
        ds.save_as(series_dir / f"slice_{i:03d}.dcm", enforce_file_format=True)


def test_missing_dir_raises(tmp_path):
    with pytest.raises(DicomConversionError):
        convert_dicom_series(tmp_path / "nope", tmp_path / "out")


@needs_dcm2niix
def test_garbage_input_raises(tmp_path):
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "junk.dcm").write_bytes(b"this is not dicom")
    with pytest.raises(DicomConversionError):
        convert_dicom_series(bad, tmp_path / "out")


@needs_dcm2niix
def test_synthetic_series_converts(tmp_path):
    import nibabel as nib

    series = tmp_path / "series"
    _write_synthetic_series(series)
    out = convert_dicom_series(series, tmp_path / "out")
    assert out.name.endswith(".nii.gz")
    img = nib.load(str(out))
    assert sorted(img.shape) == [8, 32, 32]  # orientation may permute axes
