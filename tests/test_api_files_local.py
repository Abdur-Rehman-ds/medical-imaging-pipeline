"""Local test for GET /v1/cases/{id}/files/{kind} (FR-6.3 support).
Covers: modality download, mask download after inference, 404 before
inference, 400 on bad kind. Regression guard for the missing
FileResponse import found 2026-09-03.
Run:  PYTHONPATH=. .venv/bin/python tests/test_api_files_local.py
"""
import io
import os
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np


def make_nifti_bytes(shape=(32, 32, 24)) -> bytes:
    img = nib.Nifti1Image(np.random.rand(*shape).astype(np.float32), np.eye(4))
    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as f:
        tmp_path = f.name
    nib.save(img, tmp_path)
    data = Path(tmp_path).read_bytes()
    os.unlink(tmp_path)
    return data


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["CASE_STORAGE_DIR"] = str(Path(tmp) / "storage")
        os.environ.pop("MODEL_CHECKPOINT", None)
        from fastapi.testclient import TestClient
        from src.api.main import app
        client = TestClient(app)

        vol = make_nifti_bytes()
        files = {m: (f"{m}.nii.gz", io.BytesIO(vol), "application/gzip")
                 for m in ("t1", "t1ce", "t2", "flair")}
        r = client.post("/v1/cases", files=files)
        assert r.status_code == 200, r.text
        case_id = r.json()["case_id"]
        print("uploaded:", case_id)

        # Modality download works pre-inference
        r = client.get(f"/v1/cases/{case_id}/files/t1ce")
        assert r.status_code == 200, r.text
        assert len(r.content) > 1000, "modality file suspiciously small"
        print("modality download OK:", len(r.content), "bytes")

        # Mask before inference -> structured 404
        r = client.get(f"/v1/cases/{case_id}/files/mask")
        assert r.status_code == 404
        assert r.json()["error_code"] == "NO_MASK"
        print("mask before inference -> structured 404")

        # Bad kind -> structured 400
        r = client.get(f"/v1/cases/{case_id}/files/bogus")
        assert r.status_code == 400
        assert r.json()["error_code"] == "BAD_KIND"
        print("bad kind -> structured 400")

        # After inference, mask downloads
        print("running inference on CPU, please wait...")
        r = client.post(f"/v1/cases/{case_id}/infer")
        assert r.status_code == 200, r.text
        r = client.get(f"/v1/cases/{case_id}/result")
        assert r.json()["status"] == "completed", r.json()
        r = client.get(f"/v1/cases/{case_id}/files/mask")
        assert r.status_code == 200, r.text
        assert len(r.content) > 1000, "mask file suspiciously small"
        print("mask download OK:", len(r.content), "bytes")

        # Unknown case -> 404
        r = client.get("/v1/cases/case_doesnotexist/files/t1")
        assert r.status_code == 404
        assert r.json()["error_code"] == "CASE_NOT_FOUND"
        print("unknown case -> structured 404")

    print("ALL_FILE_ENDPOINT_TESTS_PASSED")


if __name__ == "__main__":
    main()
