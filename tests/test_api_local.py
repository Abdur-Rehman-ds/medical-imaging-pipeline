"""Local test for src/api/main.py (FR-5.1, FR-5.2, FR-5.3, FR-5.7, FR-6.6).

Full loop against an in-memory API: upload -> infer -> result.
Run:  PYTHONPATH=. .venv/bin/python tests/test_api_local.py
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
        os.environ.pop("MODEL_CHECKPOINT", None)  # force untrained-dev path

        from fastapi.testclient import TestClient
        from src.api.main import app

        client = TestClient(app)

        # FR-5.1 — upload a valid case
        vol = make_nifti_bytes()
        files = {m: (f"{m}.nii.gz", io.BytesIO(vol), "application/gzip")
                 for m in ("t1", "t1ce", "t2", "flair")}
        r = client.post("/v1/cases", files=files)
        assert r.status_code == 200, r.text
        case_id = r.json()["case_id"]
        assert case_id.startswith("case_")
        print("uploaded:", case_id)

        # FR-5.7 — invalid upload rejected with specific code
        bad = {m: (f"{m}.nii.gz", io.BytesIO(vol), "application/gzip")
               for m in ("t1", "t1ce", "t2")}
        r = client.post("/v1/cases", files=bad)
        assert r.status_code == 422  # FastAPI: required flair file missing
        print("incomplete upload rejected")

        # FR-5.7 — unknown case
        r = client.get("/v1/cases/case_doesnotexist/result")
        assert r.status_code == 404
        assert r.json()["error_code"] == "CASE_NOT_FOUND"
        assert "correlation_id" in r.json()
        print("unknown case -> structured 404")

        # FR-5.2 — trigger inference (untrained model, CPU — takes ~30s)
        print("running inference on CPU, please wait...")
        r = client.post(f"/v1/cases/{case_id}/infer")
        assert r.status_code == 200, r.text

        # FR-5.3 — result with disclaimer (FR-6.6)
        r = client.get(f"/v1/cases/{case_id}/result")
        body = r.json()
        assert body["status"] in ("completed", "failed"), body
        assert "NOT a certified medical device" in body["disclaimer"]
        if body["status"] == "failed":
            raise AssertionError(f"inference failed: {body}")
        summary = body["summary"]
        assert summary["model_version"] == "untrained-dev"
        assert "per_label_volumes_mm3" in summary
        assert "disclaimer" in summary
        print("result:", {k: summary[k] for k in ("model_version", "per_label_volumes_mm3")})

        # FR-5.4 — stub responds 501, structured
        r = client.get("/v1/models")
        assert r.status_code == 501
        assert r.json()["error_code"] == "NOT_IMPLEMENTED"
        print("models stub -> structured 501")

    print("ALL_API_TESTS_PASSED")


if __name__ == "__main__":
    main()
