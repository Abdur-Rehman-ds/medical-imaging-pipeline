"""Local test for GET /v1/cases (Section 6.1 History screen; the single
backend change of the frontend redesign, Appendix E decision #18).
Covers: empty storage, listing after upload, newest-first ordering,
status + model_version after inference, disclaimer presence.
Run:  PYTHONPATH=. .venv/bin/python tests/test_api_cases_list_local.py
"""
import io
import os
import tempfile
import time
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


def upload_case(client, vol) -> str:
    files = {m: (f"{m}.nii.gz", io.BytesIO(vol), "application/gzip")
             for m in ("t1", "t1ce", "t2", "flair")}
    r = client.post("/v1/cases", files=files)
    assert r.status_code == 200, r.text
    return r.json()["case_id"]


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["CASE_STORAGE_DIR"] = str(Path(tmp) / "storage")
        os.environ.pop("MODEL_CHECKPOINT", None)
        from fastapi.testclient import TestClient

        from src.api.main import app
        client = TestClient(app)

        # Empty storage -> empty list, disclaimer present
        r = client.get("/v1/cases")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["cases"] == [], body
        assert "NOT a certified medical device" in body["disclaimer"]
        print("empty storage -> empty list OK")

        # One upload -> listed as uploaded, no model version yet
        vol = make_nifti_bytes()
        first = upload_case(client, vol)
        r = client.get("/v1/cases")
        cases = r.json()["cases"]
        assert [c["case_id"] for c in cases] == [first], cases
        assert cases[0]["status"] == "uploaded"
        assert cases[0]["model_version"] is None
        assert cases[0]["updated_at"]
        print("single case listed OK:", first)

        # Second upload -> newest first
        time.sleep(1.1)  # status.json mtime resolution
        second = upload_case(client, vol)
        r = client.get("/v1/cases")
        ids = [c["case_id"] for c in r.json()["cases"]]
        assert ids == [second, first], ids
        print("newest-first ordering OK")

        # Inference on the first -> completed + model_version filled
        print("running inference on CPU, please wait...")
        r = client.post(f"/v1/cases/{first}/infer")
        assert r.status_code == 200, r.text
        r = client.get(f"/v1/cases/{first}/result")
        assert r.json()["status"] == "completed", r.json()
        r = client.get("/v1/cases")
        by_id = {c["case_id"]: c for c in r.json()["cases"]}
        assert by_id[first]["status"] == "completed"
        assert by_id[first]["model_version"] == "untrained-dev"
        assert by_id[second]["status"] == "uploaded"
        # first was updated most recently -> back on top
        assert next(c["case_id"] for c in r.json()["cases"]) == first
        print("post-inference listing OK")

    print("ALL_CASES_LIST_TESTS_PASSED")


if __name__ == "__main__":
    main()
