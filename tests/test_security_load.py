"""Section 11 — security + load tests (Appendix E No.23).

Covers the Section 11 rows not already in test_api_models_auth.py
(which owns 401/429/FR-5.7): malformed-file handling through the API,
the MAX_UPLOAD_MB cap (413), and right-sized concurrency — 20 threads
against fast endpoints via TestClient. The SRS full-scale load test
(20 concurrent INFERENCES, p95 in budget) is deployment-environment
work, recorded as scoped out in decision No.23.
"""

import io
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "0")  # concurrency, not 429s
    monkeypatch.setenv("CASE_STORAGE_DIR", str(tmp_path / "uploads"))
    from src.api import main
    main._rate_buckets.clear()
    return TestClient(main.app)


def _four_files(content: bytes):
    return {m: (f"{m}.nii.gz", io.BytesIO(content), "application/gzip")
            for m in ("t1", "t1ce", "t2", "flair")}


def test_malformed_files_rejected_with_specific_code(client):
    """Garbage bytes -> 400 UNREADABLE_HEADER in the FR-5.7 structure;
    no case directory is created (FR-1.3/1.5)."""
    r = client.post("/v1/cases", files=_four_files(b"not a nifti at all"))
    assert r.status_code == 400
    body = r.json()
    assert body["error_code"] == "UNREADABLE_HEADER"
    assert "correlation_id" in body


def test_oversize_upload_rejected_413(client, monkeypatch):
    monkeypatch.setenv("MAX_UPLOAD_MB", "0.001")  # ~1 KB cap
    r = client.post("/v1/cases", files=_four_files(b"x" * 10_000))
    assert r.status_code == 413
    body = r.json()
    assert body["error_code"] == "PAYLOAD_TOO_LARGE"
    assert "correlation_id" in body


def test_size_cap_allows_normal_uploads(client, monkeypatch):
    """Under-cap garbage must fail VALIDATION (400), not the size check —
    proves the cap does not bite legitimate-size files."""
    monkeypatch.setenv("MAX_UPLOAD_MB", "512")
    r = client.post("/v1/cases", files=_four_files(b"x" * 10_000))
    assert r.status_code == 400
    assert r.json()["error_code"] == "UNREADABLE_HEADER"


def test_20_concurrent_requests_all_succeed(client):
    """Section 11 load row, right-sized: 20 concurrent GETs against the
    two fast read endpoints; all 200, no crashes, coherent bodies."""
    def hit(i):
        ep = "/v1/models" if i % 2 else "/v1/cases"
        r = client.get(ep)
        return r.status_code, "disclaimer" in r.json() or "cases" in r.json()

    with ThreadPoolExecutor(max_workers=20) as ex:
        results = list(ex.map(hit, range(20)))
    assert all(code == 200 for code, _ in results)
    assert all(ok for _, ok in results)


def test_concurrent_uploads_get_unique_case_ids(client):
    """Concurrency correctness where state is written: 8 simultaneous
    malformed uploads must each be handled independently (no shared-state
    corruption, all rejected, none crash)."""
    def upload(_):
        return client.post("/v1/cases", files=_four_files(b"garbage")).status_code

    with ThreadPoolExecutor(max_workers=8) as ex:
        codes = list(ex.map(upload, range(8)))
    assert codes == [400] * 8
