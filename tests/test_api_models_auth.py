"""Unit tests for FR-5.4 (model listing) and FR-5.6 (auth + rate limit).
TestClient only — no server, no GPU. Complements test_api_local.py.
"""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch):
    """Fresh app per test with a clean environment and rate buckets."""
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.delenv("RATE_LIMIT_PER_MINUTE", raising=False)
    from src.api import main
    main._rate_buckets.clear()
    return TestClient(main.app), main


def test_fr_5_4_lists_checkpoints_with_parsed_metadata(client, tmp_path, monkeypatch):
    c, _main = client
    (tmp_path / "fold0_best_e109_d0.8439.pt").write_bytes(b"x" * 100)
    (tmp_path / "weird_name.pt").write_bytes(b"x")          # non-conforming
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    r = c.get("/v1/models")
    assert r.status_code == 200
    body = r.json()
    by_version = {m["version"]: m for m in body["models"]}
    good = by_version["fold0_best_e109_d0.8439"]
    assert good["fold"] == 0 and good["epoch"] == 109
    assert good["val_dice_mean"] == pytest.approx(0.8439)
    assert by_version["weird_name"]["fold"] is None          # nulls, not a crash
    assert body["active_version"] == "untrained-dev"         # no MODEL_CHECKPOINT
    assert "disclaimer" in body                              # Section 14


def test_fr_5_4_active_flag_follows_model_checkpoint(client, tmp_path, monkeypatch):
    c, _main = client
    (tmp_path / "fold0_best_e109_d0.8439.pt").write_bytes(b"x")
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    monkeypatch.setenv("MODEL_CHECKPOINT", str(tmp_path / "fold0_best_e109_d0.8439.pt"))
    body = c.get("/v1/models").json()
    assert body["active_version"] == "fold0_best_e109_d0.8439"
    assert body["models"][0]["active"] is True


def test_fr_5_6_open_when_api_key_unset(client):
    c, _ = client
    assert c.get("/v1/models").status_code == 200


def test_fr_5_6_auth_enforced_when_key_set(client, monkeypatch):
    c, _ = client
    monkeypatch.setenv("API_KEY", "sekret")
    r = c.get("/v1/models")
    assert r.status_code == 401
    body = r.json()
    assert body["error_code"] == "UNAUTHORIZED" and "correlation_id" in body  # FR-5.7
    assert c.get("/v1/models", headers={"X-API-Key": "wrong"}).status_code == 401
    assert c.get("/v1/models", headers={"X-API-Key": "sekret"}).status_code == 200


def test_fr_5_6_rate_limit_returns_429(client, monkeypatch):
    c, _main = client
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "3")
    codes = [c.get("/v1/models").status_code for _ in range(5)]
    assert codes == [200, 200, 200, 429, 429]
    assert c.get("/v1/models").json()["error_code"] == "RATE_LIMITED"
