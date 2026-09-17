"""FR-7.1/7.2/7.3 — monitoring module tests (Appendix E No.22).

Pure-function tests: stats on synthetic volumes, drift score against a
temp reference file, threshold alert flag, and the JSON log line shape
captured from stdout. No model, storage, or network involved; webhook
path is NOT tested (fire-and-forget by design).
"""

import json

import numpy as np

import src.monitoring.logging as mon
from src.monitoring.logging import (
    compute_input_drift_stats,
    drift_score,
    log_inference_event,
)


def _volume(mean=0.0, std=1.0, shape=(4, 16, 16, 8), seed=0):
    rng = np.random.default_rng(seed)
    v = rng.normal(mean, std, shape)
    v[:, :4] = 0.0  # a zero "background" region so fg_fraction < 1
    return v


def _reference(tmp_path, monkeypatch, center=0.0):
    """Write a reference where every stat has mean_across_cases=center,
    std_across_cases=0.5, and point the module at it."""
    ref = {"n_cases": 10}
    for m in mon.MODALITY_ORDER:
        ref[m] = {k: {"mean_across_cases": center, "std_across_cases": 0.5}
                  for k in ("mean", "std", "p1", "p50", "p99", "fg_fraction")}
    p = tmp_path / "ref.json"
    p.write_text(json.dumps(ref))
    monkeypatch.setattr(mon, "REFERENCE_STATS_PATH", p)
    return p


def test_stats_shape_and_values():
    stats = compute_input_drift_stats(_volume())
    assert set(stats) == set(mon.MODALITY_ORDER)
    t1 = stats["t1"]
    assert set(t1) == {"mean", "std", "p1", "p50", "p99", "fg_fraction"}
    assert abs(t1["mean"]) < 0.1          # ~N(0,1) foreground
    assert abs(t1["std"] - 1.0) < 0.1
    assert 0.0 < t1["fg_fraction"] < 1.0


def test_all_zero_channel_is_safe():
    stats = compute_input_drift_stats(np.zeros((4, 8, 8, 4)))
    assert stats["flair"]["fg_fraction"] == 0.0  # no NaNs, no crash


def test_score_none_without_reference(tmp_path, monkeypatch):
    monkeypatch.setattr(mon, "REFERENCE_STATS_PATH", tmp_path / "absent.json")
    score, reason = drift_score(compute_input_drift_stats(_volume()))
    assert score is None and reason == "no-reference"


def test_score_low_for_reference_like_case(tmp_path, monkeypatch):
    _reference(tmp_path, monkeypatch, center=0.0)
    stats = compute_input_drift_stats(_volume(mean=0.0))
    score, reason = drift_score(stats)
    assert reason is None and score is not None and score < 3.0


def test_score_high_for_shifted_case(tmp_path, monkeypatch):
    _reference(tmp_path, monkeypatch, center=0.0)
    stats = compute_input_drift_stats(_volume(mean=8.0))  # blatant shift
    score, _ = drift_score(stats)
    assert score is not None and score > 3.0


def test_log_event_json_and_alert_flag(tmp_path, monkeypatch, capsys):
    _reference(tmp_path, monkeypatch)
    monkeypatch.setenv("DRIFT_THRESHOLD", "3.0")
    monkeypatch.delenv("ALERT_WEBHOOK_URL", raising=False)
    stats = compute_input_drift_stats(_volume(mean=8.0))
    score, reason = drift_score(stats)
    log_inference_event("case_test123", 12.3, (4, 16, 16, 8),
                        "fold2_best", stats, score, reason)
    event = json.loads(capsys.readouterr().out.strip())
    assert event["event"] == "inference"
    assert event["case_id"] == "case_test123"
    assert event["model_version"] == "fold2_best"
    assert event["drift_alert"] is True

