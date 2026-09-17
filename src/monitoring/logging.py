"""Operational logging and drift monitoring.

Implements:
  FR-7.1 — log every request's latency, input shape, model version
  FR-7.2 — per-request input-intensity stats + drift score vs training
  FR-7.3 — threshold alert (flag in log/summary; optional webhook)

DESIGN (decision recorded 2026-09-20, Appendix E No.22): stats are
computed on the PREPROCESSED volume (post z-score normalization), so
the training reference distribution is ~N(0,1) per modality and drift
is deviation from it; raw intensities differ per scanner and would
false-alarm. Logs are one structured JSON line per event to stdout —
captured by docker logs (Section 10.4) with no Prometheus stack, per
the decision-15 compose scope. Reference stats ship as a JSON built by
scripts/make_reference_stats.py on the training set (Kaggle CPU); with
no reference present, stats are still logged and drift_score is null
(reason no-reference) — never a fabricated number.

FR-7.4 (promotion audit log) is a separate Should, deliberately not
implemented here.
"""

import json
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np

REFERENCE_STATS_PATH = Path(
    os.environ.get("REFERENCE_STATS", "configs/monitoring/reference_stats.json")
)
MODALITY_ORDER = ("t1", "t1ce", "t2", "flair")  # channel order, matches preprocessing


def compute_input_drift_stats(volume: np.ndarray) -> dict:
    """FR-7.2. Per-modality stats of a preprocessed (C,H,W,D) volume.

    Foreground = voxels differing from the most-common value in each
    channel (the background plateau). Mask-less normalization (the API
    inference path) leaves background at a constant nonzero z-value, so
    a plain nonzero test wrongly counted it all as foreground — observed
    live 20-Sep-2026 (fg_fraction 1.0, p1==p50). The mode-based
    definition handles both masked (mode==0) and mask-less paths.
    """
    stats: dict = {}
    for i, m in enumerate(MODALITY_ORDER[: volume.shape[0]]):
        chan = np.asarray(volume[i], dtype=np.float64)
        vals, counts = np.unique(np.round(chan, 4), return_counts=True)
        background = vals[np.argmax(counts)]
        fg = chan[np.round(chan, 4) != background]
        if fg.size == 0:
            stats[m] = {"mean": 0.0, "std": 0.0, "p1": 0.0, "p50": 0.0,
                        "p99": 0.0, "fg_fraction": 0.0}
            continue
        p1, p50, p99 = np.percentile(fg, [1, 50, 99])
        stats[m] = {
            "mean": round(float(fg.mean()), 4),
            "std": round(float(fg.std()), 4),
            "p1": round(float(p1), 4),
            "p50": round(float(p50), 4),
            "p99": round(float(p99), 4),
            "fg_fraction": round(float(fg.size / chan.size), 4),
        }
    return stats


def drift_score(stats: dict) -> tuple[float | None, str | None]:
    """Mean |case - ref_mean| / ref_std over all (modality, stat) pairs.

    Returns (score, None) or (None, reason) when no reference exists.
    ~0-1 for typical training-like cases; DRIFT_THRESHOLD (default 3.0)
    flags outliers (FR-7.3).
    """
    if not REFERENCE_STATS_PATH.exists():
        return None, "no-reference"
    ref = json.loads(REFERENCE_STATS_PATH.read_text())
    distances = []
    for m, case_vals in stats.items():
        if m not in ref:
            continue
        for k, v in case_vals.items():
            r = ref[m].get(k)
            if r is None:
                continue
            spread = max(abs(r.get("std_across_cases", 0.0)), 1e-6)
            distances.append(abs(v - r["mean_across_cases"]) / spread)
    if not distances:
        return None, "reference-empty"
    return round(float(np.mean(distances)), 4), None


def log_inference_event(case_id: str, latency_s: float, input_shape,
                        model_version: str, stats: dict,
                        score: float | None, score_reason: str | None) -> None:
    """FR-7.1 + FR-7.3. One JSON line to stdout; optional webhook on alert."""
    threshold = float(os.environ.get("DRIFT_THRESHOLD", "3.0"))
    alert = score is not None and score > threshold
    event = {
        "event": "inference", "case_id": case_id,
        "latency_s": round(latency_s, 2),
        "input_shape": list(input_shape), "model_version": model_version,
        "input_stats": stats, "drift_score": score,
        "drift_score_reason": score_reason,
        "drift_threshold": threshold, "drift_alert": alert,
    }
    print(json.dumps(event), file=sys.stdout, flush=True)

    webhook = os.environ.get("ALERT_WEBHOOK_URL")
    if alert and webhook:
        try:  # fire-and-forget: alerting must never break inference
            req = urllib.request.Request(
                webhook, data=json.dumps(event).encode(),
                headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=5)
        except Exception as e:
            print(json.dumps({"event": "webhook-failed", "error": str(e)}),
                  file=sys.stdout, flush=True)
