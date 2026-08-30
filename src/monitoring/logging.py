"""Operational logging and drift monitoring.

Implements:
  FR-7.1 — log every request's latency, input shape/spacing, model version
  FR-7.2 — compute/expose input-intensity stats per request (drift signal)
  FR-7.3 — alert when p95 latency or drift score exceeds threshold
  FR-7.4 — immutable audit log of model promotions (version, when, by whom)
"""


def log_inference_event(case_id: str, latency_s: float, input_shape,
                         input_spacing, model_version: str) -> None:
    """FR-7.1. TODO: emit structured JSON log + Prometheus metrics."""
    raise NotImplementedError


def compute_input_drift_stats(volume) -> dict:
    """FR-7.2. TODO: intensity mean/std/percentiles vs. training distribution."""
    raise NotImplementedError


def record_model_promotion(model_version: str, promoted_by: str) -> None:
    """FR-7.4. Append-only — never overwrite prior entries."""
    raise NotImplementedError
