"""FastAPI inference service.

Implements:
  FR-5.1 — POST /v1/cases                     upload a case
  FR-5.2 — POST /v1/cases/{id}/infer           trigger inference
  FR-5.3 — GET  /v1/cases/{id}/result          retrieve status/result
  FR-5.4 — GET  /v1/models                     list model versions + metrics
  FR-5.5 — path-versioned API (/v1/), backward compatible within major version
  FR-5.6 — API key / bearer token auth + per-client rate limiting
  FR-5.7 — structured error responses (error code, message, correlation ID)

FR-6.6 requires the non-clinical-use disclaimer on every results view —
mirror that into every /v1/cases/{id}/result response body here too, not
just the frontend, since API consumers may bypass the UI entirely.
"""

from fastapi import FastAPI

app = FastAPI(title="Medical Imaging Diagnostic Pipeline API", version="1.0")

NON_CLINICAL_DISCLAIMER = (
    "Research/educational pipeline. NOT a certified medical device. "
    "NOT for clinical diagnosis, treatment planning, or patient-care decisions."
)


@app.post("/v1/cases")
def upload_case():
    """FR-5.1. TODO: wire to src/data/ingestion.py::register_case."""
    raise NotImplementedError


@app.post("/v1/cases/{case_id}/infer")
def trigger_inference(case_id: str):
    """FR-5.2. TODO: wire to src/inference/sliding_window.py."""
    raise NotImplementedError


@app.get("/v1/cases/{case_id}/result")
def get_result(case_id: str):
    """FR-5.3, FR-6.6. Include NON_CLINICAL_DISCLAIMER in every response body."""
    raise NotImplementedError


@app.get("/v1/models")
def list_models():
    """FR-5.4. TODO: wire to model registry (MLflow)."""
    raise NotImplementedError
