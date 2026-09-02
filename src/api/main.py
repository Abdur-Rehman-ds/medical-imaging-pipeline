"""FastAPI inference service.

Implements:
  FR-5.1 — POST /v1/cases                      upload a case
  FR-5.2 — POST /v1/cases/{id}/infer           trigger inference (async)
  FR-5.3 — GET  /v1/cases/{id}/result          retrieve status/result
  FR-5.5 — path-versioned API (/v1/)
  FR-5.7 — structured error responses (error code, message, correlation ID)
  FR-6.6 / Section 14 — non-clinical-use disclaimer in every result body

Stubs (Should-priority, decisions pending — recorded 2026-09-02):
  FR-5.4 — GET /v1/models (needs model-registry format decision)
  FR-5.6 — auth + rate limiting (needs auth-scheme decision)

DESIGN (decision recorded 2026-09-02): FR-5.2/5.3 imply asynchronous
inference — the trigger returns immediately and the caller polls the
result endpoint. Implemented with FastAPI BackgroundTasks and a per-case
status.json in the case's storage directory (simplest option; swappable
for a task queue later without changing the API surface).

MODEL SOURCE (decision recorded 2026-09-02): checkpoint path from the
MODEL_CHECKPOINT env var. If unset, an UNTRAINED model is used and every
response carries model_version="untrained-dev" so development output can
never be mistaken for real results.
"""

import json
import os
import shutil
import tempfile
import uuid
from pathlib import Path

import numpy as np
import torch
from fastapi import BackgroundTasks, FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from omegaconf import OmegaConf

from src.data.ingestion import IngestionFailure, get_storage_dir, register_case

REPO_ROOT = Path(__file__).resolve().parents[2]

app = FastAPI(title="Medical Imaging Diagnostic Pipeline API", version="1.0")

NON_CLINICAL_DISCLAIMER = (
    "Research/educational pipeline. NOT a certified medical device. "
    "NOT for clinical diagnosis, treatment planning, or patient-care decisions."
)


def error_response(status: int, code: str, message: str) -> JSONResponse:
    """FR-5.7 — every failure path returns this structure."""
    return JSONResponse(
        status_code=status,
        content={"error_code": code, "message": message,
                 "correlation_id": uuid.uuid4().hex[:12]},
    )


def case_dir(case_id: str) -> Path:
    return get_storage_dir() / case_id


def write_status(cid: str, **fields) -> None:
    (case_dir(cid) / "status.json").write_text(json.dumps(fields))


def read_status(cid: str) -> dict:
    p = case_dir(cid) / "status.json"
    return json.loads(p.read_text()) if p.exists() else {"status": "uploaded"}


def load_model():
    """Model per MODEL_CHECKPOINT decision above. Returns (model, version)."""
    from src.models.train import build_model

    model_cfg = OmegaConf.load(REPO_ROOT / "configs/model/unet3d.yaml")
    model = build_model(model_cfg)
    ckpt_path = os.environ.get("MODEL_CHECKPOINT")
    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        return model, Path(ckpt_path).stem
    return model, "untrained-dev"


def run_case_inference(cid: str) -> None:
    """Background task: preprocess -> sliding-window inference ->
    post-process -> persist results (FR-4.1..4.5 via src/inference)."""
    try:
        from src.data.preprocessing import build_case_dict, build_inference_transforms
        from src.inference.sliding_window import infer_case

        inf_cfg = OmegaConf.load(REPO_ROOT / "configs/inference/default.yaml")
        data = build_case_dict(case_dir(cid), cid, include_label=False)
        image = build_inference_transforms(inf_cfg)(data)["image"]
        affine = np.asarray(image.affine) if hasattr(image, "affine") else np.eye(4)

        model, version = load_model()
        summary = infer_case(
            torch.as_tensor(image), affine, model, inf_cfg,
            out_dir=case_dir(cid) / "results", case_id=cid,
            model_version=version,
            voxel_volume_mm3=1.0,  # FR-2.2 — 1mm isotropic after resampling
        )
        write_status(cid, status="completed", summary=summary)
    except Exception as e:  # FR-5.7 — surface failures with a correlation ID
        write_status(cid, status="failed", error=str(e),
                     correlation_id=uuid.uuid4().hex[:12])


@app.post("/v1/cases")
async def upload_case(
    t1: UploadFile = File(...), t1ce: UploadFile = File(...),
    t2: UploadFile = File(...), flair: UploadFile = File(...),
):
    """FR-5.1 — upload 4 modality files, validate (FR-1.2/1.3), register
    (FR-1.5), return the case ID."""
    with tempfile.TemporaryDirectory() as tmp:
        paths = {}
        for name, up in (("t1", t1), ("t1ce", t1ce), ("t2", t2), ("flair", flair)):
            suffix = "".join(Path(up.filename or f"{name}.nii.gz").suffixes) or ".nii.gz"
            dest = Path(tmp) / f"{name}{suffix}"
            with dest.open("wb") as f:
                shutil.copyfileobj(up.file, f)
            paths[name] = dest
        try:
            upload = register_case(paths)
        except IngestionFailure as e:
            return error_response(400, e.errors[0].value,
                                  "Upload rejected: " + ", ".join(x.value for x in e.errors))
    write_status(upload.case_id, status="uploaded")
    return {"case_id": upload.case_id, "status": "uploaded"}


@app.post("/v1/cases/{case_id}/infer")
def trigger_inference(case_id: str, background_tasks: BackgroundTasks):
    """FR-5.2 — start async inference; poll FR-5.3 for the outcome."""
    if not case_dir(case_id).exists():
        return error_response(404, "CASE_NOT_FOUND", f"No case {case_id}")
    status = read_status(case_id)
    if status.get("status") == "processing":
        return error_response(409, "ALREADY_PROCESSING", "Inference already running")
    write_status(case_id, status="processing")
    background_tasks.add_task(run_case_inference, case_id)
    return {"case_id": case_id, "status": "processing"}


@app.get("/v1/cases/{case_id}/result")
def get_result(case_id: str):
    """FR-5.3, FR-6.6 — status/result with the mandatory disclaimer."""
    if not case_dir(case_id).exists():
        return error_response(404, "CASE_NOT_FOUND", f"No case {case_id}")
    status = read_status(case_id)
    status["case_id"] = case_id
    status["disclaimer"] = NON_CLINICAL_DISCLAIMER
    return status


@app.get("/v1/models")
def list_models():
    """FR-5.4 (stub) — pending model-registry format decision."""
    return error_response(501, "NOT_IMPLEMENTED",
                          "Model listing pending registry decision (see SRS Appendix E)")
