# Project Guide — Medical Imaging Pipeline (BraTS)

The complete reference for this project: what was built, every result,
and how to operate or extend each part. Written for future readers
(including future-me) with no assumed context. Companion documents:
README.md (summary), docs/DEMO.md (demo script),
docs/SRS_Medical_Imaging_Pipeline_v1.11.docx (formal spec, all 23
recorded decisions), Project_Handoff_v9.md (closing state).

> NOT a medical device. Research/educational project only. No
> regulatory clearance. Never use for clinical decisions.

## 1. What this project is

An end-to-end system that takes four MRI scans of a brain (T1, T1ce,
T2, FLAIR) and automatically outlines the tumor in three sub-regions,
served through a web app and REST API, fully containerized. Trained
from scratch on the BraTS2020 dataset using only free Kaggle GPUs
(30 hrs/week quota), built Aug-Sep 2026 in ~6 weeks.

## 2. Results — all numbers

### 2.1 Baseline: 3D U-Net, 3-fold cross-validation

3-fold patient-level CV (decision 13: 3 folds not 5, sized to the
weekly GPU quota; each fold trains on 4/5 of the 368-case manifest
and validates on its held-out fold). Dice coefficient, higher = better
overlap with expert annotation (1.0 = perfect):

| Fold | Best epoch | ET | TC | WT | Mean |
| --- | --- | --- | --- | --- | --- |
| 0 | 109 | 0.799 | 0.835 | 0.896 | 0.8439 |
| 1 | 99 | 0.773 | 0.845 | 0.894 | 0.8372 |
| 2 | 99 | 0.797 | 0.874 | 0.906 | 0.8590 |
| **Mean ± std** | — | **0.790 ± 0.012** | **0.851 ± 0.017** | **0.899 ± 0.005** | **0.847 ± 0.009** |

Regions: ET = enhancing tumor (active tissue), TC = tumor core
(ET + necrosis), WT = whole tumor (everything including edema).

### 2.2 Quality gate (all folds pass individually)

| Region | Gate (Dice >=) | Achieved (mean) |
| --- | --- | --- |
| WT | 0.85 | 0.899 |
| TC | 0.80 | 0.851 |
| ET | 0.75 | 0.790 |

### 2.3 nnU-Net benchmark (same validation split)

nnU-Net v2, 100 epochs (epoch cap measured on the T4), its 5-fold
split overridden to validate on exactly the baseline's fold-0
patients (decision 17) — a true apples-to-apples comparison:

| Model (fold-0 val, n=74) | ET | TC | WT | Mean |
| --- | --- | --- | --- | --- |
| 3D U-Net baseline (ours) | 0.799 | 0.835 | 0.896 | 0.8439 |
| nnU-Net (100 epochs) | 0.824 | 0.860 | 0.917 | 0.8668 |

nnU-Net wins every region by +0.02-0.025 at a comparable budget —
expected, and why it exists here as the credibility benchmark. Its
checkpoint is NOT servable by this API (different architecture);
the served model is always the baseline.

### 2.4 Other measured numbers

| Metric | Value |
| --- | --- |
| Inference latency (CPU, full volume) | ~36-45 s (budget: <= 4 min) |
| Drift score, real training case | 0.6167 (threshold 3.0) |
| Training cost per fold | ~20 GPU-hrs (T4, free tier) |
| Dataset | BraTS2020, 369 cases (368 valid for training) |
| Tests / CI | 35 tests, mypy blocking, CI green |

## 3. The trained models (pre-trained checkpoints)

Three checkpoints live in models/ (150 MB each, NOT in git — they
stay on this machine and in Kaggle notebook outputs):

| File | What it is | Val Dice |
| --- | --- | --- |
| fold0_best_e109_d0.8439.pt | Fold-0 model, epoch 109 | 0.8439 |
| fold1_best_e099_d0.8372.pt | Fold-1 model, epoch 99 | 0.8372 |
| fold2_best_e099_d0.8590.pt | Fold-2 model, epoch 99 — THE SERVED DEFAULT | 0.8590 |

Filename convention (decision 5): fold{i}_best_e{epoch}_d{dice}.pt —
the API parses metadata straight from these names (decision 11).

### Switching the served model (no rebuild)

    MODEL_CHECKPOINT=models/fold0_best_e109_d0.8439.pt \
      docker compose -f docker/docker-compose.yml up -d

Unset MODEL_CHECKPOINT entirely -> the API serves an UNTRAINED model
and every response says model_version=untrained-dev, so dev output
can never be mistaken for real results (decision recorded 02-Sep).

### Reusing a checkpoint as pre-trained weights elsewhere

Each .pt holds model_state_dict + optimizer state. To load the
weights in another project:

    import torch
    from src.models.train import build_model
    from omegaconf import OmegaConf
    cfg = OmegaConf.load("configs/model/unet3d.yaml")
    model = build_model(cfg)
    ckpt = torch.load("models/fold2_best_e099_d0.8590.pt",
                      map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])

Input contract: 4-channel (T1,T1ce,T2,FLAIR), RAS orientation, 1 mm
isotropic, percentile-clipped z-score per modality — reuse
src/data/preprocessing.py or reproduce those steps exactly, or the
weights see garbage.

### Ensemble warning (decision 19)

Averaging the three folds was considered and REJECTED for
evaluation: under the 3-fold structure every case was training data
for at least two of the three models, so any measured ensemble Dice
would be leakage-inflated. Ensembling for genuinely NEW uploads is
legitimate; claiming a number for it is not, without a held-out set
that was never created.

## 4. Training — how it was done, and how to redo it

### The setup that produced the results

| Item | Value |
| --- | --- |
| Hardware | Kaggle free tier, NVIDIA T4 (16 GB) — T4 ONLY, P100 broken with this PyTorch |
| Quota | 30 GPU-hrs/week, 12-hr session cap |
| Model | 3D U-Net (MONAI), from scratch |
| Loss | Dice + Cross-Entropy, 1:1 |
| Optimizer | AdamW, cosine LR decay |
| Patches | 96^3, foreground-oversampled, AMP on |
| Validation | Full sliding-window every 5 epochs + final epoch (quota-sized, decision 3) |
| Early stopping | Patience 20 epochs without val-Dice improvement, counter persisted in checkpoints |
| Fold cost | ~20 GPU-hrs each; fold 2 spanned 3 resumed weekly sessions |

The defining constraint: sessions die at 12 hrs and quota resets
weekly, so EVERYTHING is checkpoint-resume — weights, optimizer
state, epoch counter, and the early-stop counter persist every
epoch, and a killed session resumes exactly where it stopped.

### To train again (only if something new is wanted — nothing needs it)

1. Kaggle -> New Notebook -> Accelerator: GPU T4 x1 (never P100).
2. Add Input: the BraTS2020 dataset
   (awsaf49/brats20-dataset-training-validation).
3. Use the training notebooks in notebooks/ as the template — they
   clone this repo, pin monai with pip install --no-deps monai (so
   Kaggle's torch is not upgraded), and run per-fold training.
4. Config lives in configs/training/ and configs/model/unet3d.yaml —
   change folds/epochs/patch size THERE, never hardcoded (SRS 2.5).
5. Budget: assume ~20 GPU-hrs per fold. Check remaining quota on
   the Kaggle account page BEFORE starting, and restart the session
   fully after any CUDA crash.
6. Checkpoints appear in the notebook output; download and drop
   into models/ — the filename convention is what the API reads.

### nnU-Net benchmark path (decision 17)

scripts/make_nnunet_split.py converts the fold manifest to
nnU-Net's splits_final.json so it validates on baseline fold 0.
Labels remap 4->3 (nnU-Net wants consecutive integers); regions are
recomputed from saved predictions as label unions. Trainer:
nnUNetTrainer_100epochs (~5.7 hrs wall on a T4, one session).
Per-case metrics preserved as nnunet_fold0_region_metrics.json in
the Kaggle notebook nnunet-region-metrics.

## 5. The API — every endpoint

Base: http://localhost:8000 (direct) or via the frontend proxy at
/v1. Versioned path prefix /v1 (FR-5.5). All failures return the
FR-5.7 structure: {error_code, message, correlation_id}.

| Endpoint | What it does |
| --- | --- |
| POST /v1/cases | Upload 4 modality files, get a case_id |
| POST /v1/cases/{id}/infer | Start inference (returns immediately; async) |
| GET /v1/cases/{id}/result | Poll status; when completed, the full summary |
| GET /v1/cases | List all cases, newest first (History screen) |
| GET /v1/models | List checkpoints with parsed metadata; active flagged |
| GET /v1/cases/{id}/files/{kind} | Download mask/overlay files |

### The full flow by curl

    curl -X POST http://localhost:8000/v1/cases \
      -F t1=@t1.nii -F t1ce=@t1ce.nii -F t2=@t2.nii -F flair=@flair.nii
    curl -X POST http://localhost:8000/v1/cases/CASE_ID/infer
    curl http://localhost:8000/v1/cases/CASE_ID/result

First inference after a container start is the COLD one — model
loads from scratch, can take 2x the usual latency. Silence is not
failure (gotcha 23); poll patiently.

### Auth and limits (FR-5.6, decision 12; cap: decision 23)

| Env var | Effect |
| --- | --- |
| API_KEY | Unset = open (dev). Set = every request needs X-API-Key header; wrong/missing -> 401 |
| RATE_LIMIT_PER_MINUTE | Default 60 per client; 0 disables; exceeded -> 429 |
| MAX_UPLOAD_MB | Per-file cap, default 512; oversize -> 413, cut off mid-stream |
| DRIFT_THRESHOLD | Default 3.0; score above it -> drift_alert true |
| ALERT_WEBHOOK_URL | Optional; alert events POSTed there, fire-and-forget |

### Reading a result summary

per_label_volumes_mm3 and per_label_voxels: size of each region
(NCR_NET = dead core, edema = swelling, enhancing_tumor = active
tissue). confidence_summary: mean winning-class softmax probability
per region (decision 8) — the model's certainty about its own
labels, NOT proof of correctness. drift: {score, threshold, alert}
— see section 7. model_version: exactly which checkpoint produced
this. disclaimer: always present, by design (decision 9).

## 6. The frontend — every screen

React (Router 7) SPA behind nginx, proxying /v1 to the API
(decision 18). Every route carries the persistent non-clinical
banner (FR-6.6).

| Route | Screen | What you do there |
| --- | --- | --- |
| / | Upload | Pick the 4 modality files, Run Inference, watch status |
| /cases/:id | Viewer | Axial/coronal/sagittal slices, scroll, per-label overlay toggles, opacity, per-region volumes in cm3, model badge, PDF/JSON export |
| /history | History | All past cases, newest first; click to re-open with overlay restored, no re-run |
| /about | About | Project description |

Report export is client-side (jsPDF, decision 10): the PDF snapshot
matches the exact view state on screen (slice, toggles, opacity).
JSON export is the backend summary verbatim.

Admin screen (registry/promotion controls) deliberately does NOT
exist — scoped out (decision 18) because the model registry is
deferred (decision 7); promotion is manual and documented.

## 7. Monitoring — the drift system

Every inference emits ONE structured JSON line to stdout, read via:

    docker compose -f docker/docker-compose.yml logs api | grep '"event": "inference"'

The line carries: case_id, latency_s, input_shape, model_version,
per-modality intensity stats (computed on the PREPROCESSED volume —
mean/std/p1/p50/p99/foreground-fraction), drift_score, threshold,
and drift_alert.

How drift_score works: case stats are compared against the training
distribution in configs/monitoring/reference_stats.json (built from
all 369 Kaggle-hosted cases by scripts/make_reference_stats.py, on
Kaggle CPU — no GPU needed). Score = mean normalized distance from
the training reference. Verified live: a genuine training case
scores 0.6167; threshold is 3.0. A score above threshold flips
drift_alert to true in the log AND in the case summary, and POSTs
to ALERT_WEBHOOK_URL if set. Foreground is defined by mode-based
background exclusion — the naive nonzero definition was observed
wrong live (fg_fraction 1.0) and fixed (decision 22).

What a HIGH score means in practice: the input does not look like
BraTS training data — wrong modality order, skull not stripped, a
different scanner protocol, or corruption. The segmentation may
still render but should not be trusted.

To regenerate the reference (only if preprocessing ever changes):
run scripts/make_reference_stats.py in a Kaggle CPU session against
the dataset, ~30 min, commit the JSON.

## 8. Running everything

### Daily use (the only two commands)

    cd ~/Downloads/medical-imaging-pipeline && \
      docker compose -f docker/docker-compose.yml up -d

Browser: http://localhost:8080. Stop with the same command ending
in "down". If the brats-up / brats-down aliases are installed in
~/.bashrc, those one words do it (brats-up also opens the browser).

Requirements on any machine: Docker + a checkpoint in models/
(checkpoints are volume-mounted, never in the images). Without a
checkpoint the API serves untrained-dev — UI works, results are
garbage by design.

### Getting test images

Best source: the training dataset itself. Kaggle ->
awsaf49/brats20-dataset-training-validation -> Data tab -> any
BraTS20_Training_XXX folder -> download its four modality files
(t1, t1ce, t2, flair — SKIP the _seg file, that is the answer key).
Kaggle serves them as .nii.zip; unzip, then upload the four .nii
through the UI, one per picker. Cases already processed live in
data/uploads/ and can be re-uploaded any time.

A random internet MRI will upload if it is NIfTI but produce
garbage: the model expects skull-stripped, co-registered
BraTS-style input — and the drift alert should fire, which is the
monitoring doing its job.

### Recording a demo

docs/DEMO.md is the narration script. On Wayland use GNOME's
built-in recorder (Ctrl+Alt+Shift+R; remove the time cap once with
gsettings set org.gnome.settings-daemon.plugins.media-keys
max-screencast-length 0); output lands in ~/Videos/Screencasts as
.webm. Pre-run the inference OFF camera so the video never waits.
To cut dead air, ffmpeg re-encode (copy-mode cutting FAILS on
GNOME webm — keyframe-sparse): trim two segments with
-filter_complex trim/concat and -c:v libvpx. Verified: a 165 s raw
take cut to 109 s cleanly.

### Local development (no Docker)

Always .venv/bin/python with PYTHONPATH=. from the repo root — the
(base) conda env lacks monai. Before any push: .venv/bin/ruff
check src/ tests/ AND .venv/bin/mypy src/ --ignore-missing-imports
(mypy is BLOCKING in CI). Tests: PYTHONPATH=. .venv/bin/pytest tests/

## 9. Troubleshooting — the stumbles that actually happened

| Symptom | Cause and fix |
| --- | --- |
| "No such file or directory" on any project command | You are in ~, not the repo. cd ~/Downloads/medical-imaging-pipeline first. Happened repeatedly. |
| ruff/mypy/python "command not found" | Using (base) conda, not the venv. Always .venv/bin/... |
| First inference after start seems dead — no logs, no result | COLD START (gotcha 23): model loading from scratch. Wait generously, check docker stats for CPU, only then debug. The boring answer beat three fancy theories. |
| Terminal stuck at > after a paste | Heredoc truncated (inner ``` fences break the chat copy button). Ctrl+C; the file was NOT written (heredoc is all-or-nothing); verify with tail, re-paste smaller. |
| Terminal echo shows garbage mid-command | Echo lies; disk is truth. tail/grep the file before concluding anything. |
| docx title page shows old version | python-docx edits need a LibreOffice eyeball after; stale .~lock files are crash leftovers — pgrep -a soffice, then delete. |
| "Queued" in GitHub Actions | Normal for 1-2 min. Only worry past 10. |
| Kaggle CUDA crash | Restart the session FULLY. T4 only, never P100. |
| Empty compose ps after reboot | Stack does not auto-start. Just up -d again; case data survives in the volume. |

## 10. Where everything lives

| Question | Look in |
| --- | --- |
| What was required, and every decision made | docs/SRS_..._v1.11.docx (Appendix E, decisions 1-23) |
| Final project state, gotchas 1-23 | Project_Handoff_v9.md |
| Quick results + how to run | README.md |
| Demo narration script | docs/DEMO.md |
| This operator's reference | docs/PROJECT_GUIDE.md |
| Code | src/ (data, models, inference, api, monitoring), frontend/, tests/, scripts/, configs/ |
| Repo | https://github.com/Abdur-Rehman-ds/medical-imaging-pipeline |
