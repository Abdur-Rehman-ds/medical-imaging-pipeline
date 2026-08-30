# Medical Imaging Diagnostic Pipeline — BraTS Brain Tumor Segmentation

Automated brain tumor segmentation from multi-modal MRI (T1, T1ce, T2, FLAIR)
using the BraTS dataset. Research/portfolio pipeline — **not a certified
medical device, not for clinical use** (see SRS Section 14).

Full requirements: `docs/SRS_Medical_Imaging_Pipeline.docx` (source of truth).
Every module below is traceable to a Functional/Non-Functional Requirement
ID (`FR-x.y` / NFR) from that document.

## Repo layout

```
configs/          Hydra/YAML configs (FR-2.6: preprocessing config must be versioned;
                   "all configuration must be code-defined" — Section 2.5)
  data/            dataset paths, modality list, split manifest location
  model/           architecture choice + hyperparams (unet3d.yaml, nnunet.yaml)
  training/        loss, optimizer, patch size, AMP, CV folds, epoch budget
  inference/       sliding-window patch size/overlap, post-processing thresholds

src/
  data/            Data Layer — FR-1.x (ingestion/validation), FR-2.x (preprocessing)
  models/          Model Layer — FR-3.x (training, checkpointing, CV)
  inference/       Serving Layer core — FR-4.x (sliding-window, post-processing)
  api/             Serving Layer REST — FR-5.x (FastAPI endpoints)
  monitoring/      Operations Layer — FR-7.x (logging, drift, audit log)

frontend/          Application Layer — FR-6.x (upload UI, slice viewer, reports)
notebooks/         Kaggle/Colab training notebooks (checkpoint-resume, FR-3.7)
docker/            Dockerfiles + docker-compose.yml (Section 10.1)
.github/workflows/ CI: lint, type-check, unit tests (Section 10.3)
tests/             Unit/integration tests (Section 11)
docs/              SRS and other reference docs
```

## Environment split (per SRS Section 2.4 / 2.6)

- **Local**: repo, Docker configs, FastAPI + frontend development. No GPU
  needed for this half of the project.
- **Kaggle (training)**: T4/P100, 16 GB VRAM, 30 GPU-hrs/week, 12-hr session
  cap. All training code must be checkpoint-resume (weights, optimizer
  state, epoch counter persisted at least once per epoch — FR-3.3, FR-3.7,
  Section 2.5) since a session can end mid-run.

## Setup — see the two step-by-step walkthroughs below this message
(GitHub repo creation, Kaggle notebook + dataset setup).

## Design decisions NOT specified by the SRS (flagged, not silently chosen)

These were picked as sensible defaults — override any of them if you'd
rather go a different way:

1. **License**: MIT (SRS Section 14.4 says "chosen by the author", doesn't specify which).
2. **Package management**: plain `requirements.txt` + `pip`, not Poetry —
   simplest to mirror on a Kaggle notebook image, which has no Poetry support.
3. **Src layout**: `src/` package layout (vs. flat root) — plays better
   with CI type-checking and avoids import-path collisions with notebooks.
4. **Config framework**: Hydra (SRS just says "Hydra/YAML") — chosen over
   plain YAML+argparse for composable configs across data/model/training.
5. **Frontend MVP**: SRS allows React+Cornerstone.js/OHIF OR Streamlit for
   MVP — this scaffold leaves `frontend/` empty pending that choice; flag
   back to me which you want and I'll scaffold it.
