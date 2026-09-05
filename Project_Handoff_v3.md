# Project Handoff v3 — Medical Imaging Pipeline (BraTS)

**Date: 05 September 2026.** Replaces v2 entirely. For a new Claude
conversation: attach this plus SRS v1.1 (docs/, source of truth,
Appendix E has all recorded implementation decisions).

Repo: https://github.com/Abdur-Rehman-ds/medical-imaging-pipeline
Latest commit: `e3c4274`. Branch clean.

## Headline state

- **Fold 0 training: COMPLETE and gate-passing.** Final (epoch 109 best):
  mean Dice 0.8439 — ET 0.799 / TC 0.835 / WT 0.896 vs gate 0.75/0.80/0.85.
  Trained across 3 resumed Kaggle sessions (V3: e0-29, V4: e30-89,
  V5: e90-119 with cosine LR). Early stopping ended V5 at e119.
- **Best checkpoint deployed locally**: `models/fold0_best_e109_d0.8439.pt`
  (dir is gitignored). NOTE: Kaggle serves .pt files unzipped — must be
  re-zipped with the folder as archive root (see "Kaggle checkpoint
  download" below).
- **Frontend: React+Vite+NiiVue, functionally complete for Musts**
  (FR-6.1..6.4, 6.6) with dark clinical design pass. Verified end-to-end
  on real BraTS case (Training_297, a fold-0 VAL case) with the real
  model: coherent tumor segmentation rendered in-browser.
- **API: all Must endpoints implemented + tested** (FR-5.1/5.2/5.3/5.5/5.7),
  plus GET /v1/cases/{id}/files/{kind} for viewer data. FR-5.4/5.6 remain
  501 stubs (decisions pending).

## Environment (unchanged from v2 except as noted)

- Local: Ubuntu laptop, repo at ~/Downloads/medical-imaging-pipeline,
  venv at .venv (PYTHONPATH=. .venv/bin/python). (base) conda lacks monai.
- Frontend: Node 22 via NodeSource; `cd frontend && npm run dev` (:5173,
  proxies /v1 -> :8000).
- API: `MODEL_CHECKPOINT=models/fold0_best_e109_d0.8439.pt PYTHONPATH=.
  .venv/bin/uvicorn src.api.main:app --port 8000` (omit env var =
  untrained-dev).
- Kaggle: notebook brats-baseline-training, T4 only, 6-cell structure
  (v2's cell list still accurate). Version 5 = final fold-0 chunk.
  Quota used this week: ~27 of 30 hrs — effectively exhausted until reset.
- W&B: brats-segmentation. Runs are messy (9+ all named "fold0",
  crashes + smoke tests + chunks as separate runs) — cleanup on backlog.

## Training details that matter for fold 1

- Cosine LR schedule now in code+config (scheduler: t_max 120, eta_min
  1e-6), stepped per epoch, fast-forwarded `start_epoch` steps on resume
  from pre-scheduler checkpoints (harmless PyTorch warning at startup).
  Fold 1 gets it from epoch 0 — same 120-epoch shape, ~18-19 hrs total
  across 2 chunks (~9.1 min/epoch + ~2.5 min val every 5).
- Resume mechanics proven twice: attach prior version's output as
  notebook input, Cell 5 copies fold{N}_latest.pt into
  /kaggle/working/checkpoints (guarded: won't overwrite existing).
  CHECK the attached input serves the LATEST version before launching.
- Always stop the interactive session right after Save & Run All.

## Kaggle checkpoint download (hard-won)

Kaggle serves .pt (which are zip archives internally) UNPACKED, and
per-file downloads arrive as .zip of the contents. To restore:
`cd <extracted_parent> && zip -r -0 out.pt <checkpoint_folder_name>`
— archive entries MUST be under one top-level dir or torch.load fails
with "file in archive is not in a subdirectory".

## Frontend architecture

- frontend/src: App.jsx (shell, NiiVue viewer, overlay colormap logic),
  UploadPanel.jsx (FR-6.1), InferencePanel.jsx (FR-6.2/6.4, polls every
  3s), OverlayControls.jsx (FR-6.3 toggles+opacity), index.css (design
  system: CSS vars, .card/.btn/.banner classes).
- API status strings are VERIFIED: uploaded/processing/completed/failed
  (frontend checks "completed", reads summary.*, error under `error`).
- Per-label toggling: NVImage.setColormapLabel with alpha-0 for hidden
  labels; colors NCR/NET red, edema green, enhancing yellow (recorded
  decision; SRS silent).
- Real BraTS files are .nii (not .nii.gz) — files endpoint globs
  `{case_id}_{kind}.nii*` (bug fixed 2026-09-05, commit e3c4274).

## Remaining work (priority order)

1. Fold 1 on quota reset (config unchanged, fold_idx=1, max_epochs 120,
   two chunks). Then folds 2+ / the 3-fold vs 5-fold decision (SRS §2.6).
2. Quality-gate proposal (SRS Appendix E, PENDING APPROVAL): fold 0
   passes the gate as written — proposal likely withdrawable; needs
   cross-fold results for formal closure. Update Appendix E either way.
3. FR-6.5 report export (PDF/JSON + slice snapshot) — last frontend Should.
4. Frontend animation pass (subtle: card fade-ins, processing pulse) —
   user explicitly wants this.
5. FR-5.4 model listing + FR-5.6 auth (decisions needed first).
6. Cleanup backlog: torch.cuda.amp -> torch.amp; `data/` to .gitignore;
   W&B run naming (name=fold{N}-chunk{M} or similar); Cell 6 comment
   says "chunk 2" (cosmetic); empty trailing notebook cell.
7. Later: nnU-Net benchmark path, Docker/serving polish (SRS §10), CI
   extension.

## Known gotchas (v2 list still valid, plus)

9. Kaggle .pt download quirk (see above).
10. Real .nii vs test .nii.gz — test both extensions for any new
    file-path code.
11. A failed `git push` claiming remote "is at <your own new hash>"
    means the push actually landed (duplicate attempt) — verify with
    `git fetch && git status -sb`, never force-push.
12. Untrained-model CPU inference ~2 s on 64^3 synthetic; real volume
    with real weights: minutes on laptop CPU — patience, not a hang.

## How the user works (unchanged, essential — see v2 for full text)

One literal copy-paste block at a time; plain "what and why" before each
task; reads raw pasted output carefully (multiple real bugs surfaced
there); terminal paste display often truncates/mangles echoes — verify
file state on disk (wc -l / tail) rather than trusting the display.
