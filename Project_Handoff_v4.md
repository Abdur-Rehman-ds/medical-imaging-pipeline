# Project Handoff v4 — Medical Imaging Pipeline (BraTS)

**Date: 08 September 2026.** Replaces v3 entirely. For a new Claude
conversation: attach this plus SRS v1.3 (docs/, source of truth,
Appendix E decisions 1-13 + withdrawn gate proposal).

Repo: https://github.com/Abdur-Rehman-ds/medical-imaging-pipeline
Latest commit: `e140f4b`. Branch clean.

## Headline state

- **Folds 0 AND 1 trained, both GATE-PASSING** (gate: ET>=0.75 /
  TC>=0.80 / WT>=0.85, unchanged):
  - Fold 0 (best e109): ET 0.799 / TC 0.835 / WT 0.896, mean 0.8439
  - Fold 1 (best e099): ET 0.773 / TC 0.845 / WT 0.894, mean 0.8372
  - Cross-fold mean±std: ET 0.786±0.013, TC 0.840±0.005, WT 0.895±0.001
- **Quality-gate proposal WITHDRAWN** (SRS rev 1.3) — baseline passes as
  written. **CV scope fixed at 3 folds** (Appendix E decision 13).
- **Fold 2 chunk 1 RUNNING** (Kaggle Version 8, e0-39, max_epochs=40,
  W&B `fold2-from-e0` run p8lsfm3y, launched 08-Sep on expiring quota,
  ~7 hrs). Chunks 2-3 next week: e40-79 then e80-119, max_epochs 80
  then 120, ~7 hrs each, resume-copy `fold2_latest.pt` from prior
  version's attached output (VERIFY input serves latest version).
- **Both best checkpoints deployed locally**: models/ holds
  fold0_best_e109_d0.8439.pt and fold1_best_e099_d0.8372.pt (gitignored).
- **API 100% done** — FR-5.1..5.7 all implemented incl. FR-5.4 model
  listing (filename-parsed metadata, decision 11) and FR-5.6 X-API-Key
  auth + rate limiting (decision 12; open when API_KEY env unset).
- **Frontend 100% done** — FR-6.1..6.6 incl. FR-6.5 JSON+PDF report
  export (client-side jsPDF, decision 10; getSnapshot forces NiiVue
  redraw before toDataURL) and animation pass (reduced-motion aware).
- **Tests: coverage NFR MET** — preprocessing 80%, sliding_window 97%
  (target >=70%, Section 5). 16 unit tests: test_preprocessing.py
  (FR-2.x on synthetic NIfTI), test_postprocessing.py (FR-4.x),
  test_api_models_auth.py (FR-5.4/5.6). Run:
  `PYTHONPATH=. .venv/bin/python -m pytest`

## Environment (unchanged from v3 except as noted)

- Local: Ubuntu laptop, repo ~/Downloads/medical-imaging-pipeline,
  venv .venv (PYTHONPATH=. .venv/bin/python; base conda lacks monai).
  Local torch is 2.13.0+cpu (Kaggle: 2.10.0+cu128) — both fine.
- API: `MODEL_CHECKPOINT=models/fold0_best_e109_d0.8439.pt PYTHONPATH=.
  .venv/bin/uvicorn src.api.main:app --port 8000`. New env vars:
  MODEL_DIR (FR-5.4), API_KEY + RATE_LIMIT_PER_MINUTE (FR-5.6).
- Frontend: `cd frontend && npm run dev` (:5173 -> proxies :8000).
  New dep: jspdf ^4.2.1.
- Kaggle notebook brats-baseline-training, T4, 6-cell structure.
  Version 8 = fold-2 chunk 1. W&B run naming now
  `fold{N}-from-e{start}`, group=fold{N} (resume="allow" dropped).
- Quota after V8 completes: ~1.4 hrs left this week; resets weekly.

## Remaining work (priority order)

1. Fold 2 chunks 2-3 (next week's quota) -> append fold-2 results to
   Appendix E withdrawal note; 3-fold mean±std final.
2. **Docker + CI/CD (SRS Section 10)** — biggest untouched area, zero
   GPU needed: pinned requirements lockfile (a 2.5 constraint), API +
   frontend Dockerfiles, docker-compose, GitHub Actions (lint, pytest,
   image build, synthetic smoke test).
3. nnU-Net benchmark path (FR-3.6) — GPU-heavy, after fold 2.
4. Smaller Shoulds — build or formally scope out in Appendix E:
   FR-1.4 DICOM stub, FR-7.2/7.3 drift stats + alerts, load/security
   tests (Section 11).
5. Polish (Section 12 phase 8): README, architecture diagram, demo
   write-up.

## Known gotchas (v3 list still valid; renumbered essentials)

1. Kaggle serves .pt UNPACKED; per-file downloads arrive as .zip of
   contents. Restore: `cd <parent> && zip -r -0 out.pt <folder>` —
   entries must sit under ONE top-level dir or torch.load fails.
2. Attached notebook inputs FREEZE on old versions. Before any resume
   launch: expand the input, check checkpoints/ filenames; fix via
   input menu "Check for updates" (or remove + re-add).
3. Interactive editor sessions get their OWN T4 and will happily run
   Cell 6 as a duplicate training run (cost us ~40 min once). Stop the
   session immediately after Save & Run All; never run Cell 6
   interactively.
4. Kaggle's top-level Logs tab shows the last FINISHED version. For a
   running version: Version History -> click the version -> Logs.
5. `.gitignore` patterns must be anchored (`/models/`, `/data/`) —
   unanchored forms silently ignore src/models/, src/data/.
6. Terminal heredoc echoes truncate/mangle — verify on disk (wc -l,
   tail, grep, py_compile) after every write; never trust the display.
7. Multi-line code replacements: anchored-Python-patch pattern
   (assert old in src; replace; write) — never sed for multi-line.
8. Real BraTS files are .nii not .nii.gz; test both for path code.
9. Failed `git push` claiming remote at your own new hash = push
   actually landed; `git fetch && git status -sb`, never force-push.
10. Cosine LR fast-forward warning on resume = expected, harmless.
11. Untrained CPU inference seconds; real weights on laptop CPU =
    minutes. Patience, not a hang.

## How the user works (unchanged, essential)

One literal copy-paste block at a time; plain "what and why" before
each task; explicit click-by-click for any UI; reads raw pasted output
carefully (multiple real bugs surfaced there); heredoc for file writes;
compile-check all touched .py files; verify file state on disk. Never
write against unseen code — cat the file first. Record every
SRS-silent decision in Appendix E with rationale.
