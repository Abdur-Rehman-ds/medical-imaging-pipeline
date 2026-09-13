# Project Handoff v6 — Medical Imaging Pipeline (BraTS)

**Date: 13 September 2026.** Replaces v5 entirely. For a new Claude
conversation: attach this plus SRS v1.5 (docs/, source of truth,
Appendix E decisions 1-16 + completed withdrawal note with full
3-fold evidence).

Repo: https://github.com/Abdur-Rehman-ds/medical-imaging-pipeline
Latest commit: `3e898e9`. Untracked locals: requirements.txt.bak-v1
(deletable safety copy); a LibreOffice .~lock file appears in docs/
while the SRS is open — ignore it.

## Headline state

- **3-FOLD BASELINE COMPLETE — ALL FOLDS GATE-PASSING** (gate:
  ET>=0.75 / TC>=0.80 / WT>=0.85, unchanged):
  - Fold 0 (best e109): ET 0.799 / TC 0.835 / WT 0.896, mean 0.8439
  - Fold 1 (best e099): ET 0.773 / TC 0.845 / WT 0.894, mean 0.8372
  - Fold 2 (best e099): ET 0.797 / TC 0.874 / WT 0.906, mean 0.8590
    (early-stopped e119, patience 20, across 3 resumed sessions)
  - **Final 3-fold mean±std: ET 0.790±0.012, TC 0.851±0.017,
    WT 0.899±0.005, overall 0.847±0.009**
- Fold-2 checkpoint restored to models/ (gotcha 1 re-zip done) and
  **promoted as compose default** (commit 7b8ae9f, CI run 48 green).
- **SRS at v1.5** (docs/SRS_Medical_Imaging_Pipeline_v1.5.docx):
  fold-2 + final 3-fold results appended to the Appendix E withdrawal
  note as promised at rev 1.3. Gate and scope unchanged. Stale
  unversioned Sep-2 docx deleted from docs/ (commit 3e898e9); docs/
  now holds versioned copies only (v1.3, v1.4, v1.5).
- **Docker + CI/CD steady state** (decisions 14-16): requirements.txt
  (direct deps) + requirements.lock.txt (full pin — what Docker/CI
  install). CPU-only api image (python:3.12-slim), frontend
  node build -> nginx proxying /v1. compose = api + frontend only.
  MODEL_CHECKPOINT env override switches served fold without rebuild.
  CI: ruff + non-blocking mypy + pytest + both image builds +
  model-less smoke test. Green runs 46-48 (49 = SRS v1.5 push).
- **API & frontend functionally done** (FR-5.1..5.7, FR-6.1..6.6).
  Tests 16/16, coverage NFR met (preprocessing 80%, sliding_window
  97%). Frontend is single-screen — redesign is now a stated goal
  (below).
- **~16.6 GPU-hrs quota left this week.**

## Environment

- Local: Ubuntu laptop, repo ~/Downloads/medical-imaging-pipeline,
  venv .venv (PYTHONPATH=. .venv/bin/python; base conda lacks monai).
  Local torch 2.13.0+cpu. python-docx in venv (SRS editing only,
  deliberately NOT in requirements.txt).
- Docker: apt docker.io 29.1.3 + docker-compose-v2 2.40.3; user in
  docker group (new shells may need `newgrp docker`). Legacy-builder
  warning on `docker build` harmless; compose uses BuildKit.
- Compose: `docker compose -f docker/docker-compose.yml up` from repo
  root; frontend :8080, API :8000. Default served model = fold-2
  checkpoint.
- API dev: `MODEL_CHECKPOINT=models/<fold2 best .pt> PYTHONPATH=.
  .venv/bin/uvicorn src.api.main:app --port 8000`. Env vars:
  MODEL_DIR, API_KEY, RATE_LIMIT_PER_MINUTE.
- Frontend dev: `cd frontend && npm run dev` (:5173 -> proxies :8000).
- Kaggle notebook brats-baseline-training, T4 only, 6-cell structure,
  W&B naming fold{N}-from-e{start}, group=fold{N}. pip install
  --no-deps monai (protect torch). Restart session after CUDA crash.

## Remaining work (priority order)

1. **nnU-Net probe run (~1 hr GPU)** to measure real epoch cost on
   the T4; then set the epoch/fold cap for the benchmark path
   (FR-3.6) FROM DATA, and record the cap as an Appendix E decision.
2. **nnU-Net benchmark run** within the measured budget. Afterwards,
   possible cheap win: 3-fold ensemble inference (avg probabilities,
   ~+0.01-0.02 Dice) — decide and record either way.
3. **Frontend redesign (NEW GOAL, scoped)**: multi-screen per SRS
   Section 6.1 — Upload / Viewer / History / About. React Router,
   clinical dark theme, loading states for slow CPU inference.
   History screen needs ONE new backend endpoint (GET listing of
   processed cases); zero other backend changes. Admin screen
   formally scoped OUT in Appendix E. Record the whole expansion as
   an Appendix E decision before building.
4. Smaller Shoulds — build or formally scope out in Appendix E:
   FR-1.4 DICOM stub, FR-7.2/7.3 drift stats + alerts, load/security
   tests (Section 11). Make mypy blocking after fixing the 5
   union-narrowing errors (decision 16 follow-up).
5. Polish (Section 12 phase 8): README (still Aug-30 scaffold text),
   architecture diagram, demo write-up. Delete
   requirements.txt.bak-v1 when comfortable.

## Known gotchas (carried forward — all still apply)

1. Kaggle serves .pt UNPACKED; per-file downloads arrive as .zip of
   contents. Restore: `cd <parent> && zip -r -0 out.pt <folder>` —
   entries must sit under ONE top-level dir or torch.load fails.
2. Attached notebook inputs FREEZE on old versions. Before any resume
   launch: expand the input, check checkpoints/ filenames; fix via
   input menu "Check for updates" (or remove + re-add).
3. Interactive editor sessions get their OWN T4 and will run Cell 6
   as a duplicate training run. Stop the session immediately after
   Save & Run All; never run Cell 6 interactively.
4. Kaggle's top-level Logs tab shows the last FINISHED version. For a
   running version: Version History -> click the version -> Logs.
5. `.gitignore` patterns must be anchored (`/models/`, `/data/`).
6. Terminal heredoc echoes truncate/mangle — verify on disk (wc -l,
   tail, grep, py_compile) after every write; never trust the display.
7. Multi-line code replacements: anchored-Python-patch pattern; sed
   only for single-line edits.
8. Real BraTS files are .nii not .nii.gz; test both for path code.
9. Failed `git push` claiming remote at your own new hash = push
   landed; `git fetch && git status -sb`, never panic-force-push.
10. Cosine LR fast-forward warning on resume = expected, harmless.
11. Untrained CPU inference seconds; real weights on laptop CPU =
    minutes. Patience, not a hang.
12. Large text pastes into Claude chat can convert to attachments
    that arrive EMPTY. For logs: screenshot the tail, or paste <=20
    lines.
13. Hidden dynamic deps: import-grep does NOT find deps loaded lazily
    (FastAPI multipart, MONAI scipy, testclient httpx). Any lockfile
    change => CI dress rehearsal BEFORE push: fresh `git clone` to
    /tmp + fresh venv from requirements.txt + pytest. This exactly
    reproduces CI; GitHub is not the test bench.
14. ruff --fix edits files silently: after any --fix, run
    `git status` and commit ALL modified files.
15. Check the Actions tab (or curl the runs API) after every push;
    the old stub failed silently 41 times.
16. python-docx SRS edits: anchor searches must use prefixes unique
    in the WHOLE document; assert paragraph order in memory before
    saving; re-verify from disk after save; eyeball in LibreOffice.
17. **NEW — pasting terminal scrollback back into the shell executes
    it as commands** (harmless syntax errors this time, but could
    bite). Paste only the single command block, never prior output.

## How the user works (unchanged, essential)

One literal copy-paste block at a time; plain "what and why" before
each task; explicit click-by-click for any UI; reads raw pasted
output carefully (multiple real bugs surfaced there); heredoc for
file writes; compile-check all touched .py files; verify file state
on disk. Never write against unseen code — cat the file first.
Record every SRS-silent decision in Appendix E with rationale.
