# Project Handoff v5 — Medical Imaging Pipeline (BraTS)

**Date: 10 September 2026.** Replaces v4 entirely. For a new Claude
conversation: attach this plus SRS v1.4 (docs/, source of truth,
Appendix E decisions 1-16 + withdrawn gate proposal).

Repo: https://github.com/Abdur-Rehman-ds/medical-imaging-pipeline
Latest commit: `346a29c`. Branch clean (except requirements.txt.bak-v1,
untracked local safety copy — deletable).

## Headline state

- **Folds 0 AND 1 trained, both GATE-PASSING** (gate: ET>=0.75 /
  TC>=0.80 / WT>=0.85, unchanged):
  - Fold 0 (best e109): ET 0.799 / TC 0.835 / WT 0.896, mean 0.8439
  - Fold 1 (best e099): ET 0.773 / TC 0.845 / WT 0.894, mean 0.8372
  - Cross-fold mean±std: ET 0.786±0.013, TC 0.840±0.005, WT 0.895±0.001
- **Fold 2 chunk 1 COMPLETE** (Kaggle Version 8, e0-39, W&B run
  p8lsfm3y). Chunks 2-3 THIS week after quota reset: e40-79 then
  e80-119, max_epochs 80 then 120, ~7 hrs each, resume-copy
  fold2_latest.pt from prior version's attached output (VERIFY input
  serves latest version). ~1.8 hrs quota was left unspent last week.
- **Docker + CI/CD DONE (09-10 Sep, SRS Section 10, decisions 14-16)**:
  - requirements.txt (direct deps) + requirements.lock.txt (full pin,
    what Docker/CI install). THREE hidden dynamic deps found & pinned:
    python-multipart (FastAPI upload endpoints — container crashed at
    start), scipy (MONAI post-processing — inference crashed), httpx
    (fastapi.testclient — CI tests crashed). Each caught by a deeper
    verification layer than the last.
  - docker/Dockerfile.api: python:3.12-slim, CPU-only (deliberate
    divergence from 10.1 CUDA wording, decision 15), installs lockfile,
    checkpoints volume-mounted never baked in. .dockerignore keeps
    context ~300 kB.
  - docker/Dockerfile.frontend: node:22-slim build (npm ci) ->
    nginx:1.27-alpine; docker/nginx.conf proxies /v1 to api service,
    500M body limit, 600s read timeout.
  - docker-compose.yml: api + frontend only (postgres/minio dropped per
    decision 7). `docker compose -f docker/docker-compose.yml up` from
    repo root; frontend :8080, API :8000. MODEL_CHECKPOINT env override
    switches served fold without rebuild — verified with fold1.
    Verified end-to-end in browser: upload -> inference -> overlay ->
    PDF/JSON export.
  - CI (.github/workflows/ci.yml): ruff + non-blocking mypy + pytest
    (lockfile install, py3.12) + both image builds + model-less API
    smoke test. ruff.toml ignores B008/BLE001 with rationale.
    **Run #46 = first green run ever; runs 1-45 all failed** (stub
    workflow failing silently since Aug 30 — nobody looked).
- **API & frontend 100% done** (unchanged from v4; FR-5.1..5.7,
  FR-6.1..6.6). Tests 16/16, coverage NFR met (preprocessing 80%,
  sliding_window 97%).
- **SRS at v1.4** (docs/SRS_Medical_Imaging_Pipeline_v1.4.docx):
  decisions 14 (lockfile scheme), 15 (Docker scope), 16 (CI shape)
  recorded 10-Sep. docs/ also holds v1.3 and the stale unversioned
  Sep-2 docx (candidate for deletion).

## Environment

- Local: Ubuntu laptop, repo ~/Downloads/medical-imaging-pipeline,
  venv .venv (PYTHONPATH=. .venv/bin/python; base conda lacks monai).
  Local torch 2.13.0+cpu. python-docx now in venv (SRS editing only,
  deliberately NOT in requirements.txt).
- **Docker installed** (apt docker.io 29.1.3 + docker-compose-v2
  2.40.3; user in docker group — new shells may need `newgrp docker`).
  Legacy builder deprecation warning on `docker build` is harmless;
  compose uses BuildKit.
- API dev: `MODEL_CHECKPOINT=models/fold0_best_e109_d0.8439.pt
  PYTHONPATH=. .venv/bin/uvicorn src.api.main:app --port 8000`.
  Env vars: MODEL_DIR, API_KEY, RATE_LIMIT_PER_MINUTE.
- Frontend dev: `cd frontend && npm run dev` (:5173 -> proxies :8000).
- Kaggle notebook brats-baseline-training, T4, 6-cell structure,
  W&B naming fold{N}-from-e{start}, group=fold{N}.

## Remaining work (priority order)

1. **Fold 2 chunks 2-3** (this week's quota) -> append fold-2 results
   to Appendix E withdrawal note; 3-fold mean±std final. Download +
   re-zip best fold-2 checkpoint into models/ (gotcha 1).
2. nnU-Net benchmark path (FR-3.6) — GPU-heavy, after fold 2. Then
   possible decision: 3-fold ensemble inference (avg probabilities,
   ~+0.01-0.02 Dice, nearly free).
3. Smaller Shoulds — build or formally scope out in Appendix E:
   FR-1.4 DICOM stub, FR-7.2/7.3 drift stats + alerts, load/security
   tests (Section 11). Also: make mypy blocking after fixing the 5
   union-narrowing errors (decision 16 follow-up).
4. Polish (Section 12 phase 8): README (still Aug-30 scaffold text),
   architecture diagram, demo write-up. Delete stale docs/ dupes and
   requirements.txt.bak-v1.

## Known gotchas (essentials — v4 list still applies, plus new)

1. Kaggle serves .pt UNPACKED; per-file downloads arrive as .zip of
   contents. Restore: `cd <parent> && zip -r -0 out.pt <folder>` —
   entries must sit under ONE top-level dir or torch.load fails.
2. Attached notebook inputs FREEZE on old versions. Before any resume
   launch: expand the input, check checkpoints/ filenames; fix via
   input menu "Check for updates" (or remove + re-add).
3. Interactive editor sessions get their OWN T4 and will run Cell 6 as
   a duplicate training run. Stop the session immediately after Save &
   Run All; never run Cell 6 interactively.
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
   (Deliberate `--amend` + `--force-with-lease` on an unshared branch
   is fine — used once for the SRS 1.4 paragraph-placement fix.)
10. Cosine LR fast-forward warning on resume = expected, harmless.
11. Untrained CPU inference seconds; real weights on laptop CPU =
    minutes. Patience, not a hang.
12. Large text pastes into Claude chat can convert to attachments that
    arrive EMPTY. For logs: screenshot the tail, or paste <=20 lines.
13. **NEW — hidden dynamic deps**: import-grep does NOT find deps
    loaded lazily (FastAPI multipart, MONAI scipy, testclient httpx).
    Any lockfile change => CI dress rehearsal BEFORE push: fresh
    `git clone` to /tmp + fresh venv from requirements.txt + pytest.
    This exactly reproduces CI; GitHub is not the test bench.
14. **NEW — ruff --fix edits files silently**: after any --fix, run
    `git status` and commit ALL modified files, not just the ones you
    edited by hand (cost us CI run 42-43 confusion).
15. **NEW — check the Actions tab** (or the curl one-liner below)
    after every push; the old stub failed silently 41 times:
    `curl -s "https://api.github.com/repos/Abdur-Rehman-ds/medical-imaging-pipeline/actions/runs?per_page=3"`
16. **NEW — python-docx SRS edits**: anchor searches must use prefixes
    unique in the WHOLE document (decision-13 text nearly collided
    with a Section 2.5 bullet); verify insertion ORDER in memory and
    assert before saving; eyeball in LibreOffice after.

## How the user works (unchanged, essential)

One literal copy-paste block at a time; plain "what and why" before
each task; explicit click-by-click for any UI; reads raw pasted output
carefully (multiple real bugs surfaced there); heredoc for file writes;
compile-check all touched .py files; verify file state on disk. Never
write against unseen code — cat the file first. Record every
SRS-silent decision in Appendix E with rationale.
