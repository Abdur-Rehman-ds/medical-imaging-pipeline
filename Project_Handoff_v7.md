# Project Handoff v7 — Medical Imaging Pipeline (BraTS)

**Date: 13 September 2026 (evening).** Replaces v6 entirely. For a new
Claude conversation: attach this plus SRS v1.6 (docs/, source of truth,
Appendix E decisions 1-18).

Repo: https://github.com/Abdur-Rehman-ds/medical-imaging-pipeline
Latest commit: `fa541a8`. Untracked locals: requirements.txt.bak-v1
(deletable safety copy); .~lock files in docs/ while LibreOffice is
open — ignore.

## Headline state

- **3-fold baseline COMPLETE + nnU-Net benchmark DONE (decision 17)**,
  same fold-0 split (decision-13 manifest), same ET/TC/WT regions:
  - Baseline fold 0 (e109): ET 0.799 / TC 0.835 / WT 0.896, mean 0.8439
  - nnU-Net 100 epochs:     ET 0.824 / TC 0.860 / WT 0.917, mean 0.8668
  - nnU-Net wins every region (+0.02-0.025) at comparable budget.
  - CAUTION: nnU-Net's own log line "Mean Validation Dice 0.777" is a
    per-LABEL mean — never compare it to region means.
  - Baseline final 3-fold: ET 0.790±0.012, TC 0.851±0.017,
    WT 0.899±0.005, overall 0.847±0.009 — all folds gate-passing.
  - nnU-Net checkpoint NOT servable by our API (different arch);
    benchmark result only, served model stays fold-2 baseline.
  - Per-case metrics: nnunet_fold0_region_metrics.json, Kaggle notebook
    nnunet-region-metrics (CPU); training in nnunet-brats-probe
    (100-epoch version, ~5.7 hrs, ~196 s/epoch measured).
- **Frontend redesign SHIPPED (decision 18)**: React Router 7 multi-
  screen per Section 6.1 — Upload / Viewer (/cases/:id, linkable,
  overlay restore on re-open) / History / About; shared Layout with
  nav + FR-6.6 banner. ONE backend change: GET /v1/cases (+ test).
  Admin screen scoped OUT (prereqs deferred per decision 7). Verified
  in browser through the compose stack: History, re-open, About,
  F5-on-/history SPA fallback.
- **SRS at v1.6** (docs/SRS_Medical_Imaging_Pipeline_v1.6.docx):
  decisions 17-18 recorded. docs/ holds v1.3-v1.6.
- **CI: 21 tests, all green** (run 57 first full-suite pass; run 58 =
  docs push). Coverage NFR still met.
- **Quota: ~9.5 GPU-hrs left this week.**

## Environment (delta from v6; rest unchanged)

- Compose stack verified today; `docker compose -f
  docker/docker-compose.yml up -d` from repo root, frontend :8080,
  API :8000. Containers survive but do NOT auto-start after
  reboot/stop — an empty `compose ps` just means bring it up.
- nginx now resolves the api upstream at REQUEST time (resolver
  127.0.0.11 + variable proxy_pass) — see gotcha 19.
- Kaggle notebooks: brats-baseline-training (fold training),
  nnunet-brats-probe (benchmark, T4), nnunet-region-metrics (CPU,
  metrics JSON in output).

## Remaining work (priority order)

1. **DECIDE: 3-fold ensemble inference** (avg probabilities across the
   three baseline fold checkpoints; expected ~+0.01-0.02 Dice; ~1 hr
   GPU to measure on fold-0 val). Do this week within remaining quota,
   or park. Either way record as Appendix E decision 19.
2. Smaller Shoulds — build or formally scope out in Appendix E:
   FR-1.4 DICOM stub, FR-7.2/7.3 drift stats + alerts, load/security
   tests (Section 11).
3. Make mypy blocking after fixing the 5 union-narrowing errors
   (decision 16 follow-up).
4. Polish (Section 12 phase 8): README (still Aug-30 scaffold),
   architecture diagram, demo write-up. Delete requirements.txt.bak-v1
   when comfortable.

## Known gotchas (carried from v6, plus three new)

1-17. All v6 gotchas still apply verbatim (Kaggle .pt re-zip; frozen
notebook inputs; interactive-session duplicate T4; per-version Logs
tab; anchored .gitignore; heredoc echo mangling — verify on disk;
anchored-Python-patch for multi-line edits; .nii not .nii.gz; failed-
push false alarm; cosine LR resume warning; CPU inference patience;
empty chat attachments; hidden dynamic deps => /tmp clone dress
rehearsal; ruff --fix silent edits; check Actions after every push;
python-docx unique anchors + disk verify + LibreOffice eyeball; never
paste terminal scrollback back into the shell).

18. **NEW — run `ruff check src/ tests/` locally BEFORE every push.**
    CI run 52 failed on a lint (RUF015) that local pytest+py_compile
    never sees. Lint locally; GitHub is not the test bench.
19. **NEW — nginx in compose must resolve upstreams at request time**
    (resolver 127.0.0.11 valid=10s; set $var; proxy_pass $var).
    Boot-time resolve crashes the frontend container whenever it wins
    the startup race against the api container. Note: with a variable
    proxy_pass, nginx stops auto-appending location paths — fine here
    because /v1 is both the location and the real API prefix.
20. **NEW — pytest does NOT collect script-style tests** (main() +
    __main__ only). 5 of 8 test files were invisible to CI for weeks;
    the stale FR-5.4 assert hid there. Fix pattern: append a
    `def test_main(): main()` wrapper. Check `pytest --collect-only`
    when adding test files.
21. **NEW — nnU-Net metric semantics**: logged Mean Validation Dice is
    per-label; BraTS-comparable ET/TC/WT must be computed from saved
    validation predictions as label unions (ET={3}, TC={1,3},
    WT={1,2,3}) — remember our conversion remapped 4->3.

## How the user works (unchanged, essential)

One literal copy-paste block at a time; plain "what and why" before
each task; explicit click-by-click for any UI; reads raw pasted output
carefully (multiple real bugs surfaced there); heredoc for file
writes; compile-check all touched .py files; verify file state on
disk. Never write against unseen code — cat the file first. Record
every SRS-silent decision in Appendix E with rationale. Ruff before
every push (gotcha 18).
