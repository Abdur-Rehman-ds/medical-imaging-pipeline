# Project Handoff v9 — Medical Imaging Pipeline (BraTS) — CLOSED

**Date: 20 September 2026.** Replaces v8. This is the CLOSING handoff:
every requirement in SRS v1.11 is implemented or formally scoped out.
For any future conversation: attach this plus SRS v1.11 (docs/, source
of truth, Appendix E decisions 1-23).

Repo: https://github.com/Abdur-Rehman-ds/medical-imaging-pipeline
Final commit: `35a007b`. CI green through run 73; 35 tests; mypy
blocking. Untracked locals: requirements.txt.bak-v1 (deletable).

## Final state

- **All Musts and Shoulds resolved.** The last three Shoulds closed
  20-Sep: FR-1.4 DICOM stub BUILT (decision 21 — dcm2niix wrapper,
  synthetic-series test, binary in image+CI); FR-7.1/7.2/7.3
  monitoring BUILT (decision 22 — FR-7.1 was discovered to be
  unimplemented scaffold despite being a Must; per-request JSON event
  logging, drift stats on preprocessed volumes, threshold alert;
  foreground bug caught LIVE and fixed); Section 11 security/load
  tests BUILT (decision 23 — upload size cap added, a real gap;
  malformed-file + concurrency tests; full-scale load test scoped out
  to deployment environments). FR-7.4 closed as manual documented
  operation per decisions 7/15.
- Training: 3-fold baseline 0.847±0.009 (all folds gate-passing),
  nnU-Net benchmark 0.8668 same-split (decision 17), ensemble scoped
  out (decision 19). Zero GPU debt; nothing will ever need quota
  unless a NEW decision reopens something.
- Served model: fold-2 baseline (fold2_best_e099_d0.8590.pt), compose
  default. Drift monitoring live in the API; drift_score is null
  (reason no-reference) until the optional follow-up below.
- SRS at v1.11; README/diagram/DEMO.md/LICENSE all current.

## Optional follow-ups (none block closure)

1. Reference stats: run scripts/make_reference_stats.py in a Kaggle
   CPU session against BraTS2020, commit the JSON to
   configs/monitoring/ — turns drift_score from null to real. No GPU.
2. Demo video: docs/DEMO.md is the script; stack via
   docker compose -f docker/docker-compose.yml up -d.
3. Delete requirements.txt.bak-v1.
4. Future-work register (all recorded in Appendix E): CUDA serving
   image, nnU-Net export/serving path, ensemble on a true held-out
   set, model registry + Admin screen (decision 7 revisit),
   p95-latency alerting (needs metrics stack), FR-7.4 automation.

## Known gotchas (carry to any future work here — or any project)

All 22 from v8 still apply. The ones that earned their keep on the
final day: mangled terminal echo lies, disk is truth; a Python
heredoc that dies mid-parse (SyntaxError) writes NOTHING — grep the
file to confirm before re-patching; ruff --fix then git diff to see
what it changed; assert-anchored python patches fail loudly instead
of mispatching; read live output numbers critically (fg_fraction 1.0
exposed a real stats bug the unit tests missed); "Queued" in GitHub
Actions is normal for a minute or two, not a failure.

## How the user works (for any future session)

One literal copy-paste block at a time; plain what-and-why before
each task; explicit click-by-click for UI; reads raw pasted output
carefully; heredoc for file writes, short blocks, no inner
triple-backtick fences (gotcha 22); compile-check touched .py files;
verify state on disk; never write against unseen code; record every
SRS-silent decision in Appendix E with rationale; ruff + mypy
(BLOCKING) before every push; LibreOffice eyeball after every docx
edit, check for stale .~lock files with pgrep first.

## Closing note

Started 10-Aug-2026 as an SRS draft; closed 20-Sep-2026 with every
requirement resolved. 23 recorded decisions, 35 tests, 73 CI runs,
~6 weeks, free-tier GPU only. The discipline (verify on disk, record
every decision, read the output) caught real bugs to the last day.
