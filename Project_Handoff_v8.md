# Project Handoff v8 — Medical Imaging Pipeline (BraTS)

**Date: 17 September 2026.** Replaces v7 entirely. For a new Claude
conversation: attach this plus SRS v1.8 (docs/, source of truth,
Appendix E decisions 1-20).

Repo: https://github.com/Abdur-Rehman-ds/medical-imaging-pipeline
Latest commit: `8bdce5d`. CI green through run 66; 21 tests.
Untracked locals: requirements.txt.bak-v1 (deletable safety copy).

## Headline state

- **Item 4 (polish) COMPLETE (17-Sep)**: README rewritten (3-fold
  results 0.847±0.009, same-split nnU-Net comparison, run
  instructions), Mermaid architecture diagram rendering on GitHub,
  docs/DEMO.md demo walkthrough (verified against the live compose
  stack, doubles as video script), LICENSE copyright filled
  (Abdur Rehman).
- **Item 3 (mypy) COMPLETE (17-Sep, decision 20, SRS v1.8)**: the 5
  union-narrowing errors fixed via typing.cast ONLY (zero runtime
  change — nib.load -> SpatialImage in ingestion.py; inferer and
  DiceMetric.aggregate() -> Tensor in train.py / sliding_window.py).
  CI mypy step now BLOCKING (continue-on-error removed). mypy:
  Success, 12 files.
- Training remains COMPLETE, nothing queued: 3-fold baseline
  (0.847±0.009, all folds gate-passing), nnU-Net benchmark done
  (decision 17), ensemble scoped out (decision 19). All remaining
  work is zero-GPU.
- **SRS at v1.8** (docs/SRS_Medical_Imaging_Pipeline_v1.8.docx):
  decision 20 recorded; title page updated to 1.8 (do not trust a
  docx title page blindly — v1.8's was initially left at 1.7 and
  caught only by eyeball).
- Served model unchanged: fold-2 baseline
  (fold2_best_e099_d0.8590.pt), compose default, verified live via
  GET /v1/models (all 3 checkpoints listed, fold 2 active,
  disclaimer in JSON).
- Quota note: weekly figure in any old doc is stale; check Kaggle
  directly. Zero GPU spent since 13-Sep.

## Environment (delta from v7; rest unchanged)

- Compose stack verified again 17-Sep (fold-2 serving, frontend 200).
  Still does NOT auto-start after reboot — empty `compose ps` just
  means bring it up.
- The docs/ LibreOffice .~lock files from v7 were STALE (crash
  leftovers) — deleted 17-Sep. If locks reappear, check
  `pgrep -a soffice` before believing them.

## Remaining work (priority order)

1. **The three Shoulds** — each gets built OR formally scoped out as
   an Appendix E decision (the ensemble/Admin precedent):
   a. FR-1.4 DICOM import stub (dcm2niix conversion path)
   b. FR-7.2/7.3 input-drift statistics + threshold alerts
   c. Load + security tests (Section 11: concurrency, auth
      enforcement, malformed-file handling)
2. Optional polish: record the demo video (docs/DEMO.md is the
   script); delete requirements.txt.bak-v1 when comfortable.
3. After the Shoulds are resolved either way, the project is
   CLOSEABLE — nothing else in SRS v1.8 is open.

## Known gotchas (v7's 21 carry verbatim, plus one new)

1-21. All v7 gotchas still apply. Highlights that fired again this
week: mangled terminal echo lies, disk is truth (tail/grep the file);
ruff locally before every push (caught an import-sort error CI would
have failed); pytest script-style collection; python-docx unique
anchors + disk verify + LibreOffice eyeball (caught a stale 1.7 title
page on v1.8).

22. **NEW — chat copy button truncates at inner triple-backtick
    fences.** Any heredoc containing a ``` fence gets cut at that
    fence when copied from chat — bash then hangs at the heredoc `>`
    prompt waiting for EOF. Fix: Ctrl+C, verify the file was NOT
    written (heredoc runs all-or-nothing), re-send with inner code
    blocks as 4-space-indented markdown instead of fenced, and keep
    heredoc blocks short (~40 lines). Bit three times on 17-Sep.

## How the user works (unchanged, essential)

One literal copy-paste block at a time; plain "what and why" before
each task; explicit click-by-click for any UI; reads raw pasted output
carefully (multiple real bugs surfaced there); heredoc for file
writes; compile-check all touched .py files; verify file state on
disk. Never write against unseen code — cat the file first. Record
every SRS-silent decision in Appendix E with rationale. Ruff before
every push. mypy is now BLOCKING in CI — run
`.venv/bin/mypy src/ --ignore-missing-imports` locally alongside ruff
before pushing any .py change.
