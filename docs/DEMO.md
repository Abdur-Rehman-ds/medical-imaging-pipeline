# Demo Walkthrough — Medical Imaging Pipeline

A guided tour of the running system. Doubles as the script for a recorded
demo. Prerequisites: the compose stack is up (`docker compose -f
docker/docker-compose.yml up -d` from the repo root) with a checkpoint in
`models/` — the default serves the fold-2 baseline
(`fold2_best_e099_d0.8590.pt`, validation Dice 0.859).

> Research/educational pipeline — NOT a certified medical device, NOT for
> clinical use. This disclaimer appears on every screen and in every API
> response by design (SRS Section 14, decision 9).

## 1. The API, before touching the UI

The FR-5.4 model listing shows what is actually being served:

    curl -s http://localhost:8000/v1/models

Response: all three cross-validation fold checkpoints with fold/epoch/Dice
metadata parsed from the checkpoint filenames (decision 11), the fold-2
model marked `"active": true`, and the non-clinical disclaimer embedded in
the JSON itself. Every subsequent inference response carries the same
disclaimer field.

## 2. Upload screen (FR-6.1, FR-6.2)

Open http://localhost:8080. The persistent disclaimer banner (FR-6.6) is
visible on every route. On the Upload screen, select the four BraTS
modality files for a case — T1, T1ce, T2, FLAIR (`.nii`). The backend
validates completeness and geometry before accepting (FR-1.2): try
uploading only three modalities and it rejects with a specific structured
error rather than a silent failure (FR-1.3, FR-5.7). On success the case
gets an ID and the raw upload is persisted before any processing (FR-1.5).

## 3. Run inference and watch status (FR-6.2, FR-4.x)

Trigger inference. Status states are shown as the case moves through
preprocessing (RAS reorientation, 1 mm³ resampling, percentile-clipped
z-score normalization — FR-2.1..2.3) and sliding-window inference
(Gaussian-blended overlapping patches — FR-4.1, FR-4.2), then
post-processing (small-component removal, label remap to the four BraTS
labels — FR-4.3, FR-4.4). On CPU this takes minutes, not seconds — the
serving image is CPU-only by design (decision 15); the NFR budget for
CPU fallback is <= 4 minutes.

## 4. Viewer screen (FR-6.3, FR-6.4)

When inference completes, the Viewer (/cases/:id — linkable URL,
decision 18) renders the axial/coronal/sagittal slice viewer with a
scroll/slider control. Toggle each label overlay independently and adjust
opacity; alongside the viewer sit the per-region volumes (ET, TC, WT in
cm³) and the active model version badge — the same fold-2 version the API
reported in step 1 (FR-6.4, auditability NFR).

## 5. Report export (FR-6.5)

Export the case as PDF or JSON. The export is client-side (jsPDF,
decision 10) so the slice-overlay snapshot in the PDF matches exactly
what is on screen — slice position, label toggles, opacity. The JSON is
the backend inference summary verbatim, disclaimer field included.

## 6. History and re-open (Section 6.1, decision 18)

The History screen lists processed cases (status, timestamp, model
version, newest-first — served by GET /v1/cases). Click a completed case:
the Viewer restores its overlay without re-running inference. Refresh the
browser on /history to show SPA-fallback routing surviving a hard reload.

## 7. The same flow, API-only (FR-5.1..5.3)

Everything the UI does is three endpoints:

    POST /v1/cases                 upload, returns case ID
    POST /v1/cases/{id}/infer      trigger inference
    GET  /v1/cases/{id}/result     status + results

Structured errors with correlation IDs on every failure path (FR-5.7);
optional API-key auth and per-client rate limiting when API_KEY is set
(decision 12).

## What this demo deliberately does not show

- Clinical use of any kind — see the disclaimer above.
- The nnU-Net benchmark model: it exceeds the baseline on the same split
  (decision 17) but is a benchmark artifact, not servable by this API.
- Ensemble inference: scoped out (decision 19) because no leakage-free
  evaluation of it exists under the 3-fold structure — the honest option
  was not to claim a number.
