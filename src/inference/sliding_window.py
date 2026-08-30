"""Full-volume inference.

Implements:
  FR-4.1 — sliding-window inference, configurable patch size / overlap
  FR-4.2 — Gaussian-weighted blending of overlapping patches
  FR-4.3 — post-process: remove small connected components, fill holes
  FR-4.4 — re-map to 4 canonical BraTS labels, compute per-label
           voxel/volume counts (mm^3)
  FR-4.5 — return mask (NIfTI) + summary JSON + optional overlay preview

Use monai.inferers.SlidingWindowInferer for FR-4.1/4.2 rather than a
hand-rolled tiler — it already implements Gaussian blending correctly.

NFR Performance (Section 5): full-volume inference must complete in
<=30s on RTX3090/4080-class GPU, <=4min CPU fallback, for a standard
240x240x155 volume. Benchmark this once postprocess.py is implemented —
patch_size/overlap in configs/inference/default.yaml directly trade
accuracy for latency here.
"""


def run_inference(volume, model, cfg) -> dict:
    """FR-4.1, FR-4.2. Returns raw logits/probabilities over the full volume."""
    raise NotImplementedError


def postprocess(raw_prediction, cfg) -> dict:
    """FR-4.3, FR-4.4. Returns dict with keys: mask (labeled volume),
    per_label_volumes_mm3, confidence_summary.
    """
    raise NotImplementedError
