// About screen — project framing, scope, and the full Section 14
// disclaimer. Placeholder content; refined in the theme pass.
function AboutPage() {
  return (
    <div className="card" style={{ maxWidth: 720 }}>
      <h2>About this system</h2>
      <p>
        End-to-end brain tumor segmentation pipeline: multi-modal MRI
        (T1, T1ce, T2, FLAIR) → 3D U-Net sliding-window inference →
        per-region volumes (ET / TC / WT) and reviewable overlays.
      </p>
      <p className="muted">
        Trained on the BraTS2020 dataset (3-fold cross-validation).
        Serving and UI are research/portfolio infrastructure.
      </p>
      <p style={{ fontStyle: "italic" }}>
        This system is a research and educational pipeline. It is NOT a
        certified medical device, has NOT received regulatory clearance
        (e.g., FDA, CE), and MUST NOT be used for clinical diagnosis,
        treatment planning, or any decision affecting patient care.
      </p>
    </div>
  );
}

export default AboutPage;
