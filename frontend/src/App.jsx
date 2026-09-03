// App shell — FR-6.1 (upload), FR-6.2 (status), FR-6.3 (viewer, per-label
// toggles, opacity), FR-6.4 (volumes), FR-6.6 (disclaimer).
import { useEffect, useRef, useState } from "react";
import { Niivue } from "@niivue/niivue";
import UploadPanel from "./UploadPanel";
import InferencePanel from "./InferencePanel";
import OverlayControls, { LABELS } from "./OverlayControls";

const DISCLAIMER =
  "Research and educational use only. NOT a certified medical device. " +
  "MUST NOT be used for clinical diagnosis or treatment decisions.";

// Build a NiiVue label colormap for BraTS labels {0,1,2,4}; hidden
// labels get alpha 0. Index space must cover 0..4.
function buildLabelColormap(visible, opacity) {
  const a = Math.round(opacity * 255);
  const R = [0, 0, 0, 0, 0];
  const G = [0, 0, 0, 0, 0];
  const B = [0, 0, 0, 0, 0];
  const A = [0, 0, 0, 0, 0];
  for (const l of LABELS) {
    R[l.value] = l.rgb[0];
    G[l.value] = l.rgb[1];
    B[l.value] = l.rgb[2];
    A[l.value] = visible[l.value] ? a : 0;
  }
  return { R, G, B, A, labels: ["bg", "NCR/NET", "edema", "", "enhancing"] };
}

function App() {
  const canvasRef = useRef(null);
  const nvRef = useRef(null);
  const [caseId, setCaseId] = useState(null);
  const [result, setResult] = useState(null);
  const [visible, setVisible] = useState({ 1: true, 2: true, 4: true });
  const [opacity, setOpacity] = useState(0.6);

  useEffect(() => {
    const nv = new Niivue({ backColor: [0.05, 0.05, 0.08, 1] });
    nv.attachToCanvas(canvasRef.current);
    nv.loadVolumes([
      { url: "https://niivue.github.io/niivue-demo-images/mni152.nii.gz" },
    ]);
    nvRef.current = nv;
  }, []);

  // Load case volumes when a result arrives.
  useEffect(() => {
    if (!result || !caseId || !nvRef.current) return;
    const nv = nvRef.current;
    async function loadCase() {
      await nv.loadVolumes([
        { url: `/v1/cases/${caseId}/files/t1ce`, name: "t1ce.nii.gz" },
        { url: `/v1/cases/${caseId}/files/mask`, name: "mask.nii.gz", opacity },
      ]);
      applyOverlayStyle();
    }
    loadCase();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [result, caseId]);

  // Re-style the mask layer when toggles/opacity change.
  function applyOverlayStyle(nextVisible = visible, nextOpacity = opacity) {
    const nv = nvRef.current;
    if (!nv || nv.volumes.length < 2) return;
    const mask = nv.volumes[1];
    mask.setColormapLabel(buildLabelColormap(nextVisible, 1.0));
    nv.setOpacity(1, nextOpacity);
    nv.updateGLVolume();
  }

  function handleToggle(labelValue) {
    const next = { ...visible, [labelValue]: !visible[labelValue] };
    setVisible(next);
    applyOverlayStyle(next, opacity);
  }

  function handleOpacity(value) {
    setOpacity(value);
    applyOverlayStyle(visible, value);
  }

  return (
    <div style={{ fontFamily: "system-ui, sans-serif", padding: 16 }}>
      <h1 style={{ margin: "0 0 4px 0" }}>BraTS Tumor Segmentation Viewer</h1>
      <div
        style={{
          background: "#7a1f1f",
          color: "#fff",
          padding: "8px 12px",
          borderRadius: 6,
          margin: "8px 0 16px 0",
          fontSize: 14,
        }}
      >
        {DISCLAIMER}
      </div>
      <UploadPanel onUploaded={setCaseId} />
      {caseId && <InferencePanel caseId={caseId} onResult={setResult} />}
      {result && (
        <OverlayControls
          visible={visible}
          opacity={opacity}
          onToggle={handleToggle}
          onOpacity={handleOpacity}
        />
      )}
      <canvas ref={canvasRef} style={{ width: "100%", height: 520 }} />
      <p style={{ color: "#888", fontSize: 13 }}>
        {result
          ? "Uploaded T1ce with segmentation overlay — toggle labels or adjust opacity above."
          : "Demo volume (MNI152) — upload a case and run inference to view results."}
      </p>
    </div>
  );
}

export default App;
