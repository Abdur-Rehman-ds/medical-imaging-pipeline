// App shell — FR-6.1/6.2/6.3/6.4/6.6, snapshot provider for FR-6.5.
// Styled per design system (index.css).
import { useEffect, useRef, useState } from "react";
import { Niivue } from "@niivue/niivue";
import UploadPanel from "./UploadPanel";
import InferencePanel from "./InferencePanel";
import OverlayControls, { LABELS } from "./OverlayControls";

const DISCLAIMER =
  "Research and educational use only. NOT a certified medical device. " +
  "MUST NOT be used for clinical diagnosis or treatment decisions.";

function buildLabelColormap(visible, opacity) {
  const a = Math.round(opacity * 255);
  const R = [0, 0, 0, 0, 0], G = [0, 0, 0, 0, 0], B = [0, 0, 0, 0, 0], A = [0, 0, 0, 0, 0];
  for (const l of LABELS) {
    R[l.value] = l.rgb[0]; G[l.value] = l.rgb[1]; B[l.value] = l.rgb[2];
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
    const nv = new Niivue({ backColor: [0.043, 0.055, 0.078, 1] });
    nv.attachToCanvas(canvasRef.current);
    nv.loadVolumes([
      { url: "https://niivue.github.io/niivue-demo-images/mni152.nii.gz" },
    ]);
    nvRef.current = nv;
  }, []);

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

  function applyOverlayStyle(nextVisible = visible, nextOpacity = opacity) {
    const nv = nvRef.current;
    if (!nv || nv.volumes.length < 2) return;
    nv.volumes[1].setColormapLabel(buildLabelColormap(nextVisible, 1.0));
    nv.setOpacity(1, nextOpacity);
    nv.updateGLVolume();
  }

  function handleToggle(v) {
    const next = { ...visible, [v]: !visible[v] };
    setVisible(next);
    applyOverlayStyle(next, opacity);
  }
  function handleOpacity(v) {
    setOpacity(v);
    applyOverlayStyle(visible, v);
  }

  // FR-6.5 — capture the viewer as a PNG data URL. WebGL clears its
  // drawing buffer between frames, so force a redraw immediately before
  // toDataURL so the capture happens in the same frame as a draw.
  function getSnapshot() {
    const nv = nvRef.current;
    if (!nv || !canvasRef.current) return null;
    nv.drawScene();
    return canvasRef.current.toDataURL("image/png");
  }

  return (
    <>
      <header style={{ marginBottom: 6 }}>
        <h1>BraTS Tumor Segmentation</h1>
        <p className="muted" style={{ margin: "4px 0 0 0" }}>
          Multi-modal MRI upload · 3D U-Net inference · interactive overlay review
        </p>
      </header>
      <div className="banner">{DISCLAIMER}</div>
      <UploadPanel onUploaded={setCaseId} />
      {caseId && (
        <InferencePanel caseId={caseId} onResult={setResult} getSnapshot={getSnapshot} />
      )}
      {result && (
        <OverlayControls
          visible={visible} opacity={opacity}
          onToggle={handleToggle} onOpacity={handleOpacity}
        />
      )}
      <canvas ref={canvasRef} className="viewer" />
      <p className="muted" style={{ marginTop: 10 }}>
        {result
          ? "Uploaded T1ce with segmentation overlay — use the controls above."
          : "Demo volume (MNI152) — upload a case and run inference to view results."}
      </p>
    </>
  );
}

export default App;
