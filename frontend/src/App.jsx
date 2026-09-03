// App shell — FR-6.1 (upload), FR-6.2 (status), FR-6.3 (viewer+overlay),
// FR-6.4 (volumes panel), FR-6.6 (disclaimer).
import { useEffect, useRef, useState } from "react";
import { Niivue } from "@niivue/niivue";
import UploadPanel from "./UploadPanel";
import InferencePanel from "./InferencePanel";

const DISCLAIMER =
  "Research and educational use only. NOT a certified medical device. " +
  "MUST NOT be used for clinical diagnosis or treatment decisions.";

function App() {
  const canvasRef = useRef(null);
  const nvRef = useRef(null);
  const [caseId, setCaseId] = useState(null);
  const [result, setResult] = useState(null);

  // One Niivue instance for the app's lifetime; starts on the demo brain.
  useEffect(() => {
    const nv = new Niivue({ backColor: [0.05, 0.05, 0.08, 1] });
    nv.attachToCanvas(canvasRef.current);
    nv.loadVolumes([
      { url: "https://niivue.github.io/niivue-demo-images/mni152.nii.gz" },
    ]);
    nvRef.current = nv;
  }, []);

  // When a result arrives, show the case's T1ce + mask overlay (FR-6.3).
  useEffect(() => {
    if (!result || !caseId || !nvRef.current) return;
    nvRef.current.loadVolumes([
      { url: `/v1/cases/${caseId}/files/t1ce`, name: "t1ce.nii.gz" },
      {
        url: `/v1/cases/${caseId}/files/mask`,
        name: "mask.nii.gz",
        colormap: "nih",
        opacity: 0.5,
      },
    ]);
  }, [result, caseId]);

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
      <canvas ref={canvasRef} style={{ width: "100%", height: 520 }} />
      <p style={{ color: "#888", fontSize: 13 }}>
        {result
          ? "Showing uploaded T1ce with segmentation overlay (50% opacity)."
          : "Demo volume (MNI152) — upload a case and run inference to view results."}
      </p>
    </div>
  );
}

export default App;
