// Frontend skeleton — FR-6.3 (slice viewer), FR-6.6 (disclaimer banner).
// Loads a public demo brain (MNI152) to verify NiiVue rendering;
// will be wired to our FastAPI backend (FR-5.1..5.3) next.
import { useEffect, useRef } from "react";
import { Niivue } from "@niivue/niivue";

const DISCLAIMER =
  "Research and educational use only. NOT a certified medical device. " +
  "MUST NOT be used for clinical diagnosis or treatment decisions.";

function App() {
  const canvasRef = useRef(null);

  useEffect(() => {
    const nv = new Niivue({ backColor: [0.05, 0.05, 0.08, 1] });
    nv.attachToCanvas(canvasRef.current);
    nv.loadVolumes([
      { url: "https://niivue.github.io/niivue-demo-images/mni152.nii.gz" },
    ]);
  }, []);

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
      <canvas ref={canvasRef} style={{ width: "100%", height: 520 }} />
      <p style={{ color: "#888", fontSize: 13 }}>
        Demo volume (MNI152) — drag to rotate/scroll slices. Backend wiring
        comes next.
      </p>
    </div>
  );
}

export default App;
