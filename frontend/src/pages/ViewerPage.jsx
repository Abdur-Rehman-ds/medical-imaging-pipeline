// Viewer screen (Section 6.1) — FR-6.2/6.3/6.4, snapshot for FR-6.5.
// Holds the Niivue canvas + overlay state (moved from the old App.jsx).
// On mount, checks the case's result so completed cases re-opened from
// History (or a page refresh) restore their overlay without re-running.
import { useEffect, useRef, useState } from "react";
import { useParams } from "react-router-dom";
import { Niivue } from "@niivue/niivue";
import InferencePanel from "../InferencePanel";
import OverlayControls, { LABELS } from "../OverlayControls";

function buildLabelColormap(visible, opacity) {
  const a = Math.round(opacity * 255);
  const R = [0, 0, 0, 0, 0], G = [0, 0, 0, 0, 0], B = [0, 0, 0, 0, 0], A = [0, 0, 0, 0, 0];
  for (const l of LABELS) {
    R[l.value] = l.rgb[0]; G[l.value] = l.rgb[1]; B[l.value] = l.rgb[2];
    A[l.value] = visible[l.value] ? a : 0;
  }
  return { R, G, B, A, labels: ["bg", "NCR/NET", "edema", "", "enhancing"] };
}

function ViewerPage() {
  const { caseId } = useParams();
  const canvasRef = useRef(null);
  const nvRef = useRef(null);
  const [result, setResult] = useState(null);

  const [visible, setVisible] = useState({ 1: true, 2: true, 4: true });
  const [opacity, setOpacity] = useState(0.6);

  useEffect(() => {
    const nv = new Niivue({ backColor: [0.043, 0.055, 0.078, 1] });
    nv.attachToCanvas(canvasRef.current);
    nvRef.current = nv;
  }, []);

  // Restore a completed case on mount / case change (quick re-open).
  useEffect(() => {
    setResult(null);
    let cancelled = false;
    fetch(`/v1/cases/${caseId}/result`)
      .then((r) => r.json())
      .then((data) => {
        if (!cancelled && data.status === "completed") {
          setResult(data.summary || data);
        }
      })
      .catch(() => {});
    return () => { cancelled = true; };
  }, [caseId]);

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

  // FR-6.5 — capture in the same frame as a forced redraw (WebGL
  // clears its drawing buffer between frames).
  function getSnapshot() {
    const nv = nvRef.current;
    if (!nv || !canvasRef.current) return null;
    nv.drawScene();
    return canvasRef.current.toDataURL("image/png");
  }

  return (
    <>
      <InferencePanel caseId={caseId} onResult={setResult} getSnapshot={getSnapshot} />
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
          : "No results yet for this case — click Run Inference above."}
      </p>
    </>
  );
}

export default ViewerPage;
