// FR-6.2 + FR-6.4 — trigger, poll, results. Styled per index.css.
import { useEffect, useRef, useState } from "react";

function InferencePanel({ caseId, onResult }) {
  const [status, setStatus] = useState("idle");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const timerRef = useRef(null);

  useEffect(() => {
    setStatus("idle"); setResult(null); setError("");
    return () => clearInterval(timerRef.current);
  }, [caseId]);

  async function poll() {
    try {
      const res = await fetch(`/v1/cases/${caseId}/result`);
      const data = await res.json();
      if (data.status === "completed") {
        clearInterval(timerRef.current);
        setStatus("done");
        setResult(data.summary || data);
        onResult(data.summary || data);
      } else if (data.status === "failed") {
        clearInterval(timerRef.current);
        setStatus("failed");
        setError(data.error || data.message || "Inference failed");
      }
    } catch { /* transient — keep polling */ }
  }

  async function run() {
    setError("");
    const res = await fetch(`/v1/cases/${caseId}/infer`, { method: "POST" });
    const data = await res.json();
    if (!res.ok) { setError(data.message || `Error ${res.status}`); return; }
    setStatus("processing");
    timerRef.current = setInterval(poll, 3000);
  }

  return (
    <div className="card">
      <h2>Inference <span className="muted">— {caseId}</span></h2>
      <div style={{ display: "flex", gap: 14, alignItems: "center", flexWrap: "wrap" }}>
        <button className="btn btn-success" onClick={run} disabled={status === "processing"}>
          {status === "processing" ? "Processing…" : "Run Inference"}
        </button>
        {status === "processing" && (
          <span className="muted">Running segmentation — checking every 3 s…</span>
        )}
        {error && <span style={{ color: "#e08080", fontSize: 13.5 }}>{error}</span>}
      </div>
      {result && (
        <div style={{ marginTop: 16, display: "flex", gap: 28, flexWrap: "wrap" }}>
          {Object.entries(result.per_label_volumes_mm3 || {}).map(([label, mm3]) => (
            <div key={label}>
              <div className="muted" style={{ textTransform: "uppercase",
                    letterSpacing: "0.06em", fontSize: 11.5 }}>{label}</div>
              <div style={{ fontSize: 22, fontWeight: 650 }}>
                {(mm3 / 1000).toFixed(2)}
                <span className="muted" style={{ fontSize: 13 }}> cm³</span>
              </div>
            </div>
          ))}
          <div>
            <div className="muted" style={{ textTransform: "uppercase",
                  letterSpacing: "0.06em", fontSize: 11.5 }}>Model</div>
            <div style={{ fontSize: 22, fontWeight: 650 }}>{result.model_version}</div>
          </div>
        </div>
      )}
    </div>
  );
}

export default InferencePanel;
