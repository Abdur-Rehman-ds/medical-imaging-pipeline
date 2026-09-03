// FR-6.2 (status states) + FR-6.4 (volumes, model version).
// Triggers POST /v1/cases/{id}/infer, polls GET /v1/cases/{id}/result.
import { useEffect, useRef, useState } from "react";

function InferencePanel({ caseId, onResult }) {
  const [status, setStatus] = useState("idle");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const timerRef = useRef(null);

  // Reset when a new case is uploaded.
  useEffect(() => {
    setStatus("idle");
    setResult(null);
    setError("");
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
    } catch {
      /* transient network error — keep polling */
    }
  }

  async function run() {
    setError("");
    const res = await fetch(`/v1/cases/${caseId}/infer`, { method: "POST" });
    const data = await res.json();
    if (!res.ok) {
      setError(data.message || `Error ${res.status}`);
      return;
    }
    setStatus("processing");
    timerRef.current = setInterval(poll, 3000);
  }

  return (
    <div
      style={{
        background: "#16161d",
        border: "1px solid #2a2a35",
        borderRadius: 8,
        padding: 16,
        marginBottom: 16,
      }}
    >
      <h2 style={{ margin: "0 0 12px 0", fontSize: 18 }}>
        Inference — {caseId}
      </h2>
      <button
        onClick={run}
        disabled={status === "processing"}
        style={{
          padding: "8px 20px",
          borderRadius: 6,
          border: "none",
          background: status === "processing" ? "#333" : "#3b8a5a",
          color: "#fff",
          cursor: status === "processing" ? "not-allowed" : "pointer",
        }}
      >
        {status === "processing" ? "Processing…" : "Run Inference"}
      </button>
      {status === "processing" && (
        <p style={{ fontSize: 14 }}>Running segmentation — this page checks
        for the result every 3 seconds.</p>
      )}
      {error && <p style={{ color: "#e08080", fontSize: 14 }}>{error}</p>}
      {result && (
        <div style={{ fontSize: 14, marginTop: 10 }}>
          <b>Per-region volumes (cm³):</b>
          <ul>
            {Object.entries(result.per_label_volumes_mm3 || {}).map(
              ([label, mm3]) => (
                <li key={label}>
                  {label}: {(mm3 / 1000).toFixed(2)}
                </li>
              )
            )}
          </ul>
          <p>Model version: {result.model_version}</p>
        </div>
      )}
    </div>
  );
}

export default InferencePanel;
