// FR-6.1 — modality file selection + upload to POST /v1/cases (FR-5.1).
// Reports upload state; passes the new case_id up via onUploaded.
import { useState } from "react";

const MODALITIES = ["t1", "t1ce", "t2", "flair"];

function UploadPanel({ onUploaded }) {
  const [files, setFiles] = useState({});
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");

  const setFile = (name, file) =>
    setFiles((prev) => ({ ...prev, [name]: file }));

  const allChosen = MODALITIES.every((m) => files[m]);

  async function upload() {
    setBusy(true);
    setMessage("Uploading…");
    try {
      const form = new FormData();
      MODALITIES.forEach((m) => form.append(m, files[m]));
      const res = await fetch("/v1/cases", { method: "POST", body: form });
      const data = await res.json();
      if (!res.ok) {
        setMessage(`Upload rejected: ${data.message || res.status}`);
      } else {
        setMessage(`Uploaded — case ID: ${data.case_id}`);
        onUploaded(data.case_id);
      }
    } catch (err) {
      setMessage(`Network error: ${err.message} — is the API running on :8000?`);
    } finally {
      setBusy(false);
    }
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
      <h2 style={{ margin: "0 0 12px 0", fontSize: 18 }}>Upload a case</h2>
      <div style={{ display: "flex", gap: 16, flexWrap: "wrap" }}>
        {MODALITIES.map((m) => (
          <label key={m} style={{ fontSize: 14 }}>
            <div style={{ marginBottom: 4, textTransform: "uppercase" }}>{m}</div>
            <input
              type="file"
              accept=".nii,.nii.gz,.gz"
              onChange={(e) => setFile(m, e.target.files[0])}
            />
          </label>
        ))}
      </div>
      <button
        onClick={upload}
        disabled={!allChosen || busy}
        style={{
          marginTop: 12,
          padding: "8px 20px",
          borderRadius: 6,
          border: "none",
          background: allChosen && !busy ? "#3b6ea5" : "#333",
          color: "#fff",
          cursor: allChosen && !busy ? "pointer" : "not-allowed",
        }}
      >
        {busy ? "Uploading…" : "Upload"}
      </button>
      {message && <p style={{ fontSize: 14, marginTop: 10 }}>{message}</p>}
    </div>
  );
}

export default UploadPanel;
