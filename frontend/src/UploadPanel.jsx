// FR-6.1 — modality upload to POST /v1/cases (FR-5.1). Styled per index.css.
import { useState } from "react";

const MODALITIES = ["t1", "t1ce", "t2", "flair"];

function UploadPanel({ onUploaded }) {
  const [files, setFiles] = useState({});
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const setFile = (name, file) => setFiles((p) => ({ ...p, [name]: file }));
  const allChosen = MODALITIES.every((m) => files[m]);

  async function upload() {
    setBusy(true);
    setMessage("Uploading…");
    try {
      const form = new FormData();
      MODALITIES.forEach((m) => form.append(m, files[m]));
      const res = await fetch("/v1/cases", { method: "POST", body: form });
      const data = await res.json();
      if (!res.ok) setMessage(`Upload rejected: ${data.message || res.status}`);
      else {
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
    <div className="card">
      <h2>Upload a case</h2>
      <div style={{
        display: "grid",
        gridTemplateColumns: "repeat(auto-fit, minmax(210px, 1fr))",
        gap: 14,
      }}>
        {MODALITIES.map((m) => (
          <label key={m} style={{ fontSize: 13 }}>
            <div style={{ marginBottom: 5, textTransform: "uppercase",
                          letterSpacing: "0.06em", color: "var(--text-dim)" }}>
              {m}
            </div>
            <input type="file" accept=".nii,.nii.gz,.gz"
                   onChange={(e) => setFile(m, e.target.files[0])} />
          </label>
        ))}
      </div>
      <div style={{ marginTop: 16, display: "flex", gap: 14, alignItems: "center" }}>
        <button className="btn btn-primary" onClick={upload} disabled={!allChosen || busy}>
          {busy ? "Uploading…" : "Upload"}
        </button>
        {message && <span style={{ fontSize: 13.5 }}>{message}</span>}
      </div>
    </div>
  );
}

export default UploadPanel;
