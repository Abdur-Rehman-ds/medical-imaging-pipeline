// History screen (Section 6.1) — previously processed cases with
// status, timestamp, quick re-open. Backed by GET /v1/cases.
import { useEffect, useState } from "react";
import { Link } from "react-router-dom";

function HistoryPage() {
  const [cases, setCases] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    fetch("/v1/cases")
      .then((r) => r.json())
      .then((data) => setCases(data.cases || []))
      .catch((e) => setError(`Could not load history: ${e.message}`));
  }, []);

  if (error) return <div className="card">{error}</div>;
  if (cases === null) return <div className="card muted pulse">Loading history…</div>;
  if (cases.length === 0)
    return (
      <div className="card">
        No cases yet — <Link to="/">upload one</Link> to get started.
      </div>
    );

  return (
    <div className="card" style={{ padding: 0 }}>
      <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 14 }}>
        <thead>
          <tr className="muted" style={{ textAlign: "left", fontSize: 12,
                textTransform: "uppercase", letterSpacing: "0.06em" }}>
            <th style={{ padding: "12px 20px" }}>Case</th>
            <th style={{ padding: "12px 8px" }}>Status</th>
            <th style={{ padding: "12px 8px" }}>Model</th>
            <th style={{ padding: "12px 8px" }}>Updated (UTC)</th>
            <th style={{ padding: "12px 20px" }}></th>
          </tr>
        </thead>
        <tbody>
          {cases.map((c) => (
            <tr key={c.case_id} style={{ borderTop: "1px solid rgba(255,255,255,0.07)" }}>
              <td style={{ padding: "10px 20px", fontFamily: "monospace" }}>{c.case_id}</td>
              <td style={{ padding: "10px 8px" }}>{c.status}</td>
              <td style={{ padding: "10px 8px" }}>{c.model_version || "—"}</td>
              <td style={{ padding: "10px 8px" }} className="muted">
                {c.updated_at?.replace("T", " ").slice(0, 19)}
              </td>
              <td style={{ padding: "10px 20px", textAlign: "right" }}>
                <Link to={`/cases/${c.case_id}`}>Open</Link>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default HistoryPage;
