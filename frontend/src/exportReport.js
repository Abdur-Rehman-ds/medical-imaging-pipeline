// FR-6.5 — report export helpers (JSON now, PDF added next).
// Design decision (SRS silent): client-side export — the slice snapshot
// reflects the user's current view state, which only exists in-browser.

const DISCLAIMER =
  "Research and educational use only. NOT a certified medical device. " +
  "MUST NOT be used for clinical diagnosis or treatment decisions.";

function triggerDownload(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

export function exportJson(caseId, result) {
  const report = {
    report_type: "brats_segmentation_case_report",
    case_id: caseId,
    exported_at: new Date().toISOString(),
    disclaimer: DISCLAIMER,
    ...result,
  };
  const blob = new Blob([JSON.stringify(report, null, 2)], {
    type: "application/json",
  });
  triggerDownload(blob, `${caseId}_report.json`);
}
