// FR-6.5 — report export helpers (JSON + PDF).
// Design decision (SRS silent): client-side export — the slice snapshot
// reflects the user's current view state, which only exists in-browser.
import { jsPDF } from "jspdf";

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
    ...result,
  };
  const blob = new Blob([JSON.stringify(report, null, 2)], {
    type: "application/json",
  });
  triggerDownload(blob, `${caseId}_report.json`);
}

// snapshotDataUrl: PNG data URL of the viewer canvas, captured by the
// caller right after a forced redraw (WebGL buffers are cleared between
// frames, so capture must happen in the same frame as a draw).
export function exportPdf(caseId, result, snapshotDataUrl) {
  const doc = new jsPDF({ unit: "mm", format: "a4" });
  const W = doc.internal.pageSize.getWidth();
  const margin = 16;
  let y = 20;

  doc.setFont("helvetica", "bold").setFontSize(16);
  doc.text("BraTS Segmentation Case Report", margin, y);
  y += 9;

  doc.setFont("helvetica", "normal").setFontSize(10).setTextColor(90);
  doc.text(`Case: ${caseId}`, margin, y);
  doc.text(`Exported: ${new Date().toISOString()}`, W - margin, y, { align: "right" });
  y += 6;
  doc.text(`Model version: ${result.model_version ?? "n/a"}`, margin, y);
  y += 9;

  doc.setTextColor(0).setFont("helvetica", "bold").setFontSize(11);
  doc.text("Per-region results", margin, y);
  y += 6;

  doc.setFont("helvetica", "normal").setFontSize(10);
  const vols = result.per_label_volumes_mm3 || {};
  const conf = result.confidence_summary || {};
  doc.text("Region", margin, y);
  doc.text("Volume (cm³)", margin + 60, y);
  doc.text("Mean confidence", margin + 110, y);
  y += 2;
  doc.line(margin, y, W - margin, y);
  y += 5;
  for (const [label, mm3] of Object.entries(vols)) {
    doc.text(String(label), margin, y);
    doc.text((mm3 / 1000).toFixed(2), margin + 60, y);
    doc.text(conf[label] != null ? String(conf[label]) : "n/a", margin + 110, y);
    y += 6;
  }
  y += 4;

  if (snapshotDataUrl) {
    doc.setFont("helvetica", "bold").setFontSize(11);
    doc.text("Viewer snapshot (current view state)", margin, y);
    y += 5;
    const imgW = W - 2 * margin;
    const imgH = imgW * 0.5; // viewer canvas is wide; keep aspect ~2:1
    doc.addImage(snapshotDataUrl, "PNG", margin, y, imgW, imgH);
    y += imgH + 8;
  }

  const disclaimer = result.disclaimer ||
    "Research and educational use only. NOT a certified medical device. " +
    "MUST NOT be used for clinical diagnosis or treatment decisions.";
  doc.setFont("helvetica", "italic").setFontSize(9).setTextColor(120);
  doc.text(doc.splitTextToSize(`Disclaimer: ${disclaimer}`, W - 2 * margin), margin, y);

  doc.save(`${caseId}_report.pdf`);
}
