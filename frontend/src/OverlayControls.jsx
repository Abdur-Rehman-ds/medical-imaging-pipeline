// FR-6.3 — per-label overlay toggles + overlay opacity control.
// BraTS canonical labels: 1=NCR/NET (red), 2=edema (green), 4=enhancing (yellow).
const LABELS = [
  { value: 1, name: "NCR/NET", rgb: [230, 60, 60] },
  { value: 2, name: "Edema", rgb: [60, 200, 90] },
  { value: 4, name: "Enhancing", rgb: [240, 220, 60] },
];

function OverlayControls({ visible, opacity, onToggle, onOpacity }) {
  return (
    <div
      style={{
        background: "#16161d",
        border: "1px solid #2a2a35",
        borderRadius: 8,
        padding: "10px 16px",
        marginBottom: 12,
        display: "flex",
        gap: 24,
        alignItems: "center",
        flexWrap: "wrap",
        fontSize: 14,
      }}
    >
      <b>Overlay:</b>
      {LABELS.map((l) => (
        <label key={l.value} style={{ display: "flex", gap: 6, alignItems: "center" }}>
          <input
            type="checkbox"
            checked={visible[l.value]}
            onChange={() => onToggle(l.value)}
          />
          <span
            style={{
              width: 12,
              height: 12,
              borderRadius: 2,
              background: `rgb(${l.rgb.join(",")})`,
              display: "inline-block",
            }}
          />
          {l.name}
        </label>
      ))}
      <label style={{ display: "flex", gap: 8, alignItems: "center" }}>
        Opacity
        <input
          type="range"
          min="0"
          max="100"
          value={Math.round(opacity * 100)}
          onChange={(e) => onOpacity(Number(e.target.value) / 100)}
        />
        {Math.round(opacity * 100)}%
      </label>
    </div>
  );
}

export { LABELS };
export default OverlayControls;
