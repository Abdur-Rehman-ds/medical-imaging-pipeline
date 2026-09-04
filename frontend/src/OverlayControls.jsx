// FR-6.3 — per-label toggles + opacity. Styled per index.css.
const LABELS = [
  { value: 1, name: "NCR/NET", rgb: [230, 60, 60] },
  { value: 2, name: "Edema", rgb: [60, 200, 90] },
  { value: 4, name: "Enhancing", rgb: [240, 220, 60] },
];

function OverlayControls({ visible, opacity, onToggle, onOpacity }) {
  return (
    <div className="card" style={{ display: "flex", gap: 26, alignItems: "center",
                                   flexWrap: "wrap", padding: "12px 20px" }}>
      <b style={{ fontSize: 14 }}>Overlay</b>
      {LABELS.map((l) => (
        <label key={l.value} style={{ display: "flex", gap: 7, alignItems: "center",
                                      fontSize: 13.5, cursor: "pointer" }}>
          <input type="checkbox" checked={visible[l.value]}
                 onChange={() => onToggle(l.value)} />
          <span style={{ width: 11, height: 11, borderRadius: 3,
                         background: `rgb(${l.rgb.join(",")})`,
                         display: "inline-block" }} />
          {l.name}
        </label>
      ))}
      <label style={{ display: "flex", gap: 10, alignItems: "center",
                      fontSize: 13.5, marginLeft: "auto" }}>
        Opacity
        <input type="range" min="0" max="100" value={Math.round(opacity * 100)}
               onChange={(e) => onOpacity(Number(e.target.value) / 100)} />
        <span className="muted" style={{ width: 34 }}>{Math.round(opacity * 100)}%</span>
      </label>
    </div>
  );
}

export { LABELS };
export default OverlayControls;
