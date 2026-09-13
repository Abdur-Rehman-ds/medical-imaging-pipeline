// Shared shell for all routes — nav + persistent FR-6.6 banner.
// Frontend redesign per SRS Section 6.1 (Appendix E decision #18).
import { NavLink, Outlet } from "react-router-dom";

const DISCLAIMER =
  "Research and educational use only. NOT a certified medical device. " +
  "MUST NOT be used for clinical diagnosis or treatment decisions.";

const navStyle = ({ isActive }) => ({
  padding: "6px 14px", borderRadius: 6, fontSize: 14, textDecoration: "none",
  color: isActive ? "var(--text)" : "var(--text-dim)",
  background: isActive ? "var(--surface-2, rgba(255,255,255,0.08))" : "transparent",
});

function Layout() {
  return (
    <>
      <header style={{ display: "flex", alignItems: "baseline", gap: 22,
                       flexWrap: "wrap", marginBottom: 6 }}>
        <h1 style={{ marginRight: 6 }}>BraTS Tumor Segmentation</h1>
        <nav style={{ display: "flex", gap: 6 }}>
          <NavLink to="/" style={navStyle} end>Upload</NavLink>
          <NavLink to="/history" style={navStyle}>History</NavLink>
          <NavLink to="/about" style={navStyle}>About</NavLink>
        </nav>
      </header>
      <div className="banner">{DISCLAIMER}</div>
      <Outlet />
    </>
  );
}

export default Layout;
