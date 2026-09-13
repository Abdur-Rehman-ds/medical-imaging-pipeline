// Upload screen (Section 6.1) — FR-6.1/6.2. On success, routes to the
// case's viewer so every case has a linkable URL.
import { useNavigate } from "react-router-dom";
import UploadPanel from "../UploadPanel";

function UploadPage() {
  const navigate = useNavigate();
  return (
    <>
      <p className="muted" style={{ margin: "2px 0 12px 0" }}>
        Upload the four MRI modalities of a case, then run inference in the viewer.
      </p>
      <UploadPanel onUploaded={(id) => navigate(`/cases/${id}`)} />
    </>
  );
}

export default UploadPage;
