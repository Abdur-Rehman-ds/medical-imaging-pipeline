"""Unit tests for src/data/preprocessing.py — Section 11 (Testing Strategy).

Naming convention: test names should reference the FR-ID they cover
(Appendix C — Requirement Traceability Summary) so coverage is traceable
back to the SRS.
"""

import pytest


@pytest.mark.skip(reason="Implements FR-2.2 — pending preprocessing.py implementation")
def test_fr_2_2_resample_preserves_label_integrity():
    """Resampling a label map must use nearest-neighbor interpolation,
    never linear/cubic — linear interpolation on integer label IDs
    produces invalid intermediate label values.
    """
    raise NotImplementedError


@pytest.mark.skip(reason="Implements FR-2.3 — pending preprocessing.py implementation")
def test_fr_2_3_zscore_normalization_after_percentile_clip():
    raise NotImplementedError
