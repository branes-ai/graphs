"""Tests for micro-architectural reporting schema and HTML template."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from graphs.reporting.microarch_html_template import (
    _safe_status_class,
    render_sku_page,
)
from graphs.reporting.microarch_schema import (
    LayerPanel,
    MicroarchReport,
    empty_report,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


class TestSafeStatusClass:
    def test_passes_known_statuses(self):
        for s in ("calibrated", "interpolated", "theoretical",
                  "not_populated", "unknown"):
            assert _safe_status_class(s) == s

    def test_normalizes_case_and_whitespace(self):
        assert _safe_status_class("  Theoretical  ") == "theoretical"
        assert _safe_status_class("CALIBRATED") == "calibrated"

    def test_coerces_unknown_to_unknown(self):
        assert _safe_status_class("invalid-status") == "unknown"
        assert _safe_status_class("") == "unknown"
        assert _safe_status_class(None) == "unknown"

    def test_blocks_attribute_injection(self):
        """A crafted status with quotes must not escape the class attribute."""
        hostile = 'x"><script>alert(1)</script>'
        assert _safe_status_class(hostile) == "unknown"


class TestSKURenderSanitization:
    def test_injection_status_does_not_escape_class_attribute(self):
        panel = LayerPanel(
            layer=empty_report("t").layers[0].layer,
            title="Layer 1: ALU",
            status='x"><script>alert(1)</script>',
            summary="",
        )
        rpt = MicroarchReport(sku="t", display_name="t", layers=[panel])
        html_out = render_sku_page(rpt, REPO_ROOT)
        # The crafted class token never appears; it collapses to "unknown".
        assert 'class="badge unknown"' in html_out
        # The crafted script tag must have been HTML-escaped in the display.
        assert "<script>alert(1)</script>" not in html_out
        assert "&lt;script&gt;" in html_out


class TestEmptyReportTimestamp:
    def test_generated_at_is_timezone_aware_utc(self):
        rpt = empty_report(sku="x")
        parsed = datetime.fromisoformat(rpt.generated_at)
        assert parsed.tzinfo is not None, (
            "generated_at must be timezone-aware; got naive datetime")
        # UTC offset is zero.
        assert parsed.utcoffset().total_seconds() == 0
