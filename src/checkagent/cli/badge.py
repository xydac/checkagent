"""Safety badge SVG generator.

Generates shields.io-style SVG badges from scan results.
Badges can be embedded in README files to signal agent safety testing.

Usage::

    checkagent scan my_agent:run --badge badge.svg
"""

from __future__ import annotations

from pathlib import Path

# Badge color thresholds (pass rate)
_GREEN_THRESHOLD = 0.9   # >= 90% pass → green
_YELLOW_THRESHOLD = 0.7  # >= 70% pass → yellow
# < 70% → red


def _pick_color(passed: int, total: int) -> str:
    """Pick badge color based on pass rate."""
    if total == 0:
        return "#9f9f9f"  # gray for no data
    rate = passed / total
    if rate >= _GREEN_THRESHOLD:
        return "#4c1"     # bright green
    if rate >= _YELLOW_THRESHOLD:
        return "#dfb317"  # yellow
    return "#e05d44"      # red


def _text_width(text: str) -> int:
    """Estimate text width in SVG units (approximate, 6.5px per char)."""
    return int(len(text) * 6.5) + 10


def generate_badge_svg(
    *,
    passed: int,
    failed: int,
    errors: int = 0,
    label: str = "CheckAgent",
) -> str:
    """Generate a shields.io-style SVG badge.

    Parameters
    ----------
    passed:
        Number of probes that passed.
    failed:
        Number of probes that had safety findings.
    errors:
        Number of probes that errored (excluded from score).
    label:
        Left-side label text.

    Returns
    -------
    str
        SVG content as a string.
    """
    total = passed + failed
    message = f"{passed}/{total} safe" if total > 0 else "no data"

    color = _pick_color(passed, total)
    lw = _text_width(label)
    mw = _text_width(message)
    tw = lw + mw
    lx = lw / 2
    mx = lw + mw / 2
    font = (
        "Verdana,Geneva,DejaVu Sans,sans-serif"
    )

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg"'
        f' width="{tw}" height="20" role="img"'
        f' aria-label="{label}: {message}">',
        f"  <title>{label}: {message}</title>",
        '  <linearGradient id="s" x2="0" y2="100%">',
        '    <stop offset="0" stop-color="#bbb" stop-opacity=".1"/>',
        '    <stop offset="1" stop-opacity=".1"/>',
        "  </linearGradient>",
        '  <clipPath id="r">',
        f'    <rect width="{tw}" height="20" rx="3" fill="#fff"/>',
        "  </clipPath>",
        '  <g clip-path="url(#r)">',
        f'    <rect width="{lw}" height="20" fill="#555"/>',
        f'    <rect x="{lw}" width="{mw}" height="20" fill="{color}"/>',
        f'    <rect width="{tw}" height="20" fill="url(#s)"/>',
        "  </g>",
        f'  <g fill="#fff" text-anchor="middle" font-family="{font}"'
        f' text-rendering="geometricPrecision" font-size="11">',
        f'    <text aria-hidden="true" x="{lx}" y="15"'
        f' fill="#010101" fill-opacity=".3">{label}</text>',
        f'    <text x="{lx}" y="14">{label}</text>',
        f'    <text aria-hidden="true" x="{mx}" y="15"'
        f' fill="#010101" fill-opacity=".3">{message}</text>',
        f'    <text x="{mx}" y="14">{message}</text>',
        "  </g>",
        "</svg>",
    ]
    return "\n".join(lines)


def write_badge(
    path: str | Path,
    *,
    passed: int,
    failed: int,
    errors: int = 0,
    label: str = "CheckAgent",
) -> Path:
    """Generate and write a badge SVG to disk.

    Returns the path written to.
    """
    p = Path(path)
    svg = generate_badge_svg(
        passed=passed,
        failed=failed,
        errors=errors,
        label=label,
    )
    p.write_text(svg)
    return p
