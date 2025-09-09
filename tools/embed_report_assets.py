#!/usr/bin/env python3
"""
Embed all external assets (images, csv/html/markdown downloads) into a single
self‑contained HTML file using data URIs. Designed for the FleetMix executive
summary report.

Usage:
  python tools/embed_report_assets.py <input_html> [output_html]

Notes:
  - Replaces <img src="..."> with data:image/*;base64 URIs
  - Replaces <iframe src="..."> with the inlined HTML content wrapped in a div
  - Rewrites <a href="*.csv|*.md" download> to data:*;base64 URIs
  - Leaves inline SVG untouched
"""

from __future__ import annotations

import base64
import mimetypes
import re
import sys
from pathlib import Path

IMG_RE = re.compile(r"(<img[^>]+src=\")((?!data:)[^\"]+)(\"[^>]*>)", re.IGNORECASE)
IFRAME_RE = re.compile(r"<iframe[^>]+src=\"([^\"]+)\"[^>]*></iframe>", re.IGNORECASE)
ANCHOR_DL_RE = re.compile(
    r"(<a[^>]+href=\")((?!data:)[^\"]+\.(?:csv|md))(\"[^>]*download[^>]*>)",
    re.IGNORECASE,
)


def to_data_uri(path: Path) -> str:
    data = path.read_bytes()
    mime, _ = mimetypes.guess_type(path.as_posix())
    if not mime:
        # Fallbacks
        if path.suffix.lower() in {".png"}:
            mime = "image/png"
        elif path.suffix.lower() in {".jpg", ".jpeg"}:
            mime = "image/jpeg"
        elif path.suffix.lower() == ".csv":
            mime = "text/csv"
        elif path.suffix.lower() == ".md":
            mime = "text/markdown"
        else:
            mime = "application/octet-stream"
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}"


def embed_images(html: str, base_dir: Path) -> str:
    def repl(match: re.Match) -> str:
        prefix, src, suffix = match.groups()
        p = (base_dir / src).resolve()
        if not p.exists():
            return match.group(0)
        return prefix + to_data_uri(p) + suffix

    return IMG_RE.sub(repl, html)


def embed_iframes(html: str, base_dir: Path) -> str:
    def repl(match: re.Match) -> str:
        src = match.group(1)
        p = (base_dir / src).resolve()
        if not p.exists():
            return match.group(0)
        # Inline the HTML content directly
        try:
            inner = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            # Fallback to base64 srcdoc if binary
            data_uri = to_data_uri(p)
            return f'<iframe src="{data_uri}"></iframe>'
        return f'<div class="embedded-iframe">{inner}</div>'

    return IFRAME_RE.sub(repl, html)


def embed_download_links(html: str, base_dir: Path) -> str:
    def repl(match: re.Match) -> str:
        prefix, href, suffix = match.groups()
        p = (base_dir / href).resolve()
        if not p.exists():
            return match.group(0)
        return prefix + to_data_uri(p) + suffix

    return ANCHOR_DL_RE.sub(repl, html)


def main() -> None:
    if len(sys.argv) < 2:
        print(
            "Usage: embed_report_assets.py <input_html> [output_html]", file=sys.stderr
        )
        sys.exit(2)
    src_path = Path(sys.argv[1]).resolve()
    out_path = (
        Path(sys.argv[2]).resolve()
        if len(sys.argv) > 2
        else src_path.with_name(src_path.stem + "_embedded.html")
    )

    base_dir = src_path.parent
    html = src_path.read_text(encoding="utf-8")

    html = embed_images(html, base_dir)
    html = embed_iframes(html, base_dir)
    html = embed_download_links(html, base_dir)

    out_path.write_text(html, encoding="utf-8")
    print(f"Wrote embedded report to: {out_path}")


if __name__ == "__main__":
    main()
