"""Draw the repository icon (docs/assets/icon.svg). Standard library only.

Usage:
    python scripts/make_assets.py

The icon is a rounded square in the area color with a simple geometric mark,
samples scheduled on two workers, one much longer than the rest; no text and no
third-party logo. The output is byte-stable: running the script twice gives the same file.
"""

import os

# area colors (fill, stroke), shared with docs/diagrams/architecture.d2
FILL, INK = "#EDE4FA", "#6D3FB3"


def mark(ink):
    """SVG elements of the mark, on a 64 x 64 canvas, drawn in `ink`."""
    f = f'fill="{ink}"'
    return (
        f'<rect x="12" y="17" width="10" height="11" rx="2" {f}/>'
        f'<rect x="25" y="17" width="27" height="11" rx="2" {f}/>'
        f'<rect x="17" y="36" width="12" height="11" rx="2" {f}/>'
        f'<rect x="32" y="36" width="8" height="11" rx="2" {f}/>'
        f'<rect x="43" y="36" width="9" height="11" rx="2" {f}/>'
    )


def icon():
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="96" height="96" viewBox="0 0 64 64">'
        f'<rect x="2" y="2" width="60" height="60" rx="14" fill="{FILL}" stroke="{INK}" stroke-width="3"/>'
        f"{mark(INK)}</svg>\n"
    )


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = os.path.join(root, "docs", "assets", "icon.svg")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write(icon())
    print(f"wrote {os.path.relpath(out, root)}")


if __name__ == "__main__":
    main()
