"""Capture a Grafana dashboard screenshot via headless Chromium.

Used by the `Capture Grafana dashboard screenshot` workflow.
Anonymous viewer auth on the Grafana side keeps this scriptless of
credentials — Playwright just navigates to the kiosk-mode URL.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="Full Grafana dashboard URL.")
    parser.add_argument("--out", type=Path, required=True, help="Output PNG path.")
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=2400)
    parser.add_argument(
        "--render-wait-ms",
        type=int,
        default=8000,
        help="Extra ms to wait after networkidle so panels finish drawing.",
    )
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": args.width, "height": args.height},
            device_scale_factor=2,
        )
        page = context.new_page()
        page.goto(args.url, wait_until="networkidle", timeout=60_000)
        page.wait_for_timeout(args.render_wait_ms)
        page.screenshot(path=str(args.out), full_page=True)
        browser.close()

    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
