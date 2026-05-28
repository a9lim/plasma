#!/usr/bin/env python3
"""Drive plasma/tests/physics-validation.html via Playwright and print a summary.

Usage:
  cd /Users/a9lim/Work/a9lim.github.io && ./dev.sh
  cd plasma && ./tests/physics-validation.py --port 8787

For the faster static path:
  cd plasma && python -m http.server 8090
  ./tests/physics-validation.py --static-root --port 8090
"""

import argparse
import json
import sys
from playwright.sync_api import sync_playwright


def validation_url(port, static_root):
    if static_root:
        return f"http://localhost:{port}/tests/physics-validation.html?auto=1"
    return f"http://localhost:{port}/plasma/tests/physics-validation.html?auto=1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument("--timeout", type=int, default=360)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--static-root", action="store_true",
                    help="server root is plasma/ rather than a9lim.github.io/")
    ap.add_argument("--headless", action="store_true",
                    help="try headless Chromium; headed is safer for WebGPU")
    args = ap.parse_args()

    url = validation_url(args.port, args.static_root)
    console_lines = []

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=args.headless,
            args=[
                "--enable-unsafe-webgpu",
                "--enable-features=Vulkan",
                "--use-vulkan",
            ],
        )
        page = browser.new_page()

        def on_console(msg):
            text = msg.text
            print(f"[browser] {text}", file=sys.stderr)
            for line in text.splitlines():
                console_lines.append(line)

        page.on("console", on_console)
        page.on("pageerror", lambda e: print(f"[pageerror] {e}", file=sys.stderr))
        print(f"[driver] navigating to {url}", file=sys.stderr)
        page.goto(url, wait_until="networkidle")

        got_end = False
        result = None
        try:
            page.wait_for_function(
                "() => window.__PHYSICS_VALIDATION_DONE === true",
                timeout=args.timeout * 1000,
            )
            result = page.evaluate("() => window.__PHYSICS_VALIDATION_RESULT")
            got_end = True
        except Exception:
            got_end = any(line.strip() == "PHYSICS_VALIDATION_JSON_END"
                          for line in console_lines)
        browser.close()

    if not got_end:
        print("[driver] TIMEOUT", file=sys.stderr)
        sys.exit(3)

    if result is None:
        in_json = False
        chunks = []
        for line in console_lines:
            line = line.strip()
            if line == "PHYSICS_VALIDATION_JSON_BEGIN":
                in_json = True
                chunks = []
                continue
            if line == "PHYSICS_VALIDATION_JSON_END":
                break
            if in_json:
                chunks.append(line)

        if not chunks:
            print("[driver] no JSON payload captured", file=sys.stderr)
            sys.exit(4)

        result = json.loads("\n".join(chunks))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fp:
            json.dump(result, fp, indent=2)
        print(f"[driver] wrote {args.out}", file=sys.stderr)

    print(f"physics validation: {result['passed']} passed, {result['failed']} failed")
    for row in result["rows"]:
        status = "PASS" if row["ok"] else "FAIL"
        print(f"{status:4} {row['case']:<45} N={row['n']:<4} steps={row['steps']:<5} {row['metric']}")

    if result["failed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
