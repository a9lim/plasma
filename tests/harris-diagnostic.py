#!/usr/bin/env python3
"""Drive plasma/tests/harris-diagnostic.html via Playwright (headed Chrome
for WebGPU), wait for the diagnostic to finish, dump JSON to stdout.

Usage:
  cd plasma && python -m http.server 8090 &
  ./tests/harris-diagnostic.py --N 256 --maxSteps 400 --sampleEvery 5 --eta 1e-3
"""

import argparse, json, sys, time
from playwright.sync_api import sync_playwright

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--N', type=int, default=256)
    ap.add_argument('--maxSteps', type=int, default=400)
    ap.add_argument('--sampleEvery', type=int, default=5)
    ap.add_argument('--eta', type=str, default='1e-3')
    ap.add_argument('--port', type=int, default=8090)
    ap.add_argument('--timeout', type=int, default=180,
                    help='wall-clock seconds before giving up')
    ap.add_argument('--out', type=str, default=None,
                    help='optional path to save full JSON result')
    ap.add_argument('--tight', action='store_true',
                    help='enable tight-loop mode (no per-step await — mirrors prod)')
    args = ap.parse_args()

    url = (f'http://localhost:{args.port}/plasma/tests/harris-diagnostic.html'
           f'?auto=1&N={args.N}&maxSteps={args.maxSteps}'
           f'&sampleEvery={args.sampleEvery}&eta={args.eta}'
           f'{"&tight=1" if args.tight else ""}')

    with sync_playwright() as p:
        # Headed because WebGPU needs a real GPU context. Chromium with
        # the flag set enables WebGPU.
        browser = p.chromium.launch(
            headless=False,
            args=['--enable-unsafe-webgpu', '--enable-features=Vulkan',
                  '--use-vulkan'],
        )
        context = browser.new_context()
        page = context.new_page()

        console_lines = []
        def on_console(msg):
            try:
                t = msg.text
            except Exception:
                t = str(msg)
            console_lines.append(t)
            print(f'[browser] {t}', file=sys.stderr)

        page.on('console', on_console)
        page.on('pageerror', lambda e: print(f'[pageerror] {e}', file=sys.stderr))

        print(f'[driver] navigating to {url}', file=sys.stderr)
        page.goto(url, wait_until='networkidle')

        # Wait for the JSON sentinel in console output.
        deadline = time.time() + args.timeout
        got_end = False
        while time.time() < deadline:
            if any(l == 'HARRIS_DIAGNOSTIC_JSON_END' for l in console_lines):
                got_end = True
                break
            if any(l.startswith('HARRIS_DIAGNOSTIC_ERROR') for l in console_lines):
                print('[driver] page reported error', file=sys.stderr)
                browser.close()
                sys.exit(2)
            time.sleep(0.5)
        if not got_end:
            print('[driver] TIMEOUT', file=sys.stderr)
        browser.close()

    # Extract the JSON payload from the captured lines.
    in_json = False
    json_chunks = []
    for line in console_lines:
        if line == 'HARRIS_DIAGNOSTIC_JSON_BEGIN':
            in_json = True
            json_chunks = []
            continue
        if line == 'HARRIS_DIAGNOSTIC_JSON_END':
            in_json = False
            break
        if in_json:
            json_chunks.append(line)

    if not json_chunks:
        print('[driver] no JSON captured', file=sys.stderr)
        sys.exit(4)

    payload = '\n'.join(json_chunks)
    try:
        result = json.loads(payload)
    except json.JSONDecodeError as e:
        print(f'[driver] JSON parse failed: {e}', file=sys.stderr)
        print(payload[:1000])
        sys.exit(5)

    if args.out:
        with open(args.out, 'w') as out_fp:
            json.dump(result, out_fp, indent=2)
        print(f'[driver] wrote {args.out}', file=sys.stderr)

    # Pretty-print summary.
    meta = result['meta']
    series = result['series']
    print(f'\n=== Harris diagnostic: N={meta["N"]}, eta={meta["eta"]}, '
          f'maxSteps={meta["maxSteps"]}, sampleEvery={meta["sampleEvery"]} ===')
    print(f'NaN onset: step {meta["nanStep"]}' if meta['nanStep'] is not None
          else 'No NaN within maxSteps.')
    print()
    cols = ['step', 'tEst', 'dtHyp', 'rhoMin', 'rhoMax', 'pMin', 'pMax',
            'vMax', 'bMax', 'jMax', 'rhoFloorCount', 'pFloorCount', 'nanCount',
            'divbAvg']
    print(' '.join(f'{c:>11}' for c in cols))
    for row in series:
        def f(v):
            if v is None: return 'None'
            if isinstance(v, float):
                if v == 0: return '0'
                a = abs(v)
                if a >= 0.01 and a < 1000: return f'{v:.3f}'
                return f'{v:.2e}'
            return str(v)
        print(' '.join(f'{f(row.get(c, "")):>11}' for c in cols))

if __name__ == '__main__':
    main()
