#!/usr/bin/env python3
"""Launch the consolidated Next.js workbench."""

from __future__ import annotations

import argparse
from pathlib import Path
import socket
import subprocess
import threading
import webbrowser


def main() -> None:
    parser = argparse.ArgumentParser(description="Open the CPU GameBench profiler dashboard.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    web = root / "web"
    if not (web / "node_modules" / ".bin" / "next").exists():
        raise SystemExit("Next.js is not installed. Run: cd web && npm install")

    actual_port = args.port
    with socket.socket() as probe:
        try:
            probe.bind((args.host, actual_port))
        except OSError:
            probe.bind((args.host, 0))
            actual_port = probe.getsockname()[1]
            print(f"Port {args.port} is busy; selected {actual_port} instead.")
    url = f"http://{args.host}:{actual_port}/profiler"
    print(f"CPU Workbench: {url}")
    print("Press Ctrl+C to stop.")
    if not args.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    try:
        subprocess.run(
            ["npm", "run", "dev", "--", "--hostname", args.host, "--port", str(actual_port)],
            cwd=web,
            check=True,
        )
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.returncode) from error


if __name__ == "__main__":
    main()
