# Starts a cloudflared quick tunnel, then the app with REWIND_PUBLIC_BASE_URL set to it.
# The quick-tunnel URL changes on every start, so nobody has to copy it by hand. Usage:
#   uv run python scripts/serve_live.py [--cloudflared PATH] [--port 8000]

import argparse
import os
import re
import subprocess
import threading
import time

import uvicorn

DEFAULT_CLOUDFLARED = r"C:\Program Files (x86)\cloudflared\cloudflared.exe"
TUNNEL_URL = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")
WAIT_S = 60


def start_tunnel(cloudflared: str, port: int) -> tuple[subprocess.Popen[str], str]:
    process = subprocess.Popen(
        [cloudflared, "tunnel", "--no-autoupdate", "--url", f"http://localhost:{port}"],
        stderr=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        text=True,
    )
    assert process.stderr is not None
    found: list[str] = []

    # cloudflared logs to stderr for as long as it runs; it must be drained or it blocks.
    def drain() -> None:
        assert process.stderr is not None
        for line in process.stderr:
            match = TUNNEL_URL.search(line)
            if match and not found:
                found.append(match.group(0))

    threading.Thread(target=drain, daemon=True).start()
    deadline = time.monotonic() + WAIT_S
    while not found:
        if process.poll() is not None or time.monotonic() > deadline:
            process.kill()
            raise SystemExit("cloudflared did not report a tunnel URL")
        time.sleep(0.2)
    return process, found[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cloudflared", default=DEFAULT_CLOUDFLARED)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    tunnel, url = start_tunnel(args.cloudflared, args.port)
    print(f"tunnel: {url}", flush=True)
    os.environ["REWIND_PUBLIC_BASE_URL"] = url
    try:
        uvicorn.run("rewind.app.main:create_app", factory=True, host="127.0.0.1", port=args.port)
    finally:
        tunnel.terminate()


if __name__ == "__main__":
    main()
