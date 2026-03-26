#!/usr/bin/env python3
"""
PACE Web UI — Launch / Restart / Stop helper
Usage:
    python scripts/launch_web_app.py start   [--port 5000] [--host 127.0.0.1] [--debug]
    python scripts/launch_web_app.py restart [--port 5000] [--host 127.0.0.1] [--debug]
    python scripts/launch_web_app.py stop
    python scripts/launch_web_app.py status
"""

import argparse
import os
import signal
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEB_APP_DIR = PROJECT_ROOT / "web_app"
PID_FILE = WEB_APP_DIR / ".web_app.pid"


def _read_pid() -> int | None:
    """Return the stored PID or None."""
    if PID_FILE.exists():
        try:
            pid = int(PID_FILE.read_text().strip())
            # Check if that process is actually alive
            os.kill(pid, 0)
            return pid
        except (ValueError, ProcessLookupError, PermissionError):
            PID_FILE.unlink(missing_ok=True)
    return None


def _write_pid(pid: int):
    PID_FILE.write_text(str(pid))


def _kill(pid: int):
    """Terminate process and its entire process group, then force-kill after timeout."""
    print(f"  Sending SIGTERM to PID {pid} …")
    try:
        # Kill the entire process group (handles Werkzeug worker threads)
        pgid = os.getpgid(pid)
        os.killpg(pgid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        # Fall back to killing just the process
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            print("  Process already gone.")
            PID_FILE.unlink(missing_ok=True)
            return

    # Wait up to 5 seconds for graceful shutdown
    for _ in range(50):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.1)
    else:
        print("  Graceful shutdown timed-out, sending SIGKILL …")
        try:
            pgid = os.getpgid(pid)
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass  # Process already exited between SIGKILL and check
    PID_FILE.unlink(missing_ok=True)
    print("  Stopped.")


def cmd_status():
    pid = _read_pid()
    if pid:
        print(f"PACE Web UI is RUNNING  (PID {pid})")
    else:
        print("PACE Web UI is NOT running.")


def cmd_stop():
    pid = _read_pid()
    if pid is None:
        print("PACE Web UI is not running (no PID file found).")
    else:
        _kill(pid)
    # Also kill any stale processes on the port
    _kill_port_holders(5000)


def _kill_port_holders(port: int):
    """Kill any processes still holding the given port."""
    import subprocess
    try:
        result = subprocess.run(
            ["lsof", "-ti", f":{port}"],
            capture_output=True, text=True, timeout=5
        )
        pids = result.stdout.strip().split()
        for p in pids:
            try:
                os.kill(int(p), signal.SIGKILL)
            except (ProcessLookupError, ValueError, PermissionError):
                pass  # Process already gone or insufficient permissions
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass  # lsof not available or timed out; skip port cleanup


def cmd_start(host: str, port: int, debug: bool):
    pid = _read_pid()
    if pid:
        print(f"PACE Web UI is already running (PID {pid}).  Use 'restart' instead.")
        return

    print(f"Starting PACE Web UI on {host}:{port} …")

    # Fork a child so that the launcher can return immediately.
    child_pid = os.fork()
    if child_pid == 0:
        # ── Child process ──
        # Detach from terminal
        os.setsid()

        # Redirect stdout/stderr to a log file
        log_path = WEB_APP_DIR / "web_app.log"
        log_fd = os.open(str(log_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o640)
        os.dup2(log_fd, 1)
        os.dup2(log_fd, 2)
        os.close(log_fd)

        # Change to project root
        os.chdir(str(PROJECT_ROOT))
        sys.path.insert(0, str(PROJECT_ROOT))

        # Now import and run Flask
        from web_app.app import run_server
        run_server(host=host, port=port, debug=debug)
        sys.exit(0)
    else:
        # ── Parent process ──
        _write_pid(child_pid)
        # Give it a moment to either start or crash
        time.sleep(1.5)
        alive = _read_pid()
        if alive:
            print(f"  Started  (PID {child_pid})")
            print(f"  URL:  http://localhost:{port}")
            print(f"  Logs: {WEB_APP_DIR / 'web_app.log'}")
        else:
            print("  Failed to start. Check web_app.log for details.")


def cmd_restart(host: str, port: int, debug: bool):
    pid = _read_pid()
    if pid:
        print("Stopping current instance …")
        _kill(pid)
        time.sleep(0.5)
    cmd_start(host, port, debug)


# ─── CLI ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="PACE Web UI — start / restart / stop",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="action")

    p_start = sub.add_parser("start", help="Start the web server")
    p_start.add_argument("--port", type=int, default=5000)
    p_start.add_argument("--host", type=str, default="127.0.0.1")
    p_start.add_argument("--debug", action="store_true")

    p_restart = sub.add_parser("restart", help="Restart the web server")
    p_restart.add_argument("--port", type=int, default=5000)
    p_restart.add_argument("--host", type=str, default="127.0.0.1")
    p_restart.add_argument("--debug", action="store_true")

    sub.add_parser("stop", help="Stop the web server")
    sub.add_parser("status", help="Check if the server is running")

    args = parser.parse_args()

    if args.action is None:
        parser.print_help()
        sys.exit(1)

    if args.action == "start":
        cmd_start(args.host, args.port, args.debug)
    elif args.action == "restart":
        cmd_restart(args.host, args.port, args.debug)
    elif args.action == "stop":
        cmd_stop()
    elif args.action == "status":
        cmd_status()


if __name__ == "__main__":
    main()
