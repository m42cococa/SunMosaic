"""Desktop launcher: the browser UI in its own window, as a Mac application.

The Streamlit server runs as a child process on a free local port, and the window is a native
WebKit view (pywebview) on the main thread, which Cocoa requires.  Closing the window, Cmd-Q
or Quit from the Dock stops the server.  Output from the server goes to a log file, since a
desktop application has no terminal.
"""

from __future__ import annotations

import argparse
import atexit
import datetime as _dt
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

from . import __version__

HOST = "127.0.0.1"
LOG_PATH = Path.home() / "Library" / "Logs" / "SunMosaic.log"
STARTUP_TIMEOUT_S = 120.0
WINDOW_SIZE = (1440, 1000)
PARENT_POLL_S = 1.0


def free_port(host: str = HOST) -> int:
    """A port nothing is listening on right now, chosen by the operating system."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind((host, 0))
        return int(probe.getsockname()[1])


def server_environment(port: int, base: dict[str, str] | None = None) -> dict[str, str]:
    """Streamlit settings passed as environment variables.

    Inside the application bundle there is no ``.streamlit/config.toml`` to find, so every
    setting the project relies on is given here instead.
    """
    env = dict(os.environ if base is None else base)
    for leaked in ("PYTHONHOME", "PYTHONPATH"):
        env.pop(leaked, None)
    env.update({
        "PYTHONDONTWRITEBYTECODE": "1",  # never write into the signed bundle
        "STREAMLIT_SERVER_HEADLESS": "true",
        "STREAMLIT_SERVER_ADDRESS": HOST,
        "STREAMLIT_SERVER_PORT": str(port),
        "STREAMLIT_SERVER_MAX_UPLOAD_SIZE": "500",
        "STREAMLIT_SERVER_FILE_WATCHER_TYPE": "none",
        "STREAMLIT_SERVER_RUN_ON_SAVE": "false",
        "STREAMLIT_BROWSER_GATHER_USAGE_STATS": "false",
        "STREAMLIT_GLOBAL_DEVELOPMENT_MODE": "false",
        "STREAMLIT_CLIENT_TOOLBAR_MODE": "minimal",
    })
    return env


def server_command(parent_pid: int | None = None) -> list[str]:
    """The child process: this module again, in server mode, told which process to outlive."""
    parent = os.getpid() if parent_pid is None else parent_pid
    # -B, not PYTHONDONTWRITEBYTECODE: -I makes Python ignore that variable.
    return [sys.executable, "-I", "-B", "-m", "sunmosaic.desktop", "--serve", str(parent)]


def watch_parent(parent_pid: int, poll_s: float = PARENT_POLL_S) -> threading.Thread:
    """End this process as soon as ``parent_pid`` is no longer its parent.

    Quitting from the menu or the Dock ends the window process without running Python's exit
    handlers, and Force Quit or a crash never runs them, so the server cannot rely on being
    told to stop.  It watches for its parent instead: once the parent is gone, the process is
    handed to launchd and its parent id changes.
    """

    def watch() -> None:
        while True:
            if os.getppid() != parent_pid:
                os._exit(0)
            time.sleep(poll_s)

    thread = threading.Thread(target=watch, name="sunmosaic-parent-watch", daemon=True)
    thread.start()
    return thread


def serve(parent_pid: int) -> int:
    """Server side: run the Streamlit app until the window process goes away."""
    watch_parent(parent_pid)
    from streamlit.web import cli as streamlit_cli

    sys.argv = ["streamlit", "run", str(Path(__file__).with_name("app.py"))]
    return int(streamlit_cli.main() or 0)


def wait_until_healthy(
    url: str, timeout_s: float, process: subprocess.Popen | None = None, poll_s: float = 0.25,
) -> bool:
    """True once the server answers its health check; False if it exits or time runs out."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(f"{url}/_stcore/health", timeout=2) as response:
                if response.status == 200 and response.read().strip() == b"ok":
                    return True
        except OSError:
            pass
        time.sleep(poll_s)
    return False


def stop_server(process: subprocess.Popen, grace_s: float = 5.0) -> None:
    """Ask the server to stop, and make sure it has."""
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=grace_s)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _alert(message: str) -> None:
    """Tell a user who has no terminal that the application could not start."""
    script = f'display alert "SunMosaic could not start" message {message!r} as critical'
    subprocess.run(["osascript", "-e", script.replace("'", '"')], check=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sunmosaic-app", description=__doc__.splitlines()[0])
    parser.add_argument("--browser", action="store_true",
                        help="open the default browser instead of an application window")
    parser.add_argument("--port", type=int, default=0, help="port to serve on (default: any free)")
    parser.add_argument("--serve", type=int, metavar="PARENT_PID", help=argparse.SUPPRESS)
    # Finder may add arguments of its own, such as -psn_..., so unknown ones are ignored.
    args, _ = parser.parse_known_args(argv)
    if args.serve is not None:
        return serve(args.serve)

    port = args.port or free_port()
    url = f"http://{HOST}:{port}"
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log = LOG_PATH.open("a", buffering=1, encoding="utf-8")
    stamp = _dt.datetime.now().astimezone().isoformat(timespec="seconds")
    log.write(f"\n=== SunMosaic {__version__} started {stamp}, serving {url} "
              f"with {sys.executable}\n")

    process = subprocess.Popen(
        server_command(), env=server_environment(port), stdout=log, stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL, cwd=str(Path.home()),
    )
    atexit.register(stop_server, process)

    if not wait_until_healthy(url, STARTUP_TIMEOUT_S, process):
        log.write("the server did not answer its health check\n")
        stop_server(process)
        if not args.browser:
            _alert(f"The image server did not start. Details are in {LOG_PATH}.")
        print(f"SunMosaic could not start; see {LOG_PATH}", file=sys.stderr)
        return 1
    log.write(f"server ready at {url}\n")

    if args.browser:
        webbrowser.open(url)
        try:
            process.wait()
        except KeyboardInterrupt:
            pass
        finally:
            stop_server(process)
        return 0

    import webview  # only needed for the window, and only installed with the desktop extra

    webview.settings["ALLOW_DOWNLOADS"] = True
    webview.create_window("SunMosaic", url, width=WINDOW_SIZE[0], height=WINDOW_SIZE[1],
                          min_size=(900, 650))
    webview.start()
    stop_server(process)
    log.write("window closed, server stopped\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
