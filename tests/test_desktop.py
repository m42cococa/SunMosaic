"""The desktop launcher's plumbing, without opening a window."""

from __future__ import annotations

import http.server
import os
import socket
import subprocess
import sys
import threading
import time

from sunmosaic import desktop


def test_importing_the_launcher_does_not_need_the_window_library():
    code = "import sys, sunmosaic.desktop; print('webview' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_free_port_can_be_bound():
    port = desktop.free_port()
    with socket.socket() as probe:
        probe.bind((desktop.HOST, port))


def test_server_environment_carries_every_setting_the_bundle_cannot_read_from_a_file():
    env = desktop.server_environment(8765, base={"PYTHONPATH": "/elsewhere", "HOME": "/Users/x"})
    assert env["STREAMLIT_SERVER_PORT"] == "8765"
    assert env["STREAMLIT_SERVER_HEADLESS"] == "true"
    assert env["PYTHONDONTWRITEBYTECODE"] == "1"
    assert env["STREAMLIT_SERVER_ADDRESS"] == "127.0.0.1"
    assert env["STREAMLIT_SERVER_MAX_UPLOAD_SIZE"] == "500"
    assert env["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] == "false"
    assert "PYTHONPATH" not in env and env["HOME"] == "/Users/x"
    command = desktop.server_command()
    assert command[0] == sys.executable
    assert command[-2:] == ["--serve", str(os.getpid())]


def _serve(body: bytes):
    class Health(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            status = 200 if self.path == "/_stcore/health" else 404
            self.send_response(status)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer((desktop.HOST, 0), Health)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def test_waits_for_a_healthy_server():
    server = _serve(b"ok")
    try:
        url = f"http://{desktop.HOST}:{server.server_address[1]}"
        assert desktop.wait_until_healthy(url, timeout_s=5.0)
    finally:
        server.shutdown()


def test_gives_up_when_nothing_answers_or_the_server_exits():
    url = f"http://{desktop.HOST}:{desktop.free_port()}"
    assert not desktop.wait_until_healthy(url, timeout_s=0.6, poll_s=0.1)
    finished = subprocess.Popen([sys.executable, "-c", "pass"])
    finished.wait()
    assert not desktop.wait_until_healthy(url, timeout_s=30.0, process=finished)


def test_stop_server_ends_the_child():
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    desktop.stop_server(child, grace_s=5.0)
    assert child.poll() is not None


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_the_server_side_ends_when_the_window_process_disappears():
    # The parent starts a child that watches it, reports the child's pid, and exits at once,
    # the way Quit from the Dock ends the window process without any clean-up.
    child = ("import sys, time; from sunmosaic.desktop import watch_parent; "
             "watch_parent(int(sys.argv[1]), 0.1); time.sleep(60)")
    parent = ("import os, subprocess, sys; "
              f"p = subprocess.Popen([sys.executable, '-c', {child!r}, str(os.getpid())]); "
              "print(p.pid, flush=True)")
    out = subprocess.run([sys.executable, "-c", parent], capture_output=True, text=True,
                         check=True, timeout=30)
    pid = int(out.stdout.strip())
    deadline = time.monotonic() + 15.0
    while _alive(pid) and time.monotonic() < deadline:
        time.sleep(0.1)
    if _alive(pid):
        os.kill(pid, 9)
        raise AssertionError("the server side outlived the window process")


def test_the_server_side_keeps_running_while_its_parent_lives():
    code = ("import os, time; from sunmosaic.desktop import watch_parent; "
            "watch_parent(os.getppid(), 0.05); time.sleep(1.0); print('alive')")
    child = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
    out, _ = child.communicate(timeout=30)
    assert out.strip() == "alive"
