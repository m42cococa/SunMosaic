"""Entry point that starts the Streamlit UI in the browser."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    from streamlit.web import cli as streamlit_cli

    app = Path(__file__).with_name("app.py")
    sys.argv = ["streamlit", "run", str(app), "--server.headless=false"]
    return int(streamlit_cli.main() or 0)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
