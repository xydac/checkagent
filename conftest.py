"""Root conftest — checks that tests run against the editable source install."""
from __future__ import annotations

import importlib.util
import os


def pytest_configure(config):
    spec = importlib.util.find_spec("checkagent")
    if spec is None:
        return
    origin = spec.origin or ""
    src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "src"))
    if not origin.startswith(src_dir):
        raise SystemExit(
            f"\n[checkagent] Tests must run against the LOCAL source, not the installed package.\n"
            f"  Installed: {origin!r}\n"
            f"  Expected:  {src_dir!r}\n"
            f"  Fix: pip install -e . --break-system-packages\n"
        )
