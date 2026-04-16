"""Root conftest — checks that tests run against the editable source install."""
from __future__ import annotations

import importlib.util
import os
import warnings


def pytest_configure(config):
    spec = importlib.util.find_spec("checkagent")
    if spec is None:
        return
    origin = spec.origin or ""
    src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "src"))
    if not origin.startswith(src_dir):
        warnings.warn(
            f"Tests are running against an installed checkagent package at "
            f"{origin!r}, NOT the local source in src/. "
            "Run 'pip install -e . --break-system-packages' to fix this.",
            stacklevel=1,
        )
