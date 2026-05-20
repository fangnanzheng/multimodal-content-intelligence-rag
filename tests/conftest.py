from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def pytest_sessionstart(session):  # type: ignore[no-untyped-def]
    from content_intel.pipeline.build_artifacts import build_demo_artifacts
    from content_intel.storage import load_joined_content

    try:
        if len(load_joined_content()) >= 1000:
            return
    except Exception:
        pass

    build_demo_artifacts(rows=1200, seed=42)
