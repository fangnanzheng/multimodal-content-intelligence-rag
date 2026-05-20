from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from content_intel.config import config_path
from content_intel.pipeline.build_artifacts import build_demo_artifacts


def main() -> None:
    required = [
        config_path("paths.sqlite_db"),
        config_path("paths.vectorizer"),
        config_path("paths.vector_matrix"),
        config_path("paths.risk_model"),
    ]
    if all(path.exists() for path in required):
        print("Demo artifacts already exist.")
        return

    rows = int(os.getenv("DEMO_ROWS", "1200"))
    print(f"Building demo artifacts with {rows} rows...")
    build_demo_artifacts(rows=rows)


if __name__ == "__main__":
    main()

