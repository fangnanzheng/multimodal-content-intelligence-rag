from __future__ import annotations

from pathlib import Path

from content_intel.config import config_path
from content_intel.storage import load_joined_content


def test_artifacts_are_created() -> None:
    assert config_path("paths.sqlite_db").exists()
    assert config_path("paths.vectorizer").exists()
    assert config_path("paths.vector_matrix").exists()
    assert config_path("paths.risk_model").exists()


def test_joined_content_has_predictions() -> None:
    df = load_joined_content()
    assert len(df) >= 1000
    assert {"content_id", "manual_label", "risk_label", "risk_score"}.issubset(df.columns)
    assert {"image_path", "image_type", "ocr_text", "visual_risk_signal"}.issubset(df.columns)
    assert df["risk_score"].between(0, 1).all()
    assert df["image_path"].map(lambda path: Path(path).exists()).all()
