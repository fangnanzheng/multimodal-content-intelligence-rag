from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

from content_intel.config import config_path


def db_path() -> Path:
    return config_path("paths.sqlite_db")


def read_table(table: str) -> pd.DataFrame:
    with sqlite3.connect(db_path()) as conn:
        return pd.read_sql_query(f"select * from {table}", conn)


def get_content(content_id: str) -> dict[str, Any] | None:
    with sqlite3.connect(db_path()) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            """
            select c.*, p.risk_label, p.risk_score, p.recommended_action
            from content c
            left join predictions p using (content_id)
            where c.content_id = ?
            """,
            (content_id,),
        ).fetchone()
    return dict(row) if row else None


def load_joined_content() -> pd.DataFrame:
    with sqlite3.connect(db_path()) as conn:
        return pd.read_sql_query(
            """
            select c.*, p.risk_label, p.risk_score, p.recommended_action
            from content c
            left join predictions p using (content_id)
            """,
            conn,
        )

