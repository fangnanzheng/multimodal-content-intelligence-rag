from __future__ import annotations

import json
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from scipy import sparse

from content_intel.config import config_path, load_config
from content_intel.storage import load_joined_content


@dataclass
class SearchFilters:
    risk_label: str | None = None
    date_from: str | None = None
    date_to: str | None = None


class VectorSearcher:
    """Small retrieval adapter.

    The local demo uses sparse TF-IDF cosine similarity so it can run without
    FAISS. The artifact layout intentionally mirrors a FAISS-backed service:
    vectorizer/model artifacts are loaded once, while metadata filtering stays
    outside the vector index.
    """

    def __init__(self) -> None:
        self.cfg = load_config()
        self.vectorizer = joblib.load(config_path("paths.vectorizer"))
        self.matrix = sparse.load_npz(config_path("paths.vector_matrix"))
        self.content = load_joined_content()
        self.metadata = json.loads(config_path("paths.index_metadata").read_text())

    def search(
        self, query: str, top_k: int | None = None, filters: SearchFilters | None = None
    ) -> list[dict[str, object]]:
        top_k = top_k or int(self.cfg["retrieval"]["default_top_k"])
        filters = filters or SearchFilters()
        q = self.vectorizer.transform([query])
        scores = np.asarray((q @ self.matrix.T).toarray()).ravel()
        candidates = self.content.copy()
        candidates["similarity"] = scores

        if filters.risk_label:
            candidates = candidates[candidates["risk_label"] == filters.risk_label]
        if filters.date_from:
            candidates = candidates[candidates["timestamp"] >= filters.date_from]
        if filters.date_to:
            candidates = candidates[candidates["timestamp"] <= filters.date_to]

        candidates = candidates.sort_values("similarity", ascending=False).head(top_k)
        return [
            {
                "content_id": row.content_id,
                "similarity": round(float(row.similarity), 4),
                "post_text": row.post_text,
                "ocr_text": row.ocr_text,
                "image_type": getattr(row, "image_type", None),
                "visual_risk_signal": getattr(row, "visual_risk_signal", None),
                "risk_label": row.risk_label,
                "risk_score": float(row.risk_score),
                "timestamp": row.timestamp,
                "category": row.category,
            }
            for row in candidates.itertuples()
        ]

    def similar_to_content(self, content: dict[str, object], top_k: int = 5) -> list[dict[str, object]]:
        query = "\n".join(
            [
                str(content.get("post_text", "")),
                str(content.get("ocr_text", "")),
                str(content.get("image_type", "")),
                str(content.get("visual_risk_signal", "")),
            ]
        )
        results = self.search(query=query, top_k=top_k + 1)
        return [r for r in results if r["content_id"] != content["content_id"]][:top_k]


_SEARCHER: VectorSearcher | None = None


def get_searcher() -> VectorSearcher:
    global _SEARCHER
    if _SEARCHER is None:
        _SEARCHER = VectorSearcher()
    return _SEARCHER
