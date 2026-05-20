from __future__ import annotations

from content_intel.rag.explainer import explain_content
from content_intel.retrieval.search import get_searcher
from content_intel.storage import load_joined_content


def test_search_returns_ranked_results() -> None:
    results = get_searcher().search("guaranteed return private contact", top_k=5)
    assert len(results) == 5
    assert results[0]["similarity"] >= results[-1]["similarity"]
    assert "content_id" in results[0]


def test_explanation_is_structured() -> None:
    content_id = load_joined_content().iloc[0]["content_id"]
    explanation = explain_content(content_id)
    assert explanation.content_id == content_id
    assert explanation.evidence
    assert any("Image modality" in item for item in explanation.evidence)
    assert explanation.policy_evidence
    assert 0 <= explanation.risk_score <= 1
