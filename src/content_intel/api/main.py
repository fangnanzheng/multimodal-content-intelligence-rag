from __future__ import annotations

from fastapi import FastAPI, HTTPException

from content_intel.analytics.insights import generate_insight
from content_intel.rag.explainer import explain_content
from content_intel.retrieval.search import SearchFilters, get_searcher
from content_intel.schemas import (
    InsightRequest,
    InsightResponse,
    ModerateRequest,
    ModerationExplanation,
    SearchRequest,
    SearchResult,
)
from content_intel.storage import get_content


app = FastAPI(
    title="Multimodal Content Intelligence & RAG API",
    version="0.1.0",
    description="Lightweight trust & safety analytics API with retrieval-grounded explanations.",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/search", response_model=list[SearchResult])
def search(request: SearchRequest) -> list[SearchResult]:
    filters = SearchFilters(
        risk_label=request.risk_label,
        date_from=request.date_from,
        date_to=request.date_to,
    )
    results = get_searcher().search(request.query, top_k=request.top_k, filters=filters)
    return [SearchResult(**result) for result in results]


@app.get("/content/{content_id}")
def content(content_id: str) -> dict[str, object]:
    row = get_content(content_id)
    if row is None:
        raise HTTPException(status_code=404, detail="content_id not found")
    return row


@app.post("/moderate", response_model=ModerationExplanation)
def moderate(request: ModerateRequest) -> ModerationExplanation:
    try:
        return explain_content(request.content_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/analytics/insight", response_model=InsightResponse)
def insight(request: InsightRequest) -> InsightResponse:
    return generate_insight(request.question)

