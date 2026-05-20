from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


RiskLabel = Literal[
    "safe",
    "financial_scam",
    "spam",
    "counterfeit",
    "health_misinformation",
]


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=2)
    top_k: int = Field(default=5, ge=1, le=20)
    risk_label: str | None = None
    date_from: str | None = None
    date_to: str | None = None


class SearchResult(BaseModel):
    content_id: str
    similarity: float
    post_text: str
    ocr_text: str
    image_type: str | None = None
    visual_risk_signal: str | None = None
    risk_label: str
    risk_score: float
    timestamp: str
    category: str


class ModerateRequest(BaseModel):
    content_id: str


class ModerationExplanation(BaseModel):
    content_id: str
    risk_label: str
    risk_score: float = Field(ge=0.0, le=1.0)
    uncertainty: Literal["low", "medium", "high"]
    needs_human_review: bool
    evidence: list[str]
    similar_cases: list[SearchResult]
    policy_evidence: list[str]
    recommended_action: str
    reasoning: str


class InsightRequest(BaseModel):
    question: str = Field(..., min_length=5)


class InsightResponse(BaseModel):
    question: str
    answer: str
    supporting_metrics: dict[str, float | int | str]
    representative_cases: list[SearchResult]
