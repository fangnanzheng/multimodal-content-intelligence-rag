from __future__ import annotations

import pandas as pd

from content_intel.retrieval.search import get_searcher
from content_intel.schemas import InsightResponse, SearchResult
from content_intel.storage import load_joined_content


def _humanize(value: str) -> str:
    return " ".join(word.capitalize() for word in value.replace("_", " ").split())


def generate_insight(question: str) -> InsightResponse:
    df = load_joined_content()
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date
    high_risk = df[df["risk_label"] != "safe"]
    latest_day = str(df["date"].max())
    high_risk_rate = float((df["risk_label"] != "safe").mean())
    top_label = str(high_risk["risk_label"].value_counts().idxmax()) if len(high_risk) else "safe"
    avg_reports = float(df["reports"].mean())
    cases = get_searcher().search(question, top_k=3)

    answer = (
        f"High-risk content accounts for {high_risk_rate:.1%} of the demo corpus. "
        f"The most frequent risk class is {_humanize(top_label)}. On {latest_day}, operators "
        f"should prioritize review queues with elevated reports and retrieve similar "
        f"cases before taking action."
    )
    return InsightResponse(
        question=question,
        answer=answer,
        supporting_metrics={
            "rows": int(len(df)),
            "high_risk_rate": round(high_risk_rate, 4),
            "top_risk_label": top_label,
            "average_reports": round(avg_reports, 3),
        },
        representative_cases=[SearchResult(**case) for case in cases],
    )
