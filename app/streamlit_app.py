from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from content_intel.analytics.insights import generate_insight
from content_intel.rag.explainer import explain_content
from content_intel.retrieval.search import SearchFilters, get_searcher
from content_intel.storage import load_joined_content


st.set_page_config(
    page_title="Content Intelligence RAG",
    page_icon="",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    df = load_joined_content()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


@st.cache_resource(show_spinner=False)
def searcher():
    return get_searcher()


def humanize_name(name: object) -> str:
    text = str(name).replace("_", " ").strip()
    replacements = {
        "id": "ID",
        "ocr": "OCR",
        "api": "API",
        "rag": "RAG",
    }
    words = [replacements.get(word.lower(), word.capitalize()) for word in text.split()]
    return " ".join(words)


def humanize_value(value: object) -> object:
    if isinstance(value, str):
        return humanize_name(value)
    return value


def display_frame(frame: pd.DataFrame) -> pd.DataFrame:
    display = frame.copy()
    for column in display.select_dtypes(include=["object", "string"]).columns:
        if column.endswith("label") or column in {
            "risk_label",
            "manual_label",
            "recommended_action",
            "review_status",
            "category",
        }:
            display[column] = display[column].map(humanize_value)
    display = display.rename(columns={column: humanize_name(column) for column in display.columns})
    return display


st.title("Multimodal Content Intelligence & RAG")
st.caption(
    "A lightweight demo for content risk monitoring, semantic retrieval, "
    "case-level moderation explanation, and business-facing insight generation."
)

tabs = st.tabs(["Overview", "Semantic Search", "Moderation Explorer", "Model Monitoring", "Insight"])
df = load_data()

with tabs[0]:
    st.info(
        "This page is the executive overview for a trust-and-safety or content operations team. "
        "It answers three quick questions: how much content is in the demo corpus, what share is "
        "currently predicted as risky, and whether risky content is changing over time. The left "
        "chart shows the model's risk-category mix; the right chart shows daily high-risk volume. "
        "The system is multimodal in a lightweight way: it combines post text, synthetic demo images, "
        "OCR-style text extracted from those images, image type signals, and engagement metadata. "
        "In a real platform, this is the first screen an operator or business stakeholder would use "
        "to spot unusual risk movement before drilling into search, case review, or model diagnostics."
    )
    total = len(df)
    high_risk_rate = (df["risk_label"] != "safe").mean()
    avg_reports = df["reports"].mean()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Content", f"{total:,}")
    c2.metric("High-risk Rate", f"{high_risk_rate:.1%}")
    c3.metric("Avg Reports", f"{avg_reports:.2f}")
    c4.metric("Risk Classes", df["risk_label"].nunique())

    left, right = st.columns(2)
    with left:
        st.subheader("Risk Label Distribution")
        risk_counts = (
            df["risk_label"]
            .map(humanize_value)
            .value_counts()
            .rename_axis("Risk Label")
            .reset_index(name="Count")
        )
        fig = px.bar(risk_counts, x="Risk Label", y="Count")
        fig.update_layout(xaxis_tickangle=0, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, width="stretch")
    with right:
        st.subheader("Daily High-risk Volume")
        trend = (
            df.assign(date=df["timestamp"].dt.date, high_risk=df["risk_label"] != "safe")
            .groupby("date")["high_risk"]
            .sum()
            .reset_index(name="High-risk Volume")
        )
        fig = px.line(trend, x="date", y="High-risk Volume")
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="High-risk Volume",
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig, width="stretch")

    st.subheader("Image Type Mix")
    image_mix = (
        df["image_type"]
        .map(humanize_value)
        .value_counts()
        .rename_axis("Image Type")
        .reset_index(name="Count")
    )
    fig = px.bar(image_mix, x="Image Type", y="Count")
    fig.update_layout(xaxis_tickangle=0, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, width="stretch")

with tabs[1]:
    st.subheader("Semantic Search")
    st.info(
        "This page demonstrates semantic retrieval, the retrieval part of RAG. Instead of filtering "
        "only by exact keywords, a reviewer can describe a risk pattern in plain English, such as "
        "`guaranteed return private contact investment`, and retrieve similar historical cases from "
        "the content corpus. The current MVP uses a lightweight TF-IDF vector index so it runs on a "
        "small CPU machine; the same interface can later be swapped to FAISS plus embedding models "
        "such as OpenAI, Qwen, BGE, or sentence-transformers."
    )
    query = st.text_input("Query", "guaranteed return private contact investment")
    raw_labels = sorted(df["risk_label"].dropna().unique().tolist())
    label_options = {"Any": None} | {humanize_name(label): label for label in raw_labels}
    selected_label_name = st.selectbox("Risk Filter", list(label_options.keys()))
    top_k = st.slider("Top K", 3, 12, 5)
    if st.button("Search", type="primary"):
        results = searcher().search(
            query,
            top_k=top_k,
            filters=SearchFilters(
                risk_label=label_options[selected_label_name]
            ),
        )
        st.dataframe(display_frame(pd.DataFrame(results)), width="stretch")

with tabs[2]:
    st.subheader("Moderation Explorer")
    st.info(
        "This is the case-level review workflow. Pick one content ID and the page shows the synthetic "
        "demo image, original post text, OCR-style text extracted from the image, image type, "
        "engagement/reporting metadata, model prediction, recommended action, evidence, policy snippets, "
        "and similar historical cases. "
        "The goal is not only to classify content, but to make the model output actionable for a "
        "human reviewer: what was detected, why it matters, how confident the system is, and what "
        "the reviewer should do next."
    )
    risky_first = df.sort_values(["risk_label", "risk_score"], ascending=[True, False])
    content_id = st.selectbox("Content ID", risky_first["content_id"].tolist())
    content = df[df["content_id"] == content_id].iloc[0]
    left, right = st.columns([1, 1])
    with left:
        image_path = ROOT / str(content["image_path"])
        st.caption("Image")
        if image_path.exists():
            st.image(str(image_path), width="stretch")
        else:
            st.warning(f"Image file not found: {content['image_path']}")
        st.caption("Post Text")
        st.write(content["post_text"])
        st.caption("OCR Text")
        st.write(content["ocr_text"])
        st.caption("Metadata")
        st.dataframe(
            display_frame(pd.DataFrame(
                    [
                        {
                            "category": content["category"],
                            "image_type": content["image_type"],
                            "image_text_density": content["image_text_density"],
                            "visual_risk_signal": content["visual_risk_signal"],
                            "views": content["views"],
                            "shares": content["shares"],
                            "reports": content["reports"],
                            "manual_label": content["manual_label"],
                            "review_status": content["review_status"],
                        }
                    ]
                )
            ),
            width="stretch",
        )
    with right:
        explanation = explain_content(content_id)
        st.metric("Predicted Risk", humanize_value(explanation.risk_label), f"{explanation.risk_score:.2f}")
        st.write("Recommended action:", humanize_value(explanation.recommended_action))
        st.write("Uncertainty:", humanize_value(explanation.uncertainty))
        st.write("Needs human review:", explanation.needs_human_review)
        st.caption("Evidence")
        for item in explanation.evidence:
            st.write(f"- {item}")
        st.caption("Policy Evidence")
        for item in explanation.policy_evidence:
            st.write(f"- {item}")
    st.caption("Similar Cases")
    st.dataframe(display_frame(pd.DataFrame([r.model_dump() for r in explanation.similar_cases])), width="stretch")

with tabs[3]:
    st.subheader("Model Monitoring")
    st.info(
        "This page explains whether the model is behaving reliably enough to support operations. "
        "Agreement compares the model prediction with the simulated manual review label. The corpus "
        "intentionally includes ambiguous and borderline human-review cases, so the score is high but "
        "not perfect, which is closer to real moderation work. The confusion table shows which classes "
        "are mixed up, and the disagreement samples are exactly the cases a data scientist or model "
        "owner would inspect before changing labels, thresholds, prompts, or training data."
    )
    agreement = (df["manual_label"] == df["risk_label"]).mean()
    c1, c2 = st.columns(2)
    c1.metric("Agreement", f"{agreement:.1%}")
    c2.metric("Low-confidence Cases", int((df["risk_score"] < 0.60).sum()))
    st.caption("Manual vs Predicted")
    confusion = pd.crosstab(df["manual_label"], df["risk_label"])
    confusion.index = [humanize_value(value) for value in confusion.index]
    confusion.columns = [humanize_value(value) for value in confusion.columns]
    confusion.index.name = "Manual Label"
    st.dataframe(confusion, width="stretch")
    st.caption("Disagreement Samples")
    st.dataframe(
        display_frame(
            df[df["manual_label"] != df["risk_label"]][
                [
                    "content_id",
                    "post_text",
                    "ocr_text",
                    "manual_label",
                    "risk_label",
                    "risk_score",
                    "review_status",
                ]
            ].head(50)
        ),
        width="stretch",
    )

with tabs[4]:
    st.subheader("Agentic Insight")
    st.info(
        "This page turns model outputs into a short business-facing answer. The demo implementation "
        "uses local Python analytics plus vector retrieval, and the same workflow is exposed through "
        "the local FastAPI endpoint `POST /analytics/insight`. No paid external LLM API is required "
        "for this MVP. In production, the final summary step could call OpenAI, Qwen, DeepSeek, or "
        "another LLM provider, while keeping the same guardrails: retrieve evidence first, compute "
        "supporting metrics, then generate a concise recommendation that business stakeholders can "
        "understand in a few minutes."
    )
    question = st.text_area(
        "Business question",
        "Are high-risk financial promotion posts increasing, and what cases should reviewers inspect?",
    )
    if st.button("Generate Insight", type="primary"):
        insight = generate_insight(question)
        st.write(insight.answer)
        st.caption("Supporting Metrics")
        metric_display = {humanize_name(key): humanize_value(value) for key, value in insight.supporting_metrics.items()}
        st.json(metric_display)
        st.caption("Representative Cases")
        st.dataframe(display_frame(pd.DataFrame([r.model_dump() for r in insight.representative_cases])), width="stretch")
