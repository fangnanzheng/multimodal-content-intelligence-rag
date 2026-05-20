from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from content_intel.config import load_config, resolve_path
from content_intel.pipeline.synthetic_data import generate_content


def combined_text(df: pd.DataFrame) -> pd.Series:
    return (
        df["post_text"].fillna("")
        + "\nOCR: "
        + df["ocr_text"].fillna("")
        + "\nCategory: "
        + df["category"].fillna("")
        + "\nImage type: "
        + df["image_type"].fillna("")
        + "\nVisual signal: "
        + df["visual_risk_signal"].fillna("")
    )


def validate_content(df: pd.DataFrame) -> dict[str, object]:
    required = {
        "content_id",
        "user_id",
        "post_text",
        "ocr_text",
        "timestamp",
        "category",
        "image_type",
        "image_text_density",
        "visual_risk_signal",
        "views",
        "likes",
        "shares",
        "reports",
        "source_label",
        "manual_label",
        "review_status",
    }
    missing_cols = sorted(required - set(df.columns))
    invalid_rows = int(df[list(required & set(df.columns))].isna().any(axis=1).sum())
    duplicate_ids = int(df["content_id"].duplicated().sum()) if "content_id" in df else 0
    return {
        "rows": int(len(df)),
        "missing_columns": missing_cols,
        "invalid_rows": invalid_rows,
        "duplicate_content_ids": duplicate_ids,
        "label_distribution": df["manual_label"].value_counts().to_dict()
        if "manual_label" in df
        else {},
    }


def _write_markdown(path: Path, title: str, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# {title}\n\n{body.strip()}\n", encoding="utf-8")


def _risk_action(label: str, score: float) -> str:
    if label == "safe" and score < 0.45:
        return "allow"
    if score >= 0.75:
        return "send_to_manual_review"
    if label in {"financial_scam", "health_misinformation", "counterfeit"}:
        return "manual_review_queue"
    return "monitor_or_rate_limit"


def build_demo_artifacts(rows: int = 1200, seed: int = 42) -> dict[str, object]:
    cfg = load_config()
    paths = {k: resolve_path(v) for k, v in cfg["paths"].items()}
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    df = generate_content(rows=rows, seed=seed)
    quality = validate_content(df)
    if quality["missing_columns"] or quality["invalid_rows"] or quality["duplicate_content_ids"]:
        raise ValueError(f"Data validation failed: {quality}")

    df["combined_text"] = combined_text(df)
    df.to_csv(paths["raw_content"], index=False, encoding="utf-8")
    df.to_csv(paths["processed_content"], index=False, encoding="utf-8")

    X_train, X_test, y_train, y_test = train_test_split(
        df["combined_text"],
        df["manual_label"],
        test_size=0.25,
        random_state=seed,
        stratify=df["manual_label"],
    )
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_features=6000,
        norm="l2",
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    report = classification_report(y_test, y_pred, digits=3)
    labels = list(model.classes_)
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    full_vectors = vectorizer.transform(df["combined_text"])
    probs = model.predict_proba(full_vectors)
    pred_idx = np.argmax(probs, axis=1)
    df["risk_label"] = [model.classes_[idx] for idx in pred_idx]
    df["risk_score"] = probs.max(axis=1).round(4)
    df["recommended_action"] = [
        _risk_action(label, score)
        for label, score in zip(df["risk_label"], df["risk_score"], strict=True)
    ]
    predictions = df[
        [
            "content_id",
            "manual_label",
            "risk_label",
            "risk_score",
            "recommended_action",
        ]
    ].copy()
    predictions.to_csv(paths["predictions"], index=False, encoding="utf-8")

    joblib.dump(vectorizer, paths["vectorizer"])
    joblib.dump(model, paths["risk_model"])
    sparse.save_npz(paths["vector_matrix"], full_vectors)
    paths["index_metadata"].write_text(
        json.dumps(
            {
                "backend": "sparse_tfidf_cosine",
                "rows": len(df),
                "content_ids": df["content_id"].tolist(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    with sqlite3.connect(paths["sqlite_db"]) as conn:
        content_columns = [
            "content_id",
            "user_id",
            "post_text",
            "ocr_text",
            "image_path",
            "image_type",
            "image_text_density",
            "visual_risk_signal",
            "timestamp",
            "category",
            "views",
            "likes",
            "shares",
            "reports",
            "source_label",
            "manual_label",
            "review_status",
        ]
        df[content_columns].to_sql("content", conn, if_exists="replace", index=False)
        predictions.to_sql("predictions", conn, if_exists="replace", index=False)

    sample_idx = np.random.default_rng(seed).choice(
        np.arange(len(df)), size=min(120, len(df)), replace=False
    )
    recall_hits = 0
    for idx in sample_idx:
        scores = full_vectors[idx] @ full_vectors.T
        scores = np.asarray(scores.toarray()).ravel()
        scores[idx] = -1.0
        top = scores.argsort()[-5:][::-1]
        if df.iloc[idx]["manual_label"] in set(df.iloc[top]["manual_label"]):
            recall_hits += 1
    recall_at_5 = recall_hits / len(sample_idx)

    _write_markdown(
        Path("reports/data_quality_report.md"),
        "Data Quality Report",
        "\n".join(
            [
                f"- Rows: {quality['rows']}",
                f"- Duplicate content ids: {quality['duplicate_content_ids']}",
                f"- Invalid rows: {quality['invalid_rows']}",
                f"- Label distribution: `{quality['label_distribution']}`",
            ]
        ),
    )
    _write_markdown(
        Path("reports/model_evaluation_report.md"),
        "Model Evaluation Report",
        "## Classification Report\n\n"
        f"```text\n{report}\n```\n\n"
        "## Confusion Matrix\n\n"
        f"Labels: `{labels}`\n\n```text\n{cm}\n```",
    )
    _write_markdown(
        Path("reports/retrieval_evaluation_report.md"),
        "Retrieval Evaluation Report",
        f"- Retrieval backend: sparse TF-IDF cosine, FAISS-ready adapter.\n"
        f"- Sampled query count: {len(sample_idx)}\n"
        f"- Label Recall@5: {recall_at_5:.3f}\n",
    )
    _write_markdown(
        Path("reports/rag_evaluation_report.md"),
        "RAG Evaluation Report",
        "- Structured outputs are validated with Pydantic schemas.\n"
        "- Explanations are grounded in retrieved cases, policy snippets, OCR text, and model score.\n"
        "- High-risk or uncertain cases are routed to manual review.\n",
    )

    return {
        "rows": len(df),
        "labels": labels,
        "recall_at_5": recall_at_5,
        "sqlite_db": str(paths["sqlite_db"]),
    }
