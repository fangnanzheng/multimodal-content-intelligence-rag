from __future__ import annotations

from pathlib import Path

from content_intel.config import load_config, load_yaml
from content_intel.paths import CONFIG_DIR
from content_intel.retrieval.search import get_searcher
from content_intel.schemas import ModerationExplanation, SearchResult
from content_intel.storage import get_content


KEYWORD_EVIDENCE = {
    "financial_scam": ["guaranteed", "return", "profit", "principal", "investment", "contact"],
    "spam": ["coupon", "forward", "followers", "traffic", "referral", "signup"],
    "counterfeit": ["luxury", "factory", "receipt", "deposit", "brand", "lowest"],
    "health_misinformation": ["miracle", "cure", "guaranteed", "therapy", "medication"],
}


def _humanize(value: str) -> str:
    return " ".join(word.capitalize() for word in value.replace("_", " ").split())


def _policy_for(label: str) -> list[str]:
    policy_file = load_yaml(CONFIG_DIR / "policy_snippets.yaml")
    return [
        f"{item['title']}: {item['text']}"
        for item in policy_file["policies"]
        if item["label"] in {label, "safe"}
    ][:2]


def _uncertainty(score: float) -> str:
    if score >= 0.78 or score <= 0.35:
        return "low"
    if score >= 0.55:
        return "medium"
    return "high"


def _keyword_hits(label: str, text: str) -> list[str]:
    text_lower = text.lower()
    return [kw for kw in KEYWORD_EVIDENCE.get(label, []) if kw in text_lower]


def explain_content(content_id: str) -> ModerationExplanation:
    content = get_content(content_id)
    if not content:
        raise KeyError(f"Unknown content_id: {content_id}")

    label = str(content["risk_label"])
    score = float(content["risk_score"])
    text = f"{content.get('post_text', '')}\n{content.get('ocr_text', '')}"
    hits = _keyword_hits(label, text)
    similar_cases = get_searcher().similar_to_content(content, top_k=3)
    policy_evidence = _policy_for(label)

    evidence = [
        f"Model predicted {_humanize(label)} with score {score:.2f}.",
        f"Image modality: {_humanize(str(content.get('image_type', 'unknown')))}; {content.get('visual_risk_signal', 'no visual signal')}.",
        f"OCR/post text matched risk terms: {', '.join(hits) if hits else 'no strong keyword hit'}.",
        f"Retrieved {len(similar_cases)} similar historical cases for grounding.",
    ]
    if int(content.get("reports", 0)) > 2:
        evidence.append(f"User reports are elevated: {content['reports']} reports.")

    uncertainty = _uncertainty(score)
    needs_review = label != "safe" or uncertainty != "low"
    if needs_review and score >= 0.75:
        action = "send_to_manual_review"
    elif label == "safe":
        action = "allow_with_monitoring" if uncertainty != "low" else "allow"
    else:
        action = "monitor_or_rate_limit"

    reasoning = (
        f"The content is classified as {_humanize(label)}. The explanation is grounded in "
        f"model confidence, image-derived signals, OCR/post-text signals, policy "
        f"snippets, and similar cases rather than free-form external assumptions."
    )

    return ModerationExplanation(
        content_id=content_id,
        risk_label=label,
        risk_score=score,
        uncertainty=uncertainty,  # type: ignore[arg-type]
        needs_human_review=needs_review,
        evidence=evidence,
        similar_cases=[SearchResult(**case) for case in similar_cases],
        policy_evidence=policy_evidence,
        recommended_action=action,
        reasoning=reasoning,
    )
