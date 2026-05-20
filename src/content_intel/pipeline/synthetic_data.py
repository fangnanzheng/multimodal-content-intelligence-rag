from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


LABELS = [
    "safe",
    "financial_scam",
    "spam",
    "counterfeit",
    "health_misinformation",
]


TEXT_TEMPLATES = {
    "safe": [
        "Weekend reading notes on product design and community operations.",
        "Sharing a normal delivery experience and a few photos from today's meal.",
        "A short review of a new productivity app with pros and cons.",
        "Discussion thread about local events, transportation, and family plans.",
    ],
    "financial_scam": [
        "Limited investment opportunity with guaranteed high return. Add private contact for details.",
        "Daily profit plan, no risk, principal protected, first group gets bonus.",
        "Crypto arbitrage signal with insider channel access. Contact now before slots close.",
        "Passive income course promises 30 percent weekly return and private payment link.",
    ],
    "spam": [
        "Flash coupon blast, click the same link and forward to five groups.",
        "Bulk promotion for followers, likes, and traffic packages. Message for price list.",
        "Repeated lucky draw campaign with referral code and external signup link.",
        "Mass discount post repeated across groups with contact harvesting form.",
    ],
    "counterfeit": [
        "Luxury watch clearance at 90 percent off, original factory source, off-platform payment.",
        "Branded sneakers direct from special channel, no receipt, private seller only.",
        "Premium cosmetics warehouse deal, suspiciously low price, limited stock.",
        "Electronics flagship model at impossible discount, pay deposit through private link.",
    ],
    "health_misinformation": [
        "Miracle herbal therapy cures chronic disease in seven days with no clinical visit.",
        "Secret supplement eliminates all symptoms immediately, guaranteed result.",
        "Unverified treatment plan claims to replace professional medical advice.",
        "Detox product promises permanent cure and asks users to stop medication.",
    ],
}


OCR_TEMPLATES = {
    "safe": [
        "menu schedule open hours",
        "book notes project plan",
        "travel photos receipt",
        "community event poster",
    ],
    "financial_scam": [
        "guaranteed return add contact limited slots",
        "principal protected high yield private group",
        "daily profit scan QR code",
        "investment bonus contact advisor",
    ],
    "spam": [
        "click link forward reward coupon",
        "traffic package followers likes",
        "limited coupon scan code",
        "external signup referral bonus",
    ],
    "counterfeit": [
        "factory source luxury clearance",
        "no receipt private payment",
        "brand goods lowest price",
        "deposit only limited stock",
    ],
    "health_misinformation": [
        "miracle cure guaranteed effect",
        "stop medication natural therapy",
        "secret formula no hospital",
        "permanent cure seven days",
    ],
}


IMAGE_TYPES = {
    "safe": ["normal_photo", "event_poster", "receipt_snapshot"],
    "financial_scam": ["promo_poster", "chat_screenshot", "qr_poster"],
    "spam": ["promo_poster", "link_screenshot", "coupon_banner"],
    "counterfeit": ["product_image", "promo_poster", "payment_screenshot"],
    "health_misinformation": ["promo_poster", "article_screenshot", "product_image"],
}


CATEGORIES = {
    "safe": ["lifestyle", "education", "local_news", "commerce"],
    "financial_scam": ["finance", "education", "community"],
    "spam": ["commerce", "marketing", "community"],
    "counterfeit": ["commerce", "fashion", "electronics"],
    "health_misinformation": ["health", "lifestyle"],
}


CONFUSION_LABELS = {
    "safe": ["spam", "counterfeit"],
    "financial_scam": ["spam", "safe"],
    "spam": ["financial_scam", "safe"],
    "counterfeit": ["spam", "safe"],
    "health_misinformation": ["safe", "spam"],
}


def _risk_weight(label: str) -> float:
    return {
        "safe": 0.08,
        "spam": 0.55,
        "counterfeit": 0.68,
        "health_misinformation": 0.72,
        "financial_scam": 0.82,
    }[label]


def _manual_review_label(true_label: str, rng: random.Random) -> tuple[str, str]:
    """Simulate imperfect human labels for a more realistic monitoring view."""
    roll = rng.random()
    if roll < 0.07:
        return rng.choice(CONFUSION_LABELS[true_label]), "ambiguous_review"
    if roll < 0.12:
        return true_label, "borderline_case"
    return true_label, "standard_review"


def _visual_risk_signal(label: str, image_type: str) -> str:
    if label == "safe":
        return "low visual risk"
    if "screenshot" in image_type:
        return "screenshot with embedded claims"
    if "poster" in image_type or "banner" in image_type:
        return "promotional image with embedded text"
    if image_type == "product_image":
        return "product image with commerce risk cues"
    return "image contains risk-related text"


def _render_demo_image(path: Path, image_type: str, label: str, ocr_text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    palette = {
        "safe": ("#f3f7fb", "#2f6f9f", "#1f2933"),
        "financial_scam": ("#fff3d9", "#c2410c", "#3f1d0b"),
        "spam": ("#eef2ff", "#7c3aed", "#271a45"),
        "counterfeit": ("#fef2f2", "#b91c1c", "#3b0a0a"),
        "health_misinformation": ("#ecfdf5", "#047857", "#052e22"),
    }
    bg, accent, text_color = palette[label]
    img = Image.new("RGB", (480, 270), bg)
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle((18, 18, 462, 252), radius=14, outline=accent, width=3)
    draw.rectangle((18, 18, 462, 74), fill=accent)
    title = image_type.replace("_", " ").title()
    draw.text((34, 36), title, fill="white")
    draw.text((34, 104), label.replace("_", " ").title(), fill=text_color)
    words = ocr_text.split()
    line_1 = " ".join(words[:5])
    line_2 = " ".join(words[5:10])
    draw.text((34, 146), line_1, fill=text_color)
    draw.text((34, 178), line_2, fill=text_color)
    draw.text((34, 222), "Synthetic demo image", fill=accent)
    img.save(path, format="JPEG", quality=82, optimize=True)


def generate_content(rows: int = 1200, seed: int = 42) -> pd.DataFrame:
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    weights = [0.46, 0.18, 0.16, 0.12, 0.08]
    start = datetime.now(UTC).replace(tzinfo=None) - timedelta(days=45)
    records: list[dict[str, object]] = []

    for i in range(rows):
        label = rng.choices(LABELS, weights=weights, k=1)[0]
        manual_label, review_status = _manual_review_label(label, rng)
        image_type = rng.choice(IMAGE_TYPES[label])
        ocr_text = rng.choice(OCR_TEMPLATES[label])
        image_path = Path(f"data/raw/images/img_{i + 1:05d}.jpg")
        _render_demo_image(image_path, image_type, label, ocr_text)
        risk = _risk_weight(label)
        timestamp = start + timedelta(minutes=int(np_rng.integers(0, 45 * 24 * 60)))
        views = int(np_rng.lognormal(mean=6.2, sigma=0.8))
        likes = int(max(0, np_rng.normal(views * (0.06 if label == "safe" else 0.03), 8)))
        shares = int(max(0, np_rng.normal(views * (0.018 + risk * 0.03), 5)))
        reports = int(max(0, np_rng.poisson(0.2 + risk * 5.2)))

        records.append(
            {
                "content_id": f"c{i + 1:05d}",
                "user_id": f"u{np_rng.integers(1, max(20, rows // 4)):05d}",
                "post_text": rng.choice(TEXT_TEMPLATES[label]),
                "ocr_text": ocr_text,
                "image_path": str(image_path).replace("\\", "/"),
                "image_type": image_type,
                "image_text_density": int(np_rng.integers(1, 5) if label == "safe" else np_rng.integers(5, 10)),
                "visual_risk_signal": _visual_risk_signal(label, image_type),
                "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "category": rng.choice(CATEGORIES[label]),
                "views": views,
                "likes": likes,
                "shares": shares,
                "reports": reports,
                "source_label": label,
                "manual_label": manual_label,
                "review_status": review_status,
            }
        )

    return pd.DataFrame.from_records(records)
