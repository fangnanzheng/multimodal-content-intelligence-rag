from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from content_intel.rag.explainer import explain_content
from content_intel.retrieval.search import get_searcher
from content_intel.storage import load_joined_content

OUT = ROOT / "docs" / "assets"
OUT.mkdir(parents=True, exist_ok=True)

BG = "#0e1117"
PANEL = "#171b24"
GRID = "#303743"
TEXT = "#f4f6f8"
MUTED = "#a7adb8"
ACCENT = "#7cc4f8"
RED = "#ff4b4b"
GREEN = "#2fbf71"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


F_TITLE = font(42, True)
F_H1 = font(30, True)
F_H2 = font(22, True)
F_BODY = font(18)
F_SMALL = font(15)
F_MONO = font(16)


def humanize(value: object) -> str:
    text = str(value).replace("_", " ")
    return " ".join(word.upper() if word.lower() in {"id", "ocr", "rag"} else word.capitalize() for word in text.split())


def canvas(title: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (1600, 950), BG)
    draw = ImageDraw.Draw(img)
    draw.text((56, 42), title, fill=TEXT, font=F_TITLE)
    return img, draw


def rounded(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], fill: str = PANEL, outline: str | None = None) -> None:
    draw.rounded_rectangle(box, radius=14, fill=fill, outline=outline or GRID, width=1)


def metric(draw: ImageDraw.ImageDraw, x: int, y: int, label: str, value: str) -> None:
    rounded(draw, (x, y, x + 320, y + 120))
    draw.text((x + 24, y + 22), label, fill=MUTED, font=F_BODY)
    draw.text((x + 24, y + 58), value, fill=TEXT, font=F_H1)


def bar_chart(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], title: str, items: list[tuple[str, int]]) -> None:
    x1, y1, x2, y2 = box
    rounded(draw, box)
    draw.text((x1 + 24, y1 + 20), title, fill=TEXT, font=F_H2)
    chart_x1, chart_y1 = x1 + 60, y1 + 80
    chart_x2, chart_y2 = x2 - 30, y2 - 70
    max_val = max(value for _, value in items) or 1
    bar_w = max(24, int((chart_x2 - chart_x1) / max(len(items) * 1.7, 1)))
    gap = int((chart_x2 - chart_x1 - bar_w * len(items)) / max(len(items), 1))
    for i in range(5):
        yy = chart_y2 - int((chart_y2 - chart_y1) * i / 4)
        draw.line((chart_x1, yy, chart_x2, yy), fill=GRID)
    for idx, (name, value) in enumerate(items):
        bx = chart_x1 + gap // 2 + idx * (bar_w + gap)
        bh = int((chart_y2 - chart_y1) * value / max_val)
        draw.rounded_rectangle((bx, chart_y2 - bh, bx + bar_w, chart_y2), radius=5, fill=ACCENT)
        draw.text((bx - 6, chart_y2 + 14), humanize(name), fill=MUTED, font=F_SMALL)


def line_chart(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], title: str, values: list[int]) -> None:
    x1, y1, x2, y2 = box
    rounded(draw, box)
    draw.text((x1 + 24, y1 + 20), title, fill=TEXT, font=F_H2)
    chart_x1, chart_y1 = x1 + 60, y1 + 80
    chart_x2, chart_y2 = x2 - 30, y2 - 50
    max_val = max(values) or 1
    points = []
    for idx, value in enumerate(values):
        x = chart_x1 + int((chart_x2 - chart_x1) * idx / max(len(values) - 1, 1))
        y = chart_y2 - int((chart_y2 - chart_y1) * value / max_val)
        points.append((x, y))
    for i in range(5):
        yy = chart_y2 - int((chart_y2 - chart_y1) * i / 4)
        draw.line((chart_x1, yy, chart_x2, yy), fill=GRID)
    draw.line(points, fill=ACCENT, width=4)
    for point in points[:: max(1, len(points) // 8)]:
        draw.ellipse((point[0] - 4, point[1] - 4, point[0] + 4, point[1] + 4), fill=ACCENT)


def table(draw: ImageDraw.ImageDraw, x: int, y: int, columns: list[str], rows: list[list[object]], widths: list[int]) -> None:
    row_h = 44
    total_w = sum(widths)
    rounded(draw, (x, y, x + total_w, y + row_h * (len(rows) + 1)), fill="#10141d")
    cx = x
    for col, width in zip(columns, widths, strict=True):
        draw.text((cx + 12, y + 12), col, fill=MUTED, font=F_SMALL)
        cx += width
    for ridx, row in enumerate(rows):
        yy = y + row_h * (ridx + 1)
        draw.line((x, yy, x + total_w, yy), fill=GRID)
        cx = x
        for value, width in zip(row, widths, strict=True):
            text = str(value)
            if len(text) > 48:
                text = text[:45] + "..."
            draw.text((cx + 12, yy + 12), text, fill=TEXT, font=F_SMALL)
            cx += width


def overview(df: pd.DataFrame) -> None:
    img, draw = canvas("Overview")
    draw.text((58, 100), "Executive risk monitoring across text, image-derived OCR, image type, and reporting signals.", fill=MUTED, font=F_BODY)
    metric(draw, 56, 150, "Content", f"{len(df):,}")
    metric(draw, 400, 150, "High-risk Rate", f"{(df.risk_label != 'safe').mean():.1%}")
    metric(draw, 744, 150, "Avg Reports", f"{df.reports.mean():.2f}")
    metric(draw, 1088, 150, "Image Types", f"{df.image_type.nunique()}")
    risk_items = list(df.risk_label.value_counts().items())
    image_items = list(df.image_type.value_counts().head(8).items())
    trend = df.assign(date=pd.to_datetime(df.timestamp).dt.date, high=df.risk_label != "safe").groupby("date").high.sum().tail(30).tolist()
    bar_chart(draw, (56, 310, 760, 650), "Risk Label Distribution", risk_items)
    line_chart(draw, (804, 310, 1540, 650), "Daily High-risk Volume", trend)
    bar_chart(draw, (56, 690, 1540, 910), "Image Type Mix", image_items)
    img.save(OUT / "overview.png")


def semantic_search() -> None:
    img, draw = canvas("Semantic Search")
    draw.text((58, 100), "Plain-English retrieval over post text, OCR-style image text, and image-derived signals.", fill=MUTED, font=F_BODY)
    query = "guaranteed return private contact investment"
    rounded(draw, (56, 150, 1540, 235))
    draw.text((82, 178), f"Query: {query}", fill=TEXT, font=F_H2)
    rows = get_searcher().search(query, top_k=6)
    table(
        draw,
        56,
        290,
        ["Content ID", "Similarity", "Risk Label", "Risk Score", "Image Type", "Post Text"],
        [[r["content_id"], r["similarity"], humanize(r["risk_label"]), r["risk_score"], humanize(r["image_type"]), r["post_text"]] for r in rows],
        [150, 140, 190, 140, 190, 750],
    )
    img.save(OUT / "semantic-search.png")


def moderation(df: pd.DataFrame) -> None:
    sample = df[df.risk_label != "safe"].sort_values("risk_score", ascending=False).iloc[0]
    explanation = explain_content(str(sample.content_id))
    img, draw = canvas("Moderation Explorer")
    draw.text((58, 100), "Case-level review with image preview, OCR text, metadata, model score, evidence, and similar cases.", fill=MUTED, font=F_BODY)
    image_path = ROOT / str(sample.image_path)
    if image_path.exists():
        preview = Image.open(image_path).resize((480, 270))
        img.paste(preview, (56, 155))
    rounded(draw, (56, 450, 760, 700))
    draw.text((82, 476), "Content Signals", fill=TEXT, font=F_H2)
    signals = [
        ("Post", sample.post_text),
        ("OCR", sample.ocr_text),
        ("Image Type", humanize(sample.image_type)),
        ("Visual Signal", humanize(sample.visual_risk_signal)),
    ]
    yy = 520
    for label, value in signals:
        draw.text((82, yy), f"{label}: ", fill=MUTED, font=F_BODY)
        draw.text((200, yy), str(value)[:66], fill=TEXT, font=F_BODY)
        yy += 40
    rounded(draw, (804, 155, 1540, 700))
    draw.text((832, 186), "Structured Explanation", fill=TEXT, font=F_H2)
    draw.text((832, 238), f"Predicted Risk: {humanize(explanation.risk_label)}", fill=TEXT, font=F_H1)
    draw.text((832, 286), f"Risk Score: {explanation.risk_score:.2f}", fill=GREEN if explanation.risk_score < 0.5 else RED, font=F_H2)
    draw.text((832, 334), f"Action: {humanize(explanation.recommended_action)}", fill=TEXT, font=F_BODY)
    draw.text((832, 384), "Evidence", fill=MUTED, font=F_BODY)
    yy = 424
    for item in explanation.evidence[:4]:
        draw.text((852, yy), f"- {item[:72]}", fill=TEXT, font=F_SMALL)
        yy += 34
    table(
        draw,
        56,
        740,
        ["Similar Case", "Similarity", "Risk", "OCR Text"],
        [[r.content_id, r.similarity, humanize(r.risk_label), r.ocr_text] for r in explanation.similar_cases],
        [170, 150, 190, 970],
    )
    img.save(OUT / "moderation-explorer.png")


def monitoring(df: pd.DataFrame) -> None:
    img, draw = canvas("Model Monitoring")
    agreement = (df.manual_label == df.risk_label).mean()
    disagreements = df[df.manual_label != df.risk_label].head(6)
    draw.text((58, 100), "Operational diagnostics for agreement, low-confidence cases, and human/model disagreement.", fill=MUTED, font=F_BODY)
    metric(draw, 56, 150, "Agreement", f"{agreement:.1%}")
    metric(draw, 400, 150, "Disagreements", f"{(df.manual_label != df.risk_label).sum():,}")
    metric(draw, 744, 150, "Low-confidence", f"{(df.risk_score < 0.60).sum():,}")
    cm = pd.crosstab(df.manual_label, df.risk_label)
    cm_rows = [[humanize(idx)] + [int(row.get(col, 0)) for col in cm.columns] for idx, row in cm.iterrows()]
    table(draw, 56, 310, ["Manual Label"] + [humanize(col) for col in cm.columns], cm_rows, [250] + [210] * len(cm.columns))
    table(
        draw,
        56,
        650,
        ["Content ID", "Manual Label", "Predicted Label", "Review Status", "Post Text"],
        [[row.content_id, humanize(row.manual_label), humanize(row.risk_label), humanize(row.review_status), row.post_text] for row in disagreements.itertuples()],
        [150, 200, 220, 220, 760],
    )
    img.save(OUT / "model-monitoring.png")


def main() -> None:
    df = load_joined_content()
    overview(df)
    semantic_search()
    moderation(df)
    monitoring(df)
    print(f"Rendered README assets to {OUT}")


if __name__ == "__main__":
    main()

