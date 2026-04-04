import html
import math
import re
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw

from dataset.Flickr8kCaptionDataset import Flickr8kCaptionDataset


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
IMAGE_DIR = DATA_DIR / "Flicker8k_Dataset"
TOKEN_PATH = DATA_DIR / "Flickr8k.token.txt"
TRAIN_SPLIT_PATH = DATA_DIR / "Flickr_8k.trainImages.txt"
VAL_SPLIT_PATH = DATA_DIR / "Flickr_8k.devImages.txt"
TEST_SPLIT_PATH = DATA_DIR / "Flickr_8k.testImages.txt"
OUTPUT_DIR = ROOT_DIR / "report_visuals"


def _escape(text: str) -> str:
    return html.escape(text, quote=True)


def _load_split_names(path: Path) -> list[str]:
    return sorted(Flickr8kCaptionDataset.load_image_names(str(path)) or [])


def _load_grouped_captions() -> dict[str, list[str]]:
    return Flickr8kCaptionDataset.load_grouped_captions(str(TOKEN_PATH))


def _tokenize_caption(text: str) -> list[str]:
    return re.findall(r"[a-z']+", text.lower())


def _svg_canvas(width: int, height: int, body: list[str]) -> str:
    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="#ffffff"/>',
            *body,
            "</svg>",
        ]
    )


def _write_text(x: float, y: float, value: str, size: int = 14, weight: str = "normal", anchor: str = "middle") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, sans-serif" font-size="{size}" '
        f'font-weight="{weight}" text-anchor="{anchor}" fill="#222">{_escape(value)}</text>'
    )


def _render_bar_chart(title: str, labels: list[str], values: list[int], output_path: Path, colors: list[str]) -> None:
    width = 920
    height = 540
    left = 90
    right = 40
    top = 80
    bottom = 90
    chart_width = width - left - right
    chart_height = height - top - bottom
    max_value = max(values) if values else 1
    gap = 24
    bar_width = (chart_width - gap * (len(values) - 1)) / max(1, len(values))
    body: list[str] = [
        _write_text(width / 2, 36, title, size=24, weight="bold"),
        f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#888" stroke-width="2"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#888" stroke-width="2"/>',
    ]

    for i, value in enumerate(values):
        x = left + i * (bar_width + gap)
        y = top + (1 - (value / max_value)) * chart_height
        bar_height = (value / max_value) * chart_height
        body.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" fill="{colors[i % len(colors)]}"/>'
        )
        body.append(_write_text(x + bar_width / 2, height - bottom + 28, labels[i], size=16))
        body.append(_write_text(x + bar_width / 2, y - 10, f"{value}", size=14, weight="bold"))

    tick_count = 6
    for tick in range(tick_count):
        ratio = tick / (tick_count - 1)
        y = height - bottom - ratio * chart_height
        tick_value = round(max_value * ratio)
        body.append(f'<line x1="{left - 6}" y1="{y:.1f}" x2="{left}" y2="{y:.1f}" stroke="#888" stroke-width="2"/>')
        body.append(_write_text(left - 12, y + 5, str(tick_value), size=14, anchor="end"))

    output_path.write_text(_svg_canvas(width, height, body), encoding="utf-8")


def _render_histogram(title: str, subtitle: str, values: list[int], output_path: Path, bucket_size: int = 2) -> None:
    width = 1120
    height = 560
    left = 80
    right = 40
    top = 70
    bottom = 100
    chart_width = width - left - right
    chart_height = height - top - bottom
    max_value = max(values) if values else 1
    bucket_max = ((max_value // bucket_size) + 1) * bucket_size
    buckets: list[tuple[str, int]] = []
    for start in range(0, bucket_max, bucket_size):
        end = start + bucket_size - 1
        count = sum(1 for value in values if start <= value <= end)
        buckets.append((f"{start}-{end}", count))

    y_max = max((count for _, count in buckets), default=1)
    bar_width = chart_width / max(1, len(buckets))
    body: list[str] = [
        _write_text(width / 2, 36, title, size=24, weight="bold"),
        _write_text(width / 2, 58, subtitle, size=14),
        f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#888" stroke-width="2"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#888" stroke-width="2"/>',
    ]

    for idx, (label, count) in enumerate(buckets):
        x = left + idx * bar_width + 1
        h = 0 if y_max == 0 else (count / y_max) * chart_height
        y = height - bottom - h
        body.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(bar_width - 2, 1):.1f}" height="{h:.1f}" fill="#4f81bd"/>')
        if idx % 2 == 0:
            body.append(_write_text(x + bar_width / 2, height - bottom + 22, label, size=11))

    tick_values = [0, y_max // 2, y_max]
    for tick_value in tick_values:
        y = height - bottom - (0 if y_max == 0 else (tick_value / y_max) * chart_height)
        body.append(f'<line x1="{left - 6}" y1="{y:.1f}" x2="{left}" y2="{y:.1f}" stroke="#888" stroke-width="2"/>')
        body.append(_write_text(left - 10, y + 4, str(tick_value), size=12, anchor="end"))

    mean_value = sum(values) / max(1, len(values))
    body.append(_write_text(width - 240, top + 24, f"mean={mean_value:.2f}", size=15, weight="bold", anchor="start"))
    body.append(_write_text(width - 240, top + 47, f"max={max_value}", size=15, anchor="start"))
    body.append(_write_text(width - 240, top + 70, f"samples={len(values)}", size=15, anchor="start"))
    output_path.write_text(_svg_canvas(width, height, body), encoding="utf-8")


def _render_top_words_chart(word_counts: Counter[str], output_path: Path, limit: int = 12) -> None:
    top_words = word_counts.most_common(limit)
    labels = [word for word, _ in top_words]
    values = [count for _, count in top_words]
    _render_bar_chart(
        "Top Caption Words in Flickr8k",
        labels,
        values,
        output_path,
        ["#4f81bd", "#9bbb59", "#8064a2", "#c0504d", "#f79646", "#4bacc6"],
    )


def _wrap_text(text: str, width: int = 28) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    current_len = 0
    for word in words:
        extra = len(word) + (1 if current else 0)
        if current and current_len + extra > width:
            lines.append(" ".join(current))
            current = [word]
            current_len = len(word)
        else:
            current.append(word)
            current_len += extra
    if current:
        lines.append(" ".join(current))
    return lines[:2]


def _render_sample_grid(image_names: list[str], grouped_captions: dict[str, list[str]], output_path: Path) -> None:
    columns = 3
    rows = math.ceil(len(image_names) / columns)
    image_w = 256
    image_h = 256
    card_h = 330
    margin = 24
    width = columns * image_w + (columns + 1) * margin
    height = rows * card_h + (rows + 1) * margin + 20
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    for idx, image_name in enumerate(image_names):
        row = idx // columns
        col = idx % columns
        x = margin + col * (image_w + margin)
        y = margin + row * card_h
        image_path = IMAGE_DIR / image_name
        with Image.open(image_path).convert("RGB") as image:
            image.thumbnail((image_w, image_h))
            image_x = x + (image_w - image.width) // 2
            image_y = y + (image_h - image.height) // 2
            canvas.paste(image, (image_x, image_y))

        draw.rectangle([x, y, x + image_w, y + image_h], outline="#cccccc", width=2)
        draw.text((x, y + image_h + 10), image_name, fill="#222")
        caption = grouped_captions.get(image_name, [""])[0]
        caption_lines = _wrap_text(caption)
        for line_idx, line in enumerate(caption_lines):
            draw.text((x, y + image_h + 34 + line_idx * 18), line, fill="#444")

    canvas.save(output_path)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_names = _load_split_names(TRAIN_SPLIT_PATH)
    val_names = _load_split_names(VAL_SPLIT_PATH)
    test_names = _load_split_names(TEST_SPLIT_PATH)
    grouped_captions = _load_grouped_captions()

    all_captions = [caption for captions in grouped_captions.values() for caption in captions]
    caption_lengths = [len(caption.split()) for caption in all_captions]
    word_counts = Counter(
        token
        for caption in all_captions
        for token in _tokenize_caption(caption)
        if token not in {"a", "an", "the", "in", "on", "of", "and", "to", "is", "are"}
    )

    _render_bar_chart(
        "Flickr8k Split Sizes",
        ["train", "dev", "test"],
        [len(train_names), len(val_names), len(test_names)],
        OUTPUT_DIR / "split_sizes.svg",
        ["#4f81bd", "#9bbb59", "#c0504d"],
    )
    _render_histogram(
        "Caption Length Distribution",
        "Word count per caption",
        caption_lengths,
        OUTPUT_DIR / "caption_length_histogram.svg",
        bucket_size=2,
    )
    _render_top_words_chart(word_counts, OUTPUT_DIR / "top_caption_words.svg")
    _render_sample_grid(train_names[:6], grouped_captions, OUTPUT_DIR / "sample_images_grid.png")


if __name__ == "__main__":
    main()
