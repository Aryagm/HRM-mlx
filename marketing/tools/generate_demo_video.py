#!/usr/bin/env python3
"""Generate a side-by-side HRM-Text-1B inference speed comparison video."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "assets" / "demo-comparison.mp4"
CAPTURE_DIR = ROOT / "assets" / "captures"
FPS = 30
WIDTH, HEIGHT = 1920, 1080

BG = (13, 17, 23)
PANEL_BG = (22, 27, 34)
BORDER = (48, 54, 61)
PROMPT_BG = (30, 35, 44)
TEXT_DIM = (120, 130, 145)
TEXT_NORMAL = (200, 210, 220)
TEXT_BRIGHT = (240, 245, 250)
GREEN = (80, 220, 120)
BLUE = (95, 143, 240)
RED = (255, 80, 80)
YELLOW = (255, 220, 60)

FONT_MONO = "/System/Library/Fonts/Menlo.ttc"
FONT_HEADER = ImageFont.truetype(FONT_MONO, 18)
FONT_MODEL = ImageFont.truetype(FONT_MONO, 14)
FONT_TEXT = ImageFont.truetype(FONT_MONO, 15)
FONT_PROMPT = ImageFont.truetype(FONT_MONO, 14)
FONT_TITLE = ImageFont.truetype(FONT_MONO, 14)
FONT_TIMER = ImageFont.truetype(FONT_MONO, 15)
FONT_SPEED = ImageFont.truetype(FONT_MONO, 76)
FONT_TAG = ImageFont.truetype(FONT_MONO, 30)

PROMPT_TEXT = (
    "Differentiate f(x) = x^2 / ln(x). Write a detailed solution with quotient-rule "
    "setup, derivative substitution, simplification, a product-rule cross-check, "
    "and a domain note. Put the final derivative in boxed braces."
)


def load_capture(filename: str) -> tuple[str, float]:
    with (CAPTURE_DIR / filename).open() as f:
        data = json.load(f)
    return data["text"], float(data["tps"])


cpu_text, cpu_tps = load_capture("pytorch_cpu.json")
mps_text, mps_tps = load_capture("pytorch_mps.json")
hrm_text, hrm_tps = load_capture("hrm_mlx_fast.json")

FRAMEWORKS = [
    {
        "name": "PyTorch CPU",
        "tps": cpu_tps,
        "speed_color": RED,
        "text": cpu_text,
        "model_label": "HRM-Text-1B · FP32",
        "header_color": (60, 30, 30),
        "header_border": (120, 50, 50),
        "tag": None,
    },
    {
        "name": "PyTorch MPS",
        "tps": mps_tps,
        "speed_color": YELLOW,
        "text": mps_text,
        "model_label": "HRM-Text-1B · BF16",
        "header_color": (50, 45, 20),
        "header_border": (100, 90, 30),
        "tag": None,
    },
    {
        "name": "HRM-mlx",
        "tps": hrm_tps,
        "speed_color": GREEN,
        "text": hrm_text,
        "model_label": "MXFP4 + Metal SwiGLU",
        "header_color": (25, 50, 35),
        "header_border": (50, 120, 70),
        "tag": f"{hrm_tps / cpu_tps:.1f}x VS CPU",
    },
]

NUM_PANELS = len(FRAMEWORKS)
PANEL_W = (WIDTH - 20 * (NUM_PANELS + 1)) // NUM_PANELS
PROMPT_H = 72
PANEL_TOP = PROMPT_H + 18
PANEL_H = HEIGHT - PANEL_TOP - 12
HEADER_H = 52
TEXT_AREA_TOP = HEADER_H + 6


def text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont) -> int:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0]


def split_long_word(
    draw: ImageDraw.ImageDraw,
    word: str,
    font: ImageFont.FreeTypeFont,
    max_width: int,
) -> list[str]:
    chunks = []
    current = ""
    for char in word:
        candidate = current + char
        if not current or text_width(draw, candidate, font) <= max_width:
            current = candidate
        else:
            chunks.append(current)
            current = char
    if current:
        chunks.append(current)
    return chunks


def wrap_text(
    text: str,
    draw: ImageDraw.ImageDraw,
    font: ImageFont.FreeTypeFont,
    max_width: int,
) -> list[str]:
    lines = []
    for paragraph in text.split("\n"):
        if paragraph.strip() == "":
            lines.append("")
            continue

        current = ""
        for word in paragraph.split(" "):
            candidate = word if not current else f"{current} {word}"
            if text_width(draw, candidate, font) <= max_width:
                current = candidate
                continue

            if current:
                lines.append(current)
                current = ""

            if text_width(draw, word, font) <= max_width:
                current = word
                continue

            pieces = split_long_word(draw, word, font, max_width)
            lines.extend(pieces[:-1])
            current = pieces[-1] if pieces else ""

        lines.append(current)
    return lines


WRAP_DRAW = ImageDraw.Draw(Image.new("RGB", (1, 1), BG))
PROMPT_LINES = wrap_text(PROMPT_TEXT, WRAP_DRAW, FONT_PROMPT, WIDTH - 48)
for framework in FRAMEWORKS:
    framework["wrapped_lines"] = wrap_text(
        framework["text"], WRAP_DRAW, FONT_TEXT, PANEL_W - 24
    )
    total_tokens = max(1, len(framework["text"]) / 4)
    framework["finish_time"] = 0.5 + total_tokens / framework["tps"]

DURATION_S = int(max(framework["finish_time"] for framework in FRAMEWORKS) + 5)


def draw_frame(t: float) -> np.ndarray:
    img = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(img)

    draw.rounded_rectangle(
        [10, 8, WIDTH - 10, PROMPT_H + 8],
        radius=6,
        fill=PROMPT_BG,
        outline=BORDER,
    )
    draw.text((24, 12), "Input Prompt", fill=TEXT_DIM, font=FONT_TITLE)
    for i, line in enumerate(PROMPT_LINES[:3]):
        draw.text((24, 30 + i * 18), line, fill=TEXT_NORMAL, font=FONT_PROMPT)

    for i, framework in enumerate(FRAMEWORKS):
        x = 20 + i * (PANEL_W + 20)
        y = PANEL_TOP

        draw.rounded_rectangle(
            [x, y, x + PANEL_W, y + PANEL_H],
            radius=6,
            fill=PANEL_BG,
            outline=BORDER,
        )

        tps = framework["tps"]
        gen_t = max(0, t - 0.5)
        total_chars = int(gen_t * tps * 4)
        finished = total_chars >= len(framework["text"])

        draw.rounded_rectangle(
            [x + 1, y + 1, x + PANEL_W - 1, y + HEADER_H],
            radius=5,
            fill=framework["header_color"],
        )
        draw.line(
            [(x + 6, y + 1), (x + PANEL_W - 6, y + 1)],
            fill=framework["header_border"],
            width=2,
        )

        draw.text((x + 12, y + 8), framework["name"], fill=TEXT_BRIGHT, font=FONT_HEADER)
        draw.text((x + 12, y + 30), framework["model_label"], fill=TEXT_DIM, font=FONT_MODEL)

        if gen_t > 0:
            elapsed = min(gen_t, framework["finish_time"] - 0.5) if finished else gen_t
            timer = f"({elapsed:.1f}s)"
            timer_w = text_width(draw, timer, FONT_TIMER)
            draw.text(
                (x + PANEL_W - timer_w - 12, y + 18),
                timer,
                fill=GREEN if finished else TEXT_DIM,
                font=FONT_TIMER,
            )

        all_lines = framework["wrapped_lines"]
        max_lines = (PANEL_H - TEXT_AREA_TOP - 10) // 19
        char_count = 0
        total_lines_generated = 0
        for line in all_lines:
            line_chars = len(line) + 1
            if char_count + line_chars > total_chars:
                break
            char_count += line_chars
            total_lines_generated += 1

        page = total_lines_generated // max_lines
        line_on_page = total_lines_generated % max_lines
        visible_lines = all_lines[page * max_lines : page * max_lines + line_on_page]

        text_x = x + 12
        text_y = y + TEXT_AREA_TOP + 4
        for line in visible_lines:
            if text_y > y + PANEL_H - 10:
                break
            draw.text((text_x, text_y), line, fill=TEXT_NORMAL, font=FONT_TEXT)
            text_y += 19

        if not finished and visible_lines and int(t * 3) % 2 == 0:
            cursor_x = min(
                text_x + text_width(draw, visible_lines[-1], FONT_TEXT),
                x + PANEL_W - 21,
            )
            cursor_y = min(text_y - 19, y + PANEL_H - 20)
            draw.rectangle([cursor_x, cursor_y, cursor_x + 9, cursor_y + 17], fill=TEXT_BRIGHT)

        speed_color = framework["speed_color"]
        tps_str = f"{tps:.1f} TOK/S"
        tag_text = framework["tag"]
        if finished:
            tag_text = f"DONE | {tag_text}" if tag_text else "DONE"

        tps_bbox = draw.textbbox((0, 0), tps_str, font=FONT_SPEED)
        tps_w = tps_bbox[2] - tps_bbox[0]
        tps_h = tps_bbox[3] - tps_bbox[1]
        tag_w = tag_h = 0
        if tag_text:
            tag_bbox = draw.textbbox((0, 0), tag_text, font=FONT_TAG)
            tag_w = tag_bbox[2] - tag_bbox[0]
            tag_h = tag_bbox[3] - tag_bbox[1]

        gap = 14 if tag_text else 0
        content_w = max(tps_w, tag_w)
        total_h = tps_h + gap + tag_h
        center_x = x + PANEL_W // 2
        center_y = HEIGHT // 2
        tps_x = center_x - tps_w // 2
        tps_y = center_y - total_h // 2

        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        overlay_draw.rounded_rectangle(
            [
                center_x - content_w // 2 - 20,
                tps_y - 14,
                center_x + content_w // 2 + 20,
                tps_y + total_h + 14,
            ],
            radius=10,
            fill=(4, 6, 10, 235),
        )
        img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
        draw = ImageDraw.Draw(img)
        draw.text((tps_x, tps_y), tps_str, fill=speed_color, font=FONT_SPEED)
        if tag_text:
            draw.text(
                (center_x - tag_w // 2, tps_y + tps_h + gap),
                tag_text,
                fill=speed_color,
                font=FONT_TAG,
            )

    return np.array(img)[:, :, ::-1]


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(OUTPUT),
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        (WIDTH, HEIGHT),
    )

    for frame_idx in range(FPS * DURATION_S):
        t = frame_idx / FPS
        writer.write(draw_frame(t))
        if frame_idx % FPS == 0:
            print(f"  {int(t)}s / {DURATION_S}s")

    final_frame = draw_frame(DURATION_S)
    for _ in range(FPS * 2):
        writer.write(final_frame)

    writer.release()
    print(f"Saved to {OUTPUT}")


if __name__ == "__main__":
    main()
