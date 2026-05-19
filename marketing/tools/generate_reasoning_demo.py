#!/usr/bin/env python3
"""Generate the functional-equation reasoning image for the README."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT.parent / "assets" / "reasoning-demo.png"

WIDTH, HEIGHT = 1400, 900
BG = (13, 17, 23)
PANEL = (22, 27, 34)
BORDER = (48, 54, 61)
TEXT = (226, 232, 240)
TEXT_DIM = (137, 148, 164)
GREEN = (80, 220, 120)
BLUE = (95, 143, 240)

FONT_MONO = "/System/Library/Fonts/Menlo.ttc"
FONT_TITLE = ImageFont.truetype(FONT_MONO, 34)
FONT_LABEL = ImageFont.truetype(FONT_MONO, 21)
FONT_TEXT = ImageFont.truetype(FONT_MONO, 24)
FONT_SMALL = ImageFont.truetype(FONT_MONO, 20)

PROMPT = (
    "The function f satisfies the functional equation f(x) + f(y) = f(x + y - xy) "
    "for all real numbers x and y. If f(1) = 1, then find all integers n such that "
    "f(n) = n. Enter all your integers, separated by commas. Please reason step by "
    "step, and put your final answer within boxed braces."
)


def draw_panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], title: str) -> None:
    draw.rounded_rectangle(box, radius=8, fill=PANEL, outline=BORDER, width=2)
    draw.text((box[0] + 28, box[1] + 22), title, fill=TEXT_DIM, font=FONT_LABEL)


def text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont) -> int:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0]


def wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont,
    max_width: int,
) -> list[str]:
    lines = []
    current = ""
    for word in text.split(" "):
        candidate = word if not current else f"{current} {word}"
        if text_width(draw, candidate, font) <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def draw_wrapped(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.FreeTypeFont,
    fill: tuple[int, int, int],
    max_width: int,
    line_height: int,
) -> int:
    x, y = xy
    for line in wrap_text(draw, text, font, max_width):
        draw.text((x, y), line, fill=fill, font=font)
        y += line_height
    return y


def main() -> None:
    img = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(img)

    draw.text((50, 38), "HRM-mlx reasoning check", fill=TEXT, font=FONT_TITLE)
    draw.text(
        (50, 84),
        "Same prompt as the dflash-mlx comparison video",
        fill=TEXT_DIM,
        font=FONT_SMALL,
    )

    draw_panel(draw, (50, 140, WIDTH - 50, 345), "Prompt")
    draw_wrapped(draw, (84, 192), PROMPT, FONT_TEXT, TEXT, WIDTH - 168, 34)

    draw_panel(draw, (50, 380, WIDTH - 50, 805), "Visible model reasoning")
    reasoning = [
        "Set x = 1 and y = 1 in f(x) + f(y) = f(x + y - xy).",
        "This gives f(1) + f(1) = f(1 + 1 - 1) = f(1).",
        "Using f(1) = 1, the equation becomes 2 = 1.",
        "So the stated assumptions are inconsistent; no such function exists.",
    ]
    y = 437
    for step in reasoning:
        y = draw_wrapped(draw, (84, y), step, FONT_SMALL, TEXT_DIM, WIDTH - 168, 28)
        y += 18

    draw.text(
        (84, 722),
        "Final: no such function exists",
        fill=GREEN,
        font=FONT_TEXT,
    )

    badge_text = "contradiction detected"
    bbox = draw.textbbox((0, 0), badge_text, font=FONT_SMALL)
    badge_w = bbox[2] - bbox[0] + 36
    draw.rounded_rectangle(
        (WIDTH - 50 - badge_w, 36, WIDTH - 50, 76),
        radius=8,
        fill=(20, 55, 38),
        outline=(45, 120, 70),
    )
    draw.text((WIDTH - 50 - badge_w + 18, 47), badge_text, fill=BLUE, font=FONT_SMALL)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    img.save(OUTPUT)
    print(f"Saved to {OUTPUT}")


if __name__ == "__main__":
    main()
