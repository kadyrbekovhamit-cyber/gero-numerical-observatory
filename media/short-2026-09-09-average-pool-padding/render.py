from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import json
import math
import subprocess
import textwrap

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output"
ASSETS = ROOT / "assets"
OUT.mkdir(exist_ok=True)
ASSETS.mkdir(exist_ok=True)

W, H = 1080, 1920
INK = "#102f2b"
CREAM = "#f4f1e7"
MINT = "#9ce0bf"
AMBER = "#f3b45e"
RED = "#d86f5a"
MUTED = "#78958b"
FONT = "/System/Library/Fonts/Helvetica.ttc"


def font(size, bold=False):
    return ImageFont.truetype(FONT, size, index=1 if bold else 0)


def run(args):
    return subprocess.run(args, check=True, capture_output=True, text=True)


def duration(path):
    return float(
        run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ]
        ).stdout
    )


SCENES = [
    {
        "label": "A NATIVE C++ COUNTEREXAMPLE",
        "title": "Forward right.\nBackward wrong.",
        "speech": "Can average pooling be right in forward and wrong in backward? In Samsung's N N trainer, one asymmetric SAME-padding case did exactly that.",
        "caption": "AveragePool · padding=same\nFP32 · NCHW · CPU",
        "source": "S1–S2",
        "kind": "hook",
    },
    {
        "label": "THE FORWARD PASS",
        "title": "2.5  3\n3.5  4",
        "speech": "For input one, two, three, four, the forward result was correct: two point five, three, three point five, four.",
        "caption": "input [[1,2],[3,4]]\nwindow 2×2 · stride 1",
        "source": "S2–S3",
        "kind": "forward",
    },
    {
        "label": "THE TRANSPOSE JACOBIAN",
        "title": "75% of the\ngradient sum lost",
        "speech": "But with unit output gradients, nntrainer returned four quarters. The Jacobian and finite differences require zero point two five, zero point seven five, zero point seven five, two point two five.",
        "caption": "observed sum: 1\nrequired sum: 4",
        "source": "S2–S3",
        "kind": "gradient",
    },
    {
        "label": "TWO BOUNDS, ONE REPAIR",
        "title": "top,left\nTO bottom,right",
        "speech": "The backward loop used top and left padding in its end bounds. Replacing them with bottom and right made all fifty-eight focused tests pass.",
        "caption": "before: 50/58 pass\nafter: 58/58 pass",
        "source": "S1–S3",
        "kind": "repair",
    },
    {
        "label": "THE NOVELTY BOUNDARY",
        "title": "Confirmed now.\nNot claimed first.",
        "speech": "A 2021 pull request already contained the correct boundaries, so I am not claiming first discovery. The complete reproducer and limits are linked.",
        "caption": "PR #1360 disclosed\nEvidence: GitHub + gero.uz",
        "source": "S2–S4",
        "kind": "boundary",
    },
]

(ROOT / "storyboard.json").write_text(
    json.dumps(SCENES, indent=2) + "\n", encoding="utf-8"
)


def lines(draw, text, x, y, size, fill, bold=False, gap=12):
    for line in text.split("\n"):
        assert draw.textlength(line, font=font(size, bold)) <= W - x - 72
        draw.text((x, y), line, font=font(size, bold), fill=fill)
        y += size + gap
    return y


def matrix(draw, values, x, y, cell=170, fill=MINT, text_fill=INK):
    for i, value in enumerate(values):
        row, col = divmod(i, 2)
        left = x + col * (cell + 18)
        top = y + row * (cell + 18)
        draw.rounded_rectangle((left, top, left + cell, top + cell), 24, fill=fill)
        label = str(value)
        bbox = draw.textbbox((0, 0), label, font=font(49, True))
        draw.text(
            (left + (cell - (bbox[2] - bbox[0])) / 2, top + 54),
            label,
            font=font(49, True),
            fill=text_fill,
        )


parts = []
timings = []
total = 0.0
for index, scene in enumerate(SCENES, 1):
    dark = index in (1, 3, 5)
    bg, fg = (INK, CREAM) if dark else (CREAM, INK)
    image = Image.new("RGB", (W, H), bg)
    draw = ImageDraw.Draw(image)

    draw.rounded_rectangle((72, 82, 154, 164), 20, fill=MINT if dark else INK)
    draw.text((93, 87), "g", font=font(60, True), fill=INK if dark else CREAM)
    draw.text((178, 105), "GERO / TECHNOLOGY PRODUCT", font=font(28, True), fill=fg)
    draw.line((72, 215, 1008, 215), fill=MUTED, width=2)
    lines(draw, scene["label"], 72, 282, 28, MINT if dark else INK, True)
    title_size = 91 if index != 3 else 80
    lines(draw, scene["title"], 72, 365, title_size, fg, True, 15)

    if scene["kind"] == "hook":
        matrix(draw, [1, 2, 3, 4], 170, 770, fill=CREAM, text_fill=INK)
        draw.text((454, 1148), "VS", font=font(62, True), fill=MINT)
        lines(draw, "same forward\ndifferent derivative", 195, 1250, 48, fg, True, 20)
    elif scene["kind"] == "forward":
        matrix(draw, [1, 2, 3, 4], 120, 760, cell=150, fill="#dde7dc")
        draw.text((508, 895), "TO", font=font(45, True), fill=INK)
        matrix(draw, [2.5, 3, 3.5, 4], 660, 760, cell=150, fill=MINT)
        lines(draw, "Four valid windows", 295, 1240, 49, INK, True)
    elif scene["kind"] == "gradient":
        lines(draw, "OBSERVED", 72, 760, 27, AMBER, True)
        lines(draw, "0.25  0.25  0.25  0.25", 72, 820, 45, fg, True)
        draw.line((72, 920, 1008, 920), fill=MUTED, width=2)
        lines(draw, "FINITE DIFFERENCE", 72, 982, 27, MINT, True)
        lines(draw, "0.25  0.75  0.75  2.25", 72, 1042, 45, fg, True)
        draw.rounded_rectangle((72, 1210, 1008, 1360), 24, fill="#6b352d")
        lines(draw, "max absolute error = 2.0", 116, 1251, 45, CREAM, True)
    elif scene["kind"] == "repair":
        for y, left, right in [
            (760, "height + top", "height + bottom"),
            (1010, "width + left", "width + right"),
        ]:
            draw.rounded_rectangle((72, y, 1008, y + 180), 24, fill="#e2e8dc")
            lines(draw, left, 110, y + 35, 41, RED, True)
            draw.text((480, y + 53), "TO", font=font(38, True), fill=INK)
            lines(draw, right, 610, y + 35, 41, INK, True)
        lines(draw, "8 failures TO 0", 245, 1285, 62, INK, True)
    else:
        draw.rounded_rectangle((72, 760, 1008, 1055), 28, outline=MINT, width=4)
        lines(draw, "2021", 116, 805, 92, MINT, True)
        lines(draw, "PR #1360 already used\nthe correct bounds", 405, 807, 41, fg, True, 18)
        lines(draw, "Reproduced current behavior.\nNovelty remains unclaimed.", 72, 1205, 51, fg, True, 22)

    draw.rounded_rectangle((50, 1538, 1030, 1732), 24, fill="#0a211e" if dark else "#dde7dc")
    lines(draw, scene["caption"], 82, 1575, 35, fg if dark else INK, False, 18)
    draw.text((72, 1795), f'EVIDENCE {scene["source"]} · 09 SEP 2026', font=font(27), fill=MUTED)
    draw.text((72, 1840), "Synthetic narration · Xamit Kadirbekov / GERO", font=font(25), fill=MUTED)
    draw.text((940, 1795), f"{index:02d}", font=font(34, True), fill=fg)

    frame = ASSETS / f"scene-{index:02d}.png"
    image.save(frame)
    text_path = ASSETS / f"scene-{index:02d}.txt"
    text_path.write_text(scene["speech"], encoding="utf-8")
    audio = ASSETS / f"scene-{index:02d}.aiff"
    run(["say", "-v", "Daniel", "-r", "188", "-f", str(text_path), "-o", str(audio)])
    seconds = math.ceil((duration(audio) + 0.55) * 30) / 30
    segment = ASSETS / f"scene-{index:02d}.mp4"
    run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-loop",
            "1",
            "-framerate",
            "30",
            "-i",
            str(frame),
            "-i",
            str(audio),
            "-t",
            str(seconds),
            "-vf",
            f"fade=t=in:st=0:d=0.15,fade=t=out:st={seconds - 0.18}:d=0.18",
            "-af",
            "apad=pad_dur=1",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "160k",
            "-ar",
            "48000",
            str(segment),
        ]
    )
    parts.append(segment)
    timings.append({"start": total, "end": total + seconds, **scene})
    total += seconds
    print(f"Scene {index}: {seconds:.2f}s", flush=True)

concat = ASSETS / "concat.txt"
concat.write_text("".join(f"file '{part}'\n" for part in parts), encoding="utf-8")
run(
    [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat),
        "-c",
        "copy",
        "-movflags",
        "+faststart",
        str(OUT / "episode.mp4"),
    ]
)
(OUT / "timings.json").write_text(json.dumps(timings, indent=2) + "\n", encoding="utf-8")


def stamp(value):
    millis = round(value * 1000)
    return f"{millis // 3600000:02}:{millis // 60000 % 60:02}:{millis // 1000 % 60:02},{millis % 1000:03}"


captions = []
number = 1
for scene in timings:
    chunks = textwrap.wrap(scene["speech"], width=70, break_long_words=False, break_on_hyphens=False)
    weights = [len(chunk.split()) for chunk in chunks]
    cursor = scene["start"]
    usable_end = scene["end"] - 0.4
    for chunk, weight in zip(chunks, weights):
        next_cursor = cursor + (usable_end - scene["start"]) * weight / sum(weights)
        captions.append(
            f"{number}\n{stamp(cursor)} --> {stamp(next_cursor)}\n"
            + textwrap.fill(chunk, width=42, break_on_hyphens=False)
            + "\n"
        )
        cursor = next_cursor
        number += 1
(OUT / "captions.en.srt").write_text("\n".join(captions), encoding="utf-8")

Image.open(ASSETS / "scene-01.png").save(OUT / "thumbnail.png")
sheet = Image.new("RGB", (3 * 270, 2 * 480), "#d8ddd5")
for index in range(1, len(SCENES) + 1):
    sheet.paste(
        Image.open(ASSETS / f"scene-{index:02d}.png").resize((270, 480)),
        (((index - 1) % 3) * 270, ((index - 1) // 3) * 480),
    )
sheet.save(OUT / "contact-sheet.jpg")
print(f"Complete: {total:.2f}s", flush=True)
