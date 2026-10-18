"""Sohayok's face: the app, tray, and shortcut icon, drawn from code at any size."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

CYAN = (46, 230, 255)
PINK = (255, 79, 216)
VIOLET = (155, 123, 255)


def _radial(size: int, inner: tuple, outer: tuple, center: tuple[float, float], radius: float):
    """A radial gradient image (inner colour at `center`, outer colour at `radius`)."""
    image = Image.new("RGBA", (size, size))
    pixels = image.load()
    cx, cy = center
    for y in range(size):
        for x in range(size):
            d = min(1.0, ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5 / radius)
            pixels[x, y] = tuple(int(inner[i] + (outer[i] - inner[i]) * d) for i in range(4))
    return image


def face(size: int = 256, mood: str = "happy", color: tuple = CYAN) -> Image.Image:
    """Draw the character. Moods: happy (default), listening (wide eyes), sleeping."""
    scale = 4  # draw large, then downsample for smooth edges
    s = size * scale
    image = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    c = s / 2
    r = s * 0.46

    glow = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    ImageDraw.Draw(glow).ellipse((c - r, c - r, c + r, c + r), fill=color + (150,))
    image.alpha_composite(glow.filter(ImageFilter.GaussianBlur(s * 0.04)))

    sphere = _radial(
        s // 8, color + (255,), (8, 14, 40, 255), (s / 8 * 0.36, s / 8 * 0.32), s / 8 * 0.75
    ).resize((s, s), Image.Resampling.BICUBIC)
    mask = Image.new("L", (s, s), 0)
    ImageDraw.Draw(mask).ellipse((c - r * 0.92, c - r * 0.92, c + r * 0.92, c + r * 0.92), fill=255)
    image.paste(sphere, (0, 0), mask)

    draw = ImageDraw.Draw(image)
    ring = r * 0.92
    draw.ellipse(
        (c - ring, c - ring, c + ring, c + ring),
        outline=color + (255,),
        width=max(2, int(s * 0.018)),
    )

    eye_w, eye_h = s * 0.075, s * (0.16 if mood == "listening" else 0.13)
    eye_y = c - s * 0.06
    for side in (-1, 1):
        ex = c + side * s * 0.15
        if mood == "sleeping":
            draw.arc(
                (ex - eye_w, eye_y - eye_w * 0.6, ex + eye_w, eye_y + eye_w * 0.9),
                20,
                160,
                fill=(235, 250, 255, 255),
                width=int(s * 0.022),
            )
            continue
        draw.ellipse(
            (ex - eye_w, eye_y - eye_h, ex + eye_w, eye_y + eye_h), fill=(240, 252, 255, 255)
        )
        draw.ellipse(
            (ex - eye_w * 0.1, eye_y + eye_h * 0.05, ex + eye_w * 0.75, eye_y + eye_h * 0.55),
            fill=color + (255,),
        )
        draw.ellipse(
            (ex - eye_w * 0.65, eye_y - eye_h * 0.62, ex - eye_w * 0.05, eye_y - eye_h * 0.18),
            fill=(255, 255, 255, 255),
        )
    for side in (-1, 1):  # blush
        bx = c + side * s * 0.25
        draw.ellipse(
            (bx - s * 0.06, c + s * 0.06, bx + s * 0.06, c + s * 0.105), fill=(255, 130, 220, 150)
        )
    smile = s * 0.075
    draw.arc(
        (c - smile, c + s * 0.03, c + smile, c + s * 0.03 + smile * 1.5),
        25,
        155,
        fill=(240, 252, 255, 255),
        width=max(2, int(s * 0.02)),
    )
    return image.resize((size, size), Image.Resampling.LANCZOS)


def write_icons(folder: Path) -> tuple[Path, Path]:
    """Write icon.ico (16–256 px) and icon.png (256 px) into `folder`."""
    folder.mkdir(parents=True, exist_ok=True)
    big = face(256)
    ico, png = folder / "icon.ico", folder / "icon.png"
    big.save(png)
    big.save(ico, sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    return ico, png


if __name__ == "__main__":
    import sys

    print(write_icons(Path(sys.argv[1] if len(sys.argv) > 1 else "src/sohayok/ui")))
