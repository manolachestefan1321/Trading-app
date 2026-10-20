"""Generate the PWA icons (run once: python -m webapp.make_icons)."""

from pathlib import Path

from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parent / "static" / "icons"
BG, UP, DOWN, WICK = (15, 23, 42), (16, 185, 129), (239, 68, 68), (203, 213, 225)
# (x centre, wick top, wick bottom, body top, body bottom, bullish) on a 100x100 grid
CANDLES = [(25, 52, 84, 60, 76, False), (42, 40, 76, 46, 66, True), (59, 28, 64, 34, 52, True), (76, 16, 50, 22, 40, True)]


def draw(size: int, safe: float) -> Image.Image:
    """safe = fraction of the canvas the artwork may use (maskable icons need ~80%)."""
    img = Image.new("RGB", (size, size), BG)
    d = ImageDraw.Draw(img)
    pad = size * (1 - safe) / 2
    k = (size - 2 * pad) / 100

    def xy(x, y):
        return pad + x * k, pad + y * k

    for x, wt, wb, bt, bb, bull in CANDLES:
        col = UP if bull else DOWN
        d.line([xy(x, wt), xy(x, wb)], fill=WICK, width=max(2, int(2.2 * k)))
        (x0, y0), (x1, y1) = xy(x - 6, bt), xy(x + 6, bb)
        d.rounded_rectangle([x0, y0, x1, y1], radius=2 * k, fill=col)
    return img


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for s in (180, 192, 512):
        draw(s, 0.86).save(OUT / f"icon-{s}.png")
    draw(512, 0.68).save(OUT / "icon-maskable-512.png")
    print("icons written to", OUT)
