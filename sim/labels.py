import random
from pathlib import Path

import segno
from PIL import Image, ImageDraw, ImageFont

JAR_BRANDS = ("ORCHARD", "HILLSIDE", "GOLDEN MEADOW", "OLD MILL")
JAR_LABEL_COLOURS = ((196, 42, 38), (34, 92, 60), (220, 120, 30), (38, 58, 120))
CITIES = ("PUNE", "LEEDS", "AUSTIN", "LYON", "OSAKA", "PORTO", "DENVER", "TURIN")


def jar_label(path: Path, design: int, contents: str) -> Path:
    width, height = 1024, 320
    # A paper-white label with a coloured band stands out against any jar contents.
    band = JAR_LABEL_COLOURS[design % len(JAR_LABEL_COLOURS)]
    ink = (25, 25, 25)
    image = Image.new("RGB", (width, height), (240, 236, 222))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, width, 60), fill=band)
    draw.rectangle((0, height - 60, width, height), fill=band)
    # The front of the jar shows the middle of the label, so the text sits there.
    middle = width // 2
    draw.text(
        (middle, 120), JAR_BRANDS[design % len(JAR_BRANDS)], fill=ink, font=_font(64), anchor="mm"
    )
    draw.text((middle, 200), contents.upper(), fill=ink, font=_font(40), anchor="mm")
    draw.text((middle, 255), "NET WT 340 g", fill=ink, font=_font(24), anchor="mm")
    image.save(path)
    return path


def shipping_label(path: Path, order_id: str, with_qr: bool, rng: random.Random) -> Path:
    width, height = 800, 600
    image = Image.new("RGB", (width, height), (246, 246, 240))
    draw = ImageDraw.Draw(image)
    draw.rectangle((8, 8, width - 8, height - 8), outline=(10, 10, 10), width=6)
    if with_qr:
        qr = segno.make_qr(f"ORDER:{order_id}", error="m")
        rows = [[0 if dark else 255 for dark in row] for row in qr.matrix_iter(scale=1, border=2)]
        code = Image.new("L", (len(rows[0]), len(rows)))
        code.putdata([value for row in rows for value in row])
        side = 330
        image.paste(code.resize((side, side), Image.Resampling.NEAREST).convert("RGB"), (30, 40))
    left = 390
    draw.text((left, 50), "SHIP TO", fill=(10, 10, 10), font=_font(34))
    draw.text((left, 100), f"ORDER {order_id}", fill=(10, 10, 10), font=_font(52))
    draw.text((left, 175), rng.choice(CITIES), fill=(10, 10, 10), font=_font(40))
    draw.text((left, 230), f"{rng.randint(10000, 99999)}", fill=(10, 10, 10), font=_font(34))
    # A 1D barcode strip, as on real carrier labels.
    x = 40
    while x < width - 40:
        bar = rng.choice((3, 5, 8))
        draw.rectangle((x, 420, x + bar, 560), fill=(10, 10, 10))
        x += bar + rng.choice((3, 5, 7))
    image.save(path)
    return path


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in ("arialbd.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size)
