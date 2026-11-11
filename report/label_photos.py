"""Labelled copies of the setup photos (originals untouched) -> report/photos/."""
import os
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "report", "photos")
os.makedirs(OUT, exist_ok=True)


def font(sz):
    for f in ("segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(f, sz)
        except OSError:
            pass
    return ImageFont.load_default()


def label(d, xy, text, sz=20, fill=(255, 255, 255), bg=(20, 60, 140), arrow_to=None):
    f = font(sz)
    x, y = xy
    l, t, r, b = d.textbbox((x, y), text, font=f)
    pad = 6
    if arrow_to:
        d.line([((l + r) / 2, (t + b) / 2), arrow_to], fill=bg, width=4)
        ax, ay = arrow_to
        d.ellipse([ax - 6, ay - 6, ax + 6, ay + 6], fill=bg)
    d.rounded_rectangle([l - pad, t - pad, r + pad, b + pad], radius=6, fill=bg)
    d.text((x, y), text, font=f, fill=fill)


ANT = (20, 60, 140)
ZONE = (0, 110, 70)
BOX = (40, 40, 40)
NOTE = (150, 60, 0)

# 2.jpeg: aisle-side view of the whole 4-zone setup
im = Image.open(os.path.join(ROOT, "2.jpeg")).convert("RGB")
d = ImageDraw.Draw(im)
for n, (x, y) in enumerate([(88, 352), (325, 345), (597, 345), (862, 362)], 1):
    label(d, (x - 34, y - 72), f"Ant {n}", 22, bg=ANT, arrow_to=(x, y - 8))
for n, x in enumerate([258, 405, 572, 732], 1):
    label(d, (x - 34, 600), f"Zone {n}", 22, bg=ZONE)
boxes = {"649": (255, 497), "650": (310, 425), "651": (345, 382), "653": (400, 502), "654": (433, 428),
         "655": (455, 372), "656": (570, 512), "657": (565, 422), "658": (555, 372), "659": (730, 517),
         "660": (698, 432), "661": (680, 372)}
for t, (x, y) in boxes.items():
    label(d, (x - 18, y - 12), t, 15, bg=BOX)
label(d, (150, 160), "R700 reader", 18, bg=NOTE, arrow_to=(215, 210))
label(d, (330, 25), "Battery pack", 18, bg=NOTE)
label(d, (690, 640), "Aisle edge (antenna side)", 18, bg=NOTE)
im.save(os.path.join(OUT, "setup_aisle_view.jpg"), quality=88)

# 3.jpeg: view along the shelf from one end
im = Image.open(os.path.join(ROOT, "3.jpeg")).convert("RGB")
d = ImageDraw.Draw(im)
label(d, (300, 70), "Antennas: roof, aisle edge,", 20, bg=ANT, arrow_to=(205, 165))
label(d, (300, 104), "tilted 45° down", 20, bg=ANT)
label(d, (20, 560), "Aisle side (open)", 20, bg=NOTE)
label(d, (560, 290), "Back wall", 20, bg=NOTE, arrow_to=(720, 380))
label(d, (330, 690), "Blue tape = zone divider", 20, bg=ZONE, arrow_to=(470, 765))
label(d, (60, 900), "Metal shelf floor", 20, bg=NOTE)
im.save(os.path.join(OUT, "setup_end_view.jpg"), quality=88)

# 1.jpeg: close-up from above
im = Image.open(os.path.join(ROOT, "1.jpeg")).convert("RGB")
d = ImageDraw.Draw(im)
label(d, (420, 600), "RFID tag label on the box top", 20, bg=BOX, arrow_to=(362, 470))
label(d, (600, 520), "Blue tape = zone divider", 20, bg=ZONE, arrow_to=(578, 470))
label(d, (40, 700), "Aisle-edge rail (antennas above)", 20, bg=NOTE)
label(d, (760, 250), "Back wall", 20, bg=NOTE)
im.save(os.path.join(OUT, "setup_top_view.jpg"), quality=88)
print("wrote", os.listdir(OUT))
