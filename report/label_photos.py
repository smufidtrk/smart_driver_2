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

# Nov 13 beam-steering setup (IMG_6371/6372/6374): positions below are in a 1200x900 frame, scaled to 1600x1200
from PIL import ImageOps
K = 1600 / 1200


def load_new(name):
    im = ImageOps.exif_transpose(Image.open(os.path.join(ROOT, name))).convert("RGB")
    return im.resize((1600, 1200), Image.LANCZOS)


def lb(d, xy, text, bg, arrow_to=None, sz=26):
    label(d, (xy[0] * K, xy[1] * K), text, sz, bg=bg, arrow_to=None if arrow_to is None else (arrow_to[0] * K, arrow_to[1] * K))


im = load_new("IMG_6371.jpg"); d = ImageDraw.Draw(im)
lb(d, (165, 470), "4-patch array, 16 cm apart", ANT, arrow_to=(560, 345))
lb(d, (660, 100), "RF cables to the reader", NOTE, arrow_to=(655, 255))
for n, x in enumerate([200, 440, 690, 960], 1):
    lb(d, (x, 538), f"Zone {n}", ZONE)
lb(d, (140, 842), "Aisle edge (front rail)", NOTE)
im.save(os.path.join(OUT, "beam_aisle_view.jpg"), quality=86)

im = load_new("IMG_6372.jpg"); d = ImageDraw.Draw(im)
lb(d, (330, 140), "Array: 4 patches on foam, 16 cm apart", ANT, arrow_to=(935, 465))
lb(d, (330, 290), "Metal roof", NOTE)
lb(d, (420, 760), "RF cables", NOTE, arrow_to=(900, 545))
im.save(os.path.join(OUT, "beam_array_view.jpg"), quality=86)

im = load_new("IMG_6374.jpg"); d = ImageDraw.Draw(im)
lb(d, (430, 60), "Array at the roof edge", ANT, arrow_to=(985, 110))
lb(d, (40, 400), "32 boxes, 4 rows x 8 columns", BOX, arrow_to=(600, 545))
lb(d, (40, 470), "Metal side walls", NOTE)
im.save(os.path.join(OUT, "beam_shelf_view.jpg"), quality=86)
print("wrote beam photos")

# Nov 15 Zircon back-metal setup (zircon_setup_15..18.jpg, 2000x1500): positions below are in that frame, scaled to 1600x1200
K2 = 1600 / 2000


def lb2(d, xy, text, bg, arrow_to=None, sz=26):
    label(d, (xy[0] * K2, xy[1] * K2), text, sz, bg=bg,
          arrow_to=None if arrow_to is None else (arrow_to[0] * K2, arrow_to[1] * K2))


im = load_new("zircon_setup_15.jpg"); d = ImageDraw.Draw(im)
lb2(d, (900, 330), "4-patch array on the metal back plate", ANT, arrow_to=(1300, 570))
lb2(d, (1180, 120), "Metal back plate", NOTE, arrow_to=(1560, 330))
lb2(d, (180, 1010), "Tagged boxes (EPC label on each)", BOX, arrow_to=(640, 760))
lb2(d, (1180, 900), "Aisle", NOTE)
im.save(os.path.join(OUT, "zircon_aisle_view.jpg"), quality=86)

im = load_new("zircon_setup_16.jpg"); d = ImageDraw.Draw(im)
lb2(d, (1150, 1180), "4 patches on the metal back plate", ANT, arrow_to=(1230, 920))
lb2(d, (1400, 520), "RF cables", NOTE, arrow_to=(1560, 640))
lb2(d, (200, 160), "Metal roof", NOTE)
im.save(os.path.join(OUT, "zircon_array_view.jpg"), quality=86)

im = load_new("zircon_setup_17.jpg"); d = ImageDraw.Draw(im)
lb2(d, (620, 110), "Back of the array: metal back plate", ANT, arrow_to=(1000, 260))
lb2(d, (1250, 40), "RF cables", NOTE, arrow_to=(1090, 95))
for n, x in enumerate([250, 690, 1060, 1420], 1):
    lb2(d, (x, 860), f"Zone {n}", ZONE)
lb2(d, (60, 1000), "Metal shelf and back wall", NOTE)
im.save(os.path.join(OUT, "zircon_shelf_view.jpg"), quality=86)

im = load_new("zircon_setup_18.jpg"); d = ImageDraw.Draw(im)
lb2(d, (850, 230), "Array on the metal back plate", ANT, arrow_to=(640, 470))
lb2(d, (60, 1180), "Laptop: J-Link RTT log", NOTE, arrow_to=(280, 880))
lb2(d, (1150, 1300), "Tagged boxes", BOX, arrow_to=(1350, 950))
im.save(os.path.join(OUT, "zircon_end_view.jpg"), quality=86)
print("wrote zircon photos")

# Back metal v1 (zircon_setup_19/20.jpg, 2000x1500): plate held perpendicular to the shelf behind the foam-mounted array
im = load_new("zircon_setup_19.jpg"); d = ImageDraw.Draw(im)
lb2(d, (1330, 180), "Metal plate, perpendicular to the shelf (v1)", NOTE, arrow_to=(1250, 600))
lb2(d, (40, 1180), "32 boxes, one layer", BOX, arrow_to=(560, 1000))
im.save(os.path.join(OUT, "zircon_v1_shelf_view.jpg"), quality=86)

im = load_new("zircon_setup_20.jpg"); d = ImageDraw.Draw(im)
lb2(d, (420, 560), "4 patches on foam, 16 cm apart", ANT, arrow_to=(1110, 820))
lb2(d, (1380, 1120), "Metal plate behind the array (v1)", NOTE, arrow_to=(1240, 990))
lb2(d, (60, 1000), "32 boxes, one layer", BOX, arrow_to=(560, 1260))
lb2(d, (800, 220), "Metal roof", NOTE)
im.save(os.path.join(OUT, "zircon_v1_array_view.jpg"), quality=86)
print("wrote zircon v1 photos")
