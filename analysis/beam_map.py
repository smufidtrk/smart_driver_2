"""
Box-layout map of a beam check: which boxes each port reads alone, and which port is strongest per box.

  python analysis/beam_map.py sessions/beam_check_silion10_16cm [more session dirs ...]

Needs analysis/beam_check.py run on each dir first (reads its beam_check.csv).
Writes <dir>/beam_map.png. Shelf seen from above: back wall at the top, aisle (array) at the bottom.
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle, FancyBboxPatch

SURF, INK, INK2, MUTED, LINE = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#d6d5d0"
PORT_C = {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a", 4: "#eda100"}  # categorical slots 1-4
SEQ = LinearSegmentedColormap.from_list("blue", ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"])
LO, HI = -56.0, -35.0
NORM = Normalize(LO, HI)
BOX = 5.0  # drawn box size, inches
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "text.color": INK})


def shelf(ax, title):
    ax.set_xlim(-1, 49); ax.set_ylim(-6.5, 28.5); ax.set_aspect("equal"); ax.axis("off")
    ax.add_patch(Rectangle((0, 0), 48, 24, fill=False, ec=LINE, lw=1))
    ax.add_patch(Rectangle((0, 24), 48, 1.2, color=MUTED, lw=0))
    for e in (12, 24, 36):
        ax.plot([e, e], [0, 25.2], color=MUTED, lw=0.8, ls=(0, (3, 3)))
    for z in range(4):
        ax.text(6 + 12 * z, 26.4, f"Zone {z + 1}", ha="center", va="bottom", color=INK2, fontsize=8, fontweight="bold")
    # array at the aisle edge, centred: 4 elements 16 cm (6.3") apart
    for i in range(4):
        x = 24 + (i - 1.5) * 6.3
        ax.add_patch(Rectangle((x - 1.4, -2.6), 2.8, 1.6, color=INK2, lw=0))
        ax.text(x, -3.6, f"A{i + 1}", ha="center", va="top", fontsize=6, color=INK2)
    ax.text(24, -5.6, "aisle · array", ha="center", va="top", fontsize=7, color=INK2)
    ax.set_title(title, loc="left", fontsize=9, fontweight="bold", color=INK, pad=4)


def unread(ax, x, y, box):
    ax.add_patch(FancyBboxPatch((x - BOX / 2, y - BOX / 2), BOX, BOX, boxstyle="round,pad=0,rounding_size=0.6",
                                fc=SURF, ec=LINE, lw=1, ls=(0, (2, 2))))
    ax.text(x, y + 0.9, box, ha="center", va="center", fontsize=7, color=MUTED)
    ax.text(x, y - 1.1, "–", ha="center", va="center", fontsize=7, color=MUTED)


def run(sd):
    R = pd.read_csv(os.path.join(sd, "beam_check.csv"), dtype={"box": str})
    name = os.path.basename(os.path.normpath(sd))
    fig, axg = plt.subplots(2, 3, figsize=(15, 7.6), facecolor=SURF)
    axs = [axg[0, 0], axg[0, 1], axg[1, 0], axg[1, 1], axg[0, 2]]
    for ax, (p, g) in zip(axs, R.groupby("port")):
        tot = g.reads.sum()
        share = g.groupby("zone").reads.sum() / max(tot, 1) * 100
        shelf(ax, f"Port {p} alone · {int((g.reads > 0).sum())}/32 boxes\n" +
              "  ".join(f"Z{z} {share.get(z, 0):.0f}%" for z in (1, 2, 3, 4)) + "  of reads")
        for _, r in g.iterrows():
            if r.reads == 0 or not np.isfinite(r.rssi):
                unread(ax, r.x, r.y, r.box); continue
            c = SEQ(NORM(np.clip(r.rssi, LO, HI)))
            ax.add_patch(FancyBboxPatch((r.x - BOX / 2, r.y - BOX / 2), BOX, BOX, boxstyle="round,pad=0,rounding_size=0.6",
                                        fc=c, ec=SURF, lw=2))
            tc = "#ffffff" if NORM(r.rssi) > 0.35 else INK
            ax.text(r.x, r.y + 0.9, r.box, ha="center", va="center", fontsize=7, color=tc, fontweight="bold")
            ax.text(r.x, r.y - 1.1, f"{r.rssi:.0f}", ha="center", va="center", fontsize=7, color=tc)
    # strongest port per box
    ax = axs[4]
    P = R.pivot_table(index=["box", "zone", "x", "y"], columns="port", values="rssi").reset_index()
    ports = [c for c in P.columns if isinstance(c, (int, np.integer))]
    shelf(ax, "Strongest port per box\nmargin over the next port (dB)")
    for _, r in P.iterrows():
        v = r[ports].astype(float)
        if v.notna().sum() == 0:
            unread(ax, r.x, r.y, r.box); continue
        bp = int(v.idxmax()); s = np.sort(v.dropna().values)
        m = f"+{s[-1] - s[-2]:.0f}" if len(s) > 1 else "only"
        ax.add_patch(FancyBboxPatch((r.x - BOX / 2, r.y - BOX / 2), BOX, BOX, boxstyle="round,pad=0,rounding_size=0.6",
                                    fc=PORT_C[bp], ec=SURF, lw=2))
        ax.text(r.x, r.y + 0.9, f"P{bp}", ha="center", va="center", fontsize=7, color=INK, fontweight="bold")
        ax.text(r.x, r.y - 1.1, m, ha="center", va="center", fontsize=7, color=INK)
    # legend panel
    lg = axg[1, 2]; lg.axis("off"); lg.set_xlim(0, 1); lg.set_ylim(0, 1)
    lg.text(0.04, 0.93, "How to read", fontsize=9, fontweight="bold", va="top")
    lg.text(0.04, 0.84, "Seen from above: back wall at the top (grey bar),\n"
                        "the 4-element array at the aisle edge at the bottom.\n"
                        "Each square is a box; number = box (last 3 EPC digits).", fontsize=8, color=INK2, va="top")
    lg.text(0.04, 0.58, "Port panels: shade = median RSSI on that port alone;\n"
                        "dashed grey = not read by that port.", fontsize=8, color=INK2, va="top")
    lg.text(0.04, 0.24, "Strongest-port panel: colour = port with the highest\n"
                        "RSSI; 'only' = the only port that read it.", fontsize=8, color=INK2, va="top")
    for i, (p, c) in enumerate(PORT_C.items()):
        lg.add_patch(Rectangle((0.04 + i * 0.23, 0.05), 0.04, 0.05, color=c, lw=0))
        lg.text(0.10 + i * 0.23, 0.075, f"Port {p}", va="center", fontsize=8, color=INK2)
    cax = lg.inset_axes([0.04, 0.38, 0.6, 0.05])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=NORM, cmap=SEQ), cax=cax, orientation="horizontal")
    cb.set_label("median RSSI (dBm)", fontsize=7, color=INK2)
    cb.outline.set_visible(False); cax.tick_params(labelsize=7, colors=INK2)
    run_name = {"beam_check_silion8_16cm": "Silion · 8 dBm · 30 s per port · elements 16 cm apart",
                "beam_check_silion10_16cm": "Silion · 10 dBm · 30 s per port · elements 16 cm apart",
                "beam_check_silion8": "Silion · 8 dBm · 30 s per port · wrong element spacing",
                "beam_check_silion5": "Silion · 5 dBm · 20 s per port · wrong element spacing",
                "beam_check": "R700 · 10 dBm · 20 s per port · wrong element spacing",
                **{f"beam_check_gap3_{p}dbm": f"Silion · {p} dBm · 30 s per port · zones 3\" apart (boxes drawn on the old grid)"
                   for p in (5, 10, 15)}}.get(name, name)
    fig.suptitle(f"Beam-steering check: each port alone · {run_name}", x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.subplots_adjust(left=0.01, right=0.995, top=0.90, bottom=0.01, wspace=0.04, hspace=0.18)
    out = os.path.join(sd, "beam_map.png")
    fig.savefig(out, dpi=130, facecolor=SURF); plt.close(fig)
    print("->", out)


if __name__ == "__main__":
    for d in sys.argv[1:]:
        run(d)
