"""Summarise sim/results/sim_rows.csv -> summary.csv + plots."""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
df = pd.read_csv(os.path.join(OUT, "sim_rows.csv"))
METHODS = ["rssi", "phase", "both", "rssi_max"]
LAYOUT_ORDER = list(dict.fromkeys(df.layout))
SEVS = ["mild", "typical", "harsh"]
CONF_GATE = 0.90

rows = []
for (lay, sev), g in df.groupby(["layout", "sev"], sort=False):
    for m in METHODS:
        if m == "rssi_max" and not lay.startswith("4 ant"):
            continue
        ok = g[f"{m}_q"] == g.q
        far = g.bdist >= 3                       # tag >= 3" from a quadrant boundary
        conf = g[f"{m}_conf"] >= CONF_GATE
        rows.append(dict(
            layout=lay, sev=sev, method=m, n=len(g),
            acc=ok.mean(),
            acc_3in_from_edge=ok[far].mean(),
            acc_when_conf90=ok[conf].mean() if conf.any() else np.nan,
            top2_acc=(ok | (g[f"{m}_q2"] == g.q)).mean(),
            coverage_conf90=conf.mean() if m != "rssi_max" else np.nan,
            p_err_lt_12in=(g[f"{m}_err_in"] < 12).mean(),
            p_err_lt_18in=(g[f"{m}_err_in"] < 18).mean(),
            median_err_in=g[f"{m}_err_in"].median(),
        ))
S = pd.DataFrame(rows)
S.to_csv(os.path.join(OUT, "summary.csv"), index=False)

pd.set_option("display.width", 200)
for sev in SEVS:
    print(f"\n=== {sev} multipath ===")
    t = S[S.sev == sev].copy()
    for c in ["acc", "acc_3in_from_edge", "acc_when_conf90", "top2_acc", "coverage_conf90", "p_err_lt_12in", "p_err_lt_18in"]:
        t[c] = (100 * t[c]).round(1)
    t["median_err_in"] = t.median_err_in.round(1)
    print(t.drop(columns=["sev", "n"]).to_string(index=False))

# ── Plot 1: quadrant accuracy vs layout, per severity, 'both' vs 'rssi' vs 'phase'
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
colors = {"rssi": "#8c8c8c", "phase": "#2a7fff", "both": "#e8590c"}
x = np.arange(len(LAYOUT_ORDER))
for ax, sev in zip(axes, SEVS):
    t = S[S.sev == sev]
    for i, m in enumerate(["rssi", "phase", "both"]):
        v = [t[(t.layout == L) & (t.method == m)].acc.values[0] * 100 for L in LAYOUT_ORDER]
        ax.bar(x + (i - 1) * 0.27, v, 0.27, label={"rssi": "RSSI only", "phase": "Phase-slope only",
                                                    "both": "RSSI + phase"}[m], color=colors[m])
    rm = t[t.method == "rssi_max"]
    if len(rm):
        j = LAYOUT_ORDER.index(rm.layout.values[0])
        ax.scatter([j], [rm.acc.values[0] * 100], marker="_", s=600, color="k", zorder=5,
                   label="strongest-antenna (naive)")
    ax.axhline(95, ls=":", color="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([L.replace(" - ", "\n") for L in LAYOUT_ORDER], fontsize=8)
    ax.set_title(f"{sev} multipath"); ax.set_ylim(0, 100); ax.grid(axis="y", alpha=0.3)
axes[0].set_ylabel("quadrant accuracy, all tag positions (%)")
axes[0].legend(fontsize=8, loc="upper left")
fig.suptitle("Which quadrant? 32\"x24\"x20\" metal shelf zone, ceiling-mounted antennas (simulation)")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "accuracy_vs_antennas.png"), dpi=140)

# ── Plot 2: accuracy vs distance from quadrant boundary (typical, 'both')
fig, ax = plt.subplots(figsize=(7.5, 4.6))
bins = np.array([0, 1, 2, 3, 4, 6, 8, 12])
for L in LAYOUT_ORDER:
    g = df[(df.layout == L) & (df.sev == "typical")]
    b = np.digitize(g.bdist, bins) - 1
    acc = [((g.both_q == g.q)[b == k]).mean() * 100 for k in range(len(bins) - 1)]
    ax.plot((bins[:-1] + bins[1:]) / 2, acc, "o-", label=L)
ax.axhline(95, ls=":", color="k", lw=0.8)
ax.set_xlabel("tag distance from nearest quadrant boundary (in)")
ax.set_ylabel("quadrant accuracy (%)"); ax.set_ylim(0, 101); ax.grid(alpha=0.3)
ax.set_title("Typical multipath, RSSI + phase: errors concentrate at the boundaries")
ax.legend(fontsize=8); fig.tight_layout()
fig.savefig(os.path.join(OUT, "accuracy_vs_boundary_distance.png"), dpi=140)

# ── Plot 3: error maps (x, y) for 2/3/4 ant, typical, 'both'
show = [L for L in LAYOUT_ORDER if not L.startswith("1 ant")]
fig, axes = plt.subplots(1, len(show), figsize=(3.6 * len(show), 3.4))
from quadrant_sim import LAYOUTS  # noqa: E402
for ax, L in zip(axes, show):
    g = df[(df.layout == L) & (df.sev == "typical")]
    ok = g.both_q == g.q
    ax.scatter(g.x[ok], g.y[ok], s=4, c="#9fd39f")
    ax.scatter(g.x[~ok], g.y[~ok], s=7, c="#d62728")
    for (ax_, ay_) in LAYOUTS[L]:
        ax.add_patch(plt.Rectangle((ax_ - 4, ay_ - 4), 8, 8, fill=False, ec="k", lw=1.2))
    ax.axvline(16, color="k", lw=0.6); ax.axhline(12, color="k", lw=0.6)
    ax.set_xlim(0, 32); ax.set_ylim(0, 24); ax.set_aspect("equal")
    ax.set_title(f"{L}\n{ok.mean()*100:.0f}% correct", fontsize=9)
    ax.set_xticks([0, 16, 32]); ax.set_yticks([0, 12, 24])
fig.suptitle("Typical multipath, RSSI+phase: red = wrong quadrant (squares = 8\" antennas, top view)", fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "error_maps_typical.png"), dpi=140)
print(f"\nplots + summary.csv -> {OUT}")
