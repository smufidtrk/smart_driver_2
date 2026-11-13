"""
Per-antenna phase-distance histograms for one tag in one capture.

  python analysis/tag_phase_hist.py --file sessions/ui_runs/20261112_130952/cap_scan_20261112_130953.csv --epc 261

Phase gives distance through its slope across frequency: a path d long turns the phase by
4*pi*d*f/c, so the phase change between two neighbouring channels (0.5-1 MHz apart) is one
estimate of the path length.  For each antenna:
  1. phase per channel = circular mean of the reads (doubled, so the R700's 180-deg flips cancel)
  2. one slope estimate per neighbouring-channel step: d(phase)/d(freq), deg/MHz
     (the median of these is the slope the Zone Locator uses)
  3. each step -> raw round-trip path (metres, includes cable and reader delay)
     and -> horizontal distance from the antenna (inches) using the per-antenna fit
        slope = c + k * |x - antenna x|  on the 32 calibration boxes (locator.slope_model),
     which removes the cable delay.
Row 1: raw path per step (m).  Row 2: horizontal distance per step (in), median and the RSSI
median for comparison.  Silion captures: rows 1 only (the calibration is from the R700).
Writes <capture folder>/analysis/phase_hist_<box>.png and .csv.
"""
import argparse, glob, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools", "zone_ui"))
sys.path.insert(0, os.path.join(ROOT, "analysis"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
from tag_distance_hist import default_file, power_of  # noqa: E402
ZC = {1: "#1f77b4", 2: "#2ca02c", 3: "#ff7f0e", 4: "#d62728"}
C = 299792458.0


def steps(g):
    """Slope (deg/MHz) of every neighbouring-channel step of one tag on one antenna."""
    z = g.groupby("freq_mhz").phase_deg.apply(lambda a: np.mean(np.exp(2j * np.radians(a.values))))
    z = z.sort_index()
    if len(z) < 3:
        return np.array([])
    f = z.index.values.astype(float)
    ph = np.degrees(np.unwrap(np.angle(z.values))) / 2
    df, dp = np.diff(f), np.diff(ph)
    m = (df > 0) & (df <= 1.01)
    return dp[m] / df[m]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file")
    ap.add_argument("--epc", default="261")
    ap.add_argument("--power", type=float)
    a = ap.parse_args()
    path = os.path.abspath(a.file) if a.file else default_file(a.epc)
    cap_dir = os.path.dirname(path)
    power = a.power or power_of(path, cap_dir)
    d = pd.read_csv(path, dtype={"epc": str}); d["epc"] = d.epc.str.upper()
    hit = [e for e in d.epc.unique() if e.endswith(a.epc.upper())]
    if not hit:
        raise SystemExit(f"...{a.epc} is not in {path}")
    epc = hit[0]; t = d[d.epc == epc]
    silion = "end_phase_deg" in d.columns
    sm = None
    if not silion and power is not None:
        import locator as L
        sm = L.slope_model(L.library(["4zone_cal32", "4zone_cal32_repeat"], power))
    rssi_dist = {}
    rf = os.path.join(cap_dir, "analysis", f"dist_hist_{epc[-3:]}.csv")
    if os.path.exists(rf):
        rr = pd.read_csv(rf)
        rssi_dist = {int(r.antenna): r.dist_median_in for r in rr.itertuples() if pd.notna(getattr(r, "dist_median_in", np.nan))}

    fig, ax = plt.subplots(2, 4, figsize=(17, 7.5), sharey="row")
    out = []
    for j, p in enumerate([1, 2, 3, 4]):
        g = t[t.antenna == p]
        s = steps(g)
        a0, a1 = ax[0, j], ax[1, j]
        a0.set_title(f"Antenna {p} (zone {p})\n{len(g)} reads, {g.freq_mhz.nunique()} ch, {len(s)} channel steps", fontsize=10)
        if len(s) < 3:
            for x in (a0, a1):
                x.text(0.5, 0.5, "too few channels", ha="center", transform=x.transAxes, color="#888")
            out.append(dict(antenna=p, steps=len(s))); continue
        raw_m = np.abs(np.radians(s)) / 1e6 * C / (4 * np.pi)          # round trip -> one-way path, m
        a0.hist(np.clip(raw_m, 0, 40), bins=np.arange(0, 40.5, 1), color=ZC[p], alpha=0.85)
        a0.axvline(np.median(raw_m), color="k", lw=1.2, label=f"median {np.median(raw_m):.1f} m")
        a0.set_xlabel("path implied by each channel step (m, incl. cable)")
        a0.legend(fontsize=8)
        row = dict(antenna=p, reads=len(g), channels=g.freq_mhz.nunique(), steps=len(s),
                   slope_median=round(float(np.median(s)), 1), slope_iqr=round(float(np.subtract(*np.percentile(s, [75, 25]))), 1),
                   raw_path_median_m=round(float(np.median(raw_m)), 2))
        if sm:
            dx = np.clip((s - sm[p]["c"]) / sm[p]["k"], -20, 100)
            a1.hist(dx, bins=np.arange(-20, 101, 4), color=ZC[p], alpha=0.85)
            med = float(np.clip((np.median(s) - sm[p]["c"]) / sm[p]["k"], 0, 100))
            a1.axvline(med, color="k", lw=1.2, label=f"phase median {med:.0f}\"")
            if p in rssi_dist:
                a1.axvline(rssi_dist[p], color="k", ls=":", lw=1.2, label=f"RSSI median {rssi_dist[p]:.0f}\"")
            a1.axvspan(-20, 0, color="#eee")
            a1.set_xlabel("horizontal distance from antenna (in)")
            a1.legend(fontsize=8)
            a1.text(0.02, 0.97, f"fit scatter ±{sm[p]['sd_in']:.0f}\"", transform=a1.transAxes, va="top", fontsize=8)
            p10, p90 = np.percentile(dx, [10, 90])
            row.update(dist_median_in=round(med, 1), dist_p10_in=round(float(p10), 1), dist_p90_in=round(float(p90), 1))
        else:
            a1.text(0.5, 0.5, "no R700 calibration for this capture", ha="center", transform=a1.transAxes, color="#c00")
        out.append(row)
    ax[0, 0].set_ylabel("channel steps"); ax[1, 0].set_ylabel("channel steps")
    fig.suptitle(f"Tag ...{epc[-6:]} — {os.path.relpath(path, ROOT)} ({power:g} dBm)\n"
                 "phase distance: one estimate per neighbouring-channel step (phase slope across frequency)", fontsize=11)
    fig.tight_layout()
    od = os.path.join(cap_dir, "analysis"); os.makedirs(od, exist_ok=True)
    png = os.path.join(od, f"phase_hist_{epc[-3:]}.png")
    fig.savefig(png, dpi=110); plt.close(fig)
    O = pd.DataFrame(out); O.to_csv(png[:-4] + ".csv", index=False)
    pd.set_option("display.width", 250)
    print(O.to_string(index=False)); print("->", png)


if __name__ == "__main__":
    main()
