"""
Phase vs RSSI vs frequency for tags in one raw capture file.

  python analysis/raw_plot.py                                   (default file and one box per zone)
  python analysis/raw_plot.py --file sessions/4zone_mode2_d/cap_scan_20261111_132111-p20.csv --epc 654 661
  python analysis/raw_plot.py --file <csv> --all                (one figure for every tag)

--epc takes full EPCs or the last 3 digits.  Without --epc: the best-read box of each zone
(zones from the session's layout.json, if there is one).

Per tag, a 3 x 4 figure (rows: phase vs frequency, RSSI vs frequency, phase vs RSSI;
columns: antennas 1-4), plus one comparison figure of all the chosen tags.
Phase is folded to 0-180 deg: the R700's phase has a 180 deg ambiguity (reads flip by 180).
Writes PNGs and raw_stats.csv to <session>/analysis/raw_<file>/.
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT = os.path.join(ROOT, "sessions", "4zone_mode2_d", "cap_scan_20261111_132111-p20.csv")
ZC = {1: "#1f77b4", 2: "#2ca02c", 3: "#ff7f0e", 4: "#d62728", 0: "#555555"}


def fold(deg):
    return np.mod(deg, 180.0)


def chan_table(g):
    """Per-channel stats for one tag on one antenna."""
    def agg(d):
        z = np.mean(np.exp(2j * np.radians(d.phase_deg.values)))        # doubled phase: flip-safe
        R = min(abs(z), 1.0)
        return pd.Series(dict(
            reads=len(d),
            rssi=d.rssi_dbm.median(),
            rssi_sd=d.rssi_dbm.std(ddof=0),
            phase=np.mod(np.degrees(np.angle(z)) / 2, 180),               # channel mean, 0-180
            phase_sd=np.degrees(np.sqrt(-2 * np.log(max(R, 1e-9)))) / 2,  # circular std, deg
        ))
    return g.groupby("freq_mhz").apply(agg, include_groups=False)


def unwrap_slope(ch):
    """Phase slope across frequency (deg/MHz) from the folded channel means, and the
    distance it implies (d = c * slope / (4 pi), folded phase -> x2)."""
    if len(ch) < 6:
        return np.nan, np.nan
    f = ch.index.values
    ph = np.unwrap(np.radians(ch.phase.values) * 2) / 2                   # back to real units
    k = np.polyfit(f, np.degrees(ph), 1)[0]                                # deg / MHz
    d = abs(np.radians(k) / 1e6) * 3e8 / (4 * np.pi)                       # metres (round trip)
    return k, d


def tag_stats(cap, epc, ports):
    rows, chans = [], {}
    for p in ports:
        g = cap[(cap.epc == epc) & (cap.antenna == p)]
        if g.empty:
            rows.append(dict(antenna=p, reads=0)); continue
        ch = chan_table(g)
        chans[p] = ch
        k, d = unwrap_slope(ch)
        rows.append(dict(
            antenna=p, reads=len(g), channels=len(ch),
            rssi_median=g.rssi_dbm.median(),
            rssi_fade_db=ch.rssi.max() - ch.rssi.min(),                       # spread across channels
            rssi_noise_db=(g.rssi_dbm - g.groupby("freq_mhz").rssi_dbm.transform("median")).abs().median(),
            phase_noise_deg=ch.phase_sd.median(),
            flip_share=float(np.mean(np.abs(((g.phase_deg - g.groupby("freq_mhz").phase_deg.transform(
                lambda a: np.degrees(np.angle(np.mean(np.exp(1j * np.radians(a))))))) + 90) % 360 - 90) > 90)),
            phase_slope_deg_mhz=k, slope_range_m=d,
            corr_phase_rssi=np.corrcoef(ch.phase, ch.rssi)[0, 1] if len(ch) > 3 else np.nan,
        ))
    return pd.DataFrame(rows), chans


def tag_figure(cap, epc, label, zone, ports, out):
    S, chans = tag_stats(cap, epc, ports)
    fig, ax = plt.subplots(3, len(ports), figsize=(4.2 * len(ports), 10.5), squeeze=False)
    fmin, fmax = cap.freq_mhz.min() - 0.5, cap.freq_mhz.max() + 0.5
    for j, p in enumerate(ports):
        g = cap[(cap.epc == epc) & (cap.antenna == p)]
        st = S[S.antenna == p].iloc[0]
        a0, a1, a2 = ax[0, j], ax[1, j], ax[2, j]
        a0.set_title(f"Antenna {p}" + ("" if g.empty else
                     f"\n{int(st.reads)} reads, {int(st.channels)} ch, median {st.rssi_median:.1f} dBm"), fontsize=10)
        if g.empty:
            for a in (a0, a1, a2):
                a.text(0.5, 0.5, "not read", ha="center", va="center", transform=a.transAxes, color="#888")
            continue
        ch = chans[p]
        # phase vs frequency
        a0.scatter(g.freq_mhz, fold(g.phase_deg), s=3, alpha=0.25, color="#888", lw=0)
        a0.errorbar(ch.index, ch.phase, yerr=ch.phase_sd, fmt="o", ms=3, color=ZC[zone], lw=0.8)
        a0.set_ylim(0, 180); a0.set_xlim(fmin, fmax)
        a0.text(0.02, 0.97, f"noise {st.phase_noise_deg:.1f}°", transform=a0.transAxes, va="top", fontsize=8)
        # RSSI vs frequency
        a1.scatter(g.freq_mhz, g.rssi_dbm, s=3, alpha=0.25, color="#888", lw=0)
        a1.plot(ch.index, ch.rssi, "-o", ms=3, color=ZC[zone], lw=1)
        a1.set_xlim(fmin, fmax)
        a1.text(0.02, 0.97, f"fade {st.rssi_fade_db:.1f} dB across channels", transform=a1.transAxes,
                va="top", fontsize=8)
        # phase vs RSSI (per channel), coloured by frequency
        sc = a2.scatter(ch.rssi, ch.phase, c=ch.index, cmap="viridis", s=22, vmin=fmin, vmax=fmax)
        a2.set_ylim(0, 180)
        a2.text(0.02, 0.97, f"r = {st.corr_phase_rssi:.2f}", transform=a2.transAxes, va="top", fontsize=8)
        a0.set_xlabel("frequency (MHz)"); a1.set_xlabel("frequency (MHz)"); a2.set_xlabel("RSSI (dBm, channel median)")
    ax[0, 0].set_ylabel("phase (deg, folded 0–180)")
    ax[1, 0].set_ylabel("RSSI (dBm)")
    ax[2, 0].set_ylabel("phase (deg, channel mean)")
    # shared RSSI axis across antennas so the levels compare
    lo = cap[cap.epc == epc].rssi_dbm.min() - 2; hi = cap[cap.epc == epc].rssi_dbm.max() + 2
    for j in range(len(ports)):
        ax[1, j].set_ylim(lo, hi); ax[2, j].set_xlim(lo, hi)
    cb = fig.colorbar(sc, ax=ax[2, :].tolist(), shrink=0.8, pad=0.01) if "sc" in dir() else None
    if cb: cb.set_label("frequency (MHz)")
    fig.suptitle(f"Box {label}  (zone {zone if zone else '?'})  —  {os.path.basename(out).split('__')[0]}\n"
                 "grey = every read, coloured = per-channel value (bars = phase spread within a channel)",
                 fontsize=11)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    S.insert(0, "zone", zone); S.insert(0, "box", label)
    return S


def compare_figure(cap, tags, ports, out):
    """Per-channel RSSI and phase of every chosen tag on each antenna."""
    fig, ax = plt.subplots(2, len(ports), figsize=(4.2 * len(ports), 7), squeeze=False, sharex=True)
    ls = ["-", "--", ":", "-."]
    seen = {}
    for epc, label, zone in tags:
        k = seen.get(zone, 0); seen[zone] = k + 1
        for j, p in enumerate(ports):
            g = cap[(cap.epc == epc) & (cap.antenna == p)]
            if g.empty:
                continue
            ch = chan_table(g)
            ax[0, j].plot(ch.index, ch.rssi, ls[k % 4], marker="o", ms=2.5, lw=1, color=ZC[zone],
                          label=f"{label} (z{zone})")
            ax[1, j].plot(ch.index, ch.phase, ls[k % 4], marker="o", ms=2.5, lw=0.8, color=ZC[zone])
    for j, p in enumerate(ports):
        ax[0, j].set_title(f"Antenna {p}", fontsize=10)
        ax[1, j].set_ylim(0, 180); ax[1, j].set_xlabel("frequency (MHz)")
    ax[0, 0].set_ylabel("RSSI (dBm, channel median)")
    ax[1, 0].set_ylabel("phase (deg, channel mean)")
    ax[0, 0].legend(fontsize=7, loc="lower left")
    fig.suptitle("Chosen boxes compared (colour = zone)", fontsize=11)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=DEFAULT)
    ap.add_argument("--epc", nargs="*", default=None)
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    path = os.path.abspath(a.file)
    sd = os.path.dirname(path)
    cap = pd.read_csv(path, dtype={"epc": str})
    cap["epc"] = cap.epc.str.upper()
    ports = sorted(cap.antenna.unique().tolist())
    lay = {}
    if os.path.exists(os.path.join(sd, "layout.json")):
        lay = {k.upper(): int(v[0]) for k, v in json.load(open(os.path.join(sd, "layout.json")))["tags"].items()}
    counts = cap.epc.value_counts()

    if a.all:
        epcs = counts.index.tolist()
    elif a.epc:
        epcs = []
        for e in a.epc:
            m = [x for x in counts.index if x == e.upper() or x.endswith(e.upper())]
            if not m:
                print(f"{e}: not in this file"); continue
            epcs.append(m[0])
    else:                                   # best-read box of each zone
        epcs = []
        for z in sorted(set(lay.values())):
            c = [e for e in counts.index if lay.get(e) == z]
            if c:
                epcs.append(c[0])
        epcs = epcs or counts.index[:4].tolist()

    stem = os.path.splitext(os.path.basename(path))[0]
    od = os.path.join(sd, "analysis", f"raw_{stem}")
    os.makedirs(od, exist_ok=True)
    tags = [(e, e[-3:], lay.get(e, 0)) for e in epcs]
    allS = []
    for e, label, z in tags:
        out = os.path.join(od, f"{stem}__box{label}.png")
        allS.append(tag_figure(cap, e, label, z, ports, out))
        print(f"-> {out}")
    if len(tags) > 1:
        out = os.path.join(od, f"{stem}__compare.png")
        compare_figure(cap, tags, ports, out)
        print(f"-> {out}")
    S = pd.concat(allS, ignore_index=True)
    S.to_csv(os.path.join(od, "raw_stats.csv"), index=False)
    pd.set_option("display.width", 220)
    print(S.round(2).to_string(index=False))
    print(f"\nFile: {path}\n{len(cap)} reads, {cap.epc.nunique()} tags, {cap.freq_mhz.nunique()} channels, "
          f"{(cap.timestamp_ms.max() - cap.timestamp_ms.min()) / 1000:.1f} s")


if __name__ == "__main__":
    main()
