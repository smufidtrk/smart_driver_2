"""
Per-antenna histograms for one tag in one capture: RSSI of every read, and the distance
each read's RSSI implies from that antenna.

  python analysis/tag_distance_hist.py                                  (tag 261, newest UI scan of it)
  python analysis/tag_distance_hist.py --file sessions/ui_runs/20261112_130952/cap_scan_20261112_130953.csv --epc 261
  python analysis/tag_distance_hist.py --file sessions/silion_test/cap_sdcap_s0-p20.csv --epc 261 --power 20

Distance model, fitted per antenna on the calibration boxes (known x, y) at the same power:
  median RSSI = a + b * log10(d),   d = sqrt((x - antenna x)^2 + y^2 + H^2)  inches
(antenna at the aisle edge, H = height of the antenna above the tag, --height, default 12").
Each read's distance = 10^((RSSI - a) / b).  The fit's scatter (dB) is shown: multipath on
this shelf makes it large, so read the histograms as "near / mid / far", not inches.
Also marked: the horizontal distance the antenna's phase slope implies (locator.slope_model).
Writes <capture folder>/analysis/dist_hist_<box>.png and .csv.
"""
import argparse, glob, json, os, sys
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
ANT_X = {1: 6.0, 2: 18.0, 3: 30.0, 4: 42.0}
ZC = {1: "#1f77b4", 2: "#2ca02c", 3: "#ff7f0e", 4: "#d62728"}


def default_file(tail):
    files = sorted(glob.glob(os.path.join(ROOT, "sessions", "ui_runs", "*", "cap_*.csv")), reverse=True)
    for f in files:
        with open(f, encoding="utf-8", errors="replace") as fh:
            if tail.upper() in fh.read().upper():
                return f
    raise SystemExit(f"no UI scan contains ...{tail}")


def power_of(path, cap_dir):
    b = os.path.basename(path)
    if "-p" in b:
        try:
            return float(b.rsplit("-p", 1)[1].split(".")[0])
        except ValueError:
            pass
    mf = os.path.join(cap_dir, "manifest.csv")
    if os.path.exists(mf):
        m = pd.read_csv(mf)
        hit = m[m.file == b]
        if len(hit):
            return float(hit.tx_power_dbm.iloc[0])
    return None


def fit_models(power, height, sessions=("4zone_cal32", "4zone_cal32_repeat")):
    rows = []
    for s in sessions:
        sd = os.path.join(ROOT, "sessions", s)
        lay = {k.upper(): v for k, v in json.load(open(os.path.join(sd, "layout.json")))["tags"].items()}
        f = glob.glob(os.path.join(sd, f"cap_*-p{power:g}.csv"))
        if not f:
            continue
        d = pd.read_csv(f[0], dtype={"epc": str}); d["epc"] = d.epc.str.upper()
        med = d[d.epc.isin(lay)].groupby(["epc", "antenna"]).rssi_dbm.median().reset_index()
        for r in med.itertuples():
            z, y, x = lay[r.epc][0], float(lay[r.epc][1]), float(lay[r.epc][2])
            dd = np.sqrt((x - ANT_X[r.antenna]) ** 2 + y ** 2 + height ** 2)
            rows.append(dict(antenna=r.antenna, d=dd, rssi=r.rssi_dbm))
    R = pd.DataFrame(rows)
    if R.empty:
        raise SystemExit(f"no calibration capture at {power:g} dBm")
    m = {}
    for p, g in R.groupby("antenna"):
        b, a = np.polyfit(np.log10(g.d), g.rssi, 1)
        m[p] = dict(a=a, b=b, sd=float(np.std(g.rssi - (a + b * np.log10(g.d)))), n=len(g))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file")
    ap.add_argument("--epc", default="261")
    ap.add_argument("--power", type=float, help="tx power of the capture (read from the file/manifest if omitted)")
    ap.add_argument("--height", type=float, default=12.0, help="antenna height above the tag, inches")
    a = ap.parse_args()
    path = os.path.abspath(a.file) if a.file else default_file(a.epc)
    cap_dir = os.path.dirname(path)
    power = a.power or power_of(path, cap_dir)
    if power is None:
        raise SystemExit("can't tell the capture's power: pass --power")
    d = pd.read_csv(path, dtype={"epc": str}); d["epc"] = d.epc.str.upper()
    hit = [e for e in d.epc.unique() if e.endswith(a.epc.upper())]
    if not hit:
        raise SystemExit(f"...{a.epc} is not in {path}")
    epc = hit[0]; t = d[d.epc == epc].copy()
    M = fit_models(power, a.height)
    other_reader = "end_phase_deg" in d.columns          # sdcap (Silion) capture
    if other_reader:
        print("NOTE: this is a Silion capture; the distance fit and phase-slope model come from R700 "
              "captures (different RSSI scale and cable delays), so the distances are NOT valid. "
              "The RSSI histograms are.")

    # phase-slope horizontal distance per antenna (locator's model), if the reads allow it
    slope_dx = {}
    try:
        if other_reader:
            raise RuntimeError("R700 model, Silion capture")
        import locator as L, shelfloc as sl
        lib = L.library(["4zone_cal32", "4zone_cal32_repeat"], power)
        sm = L.slope_model(lib)
        if sm:
            sig = sl.capture_signatures(d, epc, L.PORTS)
            for p in L.PORTS:
                s = L.sig_slope(sig.get(p))
                if np.isfinite(s):
                    slope_dx[p] = float(np.clip((s - sm[p]["c"]) / sm[p]["k"], 0, 60))
    except Exception as ex:  # noqa: BLE001  (slope marker is optional)
        print("phase-slope distance skipped:", ex)

    ports = [1, 2, 3, 4]
    t["dist_in"] = np.nan
    for p in ports:
        if p in M:
            k = t.antenna == p
            t.loc[k, "dist_in"] = np.clip(10 ** ((t.loc[k, "rssi_dbm"] - M[p]["a"]) / M[p]["b"]), a.height, 120)

    fig, ax = plt.subplots(2, 4, figsize=(17, 7.5), sharey="row")
    rb = np.arange(np.floor(t.rssi_dbm.min()) - 1, np.ceil(t.rssi_dbm.max()) + 2, 1.0)
    db = np.arange(a.height, 85, 3.0)
    summary = []
    for j, p in enumerate(ports):
        g = t[t.antenna == p]
        a0, a1 = ax[0, j], ax[1, j]
        a0.set_title(f"Antenna {p} (zone {p})\n{len(g)} reads, {g.freq_mhz.nunique()} ch", fontsize=10)
        if g.empty:
            for x in (a0, a1):
                x.text(0.5, 0.5, "not read", ha="center", transform=x.transAxes, color="#888")
            summary.append(dict(antenna=p, reads=0)); continue
        a0.hist(g.rssi_dbm, bins=rb, color=ZC[p], alpha=0.85)
        a0.axvline(g.rssi_dbm.median(), color="k", lw=1)
        a0.set_xlabel("RSSI (dBm)")
        a1.hist(g.dist_in, bins=db, color=ZC[p], alpha=0.85)
        med = float(g.dist_in.median())
        a1.axvline(med, color="k", lw=1.2, label=f"median {med:.0f}\"")
        lo, hi = np.percentile(g.dist_in, [10, 90])
        if p in slope_dx:
            a1.axvline(slope_dx[p], color="k", ls="--", lw=1, label=f"phase slope |dx| {slope_dx[p]:.0f}\"")
        a1.set_xlabel("distance from antenna implied by RSSI (in)")
        a1.legend(fontsize=8, loc="upper right")
        a1.text(0.02, 0.97, f"fit scatter ±{M[p]['sd']:.1f} dB", transform=a1.transAxes, va="top", fontsize=8)
        summary.append(dict(antenna=p, reads=len(g), channels=g.freq_mhz.nunique(), rssi_median=g.rssi_dbm.median(),
                            rssi_p10=g.rssi_dbm.quantile(.1), rssi_p90=g.rssi_dbm.quantile(.9),
                            dist_median_in=round(med, 1), dist_p10_in=round(lo, 1), dist_p90_in=round(hi, 1),
                            slope_dx_in=round(slope_dx[p], 1) if p in slope_dx else None,
                            fit=f"RSSI = {M[p]['a']:.1f} {M[p]['b']:+.1f} log10(d), scatter {M[p]['sd']:.1f} dB"))
    ax[0, 0].set_ylabel("reads"); ax[1, 0].set_ylabel("reads")
    rel = os.path.relpath(path, ROOT)
    fig.suptitle(f"Tag ...{epc[-6:]} — {rel} ({power:g} dBm)\nRSSI per read and the distance it implies "
                 f"(per-antenna fit on the 32 calibration boxes, antenna {a.height:g}\" above the tag)", fontsize=11)
    if other_reader:
        fig.text(0.5, 0.47, "Silion capture: distances use the R700 calibration and are not valid",
                 ha="center", color="#c00", fontsize=12, weight="bold")
    fig.tight_layout()
    od = os.path.join(cap_dir, "analysis"); os.makedirs(od, exist_ok=True)
    out = os.path.join(od, f"dist_hist_{epc[-3:]}.png")
    fig.savefig(out, dpi=110); plt.close(fig)
    S = pd.DataFrame(summary); S.to_csv(out[:-4] + ".csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 60)
    print(S.to_string(index=False)); print("->", out)


if __name__ == "__main__":
    main()
