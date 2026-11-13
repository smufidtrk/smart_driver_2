"""
Phase slope across frequency as a smooth "distance to each antenna" feature.

  python analysis/phase_slope.py

For every box with a known [zone, y, x] layout and every antenna: the phase slope (deg/MHz)
across the hop channels, taken as the MEDIAN of channel-to-channel steps of the unwrapped
(doubled -> 180-deg-flip-safe) phase, so a few multipath kinks don't bend it.
A longer path -> a steeper (more negative) slope.  Cable/reader delay is the same for every tag
on one antenna, so it is removed by subtracting that antenna's median slope over all boxes
(in the field: a reference tag per antenna would do this).

Reports, per capture: rank correlation of slope with |x - antenna x| per antenna, how well slope
repeats between captures, and the zone call "antenna with the shortest relative path"
vs "strongest relative RSSI" vs both.  No lookup table is used.
Writes sessions/phase_slope/phase_slope.csv.
"""
import glob, json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from raw_plot import chan_table  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XA = {1: 6, 2: 18, 3: 30, 4: 42}      # antenna x (in) = zone centres
SESSIONS = ["4zone_cal32", "4zone_cal32_repeat", "4zone_mode2_c", "4zone_mode2_d", "4zone_mode1002_c", "4zone_mode1002_d"]


def robust_slope(ch):
    f = ch.index.values
    ph = np.degrees(np.unwrap(np.radians(ch.phase.values) * 2) / 2)
    df, dp = np.diff(f), np.diff(ph)
    m = (df > 0) & (df <= 1.01)          # neighbouring channels only (0.5-1 MHz apart)
    return float(np.median(dp[m] / df[m])) if m.sum() > 5 else np.nan


def rank_corr(a, b):
    return pd.Series(np.asarray(a)).rank().corr(pd.Series(np.asarray(b)).rank())


def main():
    rows = []
    for s in SESSIONS:
        sd = os.path.join(ROOT, "sessions", s)
        if not os.path.exists(os.path.join(sd, "layout.json")):
            continue
        lay = json.load(open(os.path.join(sd, "layout.json")))["tags"]
        if not all(len(v) > 2 for v in lay.values()):
            continue
        for f in sorted(glob.glob(os.path.join(sd, "cap_*-p*.csv"))):
            pw = f.rsplit("-p", 1)[1][:-4]
            d = pd.read_csv(f, dtype={"epc": str})
            d["epc"] = d.epc.str.upper()
            for e, v in lay.items():
                for p in XA:
                    g = d[(d.epc == e.upper()) & (d.antenna == p)]
                    if len(g) < 20:
                        continue
                    rows.append(dict(capture=f"{s} @{pw}", box=e[-3:], zone=int(v[0]), y=v[1], x=v[2], ant=p,
                                     slope=robust_slope(chan_table(g)), rssi=g.rssi_dbm.median()))
    R = pd.DataFrame(rows)
    R["dx"] = (R.x - R.ant.map(XA)).abs()
    out = []
    for cap, G in R.groupby("capture", sort=False):
        G = G.copy()
        G["rel_slope"] = (G.slope - G.groupby("ant").slope.transform("median")) / G.slope.std()
        G["rel_rssi"] = (G.rssi - G.groupby("ant").rssi.transform("median")) / G.rssi.std()
        G["both"] = G.rel_slope + G.rel_rssi
        res = dict(capture=cap, boxes=G.box.nunique(),
                   **{f"slope_vs_dx_A{p}": round(rank_corr(g.slope, g.dx), 2) for p, g in G.groupby("ant")})
        for k, name in (("rel_slope", "call_slope"), ("rel_rssi", "call_rssi"), ("both", "call_both")):
            c = G.loc[G.groupby("box")[k].idxmax()]
            res[name] = f"{(c.ant == c.zone).sum()}/{len(c)}"
        out.append(res)
    T = pd.DataFrame(out)
    pd.set_option("display.width", 220)
    print(T.to_string(index=False))
    P = R.pivot_table(index=["box", "ant"], columns="capture", values="slope")
    print("\nSlope repeat between captures (correlation over box x antenna):")
    print(P.corr().round(2).to_string())
    od = os.path.join(ROOT, "sessions", "phase_slope")
    os.makedirs(od, exist_ok=True)
    R.to_csv(os.path.join(od, "phase_slope.csv"), index=False)
    T.to_csv(os.path.join(od, "phase_slope_summary.csv"), index=False)
    print(f"\n-> {od}")


if __name__ == "__main__":
    main()
