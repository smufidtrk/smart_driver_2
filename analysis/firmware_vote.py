"""
The firmware's Zircon rule vote (trkappspsf SmartDriver_Zone.cpp, sdCOMB_buildZirconVote),
applied to our 4-zone captures (one antenna per zone, antenna p -> zone p).

  per antenna: signal = 0.7 * peak + 0.3 * avg RSSI;  dropped if avg < -65 dBm
               w = 10^(signal / 8) * seen_cnt * (1 - min(1, stddev / 12))      (SD_ALGO_ZIR_DB_X10 = 80)
  p(zone) = sum of w over the zone's antennas / total;  call = argmax
Compared with: strongest antenna (median RSSI), and the old linear weight (signal + 100) * seen.

  python analysis/firmware_vote.py
"""
import glob, json, os, sys
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SESS = ["4zone_cal32", "4zone_cal32_repeat", "4zone_mode2_c", "4zone_mode2_d", "4zone_mode1002_c",
        "4zone_mode1002_d", "4zone_12box", "4zone_centers", "4zone_rot1", "4zone_rot2", "4zone_rot3", "4zone_rot4"]

def vote(g, db=8.0, linear=False, thresh=-65.0, stab=True, use_seen=True):
    w = {}
    for p, a in g.groupby("antenna"):
        avg, peak, sd, n = a.rssi_dbm.mean(), a.rssi_dbm.max(), a.rssi_dbm.std(ddof=0), len(a)
        if avg < thresh: continue
        s = 0.7 * peak + 0.3 * avg
        x = (s + 100.0) if linear else 10 ** (s / db)
        x *= (n if use_seen else 1) * ((1 - min(1.0, sd / 12.0)) if stab else 1)
        w[p] = max(x, 0.0)
    if not w or sum(w.values()) <= 0: return None, None
    tot = sum(w.values()); best = max(w, key=w.get)
    return best, w[best] / tot

def main():
    rows = []
    for s in SESS:
        sd = os.path.join(ROOT, "sessions", s)
        if not os.path.exists(os.path.join(sd, "layout.json")): continue
        lay = {k.upper(): int(v[0]) for k, v in json.load(open(os.path.join(sd, "layout.json")))["tags"].items()}
        for f in sorted(glob.glob(os.path.join(sd, "cap_*-p*.csv"))):
            pw = f.rsplit("-p", 1)[1][:-4]
            d = pd.read_csv(f, dtype={"epc": str}); d["epc"] = d.epc.str.upper()
            for e, z in lay.items():
                g = d[d.epc == e]
                if g.empty: continue
                med = g.groupby("antenna").rssi_dbm.median()
                r = dict(session=s, power=pw, box=e[-3:], zone=z, strongest=int(med.idxmax()))
                r["fw_vote"], r["fw_conf"] = vote(g)
                r["fw_vote_no_threshold"], _ = vote(g, thresh=-999)
                r["linear_weight"], _ = vote(g, linear=True)
                r["no_read_count"], _ = vote(g, use_seen=False)
                rows.append(r)
    D = pd.DataFrame(rows)
    meth = ["strongest", "fw_vote", "fw_vote_no_threshold", "linear_weight", "no_read_count"]
    out = []
    for (s, pw), g in D.groupby(["session", "power"], sort=False):
        out.append(dict(session=s, dBm=pw, boxes=len(g), **{m: f"{(g[m] == g.zone).sum()}/{len(g)}" for m in meth},
                        no_call=int(g.fw_vote.isna().sum())))
    T = pd.DataFrame(out); pd.set_option("display.width", 220)
    print(T.to_string(index=False))
    print("\nall captures:", {m: f"{(D[m] == D.zone).mean():.0%}" for m in meth}, f"| boxes x captures: {len(D)}")
    ok = D.dropna(subset=["fw_vote"])
    for lo, hi in ((0, .5), (.5, .7), (.7, .9), (.9, 1.01)):
        b = ok[(ok.fw_conf >= lo) & (ok.fw_conf < hi)]
        if len(b): print(f"  vote share {lo:.0%}-{min(hi,1):.0%}: n={len(b)}  right {(b.fw_vote == b.zone).mean():.0%}")
    print("agrees with strongest antenna:", f"{(ok.fw_vote == ok.strongest).mean():.0%}")
    print("\nconfusion (rows true zone, cols fw vote):")
    print(pd.crosstab(ok.zone, ok.fw_vote.astype(int)).to_string())


if __name__ == "__main__":
    main()
