"""
RF-mode comparison on the 32-box calibration layout.

  python analysis/mode_compare.py --modes 1002 4180 4189

Expects sessions/4zone_mode<M>_a and _b (same boxes, same spots, captured
interleaved), each with the calibration layout.json.  Per mode and power:
  reads       boxes read (of 32), median reads per box, median channels per antenna
  noise       median per-channel phase spread (deg, mod 180) on the box's own antenna / all
  repeat      pass b vs pass a: own-antenna pattern match (median, 10th percentile),
              boxes whose most similar pass-a pattern is their own spot
  zone calls  pass b looked up against pass a, and against 4zone_cal32 (yesterday, mode 1002):
              phase pattern, RSSI lookup, strongest antenna
"""
import argparse, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import shelfloc as sl  # noqa: E402
import calibration as C  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S = lambda n: os.path.join(ROOT, "sessions", n)


def noise_and_channels(sd, cfg, pw):
    _, man = sl.load_session(sd)
    r = man[man.tx_power_dbm.astype(float) == pw]
    if r.empty:
        return {}
    cap = sl.load_capture(os.path.join(sd, r.file.iloc[0]))
    f = sl.capture_features(cap, cfg)
    return {(e, int(p)): (g.phase_sd_mod180, g.n_ch) for e, p, g in
            ((row.epc, row.port, row) for row in f.itertuples())}


def call(r, lib, ports, how):
    if how == "phase":
        sc = [C.mean_match(r["sig"], c["sig"], ports) for c in lib]
        return lib[int(np.argmax(sc))]["zone"], lib[int(np.argmax(sc))]["box"]
    if how == "rssi":
        d = [np.linalg.norm(c["rel"] - r["rel"]) for c in lib]
        idx = np.argsort(d)[:3]
        return int(pd.Series([lib[i]["zone"] for i in idx]).mode().iloc[0]), None
    return r["strongest"], None


def prep(rs, cfg):
    PZ = np.array([int(sl.zone_names(cfg)[cfg["port_zone"][p]]) for p in cfg["ports"]])
    for r in rs:
        v = np.where(np.isnan(r["rssi"]), C.FLOOR, r["rssi"])
        r["rel"] = v - v.max()
        r["strongest"] = int(PZ[np.argmax(v)]) if np.isfinite(r["rssi"]).any() else 0
    return rs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=["1002", "4180", "4189"])
    ap.add_argument("--passes", nargs=2, default=["a", "b"], help="session suffixes: lookup pass, test pass")
    a = ap.parse_args()
    cfg, _ = sl.load_session(S("4zone_cal32"))
    ports = cfg["ports"]
    yesterday = prep(C.load_readings(S("4zone_cal32"), cfg), cfg)
    rows = []
    for m in a.modes:
        sa, sb = S(f"4zone_mode{m}_{a.passes[0]}"), S(f"4zone_mode{m}_{a.passes[1]}")
        if not (os.path.exists(os.path.join(sa, "manifest.csv")) and os.path.exists(os.path.join(sb, "manifest.csv"))):
            print(f"mode {m}: missing session(s) - skipped")
            continue
        A = prep(C.load_readings(sa, cfg), cfg)
        B = prep(C.load_readings(sb, cfg), cfg)
        for pw in sorted({r["power"] for r in B}):
            Ap = [r for r in A if r["power"] == pw]
            Bp = [r for r in B if r["power"] == pw]
            Yp = [r for r in yesterday if r["power"] == pw]
            byA = {r["box"]: r for r in Ap}
            nz = noise_and_channels(sb, cfg, pw)
            own_noise, all_noise, chans, read_n = [], [], [], []
            for r in Bp:
                read_n.append(int(r["reads"].sum()))
                for p in ports:
                    v = nz.get((r["epc"], p))
                    if v:
                        all_noise.append(v[0]); chans.append(v[1])
                        if int(sl.zone_names(cfg)[cfg["port_zone"][p]]) == r["zone"]:
                            own_noise.append(v[0])
            rep_own, self_hits = [], 0
            hits = {k: 0 for k in ("A phase", "A rssi", "Y phase", "Y rssi", "strongest")}
            for r in Bp:
                ra = byA.get(r["box"])
                if ra:
                    rep_own.append(sl.sig_match(ra["sig"].get(r["zone"]), r["sig"].get(r["zone"])))
                if Ap:
                    z, b = call(r, Ap, ports, "phase"); hits["A phase"] += z == r["zone"]; self_hits += b == r["box"]
                    hits["A rssi"] += call(r, Ap, ports, "rssi")[0] == r["zone"]
                if Yp:
                    hits["Y phase"] += call(r, Yp, ports, "phase")[0] == r["zone"]
                    hits["Y rssi"] += call(r, Yp, ports, "rssi")[0] == r["zone"]
                hits["strongest"] += call(r, None, ports, "strongest")[0] == r["zone"]
            n = len(Bp)
            rep_own = np.array([x for x in rep_own if np.isfinite(x)])
            rows.append({"mode": f"{m} ({a.passes[0]}/{a.passes[1]})", "power": pw,
                         "boxes read": f"{sum(1 for r in Bp if r['reads'].sum() > 0)}/32",
                         "reads/box": int(np.median(read_n)) if read_n else 0,
                         "channels/ant": int(np.median(chans)) if chans else 0,
                         "noise own°": round(float(np.median(own_noise)), 1) if own_noise else np.nan,
                         "noise all°": round(float(np.median(all_noise)), 1) if all_noise else np.nan,
                         "repeat own (med)": round(float(np.median(rep_own)), 2) if len(rep_own) else np.nan,
                         "repeat own (p10)": round(float(np.percentile(rep_own, 10)), 2) if len(rep_own) else np.nan,
                         "own spot best": f"{self_hits}/{n}",
                         "zone: phase vs a": f"{hits['A phase']}/{n}", "zone: RSSI vs a": f"{hits['A rssi']}/{n}",
                         "zone: phase vs yday": f"{hits['Y phase']}/{n}", "zone: RSSI vs yday": f"{hits['Y rssi']}/{n}",
                         "zone: strongest": f"{hits['strongest']}/{n}"})
    R = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(R.to_string(index=False))
    od = os.path.join(ROOT, "sessions", "mode_compare")
    os.makedirs(od, exist_ok=True)
    R.to_csv(os.path.join(od, "mode_compare.csv"), index=False)
    print(f"\n-> {os.path.join(od, 'mode_compare.csv')}")


if __name__ == "__main__":
    main()
