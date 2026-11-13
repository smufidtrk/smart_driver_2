"""
What if some antennas were removed?  Re-runs the zone calls using only a subset of
the 4 antennas (the R700 reads one antenna at a time, so the remaining antennas'
data is what they would have heard anyway; they would only get more reads).

  python analysis/antenna_subsets.py

Tests (each box called against a lookup table of calibration boxes at known spots):
  same spot, minutes   4zone_mode2_d   vs 4zone_cal32 + 4zone_cal32_repeat (also mode2_c), 15/20 dBm
  same spot, overnight 4zone_cal32_repeat vs 4zone_cal32
  6" neighbours        4zone_cal32, each box vs the other 31 (leave-one-out)
  ~4" off-spot         4zone_12box vs 4zone_cal32 + 4zone_cal32_repeat
Methods: phase pattern (nearest box), RSSI pattern (3 nearest), combined, strongest antenna.
"""
import os, sys
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
cfg, _ = sl.load_session(S("4zone_cal32"))
PORTS = cfg["ports"]
PZ = {p: int(sl.zone_names(cfg)[cfg["port_zone"][p]]) for p in PORTS}
_cache = {}


def readings(name):
    if name not in _cache:
        _cache[name] = C.load_readings(S(name), cfg)
    return _cache[name]


def call(t, lib, ports):
    idx = [PORTS.index(p) for p in ports]
    tv = np.where(np.isnan(t["rssi"][idx]), C.FLOOR, t["rssi"][idx])
    trel = tv - tv.max()
    ph, rd = [], []
    for c in lib:
        m = [sl.sig_match(t["sig"].get(p), c["sig"].get(p)) for p in ports]
        m = [x for x in m if np.isfinite(x)]
        ph.append(np.mean(m) if m else -1)
        cv = np.where(np.isnan(c["rssi"][idx]), C.FLOOR, c["rssi"][idx])
        rd.append(np.linalg.norm((cv - cv.max()) - trel))
    ph, rd = np.array(ph), np.array(rd)
    zones = np.array([c["zone"] for c in lib])
    z_phase = zones[int(np.argmax(ph))]
    z_rssi = int(pd.Series(zones[np.argsort(rd)[:3]]).mode().iloc[0])
    comb = (1 - ph) / max(np.median(1 - ph), 1e-9) + 0.5 * rd / max(np.median(rd), 1e-9)
    z_comb = zones[int(np.argmin(comb))]
    z_str = PZ[ports[int(np.argmax(tv))]] if np.isfinite(t["rssi"][idx]).any() else 0
    return dict(phase=z_phase, rssi=z_rssi, combined=z_comb, strongest=z_str)


def run(test_name, tests, lib_fn, subsets):
    rows = []
    for sub in subsets:
        hits = {k: 0 for k in ("phase", "rssi", "combined", "strongest")}
        n = 0
        for t in tests:
            lib = lib_fn(t)
            if not lib or not np.isfinite(t["rssi"]).any():
                continue
            r = call(t, lib, sub)
            n += 1
            for k in hits:
                hits[k] += r[k] == t["zone"]
        rows.append(dict(test=test_name, antennas="+".join(map(str, sub)), n=n,
                         **{k: f"{v}/{n} ({100 * v / n:.0f}%)" for k, v in hits.items()}))
    return rows


def main():
    subsets = [[1, 2, 3, 4], [1, 4], [2, 3], [1, 2, 4], [1, 3, 4], [1, 3], [2, 4]]
    out = []
    for pw in (20.0, 15.0):
        look = [r for n in ("4zone_cal32", "4zone_cal32_repeat") for r in readings(n) if r["power"] == pw]
        for name, sess in (("same spot, minutes", "4zone_mode2_d"), ("~4in off-spot (12-box)", "4zone_12box")):
            tests = [r for r in readings(sess) if r["power"] == pw]
            if tests:
                out += run(f"{name} @{pw:g}", tests, lambda t: look, subsets)
        lib_y = [r for r in readings("4zone_cal32") if r["power"] == pw]
        out += run(f"same spot, overnight @{pw:g}", [r for r in readings("4zone_cal32_repeat") if r["power"] == pw],
                   lambda t: lib_y, subsets)
        out += run(f"6in neighbours (LOO) @{pw:g}", lib_y, lambda t: [c for c in lib_y if c is not t], subsets)
    R = pd.DataFrame(out)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.max_rows", 200)
    for t, g in R.groupby("test", sort=False):
        print(f"\n== {t}")
        print(g.drop(columns="test").to_string(index=False))
    od = S("antenna_subsets")
    os.makedirs(od, exist_ok=True)
    R.to_csv(os.path.join(od, "antenna_subsets.csv"), index=False)
    print(f"\n-> {os.path.join(od, 'antenna_subsets.csv')}")


if __name__ == "__main__":
    main()
