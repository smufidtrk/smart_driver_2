"""
Silion E710 (Sapphire, sdcap) vs Impinj R700 on the same 32-box calibration grid.

  python analysis/reader_compare.py

Test sets (each box called against a lookup of calibration boxes with known zones):
  6" neighbours  each calibration box vs the other 31 of the same capture (leave-one-out)
  same spot      a later capture, boxes untouched, vs the calibration capture
  overnight      (R700 only) next-day capture vs the calibration
Methods (all per box):
  strongest      antenna with the highest median RSSI
  fw vote        the firmware's Zircon vote (10^(signal/8) x reads x stability, analysis/firmware_vote.py)
  WCL RSSI       weighted centroid, weights = RSSI in mW, zone by the 12" edges
  WCL slope+RSSI weighted centroid, weights = exp(relative slope) x 10^(relative RSSI / 10)
  slope          antenna with the shortest relative phase-slope path
  RSSI lookup    closest relative-RSSI vector among the lookup boxes
  phase lookup   best phase-pattern match among the lookup boxes
  locator        the Zone Locator's combined call (phase + RSSI gated by match, + slope)
Also signal quality per capture: reads, channels per box per antenna, phase noise, RSSI spread.
Border-tolerant score: a box in a column next to a zone line also counts as right when called the zone
across that line (box 655 at x=9" may be Z1 or Z2); the shelf-end columns (x=3", 45") must be exact.
Writes sessions/reader_compare/reader_compare.csv, reader_compare_border.csv, quality.csv and per_box.csv.
"""
import glob, json, os, sys, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools", "zone_ui"))
sys.path.insert(0, os.path.join(ROOT, "analysis"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
import locator as L  # noqa: E402
from firmware_vote import vote  # noqa: E402

S = lambda n: os.path.join(ROOT, "sessions", n)
ANT_X = np.array([L.ANT_X[p] for p in L.PORTS])
EDGES = [12, 24, 36]


def near_zone(zone, x):
    """The zone just across the line a box sits next to (within 6"), or None for the shelf-end columns.
    Border-tolerant scoring accepts a call of the box's own zone or this one."""
    for e in EDGES:
        if abs(x - e) <= 6:
            return zone + 1 if x < e else zone - 1
    return None

# (reader, test name, test session, power, lookup sessions, leave-one-out)
SETS = [
    ("R700", "6in neighbours", "4zone_cal32", 12.0, ["4zone_cal32"], True),
    ("R700", "6in neighbours", "4zone_cal32", 15.0, ["4zone_cal32"], True),
    ("R700", "same spot, minutes", "4zone_mode2_d", 15.0, ["4zone_mode2_c"], False),          # same mode (2) both
    ("R700", "same spot, minutes", "4zone_mode2_d", 20.0, ["4zone_mode2_c"], False),
    ("R700", "same spot, other mode", "4zone_mode2_d", 15.0, ["4zone_cal32", "4zone_cal32_repeat"], False),  # mode 2 vs 1002
    ("R700", "same spot, overnight", "4zone_cal32_repeat", 12.0, ["4zone_cal32"], False),
    ("R700", "same spot, overnight", "4zone_cal32_repeat", 15.0, ["4zone_cal32"], False),
    ("Silion", "6in neighbours", "silion_cal32", 12.0, ["silion_cal32"], True),
    ("Silion", "same spot, minutes", "silion_test_s0", 12.0, ["silion_cal32"], False),
]
METHODS = ["strongest", "fw vote", "WCL RSSI", "WCL slope+RSSI", "slope", "RSSI lookup", "phase lookup", "locator"]


def raw_capture(session, power):
    sd = S(session)
    man = pd.read_csv(os.path.join(sd, "manifest.csv"))
    m = man[man.tx_power_dbm.astype(float) == power]
    if m.empty:
        return None
    d = pd.read_csv(os.path.join(sd, m.file.iloc[-1]), dtype={"epc": str})
    d["epc"] = d.epc.str.upper()
    return d


def wcl(w):
    w = np.where(np.isfinite(w), w, 0.0)
    if w.sum() <= 0:
        return None
    x = float((w * ANT_X).sum() / w.sum())
    return int(np.searchsorted(EDGES, x, side="right")) + 1


def calls(t, lib, raw):
    t = {**t, "channels": t.get("channels", [99] * 4)}
    r = L.classify(t, lib)
    zones = r["zones"]
    pick = lambda p: None if p is None or any(v is None for v in p) else zones[int(np.argmax(p))]
    out = {"strongest": r["strongest"], "slope": pick(r["slope"]), "RSSI lookup": pick(r["rssi"]),
           "phase lookup": pick(r["phase"]), "locator": r["call"]}
    g = raw[raw.epc == t["epc"]] if raw is not None else pd.DataFrame()
    out["fw vote"] = vote(g)[0] if len(g) else None
    rssi = np.where(np.isfinite(t["rssi"]), t["rssi"], np.nan)
    out["WCL RSSI"] = wcl(10 ** (rssi / 10))
    med_s, sd_s = L.slope_reference(lib)
    rel_s = (L._slopes(t) - med_s) / sd_s
    lib_r = np.nanmedian(np.array([c["rssi"] for c in lib]), axis=0)
    w = np.exp(np.where(np.isfinite(rel_s), rel_s, -3)) * 10 ** ((rssi - lib_r) / 10)
    out["WCL slope+RSSI"] = wcl(w)
    out["best_match"] = r["best_match"]
    return out


def quality(reader, session, power):
    d = raw_capture(session, power)
    lay = {k.upper() for k in json.load(open(os.path.join(S(session), "layout.json")))["tags"]}
    d = d[d.epc.isin(lay)]
    ch = d.groupby(["epc", "antenna"]).freq_mhz.nunique()
    z = d.groupby(["epc", "antenna", "freq_mhz"]).phase_deg.apply(
        lambda a: abs(np.mean(np.exp(2j * np.radians(a.values)))) if len(a) >= 2 else np.nan).dropna()
    noise = np.degrees(np.sqrt(-2 * np.log(z.clip(1e-9, 1)))) / 2
    rs = d.groupby(["epc", "antenna", "freq_mhz"]).rssi_dbm.std(ddof=0).dropna()
    secs = (d.timestamp_ms.max() - d.timestamp_ms.min()) / 1000
    return dict(reader=reader, capture=f"{session} @{power:g}", seconds=round(secs),
                reads_per_box=round(len(d) / max(1, d.epc.nunique())), boxes_read=d.epc.nunique(),
                channels_per_box_ant=float(ch.median()), weak_pairs_lt6ch=int((ch < 6).sum() + (128 - len(ch))),
                phase_noise_deg=round(float(noise.median()), 2), rssi_spread_db=round(float(rs.median()), 2))


def main():
    rows, rows_b, per_box, qual = [], [], [], []
    seen_q = set()
    for reader, name, sess, pw, look, loo in SETS:
        if not os.path.exists(os.path.join(S(sess), "manifest.csv")):
            print(f"skip {reader} {name}: {sess} not captured yet")
            continue
        tests = L.library([sess], pw)
        if not tests:
            print(f"skip {reader} {name}: no {pw:g} dBm capture"); continue
        lib_all = L.library(look, pw)
        raw = raw_capture(sess, pw)
        hits = {m: 0 for m in METHODS}
        hits_b = {m: 0 for m in METHODS}
        for t in tests:
            lib = [c for c in lib_all if c is not t] if loo else [c for c in lib_all if c["session"] != t["session"]]
            c = calls(t, lib, raw)
            nz = near_zone(t["zone"], t["x"])
            for m in METHODS:
                hits[m] += c[m] == t["zone"]
                hits_b[m] += c[m] == t["zone"] or (nz is not None and c[m] == nz)
            per_box.append(dict(reader=reader, test=name, power=pw, box=t["box"], zone=t["zone"], x=t["x"], y=t["y"],
                                **{m: c[m] for m in METHODS}, best_match=c["best_match"],
                                rssi=json.dumps([None if not np.isfinite(v) else round(float(v), 1) for v in t["rssi"]])))
        n = len(tests)
        rows.append(dict(reader=reader, test=name, dBm=f"{pw:g}", boxes=n, **{m: f"{hits[m]}/{n}" for m in METHODS}))
        rows_b.append(dict(reader=reader, test=name, dBm=f"{pw:g}", boxes=n, **{m: f"{hits_b[m]}/{n}" for m in METHODS}))
        for s in {sess, *look}:
            if (s, pw) not in seen_q:
                seen_q.add((s, pw)); qual.append(quality(reader, s, pw))
    R, Q = pd.DataFrame(rows), pd.DataFrame(qual)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    RB = pd.DataFrame(rows_b)
    print(R.to_string(index=False)); print()
    print("border-tolerant (a box next to a zone line may also be called the zone across it):")
    print(RB.to_string(index=False)); print(); print(Q.to_string(index=False))
    od = S("reader_compare"); os.makedirs(od, exist_ok=True)
    R.to_csv(os.path.join(od, "reader_compare.csv"), index=False)
    RB.to_csv(os.path.join(od, "reader_compare_border.csv"), index=False)
    Q.to_csv(os.path.join(od, "quality.csv"), index=False)
    pd.DataFrame(per_box).to_csv(os.path.join(od, "per_box.csv"), index=False)
    print("->", od)


if __name__ == "__main__":
    main()
