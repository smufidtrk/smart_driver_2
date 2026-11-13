"""
Beam-steering check: one reader port enabled at a time (sessions/beam_check/port1..4), 32 boxes in the
calibration layout. Shows, per port, which boxes it reads, their median RSSI, and how much each zone gets.

  python analysis/beam_check.py [session_dir]
"""
import glob, json, os, sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
SD = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "sessions", "beam_check")
lay = json.load(open(os.path.join(SD, "layout.json")))["tags"]
BOX = {k.upper(): dict(box=k[-3:], zone=int(v[0]), y=v[1], x=v[2]) for k, v in lay.items()}
XS, YS = [3, 9, 15, 21, 27, 33, 39, 45], [21, 15, 9, 3]

rows = []
for p in (1, 2, 3, 4):
    files = sorted(glob.glob(os.path.join(SD, f"port{p}", "cap_*.csv")))
    if not files:
        continue
    d = pd.read_csv(files[-1], dtype={"epc": str})
    d["epc"] = d.epc.str.upper()
    secs = (d.timestamp_ms.max() - d.timestamp_ms.min()) / 1000
    other = d[~d.epc.isin(BOX)].epc.nunique()
    for e, b in BOX.items():
        g = d[d.epc == e]
        rows.append(dict(port=p, **b, reads=len(g), rssi=g.rssi_dbm.median() if len(g) else np.nan,
                         peak=g.rssi_dbm.max() if len(g) else np.nan))
    print(f"port {p}: {len(d)} reads in {secs:.0f} s, {d[d.epc.isin(BOX)].epc.nunique()}/32 boxes, {other} other tags")
R = pd.DataFrame(rows)
R.to_csv(os.path.join(SD, "beam_check.csv"), index=False)

for p, g in R.groupby("port"):
    print(f"\n=== port {p} alone: median RSSI per box (dBm, '  --' = not read)   back wall at top, aisle at bottom")
    print("          " + "".join(f"{'Z' + str(int(x // 12) + 1) + ' x' + str(x):>9}" for x in XS))
    for y in YS:
        line = []
        for x in XS:
            v = g[(g.x == x) & (g.y == y)].rssi.iloc[0]
            line.append(f"{v:9.1f}" if np.isfinite(v) else "       --")
        print(f"  y={y:>2}\"  " + "".join(line))
    z = g.groupby("zone").agg(read=("reads", lambda s: int((s > 0).sum())), reads=("reads", "sum"), rssi=("rssi", "median"))
    print("  per zone: " + " | ".join(f"Z{i}: {r.read}/8 boxes, {r.reads} reads, median {r.rssi:.1f} dBm" for i, r in z.iterrows()))

# which port is strongest per box, and how far ahead
P = R.pivot_table(index=["box", "zone", "x", "y"], columns="port", values="rssi").reset_index()
ports = [c for c in P.columns if isinstance(c, (int, np.integer))]
best = P[ports].idxmax(axis=1)
srt = np.sort(P[ports].fillna(-99).values, axis=1)
P["best_port"], P["margin_db"] = best, srt[:, -1] - srt[:, -2]
zmap = P.groupby("zone").best_port.agg(lambda s: s.value_counts().idxmax())
print("\nstrongest port per box (margin over the next port, dB):")
for y in YS:
    print(f"  y={y:>2}\"  " + "".join(
        f"{('P' + str(int(r.best_port)) + f' {r.margin_db:+.1f}'):>9}" for _, r in P[P.y == y].sort_values("x").iterrows()))
print("most boxes in each zone are strongest on: " + ", ".join(f"Z{z}->P{p}" for z, p in zmap.items()))
pz = {z: p for z, p in zmap.items()}
ok = sum(P.best_port == P.zone.map(pz))
print(f"if each zone's port is taken as its beam, strongest port = right zone for {ok}/32 boxes")
P.to_csv(os.path.join(SD, "beam_check_strongest.csv"), index=False)
