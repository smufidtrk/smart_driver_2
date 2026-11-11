"""
Calibration-grid analysis: many boxes at known spots in ONE capture.  Is there a pattern
in RSSI or phase that tells the 4 zones apart?

  python analysis/calibration.py sessions/4zone_cal32
  python analysis/calibration.py sessions/4zone_cal32 --test sessions/4zone_12box

Each session folder has layout.json  {"tags": {"EPC": ["zone", y_in, x_in], ...}}
(x_in optional: default = zone center).

Zone calls compared (each box is called WITHOUT its own reading: leave-one-out inside
the calibration capture; with --test, the calibration capture is the lookup table):
  strongest antenna    zone of the antenna with the highest RSSI (no calibration)
  RSSI-weighted x      x = sum(w_i * x_antenna_i) / sum(w_i), w = RSSI in mW;
                       zone by the fixed 12" edges (no calibration)
  RSSI-weighted x, cal same x, but the zone of the 3 calibration boxes with the nearest x
  RSSI lookup (kNN)    3 nearest calibration boxes by relative RSSI (RSSI - box max)
  phase pattern (kNN)  calibration box with the most similar per-channel phase pattern
  phase + RSSI (kNN)   both distances, each scaled to its median
Writes <session>/analysis/calibration.csv, calibration.png
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import shelfloc as sl  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

FLOOR = -80.0   # RSSI used when an antenna did not read the box


def load_readings(sd, cfg):
    names = sl.zone_names(cfg)
    edges = cfg["zone_edges"] / sl.IN
    lay = json.load(open(os.path.join(sd, "layout.json")))["tags"]
    _, man = sl.load_session(sd)
    out = []
    for r in man.itertuples():
        f = os.path.join(sd, r.file)
        if not os.path.exists(f):
            continue
        cap = sl.load_capture(f)
        for e, v in lay.items():
            e = e.upper()
            z = str(v[0]); k = names.index(z)
            x = float(v[2]) if len(v) > 2 else (edges[k] + edges[k + 1]) / 2
            g = cap[cap.epc == e]
            st = g.groupby("antenna").rssi_dbm.agg(["median", "size"])
            out.append(dict(session=os.path.basename(os.path.normpath(sd)), power=float(r.tx_power_dbm),
                            epc=e, box=e[-3:], zone=int(z), x=x, y=float(v[1]),
                            rssi=np.array([float(st["median"][p]) if p in st.index else np.nan for p in cfg["ports"]]),
                            reads=np.array([int(st["size"][p]) if p in st.index else 0 for p in cfg["ports"]]),
                            sig=sl.capture_signatures(cap, e, cfg["ports"])))
    return out


def mean_match(a, b, ports):
    v = [sl.sig_match(a.get(p), b.get(p)) for p in ports]
    return float(np.nanmean(v)) if np.isfinite(v).any() else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--test", nargs="*", default=[])
    ap.add_argument("--k", type=int, default=3)
    a = ap.parse_args()

    cfg, _ = sl.load_session(a.session)
    ports = cfg["ports"]
    PZ = np.array([int(sl.zone_names(cfg)[cfg["port_zone"][p]]) for p in ports])
    AX = np.array([cfg["ants"][p][0] / sl.IN for p in ports])
    edges = cfg["zone_edges"] / sl.IN
    cal = load_readings(a.session, cfg)
    tests = [r for t in a.test for r in load_readings(t, cfg)]

    def feats(r):
        v = np.where(np.isnan(r["rssi"]), FLOOR, r["rssi"])
        w = 10 ** (v / 10)
        return v, v - v.max(), float((w * AX).sum() / w.sum())

    for r in cal + tests:
        r["v"], r["rel"], r["xw"] = feats(r)
        r["strongest"] = int(PZ[np.argmax(r["v"])]) if np.isfinite(r["rssi"]).any() else 0
        r["xw_zone"] = int(np.clip(np.searchsorted(edges[1:-1], r["xw"], side="right"), 0, 3)) + 1

    rows = []
    for pw in sorted({r["power"] for r in cal}):
        C = [r for r in cal if r["power"] == pw]
        for r in C + [t for t in tests if t["power"] == pw]:
            lib = [c for c in C if c is not r]             # leave-one-out / lookup table
            if not lib:
                continue
            dz = lambda idx: int(pd.Series([lib[i]["zone"] for i in idx]).mode().iloc[0])
            dx = np.abs(np.array([c["xw"] for c in lib]) - r["xw"])
            dr = np.array([np.linalg.norm(c["rel"] - r["rel"]) for c in lib])
            dp = np.array([1 - mean_match(r["sig"], c["sig"], ports) for c in lib])
            k = min(a.k, len(lib))
            comb = dr / max(np.median(dr), 1e-9) + dp / max(np.median(dp), 1e-9)
            rows.append(dict(set="cal (leave-one-out)" if r in C else f"test {r['session']}", power=pw,
                             box=r["box"], zone=r["zone"], x=r["x"], y=r["y"], xw=round(r["xw"], 1),
                             **{f"rssi{p}": r["rssi"][i] for i, p in enumerate(ports)},
                             **{"strongest antenna": r["strongest"], "RSSI-weighted x": r["xw_zone"],
                                "RSSI-weighted x, cal": dz(np.argsort(dx)[:k]),
                                "RSSI lookup (kNN)": dz(np.argsort(dr)[:k]),
                                "phase pattern (kNN)": dz(np.argsort(dp)[:1]),
                                "phase + RSSI (kNN)": dz(np.argsort(comb)[:k])},
                             phase_best_match=round(1 - dp.min(), 3),
                             phase_best_same_zone=round(1 - min(dp[i] for i in range(len(lib)) if lib[i]["zone"] == r["zone"]), 3)
                             if any(c["zone"] == r["zone"] for c in lib) else np.nan))
    R = pd.DataFrame(rows)
    methods = ["strongest antenna", "RSSI-weighted x", "RSSI-weighted x, cal", "RSSI lookup (kNN)",
               "phase pattern (kNN)", "phase + RSSI (kNN)"]
    od = os.path.join(a.session, "analysis")
    os.makedirs(od, exist_ok=True)
    R.to_csv(os.path.join(od, "calibration.csv"), index=False)
    pd.set_option("display.width", 250)

    # ── reads / RSSI maps ────────────────────────────────────────────────────
    for pw in sorted(R.power.unique()):
        d = R[(R.power == pw) & (R.set.str.startswith("cal"))]
        print(f"\n════ {pw:g} dBm — calibration grid: RSSI per antenna (dBm), rows = depth (back on top), columns = x (in)")
        for i, p in enumerate(ports):
            t = d.pivot_table(index="y", columns="x", values=f"rssi{p}").sort_index(ascending=False)
            print(f"  antenna {p} (zone {PZ[i]}, x={AX[i]:g}\")")
            print("    " + t.to_string(float_format=lambda v: f"{v:5.0f}", na_rep="    -").replace("\n", "\n    "))
        t = d.pivot_table(index="y", columns="x", values="xw").sort_index(ascending=False)
        print("  RSSI-weighted x (in) — should rise left to right if RSSI carries position:")
        print("    " + t.to_string(float_format=lambda v: f"{v:5.1f}").replace("\n", "\n    "))

    # ── accuracy ─────────────────────────────────────────────────────────────
    print("\n════ Zone call accuracy")
    for (st, pw), d in R.groupby(["set", "power"]):
        acc = {m: (d[m] == d.zone).mean() for m in methods}
        near = {m: (abs(d[m] - d.zone) <= 1).mean() for m in methods}
        print(f"  {st:<22s} {pw:4.0f} dBm  n={len(d):2d}  " +
              "  ".join(f"{m}: {100 * acc[m]:3.0f}%" for m in methods))
    print("\n  By depth row (calibration, leave-one-out, best power per method):")
    for m in methods:
        d = R[R.set.str.startswith("cal")]
        best = d.groupby("power").apply(lambda g: (g[m] == g.zone).mean()).idxmax()
        g = d[d.power == best]
        print(f"    {m:<22s} @{best:g} dBm: " + "  ".join(
            f"y={y:g}\": {(h[m] == h.zone).sum()}/{len(h)}" for y, h in g.groupby("y")) +
            "   |  x-column: " + " ".join(f"{(h[m] == h.zone).sum()}/{len(h)}" for _, h in g.groupby("x")))

    # ── plots ────────────────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pws = sorted(R.power.unique())
    fig, ax = plt.subplots(len(pws), len(ports) + 1, figsize=(4 * (len(ports) + 1), 2.6 * len(pws)), squeeze=False)
    for i, pw in enumerate(pws):
        d = R[(R.power == pw) & R.set.str.startswith("cal")]
        for j, p in enumerate(ports):
            t = d.pivot_table(index="y", columns="x", values=f"rssi{p}").sort_index(ascending=False)
            im = ax[i, j].imshow(t.values, cmap="viridis", vmin=-75, vmax=-30, aspect="auto",
                                 extent=[0, 48, 0, 24])
            for e in edges[1:-1]:
                ax[i, j].axvline(e, color="w", lw=1)
            ax[i, j].plot(AX[j], 0.5, "rv", ms=9)
            ax[i, j].set(title=f"{pw:g} dBm  ant{p} RSSI", xticks=edges, yticks=[0, 12, 24])
        ax[i, -1].scatter(d.x, d.xw, c=d.zone, cmap="tab10", vmin=1, vmax=10)
        ax[i, -1].plot([0, 48], [0, 48], "k:", lw=0.8)
        for e in edges[1:-1]:
            ax[i, -1].axvline(e, color="gray", lw=0.6); ax[i, -1].axhline(e, color="gray", lw=0.6)
        ax[i, -1].set(xlabel="true x (in)", ylabel="RSSI-weighted x (in)", title="RSSI-weighted position")
    fig.colorbar(im, ax=ax[:, :len(ports)].ravel().tolist(), shrink=0.6, label="RSSI (dBm)")
    fig.savefig(os.path.join(od, "calibration.png"), dpi=100, bbox_inches="tight")
    print(f"\n-> {os.path.join(od, 'calibration.csv')}\n-> {os.path.join(od, 'calibration.png')}")


if __name__ == "__main__":
    main()
