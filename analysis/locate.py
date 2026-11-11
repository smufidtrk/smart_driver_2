"""
Locate every tag in a scan capture (positions unknown) using survey sessions as
training data.

  python analysis/locate.py sessions/van1_scan --train sessions/van1_grid sessions/van1_random
  python analysis/locate.py path/to/cap_scan_xxx.csv --train sessions/van1_grid

Output per tag: zone (quadrant or numbered zone, from the shelf config),
confidence, runner-up zone, estimated x/y, and a 'call' = single zone if
confidence >= --conf (default 0.9), otherwise the top-2 pair (e.g. "FL/FR", "2/3").  Written to <scan>/analysis/locations.csv (or next to
the csv).
"""
import argparse, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import shelfloc as sl  # noqa: E402
from survey_analyze import prepare  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scan", help="scan session dir or a single cap_*.csv")
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--conf", type=float, default=0.9)
    ap.add_argument("--method", choices=["strongest", "rssi", "pattern", "fingerprint", "model"], default="rssi",
                    help="strongest = antenna with the highest RSSI wins (no training); "
                         "rssi = RSSI kNN (default); pattern = phase pattern + RSSI kNN; "
                         "fingerprint = RSSI + phase-slope kNN; model = physics model (quadrants only)")
    a = ap.parse_args()

    # training data (first session defines the fallback cable term)
    cfg, tman, train, cable, _ = prepare(a.train[0])
    tsigs = sl.session_signatures(a.train[0], tman, cfg)
    for t in a.train[1:]:
        tc, tm, tf, _, _ = prepare(t, cable)
        sfx = "@" + os.path.basename(os.path.normpath(t))
        train = pd.concat([train, tf.assign(id=tf.id.astype(str) + sfx)], ignore_index=True)
        tsigs.update(sl.session_signatures(t, tm, tc, sfx))
    ports = cfg["ports"]

    if os.path.isdir(a.scan):
        scfg, man = sl.load_session(a.scan)
        files = [os.path.join(a.scan, f) for f in man.file]
        outdir = os.path.join(a.scan, "analysis")
    else:
        scfg, files, outdir = cfg, [a.scan], os.path.dirname(os.path.abspath(a.scan))
    os.makedirs(outdir, exist_ok=True)

    fp = sl.Fingerprint(ports).fit(train, cfg)
    fr = sl.Fingerprint(ports, use_phase=False).fit(train, cfg)
    sa = sl.StrongestAntenna(cfg, ports)
    names = sl.zone_names(cfg)
    pf = sl.PatternFingerprint(ports).fit(train, tsigs, cfg)
    loc = sl.ModelLocalizer(cfg, sl.learn_model(train, cfg))
    # cable term for the scanned shelf: its own reference tag (median over the scan
    # captures); ports without reference reads fall back to the training cable.
    raw = {f: sl.load_capture(f) for f in files}
    caps = {f: sl.capture_features(c, scfg) for f, c in raw.items()}
    allf = pd.concat([c for c in caps.values() if not c.empty], ignore_index=True)
    scab, ssrc = sl.session_cable(allf, scfg, fallback=cable)
    print("cable: " + ", ".join(f"p{p}={ssrc.get(p, 'NONE')}" for p in ports))
    if not any(v.startswith("reference") for v in ssrc.values()):
        print("  ⚠ no reference-tag reads in the scan: using the TRAINING shelf's cable "
              "(only valid on the same van / same cables)")
    rows = []
    for f, feat in caps.items():
        if feat.empty:
            continue
        if scfg["ref_epc"]:
            feat = feat[feat.epc != scfg["ref_epc"]]
        feat = sl.apply_cable(feat, scab).assign(id=lambda d: d.epc)
        for epc, g in feat.groupby("epc"):
            if a.method in ("strongest", "rssi"):
                _, P, xy = (sa if a.method == "strongest" else fr).predict(g)
                qp, xy = P[0], xy[0]
            elif a.method == "pattern":
                _, P, xy = pf.predict(g, {epc: sl.capture_signatures(raw[f], epc, ports, scfg["ref_epc"])})
                qp, xy = P[0], xy[0]
            elif a.method == "fingerprint":
                _, P, xy = fp.predict(g)
                qp, xy = P[0], xy[0]
            else:
                qp, est = loc.locate(g)
                xy = est[:2] / sl.IN
            order = np.argsort(qp)[::-1]
            best, second = names[order[0]], names[order[1]]
            rows.append(dict(capture=os.path.basename(f), epc=epc, ports_seen=len(g),
                             zone=best, conf=round(float(qp[order[0]]), 3), second=second,
                             call=best if qp[order[0]] >= a.conf else f"{best}/{second}",
                             x_in=round(float(xy[0]), 1), y_in=round(float(xy[1]), 1),
                             note="read by 1 port only — low confidence" if len(g) < 2 else ""))
    out = pd.DataFrame(rows)
    path = os.path.join(outdir, "locations.csv")
    out.to_csv(path, index=False)
    pd.set_option("display.width", 200)
    print(out.to_string(index=False) if len(out) else "no tags found")
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
