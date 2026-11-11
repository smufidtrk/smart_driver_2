"""
Fingerprinting variant: learn the shelf's (fixed) metal multipath once, then
classify quadrants with kNN on per-antenna features.

Features per antenna (relative, so the unknown per-tag gain / tag-slope cancel):
  rssi_rel  = rssi - max(rssi over antennas)
  range_rel = slope_range - median(slope_range over antennas)   [mod-180-safe]
Unseen antenna -> rssi_rel = -40 dB, range_rel = +0.6 m.

Two scenarios:
  same-shelf : train + test on the same shelf instance (load / other boxes /
               tag orientation / tag gain all re-randomised per tag)
  transfer   : train on shelf A, test on shelf B (same design, different install:
               new cable-cal error + reflection-phase jitter)  -> fleet roll-out case
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import quadrant_sim as qs  # noqa: E402

IN = qs.IN
N_TRAIN, N_TEST, K = 1500, 500, 15
LAYS = ["3 ant - triangle", "4 ant - 1 per quadrant", "6 ant - 3x2 grid"]


def feats(phi, rssi, seen):
    r = np.where(seen, rssi - rssi[seen].max(), -40.0)
    d, _ = qs.slope_range(phi)
    rg = np.where(seen, d - np.median(d[seen]), 0.6)
    return r, rg


def gen(rng, shelf, ants, n):
    R, G, Q, XY, BD = [], [], [], [], []
    for _ in range(n):
        pos, axis = qs.random_tag(rng)
        phi, rssi, seen = qs.measure(rng, shelf, ants, pos, axis)
        if not seen.any():
            continue
        r, g = feats(phi, rssi, seen)
        R.append(r); G.append(g); Q.append(qs.quadrant(pos)); XY.append(pos[:2] / IN)
        BD.append(qs.boundary_dist_in(pos))
    return np.array(R), np.array(G), np.array(Q), np.array(XY), np.array(BD)


def knn(trX, trQ, trXY, teX):
    mu, sd = trX.mean(0), trX.std(0) + 1e-9
    A, B = (trX - mu) / sd, (teX - mu) / sd
    D = ((B[:, None, :] - A[None, :, :]) ** 2).sum(-1)
    nn = np.argsort(D, 1)[:, :K]
    w = 1 / (np.sqrt(np.take_along_axis(D, nn, 1)) + 1e-3)
    P = np.zeros((len(teX), 4))
    for q in range(4):
        P[:, q] = (w * (trQ[nn] == q)).sum(1)
    P /= P.sum(1, keepdims=True)
    xy = (w[..., None] * trXY[nn]).sum(1) / w.sum(1, keepdims=True)
    return P, xy


rows = []
for lay in LAYS:
    ants = np.array([[x * IN, y * IN, qs.ANT_Z] for x, y in qs.LAYOUTS[lay]])
    for sev in qs.SEVERITY:
        for scen in ["same-shelf", "transfer"]:
            for rep in range(3):
                rng = np.random.default_rng(1000 + rep)
                shA = qs.Shelf(rng, sev, len(ants))
                shB = shA if scen == "same-shelf" else qs.Shelf(rng, sev, len(ants))
                tr = gen(rng, shA, ants, N_TRAIN)
                te = gen(rng, shB, ants, N_TEST)
                for feat in ["rssi", "both"]:
                    trX = tr[0] if feat == "rssi" else np.hstack([tr[0], tr[1] * 10])
                    teX = te[0] if feat == "rssi" else np.hstack([te[0], te[1] * 10])
                    P, xy = knn(trX, tr[2], tr[3], teX)
                    qh = P.argmax(1); conf = P.max(1); q2 = np.argsort(P, 1)[:, -2]
                    ok = qh == te[2]
                    err = np.linalg.norm(xy - te[3], axis=1)
                    rows.append(dict(layout=lay, sev=sev, scenario=scen, feat=feat, rep=rep,
                                     acc=ok.mean(), acc_3in_from_edge=ok[te[4] >= 3].mean(),
                                     acc_when_conf90=ok[conf >= 0.9].mean(),
                                     coverage_conf90=(conf >= 0.9).mean(),
                                     top2_acc=(ok | (q2 == te[2])).mean(),
                                     p_err_lt_12in=(err < 12).mean(), p_err_lt_18in=(err < 18).mean()))
        print(f"  {lay:24s} {sev}", flush=True)

df = pd.DataFrame(rows)
S = df.groupby(["sev", "layout", "scenario", "feat"], sort=False).mean(numeric_only=True).drop(columns="rep")
S = (S * 100).round(1)
S.to_csv(os.path.join(qs.OUT, "fingerprint_summary.csv"))
pd.set_option("display.width", 200)
print(S.to_string())
