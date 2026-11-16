"""Collects the per-box results of every test into report/report_data.json
(used by report/zone_test_report.html).  Run from the project folder:
    .venv\\Scripts\\python report\\build_data.py
"""
import glob, json, os, sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "analysis"))
import shelfloc as sl  # noqa: E402

S = lambda *p: os.path.join(ROOT, "sessions", *p)
out = {}

# ── A. 2-antenna quadrant survey (side walls, 17 dBm, 48 spots) ─────────────
pr = pd.read_csv(S("2ant_grid_17dbm", "analysis", "predictions.csv"))
ft = pd.read_csv(S("2ant_grid_17dbm", "analysis", "features.csv"), dtype={"epc": str})
ft = ft[ft.epc.str.upper().str.endswith("0261")]
qn = {"BR": 1, "BL": 2, "FL": 3, "FR": 4}       # the user's quadrant numbering
spots = []
for sid, g in pr.groupby("id"):
    r0 = g.iloc[0]
    f = ft[ft.id == sid].set_index("port").rssi
    spots.append(dict(x=r0.x_true, y=r0.y_true, true=qn[r0.true_q],
                      calls={m: qn[h.pred_q.iloc[0]] for m, h in g.groupby("method")},
                      rssi=[None if p not in f else float(f[p]) for p in (1, 2)]))
acc = pr.assign(ok=pr.true_q == pr.pred_q).groupby("method").ok.mean().round(3).to_dict()
out["two_ant"] = dict(spots=spots, acc=acc, L=32, D=24,
                      antennas=[dict(port=1, x=0, y=12), dict(port=2, x=32, y=12)])

# ── B. move test: phase pattern vs distance moved ────────────────────────────
mv = pd.read_csv(S("2ant_move_15dbm", "analysis", "move_test.csv"))
out["move"] = [dict(id=r.id, d=float(r.dist_in), dir=r.direction,
                    p1=None if pd.isna(r.P1_match) else round(float(r.P1_match), 3),
                    p2=None if pd.isna(r.P2_match) else round(float(r.P2_match), 3)) for r in mv.itertuples()]

# ── C. reference tag (never moved) while the test box moved around it ───────
st = pd.read_csv(S("2ant_grid_17dbm", "analysis", "stability.csv"))
out["ref_stability"] = [dict(d=round(float(r.dist_in), 1),
                             p1=None if pd.isna(r.P1_match) else round(float(r.P1_match), 3),
                             p2=None if pd.isna(r.P2_match) else round(float(r.P2_match), 3),
                             dr1=None if pd.isna(r.P1_dRSSI) else float(r.P1_dRSSI),
                             dr2=None if pd.isna(r.P2_dRSSI) else float(r.P2_dRSSI)) for r in st.itertuples()]

# ── D/E. 4-zone tests ────────────────────────────────────────────────────────
cfg = sl.load_config(S("4zone_centers", "shelf_config.json"))
ports, names = cfg["ports"], sl.zone_names(cfg)
PZ = {p: int(names[cfg["port_zone"][p]]) for p in ports}
TAGS = ["1A1122334455667788990261", "1A1122334455667788990650",
        "1A1122334455667788990654", "1A1122334455667788990661"]
ROT = ["4zone_centers", "4zone_rot1", "4zone_rot2", "4zone_rot3", "4zone_rot4"]


def load(sess):
    _, man = sl.load_session(S(sess))
    return [(float(r.tx_power_dbm), sl.load_capture(S(sess, r.file))) for r in man.itertuples()
            if os.path.exists(S(sess, r.file))]


def per_box(cap, epc):
    g = cap[cap.epc == epc]
    st = g.groupby("antenna").agg(rssi=("rssi_dbm", "median"), reads=("rssi_dbm", "size"))
    ps = sl.capture_features(cap, cfg, {epc})
    ps = ps[ps.epc == epc].set_index("port") if len(ps) else ps
    rssi = [None if p not in st.index else float(st.rssi[p]) for p in ports]
    reads = [0 if p not in st.index else int(st.reads[p]) for p in ports]
    noise = [None if p not in getattr(ps, "index", []) else round(float(ps.phase_sd_mod180[p]), 2) for p in ports]
    have = [(v, p) for v, p in zip(rssi, ports) if v is not None]
    call = PZ[max(have)[1]] if have else None
    srt = sorted([v for v, _ in have], reverse=True)
    return dict(rssi=rssi, reads=reads, noise=noise, rssi_call=call,
                margin=round(srt[0] - srt[1], 1) if len(srt) > 1 else None)


readings = []   # (session, power, epc, zone, y, sig, record)
for n, sess in enumerate(ROT):
    for pw, cap in load(sess):
        for i, e in enumerate(TAGS):
            z = (i + n) % 4 + 1
            rec = dict(session=sess, power=pw, box=e[-3:], zone=z, y=12, **per_box(cap, e))
            readings.append((sess, pw, e, z, sl.capture_signatures(cap, e, ports), rec))
lay = json.load(open(S("4zone_12box", "layout.json")))
for pw, cap in load("4zone_12box"):
    for e, (z, y) in lay["tags"].items():
        rec = dict(session="4zone_12box", power=pw, box=e[-3:], zone=int(z), y=y, **per_box(cap, e))
        readings.append(("4zone_12box", pw, e, int(z), sl.capture_signatures(cap, e, ports), rec))


def mean_match(a, b):
    v = [sl.sig_match(a.get(p), b.get(p)) for p in ports]
    return float(np.nanmean(v)) if np.isfinite(v).any() else np.nan


# phase call: nearest reading (mean match over the 4 antennas) among the CENTERED
# rotation readings of OTHER sessions at the same power
for s1, pw, e, z, sig, rec in readings:
    sc = {}
    for s2, pw2, e2, z2, sig2, _ in readings:
        if s2 == s1 or pw2 != pw or s2 == "4zone_12box":
            continue
        m = mean_match(sig, sig2)
        if np.isfinite(m):
            sc[z2] = max(sc.get(z2, -1), m)
    if sc:
        best = max(sc, key=sc.get)
        rec.update(phase_call=best, phase_own=round(sc.get(z, np.nan), 3),
                   phase_other=round(max(v for k, v in sc.items() if k != z), 3))
    else:
        rec.update(phase_call=None, phase_own=None, phase_other=None)
    own_p = [p for p in ports if PZ[p] == z][0]
    rec["own_port"] = own_p

out["rotation"] = [r for s, _, _, _, _, r in readings if s != "4zone_12box"]
out["twelve"] = [r for s, _, _, _, _, r in readings if s == "4zone_12box"]

# pair kinds (rotation sessions, raw pattern), from zone_centers.py output
P = pd.read_csv(S("4zone_centers", "analysis", "zone_centers_pairs.csv"))
out["pair_kinds"] = {k: {int(p): round(float(h.match.median()), 3) for p, h in g.groupby("port")}
                     for k, g in P.groupby("kind")}

# 12-box similarity matrix (15 dBm, same capture)
tw = [(e, z, rec["y"], sig) for s, pw, e, z, sig, rec in readings if s == "4zone_12box" and pw == 15]
tw.sort(key=lambda t: (t[1], t[2]))
M = [[round(mean_match(a[3], b[3]), 3) if i != j else 1.0 for j, b in enumerate(tw)] for i, a in enumerate(tw)]
out["twelve_matrix"] = dict(boxes=[dict(box=e[-3:], zone=z, y=y) for e, z, y, _ in tw], m=M)

out["port_zone"] = PZ

# ── F. 32-box calibration grid (analysis/calibration.py output) ─────────────
import itertools  # noqa: E402
import calibration as CB  # noqa: E402
cc = pd.read_csv(S("4zone_cal32", "analysis", "calibration.csv"))
CM = ["strongest antenna", "RSSI-weighted x", "RSSI lookup (kNN)", "phase pattern (kNN)", "phase + RSSI (kNN)"]
nan2none = lambda v: None if pd.isna(v) else float(v)
out["cal32"] = [dict(set=r["set"], power=r.power, box=str(r.box), zone=int(r.zone), x=r.x, y=r.y, xw=r.xw,
                     rssi=[nan2none(r[f"rssi{p}"]) for p in ports], calls={m: int(r[m]) for m in CM},
                     best=nan2none(r.phase_best_match), best_same=nan2none(r.phase_best_same_zone))
                for _, r in cc.iterrows()]
out["cal32_acc"] = [dict(set=s, power=pw, **{m: round(float((g[m] == g.zone).mean()), 3) for m in CM}, n=len(g))
                    for (s, pw), g in cc.groupby(["set", "power"])]
ccfg, _ = sl.load_session(S("4zone_cal32"))
dist = []
for pw in (12.0, 15.0, 20.0):
    calr = [r for r in CB.load_readings(S("4zone_cal32"), ccfg) if r["power"] == pw]
    for a_, b_ in itertools.combinations(calr, 2):
        dist.append(dict(power=pw, d=round(float(np.hypot(a_["x"] - b_["x"], a_["y"] - b_["y"])), 1),
                         same=a_["zone"] == b_["zone"], m=CB.mean_match(a_["sig"], b_["sig"], ports)))
DD = pd.DataFrame(dist)
DD["bin"] = DD.d.where(DD.d <= 12, 99)
out["cal32_dist"] = [dict(power=pw, d=float(b), same=bool(s), median=round(float(g.m.median()), 3), n=len(g))
                     for (pw, b, s), g in DD.groupby(["power", "bin", "same"])]
# ── G. Silion E710 vs R700 (analysis/reader_compare.py output) ─────────────
RC = S("reader_compare")
if os.path.exists(os.path.join(RC, "reader_compare.csv")):
    cmp_ = pd.read_csv(os.path.join(RC, "reader_compare.csv"), dtype=str)
    qual = pd.read_csv(os.path.join(RC, "quality.csv"))
    pb = pd.read_csv(os.path.join(RC, "per_box.csv"))
    RM = ["strongest", "fw vote", "WCL RSSI", "WCL slope+RSSI", "slope", "RSSI lookup", "phase lookup", "locator"]
    toi = lambda v: None if pd.isna(v) else int(v)
    bpath = os.path.join(RC, "reader_compare_border.csv")
    cmpb = pd.read_csv(bpath, dtype=str) if os.path.exists(bpath) else None
    accb = lambda r: None if cmpb is None else {m: [int(x) for x in cmpb[(cmpb.reader == r.reader) & (cmpb.test == r.test) & (cmpb.dBm == r.dBm)][m].iloc[0].split("/")] for m in RM}
    out["readers"] = dict(
        methods=RM,
        compare=[dict(reader=r.reader, test=r.test, dBm=r.dBm, boxes=int(r.boxes),
                      acc={m: [int(x) for x in r[m].split("/")] for m in RM}, acc_border=accb(r)) for _, r in cmp_.iterrows()],
        quality=json.loads(qual.to_json(orient="records")),
        boxes=[dict(reader=r.reader, test=r.test, power=float(r.power), box=str(r.box), zone=int(r.zone),
                    x=float(r.x), y=float(r.y), rssi=json.loads(r.rssi), calls={m: toi(r[m]) for m in RM},
                    best=nan2none(r.best_match)) for _, r in pb.iterrows()])
    out["readers"]["findings"] = [
        ["SAME", "ok", "<b>On untouched boxes both readers are near perfect, as long as the lookup and the test use the same RF mode.</b> "
         "Silion: locator 32/32, phase pattern 31/32. R700 with mode 2 for both: 32/32 and 31/32. The R700 with a mode 1002 lookup and a "
         "mode 2 test drops to 28/32 (phase 21/32), so the lookup has to be captured in the mode that will be used."],
        ["6\"", "no", "<b>Away from calibrated spots the Silion is no better.</b> Called from the other 31 boxes (6\" away): Silion locator "
         "18/32, R700 16–18/32. The phase slope does best on both, 20/32 on the Silion and 20–22/32 on the R700."],
        ["RSSI", "no", "<b>Methods without a lookup stay near 50% on both readers.</b> Strongest RSSI 13–15/32, firmware vote 14–15/32, "
         "WCL with RSSI weights 10–14/32, WCL with slope and RSSI weights 15–21/32."],
        ["BORDER", "mid", "<b>Most wrong calls are one zone over, across the line the box sits next to.</b> If a box beside a zone line may "
         "also be called the zone across that line (655 as Z1 or Z2; the end boxes such as 650 must be exact), the 6\" test rises to "
         "Silion locator 24/32 and R700 24–26/32, and phase slope to 23/32 and 27–28/32. Strongest RSSI and the firmware vote reach "
         "20–23/32. This is a lenient score: 24 of the 32 boxes sit next to a line."],
        ["PHASE", "ok", "<b>The Silion's phase is cleaner.</b> Repeated reads on a channel agree to 0.9–1.1°, against 1.6–2.7° on the R700, "
         "with no 180° flips. An untouched box's pattern repeats at 0.995 (R700, same mode: 0.989)."],
        ["POWER", "mid", "<b>12 dBm on antennas 1–2 and 14 dBm on 3–4 gave even coverage.</b> Reads per antenna 4,150–5,270 in 120 s, "
         "a median 43 channels per box per antenna, and 16 of 128 box–antenna pairs under 6 channels (R700: 6–31)."],
    ]

# ── H. Beam-steering array, one port at a time (analysis/beam_check.py output) ──
# Only runs with the elements at the correct 16 cm spacing (earlier runs had the wrong spacing).
BEAM_RUNS = [("beam_check_silion8_16cm", 1, 8, "Nov 13"), ("beam_check_silion10_16cm", 1, 10, "Nov 13"),
             ("beam_check_gap3_5dbm", 3, 5, "Nov 13"), ("beam_check_gap3_10dbm", 3, 10, "Nov 13"),
             ("beam_check_gap3_15dbm", 3, 15, "Nov 13")]
PORT_ZONE = {1: 1, 4: 2, 3: 3, 2: 4}  # zone each port favoured at 5 dBm (3" gaps), used as the reference beam map


def beam_run(sess, pz):
    """Per-box and per-port summary of one beam check (each port run alone). pz: port -> zone map for scoring."""
    p = S(sess, "beam_check.csv")
    if not os.path.exists(p):
        return None
    B = pd.read_csv(p, dtype={"box": str})
    boxes = []
    for (bx, z, x, y), g in B.groupby(["box", "zone", "x", "y"]):
        g = g.set_index("port")
        rs = {pt: (None if pd.isna(g.rssi.get(pt)) else float(g.rssi[pt])) for pt in (1, 2, 3, 4)}
        rd = {pt: int(g.reads.get(pt, 0)) for pt in (1, 2, 3, 4)}
        best = max((v, pt) for pt, v in rs.items() if v is not None)[1] if any(v is not None for v in rs.values()) else None
        boxes.append(dict(box=bx, zone=int(z), x=float(x), y=float(y), rssi=[rs[k] for k in (1, 2, 3, 4)],
                          reads=[rd[k] for k in (1, 2, 3, 4)], best=best))
    nport = [sum(1 for b in boxes if b["reads"][i] > 0) for i in range(4)]
    seen = [b for b in boxes if any(b["reads"])]
    share = []
    for pt in (1, 2, 3, 4):
        tot = B[B.port == pt].reads.sum()
        share.append([round(float(B[(B.port == pt) & (B.zone == z)].reads.sum() / max(tot, 1) * 100)) for z in (1, 2, 3, 4)])
    # mean RSSI over every read of the 32 box tags (all ports)
    tags = {str(bx).zfill(4) for bx in B.box}
    cap = pd.concat(pd.read_csv(f) for f in glob.glob(os.path.join(S(sess), "port*", "cap_*.csv")))
    cap = cap[cap.epc.astype(str).str[-4:].isin(tags)]
    return dict(id=sess, boxes=boxes, nport=nport, share=share,
                reads=[int(B[B.port == pt].reads.sum()) for pt in (1, 2, 3, 4)],
                seen=len(seen), multi=sum(1 for b in seen if sum(r > 0 for r in b["reads"]) >= 2),
                right=sum(1 for b in seen if b["best"] is not None and pz[b["best"]] == b["zone"]),
                rssi_mean=round(float(cap.rssi_dbm.mean()), 1), pz=pz)


beam = []
for sess, gap, pw, day in BEAM_RUNS:
    r = beam_run(sess, PORT_ZONE)
    if r:
        beam.append(dict(r, gap=gap, power=pw, day=day))
if beam:
    out["beam"] = dict(runs=beam, port_zone=PORT_ZONE, findings=[
        ["5 dBm", "mid", "<b>At 5 dBm with 3\" gaps each port has its own zone.</b> Port 1 reads only zone 1 and port 2 only zone 4; "
         "port 3 favours zone 3 (47% of its reads) and port 4 zone 2 (49%). But only 16 of 32 boxes are read at all, and port 1 "
         "reads just 2 boxes, so this is the edge of reading, not a working setup."],
        ["10 dBm", "no", "<b>At 8–10 dBm the beams overlap.</b> Port 1 covers the left half, port 3 the right three zones, port 2 both "
         "ends and port 4 everything. 63–88% of the boxes read are seen on 2 or more ports. The 3\" gaps changed nothing measurable "
         "at 10 dBm."],
        ["15 dBm", "no", "<b>At 15 dBm every port reads 28–30 boxes across all four zones</b> (30 of 31 boxes on 2+ ports): the "
         "beams no longer separate the zones."],
        ["CALL", "no", "<b>The strongest port names the right zone for at most about half the boxes.</b> Taking the 5 dBm beam map "
         "(P1→Z1, P4→Z2, P3→Z3, P2→Z4) for every run: 8 and 7 of 32 with 1\" gaps at 8 and 10 dBm; 12, 11 and 17 of 32 with 3\" gaps "
         "at 5, 10 and 15 dBm (at 5 dBm, 12 of the 16 boxes read). One antenna per zone gave 13–15 of 32."],
        ["ORDER", "mid", "<b>The order along the shelf is P1, P4, P3, P2.</b> The design angles (P3 41°, P1 76°, P4 105°, P2 139°) "
         "give P3, P1, P4, P2, so ports 1 and 3 look swapped, at the reader or at the Butler matrix."],
    ])

# ── I. Zircon (QM100, one reader port + RF switch): same beam check, array and metal changed between setups ──
# v1: array on foam (16 cm) with a metal plate held perpendicular to the shelf behind it.
# v2: array remounted on the plate, elements 18 cm apart.  v3: v2's antenna, boxes stacked instead of one layer.
# The port order moved with the remount: no metal and v1 favoured A2/A3/A4/A1 for zones 1-4; v2 and v3 P1/P4/P3/P2
# (the Silion order). Each setup is scored with its own map.
ZX_OLD = {2: 1, 3: 2, 4: 3, 1: 4}
ZX_SETUPS = [("zircon_beam", "No back metal", "Nov 13", ZX_OLD, 16),
             ("zircon_beam_metal", "Back metal v1", "Nov 13", ZX_OLD, 16),
             ("zircon_beam_metal_new", "Back metal v2", "Nov 15", PORT_ZONE, 18),
             ("zircon_beam_metal_new2", "Back metal v3", "Nov 15", PORT_ZONE, 18)]
zx = []
for base, setup, day, pz, cm in ZX_SETUPS:
    for pw in (5, 10, 15):
        r = beam_run(f"{base}_{pw}dbm", pz)
        if r:
            zx.append(dict(r, setup=setup, power=pw, day=day, spacing=cm, stacked=base.endswith("new2")))
if zx:
    out["zircon"] = dict(runs=zx, setups=[s[1] for s in ZX_SETUPS], findings=[
        ["v2", "ok", "<b>Back metal v2 (array mounted on the plate, elements 18 cm apart) is the best Zircon setup.</b> The strongest port names the right zone for 23, 22 and 24 of "
         "32 boxes at 5, 10 and 15 dBm, every box is read, and every zone gets at least 4 right. At 5 dBm only 56% of boxes are "
         "seen on 2+ ports and 73% of port 2's reads land in zone 4."],
        ["ORDER", "mid", "<b>Mounting the array on the plate at 18 cm changed the port order to P1, P4, P3, P2</b>, the same as "
         "the Silion. Without metal and with v1 (16 cm, plate only behind it) it was A2, A3, A4, A1."],
        ["v3", "mid", "<b>Stacking the boxes (v3, same antenna as v2) lost zone 4.</b> Ports 3 and 4 stay tight (76% of port 3's "
         "reads at 5 dBm in zone 3), but only 2–3 of 8 zone-4 boxes are called right: 19, 20 and 20 of 32, with 4 boxes unread at "
         "5 dBm. Box positions are drawn on the one-layer grid."],
        ["v1", "mid", "<b>A plate held perpendicular to the shelf behind the array (v1) helped zones 1–3 but not zone 4</b>: 19, 19 "
         "and 20 of 32, with zone 4 at 2 of 8 because port 4 spills into it and beats port 1."],
        ["NONE", "no", "<b>Without back metal the beams barely separate</b>: 81–96% of boxes are seen on 2+ ports and the "
         "strongest port is right for 13, 19 and 16 of 32."],
        ["RSSI", "mid", "<b>The Zircon reports about 5 dB lower RSSI than the Silion</b> (−48.5 to −52.7 dBm mean against −42.7 "
         "to −46.0). Its RSSI is raw and goes through the RF switch, so compare RSSI within one reader only."],
    ])

json.dump(out, open(os.path.join(ROOT, "report", "report_data.json"), "w"), indent=0)
tpl = open(os.path.join(ROOT, "report", "template.html"), encoding="utf-8").read()
blob = json.dumps(out, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
open(os.path.join(ROOT, "report", "zone_test_report.html"), "w", encoding="utf-8").write(tpl.replace("__DATA__", blob))

# console summary
R = pd.DataFrame(out["rotation"])
T = pd.DataFrame(out["twelve"])
for name, D in (("rotation", R[R.power != 20]), ("12-box", T)):
    print(f"{name}: RSSI {(D.rssi_call == D.zone).sum()}/{len(D)}  phase {(D.phase_call == D.zone).sum()}/{len(D)}")
print("2-ant acc", acc)
