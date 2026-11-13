"""
Zone locator: classifies one tag against a lookup table of calibration captures
(sessions with a layout.json: boxes at known zone / x / y).

Per zone it reports three confidences (0-1, summing to 1 over the zones):
  phase   best per-channel phase-pattern match of the tag to any lookup box in that
          zone (mean over antennas, optionally weighted by the tag's RSSI on each),
          turned into a confidence with softmax(match / T_PHASE)
  rssi    smallest distance between the tag's relative RSSI vector (RSSI - its max)
          and a lookup box in that zone, softmax(-distance / T_RSSI)
  slope   phase slope across frequency on each antenna (median of channel-to-channel
          steps, deg/MHz): a longer path -> a steeper slope.  Each antenna's cable delay is
          removed by subtracting that antenna's median slope over the lookup boxes; the
          antenna with the shortest relative path is the likely zone,
          softmax(relative slope / T_SLOPE).  Changes smoothly with position, so it does
          not need the tag to sit on a calibrated spot (analysis/phase_slope.py).
  combined  log p = g (W_PHASE log p_phase + W_RSSI log p_rssi) + W_SLOPE log p_slope, renormalised;
          g (0-1) = how closely the tag matches any lookup box (PHASE_GATE): both lookup
          scores only know calibrated spots, so they only count when the tag is on one
The confidences are a ranking with a sensible spread, not calibrated probabilities:
log the true zone of each test (the UI does) to see how often "80%" is right.
"""
import glob, json, os, sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "analysis"))
import shelfloc as sl  # noqa: E402
import calibration as C  # noqa: E402

SESS = os.path.join(ROOT, "sessions")
T_PHASE = 0.04     # match units: 0.1 better match -> e^2.5 = 12x more likely
T_RSSI = 3.0       # dB of RSSI-vector distance
T_SLOPE = 1.0      # relative slope, in units of its spread over the lookup boxes
W_PHASE, W_RSSI, W_SLOPE = 1.0, 0.5, 1.0
PHASE_GATE = (0.80, 0.95)   # weight of the lookup scores (phase pattern and RSSI pattern): 0 if the
                            # best phase match to any lookup box is below 0.80 (tag is off the
                            # calibrated spots), full above 0.95.  Tuned on saved captures: on-spot
                            # 113/128 (was 118), off-spot 50/76 (was 42).  See README.
MIN_CH = 6

_cfg = sl.load_config(os.path.join(ROOT, "shelf_config_4zone.json"))
PORTS = _cfg["ports"]
NAMES = sl.zone_names(_cfg)
PZ = {p: int(NAMES[_cfg["port_zone"][p]]) for p in PORTS}
_lib_cache = {}


def run_info(sd):
    f = sorted(glob.glob(os.path.join(sd, "run_*.json")))
    if not f:
        return {}
    try:
        return json.load(open(f[-1]))
    except Exception:
        return {}


def lookup_sessions():
    """Sessions usable as a lookup table: a layout.json with box positions and captures."""
    out = []
    for sd in sorted(glob.glob(os.path.join(SESS, "*"))):
        lj, mf = os.path.join(sd, "layout.json"), os.path.join(sd, "manifest.csv")
        if not (os.path.exists(lj) and os.path.exists(mf)):
            continue
        try:
            lay = json.load(open(lj)).get("tags", {})
            man = pd.read_csv(mf)
        except Exception:
            continue
        if not lay or man.empty:
            continue
        ri = run_info(sd)
        out.append(dict(name=os.path.basename(sd), boxes=len(lay), rf_mode=ri.get("RfMode"), gen2_session=ri.get("Session"),
                        started=(ri.get("started") or "")[:16].replace("T", " "),
                        powers=sorted({float(p) for p in man.tx_power_dbm})))
    return out


def capture_sessions():
    """Any session with captures (for classifying a tag from an existing capture)."""
    out = []
    for mf in sorted(glob.glob(os.path.join(SESS, "**", "manifest.csv"), recursive=True)):
        sd = os.path.dirname(mf)
        rel = os.path.relpath(sd, SESS).replace("\\", "/")
        if rel.startswith("fake_"):
            continue
        try:
            man = pd.read_csv(mf)
        except Exception:
            continue
        if man.empty:
            continue
        ri = run_info(sd)
        out.append(dict(name=rel, rf_mode=ri.get("RfMode"), gen2_session=ri.get("Session"), search=ri.get("search"),
                        started=(ri.get("started") or "")[:16].replace("T", " "),
                        powers=sorted({float(p) for p in man.tx_power_dbm})))
    return out


def library(names, power):
    lib = []
    for n in names:
        if n not in _lib_cache:
            _lib_cache[n] = C.load_readings(os.path.join(SESS, n), _cfg)
        lib += [r for r in _lib_cache[n] if r["power"] == power and np.isfinite(r["rssi"]).any()]
    return lib


def tag_reading(cap, epc):
    epc = epc.replace(" ", "").upper()
    g = cap[cap.epc == epc]
    st = g.groupby("antenna").agg(rssi=("rssi_dbm", "median"), reads=("rssi_dbm", "size"),
                                  ch=("freq_mhz", "nunique"))
    return dict(epc=epc,
                rssi=np.array([float(st.rssi[p]) if p in st.index else np.nan for p in PORTS]),
                reads=np.array([int(st.reads[p]) if p in st.index else 0 for p in PORTS]),
                channels=[int(st.ch[p]) if p in st.index else 0 for p in PORTS],
                sig=sl.capture_signatures(cap, epc, PORTS))


def sig_slope(s, min_ch=MIN_CH):
    """Phase slope (deg/MHz) from a per-channel doubled-phase signature: median of the
    steps between neighbouring channels, so a few multipath kinks don't bend it."""
    raw = s.get("raw") if isinstance(s, dict) else s
    if raw is None or len(raw) < min_ch:
        return np.nan
    raw = raw.sort_index()
    f = raw.index.values.astype(float)
    ph = np.degrees(np.unwrap(np.angle(raw.values))) / 2
    df, dp = np.diff(f), np.diff(ph)
    m = (df > 0) & (df <= 1.01)
    return float(np.median(dp[m] / df[m])) if m.sum() > 5 else np.nan


def _slopes(r):
    if "_slope" not in r:
        r["_slope"] = np.array([sig_slope(r["sig"].get(p)) for p in PORTS])
    return r["_slope"]


def slope_reference(lib):
    """Per-antenna median slope over the lookup boxes (removes cable/reader delay) and
    the spread of the relative slopes."""
    S = np.array([_slopes(c) for c in lib])
    med = np.nanmedian(S, axis=0)
    sd = float(np.nanstd(S - med)) or 1.0
    return med, sd


_EDGES = np.asarray(_cfg["zone_edges"]) / sl.IN                  # zone edges, inches
ANT_X = {p: float((_EDGES[NAMES.index(str(PZ[p]))] + _EDGES[NAMES.index(str(PZ[p])) + 1]) / 2) for p in PORTS}
X_GRID = np.arange(_EDGES[0], _EDGES[-1] + 0.01, 0.5)
SLOPE_NU = 4.0     # Student-t tails: one antenna with a bent slope can't veto a position
SLOPE_SCALE = 1.0  # widens the fitted scatter (1 = as fitted)


def slope_model(lib, min_boxes=8):
    """Per-antenna fit over the lookup boxes (known x):  slope = c + k * |x - antenna x|.
    c absorbs the antenna's cable/reader delay, k is how fast its slope steepens per inch,
    sd the scatter around the fit.  None if there are too few boxes or no usable trend."""
    m = {}
    for i, p in enumerate(PORTS):
        d, s = [], []
        for c in lib:
            v = _slopes(c)[i]
            if np.isfinite(v) and c.get("x") is not None:
                d.append(abs(c["x"] - ANT_X[p])); s.append(v)
        if len(d) < min_boxes:
            return None
        k, c0 = np.polyfit(d, s, 1)
        if k > -0.05:                                    # no distance trend on this antenna
            return None
        sd = float(np.std(np.array(s) - (c0 + k * np.array(d)), ddof=2))
        m[p] = dict(c=float(c0), k=float(k), sd=sd, sd_in=sd / abs(k))
    return m


def slope_position(t, model):
    """Likelihood of the tag's x along the shelf from its slopes on every antenna, and the
    per-zone probability (likelihood summed over each zone's x range).  Also the distance
    each antenna's slope implies on its own: (slope - c) / k."""
    sl_t = _slopes(t)
    ll = np.zeros_like(X_GRID)
    used = 0
    for i, p in enumerate(PORTS):
        if not np.isfinite(sl_t[i]):
            continue
        f = model[p]
        r = (sl_t[i] - (f["c"] + f["k"] * np.abs(X_GRID - ANT_X[p]))) / (f["sd"] * SLOPE_SCALE)
        ll += -(SLOPE_NU + 1) / 2 * np.log1p(r ** 2 / SLOPE_NU)
        used += 1
    dist = [None if not np.isfinite(sl_t[i]) else round((sl_t[i] - model[p]["c"]) / model[p]["k"], 1)
            for i, p in enumerate(PORTS)]
    if used < 2:
        return None
    w = np.exp(ll - ll.max())
    zones = sorted(PZ.values())
    pz = np.array([w[(X_GRID >= _EDGES[k]) & (X_GRID < _EDGES[k + 1] + (k == len(zones) - 1))].sum()
                   for k in range(len(zones))])
    return dict(p=pz / pz.sum(), x=float(X_GRID[np.argmax(w)]), dist=dist)


def _rel(r):
    v = np.where(np.isnan(r["rssi"]), C.FLOOR, r["rssi"])
    return v - v.max()


def _softmax(x):
    x = np.asarray(x, float)
    x = x - np.nanmax(x)
    e = np.where(np.isfinite(x), np.exp(x), 0.0)
    s = e.sum()
    return e / s if s > 0 else np.full(len(x), 1 / len(x))


def classify(t, lib, weighting="equal"):
    zones = sorted(PZ.values())
    if not np.isfinite(t["rssi"]).any():
        return dict(error="The tag was not read by any antenna.")
    v = np.where(np.isnan(t["rssi"]), C.FLOOR, t["rssi"])
    if weighting == "rssi":                          # stronger antennas count more
        w = {p: float(10 ** ((v[i] - v.max()) / 20)) for i, p in enumerate(PORTS)}
    else:
        w = {p: 1.0 for p in PORTS}
    rel = v - v.max()
    near = []
    for c in lib:
        ms, ws = [], []
        for p in PORTS:
            m = sl.sig_match(t["sig"].get(p), c["sig"].get(p), min_ch=MIN_CH)
            if np.isfinite(m):
                ms.append(m); ws.append(w[p])
        pm = float(np.average(ms, weights=ws)) if ms else np.nan
        near.append(dict(box=c["box"], zone=c["zone"], x=c["x"], y=c["y"], session=c["session"],
                         phase=pm, rssi_dist=float(np.linalg.norm(_rel(c) - rel))))
    N = pd.DataFrame(near)
    ph_best = [N[N.zone == z].phase.max() if (N.zone == z).any() else np.nan for z in zones]
    rs_best = [N[N.zone == z].rssi_dist.min() if (N.zone == z).any() else np.nan for z in zones]
    p_ph = _softmax(np.array(ph_best) / T_PHASE) if np.isfinite(ph_best).any() else np.full(len(zones), np.nan)
    p_rs = _softmax(-np.array(rs_best) / T_RSSI)
    # phase slope: shortest relative path -> that antenna's zone
    med, sd = slope_reference(lib)
    t_sl = _slopes(t)
    rel_sl = (t_sl - med) / sd
    z_sl = np.array([rel_sl[PORTS.index(next(p for p in PORTS if PZ[p] == z))] for z in zones])
    if np.isfinite(z_sl).sum() >= 2:
        z_sl = np.where(np.isfinite(z_sl), z_sl, np.nanmin(z_sl) - 1)   # unusable antenna = farther
        p_sl = _softmax(z_sl / T_SLOPE)
    else:
        p_sl = np.full(len(zones), np.nan)
    best = float(np.nanmax(ph_best)) if np.isfinite(ph_best).any() else np.nan
    have_slope = bool(np.isfinite(p_sl).all())
    if not have_slope:
        gate = 1.0                                   # nothing else to lean on
    elif np.isfinite(best):
        gate = float(np.clip((best - PHASE_GATE[0]) / (PHASE_GATE[1] - PHASE_GATE[0]), 0, 1))
    else:
        gate = 0.0
    lc = gate * W_RSSI * np.log(p_rs + 1e-9)
    if np.isfinite(p_ph).all():
        lc = lc + gate * W_PHASE * np.log(p_ph + 1e-9)
    if have_slope:
        lc = lc + W_SLOPE * np.log(p_sl + 1e-9)
    p_cb = _softmax(lc)
    strongest = PZ[PORTS[int(np.nanargmax(t["rssi"]))]]
    warn = []
    weak = [f"A{p}" for p, ch in zip(PORTS, t["channels"]) if 0 < ch < MIN_CH]
    if weak:
        warn.append(f"Few channels on {', '.join(weak)} (under {MIN_CH}): their phase is not used.")
    if not np.isfinite(ph_best).any():
        warn.append("No usable phase pattern: the zone call uses RSSI only.")
    if np.isfinite(best) and gate < 1:
        warn.append(f"Best phase-pattern match is {best:.2f}: the tag is not on a calibrated spot, so the "
                    f"lookup (phase and RSSI pattern) counts {gate:.0%} and the call leans on the phase slope.")
    if not np.isfinite(p_sl).all():
        warn.append("Phase slope needs 2+ antennas with enough channels: not used.")
    top = N.sort_values("phase", ascending=False).head(6)
    zr = lambda a: [None if not np.isfinite(x) else round(float(x), 4) for x in a]
    return dict(
        zones=zones, combined=zr(p_cb), phase=zr(p_ph), rssi=zr(p_rs), slope=zr(p_sl),
        phase_best_match=zr(ph_best), rssi_best_dist=zr(rs_best),
        call=int(zones[int(np.nanargmax(p_cb))]), call_conf=round(float(np.nanmax(p_cb)), 4),
        phase_weight=round(gate, 2), best_match=None if not np.isfinite(best) else round(best, 3),
        strongest=strongest,
        antennas=[dict(port=p, zone=PZ[p], rssi=None if np.isnan(t["rssi"][i]) else float(t["rssi"][i]),
                       reads=int(t["reads"][i]), channels=t["channels"][i],
                       slope=None if not np.isfinite(t_sl[i]) else round(float(t_sl[i]), 1),
                       rel_slope=None if not np.isfinite(rel_sl[i]) else round(float(rel_sl[i]), 2))
                  for i, p in enumerate(PORTS)],
        nearest=[dict(box=r.box, zone=int(r.zone), x=r.x, y=r.y, session=r.session,
                      phase=None if not np.isfinite(r.phase) else round(r.phase, 3), rssi_dist=round(r.rssi_dist, 1))
                 for r in top.itertuples()],
        lookup_boxes=int(len(lib)), warnings=warn)


def classify_from_session(session, epc, power, lookup, weighting):
    sd = os.path.join(SESS, session)
    _, man = sl.load_session(sd)
    m = man[man.tx_power_dbm.astype(float) == float(power)]
    if m.empty:
        return dict(error=f"{session} has no capture at {power:g} dBm.")
    cap = sl.load_capture(os.path.join(sd, m.file.iloc[-1]))
    lib = [c for c in library(lookup, float(power)) if not (c["session"] == os.path.basename(sd))]
    if not lib:
        return dict(error=f"The lookup sessions have no capture at {power:g} dBm.")
    res = classify(tag_reading(cap, epc), lib, weighting)
    ri = run_info(sd)
    rd = [f"mode {ri['RfMode']}" if ri.get("RfMode") is not None else "",
          f"S{ri['Session']}" if ri.get("Session") is not None else "", ri.get("search") or ""]
    rd = ", ".join(x for x in rd if x)
    res["source"] = f"{session} ({power:g} dBm{', ' + rd if rd else ''})"
    res["reader"] = dict(rf_mode=ri.get("RfMode"), gen2_session=ri.get("Session"), search=ri.get("search"))
    # truth, if the session's layout knows this tag
    lj = os.path.join(sd, "layout.json")
    if os.path.exists(lj):
        v = json.load(open(lj)).get("tags", {}).get(epc.replace(" ", "").upper())
        if v:
            res["truth_zone"] = int(v[0])
    return res


def evaluate(session, power, lookup, weighting):
    """Classify every box of a session with a layout.json; lookup = other sessions."""
    sd = os.path.join(SESS, session)
    lay = json.load(open(os.path.join(sd, "layout.json")))["tags"]
    rows = []
    for e, v in lay.items():
        r = classify_from_session(session, e, power, lookup, weighting)
        if "error" in r:
            rows.append(dict(epc=e, box=e[-3:], zone=int(v[0]), error=r["error"]))
            continue
        rows.append(dict(epc=e, box=e[-3:], zone=int(v[0]), y=v[1], x=v[2] if len(v) > 2 else None,
                         call=r["call"], conf=r["call_conf"], strongest=r["strongest"]))
    D = pd.DataFrame(rows)
    ok = D[D.get("call").notna()] if "call" in D else D.iloc[0:0]
    return dict(session=session, power=power, n=len(ok),
                correct=int((ok.call == ok.zone).sum()), strongest_correct=int((ok.strongest == ok.zone).sum()),
                rows=json.loads(D.to_json(orient="records")))
