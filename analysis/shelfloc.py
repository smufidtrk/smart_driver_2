"""
shelfloc — shared processing for Smart Driver shelf captures.

Pipeline per capture (cap_*.csv from capture/ShelfCapture):
  1. per (tag, port, channel): circular mean of the DOUBLED phase exp(j*2*phi)
     -> removes Impinj's random 180-deg flips (seen in the Walmart refs: per-read
        jitter 30-65 deg raw -> 2-6 deg mod 180).
  2. per (tag, port): phase-slope across channels -> raw round-trip range
     (includes the port's cable/reader delay, ~8 m equivalent).
  3. cable removal: the fixed REFERENCE tag in the same capture gives each port's
     cable term (raw_ref - true distance), so drift between captures cancels.
     Fallback: a per-port cable term learned from a survey session.
  4. features per port: calibrated range (m), median RSSI, channels, reads.

Localization:
  * model-based:  grid search, Gaussian likelihood of per-port range & RSSI vs the
                  direct-path model, unknown per-tag common offset marginalised,
                  error sizes binned by read strength (learned from a survey).
  * fingerprint:  kNN on relative features (rssi - max, range - median), trained
                  on survey/random sessions with known positions.
"""
import json, os
import numpy as np
import pandas as pd

IN = 0.0254
C = 299_792_458.0
DEG_PER_MHZ_PER_M = 720e6 / C            # round-trip phase slope per metre (~2.40)
F_REF = 915.0
RSSI_BIN_EDGES = np.array([-65.0, -55.0])  # weak | mid | strong
QNAMES = ["FL", "FR", "BL", "BR"]


# ── config / session I/O ─────────────────────────────────────────────────────
def load_config(path):
    c = json.load(open(path))
    z = c["zone"]
    ants = {int(p): np.array([a["x_in"], a["y_in"], a["z_in"]]) * IN for p, a in c["antennas"].items()}
    r = c.get("reference_tag", {})
    cfg = dict(L=z["length_in"] * IN, D=z["depth_in"] * IN, H=z["height_in"] * IN,
               ports=sorted(ants), ants=ants,
               ref_epc=(r.get("epc") or "").replace(" ", "").upper(),
               ref_pos=np.array([r.get("x_in", 0), r.get("y_in", 0), r.get("z_in", 0)]) * IN)
    # zones: optional "zones": {"axis": "x", "edges_in": [0,12,24,36,48], "names": ["1","2","3","4"]}
    # (side-by-side zones along the shelf length).  Default = 2x2 quadrants.
    zc = c.get("zones")
    if zc:
        cfg["zone_axis"] = zc.get("axis", "x")
        cfg["zone_edges"] = np.array(zc["edges_in"], float) * IN
        cfg["zone_names"] = list(zc.get("names") or [str(i + 1) for i in range(len(zc["edges_in"]) - 1)])
    else:
        cfg["zone_axis"], cfg["zone_edges"], cfg["zone_names"] = "quad", None, list(QNAMES)
    # which zone each antenna serves: explicit "zone" name, else the zone it sits in
    names = cfg["zone_names"]
    cfg["port_zone"] = {int(p): names.index(str(a["zone"])) if "zone" in a
                        else quadrant(cfg, a["x_in"] * IN, a["y_in"] * IN)
                        for p, a in c["antennas"].items()}
    return cfg


def load_session(sdir):
    cfg = load_config(os.path.join(sdir, "shelf_config.json"))
    man = pd.read_csv(os.path.join(sdir, "manifest.csv"), dtype={"epc": str, "notes": str})
    man = man.drop_duplicates("id", keep="last").reset_index(drop=True)   # 'b'(redo) -> last wins
    man["epc"] = man.epc.fillna("").str.upper()
    return cfg, man


def load_capture(path):
    df = pd.read_csv(path, dtype={"epc": str})
    df["epc"] = df.epc.str.upper()
    return df


def quadrant(cfg, x_m, y_m):
    """Zone index of a position (metres).  Quadrant mode: FL=0 FR=1 BL=2 BR=3."""
    if cfg.get("zone_axis", "quad") == "quad":
        return int(x_m >= cfg["L"] / 2) + 2 * int(y_m >= cfg["D"] / 2)
    v = x_m if cfg["zone_axis"] == "x" else y_m
    return int(np.clip(np.searchsorted(cfg["zone_edges"][1:-1], v, side="right"), 0, len(cfg["zone_names"]) - 1))


zone_index = quadrant


def n_zones(cfg):
    return len(cfg.get("zone_names", QNAMES))


def zone_names(cfg):
    return cfg.get("zone_names", QNAMES)


def boundary_dist_in(cfg, x_in, y_in):
    """Distance (in) from a position to the nearest INTERNAL zone boundary."""
    x_in, y_in = np.asarray(x_in, float), np.asarray(y_in, float)
    if cfg.get("zone_axis", "quad") == "quad":
        return np.minimum(np.abs(x_in - cfg["L"] / IN / 2), np.abs(y_in - cfg["D"] / IN / 2))
    v = x_in if cfg["zone_axis"] == "x" else y_in
    e = cfg["zone_edges"][1:-1] / IN
    return np.min(np.abs(v[..., None] - e), axis=-1) if len(e) else np.full(v.shape, np.inf)


# ── per-port phase-slope ranging (mod-180 safe) ──────────────────────────────
_COARSE = np.arange(-160.0, 40.0, 0.1)        # doubled-phase slope, deg/MHz


def _slope_fit(dfq, z):
    def best(grid):
        sc = np.abs(np.exp(-1j * np.radians(np.outer(grid, dfq))) @ z)
        k = int(sc.argmax())
        return grid[k], sc[k]
    s, _ = best(_COARSE)
    s, sc = best(np.arange(s - 0.15, s + 0.15, 0.002))
    return s, sc / np.sum(np.abs(z))


def port_stats(g):
    """g = reads of ONE tag on ONE port.  Returns dict of per-port features."""
    ph = np.radians(g.phase_deg.values)
    chans, z, raw_sd, m180_sd = [], [], [], []
    for f, idx in g.groupby("freq_mhz").indices.items():
        a = ph[idx]
        z1, z2 = np.mean(np.exp(1j * a)), np.mean(np.exp(2j * a))
        chans.append(f); z.append(z2)
        raw_sd.append(np.degrees(np.sqrt(-2 * np.log(max(abs(z1), 1e-9)))))
        m180_sd.append(np.degrees(np.sqrt(-2 * np.log(max(abs(z2), 1e-9)))) / 2)
    chans, z = np.array(chans), np.array(z)
    out = dict(n_reads=len(g), n_ch=len(chans), rssi=float(g.rssi_dbm.median()),
               phase_sd_raw=float(np.median(raw_sd)), phase_sd_mod180=float(np.median(m180_sd)),
               slope2=np.nan, conc=np.nan, raw_range=np.nan)
    if len(chans) >= 8:
        s2, conc = _slope_fit(chans - F_REF, z)
        out.update(slope2=s2, conc=conc,
                   # Walmart refs: slope gets MORE NEGATIVE with distance -> range = -slope
                   raw_range=-(s2 / 2) / DEG_PER_MHZ_PER_M)
    return out


def capture_features(cap, cfg, epcs=None, min_reads=5):
    """-> DataFrame rows (epc, port, features..., cable_ref, range_ref)."""
    rows = []
    for (epc, port), g in cap.groupby(["epc", "antenna"]):
        if port not in cfg["ants"] or len(g) < min_reads:
            continue
        if epcs is not None and epc not in epcs and epc != cfg["ref_epc"]:
            continue
        rows.append(dict(epc=epc, port=int(port), **port_stats(g)))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # per-capture cable term from the reference tag
    df["cable_ref"] = np.nan
    if cfg["ref_epc"]:
        ref = df[df.epc == cfg["ref_epc"]].set_index("port")
        for p, r in ref.iterrows():
            d_true = np.linalg.norm(cfg["ants"][p] - cfg["ref_pos"])
            df.loc[df.port == p, "cable_ref"] = r.raw_range - d_true
    df["range_ref"] = df.raw_range - df.cable_ref
    return df


def session_features(sdir, man=None, cfg=None):
    """Features for every capture of a session -> long DataFrame (one row per capture x port, test tag only)."""
    if cfg is None:
        cfg, man = load_session(sdir)
    out = []
    for _, m in man.iterrows():
        f = os.path.join(sdir, m.file)
        if not os.path.exists(f):
            continue
        cap = load_capture(f)
        epcs = {m.epc} if m.epc else None
        df = capture_features(cap, cfg, epcs)
        if df.empty:
            continue
        df = df[df.epc != cfg["ref_epc"]] if cfg["ref_epc"] else df
        df.insert(0, "id", m.id)
        for c in ["x_in", "y_in", "z_in", "orient", "quadrant", "tx_power_dbm"]:
            if c in m:
                df[c] = m[c]
        out.append(df)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def fit_session_cable(feat, cfg):
    """Per-port cable term from known positions (fallback when no reference tag)."""
    t = truth_xyz(feat)
    d = np.array([np.linalg.norm(cfg["ants"][p] - xyz) for p, xyz in zip(feat.port, t)])
    return (feat.raw_range - d).groupby(feat.port).median().to_dict()


def session_cable(feat, cfg, fallback=None):
    """Per-port cable term for a session.

    Preferred: per-port MEDIAN of the reference-tag estimate over all captures of
    the session.  (Using each capture's own reference reading is noisier: the
    reference tag sees multipath that changes as boxes move -> ~25-35 cm per-port
    noise in the fake data.  The per-capture value is kept only as a drift check.)
    Missing ports -> `fallback` dict, else a fit to known positions if available."""
    cab, src = {}, {}
    if "cable_ref" in feat and feat.cable_ref.notna().any():
        g = feat.dropna(subset=["cable_ref"]).groupby("port")
        cab.update({p: v for p, v in g.cable_ref.median().items() if g.size()[p] >= 3})
        src.update({p: "reference tag (session median)" for p in cab})
    missing = [p for p in cfg["ports"] if p not in cab]
    if missing and fallback:
        for p in missing:
            if p in fallback:
                cab[p], src[p] = fallback[p], "fallback (other session)"
    missing = [p for p in cfg["ports"] if p not in cab]
    if missing and "x_in" in feat and feat.x_in.notna().any():
        fit = fit_session_cable(feat.dropna(subset=["x_in"]), cfg)
        for p in missing:
            if p in fit:
                cab[p], src[p] = fit[p], "fit to known positions"
    return cab, src


def apply_cable(feat, cable):
    feat = feat.copy()
    feat["range"] = feat.raw_range - feat.port.map(cable)
    return feat


def truth_xyz(feat):
    return np.column_stack([feat.x_in, feat.y_in, feat.z_in]).astype(float) * IN


# ── antenna model (down-facing patch, same as the simulation) ────────────────
def patch_gain(vec):
    n = np.linalg.norm(vec, axis=-1)
    cos_t = -vec[..., 2] / np.maximum(n, 1e-9)
    return np.maximum(np.where(cos_t > 0, np.clip(cos_t, 0, 1) ** 1.74, 0.0), 0.1)


def rssi_bin(r):
    return np.digitize(r, RSSI_BIN_EDGES)


def model_residuals(feat, cfg):
    """Per-port residuals vs direct-path model at the TRUE position, per-capture
    common offset removed (median over ports)."""
    t = truth_xyz(feat)
    A = np.array([cfg["ants"][p] for p in feat.port])
    d = np.linalg.norm(t - A, axis=1)
    g = patch_gain(t - A)
    f = feat.assign(d_true=d, rssi_model=40 * np.log10(g / d))
    p0 = float(np.median(f.rssi - f.rssi_model))
    f["r_range"] = f.range - f.d_true
    f["r_rssi"] = f.rssi - f.rssi_model - p0
    for c in ["r_range", "r_rssi"]:
        f[c + "_rel"] = f[c] - f.groupby("id")[c].transform("median")
    return f, p0


def learn_model(feat, cfg):
    """Error sizes (by read-strength bin) + RSSI offset from a survey with known positions."""
    f, p0 = model_residuals(feat, cfg)
    mad = lambda x: 1.4826 * np.median(np.abs(x - np.median(x))) if len(x) > 3 else np.nan
    b = rssi_bin(f.rssi.values)
    rr, rd = f.r_range_rel.values, f.r_rssi_rel.values
    sr = np.array([mad(rr[(b == k) & ~np.isnan(rr)]) for k in range(3)])
    sd = np.array([mad(rd[b == k]) for k in range(3)])
    # fill empty bins with the nearest available value, inflate 20 % like the sim
    for arr in (sr, sd):
        ok = ~np.isnan(arr)
        if ok.any():
            arr[~ok] = np.interp(np.flatnonzero(~ok), np.flatnonzero(ok), arr[ok])
    return dict(p0=p0, sig_range=1.2 * sr, sig_rssi=1.2 * sd)


class ModelLocalizer:
    TAG_GAIN_SD = 6.0      # dB, unknown per-tag gain (common to all ports)
    TAG_BIAS_M = 0.21      # m, unknown per-tag phase slope (0.5 deg/MHz) as common range bias

    def __init__(self, cfg, model, step_in=1.0, z_in=(2, 5, 8, 11, 14)):
        gx = np.arange(step_in / 2, cfg["L"] / IN, step_in) * IN
        gy = np.arange(step_in / 2, cfg["D"] / IN, step_in) * IN
        X, Y, Z = np.meshgrid(gx, gy, np.array(z_in) * IN, indexing="ij")
        self.P = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        self.q = np.array([quadrant(cfg, x, y) for x, y in self.P[:, :2]])
        self.nz = n_zones(cfg)
        self.ports = cfg["ports"]
        A = np.array([cfg["ants"][p] for p in self.ports])
        v = self.P[:, None, :] - A[None]
        self.d = np.linalg.norm(v, axis=2)
        self.rssi_pred = 40 * np.log10(patch_gain(v) / self.d) + model["p0"]
        self.m = model

    @staticmethod
    def _ll(r, sig, common):
        k = r.shape[1]
        S = np.diag(np.asarray(sig) ** 2) + common ** 2 * np.ones((k, k))
        return -0.5 * np.einsum("pi,ij,pj->p", r, np.linalg.inv(S), r)

    def locate(self, rows, use_phase=True, use_rssi=True):
        """rows: features of one tag in one capture (one row per port)."""
        idx = [self.ports.index(p) for p in rows.port]
        rssi = rows.rssi.values
        b = rssi_bin(rssi)
        s = np.zeros(len(self.P))
        if use_rssi:
            s += self._ll(rssi[None] - self.rssi_pred[:, idx], self.m["sig_rssi"][b], self.TAG_GAIN_SD)
        rg = rows.range.values
        ok = ~np.isnan(rg)
        if use_phase and ok.sum() >= 1:
            ii = np.array(idx)[ok]
            s += self._ll(rg[ok][None] - self.d[:, ii], self.m["sig_range"][b[ok]], self.TAG_BIAS_M)
        w = np.exp(s - s.max()); w /= w.sum()
        qp = np.bincount(self.q, weights=w, minlength=self.nz)
        return qp, self.P[int(np.argmax(s))]


# ── fingerprint (kNN) ────────────────────────────────────────────────────────
def fp_vectors(feat, ports, use_phase=True):
    """One vector per capture id: [rssi_rel per port, range_rel*10 per port]."""
    ids, X = [], []
    for cid, g in feat.groupby("id", sort=False):
        r = dict(zip(g.port, g.rssi)); d = dict(zip(g.port, g.range))
        mx = max(r.values())
        dv = [v for v in d.values() if not np.isnan(v)]
        med = np.median(dv) if dv else 0.0
        v = [r[p] - mx if p in r else -40.0 for p in ports]
        if use_phase:
            v += [10 * (d[p] - med) if p in d and not np.isnan(d[p]) else 6.0 for p in ports]
        ids.append(cid); X.append(v)
    return ids, np.array(X)


class Fingerprint:
    def __init__(self, ports, k=None, use_phase=True):
        self.ports, self.k, self.use_phase = ports, k, use_phase

    def fit(self, feat, cfg):
        ids, X = fp_vectors(feat, self.ports, self.use_phase)
        self.k = self.k or knn_k(len(ids))
        first = feat.groupby("id", sort=False).first().loc[ids]
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        self.X = (X - self.mu) / self.sd
        self.xy = first[["x_in", "y_in"]].values.astype(float)
        self.q = np.array([quadrant(cfg, x * IN, y * IN) for x, y in self.xy])
        self.nz = n_zones(cfg)
        return self

    def predict(self, feat):
        ids, X = fp_vectors(feat, self.ports, self.use_phase)
        B = (X - self.mu) / self.sd
        D = ((B[:, None, :] - self.X[None]) ** 2).sum(-1)
        k = min(self.k, len(self.X))
        nn = np.argsort(D, 1)[:, :k]
        w = 1 / (np.sqrt(np.take_along_axis(D, nn, 1)) + 1e-3)
        P = np.stack([(w * (self.q[nn] == q)).sum(1) for q in range(self.nz)], 1)
        P /= P.sum(1, keepdims=True)
        xy = (w[..., None] * self.xy[nn]).sum(1) / w.sum(1, keepdims=True)
        return ids, P, xy


def knn_k(n_train, k_max=15):
    """k for kNN that scales with the training set (a 48-spot survey -> k=5, not 15)."""
    return int(min(k_max, max(5, n_train // 8)))


# ── per-channel phase pattern ("signature") fingerprint ──────────────────────
# On the real shelf the phase is NOT linear in frequency (multipath from the metal),
# so the slope-range above is unreliable.  The whole per-channel pattern is,
# however, repeatable for the same spot (move test: match 0.90-0.95 after the box
# was lifted and put back) and changes gradually with position (~0.5 at 4-8").
# So we match the pattern itself against a surveyed lookup table.
def chan_signature(cap, epc, port):
    """Per-channel unit vector of the DOUBLED phase (180-deg flip safe), indexed by freq."""
    g = cap[(cap.epc == epc) & (cap.antenna == port)]
    if g.empty:
        return pd.Series(dtype=complex)
    z = g.groupby("freq_mhz").phase_deg.apply(lambda a: np.mean(np.exp(2j * np.radians(a.values))))
    return z / np.abs(z).clip(1e-9)


def capture_signatures(cap, epc, ports, ref_epc="", min_ref_ch=15):
    """{port: {'raw': sig, 'ref': sig / reference-tag sig (same channel), or None}}"""
    out = {}
    for p in ports:
        s = chan_signature(cap, epc, p)
        if s.empty:
            continue
        ref = None
        if ref_epc:
            r = chan_signature(cap, ref_epc, p)
            if len(r) >= min_ref_ch:
                j = s.index.intersection(r.index)
                ref = s[j] * np.conj(r[j])
        out[p] = dict(raw=s, ref=ref)
    return out


def session_signatures(sdir, man, cfg, suffix=""):
    """{capture id(+suffix): {port: {'raw','ref'}}} for the test tag of every capture."""
    out = {}
    for _, m in man.iterrows():
        f = os.path.join(sdir, m.file)
        if os.path.exists(f) and m.epc:
            out[str(m.id) + suffix] = capture_signatures(load_capture(f), m.epc, cfg["ports"], cfg["ref_epc"])
    return out


def sig_match(a, b, min_ch=6):
    """|mean(a * conj(b))| over common channels: 1 = same pattern (up to a constant
    offset), ~sqrt(pi/4/N) for unrelated patterns.  Uses the reference-tag-corrected
    pattern when both captures have it, else the raw one.  NaN if too few channels."""
    if a is None or b is None:
        return np.nan
    key = "ref" if a["ref"] is not None and b["ref"] is not None else "raw"
    sa, sb = a[key], b[key]
    j = sa.index.intersection(sb.index)
    if len(j) < min_ch:
        return np.nan
    return float(abs(np.mean(sa[j].values * np.conj(sb[j].values))))


class PatternFingerprint:
    """kNN on the per-channel phase pattern of each port, optionally + RSSI.
    distance = mean over ports of (1 - match)   [+ RSSI distance, both scaled to
    their median over the training set so neither dominates]."""

    def __init__(self, ports, use_rssi=True, k=None):
        self.ports, self.use_rssi, self.k = ports, use_rssi, k

    def fit(self, feat, sigs, cfg):
        ids, X = fp_vectors(feat, self.ports, use_phase=False)
        first = feat.groupby("id", sort=False).first().loc[ids]
        self.ids, self.sigs = ids, [sigs.get(i, {}) for i in ids]
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        self.X = (X - self.mu) / self.sd
        self.xy = first[["x_in", "y_in"]].values.astype(float)
        self.q = np.array([quadrant(cfg, x * IN, y * IN) for x, y in self.xy])
        self.nz = n_zones(cfg)
        self.k = self.k or knn_k(len(ids))
        return self

    def _dphase(self, s):
        D = np.ones(len(self.sigs))
        for i, t in enumerate(self.sigs):
            m = [sig_match(s.get(p), t.get(p)) for p in self.ports]
            m = [v for v in m if not np.isnan(v)]
            if m:
                D[i] = 1 - np.mean(m)
        return D

    def predict(self, feat, sigs):
        ids, X = fp_vectors(feat, self.ports, use_phase=False)
        B = (X - self.mu) / self.sd
        Dp = np.array([self._dphase(sigs.get(i, {})) for i in ids])
        D = Dp / max(np.median(Dp), 1e-6)
        if self.use_rssi:
            Dr = np.sqrt(((B[:, None, :] - self.X[None]) ** 2).sum(-1))
            D = D + Dr / max(np.median(Dr), 1e-6)
        k = min(self.k, len(self.X))
        nn = np.argsort(D, 1)[:, :k]
        w = 1 / (np.take_along_axis(D, nn, 1) + 1e-3)
        P = np.stack([(w * (self.q[nn] == q)).sum(1) for q in range(self.nz)], 1)
        P /= P.sum(1, keepdims=True)
        xy = (w[..., None] * self.xy[nn]).sum(1) / w.sum(1, keepdims=True)
        return ids, P, xy


class StrongestAntenna:
    """No training: the zone of the antenna with the highest median RSSI wins.
    Confidence = softmax of the per-port RSSI (temperature T dB), summed per zone,
    so a 1-2 dB lead gives a low-confidence call and a 10 dB lead a confident one.
    Estimated x/y = RSSI-weighted mean of the antenna positions."""

    def __init__(self, cfg, ports=None, T=3.0):
        self.cfg, self.ports, self.T = cfg, ports or cfg["ports"], T
        self.nz = n_zones(cfg)

    def predict(self, feat):
        ids, P, XY = [], [], []
        for cid, g in feat[feat.port.isin(self.ports)].groupby("id", sort=False):
            r = g.rssi.values
            w = np.exp((r - r.max()) / self.T); w /= w.sum()
            pz = np.zeros(self.nz)
            for p, wi in zip(g.port, w):
                pz[self.cfg["port_zone"][p]] += wi
            xy = sum(wi * self.cfg["ants"][p][:2] for p, wi in zip(g.port, w)) / IN
            ids.append(cid); P.append(pz); XY.append(xy)
        return ids, np.array(P), np.array(XY)


def score(true_q, P, est_xy, true_xy, bdist, linear_zones=False):
    qh = P.argmax(1); conf = P.max(1); q2 = np.argsort(P, 1)[:, -2]
    ok = qh == true_q
    err = np.linalg.norm(est_xy - true_xy, axis=1)
    c = conf >= 0.9
    return dict(n=len(ok), acc=ok.mean(), acc_3in_from_edge=ok[bdist >= 3].mean() if (bdist >= 3).any() else np.nan,
                acc_when_conf90=ok[c].mean() if c.any() else np.nan, coverage_conf90=c.mean(),
                top2_acc=(ok | (q2 == true_q)).mean(), p_err_lt_12in=(err < 12).mean(),
                p_err_lt_18in=(err < 18).mean(), median_err_in=float(np.median(err)),
                within_1_zone=(np.abs(qh - true_q) <= 1).mean() if linear_zones else np.nan)
