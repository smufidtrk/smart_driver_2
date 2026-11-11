"""
Smart Driver shelf — how many antennas to tell which QUADRANT a stationary tag is in?
=====================================================================================
Monte-Carlo simulation of one shelf zone (32" L x 24" D x 20" H, metal floor,
metal shelf above, metal back wall, partial side uprights, open front), split
into 4 quadrants (16" x 12" each):

        y (depth) ^  back wall
               24 |  BL  |  BR
               12 |------+------
                0 |  FL  |  FR       (front / open aisle side)
                  +-------------> x (length)
                  0     16     32

Antennas are patch antennas on the underside of the shelf above, pointing down.

Physics model (per tag, per antenna, per channel; 50 FCC channels, 902.75-927.25 MHz)
  * One-way channel h = sum over paths: direct + image paths off the metal
    surfaces (up to one bounce per axis, incl. floor<->ceiling double bounce)
    + random point scatterers (other boxes / contents).
  * Backscatter round trip H = h^2 (monostatic).
  * Patch pattern cos^1.74 (~70 deg beamwidth, -20 dB back lobe); tag = dipole
    with random orientation (top-face or side-face label).
  * Reported phase = angle(H) + cable(ant,f) + tag_offset(f) + noise, then
    MOD 180 deg (Impinj pi-ambiguity seen in the Walmart ref captures).
  * Per-read phase noise 3-4 deg at good RSSI (measured: 2-6 deg on Walmart refs
    after mod-180), grows at low RSSI; 4 reads/channel/antenna averaged.
  * Tag phase offset has an unknown per-tag slope ~N(0, 0.5 deg/MHz) (tag chip
    detuning by box contents) -> this is what killed absolute single-antenna
    ranging in the Walmart 1-5 ft refs.
  * Per-antenna cable term calibrated with a reference tag; residual error
    5 deg offset + 0.1 deg/MHz slope.
  * RSSI: 0.5 dB quantised, +/-1.5 dB noise, unknown per-tag gain (+/-6 dB),
    read threshold -75 dBm.

Localizer (model-based; only a per-site error-size calibration, no fingerprinting):
  PHASE  -> per antenna, phase-slope range across the 50 channels on DOUBLED
            (mod-180-safe) phase  (= Walmart estimate_distance.py method).
            The unknown tag-slope bias is common to all antennas and is
            marginalised, so effectively the cross-antenna range DIFFERENCES
            carry the information.
  RSSI   -> per-antenna RSSI vs direct-path model, unknown per-tag gain
            marginalised (again: cross-antenna differences).
  Grid search over (x, y, z); Gaussian log-likelihood; quadrant posterior =
  likelihood mass per quadrant.  Methods: rssi / phase / both, plus the naive
  'rssi_max' (strongest antenna wins) baseline for the 1-antenna-per-quadrant layout.

  Note: sub-wavelength phase 'holography' was tried first and rejected — with
  the 180-deg ambiguity it has grating lobes every ~3 in and picks the wrong
  lobe even with zero multipath.

Run:  python sim/quadrant_sim.py            (writes sim/results/*)
"""
import os, sys, json, time
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

IN = 0.0254
C = 299_792_458.0
LX, LY, LZ = 32 * IN, 24 * IN, 20 * IN
ANT_Z = 19.0 * IN                     # antenna face, 1" below shelf above
FREQS = 902.75e6 + 0.5e6 * np.arange(50)
F0 = FREQS.mean()
DF_MHZ = (FREQS - F0) / 1e6
QNAMES = ["FL", "FR", "BL", "BR"]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# ── Antenna layouts (x, y in inches; all on the ceiling, facing down) ────────
LAYOUTS = {
    "1 ant - center":          [(16, 12)],
    "2 ant - left/right":      [(8, 12), (24, 12)],
    "2 ant - diagonal":        [(8, 6), (24, 18)],
    "3 ant - triangle":        [(8, 6), (24, 6), (16, 18)],
    "4 ant - 1 per quadrant":  [(8, 6), (24, 6), (8, 18), (24, 18)],
    "6 ant - 3x2 grid":        [(5.5, 6), (16, 6), (26.5, 6), (5.5, 18), (16, 18), (26.5, 18)],
}

# ── Multipath severity scenarios ─────────────────────────────────────────────
SEVERITY = {
    #            floor ceil  back  sides  n_scat  scat_scale(m)
    "mild":     (0.50, 0.50, 0.40, 0.20, 2, 0.010),
    "typical":  (0.80, 0.80, 0.70, 0.40, 4, 0.020),
    "harsh":    (0.92, 0.92, 0.90, 0.70, 8, 0.035),
}

RSSI_P0 = -54.0          # dBm offset: tag 30 cm on boresight -> ~-33 dBm (Walmart ref_30cm: -32 dBm)
RSSI_FLOOR = -78.0
N_READS = 4
TAG_SLOPE_SD = 0.5       # deg/MHz (round trip)
CAL_OFF_SD, CAL_SLOPE_SD = 5.0, 0.1
TAG_GAIN_SD = 6.0
RSSI_NOISE = 1.5
RSSI_MODEL_SD = 4.0      # localizer's assumed RSSI model mismatch (dB) - overwritten per severity
RANGE_MODEL_SD = 0.06    # localizer's assumed per-antenna slope-range error (m) - "
TAG_BIAS_M = TAG_SLOPE_SD / 2.4   # tag slope (deg/MHz) -> common range bias (m)

GRID_STEP = 1.0          # inches
GZ = np.array([2, 5, 8, 11, 14]) * IN


def patch_gain(vec):
    """Amplitude pattern of a down-facing patch; vec = direction(s) from antenna."""
    n = np.linalg.norm(vec, axis=-1)
    cos_t = -vec[..., 2] / np.maximum(n, 1e-9)          # boresight = -z
    g = np.where(cos_t > 0, np.clip(cos_t, 0, 1) ** 1.74, 0.0)
    return np.maximum(g, 0.1)                             # -20 dB back/side floor


def dipole_gain(axis, vec):
    u = vec / np.maximum(np.linalg.norm(vec, axis=-1, keepdims=True), 1e-12)
    c = np.abs(u @ axis)
    return np.maximum(np.sqrt(np.clip(1 - c ** 2, 0, 1)), 0.05)


def image_set(gam):
    """List of (sign_vec D, offset o, coefficient) so image = D*t + o."""
    gf, gc, gb, gs = gam
    xs = [((1, 0.0), 1.0), ((-1, 0.0), gs), ((-1, 2 * LX), gs)]          # none / left / right
    ys = [((1, 0.0), 1.0), ((-1, 2 * LY), gb)]                            # none / back
    zs = [((1, 0.0), 1.0), ((-1, 0.0), gf), ((-1, 2 * LZ), gc),           # none / floor / ceil
          ((1, 2 * LZ), gf * gc), ((1, -2 * LZ), gf * gc)]                # floor<->ceil double
    out = []
    for (sx, ox), cx in xs:
        for (sy, oy), cy in ys:
            for (sz, oz), cz in zs:
                out.append((np.array([sx, sy, sz], float), np.array([ox, oy, oz]), cx * cy * cz))
    return out


class Shelf:
    """One physical shelf instance (fixed structure + per-antenna cable cal error)."""

    def __init__(self, rng, sev, n_ant):
        gf, gc, gb, gs, self.n_scat, self.scat_scale = SEVERITY[sev]
        jit = lambda g: -g * np.exp(1j * np.radians(rng.normal(0, 15)))  # metal ~ -1
        self.images = image_set((jit(gf), jit(gc), jit(gb), jit(gs)))
        self.cal_off = np.radians(rng.normal(0, CAL_OFF_SD, n_ant))
        self.cal_slope = np.radians(rng.normal(0, CAL_SLOPE_SD, n_ant))  # rad/MHz


def one_way_h(shelf, ants, tag, axis, scat_pos, scat_amp):
    """h[a, f] one-way complex channel."""
    k = 2 * np.pi * FREQS / C                                           # (F,)
    h = np.zeros((len(ants), len(FREQS)), complex)
    for D, o, coef in shelf.images:
        img = D * tag + o
        v = img - ants                                                  # (A,3) antenna->image
        L = np.linalg.norm(v, axis=1)
        v_tag = D * (ants - img)                                        # arrival dir at tag
        amp = coef * patch_gain(v) * dipole_gain(axis, v_tag) / L
        h += amp[:, None] * np.exp(-1j * np.outer(L, k))
    for sp, sa in zip(scat_pos, scat_amp):
        v1 = sp - ants
        d1 = np.linalg.norm(v1, axis=1)
        d2 = np.linalg.norm(tag - sp)
        amp = sa * patch_gain(v1) * dipole_gain(axis, sp - tag) / (d1 * d2)
        h += amp[:, None] * np.exp(-1j * np.outer(d1 + d2, k))
    return h


def random_tag(rng):
    pos = np.array([rng.uniform(1, 31) * IN, rng.uniform(1, 23) * IN, rng.uniform(1, 14) * IN])
    if rng.random() < 0.5:                         # label on top face: horizontal dipole
        a = rng.uniform(0, np.pi); axis = np.array([np.cos(a), np.sin(a), 0.0])
    else:                                          # label on side face
        axis = np.array([0, 0, 1.0]) if rng.random() < 0.5 else \
            np.array([np.cos(a := rng.uniform(0, np.pi)), np.sin(a), 0.0])
    return pos, axis


def measure(rng, shelf, ants, pos, axis):
    n_s = rng.poisson(shelf.n_scat)
    sp = np.column_stack([rng.uniform(0, LX, n_s), rng.uniform(0, LY, n_s), rng.uniform(0, LZ, n_s)])
    sa = shelf.scat_scale * rng.rayleigh(1.0, n_s) * np.exp(1j * rng.uniform(0, 2 * np.pi, n_s))
    h = one_way_h(shelf, ants, pos, axis, sp, sa)
    H = h ** 2
    tag_gain = rng.normal(0, TAG_GAIN_SD)
    pw = 10 * np.log10(np.mean(np.abs(H) ** 2, axis=1)) + RSSI_P0 + tag_gain
    rssi = np.round((pw + rng.normal(0, RSSI_NOISE, len(ants))) * 2) / 2
    seen = rssi > RSSI_FLOOR
    # phase
    tag_ph = rng.uniform(0, 2 * np.pi) + np.radians(rng.normal(0, TAG_SLOPE_SD)) * DF_MHZ
    cable_resid = shelf.cal_off[:, None] + shelf.cal_slope[:, None] * DF_MHZ[None, :]
    sig_read = np.radians(np.clip(3.5 * 10 ** ((-50 - rssi) / 20), 3.0, 40.0))
    noise = rng.normal(0, 1, H.shape) * (sig_read / np.sqrt(N_READS))[:, None]
    phi = np.angle(H) + tag_ph[None, :] + cable_resid + noise
    phi = np.mod(phi, np.pi)                       # Impinj 180-deg ambiguity
    return phi, rssi, seen


SLOPE_GRID = np.radians(np.arange(-12.0, 4.0, 0.005))          # doubled-phase slope, rad/MHz
_STEER = np.exp(-1j * np.outer(DF_MHZ, SLOPE_GRID))              # (F,S)
RAD_PER_MHZ_PER_M = 8 * np.pi * 1e6 / C                          # doubled round-trip slope


def slope_range(phi):
    """Per-antenna phase-slope range (m) from mod-180 phases phi[a,f].
    Same method as Walmart estimate_distance.py, but on DOUBLED phase so the
    180-deg flips vanish.  Includes the unknown tag-slope bias (common to all
    antennas) -> the localizer marginalises it."""
    z = np.exp(1j * 2 * phi)                                      # (a,F)
    sc = np.abs(z @ _STEER)                                       # (a,S)
    k = sc.argmax(1)
    return -SLOPE_GRID[k] / RAD_PER_MHZ_PER_M, sc.max(1) / phi.shape[1]


RSSI_BIN_EDGES = np.array([-65.0, -55.0])       # weak | mid | strong read


def rssi_bin(r):
    return np.digitize(r, RSSI_BIN_EDGES)


class Localizer:
    def __init__(self, ants):
        gx = np.arange(0.5, 32, GRID_STEP) * IN
        gy = np.arange(0.5, 24, GRID_STEP) * IN
        X, Y, Z = np.meshgrid(gx, gy, GZ, indexing="ij")
        self.P = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        self.q = (self.P[:, 0] > LX / 2).astype(int) + 2 * (self.P[:, 1] > LY / 2).astype(int)
        v = self.P[:, None, :] - ants[None, :, :]                        # (P,A,3)
        self.d = np.linalg.norm(v, axis=2)                               # (P,A) direct range
        self.rssi_pred = 40 * np.log10(patch_gain(v) / self.d) + RSSI_P0  # direct path, no tag
        self.n_ant = len(ants)
        self.sig_rssi = np.full(3, RSSI_MODEL_SD)
        self.sig_range = np.full(3, RANGE_MODEL_SD)

    @staticmethod
    def _gauss_common(r, sig, sig_common):
        """log-lik of residuals r (P,k) = common_offset + independent noise with
        per-antenna sd sig (k,), common offset marginalised."""
        k = r.shape[1]
        Sig = np.diag(np.asarray(sig) ** 2) + sig_common ** 2 * np.ones((k, k))
        return -0.5 * np.einsum("pi,ij,pj->p", r, np.linalg.inv(Sig), r)

    def score(self, phi, rssi, seen, use_phase=True, use_rssi=True):
        s = np.zeros(len(self.P))
        idx = np.where(seen)[0]
        if len(idx) == 0:
            return s
        if use_phase:
            d_est, _ = slope_range(phi[idx])
            r = d_est[None, :] - self.d[:, idx]
            s += self._gauss_common(r, self.sig_range[rssi_bin(rssi[idx])], TAG_BIAS_M)
        if use_rssi:
            r = rssi[idx][None, :] - self.rssi_pred[:, idx]
            s += self._gauss_common(r, self.sig_rssi[rssi_bin(rssi[idx])], TAG_GAIN_SD)
        return s

    def decide(self, s):
        w = np.exp(s - s.max()); w /= w.sum()
        qp = np.bincount(self.q, weights=w, minlength=4)
        best = int(np.argmax(s))
        self.last_q2 = int(np.argsort(qp)[-2])          # runner-up quadrant
        return int(np.argmax(qp)), float(qp.max()), self.P[best]


def quadrant(pos):
    return int(pos[0] > LX / 2) + 2 * int(pos[1] > LY / 2)


def boundary_dist_in(pos):
    return min(abs(pos[0] - LX / 2), abs(pos[1] - LY / 2)) / IN


def calibrate_sigmas(n_tags=300, seed=99):
    """Pilot (independent seed): per severity, measure how far the per-antenna
    slope-range and RSSI deviate from the direct-path model after removing the
    per-tag common offset.  In the field this is the 'calibration walk' with a
    tag at known spots.  Error sizes are binned by read strength (weak antennas
    see relatively more multipath).  Returns {sev: (range_sd_m[3], rssi_sd_db[3])}."""
    ants = np.array([[x * IN, y * IN, ANT_Z] for x, y in LAYOUTS["4 ant - 1 per quadrant"]])
    mad = lambda x: 1.4826 * np.median(np.abs(x - np.median(x)))
    out = {}
    for sev in SEVERITY:
        rng = np.random.default_rng(seed)
        rr, dr, bb = [], [], []
        for i in range(n_tags):
            if i % 50 == 0:
                shelf = Shelf(rng, sev, len(ants))
            pos, axis = random_tag(rng)
            phi, rssi, seen = measure(rng, shelf, ants, pos, axis)
            if seen.sum() < 2:
                continue
            dt = np.linalg.norm(ants - pos, axis=1)
            de, _ = slope_range(phi)
            rp = 40 * np.log10(patch_gain(pos - ants) / dt) + RSSI_P0
            e = (de - dt)[seen]; rr.extend(e - np.median(e))
            e = (rssi - rp)[seen]; dr.extend(e - np.median(e))
            bb.extend(rssi_bin(rssi[seen]))
        rr, dr, bb = map(np.array, (rr, dr, bb))
        out[sev] = (np.array([1.2 * mad(rr[bb == k]) for k in range(3)]),
                    np.array([1.2 * mad(dr[bb == k]) for k in range(3)]))
        print(f"  calib {sev:8s} range sd weak/mid/strong {np.round(out[sev][0]*100,1)} cm"
              f"   rssi sd {np.round(out[sev][1],1)} dB")
    return out


def run(sigmas, n_tags=500, seed=1, layouts=LAYOUTS, severities=SEVERITY,
        methods=("rssi", "phase", "both")):
    rows = []
    for lname, axy in layouts.items():
        ants = np.array([[x * IN, y * IN, ANT_Z] for x, y in axy])
        loc = Localizer(ants)
        for sev in severities:
            loc.sig_range, loc.sig_rssi = sigmas[sev]
            rng = np.random.default_rng(seed)
            t0 = time.time()
            for i in range(n_tags):
                if i % 50 == 0:
                    shelf = Shelf(rng, sev, len(ants))
                pos, axis = random_tag(rng)
                phi, rssi, seen = measure(rng, shelf, ants, pos, axis)
                qt = quadrant(pos)
                rec = dict(layout=lname, n_ant=len(ants), sev=sev, x=pos[0] / IN, y=pos[1] / IN,
                           z=pos[2] / IN, q=qt, bdist=boundary_dist_in(pos), n_seen=int(seen.sum()))
                for m in methods:
                    if m == "rssi_max":
                        order = np.argsort(np.where(seen, rssi, -999))
                        qh, conf = quadrant(ants[order[-1]]), 1.0
                        loc.last_q2 = quadrant(ants[order[-2]]) if len(ants) > 1 else qh
                        est = ants[order[-1]]
                    else:
                        sc = loc.score(phi, rssi, seen, use_phase=m != "rssi", use_rssi=m != "phase")
                        qh, conf, est = loc.decide(sc)
                    rec[f"{m}_q"] = qh
                    rec[f"{m}_conf"] = conf
                    rec[f"{m}_q2"] = loc.last_q2
                    rec[f"{m}_err_in"] = float(np.linalg.norm((est - pos)[:2]) / IN)
                rows.append(rec)
            print(f"  {lname:24s} {sev:8s} {n_tags} tags  {time.time() - t0:5.1f}s", flush=True)
    return rows


if __name__ == "__main__":
    import pandas as pd
    os.makedirs(OUT, exist_ok=True)
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    methods = ("rssi", "phase", "both", "rssi_max")
    sig = calibrate_sigmas()
    json.dump({k: [v[0].tolist(), v[1].tolist()] for k, v in sig.items()},
              open(os.path.join(OUT, "sigmas.json"), "w"), indent=1)
    rows = run(sig, n_tags=n, methods=methods)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "sim_rows.csv"), index=False)
    print(f"wrote {len(df)} rows -> {OUT}")
