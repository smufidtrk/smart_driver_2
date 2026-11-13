"""
Free-space model of a 4x4 Butler-matrix array (4 patches, 16 cm apart) over the truck shelf,
compared with the current 4 separate antennas. No reflections: this is the best case.

  python analysis/butler_sim.py [height_in ...]

Shelf: 48" wide (x), 24" deep (y, 0 = aisle edge), tags on the floor (z = 0).
Butler array: centred at x = 24", at the aisle edge, `height` above the tags, elements in a row along x,
tilted 45 deg down toward the back wall (same mount as the current antennas).
Port phases from the circuit table (Ant 1..4): 1R -45/-90/-135/-180, 2L -135/0/135/270,
2R -90/-225/0/-135, 1L -180/-135/-90/-45. Reader TX and RX on the same port -> two-way power = |E|^4.
Current setup: one patch per zone above the zone centre (x = 6, 18, 30, 42"), same tilt, same height.
Field sums spherical waves from each element (near field included), patch pattern cos^q (70 deg beam).
Reports, per box of the 32-box grid, the strongest port and own-vs-best-other contrast (dB).
"""
import sys
import numpy as np

C = 299792458.0
FREQS = np.linspace(902.75e6, 927.25e6, 50)
IN = 0.0254
Q = np.log(0.5) / (2 * np.log(np.cos(np.radians(35))))  # field exponent: -3 dB (power) at 35 deg off boresight
BORE = np.array([0, np.cos(np.radians(45)), -np.sin(np.radians(45))])
XS = np.array([3, 9, 15, 21, 27, 33, 39, 45.0])
YS = np.array([3, 9, 15, 21.0])
ZONE = lambda x: int(x // 12) + 1
PHASES = {  # port: per-element phase (deg), Ant 1..4
    "1R": [-45, -90, -135, -180], "2L": [-135, 0, 135, 270],
    "2R": [-90, -225, 0, -135], "1L": [-180, -135, -90, -45]}


def field(elems, phases_deg, tag, f):
    k = 2 * np.pi * f / C
    v = tag[None, :] - elems
    r = np.linalg.norm(v, axis=1)
    cosb = np.clip((v @ BORE) / r, 1e-6, 1)
    e = cosb ** Q * np.exp(-1j * (k * r - np.radians(phases_deg))) / r
    return e.sum()


def twoway_db(elems, phases, tag):
    p = np.mean([abs(field(elems, phases, tag, f)) ** 4 for f in FREQS])  # average over the hop band
    return 10 * np.log10(p)


def run(h_in):
    h = h_in * IN
    off = (np.arange(4) - 1.5) * 0.16
    arr = np.stack([24 * IN + off, np.zeros(4), np.full(4, h)], axis=1)  # Ant 1 at the zone-1 end
    tags = [(x, y) for y in YS for x in XS]
    P = {p: np.array([twoway_db(arr, ph, np.array([x * IN, y * IN, 0])) for x, y in tags]) for p, ph in PHASES.items()}
    # assign each beam to the zone where it is strongest (relative to the other beams)
    M = np.array([P[p] for p in PHASES])
    rel = M - M.max(axis=0)
    zmean = {p: [rel[i][[ZONE(x) == z for x, _ in tags]].mean() for z in (1, 2, 3, 4)] for i, p in enumerate(PHASES)}
    beam_of = {}
    for p in sorted(PHASES, key=lambda p: -max(zmean[p])):
        z = int(np.argmax(zmean[p])) + 1
        while z in beam_of.values():
            zm = list(zmean[p]); zm[z - 1] = -1e9; z = int(np.argmax(zm)) + 1; zmean[p] = zm
        beam_of[p] = z
    order = sorted(PHASES, key=lambda p: beam_of[p])
    B = np.array([P[p] for p in order])  # rows = zones 1..4
    # current setup: one antenna per zone centre
    cur = []
    for zc in (6, 18, 30, 42):
        el = np.array([[zc * IN, 0, h]])
        cur.append([twoway_db(el, [0], np.array([x * IN, y * IN, 0])) for x, y in tags])
    A = np.array(cur)
    print(f"\n=== antennas {h_in}\" above the tags | beam -> zone: " + ", ".join(f"{p}->Z{beam_of[p]}" for p in order))
    for name, G in (("Butler beams", B), ("current 4 antennas", A)):
        ok, con = 0, []
        grid = []
        for i, (x, y) in enumerate(tags):
            z = ZONE(x)
            own = G[z - 1, i]; other = np.delete(G[:, i], z - 1).max()
            con.append(own - other); ok += int(np.argmax(G[:, i]) + 1 == z)
            grid.append(own - other)
        con = np.array(con).reshape(len(YS), len(XS))
        print(f"{name}: strongest = right zone {ok}/32, own-vs-best-other (two-way dB), rows y=21..3\", cols x=3..45\":")
        for j in range(len(YS) - 1, -1, -1):
            print(f"   y={YS[j]:>2.0f}\"  " + " ".join(f"{v:+6.1f}" for v in con[j]))
        print(f"   median {np.median(con):+.1f} dB, zone-centre-ish boxes (x=9,15,33,39 excluded edges) "
              f"min {con.min():+.1f} dB")


if __name__ == "__main__":
    lam = C / 915e6
    print(f"lambda {lam*100:.1f} cm, spacing 16 cm = {0.16/lam:.2f} lambda; far-field distance 2D^2/lambda for D=0.64 m: {2*0.64**2/lam:.1f} m")
    for dphi in (45, 135):
        th = np.degrees(np.arcsin(np.radians(dphi) / (2 * np.pi * 0.16 / lam)))
        hp = np.degrees(0.886 * lam / (4 * 0.16 * np.cos(np.radians(th))))
        print(f"  progressive {dphi:>3} deg -> beam {th:4.1f} deg off broadside, one-way HPBW ~{hp:.0f} deg (array factor only)")
    for h in [float(a) for a in sys.argv[1:]] or [12.0, 24.0]:
        run(h)
