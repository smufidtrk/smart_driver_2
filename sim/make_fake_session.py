"""
Generate FAKE sessions in exactly the format capture/ShelfCapture writes, using
the shelf simulator -> lets the analysis pipeline be tested before hardware.

Per-read rows include Impinj-style random 180-deg flips, a per-port cable phase
slope (~-20 deg/MHz like the Walmart refs) with small drift between captures,
per-channel RSSI, and the fixed reference tag in every capture.

  python sim/make_fake_session.py            -> sessions/fake_grid, fake_random,
                                                fake_van2_random, fake_scan
"""
import json, os, shutil, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import quadrant_sim as qs  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN = qs.IN
TEST_EPC = "E2801190200000000000AAAA"
REF_EPC = "E2801190200000000000FFFF"
SEV = sys.argv[1] if len(sys.argv) > 1 else "typical"


def cfg_for_sessions():
    c = json.load(open(os.path.join(ROOT, "shelf_config.json")))
    c["reference_tag"]["epc"] = REF_EPC
    return c


class Van:
    def __init__(self, rng, cfg):
        self.ants = np.array([[a["x_in"], a["y_in"], a["z_in"]] for _, a in sorted(cfg["antennas"].items())]) * IN
        self.shelf = qs.Shelf(rng, SEV, len(self.ants))
        self.cable_slope = rng.uniform(-22, -18, len(self.ants))       # deg/MHz, round trip
        self.cable_off = rng.uniform(0, 360, len(self.ants))
        r = cfg["reference_tag"]
        self.ref_pos = np.array([r["x_in"], r["y_in"], r["z_in"]]) * IN
        self.ref_axis = np.array([1.0, 0, 0])                            # label flat on back wall


def tag_axis(rng, orient):
    a = rng.uniform(0, np.pi)
    if orient == "top" or rng.random() < 0.5:
        return np.array([np.cos(a), np.sin(a), 0.0])
    return np.array([0, 0, 1.0])


def reads_for(rng, van, pos, axis, epc, drift, scat):
    h = qs.one_way_h(van.shelf, van.ants, pos, axis, *scat)
    H = h ** 2                                                           # (A,F)
    gain = rng.normal(0, qs.TAG_GAIN_SD)
    rssi_cf = 20 * np.log10(np.abs(H)) + qs.RSSI_P0 + gain               # per channel
    tag_ph = rng.uniform(0, 360) + rng.normal(0, qs.TAG_SLOPE_SD) * qs.DF_MHZ
    rows = []
    t = 0.0
    for a in range(len(van.ants)):
        if np.median(rssi_cf[a]) < qs.RSSI_FLOOR:
            continue
        cable = van.cable_off[a] + drift[a, 0] + (van.cable_slope[a] + drift[a, 1]) * (qs.FREQS / 1e6 - 902.75)
        for f in range(len(qs.FREQS)):
            r = rssi_cf[a, f]
            if r < qs.RSSI_FLOOR:
                continue
            sig = np.clip(3.5 * 10 ** ((-50 - r) / 20), 3.0, 40.0)
            for _ in range(rng.poisson(4) + 1):
                ph = np.degrees(np.angle(H[a, f])) + tag_ph[f] + cable[f] + rng.normal(0, sig) \
                     + 180.0 * rng.integers(0, 2)
                rr = np.round((r + rng.normal(0, 1.0)) * 2) / 2
                t += rng.uniform(1, 5)
                rows.append(f"{t:.2f},{epc},{ph % 360:.3f},{qs.FREQS[f] / 1e6:.2f},{rr:.1f},{a + 1}")
    return rows


def capture(rng, van, tags):
    n_s = rng.poisson(van.shelf.n_scat)
    sp = np.column_stack([rng.uniform(0, qs.LX, n_s), rng.uniform(0, qs.LY, n_s), rng.uniform(0, qs.LZ, n_s)])
    sa = van.shelf.scat_scale * rng.rayleigh(1.0, n_s) * np.exp(1j * rng.uniform(0, 2 * np.pi, n_s))
    drift = np.column_stack([rng.normal(0, 3, len(van.ants)), rng.normal(0, 0.03, len(van.ants))])
    rows = reads_for(rng, van, van.ref_pos, van.ref_axis, REF_EPC, drift, (sp, sa))
    for pos, axis, epc in tags:
        rows += reads_for(rng, van, pos, axis, epc, drift, (sp, sa))
    return rows


def write_session(name, cfg, plan, van, rng, mode):
    d = os.path.join(ROOT, "sessions", name)
    shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
    json.dump(cfg, open(os.path.join(d, "shelf_config.json"), "w"), indent=2)
    L, D = cfg["zone"]["length_in"], cfg["zone"]["depth_in"]
    with open(os.path.join(d, "manifest.csv"), "w") as man:
        man.write("id,mode,x_in,y_in,z_in,orient,quadrant,epc,file,n_reads,duration_s,tx_power_dbm,started,notes\n")
        for pid, x, y, z, o in plan:
            pos = np.array([x, y, z]) * IN
            rows = capture(rng, van, [(pos, tag_axis(rng, o), TEST_EPC)])
            f = f"cap_{pid}.csv"
            with open(os.path.join(d, f), "w") as fh:
                fh.write("timestamp_ms,epc,phase_deg,freq_mhz,rssi_dbm,antenna\n" + "\n".join(rows) + "\n")
            q = ("F" if y < D / 2 else "B") + ("L" if x < L / 2 else "R")
            man.write(f"{pid},{mode},{x},{y},{z},{o},{q},{TEST_EPC},{f},{len(rows)},25.0,20,2026-11-09T00:00:00,fake\n")
    print(f"  {name}: {len(plan)} captures")


def grid_plan(cfg, heights=(4, 12), orients=("top", "side")):
    L, D = cfg["zone"]["length_in"], cfg["zone"]["depth_in"]
    return [(f"G-{o}-z{z}-x{(ix + .5) * L / 8:04.1f}-y{(iy + .5) * D / 6:04.1f}".replace(".", "p"),
             (ix + .5) * L / 8, (iy + .5) * D / 6, z, o)
            for o in orients for z in heights for iy in range(6) for ix in range(8)]


def random_plan(cfg, n, seed, heights=(4, 12), orients=("top", "side")):
    L, D = cfg["zone"]["length_in"], cfg["zone"]["depth_in"]
    r = np.random.default_rng(seed)
    return [(f"R{seed}-{k:03d}", round((1 + r.random() * (L - 2)) * 2) / 2, round((1 + r.random() * (D - 2)) * 2) / 2,
             int(r.choice(heights)), str(r.choice(orients))) for k in range(n)]


if __name__ == "__main__":
    cfg = cfg_for_sessions()
    rng = np.random.default_rng(7)
    van1, van2 = Van(rng, cfg), Van(rng, cfg)
    print(f"severity: {SEV}")
    write_session("fake_grid", cfg, grid_plan(cfg), van1, rng, "survey")
    write_session("fake_random", cfg, random_plan(cfg, 150, 2), van1, rng, "random")
    write_session("fake_van2_random", cfg, random_plan(cfg, 150, 3), van2, rng, "random")
    # scan: 6 boxes on the shelf at once, positions unknown to the analyzer
    d = os.path.join(ROOT, "sessions", "fake_scan"); shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
    json.dump(cfg, open(os.path.join(d, "shelf_config.json"), "w"), indent=2)
    truth = [(6, 4, 4), (25, 3, 8), (10, 19, 12), (27, 20, 4), (15, 9, 6), (18, 15, 10)]
    tags = [(np.array(p) * IN, tag_axis(rng, "top"), f"E28011902000000000BOX{k:03d}") for k, p in enumerate(truth)]
    rows = capture(rng, van1, tags)
    with open(os.path.join(d, "cap_scan_1.csv"), "w") as fh:
        fh.write("timestamp_ms,epc,phase_deg,freq_mhz,rssi_dbm,antenna\n" + "\n".join(rows) + "\n")
    with open(os.path.join(d, "manifest.csv"), "w") as man:
        man.write("id,mode,x_in,y_in,z_in,orient,quadrant,epc,file,n_reads,duration_s,tx_power_dbm,started,notes\n")
        man.write(f"scan_1,scan,,,,,,,cap_scan_1.csv,{len(rows)},25.0,20,2026-11-09T00:00:00,fake\n")
    with open(os.path.join(d, "TRUTH.txt"), "w") as fh:
        for k, p in enumerate(truth):
            q = ("F" if p[1] < 12 else "B") + ("L" if p[0] < 16 else "R")
            fh.write(f"E28011902000000000BOX{k:03d}  x={p[0]} y={p[1]} z={p[2]}  {q}\n")
    print("  fake_scan: 6 boxes in one capture (truth in TRUTH.txt)")
