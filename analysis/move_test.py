"""
Move test: how fast does a tag's phase-vs-channel signature change with position?

  python analysis/move_test.py sessions/2ant_move

Uses a session captured with  capture/run.cmd --mode move ...  (base spot, steps
left and toward the front, base spot again).

Per capture and port, the test tag's signature is its per-channel phase (doubled,
so 180-deg flips don't matter), divided channel by channel by the reference tag's
phase from the same capture and port.  That cancels the reader/cable phase on
every channel and any drift between captures.  Each capture is compared with the
first base capture:

  match     |mean(s_a * conj(s_b))| over the common channels, 0..1.
            1 = identical pattern (up to one constant offset), ~0.15 = unrelated.
  med_dev   median per-channel phase difference after removing that offset (deg, mod 180)

base-b vs base-a is the noise floor (same spot, box picked up and put back).
The distance at which 'match' falls to about halfway between the noise floor and the
unrelated level is roughly how dense a survey grid needs to be.

Writes <session>/analysis/move_test.csv and move_test.png.
"""
import argparse, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import shelfloc as sl  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def chan_vec(cap, epc, port):
    g = cap[(cap.epc == epc) & (cap.antenna == port)]
    if g.empty:
        return pd.Series(dtype=complex)
    z = g.groupby("freq_mhz").phase_deg.apply(lambda a: np.mean(np.exp(2j * np.radians(a))))
    return z / np.abs(z).clip(1e-9)


def compare(sa, sb):
    j = sa.index.intersection(sb.index)
    if len(j) < 5:
        return np.nan, np.nan, len(j)
    p = sa[j].values * np.conj(sb[j].values)
    m = np.mean(p)
    dev = np.degrees(np.angle(p * np.conj(m) / abs(m))) / 2
    return float(abs(m)), float(np.median(np.abs(dev))), len(j)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--no-ref", action="store_true", help="don't divide by the reference tag")
    a = ap.parse_args()

    cfg, man = sl.load_session(a.session)
    ports, ref = cfg["ports"], cfg["ref_epc"]
    caps = {r.id: sl.load_capture(os.path.join(a.session, r.file)) for r in man.itertuples()}
    use_ref = bool(ref) and not a.no_ref
    if use_ref:
        missing = [(i, p) for i, c in caps.items() for p in ports if chan_vec(c, ref, p).size < 20]
        if missing:
            print("⚠ reference tag weak/missing (capture, port):", missing,
                  "\n  those rows fall back to the un-referenced signature\n")

    def sig(i, epc, port):
        s = chan_vec(caps[i], epc, port)
        if use_ref:
            r = chan_vec(caps[i], ref, port)
            if r.size >= 20:
                j = s.index.intersection(r.index)
                return s[j] * np.conj(r[j])
        return s

    base = man.id.iloc[0]
    bx, by = man.x_in.iloc[0], man.y_in.iloc[0]
    rows = []
    for r in man.itertuples():
        epc = r.epc
        dist = float(np.hypot(r.x_in - bx, r.y_in - by)) if pd.notna(r.x_in) else np.nan
        row = dict(id=r.id, x_in=r.x_in, y_in=r.y_in, dist_in=dist,
                   direction="base" if r.id.startswith("M-base") else ("left" if "-x-" in r.id else "front"))
        for p in ports:
            m, d, n = compare(sig(base, epc, p), sig(r.id, epc, p))
            g = caps[r.id]
            rs = g[(g.epc == epc) & (g.antenna == p)].rssi_dbm
            row.update({f"P{p}_match": m, f"P{p}_med_dev": d, f"P{p}_ch": n,
                        f"P{p}_rssi": float(rs.median()) if len(rs) else np.nan})
        if len(ports) >= 2:
            row["rssi_P1_minus_P2"] = row.get("P1_rssi", np.nan) - row.get("P2_rssi", np.nan)
        rows.append(row)
    out = pd.DataFrame(rows)

    od = os.path.join(a.session, "analysis")
    os.makedirs(od, exist_ok=True)
    out.to_csv(os.path.join(od, "move_test.csv"), index=False)
    pd.set_option("display.width", 220)
    print(f"signature = test tag per-channel phase{' ÷ reference tag' if use_ref else ''}; compared with {base}\n")
    print(out.round(2).to_string(index=False))

    # port mapping check: moving LEFT should raise the left antenna's RSSI
    if {"P1_rssi", "P2_rssi"} <= set(out.columns):
        lf = out[out.direction.isin(["base", "left"])].sort_values("dist_in")
        if len(lf) >= 3:
            trend = np.polyfit(lf.dist_in, lf.rssi_P1_minus_P2, 1)[0]
            print(f"\nport check: moving left changes P1-P2 RSSI by {trend:+.2f} dB/inch "
                  f"-> {'port 1 looks like the LEFT antenna' if trend > 0 else 'port 1 looks like the RIGHT antenna'}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for p in ports:
        for dname, mk in [("left", "o-"), ("front", "s--")]:
            d = out[(out.direction == dname) | (out.id == base)].sort_values("dist_in")
            ax[0].plot(d.dist_in, d[f"P{p}_match"], mk, label=f"P{p} {dname}")
            ax[1].plot(d.dist_in, d[f"P{p}_rssi"], mk, label=f"P{p} {dname}")
        b2 = out[out.id == "M-base-b"]
        if len(b2):
            ax[0].axhline(float(b2[f"P{p}_match"].iloc[0]), color="gray", lw=0.8, ls=":")
    ax[0].axhline(0.15, color="red", lw=0.8, ls=":")
    ax[0].set(xlabel="distance from base (in)", ylabel="signature match (1 = same)", ylim=(0, 1.05),
              title="phase signature vs distance\n(gray = base re-placed, red ≈ unrelated)")
    ax[1].set(xlabel="distance from base (in)", ylabel="median RSSI (dBm)", title="RSSI vs distance")
    for x in ax:
        x.legend(fontsize=8)
        x.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(od, "move_test.png"), dpi=90)
    print(f"\n-> {os.path.join(od, 'move_test.csv')}\n-> {os.path.join(od, 'move_test.png')}")


if __name__ == "__main__":
    main()
