"""
Neighbour-box test: does a box that does NOT move keep its phase and RSSI while
other boxes are moved around it?  If not, no lookup table or model can rely on it.

  python analysis/stability.py sessions/4zone_disturb
  python analysis/stability.py sessions/4zone_disturb --train sessions/4zone_grid

Session from  capture/run.cmd --mode disturb ...  (test box fixed, a second box moved
to every grid spot outside the test box's zone; baselines with the second box
removed at the start and end).  The fixed position comes from disturb.json.

Also works on any session in which some tag never moved, e.g. the reference tag
during a survey:  --tag 1A00...09 --at 16,24   (manifest x/y = the moving box).

Per capture and port, against the baseline (second box removed):
  match      phase-pattern similarity (1 = unchanged, ~0.15 = unrelated; the
             same-spot repeat in the move test was 0.90-0.95)
  dRSSI      RSSI change (dB)
Zone call of the fixed box per capture: 'strongest antenna' (no training), and with
--train the lookup-table methods (RSSI kNN, phase pattern, phase pattern + RSSI).

Writes <session>/analysis/stability.csv and stability.png
"""
import argparse, json, os, sys
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
    ap.add_argument("session")
    ap.add_argument("--tag", help="EPC of the tag that never moves (default: from disturb.json)")
    ap.add_argument("--at", help="x,y (in) of that tag (default: from disturb.json)")
    ap.add_argument("--train", nargs="*", default=[], help="survey session(s) for the lookup-table zone calls")
    a = ap.parse_args()

    cfg, man = sl.load_session(a.session)
    dj = os.path.join(a.session, "disturb.json")
    info = json.load(open(dj)) if os.path.exists(dj) else {}
    tag = (a.tag or info.get("fixed_epc") or man.epc.iloc[0]).replace(" ", "").upper()
    at = [float(v) for v in a.at.split(",")] if a.at else (
        [info["fixed_x_in"], info["fixed_y_in"]] if info else None)
    ports, names = cfg["ports"], sl.zone_names(cfg)
    caps = {r.id: sl.load_capture(os.path.join(a.session, r.file)) for r in man.itertuples()
            if os.path.exists(os.path.join(a.session, r.file))}
    man = man[man.id.isin(caps)].reset_index(drop=True)
    base_ids = [i for i in man.id if str(i).startswith("D-base")] or [man.id.iloc[0]]
    print(f"Fixed tag {tag}" + (f" at x={at[0]:g}\" y={at[1]:g}\" (zone {names[sl.quadrant(cfg, at[0] * sl.IN, at[1] * sl.IN)]})" if at else "")
          + f";  baseline capture(s): {', '.join(base_ids)};  {len(man)} captures")

    ref = "" if tag == cfg["ref_epc"] else cfg["ref_epc"]   # the reference tag cannot correct itself
    sig = {i: sl.capture_signatures(c, tag, ports, ref) for i, c in caps.items()}
    rss = {i: c[c.epc == tag].groupby("antenna").rssi_dbm.median().to_dict() for i, c in caps.items()}
    b0 = base_ids[0]
    base_rssi = {p: np.nanmean([rss[b].get(p, np.nan) for b in base_ids]) for p in ports}

    rows = []
    for r in man.itertuples():
        i = r.id
        d = np.hypot(r.x_in - at[0], r.y_in - at[1]) if at and pd.notna(r.x_in) else np.nan
        row = dict(id=i, other_x_in=r.x_in, other_y_in=r.y_in, dist_in=d,
                   other_zone=names[sl.quadrant(cfg, r.x_in * sl.IN, r.y_in * sl.IN)] if pd.notna(r.x_in) else "")
        for p in ports:
            row[f"P{p}_match"] = sl.sig_match(sig[b0].get(p), sig[i].get(p))
            row[f"P{p}_dRSSI"] = rss[i].get(p, np.nan) - base_rssi[p]
            row[f"P{p}_rssi"] = rss[i].get(p, np.nan)
        rows.append(row)
    out = pd.DataFrame(rows)

    # ── zone calls for the fixed box ───────────────────────────────────────
    feat = pd.concat([sl.capture_features(caps[i], cfg, {tag}).assign(id=i) for i in man.id], ignore_index=True)
    feat = feat[feat.epc == tag]
    if at:
        feat = feat.assign(x_in=at[0], y_in=at[1], z_in=info.get("fixed_z_in", 4))
    calls = {"strongest antenna": sl.StrongestAntenna(cfg, ports).predict(feat)}
    if a.train:
        tcfg, tman, train, cable, _ = prepare(a.train[0])
        tsig = sl.session_signatures(a.train[0], tman, tcfg)
        for t in a.train[1:]:
            c2, m2, f2, _, _ = prepare(t, cable)
            sfx = "@" + os.path.basename(os.path.normpath(t))
            train = pd.concat([train, f2.assign(id=f2.id.astype(str) + sfx)], ignore_index=True)
            tsig.update(sl.session_signatures(t, m2, c2, sfx))
        feat_c = sl.apply_cable(feat, cable)
        calls["fingerprint rssi"] = sl.Fingerprint(ports, use_phase=False).fit(train, cfg).predict(feat_c)
        calls["pattern phase"] = sl.PatternFingerprint(ports, use_rssi=False).fit(train, tsig, cfg).predict(feat_c, sig)
        calls["pattern rssi+phase"] = sl.PatternFingerprint(ports, use_rssi=True).fit(train, tsig, cfg).predict(feat_c, sig)
    true_z = sl.quadrant(cfg, at[0] * sl.IN, at[1] * sl.IN) if at else None
    for m, (ids, P, _) in calls.items():
        z = dict(zip(ids, P.argmax(1)))
        out[f"call {m}"] = [names[z[i]] if i in z else "" for i in out.id]

    od = os.path.join(a.session, "analysis")
    os.makedirs(od, exist_ok=True)
    out.to_csv(os.path.join(od, "stability.csv"), index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    show = ["id", "other_zone", "dist_in"] + [f"P{p}_{k}" for p in ports for k in ("match", "dRSSI")] \
        + [c for c in out.columns if c.startswith("call ")]
    print(out[show].round(2).to_string(index=False))

    # ── summary ─────────────────────────────────────────────────────────────
    moved = out[~out.id.isin(base_ids)]
    print("\nSummary (other box present vs removed)")
    if len(base_ids) >= 2:
        print("  baseline repeat (other box removed both times, = noise floor): " + ", ".join(
            f"P{p} match {sl.sig_match(sig[base_ids[0]].get(p), sig[base_ids[-1]].get(p)):.2f} "
            f"dRSSI {rss[base_ids[-1]].get(p, np.nan) - rss[base_ids[0]].get(p, np.nan):+.1f}" for p in ports))
    near = moved.dist_in <= 12 if at else pd.Series(False, index=moved.index)
    for p in ports:
        def s(g):
            m, d = g[f"P{p}_match"].dropna(), g[f"P{p}_dRSSI"].dropna().abs()
            return (f"match median {m.median():.2f} worst {m.min():.2f} | |dRSSI| median {d.median():.1f} worst {d.max():.1f} dB"
                    if len(m) else "not read")
        print(f"  P{p}: other box <=12\" away: {s(moved[near]) if near.any() else '-'}")
        print(f"       other box  >12\" away: {s(moved[~near])}")
    for m in calls:
        c = moved[f"call {m}"]
        b = out.loc[out.id == b0, f"call {m}"].iloc[0]
        msg = f"  zone call '{m}': same as baseline in {100 * (c == b).mean():.0f}% of captures"
        if true_z is not None:
            msg += f", correct ({names[true_z]}) in {100 * (c == names[true_z]).mean():.0f}%"
        print(msg)
    print("\nHow to read it: phase is usable for zones only if 'match' stays near the baseline repeat\n"
          "(~0.9) when neighbours move, and the phase-based zone calls stay correct.")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for p in ports:
        ax[0].plot(moved.dist_in, moved[f"P{p}_match"], "o", ms=4, label=f"P{p}")
        ax[1].plot(moved.dist_in, moved[f"P{p}_dRSSI"], "o", ms=4, label=f"P{p}")
    ax[0].axhline(0.9, color="gray", ls=":", lw=0.8); ax[0].axhline(0.15, color="red", ls=":", lw=0.8)
    ax[0].set(xlabel="distance of the other box (in)", ylabel="phase-pattern match vs baseline", ylim=(0, 1.05),
              title="Fixed box: phase pattern while a neighbour moves\n(gray ≈ same-spot repeat, red ≈ unrelated)")
    ax[1].axhline(0, color="gray", lw=0.8)
    ax[1].set(xlabel="distance of the other box (in)", ylabel="RSSI change (dB)", title="Fixed box: RSSI change")
    for x in ax:
        x.grid(alpha=0.3); x.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(od, "stability.png"), dpi=110)
    print(f"\n-> {os.path.join(od, 'stability.csv')}\n-> {os.path.join(od, 'stability.png')}")


if __name__ == "__main__":
    main()
