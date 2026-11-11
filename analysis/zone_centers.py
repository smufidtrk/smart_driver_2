"""
Zone-center test: one box in the middle of each zone, all captured at once (scan mode).
What does each antenna see for each box -- RSSI and phase -- and is there a pattern
that tells the zones apart?

  python analysis/zone_centers.py sessions/4zone_centers
  python analysis/zone_centers.py sessions/4zone_centers --tags 1A..0261=1 1A..0650=2 1A..0654=3 1A..0661=4 --y 12 --z 4

Rotation test (is the phase pattern a property of the SPOT or of the BOX?):
  python analysis/zone_centers.py sessions/4zone_centers sessions/4zone_rot1 sessions/4zone_rot2 sessions/4zone_rot3 --rotate
  session n = every box moved n zones to the RIGHT (the box in zone 4 wraps to zone 1),
  so every spot is seen with 4 different boxes and every box at 4 spots.

Default tag -> zone mapping (first session): the four test boxes in order, zones 1-4.
Box position = zone center along x, --y (default 12") deep, --z (default 4") high.

Why raw phase values can't be compared box to box: every tag chip adds its own
unknown phase offset, and every port/cable adds its own.  What CAN be compared:
  * phase SLOPE across the 50 channels -> distance (tag offset is ~flat in frequency);
    the reference tag removes each port's cable delay  -> "phase distance" below
  * the per-channel phase PATTERN (match ignores a constant offset, so chip and cable
    offsets cancel): same box between captures, and -- with the rotation test --
    different boxes at the same spot
Off-center test: a session folder may hold layout.json  {"shift": 0, "dx_in": -3, "dy_in": 0}
  (shift = zones the boxes are moved right vs the first session; dx/dy = every box moved
  that far from its zone center: dx + = right, dy + = toward the back wall).  Off-center
  readings are looked up against the CENTERED readings only.
  layout.json may instead list its own boxes: {"tags": {"EPC": ["zone", y_in], ...}} (e.g. 3 boxes
  per zone at different depths); those readings are also looked up against the centered ones,
  and the boxes within each capture are compared with each other.
  python analysis/zone_centers.py sessions/4zone_centers sessions/4zone_rot1 ... sessions/4zone_rot4 \\
         sessions/4zone_offL sessions/4zone_offR sessions/4zone_offA sessions/4zone_offB --rotate

Writes <first session>/analysis/zone_centers.csv, zone_centers_pairs.csv,
zone_centers_offcenter.csv, zone_centers.png
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import shelfloc as sl  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

DEFAULT_TAGS = ["1A1122334455667788990261", "1A1122334455667788990650",
                "1A1122334455667788990654", "1A1122334455667788990661"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sessions", nargs="+")
    ap.add_argument("--rotate", action="store_true", help="session n = boxes shifted n zones right")
    ap.add_argument("--tags", nargs="*", help="EPC=zone ... for the first session (default: the 4 test boxes -> zones 1-4)")
    ap.add_argument("--y", type=float, default=12.0, help="box depth from the aisle edge (in)")
    ap.add_argument("--z", type=float, default=4.0, help="tag height (in)")
    ap.add_argument("--power", type=float, default=15.0, help="power whose tables are printed when there are several sessions")
    ap.add_argument("--phase", choices=["raw", "ref"], default="raw",
                    help="pattern to compare: raw (default; the port's cable offset is common to all boxes, and a weak "
                         "reference tag ADDS its own noise to every box, inflating matches) or ref (divided by the "
                         "reference tag channel by channel; only worth it across days, if the reference reads well)")
    a = ap.parse_args()

    cfg, _ = sl.load_session(a.sessions[0])
    ports, names = cfg["ports"], sl.zone_names(cfg)
    edges = cfg["zone_edges"] / sl.IN
    tz0 = dict(t.split("=") for t in a.tags) if a.tags else {e: names[i] for i, e in enumerate(DEFAULT_TAGS)}
    tz0 = {k.replace(" ", "").upper(): v for k, v in tz0.items()}
    short = {e: e[-4:] for e in tz0}

    def center(z, y=None):
        k = names.index(z)
        return np.array([(edges[k] + edges[k + 1]) / 2, a.y if y is None else y, a.z]) * sl.IN

    caps = []   # (capture id, power, capture, {epc: zone})
    off = {}    # capture id -> (dx, dy) in inches; (0, 0) = boxes at the zone centers
    ydep = {}   # (capture id, epc) -> tag depth from the aisle edge (in)
    custom = set()   # captures whose layout.json lists its own boxes (not the rotation set)
    for n, sd in enumerate(a.sessions):
        _, man = sl.load_session(sd)
        lj = os.path.join(sd, "layout.json")      # optional: {"shift": k, "dx_in": .., "dy_in": ..}
        lay = json.load(open(lj)) if os.path.exists(lj) else {}
        if "tags" in lay:                          # {"EPC": ["zone", y_in], ...}
            tz = {k.upper(): str(v[0]) for k, v in lay["tags"].items()}
            ys = {k.upper(): float(v[1]) for k, v in lay["tags"].items()}
        else:
            shift = int(lay.get("shift", n if a.rotate else 0))
            tz = {e: names[(names.index(z) + shift) % len(names)] for e, z in tz0.items()}
            ys = {e: a.y for e in tz}
        short.update({e: e[-4:] for e in tz})
        stag = os.path.basename(os.path.normpath(sd))
        for r in man.itertuples():
            if os.path.exists(os.path.join(sd, r.file)):
                cid = f"{stag}/{r.id}"
                caps.append((cid, float(r.tx_power_dbm), sl.load_capture(os.path.join(sd, r.file)), tz))
                off[cid] = (float(lay.get("dx_in", 0)), float(lay.get("dy_in", 0)))
                ydep.update({(cid, e): y + off[cid][1] for e, y in ys.items()})
                if "tags" in lay:
                    custom.add(cid)
    show_pw = a.power if a.power in {c[1] for c in caps} else caps[0][1]
    centered = {c for c, o in off.items() if o == (0.0, 0.0) and c not in custom}

    rows, sigs = [], {}
    for cid, pw, cap, tagzone in caps:
        pos = {e: center(z, ydep[(cid, e)]) + np.array([off[cid][0], 0, 0]) * sl.IN for e, z in tagzone.items()}
        f = sl.capture_features(cap, cfg, set(tagzone))
        refp = sorted(f[f.epc == cfg["ref_epc"]].port) if len(f) else []
        print(f"=== {cid}  ({pw:g} dBm)  reference tag "
              + (f"read on ports {','.join(map(str, refp))}" if refp else "NOT read -> no phase-distance calibration")
              + "   boxes: " + "  ".join(f"..{short[e]}->z{z}" for e, z in tagzone.items()))
        for e in tagzone:
            sigs[(cid, e)] = sl.capture_signatures(cap, e, ports, cfg["ref_epc"] if a.phase == "ref" else "")
            for p in ports:
                g = f[(f.epc == e) & (f.port == p)] if len(f) else f
                d = np.linalg.norm(cfg["ants"][p] - pos[e]) / sl.IN
                r = dict(capture=cid, power=pw, tag=short[e], zone=tagzone[e], y=ydep[(cid, e)], port=p,
                         port_zone=names[cfg["port_zone"][p]], true_dist_in=d)
                if len(g):
                    g = g.iloc[0]
                    r.update(reads=g.n_reads, channels=g.n_ch, rssi=g.rssi, phase_sd=g.phase_sd_mod180,
                             phase_range_in=g.range_ref / sl.IN if pd.notna(g.range_ref) else np.nan)
                rows.append(r)
    df = pd.DataFrame(rows)
    for c in ("reads", "channels", "rssi", "phase_sd", "phase_range_in"):
        if c not in df:
            df[c] = np.nan
    df["range_err_in"] = df.phase_range_in - df.true_dist_in

    od = os.path.join(a.sessions[0], "analysis")
    os.makedirs(od, exist_ok=True)
    df.to_csv(os.path.join(od, "zone_centers.csv"), index=False)
    pd.set_option("display.width", 250)

    for (pw, cid), d in df.groupby(["power", "capture"], sort=True):
        if len(a.sessions) > 1 and (pw != show_pw or (cid in centered
                                                      and cid.split("/")[0] != caps[0][0].split("/")[0])):
            continue     # several sessions: tables at one power, for the 1st session and off-center ones
        dx, dy = off[cid]
        print(f"\n────────── {cid}  {pw:g} dBm" + (f"   boxes shifted dx={dx:+g}\" dy={dy:+g}\"" if (dx, dy) != (0, 0) else "")
              + " ──────────")
        for k, lab, fmt in (("reads", "reads", "{:.0f}"), ("rssi", "RSSI (dBm)", "{:.1f}"),
                            ("phase_sd", "phase noise (deg, per channel)", "{:.1f}"),
                            ("true_dist_in", "TRUE distance box -> antenna (in)", "{:.0f}"),
                            ("phase_range_in", "distance from PHASE slope, ref-tag corrected (in)", "{:.0f}")):
            t = d.pivot_table(index=["zone", "y", "tag"], columns="port", values=k, aggfunc="first")
            t.columns = [f"ant{p}(z{names[cfg['port_zone'][p]]})" for p in t.columns]
            print(f"\n  {lab}:  rows = box (zone, depth y in inches from the aisle, EPC end), columns = antenna")
            print("    " + t.to_string(float_format=lambda v: fmt.format(v), na_rep="  -").replace("\n", "\n    "))
        print("\n  Zone call per box:")
        for (z, yy, t), g in d.groupby(["zone", "y", "tag"]):
            g = g.set_index("port")
            by_rssi = names[cfg["port_zone"][g.rssi.idxmax()]] if g.rssi.notna().any() else "-"
            by_ph = names[cfg["port_zone"][g.phase_range_in.idxmin()]] if g.phase_range_in.notna().any() else "-"
            gap = g.rssi.dropna().sort_values()
            margin = gap.iloc[-1] - gap.iloc[-2] if len(gap) >= 2 else np.nan
            print(f"    box ..{t} in zone {z} (y={yy:g}\"):  strongest RSSI -> zone {by_rssi} "
                  f"({'OK' if by_rssi == z else 'WRONG'}, margin {margin:.1f} dB)   "
                  f"shortest phase distance -> zone {by_ph} ({'OK' if by_ph == z else 'WRONG'})")
        e = d.range_err_in.dropna()
        if len(e):
            print(f"  Phase-distance error vs tape measure: median |err| {e.abs().median():.0f}\", worst {e.abs().max():.0f}\"")

    # ── phase-pattern comparisons (match is blind to a constant offset, so each tag's
    #    own chip offset and each port's cable offset cancel) ──────────────────
    all_items = [(cid, pw, e, z) for cid, pw, _, tz in caps for e, z in tz.items()]
    items = [t for t in all_items if t[0] in centered]          # spot comparisons: boxes on the centers only
    order = ["same box, same spot, untouched (other power)", "same box, same spot, put back",
             "SAME SPOT, different box", "same box, different spot", "different box, different spot"]
    pairs = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            c1, p1, e1, z1 = items[i]
            c2, p2, e2, z2 = items[j]
            s1, s2 = c1.split("/")[0], c2.split("/")[0]
            if s1 == s2 and e1 == e2:
                kind = order[0]                          # untouched box, captures at other powers
            elif c1 != c2 and p1 != p2:
                continue                                  # otherwise compare at the same power only
            elif e1 == e2 and z1 == z2:
                kind = order[1]
            elif z1 == z2:
                kind = order[2]
            elif e1 == e2:
                kind = order[3]
            else:
                kind = order[4]
            for p in ports:
                m = sl.sig_match(sigs[(c1, e1)].get(p), sigs[(c2, e2)].get(p))
                if np.isfinite(m):
                    pairs.append(dict(kind=kind, port=p, match=m, power=p1, a=f"..{short[e1]}@z{z1}",
                                      b=f"..{short[e2]}@z{z2}", zone_gap=abs(names.index(z1) - names.index(z2))))
    P = pd.DataFrame(pairs)
    if len(P):
        P.to_csv(os.path.join(od, "zone_centers_pairs.csv"), index=False)
        print("\nPhase-pattern match between box readings, per antenna  (median [min-max]; 1 = identical, ~0.15 = unrelated)")
        for k in order:
            g = P[P.kind == k]
            if len(g):
                print(f"  {k:<46s}" + "  ".join(
                    f"ant{p} {h.match.median():.2f} [{h.match.min():.2f}-{h.match.max():.2f}]"
                    for p, h in g.groupby("port")))
        g = P[(P.kind == order[4]) & (P.power == show_pw)]
        if len(g):
            print(f"\n  different boxes at different spots, {show_pw:g} dBm, by how many zones apart:")
            for zg, h in g.groupby("zone_gap"):
                print(f"    {zg} zone(s) apart: " + "  ".join(f"ant{p} {x.match.median():.2f}" for p, x in h.groupby("port")))
        if (P.kind == order[2]).any():
            print("  -> phase is a property of the SPOT only if 'SAME SPOT, different box' is near the top rows\n"
                  "     and clearly above 'same box, different spot'.")
        else:
            print("  -> with one box per zone each spot is seen with only one box, so 'spot' and 'box' can't be\n"
                  "     separated yet: low matches between boxes may come from the box/tag, not the position.\n"
                  "     The rotation test separates them.")

    # zone call from the phase pattern alone, trained on OTHER sessions (other boxes at each spot)
    if len(a.sessions) > 1:
        print("\nZone call per box reading using the OTHER sessions as the lookup table (same power):")
        hits = {"phase pattern": [], "strongest RSSI": []}
        for cid, pw, e, z in items:
            sess = cid.split("/")[0]
            best, bz = -1.0, None
            for cid2, pw2, e2, z2 in items:
                if cid2.split("/")[0] == sess or pw2 != pw:
                    continue
                m = np.nanmean([sl.sig_match(sigs[(cid, e)].get(p), sigs[(cid2, e2)].get(p)) for p in ports])
                if np.isfinite(m) and m > best:
                    best, bz = m, z2
            if bz is None:
                continue          # no reading at this power in the other sessions
            hits["phase pattern"].append(bz == z)
            g = df[(df.capture == cid) & (df.tag == short[e])].set_index("port").rssi
            hits["strongest RSSI"].append(bool(g.notna().any()) and names[cfg["port_zone"][g.idxmax()]] == z)
        for k, v in hits.items():
            print(f"  {k:<16s} correct {sum(v)}/{len(v)} = {100 * np.mean(v):.0f}%")

    # ── off-center boxes vs a lookup table of the CENTERED readings ─────────
    offc = [t for t in all_items if t[0] not in centered]
    if offc and items:
        print("\nOFF-CENTER boxes, looked up against the centered readings (same power).")
        print("  score(zone) = best mean-over-antennas match to any centered reading in that zone;"
              "  own = match on the zone's own antenna")
        orows = []
        for cid, pw, e, z in offc:
            sc, own = {}, []
            for cid2, pw2, e2, z2 in items:
                if pw2 != pw:
                    continue
                m = np.nanmean([sl.sig_match(sigs[(cid, e)].get(p), sigs[(cid2, e2)].get(p)) for p in ports])
                if np.isfinite(m):
                    sc[z2] = max(sc.get(z2, -1), m)
                if z2 == z:
                    p_own = next(p for p in ports if names[cfg["port_zone"][p]] == z)
                    own.append(sl.sig_match(sigs[(cid, e)].get(p_own), sigs[(cid2, e2)].get(p_own)))
            if not sc:
                continue
            ranked = sorted(sc.items(), key=lambda kv: -kv[1])
            g = df[(df.capture == cid) & (df.tag == short[e])].set_index("port").rssi
            rz = names[cfg["port_zone"][g.idxmax()]] if g.notna().any() else "-"
            other = max((v for k, v in sc.items() if k != z), default=np.nan)
            orows.append(dict(session=cid.split("/")[0], power=pw, dx=off[cid][0], dy=off[cid][1], y=ydep[(cid, e)],
                              box=f"..{short[e]}", zone=z, phase_call=ranked[0][0], rssi_call=rz,
                              score_own_zone=sc.get(z, np.nan), score_best_other=other,
                              own_ant_match=np.nanmax(own) if np.isfinite(own).any() else np.nan))
        O = pd.DataFrame(orows)
        O.to_csv(os.path.join(od, "zone_centers_offcenter.csv"), index=False)
        O["phase_ok"], O["rssi_ok"] = O.phase_call == O.zone, O.rssi_call == O.zone
        for (sname, dx, dy), g in O.groupby(["session", "dx", "dy"], sort=False):
            print(f"\n  {sname}  (dx={dx:+g}\" dy={dy:+g}\")")
            h = g[g.power == show_pw] if (g.power == show_pw).any() else g
            print("    " + h.sort_values(["zone", "y"])[["power", "box", "zone", "y", "phase_call", "rssi_call",
                              "score_own_zone", "score_best_other", "own_ant_match"]]
                  .round(2).to_string(index=False).replace("\n", "\n    "))
        print("\n  Correct zone, all powers:")
        for (s_, dx, dy, yy), g in O.groupby(["session", "dx", "dy", "y"], sort=False):
            print(f"    {s_:<14s} dx={dx:+g}\" dy={dy:+g}\" y={yy:g}\":  phase {g.phase_ok.sum()}/{len(g)}   "
                  f"RSSI {g.rssi_ok.sum()}/{len(g)}   own-antenna match median {g.own_ant_match.median():.2f}"
                  "  (centered, different box: ~0.9)")
        print(f"    TOTAL:           phase {O.phase_ok.sum()}/{len(O)} = {100 * O.phase_ok.mean():.0f}%   "
              f"RSSI {O.rssi_ok.sum()}/{len(O)} = {100 * O.rssi_ok.mean():.0f}%")

    # ── several boxes per zone in ONE capture: do boxes of the same zone look alike? ──
    for cid, pw, _, tz in caps:
        if cid not in custom or pw != show_pw:
            continue
        es = sorted(tz, key=lambda e: (tz[e], ydep[(cid, e)]))
        W = []
        for i in range(len(es)):
            for j in range(i + 1, len(es)):
                e1, e2 = es[i], es[j]
                same_z, same_y = tz[e1] == tz[e2], ydep[(cid, e1)] == ydep[(cid, e2)]
                kind = ("same zone, other depth" if same_z else
                        "other zone, same depth" if same_y else "other zone, other depth")
                for p in ports:
                    W.append(dict(kind=kind, port=p, e1=e1, e2=e2,
                                  match=sl.sig_match(sigs[(cid, e1)].get(p), sigs[(cid, e2)].get(p)),
                                  zone_gap=abs(names.index(tz[e1]) - names.index(tz[e2]))))
        W = pd.DataFrame(W).dropna(subset=["match"])
        if W.empty:
            continue
        print(f"\nWithin {cid} ({pw:g} dBm): phase-pattern match between boxes in the same capture  (median per antenna)")
        for k in ("same zone, other depth", "other zone, same depth", "other zone, other depth"):
            g = W[W.kind == k]
            print(f"  {k:<26s}" + "  ".join(f"ant{p} {h.match.median():.2f}" for p, h in g.groupby("port")))
        g = W[W.kind != "same zone, other depth"]
        print("  other zone, by zones apart:  " + "   ".join(
            f"{zg}: " + " ".join(f"{h.match.median():.2f}" for _, h in x.groupby("port")) for zg, x in g.groupby("zone_gap")))
        # nearest other box in this capture (mean match over antennas): same zone?
        mm = W.groupby(["e1", "e2"]).match.mean()
        ok = []
        for e in es:
            cand = {(e2 if e1 == e else e1): v for (e1, e2), v in mm.items() if e in (e1, e2)}
            nb = max(cand, key=cand.get)
            ok.append(tz[nb] == tz[e])
            print(f"    box ..{short[e]} z{tz[e]} y={ydep[(cid, e)]:g}\": most similar box ..{short[nb]} "
                  f"(z{tz[nb]} y={ydep[(cid, nb)]:g}\", {cand[nb]:.2f})  {'same zone' if ok[-1] else 'OTHER ZONE'}")
        print(f"  most similar box is in the same zone for {sum(ok)}/{len(ok)} boxes")
        W.to_csv(os.path.join(od, f"within_{cid.replace('/', '_')}.csv"), index=False)

    # ── plots ────────────────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d0 = df[df.power == show_pw]
    cid0 = d0.capture.iloc[0]
    tagzone = next(tz for i, _, _, tz in caps if i == cid0)
    pos = {e: center(z) for e, z in tagzone.items()}
    nc = max(3, len(ports))
    fig, ax = plt.subplots(2, nc, figsize=(4.2 * nc, 8))
    d1 = d0[d0.capture == cid0]
    M = d1.pivot_table(index="zone", columns="port", values="rssi", aggfunc="first")
    im = ax[0, 0].imshow(M.values, cmap="viridis", aspect="auto")
    for (i, j), v in np.ndenumerate(M.values):
        ax[0, 0].text(j, i, "-" if np.isnan(v) else f"{v:.0f}", ha="center", va="center", color="w", fontsize=9)
    ax[0, 0].set(xticks=range(M.shape[1]), xticklabels=[f"ant{p}" for p in M.columns], yticks=range(M.shape[0]),
                 yticklabels=[f"box in zone {z}" for z in M.index], title=f"RSSI (dBm), {cid0}")
    fig.colorbar(im, ax=ax[0, 0], shrink=0.8)
    for p in ports:
        g = df[df.port == p]
        ax[0, 1].plot(g.true_dist_in, g.rssi, "o", label=f"ant{p}")
        ax[0, 2].plot(g.true_dist_in, g.phase_range_in, "o", label=f"ant{p}")
    ax[0, 1].set(xlabel="true distance (in)", ylabel="RSSI (dBm)", title="RSSI vs distance (all captures)")
    lim = [0, max(50, np.nanmax(df.true_dist_in) + 5)]
    ax[0, 2].plot(lim, lim, "k:", lw=0.8)
    ax[0, 2].set(xlabel="true distance (in)", ylabel="distance from phase slope (in)",
                 title="Phase ranging (dotted = perfect)")
    for x in ax[0, 1:3]:
        x.grid(alpha=0.3); x.legend(fontsize=8)
    for j in range(3, nc):
        ax[0, j].axis("off")
    for j, p in enumerate(ports):
        x = ax[1, j]
        for e in tagzone:
            s = sigs[(cid0, e)].get(p)
            if not s:
                continue
            v = (s["ref"] if s["ref"] is not None else s["raw"]).sort_index()
            ph = np.degrees(np.unwrap(np.angle(v.values))) / 2
            x.plot(v.index, ph - ph[0], ".-", ms=3, lw=0.8,
                   label=f"box in zone {tagzone[e]} ({np.linalg.norm(cfg['ants'][p] - pos[e]) / sl.IN:.0f}\")")
        x.set(xlabel="channel (MHz)", ylabel="phase (deg, mod 180, unwrapped)")
        x.set_title(f"ant{p} (zone {names[cfg['port_zone'][p]]}): phase vs channel"
                    + ("  [ref-corr.]" if sigs[(cid0, next(iter(tagzone)))].get(p, {}).get("ref") is not None else "  [raw]")
                    + "\nideal: straight, steeper = farther", fontsize=9)
        x.grid(alpha=0.3); x.legend(fontsize=7)
    for j in range(len(ports), nc):
        ax[1, j].axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(od, "zone_centers.png"), dpi=110)
    print(f"\n-> {os.path.join(od, 'zone_centers.csv')}\n-> {os.path.join(od, 'zone_centers.png')}")


if __name__ == "__main__":
    main()
