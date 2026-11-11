"""
Analyze a shelf survey (known positions) captured with capture/ShelfCapture.

  python analysis/survey_analyze.py sessions/van1_grid
  python analysis/survey_analyze.py sessions/van1_grid --test sessions/van1_random sessions/van2_random

Zones come from the session's shelf_config.json: either 2x2 quadrants (default)
or side-by-side zones ("zones": {"axis": "x", "edges_in": [...], "names": [...]}).

Without --test: grouped cross-validation (folds split by x,y spot so the same
spot never appears in train and test).  With --test: train on the first session,
test on the others (= the real deployment question).
If the session holds captures at several Tx powers (--powers in the capture
tool), every power is evaluated separately.

Methods
  strongest antenna       no training: zone of the antenna with the highest RSSI
  fingerprint rssi        kNN on relative RSSI per port
  fingerprint rssi+slope  + phase-slope range per port
  pattern phase           kNN on the per-channel phase pattern per port
  pattern rssi+phase      pattern + RSSI
  model ...               physics model (down-facing patch; quadrant layout only)

Prints + writes <session>/analysis/:
  report.txt, features.csv, predictions.csv, rssi_maps.png, error_map[_p<power>].png
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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIM_SIGMAS = os.path.join(ROOT, "sim", "results", "sigmas.json")
COLS = ["acc", "acc_3in_from_edge", "acc_when_conf90", "coverage_conf90", "top2_acc",
        "within_1_zone", "p_err_lt_12in", "p_err_lt_18in"]


class Tee:
    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8")

    def __call__(self, *a):
        s = " ".join(str(x) for x in a)
        print(s); self.f.write(s + "\n")


def prepare(sdir, fallback_cable=None, need_xy=True):
    """-> cfg, manifest, features (with calibrated 'range'), cable dict, cable source dict."""
    cfg, man = sl.load_session(sdir)
    feat = sl.session_features(sdir, man, cfg)
    if feat.empty:
        raise SystemExit(f"no usable captures in {sdir}")
    if need_xy:
        feat = feat.dropna(subset=["x_in", "y_in"])
    cable, src = sl.session_cable(feat, cfg, fallback_cable)
    return cfg, man, sl.apply_cable(feat, cable), cable, src


def truth(feat, cfg, ids):
    first = feat.groupby("id", sort=False).first().loc[ids]
    xy = first[["x_in", "y_in"]].values.astype(float)
    q = np.array([sl.quadrant(cfg, x * sl.IN, y * sl.IN) for x, y in xy])
    return xy, q, sl.boundary_dist_in(cfg, xy[:, 0], xy[:, 1])


def eval_split(train, test, cfg, ports, tag, sigs=None):
    """Returns list of (method, metrics) and predictions frame."""
    out, preds = [], []
    test = test[test.port.isin(ports)]
    train = train[train.port.isin(ports)]
    ids = list(dict.fromkeys(test.id))
    xy, q, bd = truth(test, cfg, ids)
    names = sl.zone_names(cfg)
    linear = cfg["zone_axis"] != "quad"
    runs = [("strongest antenna", lambda: sl.StrongestAntenna(cfg, ports).predict(test)),
            ("fingerprint rssi", lambda: sl.Fingerprint(ports, use_phase=False).fit(train, cfg).predict(test)),
            ("fingerprint rssi+slope", lambda: sl.Fingerprint(ports, use_phase=True).fit(train, cfg).predict(test))]
    if sigs is not None:
        runs += [(name, (lambda ur: lambda: sl.PatternFingerprint(ports, use_rssi=ur).fit(train, sigs, cfg)
                         .predict(test, sigs))(ur))
                 for ur, name in [(False, "pattern phase"), (True, "pattern rssi+phase")]]
    for name, run in runs:
        pid, P, exy = run()
        assert pid == ids
        out.append((name, sl.score(q, P, exy, xy, bd, linear)))
        preds.append(pd.DataFrame(dict(id=ids, split=tag, method=name, true_q=[names[k] for k in q],
                                       pred_q=[names[k] for k in P.argmax(1)], conf=P.max(1),
                                       second_q=[names[k] for k in np.argsort(P, 1)[:, -2]],
                                       x_true=xy[:, 0], y_true=xy[:, 1], x_est=exy[:, 0], y_est=exy[:, 1])))
    if not linear:   # physics model assumes the down-facing quadrant layout
        loc = sl.ModelLocalizer(dict(cfg, ports=ports), sl.learn_model(train, cfg))
        for up, ur, name in [(False, True, "model rssi"), (True, False, "model phase"), (True, True, "model rssi+phase")]:
            P, E = [], []
            for cid in ids:
                qp, est = loc.locate(test[test.id == cid], use_phase=up, use_rssi=ur)
                P.append(qp); E.append(est[:2] / sl.IN)
            out.append((name, sl.score(q, np.array(P), np.array(E), xy, bd, linear)))
    return out, pd.concat(preds)


def evaluate(feat, tests, cfg, sigs, folds, say):
    """Accuracy tables for one Tx power.  tests = list of (name, feat) or []."""
    ports = cfg["ports"]
    results, preds = {}, []
    port_sets = {f"{len(ports)} ant (all)": ports}
    if len(ports) >= 3:
        for p in ports:
            port_sets[f"{len(ports) - 1} ant (drop port {p})"] = [x for x in ports if x != p]
    if tests:
        test_all = pd.concat([t for _, t in tests], ignore_index=True)
        for name, ps in port_sets.items():
            r, pr = eval_split(feat, test_all, cfg, ps, "test", sigs)
            results[name] = r
            if name.endswith("(all)"):
                preds.append(pr)
    else:
        spots = feat.groupby("id").apply(lambda g: (g.x_in.iloc[0], g.y_in.iloc[0]), include_groups=False)
        uniq = sorted(set(spots))
        fold_of_spot = dict(zip(uniq, np.random.default_rng(0).permutation(len(uniq)) % folds))
        fold = spots.map(fold_of_spot)
        say(f"  {folds}-fold cross-validation grouped by (x,y) spot ({len(uniq)} spots)")
        for name, ps in port_sets.items():
            acc = {}
            for k in range(folds):
                te_ids = set(fold[fold == k].index)
                tr, te = feat[~feat.id.isin(te_ids)], feat[feat.id.isin(te_ids)]
                if te.empty or tr.empty:
                    continue
                r, pr = eval_split(tr, te, cfg, ps, f"fold{k}", sigs)
                for m, s in r:
                    acc.setdefault(m, []).append(s)
                if name.endswith("(all)"):
                    preds.append(pr)
            results[name] = [(m, {c: np.average([d[c] for d in L], weights=[d["n"] for d in L])
                                  for c in L[0] if c != "n"} | {"n": sum(d["n"] for d in L)})
                             for m, L in acc.items()]
    cols = [c for c in COLS if not (c == "within_1_zone" and cfg["zone_axis"] == "quad")]
    for name, r in results.items():
        say(f"\n  {name}")
        say(f"    {'method':24s}" + "".join(f"{c.replace('_', ' ')[:11]:>12}" for c in cols) + f"{'med err in':>11}")
        for m, s in r:
            say(f"    {m:24s}" + "".join(f"{'-':>12}" if np.isnan(s[c]) else f"{100 * s[c]:11.1f}%" for c in cols)
                + f"{s['median_err_in']:10.1f}\"")
    P = pd.concat(preds, ignore_index=True)
    names = sl.zone_names(cfg)
    for m in ["strongest antenna", "fingerprint rssi", "pattern rssi+phase"]:
        g = P[P.method == m]
        if g.empty:
            continue
        cm = pd.crosstab(pd.Categorical(g.true_q, names), pd.Categorical(g.pred_q, names),
                         rownames=["true"], colnames=["predicted"], dropna=False)
        say(f"\n  confusion, {m} (rows = true zone):")
        for line in cm.to_string().splitlines():
            say("    " + line)
    return P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--test", nargs="*", default=[])
    ap.add_argument("--folds", type=int, default=5)
    a = ap.parse_args()

    outdir = os.path.join(a.session, "analysis")
    os.makedirs(outdir, exist_ok=True)
    say = Tee(os.path.join(outdir, "report.txt"))
    cfg, man, feat, cable, cable_src = prepare(a.session)
    ports = cfg["ports"]
    sigs = sl.session_signatures(a.session, man, cfg)
    feat.to_csv(os.path.join(outdir, "features.csv"), index=False)
    n_cap = feat.id.nunique()
    names = sl.zone_names(cfg)
    say(f"Session {a.session}: {len(man)} captures in manifest, {n_cap} with test-tag reads")
    say("Zones: " + (", ".join(f"{n} = x {cfg['zone_edges'][i] / sl.IN:g}-{cfg['zone_edges'][i + 1] / sl.IN:g}\""
                               for i, n in enumerate(names)) if cfg["zone_axis"] != "quad" else "2x2 quadrants")
        + "   antenna -> zone: " + ", ".join(f"p{p}->{names[z]}" for p, z in cfg["port_zone"].items()))

    # ── QC ────────────────────────────────────────────────────────────────
    say("\nPer-port QC (test tag)")
    say(f"  {'power':>5} {'port':>4} {'seen%':>6} {'chan':>5} {'reads':>6} {'rssi':>6} {'phase sd raw':>13} {'sd mod180':>10}")
    for pw, fp in feat.groupby("tx_power_dbm"):
        n = fp.id.nunique()
        for p in ports:
            g = fp[fp.port == p]
            say(f"  {pw:5g} {p:>4} {100 * g.id.nunique() / max(n, 1):6.0f} {g.n_ch.median():5.0f} {g.n_reads.median():6.0f}"
                f" {g.rssi.median():6.1f} {g.phase_sd_raw.median():12.1f}° {g.phase_sd_mod180.median():9.1f}°")
        ns = fp.groupby("id").port.nunique()
        say(f"  {pw:5g} dBm: captures read by >=2 ports {100 * (ns >= 2).mean():.0f}%, by all {len(ports)}: "
            f"{100 * (ns == len(ports)).mean():.0f}%")
    if feat.phase_sd_raw.median() > 2 * feat.phase_sd_mod180.median():
        say("  -> 180° phase ambiguity present (raw >> mod-180 jitter); mod-180 processing is required.")

    # ── reference tag ─────────────────────────────────────────────────────
    say(f"\nReference tag read in {100 * feat.cable_ref.notna().mean():.0f}% of capture-ports")
    for p in ports:
        g = feat[feat.port == p].cable_ref.dropna()
        n = sum(1 for v in sigs.values() if p in v)
        r = sum(1 for v in sigs.values() if p in v and v[p]["ref"] is not None)
        say(f"    port {p}: cable {cable.get(p, np.nan):7.3f} m [{cable_src.get(p, 'NONE')}]"
            f"  ref scatter {100 * g.std() if len(g) > 2 else np.nan:5.0f} cm;  phase pattern ref-corrected {r}/{n}")

    if cfg["zone_axis"] == "quad":
        model = sl.learn_model(feat, cfg)
        say("\nModel error sizes (common offset removed), by read strength weak / mid / strong:")
        say(f"  range : {np.round(100 * model['sig_range'] / 1.2, 1)} cm")
        say(f"  RSSI  : {np.round(model['sig_rssi'] / 1.2, 1)} dB      (RSSI offset P0 = {model['p0']:.1f} dB)")
        if os.path.exists(SIM_SIGMAS):
            sim = json.load(open(SIM_SIGMAS))
            mine = np.r_[model["sig_range"][1:] / 1.2 * 100 / 5, model["sig_rssi"][1:] / 1.2]
            best = min(sim, key=lambda k: np.nansum((np.r_[np.array(sim[k][0][1:]) * 100 / 5,
                                                         sim[k][1][1:]] - mine) ** 2))
            say(f"  closest simulation severity: '{best}'")

    # ── test sessions ─────────────────────────────────────────────────────
    tests = []
    for t in a.test:
        tcfg, tman, tf, _, tsrc = prepare(t, cable)
        sfx = "@" + os.path.basename(os.path.normpath(t))
        sigs.update(sl.session_signatures(t, tman, tcfg, sfx))
        tests.append((t, tf.assign(id=tf.id.astype(str) + sfx)))

    # ── evaluation, per Tx power ──────────────────────────────────────────
    allP = []
    powers = sorted(feat.tx_power_dbm.dropna().unique())
    for pw in powers:
        say(f"\n══ Tx power {pw:g} dBm " + "═" * 60)
        f = feat[feat.tx_power_dbm == pw]
        tt = [(n, t[t.tx_power_dbm == pw]) for n, t in tests]
        tt = [(n, t) for n, t in tt if not t.empty]
        if a.test and not tt:
            say("  (no test captures at this power)"); continue
        if tt:
            say("  train on " + a.session + ", test on " + ", ".join(n for n, _ in tt))
        P = evaluate(f, tt, cfg, sigs, a.folds, say).assign(tx_power_dbm=pw)
        plots_errmap(cfg, P, outdir, f"_p{pw:g}" if len(powers) > 1 else "")
        allP.append(P)
    pd.concat(allP, ignore_index=True).to_csv(os.path.join(outdir, "predictions.csv"), index=False)
    plot_rssi_maps(feat, cfg, outdir)
    say(f"\nWrote {outdir}")


def plots_errmap(cfg, P, outdir, sfx):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    L, D = cfg["L"] / sl.IN, cfg["D"] / sl.IN
    ms = list(dict.fromkeys(P.method))
    fig, axes = plt.subplots(1, len(ms), figsize=(min(3.4 * len(ms) * L / max(D, 1) * 0.6, 30), 3.6))
    for a, m in zip(np.atleast_1d(axes), ms):
        g = P[P.method == m]
        ok = g.true_q == g.pred_q
        a.scatter(g.x_true[ok], g.y_true[ok], s=12, c="#9fd39f")
        a.scatter(g.x_true[~ok], g.y_true[~ok], s=16, c="#d62728")
        for p, xyz in cfg["ants"].items():
            a.plot(xyz[0] / sl.IN, xyz[1] / sl.IN, "ks", ms=6)
            a.annotate(f"P{p}", (xyz[0] / sl.IN, xyz[1] / sl.IN), fontsize=7, xytext=(2, 3), textcoords="offset points")
        if cfg["zone_axis"] == "quad":
            a.axvline(L / 2, color="k", lw=0.6); a.axhline(D / 2, color="k", lw=0.6)
        else:
            for e in cfg["zone_edges"][1:-1] / sl.IN:
                (a.axvline if cfg["zone_axis"] == "x" else a.axhline)(e, color="k", lw=0.6)
        a.set(xlim=(-1, L + 1), ylim=(-1, D + 1), aspect="equal", title=f"{m}\n{100 * ok.mean():.0f}% correct")
        a.tick_params(labelsize=7)
    fig.suptitle("Top view, red = wrong zone (aisle / front = y 0)", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, f"error_map{sfx}.png"), dpi=120); plt.close(fig)


def plot_rssi_maps(feat, cfg, outdir):
    """Median RSSI per port over the surveyed spots, one row per Tx power."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    powers = sorted(feat.tx_power_dbm.dropna().unique())
    ports = cfg["ports"]
    fig, axes = plt.subplots(len(powers), len(ports), figsize=(3.3 * len(ports), 2.4 * len(powers) + 0.5),
                             squeeze=False)
    sc = None
    for i, pw in enumerate(powers):
        f = feat[feat.tx_power_dbm == pw]
        for j, p in enumerate(ports):
            a = axes[i, j]
            g = f[f.port == p]
            sc = a.scatter(g.x_in, g.y_in, c=g.rssi, cmap="viridis", vmin=-75, vmax=-30, s=40)
            xyz = cfg["ants"][p] / sl.IN
            a.plot(xyz[0], xyz[1], "rs", ms=7)
            a.set(title=f"P{p} @ {pw:g} dBm", aspect="equal",
                  xlim=(-1, cfg["L"] / sl.IN + 1), ylim=(-1, cfg["D"] / sl.IN + 1))
            a.tick_params(labelsize=7)
    if sc is not None:
        fig.colorbar(sc, ax=axes, shrink=0.8, label="RSSI (dBm); blank spot = not read")
    fig.savefig(os.path.join(outdir, "rssi_maps.png"), dpi=110); plt.close(fig)


if __name__ == "__main__":
    main()
