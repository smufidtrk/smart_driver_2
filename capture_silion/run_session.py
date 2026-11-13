"""
Run an sdcap capture on the Sapphire board and save it as a normal session folder, so the
analysis scripts and the Zone Locator can use it like an R700 session.

  .venv\\Scripts\\python capture_silion\\run_session.py --session silion_cal32 --secs 120 \\
        --power 12,12,14,14 --mode 4141 --gen2 0 --target 0 --layout 4zone_cal32

Stops trk-app-spsf, runs sdcap.sh, and ALWAYS starts trk-app-spsf again.  Writes
sessions/<session>/: cap_scan_<stamp>-p<label>.csv, manifest.csv, run_<stamp>.json,
shelf_config.json (copied from --layout), layout.json (copied from --layout, if given).

Phase sign: the Silion's phase turns the opposite way to the R700's across frequency
(its slopes come out positive).  phase_deg is stored as 360 - phase so slopes, the phase
pattern and the locator's slope score work the same for both readers; the reader's own
values are kept in silion_start_phase_deg / end_phase_deg.
The power label in the file name (-p<label>) is the first antenna's power: the locator matches
lookup and test captures by it, so use the same --power for the lookup and the tests.
"""
import argparse, datetime, json, os, shutil, subprocess, sys
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOST = "root@172.16.14.3"
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", HOST]


def run(cmd, timeout):
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, encoding="utf-8", errors="replace")
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--secs", type=int, default=120)
    ap.add_argument("--power", default="12,12,14,14", help="dBm, one value or one per antenna")
    ap.add_argument("--ants", default="1,2,3,4")
    ap.add_argument("--mode", type=int, default=4141)
    ap.add_argument("--gen2", type=int, default=0)
    ap.add_argument("--target", type=int, default=0)
    ap.add_argument("--layout", help="session whose layout.json (box positions) applies")
    ap.add_argument("--note", default="")
    a = ap.parse_args()

    sd = os.path.join(ROOT, "sessions", a.session)
    os.makedirs(sd, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    remote = f"/tmp/sdcap_{stamp}.csv"
    script = (
        "cd /home/root/sdcap; systemctl stop trk-app-spsf.service; sleep 3; "
        f"timeout -s INT {a.secs + 60} ./sdcap.sh -a {a.ants} -w {a.power} -t {a.secs} -m {a.mode} "
        f"-s {a.gen2} -g {a.target} -o {remote}; rc=$?; sleep 1; "
        "systemctl start trk-app-spsf.service; sleep 4; "
        "echo APP=$(systemctl is-active trk-app-spsf.service); exit $rc")
    started = datetime.datetime.now().isoformat(timespec="seconds")
    print(f"capturing {a.secs} s on the board ...", flush=True)
    rc, out = run(SSH + [script], a.secs + 180)
    print(out.strip())
    if "APP=active" not in out:
        print("WARNING: trk-app-spsf did not report active after the capture; check the board.")
    label = a.power.split(",")[0]
    local = os.path.join(sd, f"cap_scan_{stamp}-p{label}.csv")
    rc2, out2 = run(["scp", "-q", "-o", "BatchMode=yes", f"{HOST}:{remote}", local], 120)
    if rc2 != 0:
        sys.exit("copy failed: " + out2)

    d = pd.read_csv(local, dtype={"epc": str})
    d["silion_start_phase_deg"] = d.phase_deg
    d["phase_deg"] = (360.0 - d.phase_deg) % 360.0            # R700 sign convention
    d.to_csv(local, index=False)
    print(f"{len(d)} reads, {d.epc.nunique()} tags; per antenna {d.antenna.value_counts().sort_index().to_dict()}")

    cfg_src = os.path.join(ROOT, "sessions", a.layout or "4zone_cal32")
    shutil.copy(os.path.join(cfg_src, "shelf_config.json"), os.path.join(sd, "shelf_config.json"))
    if a.layout:
        shutil.copy(os.path.join(cfg_src, "layout.json"), os.path.join(sd, "layout.json"))
    man = os.path.join(sd, "manifest.csv")
    row = pd.DataFrame([dict(id=f"scan_{stamp}-p{label}", mode="scan", x_in="", y_in="", z_in="", orient="", quadrant="",
                             epc="", file=os.path.basename(local), n_reads=len(d), duration_s=a.secs,
                             tx_power_dbm=float(label), started=started,
                             notes=f"Silion E710 sdcap; power per antenna {a.power} dBm; {a.note}".strip())])
    (pd.concat([pd.read_csv(man), row]) if os.path.exists(man) else row).to_csv(man, index=False)
    json.dump(dict(mode="scan", reader="Silion E710 (Sapphire, sdcap)", epc="", ref_epc="1A0000000000000000000009",
                   tx_power_dbm=[float(label)], power_per_antenna_dbm=[float(x) for x in a.power.split(",")],
                   RfMode=a.mode, Session=a.gen2, search={0: "A", 1: "B", 2: "A->B", 3: "B->A"}.get(a.target, str(a.target)),
                   dwell_s=a.secs, ports=[int(x) for x in a.ants.split(",")], started=started,
                   phase_note="phase_deg = 360 - reader phase (R700 sign convention)"),
              open(os.path.join(sd, f"run_{stamp}.json"), "w"), indent=2)
    print("->", sd)
    sys.exit(rc)


if __name__ == "__main__":
    main()
