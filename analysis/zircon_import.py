"""
Import a Zircon (QM100) RTT log of a one-port-at-a-time sweep into a session folder that
analysis/beam_check.py and beam_map.py can read: <out>/portN/cap_zircon.csv (+ layout.json copied).

  python analysis/zircon_import.py <log.txt> <out_session_dir> <layout_session_dir>
"""
import json, os, re, shutil, sys
import pandas as pd

PAT = re.compile(r"\[\s*([\d\.]+)\s*s\]\s*ANT(\d)\s+EPC\s+([0-9A-Fa-f]{24})\s+RSSI\s+(-?\d+)\s+dBm")
log, out, lay_dir = sys.argv[1:4]
text = open(log, encoding="utf-8", errors="replace").read()
last = text.split("Sweeping ANT1..ANT4")[-1]  # keep only the last sweep (earlier ones were aborted/repeated)
rows = [(float(m.group(1)), int(m.group(2)), m.group(3).upper(), int(m.group(4)))
        for line in last.splitlines() if (m := PAT.search(line))]
d = pd.DataFrame(rows, columns=["t", "ant", "epc", "rssi_dbm"])
summ = re.findall(r"sweep reads A1-A4 = ([\d/]+) \(total (\d+)\)", last)
print(os.path.basename(log), f"{len(d)} reads parsed; log summary: {summ}")
os.makedirs(out, exist_ok=True)
shutil.copy(os.path.join(lay_dir, "layout.json"), os.path.join(out, "layout.json"))
for p, g in d.groupby("ant"):
    os.makedirs(os.path.join(out, f"port{p}"), exist_ok=True)
    g = g.assign(timestamp_ms=(g.t - g.t.min()) * 1000, antenna=p)[["timestamp_ms", "epc", "rssi_dbm", "antenna"]]
    g.to_csv(os.path.join(out, f"port{p}", "cap_zircon.csv"), index=False)
    print(f"  ANT{p}: {len(g)} reads over {g.timestamp_ms.max() / 1000:.1f} s")
