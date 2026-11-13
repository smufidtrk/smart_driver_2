"""
Re-implementation of the antenna engineer's lab processing (Switched_beam_antenna_measurement_wip10.exe),
so it can be run on his log and on our shelf captures the same way.

His tool, as read from the packaged script:
  1. Parse each log line "[ t s] ANTn  EPC <24 hex>  RSSI <dBm> dBm"; keep EPCs starting "1A"; timestamps unused.
  2. Per antenna, keep reads with RSSI >= threshold (slider, default -89.1 dBm) for the selected tags.
  3. Per antenna: total reads, unique tags, mean RSSI (mean of the dBm values).
  4. Each antenna is mapped to a zone (Zone 1..4 = 0-45, 45-90, 90-135, 135-180 deg from the array axis;
     default Zone1=ANT2, Zone2=ANT4, Zone3=ANT3, Zone4=ANT1). A tag read (above threshold) on antennas of
     2+ zones is a "multi-zone" tag. The multi-zone count is the overlap figure on his dashboard.

  python analysis/engineer_method.py <his_log.txt | our session dir with port1..4> [threshold ...]
"""
import glob, json, os, re, sys
import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
PAT = re.compile(r"\[\s*[\d\.]+\s*s\]\s*(ANT\d)\s+EPC\s+([0-9A-Fa-f]{24})\s+RSSI\s+(-?\d+)\s+dBm")


def load(src):
    if os.path.isfile(src):
        rows = []
        for line in open(src, encoding="utf-8", errors="replace"):
            m = PAT.search(line)
            if m:
                rows.append((m.group(1), m.group(2).upper(), float(m.group(3))))
        d = pd.DataFrame(rows, columns=["ant", "epc", "rssi"])
        return d[d.epc.str.startswith("1A")], None
    parts, lay = [], None
    lp = os.path.join(src, "layout.json")
    if os.path.exists(lp):
        lay = {k.upper(): int(v[0]) for k, v in json.load(open(lp))["tags"].items()}
    for p in (1, 2, 3, 4):
        f = sorted(glob.glob(os.path.join(src, f"port{p}", "cap_*.csv")))
        if f:
            x = pd.read_csv(f[-1], dtype={"epc": str})
            parts.append(pd.DataFrame({"ant": f"ANT{p}", "epc": x.epc.str.upper(), "rssi": x.rssi_dbm}))
    d = pd.concat(parts)
    if lay:
        d = d[d.epc.isin(lay)]
    return d, lay


def report(d, thr, lay):
    k = d[d.rssi >= thr]
    print(f"\n-- RSSI >= {thr:g} dBm --")
    for a in ("ANT1", "ANT2", "ANT3", "ANT4"):
        g = k[k.ant == a]
        print(f"  {a}: reads {len(g):5d}  unique tags {g.epc.nunique():3d}  avg RSSI {g.rssi.mean() if len(g) else 0:6.2f}")
    n_ant = k.groupby("epc").ant.nunique()
    seen = len(n_ant)
    print(f"  tags seen {seen}; on exactly 1 antenna {int((n_ant == 1).sum())}; multi-zone (2+ antennas) {int((n_ant >= 2).sum())}"
          f" ({(n_ant >= 2).mean() * 100 if seen else 0:.0f}%)")


if __name__ == "__main__":
    src = sys.argv[1]
    d, lay = load(src)
    print(src, f"{len(d)} reads, {d.epc.nunique()} tags, RSSI {d.rssi.min():g} .. {d.rssi.max():g} dBm")
    for t in [float(a) for a in sys.argv[2:]] or [-89.1]:
        report(d, t, lay)
