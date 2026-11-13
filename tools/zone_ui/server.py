"""
Zone locator UI (local).  From the project folder:

    .venv\\Scripts\\python tools\\zone_ui\\server.py            then open http://localhost:8765

Live scans run capture/run.cmd (the R700 accepts one connection at a time: close ItemTest
first).  Each live scan is saved as sessions/ui_runs/<time>/ and can be re-used.
Test results you log go to tools/zone_ui/test_log.csv.
"""
import csv, datetime, json, os, subprocess, sys, threading, traceback
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import locator as L  # noqa: E402

PORT = int(os.environ.get("ZONE_UI_PORT", "8765"))
LOG = os.path.join(HERE, "test_log.csv")
LOG_FIELDS = ["time", "epc", "true_zone", "call", "conf", "correct", "strongest", "strongest_correct",
              "source", "power", "lookup", "weighting", "p1", "p2", "p3", "p4", "note"]
_scan_lock = threading.Lock()


def known_tags():
    tags = {}
    for s in sorted(L.lookup_sessions(), key=lambda s: s["started"], reverse=True):   # newest layout wins
        lay = json.load(open(os.path.join(L.SESS, s["name"], "layout.json")))["tags"]
        for e, v in lay.items():
            tags.setdefault(e.upper(), dict(box=e[-3:], zone=int(v[0])))
    return tags


def live_scan(rf_mode, power, dwell, gen2_session=2, search="dual"):
    gen2_session = int(gen2_session)
    if gen2_session not in (0, 1, 2, 3):
        raise ValueError("Gen2 session must be 0, 1, 2 or 3.")
    if search not in ("dual", "single"):
        raise ValueError("Search mode must be dual or single.")
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    sess = f"ui_runs/{stamp}"
    cmd = ["cmd", "/c", os.path.join(L.ROOT, "capture", "run.cmd"), "--mode", "scan",
           "--rf-mode", str(int(rf_mode)), "--power", f"{float(power):g}", "--dwell", f"{float(dwell):g}",
           "--gen2-session", str(gen2_session), "--search", search,
           "--config", r"..\shelf_config_4zone.json", "--out", r"..\sessions\ui_runs", "--session", stamp]
    p = subprocess.run(cmd, cwd=L.ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace",
                       timeout=float(dwell) + 180, stdin=subprocess.DEVNULL)
    out = (p.stdout or "") + (p.stderr or "")
    ok = os.path.exists(os.path.join(L.SESS, "ui_runs", stamp, "manifest.csv"))
    if not ok or "Reader error" in out:
        tail = [l for l in out.splitlines() if l.strip()][-4:]
        raise RuntimeError("Scan failed: " + " | ".join(tail))
    return sess


def read_log():
    if not os.path.exists(LOG):
        return []
    with open(LOG, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def log_summary(rows):
    n = len(rows)
    if not n:
        return dict(n=0)
    c = sum(r["correct"] == "1" for r in rows)
    s = sum(r["strongest_correct"] == "1" for r in rows)
    bins = []
    for lo, hi in ((0, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)):
        b = [r for r in rows if lo <= float(r["conf"] or 0) < hi]
        if b:
            bins.append(dict(range=f"{int(lo * 100)}–{min(int(hi * 100), 100)}%", n=len(b),
                             correct=sum(r["correct"] == "1" for r in b)))
    conf = {}
    for r in rows:
        k = f"{r['true_zone']}->{r['call']}"
        conf[k] = conf.get(k, 0) + 1
    return dict(n=n, correct=c, strongest_correct=s, bins=bins, confusion=conf)


# The page can also be opened straight from disk (file://); its requests carry Origin "null".
# Only that origin is let in, so other web sites can't drive the reader through this server.
ALLOWED_ORIGINS = {"null"}


def _local(origin):
    return origin.startswith(("http://localhost:", "http://127.0.0.1:"))


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _cors(self):
        o = self.headers.get("Origin")
        if o in ALLOWED_ORIGINS:
            self.send_header("Access-Control-Allow-Origin", o)
            self.send_header("Vary", "Origin")

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.send_header("Access-Control-Allow-Methods", "GET, POST")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self._cors()
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    def do_GET(self):
        try:
            if self.path in ("/", "/index.html"):
                return self._send(200, open(os.path.join(HERE, "index.html"), "rb").read(), "text/html; charset=utf-8")
            if self.path == "/api/meta":
                return self._send(200, dict(lookup=L.lookup_sessions(), captures=L.capture_sessions(),
                                            tags=known_tags(), zones=sorted(L.PZ.values())))
            if self.path == "/api/log":
                rows = read_log()
                return self._send(200, dict(rows=rows[-200:], summary=log_summary(rows)))
            self._send(404, dict(error="not found"))
        except Exception as e:
            traceback.print_exc()
            self._send(500, dict(error=str(e)))

    def do_POST(self):
        try:
            o = self.headers.get("Origin")
            if o and o not in ALLOWED_ORIGINS and not _local(o):
                return self._send(403, dict(error="Requests from other web sites are not accepted."))
            b = self._body()
            if self.path == "/api/locate":
                epc = (b.get("epc") or "").replace(" ", "").upper()
                if not epc:
                    return self._send(400, dict(error="Enter an EPC."))
                lookup, power, wt = b.get("lookup") or [], float(b.get("power", 20)), b.get("weighting", "equal")
                if not lookup:
                    return self._send(400, dict(error="Pick at least one lookup session."))
                session = b.get("session")
                if b.get("source") == "live":
                    if not _scan_lock.acquire(blocking=False):
                        return self._send(409, dict(error="A scan is already running."))
                    try:
                        session = live_scan(b.get("rf_mode", 2), power, b.get("dwell", 20),
                                            b.get("gen2_session", 2), b.get("search", "dual"))
                    finally:
                        _scan_lock.release()
                res = L.classify_from_session(session, epc, power, lookup, wt)
                res["session"] = session
                return self._send(200 if "error" not in res else 422, res)
            if self.path == "/api/evaluate":
                return self._send(200, L.evaluate(b["session"], float(b.get("power", 20)), b.get("lookup") or [],
                                                  b.get("weighting", "equal")))
            if self.path == "/api/log":
                r = b.get("result") or {}
                tz = int(b["true_zone"])
                comb = r.get("combined") or [None] * 4
                row = dict(time=datetime.datetime.now().isoformat(timespec="seconds"), epc=b.get("epc", ""),
                           true_zone=tz, call=r.get("call"), conf=r.get("call_conf"),
                           correct=int(r.get("call") == tz), strongest=r.get("strongest"),
                           strongest_correct=int(r.get("strongest") == tz), source=r.get("source", ""),
                           power=b.get("power"), lookup="+".join(b.get("lookup") or []), weighting=b.get("weighting"),
                           p1=comb[0], p2=comb[1], p3=comb[2], p4=comb[3], note=(b.get("note") or "").replace(",", ";"))
                new = not os.path.exists(LOG)
                with open(LOG, "a", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
                    if new:
                        w.writeheader()
                    w.writerow(row)
                rows = read_log()
                return self._send(200, dict(rows=rows[-200:], summary=log_summary(rows)))
            self._send(404, dict(error="not found"))
        except Exception as e:
            traceback.print_exc()
            self._send(500, dict(error=str(e)))


if __name__ == "__main__":
    print(f"Zone locator: http://localhost:{PORT}   (Ctrl+C to stop)")
    ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
