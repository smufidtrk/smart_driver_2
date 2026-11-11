using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading;
using Impinj.OctaneSdk;

// ─── Smart Driver — Shelf Capture Tool (4 antennas, stationary tags) ──────────
//
// Derived from Walmart/phase_capture (same reader mode / report fields), but for
// STATIONARY boxes on a shelf: no photo eye, all 4 ports, one capture per box
// placement, every capture tagged with its ground-truth position in a manifest.
//
// Modes
//   survey  (default)  Guided grid: 4x3 spots per quadrant x heights x label
//                      orientations.  Tells you where to put the box, captures,
//                      moves on.  Resumable — rerun with the same --session.
//   random             N random spots (fingerprint TRAIN/TEST sets).
//   scan               Unknown positions: capture every tag for --dwell seconds
//                      once (what production would do).  Feed it to locate.py.
//
// Every capture records ALL tags (incl. the fixed reference tag, which the
// analyzer uses to remove each port's cable phase slope per capture).
//
// Output  sessions/<session>/
//   shelf_config.json        copy of the zone / antenna / reference-tag config
//   manifest.csv             one row per capture (position, file, stats)
//   cap_<id>.csv             timestamp_ms,epc,phase_deg,freq_mhz,rssi_dbm,antenna
//   run_<time>.json          reader settings used for that run
//
// Examples
//   dotnet run -- --epc E28011... --session van1_zoneA
//   dotnet run -- --mode random --random 150 --seed 2 --epc E28011... --session van1_train
//   dotnet run -- --mode scan --session van1_scan --dwell 30
// ─────────────────────────────────────────────────────────────────────────────

namespace ShelfCapture
{
    record Placement(string Id, double X, double Y, double Z, string Orient);

    class Program
    {
        static readonly ImpinjReader _reader = new ImpinjReader();
        static readonly object _lock = new object();
        static List<string> _rows = new List<string>();
        static volatile bool _capturing;
        static DateTime _t0;
        // (epc, port) -> stats for the live display
        static readonly Dictionary<(string, ushort), PortStats> _stats = new();

        class PortStats { public HashSet<double> Ch = new(); public int N; public double RssiSum; }

        static readonly CultureInfo Inv = CultureInfo.InvariantCulture;

        // ── args ────────────────────────────────────────────────────────────────
        static Dictionary<string, string> Args;
        static string Arg(string k, string def) => Args.TryGetValue(k, out var v) ? v : def;

        static int Main(string[] argv)
        {
            Args = ParseArgs(argv);
            if (Args.ContainsKey("help") || Args.ContainsKey("h")) { Usage(); return 0; }

            string mode    = Arg("mode", "survey").ToLowerInvariant();
            string session = Arg("session", DateTime.Now.ToString("yyyyMMdd_HHmm"));
            string outRoot = Arg("out", Path.Combine("..", "sessions"));
            string dir     = Path.Combine(outRoot, session);
            Directory.CreateDirectory(dir);

            // config: the session's own copy wins (resume), else --config / default
            string sessCfg = Path.Combine(dir, "shelf_config.json");
            string cfgPath = File.Exists(sessCfg) ? sessCfg
                           : Arg("config", File.Exists(Path.Combine("..", "shelf_config.json"))
                                           ? Path.Combine("..", "shelf_config.json") : "shelf_config.json");
            var cfg = Config.Load(cfgPath);
            if (!File.Exists(sessCfg)) File.Copy(cfgPath, sessCfg);

            string epc = Arg("epc", "").Replace(" ", "").ToUpperInvariant();
            if (mode != "scan" && epc == "")
            {
                Console.WriteLine("✗ --epc <test tag EPC> is required for survey/random mode."); return 1;
            }
            if (cfg.RefEpc == "")
                Console.WriteLine("⚠  reference_tag.epc is empty in shelf_config.json — the analyzer will have no\n" +
                                  "   per-capture cable calibration.  Strongly recommended: set it.\n");

            // per-run overrides of the config's reader settings
            cfg.RfMode   = int.Parse(Arg("rf-mode", cfg.RfMode.ToString(Inv)), Inv);
            cfg.Session  = int.Parse(Arg("gen2-session", cfg.Session.ToString(Inv)), Inv);
            cfg.Search   = Arg("search", cfg.Search).ToLowerInvariant();
            double dwell = double.Parse(Arg("dwell", "30"), Inv);
            double power = double.Parse(Arg("power", cfg.TxPower.ToString(Inv)), Inv);
            int minCh    = int.Parse(Arg("min-channels", "48"));
            // --powers 12,15,20 : capture every placement once per power (one box placement,
            // several captures) -> the analyzer evaluates each power separately
            double[] powers = Args.ContainsKey("powers") ? ParseList(Arg("powers", "")) : new[] { power };
            string CapId(Placement pl, double pw) => powers.Length > 1 ? $"{pl.Id}-p{pw.ToString("0.#", Inv)}" : pl.Id;

            // disturb mode: the TEST box stays at --at x,y (default: center of the 2nd zone /
            // back-right quadrant); a second box is moved around it, per the grid prompts.
            double[] fixedAt = ParseList(Arg("at", cfg.ZoneEdges != null && cfg.ZoneEdges.Length > 2
                ? $"{((cfg.ZoneEdges[1] + cfg.ZoneEdges[2]) / 2).ToString(Inv)},{(cfg.D / 2).ToString(Inv)}"
                : $"{(cfg.L * 3 / 4).ToString(Inv)},{(cfg.D * 3 / 4).ToString(Inv)}"));

            // ── plan ──────────────────────────────────────────────────────────
            List<Placement> plan = mode switch
            {
                "survey" => GridPlan(cfg, Arg("grid", "8x6"), ParseList(Arg("heights", "4,12")), Arg("orients", "top,side").Split(',')),
                "random" => RandomPlan(cfg, int.Parse(Arg("random", "100")), int.Parse(Arg("seed", "1")),
                                       ParseList(Arg("heights", "4,12")), Arg("orients", "top,side").Split(',')),
                "move"   => MovePlan(cfg, ParseList(Arg("at", $"{cfg.L * 3 / 4},{cfg.D * 3 / 4}")), ParseList(Arg("steps", "1,2,4,8")),
                                     ParseList(Arg("heights", "4"))[0], Arg("orients", "top").Split(',')[0]),
                "disturb" => DisturbPlan(cfg, fixedAt, Arg("grid", "12x3"), ParseList(Arg("heights", "4"))[0],
                                         Arg("orients", "top").Split(',')[0]),
                "scan"   => new List<Placement> { new Placement($"scan_{DateTime.Now:yyyyMMdd_HHmmss}", double.NaN, double.NaN, double.NaN, "") },
                _ => null
            };
            if (plan == null) { Usage(); return 1; }
            if (mode == "disturb")
            {
                string dj = Path.Combine(dir, "disturb.json");
                if (!File.Exists(dj))
                    File.WriteAllText(dj, JsonSerializer.Serialize(new
                    {
                        fixed_x_in = fixedAt[0], fixed_y_in = fixedAt[1], fixed_z_in = ParseList(Arg("heights", "4"))[0],
                        fixed_epc = epc, note = "manifest x/y/z = position of the OTHER (moving) box"
                    }, new JsonSerializerOptions { WriteIndented = true }));
            }

            string manifest = Path.Combine(dir, "manifest.csv");
            var done = File.Exists(manifest)
                ? File.ReadAllLines(manifest).Skip(1).Select(l => l.Split(',')[0]).ToHashSet()
                : new HashSet<string>();
            if (!File.Exists(manifest))
                File.WriteAllText(manifest,
                    "id,mode,x_in,y_in,z_in,orient,quadrant,epc,file,n_reads,duration_s,tx_power_dbm,started,notes\n");

            // ── reader ────────────────────────────────────────────────────────
            Console.WriteLine($"Session : {Path.GetFullPath(dir)}");
            Console.WriteLine($"Mode    : {mode}   plan {plan.Count} placements x {powers.Length} power(s) " +
                              $"({plan.Count(p => powers.All(pw => done.Contains(CapId(p, pw))))} placements already done)");
            Console.WriteLine($"Reader  : {cfg.Ip}  ports {string.Join(",", cfg.Ports)}  {string.Join("/", powers.Select(x => x.ToString(Inv)))} dBm  RfMode {cfg.RfMode}  S{cfg.Session}");
            Console.WriteLine($"Dwell   : {dwell}s per capture (auto-stops early once every port has ≥{minCh} channels)\n");
            try
            {
                Connect(cfg, powers[0]);
            }
            catch (OctaneSdkException e)
            {
                Console.WriteLine($"✗ Reader error: {e.Message}"); return 2;
            }
            Console.CancelKeyPress += (_, e) => { try { _reader.Stop(); _reader.Disconnect(); } catch { } };
            File.WriteAllText(Path.Combine(dir, $"run_{DateTime.Now:yyyyMMdd_HHmmss}.json"),
                JsonSerializer.Serialize(new
                {
                    mode, epc, ref_epc = cfg.RefEpc, tx_power_dbm = powers, cfg.RfMode, cfg.Session, search = cfg.Search,
                    rx_sens_dbm = cfg.RxSens, dwell_s = dwell, ports = cfg.Ports, started = DateTime.Now
                }, new JsonSerializerOptions { WriteIndented = true }));

            // ── loop ──────────────────────────────────────────────────────────
            int i = 0;
            while (i < plan.Count)
            {
                var p = plan[i];
                if (mode != "scan" && powers.All(pw => done.Contains(CapId(p, pw)))) { i++; continue; }

                string note = "";
                if (mode != "scan")
                {
                    Console.WriteLine($"\n[{i + 1}/{plan.Count}] {p.Id}");
                    if (mode == "disturb")
                    {
                        Console.WriteLine($"   TEST box stays at x = {fixedAt[0]:0.#}\"  y = {fixedAt[1]:0.#}\"  ({QLabel(cfg, Quadrant(cfg, fixedAt[0], fixedAt[1]))}) — don't touch it");
                        Console.WriteLine(p.Id.StartsWith("D-base")
                            ? "   OTHER box: REMOVE it from the shelf (baseline capture)"
                            : $"   OTHER box: put it at  x = {p.X,4:0.#}\"  y = {p.Y,4:0.#}\"   → {QLabel(cfg, Quadrant(cfg, p.X, p.Y))}" +
                              $"   ({Math.Sqrt(Math.Pow(p.X - fixedAt[0], 2) + Math.Pow(p.Y - fixedAt[1], 2)):0}\" from the test box)");
                    }
                    else
                    Console.WriteLine($"   Put the test tag at  x = {p.X,4:0.#}\"  y = {p.Y,4:0.#}\"  height = {p.Z,4:0.#}\"" +
                                      $"   label: {p.Orient}   → {QLabel(cfg, Quadrant(cfg, p.X, p.Y))}");
                    if (mode == "move" && p.Id.StartsWith("M-"))
                        Console.WriteLine(p.Id.StartsWith("M-base") ? "   (base spot — mark it on the shelf so you can return to it exactly)"
                                        : $"   (move from the base spot: {(p.Id[2] == 'x' ? "LEFT" : "toward the FRONT")} by {p.Id.Substring(4).Replace("p", ".")}\")");
                    Console.Write("   Enter = capture · s = skip · b = back (redo previous) · q = quit · anything else = note+capture > ");
                    string inp = (Console.ReadLine() ?? "q").Trim();
                    if (inp == "q") break;
                    if (inp == "s") { i++; continue; }
                    if (inp == "b") { i = Math.Max(0, i - 1); foreach (var pw in powers) done.Remove(CapId(plan[i], pw)); continue; }
                    note = inp.Replace(",", ";");
                }

                foreach (var pw in powers)
                {
                    string cid = CapId(p, pw);
                    if (mode != "scan" && done.Contains(cid)) continue;
                    if (powers.Length > 1) { SetPower(cfg, pw); Console.WriteLine($"   -- {pw} dBm"); }
                    var (rows, dur) = Capture(dwell, minCh, epc == "" ? null : epc, cfg);
                    string file = $"cap_{cid}.csv";
                    SafeWrite(Path.Combine(dir, file), "timestamp_ms,epc,phase_deg,freq_mhz,rssi_dbm,antenna\n", rows);
                    File.AppendAllText(manifest, string.Join(",",
                        cid, mode, F(p.X), F(p.Y), F(p.Z), p.Orient,
                        double.IsNaN(p.X) ? "" : Quadrant(cfg, p.X, p.Y),
                        epc, file, rows.Count, dur.ToString("0.0", Inv), pw.ToString(Inv),
                        DateTime.Now.ToString("s"), note) + "\n");
                    done.Add(cid);
                    Report(epc, cfg);
                }
                i++;
            }

            _reader.Disconnect();
            string script = mode switch { "move" => "move_test.py", "scan" => "locate.py", "disturb" => "stability.py", _ => "survey_analyze.py" };
            string extra  = mode == "scan" ? " --train sessions\\<survey session>" : "";
            Console.WriteLine($"\nDone.  Analyze with (from the project folder; uses the project's Python venv):\n" +
                              $"  .venv\\Scripts\\python analysis\\{script} sessions\\{session}{extra}\n");
            return 0;
        }

        // ── reader setup (same mode/report fields as Walmart phase_capture) ────
        static void Connect(Config cfg, double power)
        {
            Console.WriteLine($"Connecting to {cfg.Ip}...");
            _reader.Connect(cfg.Ip);
            _reader.ApplySettings(BuildSettings(cfg, power));
            _reader.TagsReported += OnTagsReported;
            Console.WriteLine("Connected.\n");
        }

        static void SetPower(Config cfg, double power) => _reader.ApplySettings(BuildSettings(cfg, power));

        static Settings BuildSettings(Config cfg, double power)
        {
            Settings s = _reader.QueryDefaultSettings();
            s.Antennas.DisableAll();
            foreach (ushort port in cfg.Ports)
            {
                var a = s.Antennas.GetAntenna(port);
                a.IsEnabled = true;
                a.TxPowerInDbm = power;
                a.RxSensitivityInDbm = cfg.RxSens;
            }
            s.RfMode = (uint)cfg.RfMode;                     // Impinj RF mode ID. 1002 = AutoSet Dense Reader Deep Scan: reads every tag on this shelf, 1-2 deg phase noise. (4120 missed the reference tag and had 6-10 deg phase noise.)
            s.SearchMode = cfg.Search == "single" ? SearchMode.SingleTarget : SearchMode.DualTarget;  // dual: stationary tags keep replying
            s.Session = (ushort)cfg.Session;
            s.TagPopulationEstimate = (ushort)cfg.Population;
            s.Report.IncludePhaseAngle = true;
            s.Report.IncludeChannel = true;
            s.Report.IncludePeakRssi = true;
            s.Report.IncludeAntennaPortNumber = true;
            s.Report.IncludeFirstSeenTime = true;
            s.Report.Mode = ReportMode.Individual;
            return s;
        }

        static void OnTagsReported(ImpinjReader sender, TagReport report)
        {
            if (!_capturing) return;
            lock (_lock)
            {
                foreach (Tag tag in report)
                {
                    if (!tag.IsRfPhaseAnglePresent) continue;
                    double ms = (DateTime.Now - _t0).TotalMilliseconds;
                    double ph = tag.PhaseAngleInRadians * 180.0 / Math.PI;
                    if (ph < 0) ph += 360.0;
                    string epc = tag.Epc.ToHexString().Replace(" ", "").ToUpperInvariant();
                    ushort port = tag.AntennaPortNumber;
                    _rows.Add(string.Format(Inv, "{0:F2},{1},{2:F3},{3:F2},{4:F1},{5}",
                        ms, epc, ph, tag.ChannelInMhz, tag.PeakRssiInDbm, port));
                    if (!_stats.TryGetValue((epc, port), out var st)) _stats[(epc, port)] = st = new PortStats();
                    st.Ch.Add(tag.ChannelInMhz); st.N++; st.RssiSum += tag.PeakRssiInDbm;
                }
            }
        }

        // Runs the reader for up to `dwell` s; Enter stops early.
        static (List<string>, double) Capture(double dwell, int minCh, string epc, Config cfg)
        {
            lock (_lock) { _rows = new List<string>(); _stats.Clear(); }
            _t0 = DateTime.Now;
            _capturing = true;
            _reader.Start();
            Console.WriteLine("   capturing… (Enter = stop early)");
            while (true)
            {
                Thread.Sleep(1000);
                double el = (DateTime.Now - _t0).TotalSeconds;
                Console.Write("\r   " + LiveLine(epc ?? "", cfg) + $"  {el,4:0}s ");
                if (!Console.IsInputRedirected && Console.KeyAvailable && Console.ReadKey(true).Key == ConsoleKey.Enter) break;
                if (el >= dwell) break;
                if (epc != null && AllPortsCovered(epc, cfg, minCh)
                    && (cfg.RefEpc == "" || AllPortsCovered(cfg.RefEpc, cfg, minCh))) break;
            }
            _reader.Stop();
            _capturing = false;
            Console.WriteLine();
            lock (_lock) return (new List<string>(_rows), (DateTime.Now - _t0).TotalSeconds);
        }

        static bool AllPortsCovered(string epc, Config cfg, int minCh)
        {
            lock (_lock)
                return cfg.Ports.All(p => _stats.TryGetValue((epc, p), out var s) && s.Ch.Count >= minCh);
        }

        static string LiveLine(string epc, Config cfg)
        {
            lock (_lock)
            {
                if (epc == "")
                {
                    int tags = _stats.Keys.Select(k => k.Item1).Distinct().Count();
                    return $"{_rows.Count} reads, {tags} tags";
                }
                return string.Join(" | ", cfg.Ports.Select(p =>
                    _stats.TryGetValue((epc, p), out var s)
                        ? $"P{p} {s.Ch.Count,2}ch {s.RssiSum / s.N,5:0.0}dBm"
                        : $"P{p}  --        "));
            }
        }

        static void Report(string epc, Config cfg)
        {
            void One(string label, string e)
            {
                if (e == "") return;
                int seen; int minC = 99;
                lock (_lock)
                {
                    seen = cfg.Ports.Count(p => _stats.ContainsKey((e, p)));
                    foreach (var p in cfg.Ports)
                        if (_stats.TryGetValue((e, p), out var s)) minC = Math.Min(minC, s.Ch.Count);
                }
                string warn = seen < 2 ? "  ⚠ read by <2 antennas — no position info, check power/placement"
                            : minC < 40 ? "  ⚠ some port <40 channels — consider a longer --dwell" : "  ✓";
                Console.WriteLine($"   {label}: {LiveLine(e, cfg)}{warn}");
            }
            One("test tag", epc);
            if (cfg.RefEpc != "")
            {
                bool any; lock (_lock) any = _stats.Keys.Any(k => k.Item1 == cfg.RefEpc);
                if (!any) Console.WriteLine("   ⚠ reference tag NOT read in this capture");
                else One("ref  tag", cfg.RefEpc);
            }
            if (epc == "")
                lock (_lock) Console.WriteLine($"   {_rows.Count} reads, {_stats.Keys.Select(k => k.Item1).Distinct().Count()} tags");
        }

        // ── plans ───────────────────────────────────────────────────────────────
        static List<Placement> GridPlan(Config cfg, string grid, double[] heights, string[] orients)
        {
            // nx x ny spots at cell centers (default 8x6 = 4x3 per quadrant); --grid 12x3 etc.
            var g = grid.ToLowerInvariant().Split('x');
            int nx = int.Parse(g[0]), ny = int.Parse(g[1]);
            var plan = new List<Placement>();
            double dx = cfg.L / nx, dy = cfg.D / ny;
            foreach (var o in orients)
                foreach (var z in heights)
                    for (int iy = 0; iy < ny; iy++)
                        for (int ix = 0; ix < nx; ix++)
                        {
                            int xx = (iy % 2 == 0) ? ix : nx - 1 - ix;  // snake order = less walking
                            double x = dx * (xx + 0.5), y = dy * (iy + 0.5);
                            plan.Add(new Placement($"G-{o}-z{z:0}-x{x:00.0}-y{y:00.0}".Replace(".", "p"), x, y, z, o));
                        }
            return plan;
        }

        static List<Placement> RandomPlan(Config cfg, int n, int seed, double[] heights, string[] orients)
        {
            var rng = new Random(seed);
            var plan = new List<Placement>();
            for (int k = 0; k < n; k++)
            {
                double x = Math.Round((1 + rng.NextDouble() * (cfg.L - 2)) * 2) / 2;
                double y = Math.Round((1 + rng.NextDouble() * (cfg.D - 2)) * 2) / 2;
                plan.Add(new Placement($"R{seed}-{k:000}", x, y, heights[rng.Next(heights.Length)],
                                       orients[rng.Next(orients.Length)]));
            }
            return plan;
        }

        // Move test: base spot, then steps toward -x and toward -y, then base again (drift check).
        // Disturb test: baseline (other box removed), the other box at every grid spot outside
        // the test box's zone (or > 8" away in quadrant mode), nearest first, then baseline again.
        static List<Placement> DisturbPlan(Config cfg, double[] at, string grid, double z, string orient)
        {
            string fz = Quadrant(cfg, at[0], at[1]);
            var spots = GridPlan(cfg, grid, new[] { z }, new[] { orient })
                .Where(s => cfg.ZoneEdges != null ? Quadrant(cfg, s.X, s.Y) != fz
                                                  : Math.Sqrt(Math.Pow(s.X - at[0], 2) + Math.Pow(s.Y - at[1], 2)) > 8)
                .OrderBy(s => Math.Abs(s.X - at[0]) + Math.Abs(s.Y - at[1]))
                .Select(s => s with { Id = "D-" + s.Id.Substring(2) })
                .ToList();
            var plan = new List<Placement> { new Placement("D-base-a", double.NaN, double.NaN, double.NaN, "") };
            plan.AddRange(spots);
            plan.Add(new Placement("D-base-b", double.NaN, double.NaN, double.NaN, ""));
            return plan;
        }

        static List<Placement> MovePlan(Config cfg, double[] at, double[] steps, double z, string orient)
        {
            double x0 = at[0], y0 = at[1];
            var plan = new List<Placement> { new Placement("M-base-a", x0, y0, z, orient) };
            foreach (var d in steps) plan.Add(new Placement($"M-x-{d:0.#}".Replace(".", "p"), x0 - d, y0, z, orient));
            foreach (var d in steps) plan.Add(new Placement($"M-y-{d:0.#}".Replace(".", "p"), x0, y0 - d, z, orient));
            plan.Add(new Placement("M-base-b", x0, y0, z, orient));
            return plan;
        }

        // zone name of a position: side-by-side zones if the config has "zones", else FL/FR/BL/BR
        static string Quadrant(Config cfg, double x, double y)
        {
            if (cfg.ZoneEdges != null)
            {
                double v = cfg.ZoneAxis == "y" ? y : x;
                int k = 0;
                while (k < cfg.ZoneNames.Length - 1 && v >= cfg.ZoneEdges[k + 1]) k++;
                return cfg.ZoneNames[k];
            }
            return (y < cfg.D / 2 ? "F" : "B") + (x < cfg.L / 2 ? "L" : "R");
        }

        static string QLabel(Config cfg, string q) =>
            cfg.ZoneEdges != null ? $"zone {q}"
            : cfg.QLabels.TryGetValue(q, out var n) ? $"quadrant {q} (your Q{n})" : $"quadrant {q}";

        // ── helpers ─────────────────────────────────────────────────────────────
        static string F(double v) => double.IsNaN(v) ? "" : v.ToString("0.##", Inv);
        static double[] ParseList(string s) => s.Split(',').Select(t => double.Parse(t, Inv)).ToArray();

        static Dictionary<string, string> ParseArgs(string[] a)
        {
            var d = new Dictionary<string, string>();
            for (int k = 0; k < a.Length; k++)
                if (a[k].StartsWith("-"))
                {
                    string key = a[k].TrimStart('-');
                    d[key] = (k + 1 < a.Length && !a[k + 1].StartsWith("--")) ? a[++k] : "true";
                }
            return d;
        }

        // Never lose a capture to a file lock (e.g. CSV open in Excel).
        static void SafeWrite(string path, string header, List<string> lines)
        {
            foreach (var f in new[] { path, Path.ChangeExtension(path, $".{DateTime.Now:HHmmss}.csv") })
            {
                try
                {
                    using var sw = new StreamWriter(f, false);
                    sw.Write(header);
                    foreach (var l in lines) sw.WriteLine(l);
                    if (f != path) Console.WriteLine($"   ⚠ {path} locked — saved to {f}");
                    return;
                }
                catch (IOException) { }
            }
            Console.WriteLine($"   ✗ could not write {path} — capture NOT saved");
        }

        static void Usage() => Console.WriteLine(
@"ShelfCapture — 4-antenna stationary-tag capture for the Smart Driver shelf

  dotnet run -- [--mode survey|random|scan] [--session NAME] [--epc TEST_TAG_EPC]
                [--dwell 30] [--power DBM] [--min-channels 48]
                [--rf-mode 1002] [--search dual|single] [--gen2-session 2]
                [--heights 4,12] [--orients top,side] [--random 100 --seed 1]
                [--grid 8x6] [--powers 12,15,20]
                [--config ../shelf_config.json] [--out ../sessions]

  survey : guided 8x6 grid (4x3 per quadrant) x heights x orientations, resumable
  random : N random positions (fingerprint train/test sets)
  move   : signature-vs-distance test: base spot (--at x,y; default center of the
           back-right quadrant), then --steps 1,2,4,8 inches left and toward the
           front, then the base spot again.  Analyze with analysis/move_test.py
  disturb: neighbour-box test.  The TEST box stays at --at x,y (default: middle of zone 2);
           a second box is moved to every --grid spot outside that zone, with baselines
           (second box removed) at the start and end.  Analyze with analysis/stability.py
  scan   : one capture of all tags, positions unknown (use analysis/locate.py)");
    }

    class Config
    {
        public double L, D, H;
        public ushort[] Ports;
        public string RefEpc, Ip;
        public double TxPower, RxSens;
        public int RfMode, Session, Population;
        public Dictionary<string, string> QLabels = new();
        public string Search = "dual";
        public double[] ZoneEdges; public string[] ZoneNames; public string ZoneAxis = "x";

        public static Config Load(string path)
        {
            using var doc = JsonDocument.Parse(File.ReadAllText(path));
            var r = doc.RootElement;
            var z = r.GetProperty("zone");
            var rd = r.GetProperty("reader");
            var c = new Config
            {
                L = z.GetProperty("length_in").GetDouble(),
                D = z.GetProperty("depth_in").GetDouble(),
                H = z.GetProperty("height_in").GetDouble(),
                Ports = r.GetProperty("antennas").EnumerateObject().Select(p => ushort.Parse(p.Name)).OrderBy(p => p).ToArray(),
                RefEpc = r.GetProperty("reference_tag").GetProperty("epc").GetString().Replace(" ", "").ToUpperInvariant(),
                Ip = rd.GetProperty("ip").GetString(),
                TxPower = rd.GetProperty("tx_power_dbm").GetDouble(),
                RxSens = rd.GetProperty("rx_sens_dbm").GetDouble(),
                RfMode = rd.GetProperty("rf_mode").GetInt32(),
                Session = rd.GetProperty("session").GetInt32(),
                Population = rd.GetProperty("population_estimate").GetInt32(),
                QLabels = r.TryGetProperty("quadrant_labels", out var ql)
                    ? ql.EnumerateObject().ToDictionary(p => p.Name, p => p.Value.GetString()) : new(),
            };
            if (r.TryGetProperty("zones", out var zs))
            {
                c.ZoneEdges = zs.GetProperty("edges_in").EnumerateArray().Select(e => e.GetDouble()).ToArray();
                c.ZoneNames = zs.TryGetProperty("names", out var nm)
                    ? nm.EnumerateArray().Select(e => e.GetString()).ToArray()
                    : Enumerable.Range(1, c.ZoneEdges.Length - 1).Select(k => k.ToString()).ToArray();
                if (zs.TryGetProperty("axis", out var ax)) c.ZoneAxis = ax.GetString();
            }
            return c;
        }
    }
}
