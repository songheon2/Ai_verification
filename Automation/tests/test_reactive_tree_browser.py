"""Exercise the generated, self-contained tree in an installed Chromium browser."""
from __future__ import annotations

import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import threading
import unittest
from urllib.parse import parse_qs, urlsplit

from visualization.ReluSearchTree import write_relu_search_tree_html
from visualization.SolveTrace import SolveTrace, TraceEvent
from visualization.UnifiedRealtimeDashboard import UnifiedRealtimeDashboard


def chromium_path():
    for candidate in (
        shutil.which("chromium"), shutil.which("google-chrome"),
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    ):
        if candidate and Path(candidate).is_file():
            return candidate
    return None


class ReactiveTreeBrowserTests(unittest.TestCase):
    @unittest.skipUnless(chromium_path(), "Chromium browser is not installed")
    def test_self_contained_file_refreshes_without_reloading_visible_tree(self):
        trace = SolveTrace(events=[
            TraceEvent("reluplex_branch", 1, None, 1, "z1_1", {"phase": "+", "result": "RUNNING"}),
            TraceEvent("reluplex_branch", 2, None, 2, "z2_2", {"phase": "+", "result": "RUNNING"}),
            TraceEvent("reluplex_branch", 3, 4, 3, "z3_3", {"phase": "+", "result": "UNSAT"}),
        ])
        harness = r"""
<script>
(async () => {
  if (new URLSearchParams(location.search).has('tree-snapshot')) return;
  const assert = (ok, message) => { if (!ok) throw Error(message); };
  try {
    const root = document.querySelector('details[data-id="1"]');
    root.querySelector('summary').click();
    assert(document.querySelectorAll('#tree details').length === 2, 'one child level');
    let snapshots = 0;
    window.addEventListener('message', event => {
      if (event.data?.type === 'relu-search-tree-snapshot') snapshots++;
    });
    for (let attempt = 0; attempt < 100 && snapshots === 0; attempt++) {
      await new Promise(resolve => setTimeout(resolve, 50));
    }
    assert(snapshots > 0, 'snapshot message arrived over file protocol');
    assert(document.querySelector('details[data-id="1"]') === root && root.open, 'same root and expansion');
    assert(document.querySelectorAll('#tree details').length === 2, 'hidden subtree remains absent');
    document.querySelector('#recentPath').click();
    assert(document.querySelector('details[data-id="2"]').open, 'recent path opens intermediate ancestor');
    assert(document.querySelector('details[data-id="3"]').classList.contains('recent-target'), 'recent path works in standalone file');
    assert(document.querySelectorAll('#tree details.recent-path').length === 3, 'standalone highlights full path');
    const output = document.createElement('pre'); output.id = 'browser-test-result';
    output.textContent = 'PASS: self-contained file refresh preserves visible tree'; document.body.appendChild(output);
  } catch (error) {
    const output = document.createElement('pre'); output.id = 'browser-test-result';
    output.textContent = 'FAIL: ' + error.message; document.body.appendChild(output);
  }
})();
</script>
"""
        with tempfile.TemporaryDirectory(prefix="reactive-relu-file-") as directory:
            root = Path(directory)
            report = root / "tree.html"
            write_relu_search_tree_html(trace, report, overall_status="RUNNING")
            report.write_text(report.read_text(encoding="utf-8").replace("</body>", harness + "</body>"), encoding="utf-8")
            result = subprocess.run([
                chromium_path(), "--headless", "--no-sandbox", "--disable-gpu",
                "--in-process-gpu", "--disable-software-rasterizer", "--no-first-run",
                "--no-default-browser-check", f"--user-data-dir={root / 'profile'}",
                "--virtual-time-budget=6000", "--dump-dom", report.as_uri(),
            ], capture_output=True, timeout=40)
            match = re.search(r'<pre id="browser-test-result">([^<]+)</pre>', result.stdout.decode("utf-8", errors="replace"))
            self.assertIsNotNone(match, result.stderr.decode("utf-8", errors="replace")[-2000:])
            self.assertTrue(match.group(1).startswith("PASS:"), match.group(1))

    @unittest.skipUnless(chromium_path(), "Chromium browser is not installed")
    def test_lazy_tree_preserves_dom_expansion_and_scroll_across_live_updates(self):
        with tempfile.TemporaryDirectory(prefix="reactive-relu-tree-") as directory:
            root = Path(directory)
            trace = SolveTrace(events=[
                TraceEvent("reluplex_branch", 1, None, 1, "z1_1", {"phase": "+", "result": "RUNNING"}),
                TraceEvent("reluplex_branch", 2, None, 2, "z2_2", {"phase": "+", "result": "RUNNING"}),
                TraceEvent("reluplex_branch", 3, None, 3, "z3_3", {"phase": "+", "result": "RUNNING"}),
                TraceEvent("reluplex_branch", 4, 5, 4, "z4_4", {"phase": "+", "result": "UNSAT"}),
            ])
            for index in range(100):
                trace.events.append(TraceEvent(
                    "reluplex_branch", 10 + index * 2, 11 + index * 2, 3,
                    f"z3_{index + 10}", {"phase": "+", "result": "UNSAT"},
                ))
            trace.events.append(TraceEvent(
                "reluplex_branch", 1000, None, 2, "z2_200", {"phase": "-", "result": "RUNNING"},
            ))
            stages = []
            trace.events.append(TraceEvent("reluplex_stable", 0, 0, 0, meta={
                "source": "initial", "round_index": 1, "records": [{
                    "variable": "z1_99", "layer": 1, "index": 99, "state": "active",
                    "before": [None, None], "after": [1, 2],
                }],
            }))

            def snapshot(status):
                target = root / f"stage{len(stages)}.html"
                write_relu_search_tree_html(trace, target, overall_status=status)
                stages.append(target.read_bytes())

            snapshot("RUNNING")
            trace.events[1].t_end = 999
            trace.events[1].meta["result"] = "UNSAT"
            trace.events[2].t_end = 998
            trace.events[2].meta["result"] = "UNSAT"
            trace.events.append(TraceEvent(
                "reluplex_branch", 1001, 1002, 3, "z3_201", {"phase": "+", "result": "UNSAT"},
            ))
            trace.events.append(TraceEvent("reluplex_stable", 1001.5, 1001.6, 3, meta={
                "source": "search", "round_index": 1, "records": [{
                    "variable": "z4_9", "layer": 4, "index": 9, "state": "active",
                    "before": [-1, 2], "after": [0.1, 1.9],
                }],
            }))
            snapshot("RUNNING")
            trace.events.append(TraceEvent(
                "reluplex_branch", 1003, 1004, 3, "z3_202", {"phase": "+", "result": "UNSAT"},
            ))
            trace.events.append(TraceEvent("reluplex_stable", 1004.5, 1004.6, 2, meta={
                "source": "pivot", "round_index": 1, "records": [{
                    "variable": "z2_1", "layer": 2, "index": 1, "state": "inactive",
                    "before": [-1, 1], "after": [-0.5, -0.1],
                }],
            }))
            snapshot("RUNNING")
            trace.events.append(TraceEvent(
                "reluplex_branch", 1005, 1006, 3, "z3_203", {"phase": "-", "result": "UNSAT"},
            ))
            trace.events.append(TraceEvent("reluplex_stable", 1005.5, 1005.6, 3, meta={
                "source": "search", "round_index": 1, "records": [{
                    "variable": f"z3_{index + 2}", "layer": 3, "index": index + 2,
                    "state": "inactive", "before": [-2, 1], "after": [-1.5, -0.2],
                } for index in range(60)],
            }))
            snapshot("RUNNING")
            trace.events.extend([
                TraceEvent("reluplex_branch", 1007, 1008, 3, "z3_204", {"phase": "+", "result": "UNSAT"}),
                TraceEvent("reluplex_branch", 1009, 1010, 3, "z3_205", {"phase": "-", "result": "UNSAT"}),
            ])
            for event in trace.events:
                if event.t_end is None:
                    event.t_end = 2000
                    event.meta["result"] = "TIMEOUT"
            snapshot("UNKNOWN")

            result_pages = {}
            live_subtree_stages = []
            for stage in range(6):
                live_trace = SolveTrace(events=[
                    TraceEvent("reluplex_branch", 1, None, 1, "z1_1", {"phase": "+", "result": "RUNNING"}),
                    TraceEvent("reluplex_branch", 2, 3, 2, "z2_10", {"phase": "-", "result": "UNSAT"}),
                    *[TraceEvent("reluplex_branch", depth + 2, None, depth, f"z{depth}_20", {"phase": "+", "result": "RUNNING"})
                      for depth in range(2, min(stage, 4) + 3)],
                ])
                if stage == 5:
                    live_trace.events.append(TraceEvent("reluplex_branch", 20, None, 3, "z3_20", {"phase": "-", "result": "RUNNING"}))
                live_path = root / f"live_subtree_{stage}.html"
                write_relu_search_tree_html(live_trace, live_path, overall_status="RUNNING")
                live_subtree_stages.append(live_path.read_bytes())
            deep_trace = SolveTrace(events=[
                TraceEvent("reluplex_branch", depth, 500 - depth, depth, f"z{depth}_1", {"phase": "+", "result": "TIMEOUT"})
                for depth in range(1, 81)
            ])
            deep_path = root / "deep.html"
            write_relu_search_tree_html(deep_trace, deep_path, overall_status="TIMEOUT")
            result_pages["/deep.html"] = deep_path.read_bytes()
            for status in ("SAT", "UNSAT", "UNKNOWN", "TIMEOUT"):
                result_trace = SolveTrace(events=[
                    TraceEvent("reluplex_branch", 1, 10, 1, "z1_1", {"phase": "+", "result": status}),
                    TraceEvent("reluplex_branch", 2, 3, 2, "z2_1", {"phase": "+", "result": "UNSAT"}),
                    TraceEvent("reluplex_branch", 4, 9, 2, "z2_1", {"phase": "-", "result": status}),
                    TraceEvent("reluplex_branch", 5, 8, 3, "z3_1", {"phase": "+", "result": status}),
                ])
                result_path = root / f"result_{status}.html"
                write_relu_search_tree_html(result_trace, result_path, overall_status=status)
                result_pages[f"/result_{status}.html"] = result_path.read_bytes()

            pixel = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+afo0AAAAASUVORK5CYII=")
            (root / "panel.png").write_bytes(pixel)
            dashboard = root / "index.html"
            UnifiedRealtimeDashboard(
                {"relu": root / "panel.png"}, dashboard,
                embedded_pages={"relu-tree": root / "tree.html"},
            ).write()
            harness = Path(__file__).with_name("reactive_tree_browser_harness.js").read_text(encoding="utf-8")
            document = dashboard.read_text(encoding="utf-8").replace("</body>", f"<script>{harness}</script></body>")
            current_stage = [0]

            class Handler(BaseHTTPRequestHandler):
                def do_GET(self):
                    url = urlsplit(self.path)
                    if url.path == "/stage":
                        current_stage[0] = int(parse_qs(url.query)["value"][0])
                        content, mime = b"ok", "text/plain"
                    elif url.path == "/tree.html":
                        content, mime = stages[current_stage[0]], "text/html; charset=utf-8"
                    elif url.path == "/live-subtree.html":
                        content, mime = live_subtree_stages[current_stage[0]], "text/html; charset=utf-8"
                    elif url.path in result_pages:
                        content, mime = result_pages[url.path], "text/html; charset=utf-8"
                    elif url.path == "/panel.png":
                        content, mime = pixel, "image/png"
                    else:
                        content, mime = document.encode("utf-8"), "text/html; charset=utf-8"
                    self.send_response(200)
                    self.send_header("Content-Type", mime)
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Content-Length", str(len(content)))
                    self.end_headers()
                    self.wfile.write(content)

                def log_message(self, *_args):
                    pass

            server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            try:
                result = subprocess.run([
                    chromium_path(), "--headless", "--no-sandbox", "--disable-gpu",
                    "--in-process-gpu", "--disable-software-rasterizer", "--no-first-run",
                    "--no-default-browser-check", f"--user-data-dir={root / 'profile'}",
                    "--virtual-time-budget=12000", "--dump-dom",
                    f"http://127.0.0.1:{server.server_port}/index.html",
                ], capture_output=True, timeout=40)
                dom = result.stdout.decode("utf-8", errors="replace")
                match = re.search(r'<pre id="browser-test-result">([^<]+)</pre>', dom)
                self.assertIsNotNone(match, result.stderr.decode("utf-8", errors="replace")[-2000:] + dom[-1000:])
                self.assertTrue(match.group(1).startswith("PASS:"), match.group(1))
            finally:
                server.shutdown()
                server.server_close()
                worker.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
