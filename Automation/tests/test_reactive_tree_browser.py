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
    def test_tree_search_placeholders_fit_at_fullscreen_and_narrow_widths(self):
        harness = r'''<script>
setTimeout(()=>{
  const out=document.createElement('pre');out.id='browser-test-result';
  try {
    const group=document.querySelector('.ancestor-search').getBoundingClientRect();
    if(getComputedStyle(document.body).fontSize!=='14px')throw Error('original body font not restored');
    if(getComputedStyle(document.querySelector('h1')).fontSize!=='20px')throw Error('original heading font not restored');
    const layer=document.getElementById('layerFilter'),neuron=document.getElementById('neuronFilter');
    if(layer.getAttribute('role')!=='combobox'||neuron.getAttribute('role')!=='combobox')throw Error('editable dropdown lists missing');
    const values=id=>[...document.getElementById(id).options].map(option=>option.value).join(',');
    if(values('layerOptions')!=='1,2')throw Error('all network ReLU layers must be available before exploration');
    layer.value='1';layer.dispatchEvent(new Event('input'));
    if(values('neuronOptions')!=='0,2')throw Error('only neurons with records belong in the first layer options');
    layer.value='2';layer.dispatchEvent(new Event('input'));
    if(values('neuronOptions')!=='1,4')throw Error('neuron options must follow selected layer and exclude unrecorded neurons');
    neuron.value='4';neuron.dispatchEvent(new Event('input'));
    if(neuron.value!=='4')throw Error('manual neuron input lost');
    layer.click();
    if(document.querySelectorAll('#layerFilterChoices [role="option"]').length!==2)throw Error('filled layer input must still show every layer');
    neuron.click();
    if(document.querySelectorAll('#neuronFilterChoices [role="option"]').length!==2)throw Error('filled neuron input must still show all recorded neurons of selected layer');
    document.querySelector('#neuronFilterChoices [data-value="1"]').click();
    if(neuron.value!=='1'||!document.getElementById('neuronFilterChoices').hidden)throw Error('dropdown selection must fill input and close');
    layer.value='';neuron.value='';layer.dispatchEvent(new Event('input'));
    const canvas=document.createElement('canvas').getContext('2d');
    for(const id of ['layerFilter','neuronFilter']) {
      const input=document.getElementById(id), style=getComputedStyle(input), box=input.getBoundingClientRect();
      canvas.font=style.font;
      const available=input.clientWidth-parseFloat(style.paddingLeft)-parseFloat(style.paddingRight);
      if(canvas.measureText(input.placeholder).width>available)throw Error(id+' placeholder clipped');
      if(box.width>134)throw Error(id+' wider than compact search field');
      if(box.left<group.left || box.right>group.right)throw Error(id+' outside search box');
    }
    if(innerWidth>=1400 && Math.abs(group.top-document.getElementById('defaultView').getBoundingClientRect().top)>8)throw Error('compact search should fit on the first control row: '+[...document.querySelector('.controls').children].map(item=>Math.round(item.getBoundingClientRect().width)).join(','));
    out.textContent='PASS: search fields fit';
  } catch(error) {out.textContent='FAIL: '+error.message;}
  document.body.append(out);
},100);
</script>'''
        with tempfile.TemporaryDirectory(prefix='tree-search-layout-') as directory:
            root = Path(directory)
            report = root / 'tree.html'
            trace = SolveTrace(events=[TraceEvent('reluplex_branch', i, i+0.5, 1,
                                      f'z{layer}_{neuron}', {'phase': '+', 'result': 'UNSAT',
                                                           'layer': layer, 'index': neuron})
                                      for i, (layer, neuron) in enumerate([(1, 0), (1, 2), (2, 1), (2, 4)])])
            write_relu_search_tree_html(trace, report, overall_status='UNSAT',
                                        model_layer_sizes=[5, 3, 5, 2])
            report.write_text(report.read_text(encoding='utf-8').replace('</body>', harness + '</body>'), encoding='utf-8')
            for width in (1920, 1440, 1280, 640):
                with self.subTest(width=width):
                    result = subprocess.run([
                        chromium_path(), '--headless', '--no-sandbox', '--disable-gpu', '--no-first-run',
                        f'--window-size={width},1080', f'--user-data-dir={root / str(width)}',
                        '--virtual-time-budget=1000', '--dump-dom', report.as_uri(),
                    ], capture_output=True, timeout=30)
                    match = re.search(r'<pre id="browser-test-result">([^<]+)</pre>', result.stdout.decode('utf-8', errors='replace'))
                    self.assertIsNotNone(match)
                    self.assertTrue(match.group(1).startswith('PASS:'), match.group(1))

    @unittest.skipUnless(chromium_path(), "Chromium browser is not installed")
    def test_only_mechanisms_expand_without_resizing_other_panels(self):
        with tempfile.TemporaryDirectory(prefix="mechanism-layout-") as directory:
            root = Path(directory)
            pixel = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+afo0AAAAASUVORK5CYII=")
            image = root / 'plot.svg'
            image.write_text('<svg xmlns="http://www.w3.org/2000/svg" width="1050" height="720"></svg>', encoding='utf-8')
            page = root / 'stub.html'
            page.write_text('<!doctype html><body>fixture</body>', encoding='utf-8')
            dashboard = root / 'dashboard.html'
            UnifiedRealtimeDashboard({'relu': image, 'simplex': image, 'dpll-theory': image}, dashboard,
                                     embedded_pages={'relu-tree': page, 'solver-mechanisms': page}).write()
            harness = r'''<script>
setTimeout(async ()=>{
  const out=document.createElement('pre');out.id='browser-test-result';
  try {
    const assert=(ok,message)=>{if(!ok)throw Error(message);};
    const large=document.querySelector('.mechanisms-section');
    const regular=[...document.querySelectorAll('main>section:not(.mechanisms-section)')];
    const widths=()=>regular.map(section=>section.getBoundingClientRect().width);
    const expanded=widths();large.style.display='none';const baseline=widths();large.style.display='';
    assert(expanded.every((width,index)=>Math.abs(width-baseline[index])<1),'mechanisms must not widen other panels');
    assert(large.getBoundingClientRect().width>regular[0].getBoundingClientRect().width,'only mechanisms have a larger canvas');
    assert(large.getBoundingClientRect().width<=1680,'mechanisms have a moderate maximum width');
    const dpll=document.querySelector('#dpll-theory-dashboard').getBoundingClientRect();
    const dpllFrame=document.querySelector('.dpll-theory-section').getBoundingClientRect();
    assert(Math.abs(dpll.left-dpllFrame.left-15)<1 && Math.abs(dpllFrame.right-dpll.right-15)<1,'DPLL outer frame keeps its original padding');
    const simplex=document.querySelector('#simplex-dashboard').getBoundingClientRect();
    const relu=document.querySelector('#relu-dashboard').getBoundingClientRect();
    const reluSection=document.querySelector('.relu-section').getBoundingClientRect();
    assert(Math.abs(reluSection.width-relu.width-30)<1,'ReLU frame hugs the image with only the normal padding and border');
    assert(Math.abs(reluSection.left+reluSection.width/2-(dpll.left+dpll.width/2))<1,'ReLU frame remains centered');
    assert(Math.abs(dpll.width-simplex.width)<1 && Math.abs(simplex.width-relu.width)<1,'all three panels start at the same width');
    assert(Math.abs(dpll.width/dpll.height-35/24)<0.01 && Math.abs(simplex.width/simplex.height-35/24)<0.01 && Math.abs(relu.width/relu.height-35/24)<0.01,'all three panels start at the same aspect ratio');
    const target=panels.find(panel=>panel.id==='dpll-theory-dashboard');
    target.uri='data:image/svg+xml,'+encodeURIComponent('<svg xmlns="http://www.w3.org/2000/svg" width="1500" height="720"></svg>')+'#';
    refreshPanel(target);
    for(let i=0;i<100&&document.querySelector('#dpll-theory-dashboard').naturalWidth!==1500;i++)await new Promise(resolve=>setTimeout(resolve,10));
    const wider=document.querySelector('#dpll-theory-dashboard').getBoundingClientRect();
    assert(wider.width>dpll.width && Math.abs(wider.height-dpll.height)<1,'only DPLL grows horizontally without stretching its height');
    assert(Math.abs(document.querySelector('#simplex-dashboard').getBoundingClientRect().width-simplex.width)<1 && Math.abs(document.querySelector('#relu-dashboard').getBoundingClientRect().width-relu.width)<1,'DPLL expansion does not resize Simplex or ReLU');
    assert(document.querySelector('#relu-tree-dashboard').getBoundingClientRect().height===760,'tree height stays unchanged');
    const frame=document.querySelector('#solver-mechanisms-dashboard').getBoundingClientRect();
    assert(Math.abs(frame.width/frame.height-1.5)<0.01,'mechanism canvas follows its own aspect ratio');
    assert(document.documentElement.scrollWidth<=document.documentElement.clientWidth+1,'expanded panel remains within the page');
    out.textContent='PASS: only mechanisms expand';
  } catch(error){out.textContent='FAIL: '+error.message;}
  document.body.append(out);
},100);
</script>'''
            dashboard.write_text(dashboard.read_text(encoding='utf-8').replace('</body>', harness + '</body>'), encoding='utf-8')
            result = subprocess.run([
                chromium_path(), '--headless', '--no-sandbox', '--disable-gpu', '--no-first-run',
                '--window-size=1920,1080', f'--user-data-dir={root / "profile"}',
                '--virtual-time-budget=1000', '--dump-dom', dashboard.as_uri(),
            ], capture_output=True, timeout=30)
            match = re.search(r'<pre id="browser-test-result">([^<]+)</pre>', result.stdout.decode('utf-8', errors='replace'))
            self.assertIsNotNone(match)
            self.assertTrue(match.group(1).startswith('PASS:'), match.group(1))

    @unittest.skipUnless(chromium_path(), "Chromium browser is not installed")
    def test_self_contained_file_refreshes_without_reloading_visible_tree(self):
        trace = SolveTrace(events=[
            TraceEvent("reluplex_mechanism", 0.5, 0.5, 0, None, {"kind": "settings", "backjump": False}),
            TraceEvent("reluplex_branch", 1, None, 1, "z1_1", {"phase": "+", "result": "RUNNING"}),
            TraceEvent("reluplex_mechanism", 1.5, 1.5, 1, None,
                       {"kind": "repair", "rule": "update_f", "variable": "z1_1",
                        "before": [1, 0], "after": [1, 1], "pivoted": True}),
            TraceEvent("reluplex_mechanism", 1.6, 1.6, 1, None,
                       {"kind": "bounds", "source": "search", "changed": 2, "fixed": 1}),
            TraceEvent("reluplex_branch", 2, None, 2, "z2_2", {"phase": "+", "result": "RUNNING"}),
            TraceEvent("reluplex_branch", 3, 4, 3, "z3_3", {"phase": "+", "result": "UNSAT"}),
        ])
        harness = r"""
<script>
(async () => {
  if (new URLSearchParams(location.search).has('tree-snapshot')) return;
  const assert = (ok, message) => { if (!ok) throw Error(message); };
  try {
    const sort = document.querySelector('#treeOrder');
    const resultFilter = document.querySelector('#statusFilter');
    assert(sort && sort.parentElement === resultFilter.parentElement, 'sort grouped with result filter');
    const chooseSort = value => {
      sort.value = value;
      sort.dispatchEvent(new Event('change'));
      return [...document.querySelectorAll('#tree > li details')].map(row => Number(row.dataset.id));
    };
    assert(chooseSort('layer').join(',') === '1,2,3', 'layer sort includes all nodes');
    assert(chooseSort('duration')[0] === 3, 'duration sorts longest first');
    assert(chooseSort('completion')[0] === 3, 'completed node before running nodes');
    assert(JSON.parse(window.name.slice('relu-search-tree-ui:'.length)).treeOrder === 'completion', 'sort persisted');
    chooseSort('node');
    const root = document.querySelector('details[data-id="1"]');
    assert(!document.querySelector('#mechanismRepair') && !document.querySelector('.mechanism-badges'), 'tree excludes mechanism panel and badges');
    root.querySelector('summary').dispatchEvent(new Event('mouseenter'));
    assert(!root.querySelector('summary').title.includes('Update_f (h ← ReLU(z))'), 'mechanism details removed from tree tooltip');
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
            write_relu_search_tree_html(trace, report, overall_status="RUNNING", live_snapshots=True, snapshot_version=1)
            write_relu_search_tree_html(trace, report, overall_status="RUNNING", live_snapshots=True,
                                        snapshot_only=True, snapshot_version=2)
            report.write_text(report.read_text(encoding="utf-8").replace("<script>", "<script>window.setInterval=()=>0;</script><script>", 1).replace("</body>", harness + "</body>"), encoding="utf-8")
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
