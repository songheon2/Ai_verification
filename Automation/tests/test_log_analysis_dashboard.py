"""Scoped data isolation and loopback dashboard tests; no real Gemini calls."""
import hashlib
from http.client import HTTPConnection
import json
from pathlib import Path
import re
import subprocess
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from Automation.LogAnalysisIndex import RunIndex
from Automation.LogAnalysisScope import MECHANISMS, TOPICS, ScopedIndex, normalize_scope, scope_key, scope_options
from Automation.LogAnalysisServer import AnalysisApp, make_server
from Automation.tests.test_log_analysis import FakeClient, answer
from Automation.tests.test_reactive_tree_browser import chromium_path
from visualization.LogAnalysisPanel import panel_markup
from visualization.UnifiedRealtimeDashboard import UnifiedRealtimeDashboard


class PanelAvailabilityTests(unittest.TestCase):
    @unittest.skipUnless(chromium_path(), "Chromium is not installed")
    def test_analysis_stays_beside_graphs_while_scrolling(self):
        for width, height in ((1920, 900), (1366, 768), (800, 850)):
            with self.subTest(width=width, height=height), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                path = root / "dashboard.html"
                image = root / "simplex.svg"
                image.write_text('<svg xmlns="http://www.w3.org/2000/svg" width="960" height="850"></svg>',
                                 encoding="utf-8")
                UnifiedRealtimeDashboard({"simplex": image}, path).write()
                harness = r'''<script>(async()=>{
const out=document.createElement('pre');out.id='layout-test';
try{
 const panel=document.getElementById('llm-analysis-panel');
 const graph=document.querySelector('.simplex-section');
 const image=document.getElementById('simplex-dashboard');
 const header=document.querySelector('header');
 for(let i=0;i<50&&!image.naturalWidth;i++)await new Promise(r=>setTimeout(r,20));
 if(!image.naturalWidth)throw Error('missing graph image');
 const first=panel.getBoundingClientRect(),chart=graph.getBoundingClientRect();
 if(__WIDE__){
   if(first.left<chart.right||Math.abs(first.top-chart.top)>2)throw Error('panel is not aligned beside graph');
   if(chart.bottom>innerHeight)throw Error('graph extends below viewport');
   const report=document.getElementById('llm-result');
   report.textContent='Long analysis. '.repeat(500);
   if(report.scrollHeight<=report.clientHeight||panel.scrollHeight>panel.clientHeight+2){
     throw Error('report scroll '+report.scrollHeight+'/'+report.clientHeight+' panel '+panel.scrollHeight+'/'+panel.clientHeight+
       ' '+innerWidth+'x'+innerHeight+' '+[...panel.children].map(el=>[el.id||el.className||el.tagName,Math.round(el.getBoundingClientRect().height)]).filter(v=>v[1]).join('|'));
   }
   graph.style.minHeight='2000px';
   window.scrollTo(0,500);
   const top=panel.getBoundingClientRect().top;
   if(window.scrollY<490||Math.abs(top-header.getBoundingClientRect().bottom-16)>2)throw Error('panel did not stay under header');
   if(panel.getBoundingClientRect().bottom>innerHeight)throw Error('panel extends below viewport');
   graph.querySelector('.llm-topic-button').click();
   if(window.scrollY<490)throw Error('analysis action moved page position');
 }else{
   if(first.bottom>chart.top||first.right>innerWidth)throw Error('narrow layout overlaps or overflows');
 }
 out.textContent='PASS';
}catch(error){out.textContent='FAIL '+error.message;}document.body.appendChild(out);
})();</script>'''.replace('__WIDE__', 'true' if width >= 1080 else 'false')
                document = path.read_text(encoding="utf-8")
                path.write_text(document.replace('</body>', harness + '</body>'), encoding="utf-8")
                process = subprocess.run([
                    chromium_path(), "--headless", "--no-sandbox", "--disable-gpu", "--no-first-run",
                    f"--window-size={width},{height}", f"--user-data-dir={root / 'profile'}",
                    "--virtual-time-budget=2000", "--dump-dom", path.as_uri(),
                ], capture_output=True, timeout=20)
                dom = process.stdout.decode("utf-8", errors="replace")
                match = re.search(r'<pre id="layout-test">([^<]+)</pre>', dom)
                self.assertIsNotNone(match, dom[-1000:])
                self.assertEqual(match[1], "PASS")

    @unittest.skipUnless(chromium_path(), "Chromium is not installed")
    def test_list_only_includes_panels_present_in_dashboard(self):
        cases = [
            ({"simplex": "simplex.png"}, {}, ["all", "simplex"]),
            ({"relu": "relu.png"}, {"relu-tree": "tree.html"}, ["all", "relu", "tree"]),
            ({}, {"solver-mechanisms": "mechanisms.html"}, ["all", "mechanisms"]),
            ({}, {}, ["all"]),
        ]
        for number, (images, pages, expected) in enumerate(cases):
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                path = root / "dashboard.html"
                if images or pages:
                    UnifiedRealtimeDashboard(
                        {key: root / value for key, value in images.items()}, path,
                        embedded_pages={key: root / value for key, value in pages.items()},
                    ).write()
                    document = path.read_text(encoding="utf-8")
                else:
                    document = '<html><meta charset="utf-8"><body><main>' + panel_markup() + '</main></body></html>'
                harness = r'''<script>
const out=document.createElement('pre');out.id='availability-test';
try{
 const select=document.getElementById('llm-topic');
 const values=[...select.options].map(option=>option.value);
 if(JSON.stringify(values)!==JSON.stringify(__EXPECTED__))throw Error('choices '+JSON.stringify(values));
 if(select.value!=='all')throw Error('default is not all');
 if(document.querySelectorAll('.llm-topic-button').length!==values.length-1)throw Error('heading actions differ from available choices');
 if(values.includes('mechanisms')){
   select.value='mechanisms';select.dispatchEvent(new Event('change'));
   if(document.getElementById('llm-mechanism-label').hidden)throw Error('missing mechanism tabs');
   document.querySelector('[data-mechanism="numeric"]').click();
   if(document.querySelector('[data-mechanism="numeric"]').getAttribute('aria-checked')!=='true')throw Error('inactive mechanism tab');
 }
 out.textContent='PASS';
}catch(error){out.textContent='FAIL '+error.message;}document.body.appendChild(out);
</script>'''.replace('__EXPECTED__', json.dumps(expected))
                path.write_text(document.replace('</body>', harness + '</body>'), encoding="utf-8")
                process = subprocess.run([
                    chromium_path(), "--headless", "--no-sandbox", "--disable-gpu", "--no-first-run",
                    f"--user-data-dir={root / ('profile-' + str(number))}", "--virtual-time-budget=1000", "--dump-dom", path.as_uri(),
                ], capture_output=True, timeout=20)
                dom = process.stdout.decode("utf-8", errors="replace")
                match = re.search(r'<pre id="availability-test">([^<]+)</pre>', dom)
                self.assertIsNotNone(match, dom[-1000:])
                self.assertEqual(match[1], "PASS")


class ScopeFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = self.root / "run"
        self.run.mkdir()
        self.write("result.json", {"status": "UNSAT", "model": {"layer_sizes": [2, 3, 1]},
                                   "solver": {"status": "UNSAT", "theory_stats": {"GLOBAL": 999}}})
        self.write("solver_progress.jsonl", [
            {"event": "round_start", "round": 1},
            {"event": "simplex_start", "call_id": "simplex-1", "round": 1, "depth": 1},
            {"event": "simplex_end", "call_id": "simplex-1", "duration_seconds": 2, "pivots": 3},
            {"event": "simplex_start", "call_id": "simplex-2", "round": 2, "depth": 1},
            {"event": "simplex_end", "call_id": "simplex-2", "duration_seconds": 7, "pivots": 9},
            {"event": "theory_result", "round": 1, "result": "CONFLICT"},
        ])
        self.write("relu_events_branches.jsonl", [
            {"event": "relu_mechanism", "kind": "settings", "round_index": 1, "backjump": False},
            {"event": "relu_mechanism", "kind": "bound_updates", "round_index": 1, "changed": 2,
             "changes": [{"variable": "z1_0", "after": [0, 1]}, {"variable": "z1_1", "after": [-1, 2]}]},
            {"event": "relu_stable", "round_index": 1, "source": "search",
             "records": [{"variable": "z1_0", "state": "active"}, {"variable": "z1_1", "state": "inactive"}]},
            {"event": "branch_start", "branch_id": "b", "round_index": 1, "layer": 1, "index": 0, "variable": "z1_0"},
            {"event": "branch_end", "branch_id": "b", "result": "UNSAT"},
            {"event": "relu_mechanism", "kind": "numerical", "round_index": 1, "drift": .01},
            {"event": "relu_mechanism", "kind": "backjump", "round_index": 1, "target_depth": 1},
        ])
        self.write("relu_events.jsonl", [{"event": "+", "layer": 1, "index": 0, "variable": "z1_0", "time_ns": 10}])
        self.write("relu_events_meta.json", {"round_starts": [{"round": 1, "time_ns": 5}]})

    def write(self, name, data):
        text = "".join(json.dumps(v) + "\n" for v in data) if name.endswith("jsonl") else json.dumps(data)
        (self.run / name).write_text(text, encoding="utf-8")

    def selected(self, scope):
        index = RunIndex(self.run, "run")
        self.addCleanup(index.close)
        return ScopedIndex(index, scope)


class ScopedLogTests(ScopeFixture):
    def test_multiple_selections_keep_exact_union_without_raw_scope_widening(self):
        selected = self.selected({'selections': [
            {'topic': 'simplex', 'round': 1, 'call_id': 'simplex-1'},
            {'topic': 'mechanisms', 'mechanism': 'bounds', 'round': 1, 'layer': 1, 'neuron': 0},
        ]})
        rows = selected.query({})['records']
        raw = json.dumps(rows)
        self.assertIn('simplex-1', raw)
        self.assertIn('z1_0', raw)
        self.assertNotIn('simplex-2', raw)
        self.assertNotIn('z1_1', raw)
        self.assertNotIn('numerical', raw)
        self.assertNotIn('branch_start', raw)
        self.assertEqual(selected.query({'call_id': 'simplex-2'})['records'], [])
        summary = selected.summary()
        self.assertNotIn('GLOBAL', json.dumps(summary))
        self.assertEqual(summary['indexed_completed_simplex']['calls'], 1)

    def test_multiple_neurons_project_a_shared_batch_once(self):
        selected = self.selected({'selections': [
            {'topic': 'bounds', 'round': 1, 'layer': 1, 'neuron': 0},
            {'topic': 'bounds', 'round': 1, 'layer': 1, 'neuron': 1},
        ]})
        rows = selected.query({})['records']
        stable = [row['data'] for row in rows if row['data']['event'] == 'relu_stable']
        self.assertEqual(len(stable), 1)
        self.assertEqual({item['variable'] for item in stable[0]['records']}, {'z1_0', 'z1_1'})
        self.assertNotIn('simplex_analysis', selected.summary())

    def test_multi_scope_is_canonical_bounded_and_rejects_nesting(self):
        first, second = {'topic': 'simplex', 'call_id': 'simplex-1'}, {'topic': 'numeric'}
        self.assertEqual(scope_key({'selections': [first, second, first]}), scope_key({'selections': [second, first]}))
        for value in ({'selections': []}, {'selections': [first] * 21}, {'selections': [None]},
                      {'selections': [{'selections': [first]}]}, {'topic': 'all', 'selections': [first]}):
            with self.subTest(scope=value), self.assertRaises(ValueError):
                normalize_scope(value)

    def test_default_all_keeps_all_records_and_combines_summary_context(self):
        self.assertEqual(normalize_scope({}), {"topic": "all"})
        selected = self.selected({})
        summary = selected.summary()
        self.assertEqual(summary["scope"], {"topic": "all"})
        self.assertEqual(summary["record_availability"]["selected_records"], 14)
        self.assertIn("simplex_analysis", summary)
        self.assertIn("tree_analysis", summary)
        self.assertEqual(set(summary["feature_configuration"]["features"]), set(MECHANISMS) - {"all"})
        self.assertTrue(selected.query({"event": "round_start"})["records"])
        self.assertTrue(selected.query({"event": "simplex_end"})["records"])
        self.assertTrue(selected.query({"event": "branch_start"})["records"])

    def test_mechanisms_group_and_specific_tabs_filter_actual_data(self):
        self.assertNotIn("bounds", TOPICS)
        self.assertIn("mechanisms", TOPICS)
        selected = self.selected({"topic": "mechanisms"})
        records = selected.query({})["records"]
        self.assertTrue(all(row["data"]["event"] in ("relu_mechanism", "relu_stable") for row in records))
        self.assertEqual(selected.scope, {"topic": "mechanisms", "mechanism": "all"})
        numeric = self.selected({"topic": "mechanisms", "mechanism": "numeric", "round": 1})
        self.assertEqual([row["data"]["kind"] for row in numeric.query({})["records"]], ["numerical"])
        self.assertEqual(scope_key({"topic": "numeric"}), scope_key({"topic": "mechanisms", "mechanism": "numeric"}))
        self.assertEqual(numeric.summary()["implementation_reference"]["topic"], "numeric")

    def test_tree_topic_includes_branch_results_and_stable_records_with_neuron_scope(self):
        selected = self.selected({"topic": "tree", "round": 1, "layer": 1, "neuron": 0})
        rows = selected.query({})["records"]
        self.assertEqual(sorted(row["data"]["event"] for row in rows),
                         ["branch_end", "branch_start", "relu_stable"])
        summary = selected.summary()
        self.assertEqual(summary["tree_analysis"]["started_branches"], 1)
        self.assertEqual(summary["tree_analysis"]["end_records"], 1)
        self.assertEqual(summary["tree_analysis"]["starts_without_end_record"], 0)
        self.assertEqual(summary["tree_analysis"]["stable_confirmation_occurrences"], 1)
        self.assertNotIn("z1_1", json.dumps(rows))
        self.assertEqual(summary["implementation_reference"]["topic"], "tree")

    def test_simplex_round_and_call_cannot_be_widened_by_tool_args(self):
        selected = self.selected({"topic": "simplex", "round": 1, "call_id": "simplex-1"})
        self.assertEqual(selected.summary()["indexed_completed_simplex"], {"calls": 1, "seconds": 2, "pivots": 3})
        self.assertEqual(selected.query({"round": 2})["records"], [])
        self.assertEqual(selected.query({"call_id": "simplex-2"})["records"], [])
        self.assertEqual(selected.query({"event": "theory_result"})["records"], [])

    def test_neuron_boundary_batches_are_projected_before_upload(self):
        selected = self.selected({"topic": "bounds", "round": 1, "layer": 1, "neuron": 0})
        data = selected.query({})["records"]
        encoded = json.dumps(data)
        self.assertNotIn("z1_1", encoded)
        self.assertIn("z1_0", encoded)
        self.assertNotIn('"changed"', encoded)
        self.assertTrue(all(row["data"]["scope_projection"] for row in data))
        self.assertNotIn("GLOBAL", json.dumps(selected.summary()))

    def test_branch_terminal_and_split_rounds_link_using_ids_and_metadata(self):
        selected = self.selected({"topic": "relu", "round": 1, "layer": 1, "neuron": 0})
        events = [row["data"]["event"] for row in selected.query({})["records"]]
        self.assertEqual(sorted(events), ["+", "branch_end", "branch_start"])

    def test_mechanism_topics_do_not_include_other_mechanism_records(self):
        selected = self.selected({"topic": "numeric", "round": 1})
        rows = selected.query({})["records"]
        self.assertEqual([row["data"]["kind"] for row in rows], ["numerical"])

    def test_scope_options_only_list_actual_recorded_neurons(self):
        index = RunIndex(self.run, "run")
        self.addCleanup(index.close)
        options = scope_options(index)
        self.assertEqual(options["neurons"], {"1": [0, 1]})
        self.assertEqual(options["rounds"], [1, 2])

    def test_invalid_scopes_are_rejected(self):
        for scope in ({"topic": "exec"}, {"topic": "simplex", "layer": 1}, {"topic": "bounds", "neuron": 1},
                      {"topic": "bounds", "layer": -1}, {"topic": "dpll", "path": "outside"}, {"topic": "dpll", "round": 0},
                      {"topic": "all", "mechanism": "bounds"}, {"topic": "mechanisms", "mechanism": "exec"},
                      {"topic": "mechanisms", "mechanism": "numeric", "layer": 1}, {"topic": "all", "layer": 1}):
            with self.subTest(scope=scope), self.assertRaises(ValueError):
                normalize_scope(scope)


class AnalysisServerTests(ScopeFixture):
    @unittest.skipUnless(chromium_path(), "Chromium is not installed")
    def test_browser_selects_model_and_starts_verification_from_dropdowns(self):
        from unittest.mock import Mock

        self.app.model_catalog = lambda: {"analysis_models": [self.app.model, "gemini-3.8-flash"]}
        used = []
        def client_for_model(model, _progress):
            used.append(model)
            return FakeClient(answer([{"text": "분석 결과"}]))
        self.app.model_client_factory = client_for_model
        self.app.verification = Mock()
        self.app.verification.catalog.side_effect = lambda group: (
            {"groups": [{"id": "acas", "label": "ACAS"}], "engines": ["glpk"]} if group is None else
            {"models": [{"id": "net", "label": "2_9"}], "properties": [{"id": "prop", "label": "Prop 3"}]})
        self.app.verification.start.return_value = {"id": "test-verification"}
        self.app.verification.job.return_value = {"status": "complete", "verdict": "VERIFIED"}
        path = self.run / "relu_realtime.html"
        path.write_text('<!doctype html><html><meta charset="utf-8"><body><main>' + panel_markup() + r'''
<script>(async()=>{
const out=document.createElement('pre');out.id='picker-test';
const $=id=>document.getElementById(id);
const wait=async (ready,label)=>{
 for(let i=0;i<300;i++){if(ready())return;await new Promise(r=>setTimeout(r,25));}
 throw Error('timeout '+label);
};
try{
 const picker=$('verify-picker'), model=$('llm-model');
 await wait(()=>!picker.hidden&&model.options.length===2,'catalog');
 model.value='gemini-3.8-flash';
 $('llm-consent').click();
 await wait(()=>!$('llm-start').disabled,'analysis button');
 $('llm-start').click();
 await wait(()=>$('llm-progress').textContent.includes('gemini-3.8-flash'),'selected model result');
 picker.open=true;
 const form=picker.querySelector('form'), field=form.elements;
 await wait(()=>field.group.options.length>1,'benchmark menu');
 field.group.value='acas';field.group.dispatchEvent(new Event('change',{bubbles:true}));
 await wait(()=>!field.model.disabled&&!field.property.disabled,'model and property');
 field.model.value='net';field.property.value='prop';form.dispatchEvent(new Event('change'));
 form.querySelector('[type=submit]').click();
 await wait(()=>form.querySelector('[role="status"]').textContent.includes('VERIFIED'),'verification job');
 out.textContent='PASS';
}catch(error){out.textContent='FAIL '+error.message;}document.body.append(out);
})();</script></main></body></html>''', encoding="utf-8")
        process = subprocess.run([
            chromium_path(), "--headless", "--no-sandbox", "--disable-gpu", "--no-first-run",
            f"--user-data-dir={self.root / 'picker-profile'}", "--virtual-time-budget=18000", "--dump-dom",
            f"http://127.0.0.1:{self.server.server_port}{self.base}",
        ], capture_output=True, timeout=35)
        dom = process.stdout.decode("utf-8", errors="replace")
        match = re.search(r'<pre id="picker-test">([^<]+)</pre>', dom)
        self.assertIsNotNone(match, dom[-1000:])
        self.assertEqual(match[1], "PASS")
        self.assertEqual(used, ["gemini-3.8-flash"])
        self.app.verification.start.assert_called_once_with(
            {"group": "acas", "model": "net", "property": "prop", "engine": "glpk", "timeout": 670})

    def test_selected_gemini_model_is_frozen_for_the_analysis_job(self):
        chosen = "gemini-3.8-flash"
        listed = []
        used = []

        def model_catalog():
            listed.append(True)
            return {"analysis_models": [self.app.model, chosen, "gemini-4-carbon", "gemini-3.1-flash-lite"]}

        def model_client(model, _progress):
            used.append(model)
            return FakeClient(answer([{"text": "선택한 모델의 응답입니다."}]))

        self.app.model_catalog = model_catalog
        self.app.model_client_factory = model_client
        status, raw = self.request(self.base + "api/models")
        self.assertEqual(status, 200)
        catalog = json.loads(raw)
        self.assertEqual(catalog["models"], [self.app.model, chosen, "gemini-4-carbon"])
        self.assertEqual(catalog["paid_only"], ["gemini-4-carbon"])
        self.request(self.base + "api/models")
        self.assertEqual(len(listed), 1)
        headers = {"Content-Type": "application/json", "X-Analysis-Token": self.app.csrf}
        body = {"scope": {"topic": "simplex"}, "consent": True}
        for unavailable in ("https://arbitrary-endpoint", "gemini-3.1-flash-lite",
                            "gemini-4-hydrogen"):
            self.assertEqual(self.request(self.base + "api/analyze", {**body, "model": unavailable}, headers)[0], 400)
        body["model"] = chosen
        self.assertEqual(self.request(self.base + "api/analyze", {**body, "thinking_level": "minimal"}, headers)[0], 400)
        body["thinking_level"] = "high"
        status, raw = self.request(self.base + "api/analyze", body, headers)
        self.assertEqual(status, 202)
        job_id = json.loads(raw)["id"]
        self.app.executor.shutdown(wait=True)
        status, raw = self.request(self.base + "api/job?id=" + job_id)
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(raw)["result"]["model"], chosen)
        self.assertEqual(json.loads(raw)["result"]["thinking_level"], "high")
        self.assertEqual(used, [chosen])

    def test_dashboard_serves_local_fonts_without_exposing_raw_logs(self):
        status, page = self.request(self.base)
        self.assertEqual(status, 200)
        self.assertIn(b"data-visualization-font", page)
        status, font = self.request(self.base + "_fonts/NotoSansKR-Regular.ttf")
        self.assertEqual(status, 200)
        self.assertEqual(font[:4], b"\x00\x01\x00\x00")
        self.assertEqual(self.request(self.base + "result.json")[0], 403)

    def setUp(self):
        super().setUp()
        self.clients = []
        def client_factory(progress):
            client = FakeClient(answer([{"text": "## 선택 항목 분석\n\n**관찰**: 호출 기록입니다. [run:solver_progress.jsonl:3]"}]))
            self.clients.append(client)
            return client
        self.output = self.root / "analyses"
        self.app = AnalysisApp(self.run, output_dir=self.output, client_factory=client_factory)
        self.server = make_server(self.app)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"/s/{self.app.token}/"
        self.addCleanup(self.cleanup_server)

    def cleanup_server(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        self.app.close()

    def request(self, path, body=None, headers=None):
        connection = HTTPConnection("127.0.0.1", self.server.server_port, timeout=5)
        self.addCleanup(connection.close)
        connection.request("POST" if body is not None else "GET", path, body=json.dumps(body) if body is not None else None,
                           headers=headers or {})
        response = connection.getresponse()
        return response.status, response.read()

    def start_job(self, scope, question="", selected_items=None):
        body = {"scope": scope, "question": question, "consent": True}
        if selected_items is not None:
            body["selected_items"] = selected_items
        status, raw = self.request(self.base + "api/analyze", body,
                                   {"Content-Type": "application/json", "X-Analysis-Token": self.app.csrf})
        self.assertEqual(status, 202, raw)
        job_id = json.loads(raw)["id"]
        for _ in range(100):
            status, raw = self.request(self.base + "api/job?id=" + job_id)
            job = json.loads(raw)
            if job["status"] != "running":
                return job
            time.sleep(.02)
        self.fail("job did not complete")

    def test_ai_studio_route_requires_csrf_and_opens_only_the_quota_page(self):
        route = self.base + "api/open-ai-studio-quota"
        headers = {"Content-Type": "application/json"}
        with patch("Automation.LogAnalysisServer.webbrowser.open", return_value=True) as opened:
            self.assertEqual(self.request(route, {}, headers)[0], 403)
            opened.assert_not_called()
            headers["X-Analysis-Token"] = self.app.csrf
            self.assertEqual(self.request(route, {"url": "https://example.com/"}, headers)[0], 200)
            opened.assert_called_once_with("https://aistudio.google.com/rate-limit", new=2)

    def test_analysis_job_scoped_results_and_source_preservation(self):
        before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in self.run.iterdir()}
        job = self.start_job({"topic": "simplex", "round": 1, "call_id": "simplex-1"})
        self.assertEqual(job["result"]["status"], "complete")
        self.assertEqual(job["result"]["request_budget"], {"used": 1, "limit": 8, "remaining": 7})
        prompt = str(self.clients[0].requests)
        self.assertNotIn("GLOBAL", prompt)
        self.assertEqual(job["result"]["scope"]["call_id"], "simplex-1")
        status, raw = self.request(self.base + 'api/result?scope=%7B%22topic%22%3A%22bounds%22%7D')
        self.assertIsNone(json.loads(raw)["result"])
        self.assertEqual(before, {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in self.run.iterdir()})
        self.assertEqual(len(list(self.output.glob("*/analysis.json"))), 1)

    def test_followup_uses_only_questions_and_answers_from_this_server_session(self):
        first = self.start_job({"topic": "simplex"}, "처음 질문입니다.")
        self.start_job({"topic": "dpll"}, "그 이유는 무엇인가요?")
        parts = self.clients[1].requests[0][0][0]["parts"]
        contexts = [json.loads(part["text"]) for part in parts]
        previous = next(item["session_conversation"] for item in contexts if "session_conversation" in item)
        self.assertEqual(previous, [{"question": "처음 질문입니다.", "answer": first["result"]["analysis"]}])
        reopened = AnalysisApp(self.run, output_dir=self.output, client_factory=self.app.client_factory)
        try:
            self.assertTrue(reopened.results)
            self.assertEqual(reopened.conversation.snapshot(), [])
        finally:
            reopened.close()

    def test_multi_selection_job_shares_one_frozen_snapshot_and_saves_its_union(self):
        before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in self.run.iterdir()}
        selection = {'selections': [{'topic': 'simplex', 'round': 1, 'call_id': 'simplex-1'}, {'topic': 'numeric', 'round': 1}]}
        job = self.start_job(selection)
        self.assertEqual(job['result']['status'], 'complete')
        self.assertEqual(job['result']['scope'], normalize_scope(selection))
        self.assertEqual(len(self.clients), 1)
        self.assertEqual(job['result']['api_calls'], 1)
        prompt = str(self.clients[0].requests)
        self.assertEqual(job['result']['runs'][0]['indexed_completed_simplex']['calls'], 1)
        self.assertIn('함께 분석', prompt)
        self.assertEqual(before, {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in self.run.iterdir()})

    def test_clicked_snapshots_reach_model_separately_even_with_empty_or_full_question(self):
        scope = {"topic": "simplex", "round": 1, "call_id": "simplex-1"}
        items = [{"scope": scope, "text": "클릭한 그림 정보 " + "가" * 3500 + " SNAPSHOT_END"}]
        for question in ("", "질" * 2000):
            with self.subTest(question_length=len(question)):
                job = self.start_job(scope, question, items)
                self.assertEqual(job["result"]["status"], "complete")
                self.assertEqual(job["result"]["question"], question)
                self.assertEqual(job["result"]["selected_items"], items)
                parts = self.clients[-1].requests[0][0][0]["parts"]
                self.assertEqual(json.loads(parts[1]["text"])["visualization_selection"], items)
                self.assertNotIn("SNAPSHOT_END", parts[0]["text"])

    def test_invalid_clicked_snapshots_are_rejected_before_analysis(self):
        scope = {"topic": "simplex", "call_id": "simplex-1"}
        valid = {"scope": scope, "text": "sample"}
        invalid = [None, {}, [valid] * 21,
                   [{"scope": {"topic": "tree"}, "text": "wrong scope"}],
                   [{"scope": {"selections": [scope]}, "text": "nested"}],
                   [{"scope": scope, "text": "x" * 5001}],
                   [{"scope": scope, "text": "가" * 5000}] * 3,
                   [{"scope": scope, "text": 1}], [{**valid, "key": "extra"}]]
        headers = {"Content-Type": "application/json", "X-Analysis-Token": self.app.csrf}
        for items in invalid:
            with self.subTest(items_type=type(items).__name__):
                status, _ = self.request(self.base + "api/analyze",
                    {"scope": scope, "selected_items": items, "consent": True}, headers)
                self.assertEqual(status, 400)
        self.assertFalse(self.clients)
        self.assertFalse(self.app.jobs)

    def test_missing_records_still_allow_implementation_explanations(self):
        job = self.start_job({"topic": "bounds", "layer": 999, "neuron": 999})
        self.assertEqual(job["result"]["status"], "complete")
        self.assertEqual(job["result"]["api_calls"], 1)
        self.assertEqual(job["request_budget"], {"used": 1, "limit": 8, "remaining": 7})
        summary = job["result"]["runs"][0]
        self.assertEqual(summary["record_availability"]["selected_records"], 0)
        self.assertEqual(summary["record_availability"]["analysis_mode"], "configuration_and_reference")
        self.assertTrue(summary["implementation_reference"]["principles"])
        self.assertEqual(len(self.clients), 1)

    def test_disabled_underapprox_without_events_explains_recorded_configuration(self):
        with (self.run / "relu_events_branches.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"event": "relu_mechanism", "kind": "settings", "round_index": 1,
                                     "underapprox_enabled": False, "underapprox_epsilon": 0}) + "\n")
        job = self.start_job({"topic": "underapprox", "round": 1})
        self.assertEqual(job["result"]["status"], "complete")
        summary = job["result"]["runs"][0]
        self.assertEqual(summary["record_availability"]["selected_records"], 0)
        self.assertEqual(summary["feature_configuration"]["state"], "disabled")
        self.assertIn("원래 제약의 검증 단계가 빠진 것은 아니다", str(self.clients[0].requests))

    def test_live_request_budget_counts_attempts_and_resets_for_next_analysis(self):
        entered, release = threading.Event(), threading.Event()
        class CountingClient:
            requests_made = 0
            def generate(client, contents, *, final=False):
                for _ in range(3):
                    client.requests_made += 1
                    client.request_progress(client.requests_made, client.max_requests)
                entered.set()
                release.wait(5)
                return answer([{"text": "기록 분석입니다."}])
        self.app.client_factory = lambda progress: CountingClient()
        self.addCleanup(release.set)
        headers = {"Content-Type": "application/json", "X-Analysis-Token": self.app.csrf}
        status, raw = self.request(self.base + "api/analyze", {"scope": {"topic": "simplex"}, "consent": True}, headers)
        self.assertEqual(status, 202)
        accepted = json.loads(raw)
        self.assertEqual(accepted["request_budget"], {"used": 0, "limit": 8, "remaining": 8})
        self.assertTrue(entered.wait(5))
        _, raw = self.request(self.base + "api/job?id=" + accepted["id"])
        self.assertEqual(json.loads(raw)["request_budget"], {"used": 3, "limit": 8, "remaining": 5})
        _, raw = self.request(self.base + "api/info")
        self.assertIsNone(json.loads(raw)["provider_quota_remaining"])
        release.set()
        for _ in range(100):
            if self.app.active is None:
                break
            time.sleep(.02)
        self.app.client_factory = lambda progress: FakeClient(answer([{"text": "다음 분석입니다."}]))
        next_job = self.start_job({"topic": "dpll"})
        self.assertEqual(next_job["request_budget"], {"used": 1, "limit": 8, "remaining": 7})

    def test_server_has_no_browser_api_key_and_requires_origin_session_and_csrf(self):
        status, page = self.request(self.base)
        self.assertEqual(status, 200)
        self.assertIn(b'id="llm-analysis-panel"', page)
        self.assertNotIn(b'GEMINI_API_KEY', page)
        self.assertEqual(self.request("/api/info")[0], 403)
        self.assertEqual(self.request(self.base + "api/info", headers={"Host": "evil.example"})[0], 403)
        self.assertEqual(self.request(self.base + "api/info", headers={"Origin": "https://evil.example"})[0], 403)
        body = {"scope": {"topic": "dpll"}, "consent": True}
        self.assertEqual(self.request(self.base + "api/analyze", body, {"Content-Type": "application/json"})[0], 403)
        body["consent"] = False
        self.assertEqual(self.request(self.base + "api/analyze", body, {"Content-Type": "application/json", "X-Analysis-Token": self.app.csrf})[0], 400)

    def test_path_traversal_and_source_json_download_are_denied(self):
        for path in ("../outside.txt", "%2e%2e/outside.html", "result.json", "solver_progress.jsonl"):
            self.assertEqual(self.request(self.base + path)[0], 403)

    def test_saved_item_result_is_loaded_after_restart(self):
        self.start_job({"topic": "numeric"})
        other = AnalysisApp(self.run, output_dir=self.output, client_factory=lambda _: None)
        try:
            self.assertEqual(len(other.results), 1)
            self.assertEqual(next(iter(other.results.values()))["scope"], {"topic": "mechanisms", "mechanism": "numeric"})
        finally:
            other.close()

    @unittest.skipUnless(chromium_path(), "Chromium is not installed")
    def test_browser_live_snapshot_new_records_and_selection_preservation(self):
        (self.run / "result.json").unlink()
        self.write("run_info.json", {"status": "RUNNING", "model": {"layer_sizes": [2, 3, 1]}})
        def factory(progress):
            # Simulate the solver appending while Gemini processes a frozen selection.
            with (self.run / "relu_events_branches.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"event": "relu_stable", "round_index": 3,
                                         "records": [{"variable": "z1_2", "state": "active"}]}) + "\n")
            return FakeClient(answer([{"text": "고정된 경계 유도 분석입니다."}]))
        self.app.client_factory = factory
        harness = r'''<script>(async()=>{
const $=id=>document.getElementById(id),assert=(p,m)=>{if(!p)throw Error(m);};
const wait=async(p,label)=>{for(let i=0;i<450;i++){if(p())return;await fetch(new URL('api/info',location.href));await new Promise(r=>setTimeout(r,20));}throw Error('timeout '+label);};
const out=document.createElement('pre');out.id='test-result';
try{
 await wait(()=>$('llm-connection').textContent.includes('gemini-'),'connection');
 assert($('llm-connection').textContent==='gemini-3.5-flash-lite','only model name in connection line');
 assert(!$('llm-evidence'),'raw query evidence section removed');
 assert($('llm-live').textContent.includes('RUNNING'),'live run without final result');
 $('llm-topic').value='mechanisms';$('llm-topic').dispatchEvent(new Event('change'));
 document.querySelector('[data-mechanism="bounds"]').click();
 $('llm-layer').value='1';$('llm-layer').dispatchEvent(new Event('change'));
 $('llm-neuron').value='0';$('llm-neuron').dispatchEvent(new Event('change'));
 $('llm-consent').checked=true;$('llm-consent').dispatchEvent(new Event('change'));$('llm-start').click();
 await wait(()=>$('llm-result').textContent.includes('고정된 경계 유도'),'analysis');
 await wait(()=>$('llm-new-records').textContent.includes('+1건'),'new records');
 assert($('llm-snapshot').textContent.includes('진행 중'),'snapshot marked as ongoing');
 assert([...$('llm-round').options].some(o=>o.value==='3'),'new round choice');
 assert([...$('llm-neuron').options].some(o=>o.value==='2'),'new neuron choice');
 assert($('llm-layer').value==='1' && $('llm-neuron').value==='0','selection preserved');
 assert($('llm-consent').checked,'passive refresh preserves consent');
 assert($('llm-start').textContent==='현재까지 분석','manual snapshot action');
 out.textContent='PASS';
}catch(e){out.textContent='FAIL '+e.message;}document.body.appendChild(out);
})();</script>'''
        (self.run / "relu_realtime.html").write_text('<!doctype html><meta charset="utf-8"><main>'
            '<section><h2>Mechanisms</h2><iframe id="solver-mechanisms-dashboard"></iframe></section>'
            '</main>' + harness, encoding="utf-8")
        result = subprocess.run([chromium_path(), "--headless", "--no-sandbox", "--disable-gpu", "--no-first-run",
                                 f"--user-data-dir={self.root / 'live-profile'}", "--virtual-time-budget=20000", "--dump-dom",
                                 f"http://127.0.0.1:{self.server.server_port}{self.base}"], capture_output=True, timeout=35)
        dom = result.stdout.decode("utf-8", errors="replace")
        match = re.search(r'<pre id="test-result">([^<]+)</pre>', dom)
        self.assertIsNotNone(match, dom[-1000:])
        self.assertEqual(match[1], "PASS")

    @unittest.skipUnless(chromium_path(), "Chromium is not installed")
    def test_browser_multi_select_accumulates_and_submits_all_selected_items(self):
        harness = r'''<script>(async()=>{
const $=id=>document.getElementById(id),assert=(p,m)=>{if(!p)throw Error(m);};
const wait=async(p,label)=>{for(let i=0;i<400;i++){if(p())return;await fetch(new URL('api/info',location.href));await new Promise(r=>setTimeout(r,20));}throw Error('timeout '+label);};
const out=document.createElement('pre');out.id='multi-selection-test';
const pick=(call,round,text)=>window.dispatchEvent(new CustomEvent('log-analysis-select',{detail:{scope:{topic:'simplex',call_id:call,round},text}}));
const selectedText=()=>[...$('llm-selected-items').querySelectorAll('span')].map(span=>span.title).join('\n');
try{
 await wait(()=>$('llm-connection').textContent.includes('gemini-'),'connection');
 assert($('llm-multi-select').checked,'multi select is the default');
 $('llm-question').value='이 항목들을 비교해 주세요.';
 pick('simplex-1',1,'FIRST CALL');pick('simplex-2',2,'SECOND CALL');
 assert($('llm-selected-items').children.length===2,'default multi selection accumulates');
 $('llm-multi-select').click();
 assert($('llm-selected-items').children.length===1&&!selectedText().includes('FIRST CALL'),'single select keeps the latest item');
 $('llm-multi-select').click();pick('simplex-1',1,'FIRST CALL');pick('simplex-1',1,'FIRST CALL');
 assert($('llm-selected-items').children.length===2,'multi select accumulates without identical duplicates');
 assert(selectedText().includes('FIRST CALL')&&selectedText().includes('SECOND CALL'),'both snapshots are visible in chips');
 assert($('llm-question').value==='이 항목들을 비교해 주세요.','selection stays out of the question');
 $('llm-consent').click();$('llm-start').click();
 await wait(()=>$('llm-result-tag').dataset.status==='complete','joint analysis');
 window.dispatchEvent(new CustomEvent('log-analysis-select',{detail:{scope:{topic:'mechanisms',mechanism:'numeric',round:1},text:'NUMERIC SAMPLE'}}));
 assert($('llm-selected-items').children.length===3,'items across different panels accumulate');
 $('llm-selected-items').querySelector('button').click();
 assert($('llm-selected-items').children.length===2&&!selectedText().includes('SECOND CALL'),'individual removal updates selected context');
 $('llm-multi-select').click();
 assert($('llm-selected-items').children.length===1&&selectedText().includes('NUMERIC SAMPLE')&&!selectedText().includes('FIRST CALL'),'disabling multi keeps only the latest item');
 pick('simplex-2',2,'SECOND CALL');
 assert(!selectedText().includes('NUMERIC SAMPLE')&&$('llm-selected-items').children.length===1,'single select remains active');
 assert($('llm-consent').checked,'selection modes preserve explicit consent');
 $('llm-clear-selection').click();
 assert($('llm-selected-items').hidden&&$('llm-question').value==='이 항목들을 비교해 주세요.'&&$('llm-topic').value==='all','clear removes only generated context');
 out.textContent='PASS';
}catch(error){out.textContent='FAIL '+error.message;}document.body.appendChild(out);
})();</script>'''
        (self.run / 'relu_realtime.html').write_text('<!doctype html><meta charset="utf-8"><main>'
            '<section><h2>Simplex</h2><img id="simplex-dashboard"></section>'
            '<section><h2>Mechanisms</h2><iframe id="solver-mechanisms-dashboard"></iframe></section>'
            '</main>' + harness, encoding='utf-8')
        process = subprocess.run([chromium_path(), '--headless', '--no-sandbox', '--disable-gpu', '--no-first-run',
            f'--user-data-dir={self.root / "multi-profile"}', '--virtual-time-budget=18000', '--dump-dom',
            f'http://127.0.0.1:{self.server.server_port}{self.base}'], capture_output=True, timeout=30)
        dom = process.stdout.decode('utf-8', errors='replace')
        match = re.search(r'<pre id="multi-selection-test">([^<]+)</pre>', dom)
        self.assertIsNotNone(match, dom[-1000:])
        self.assertEqual(match[1], 'PASS')
        self.assertEqual(len(self.clients), 1, 'only the explicit analysis click calls Gemini')
        result = next(iter(self.app.results.values()))
        self.assertEqual(result['scope'], normalize_scope({'selections': [
            {'topic': 'simplex', 'call_id': 'simplex-1', 'round': 1},
            {'topic': 'simplex', 'call_id': 'simplex-2', 'round': 2}]}))
        self.assertEqual(result['runs'][0]['indexed_completed_simplex']['calls'], 2)
        self.assertEqual(result['question'], '이 항목들을 비교해 주세요.')
        self.assertEqual([item['text'] for item in result['selected_items']], ['SECOND CALL', 'FIRST CALL'])
        parts = self.clients[0].requests[0][0][0]['parts']
        self.assertEqual(json.loads(parts[1]['text'])['visualization_selection'], result['selected_items'])

    @unittest.skipUnless(chromium_path(), "Chromium is not installed")
    def test_browser_tree_nodes_are_individual_selected_items_even_for_the_same_neuron(self):
        from visualization.ReluSearchTree import write_relu_search_tree_html
        from visualization.SolveTrace import SolveTrace, TraceEvent
        tree = self.run / 'tree.html'
        trace = SolveTrace(events=[
            TraceEvent('reluplex_branch', 1, 2, 1, 'z1_0',
                       {'phase': '+', 'result': 'UNSAT', 'layer': 1, 'index': 0, 'round_index': 1}),
            TraceEvent('reluplex_branch', 3, 4, 1, 'z1_0',
                       {'phase': '-', 'result': 'UNSAT', 'layer': 1, 'index': 0, 'round_index': 1}),
        ])
        write_relu_search_tree_html(trace, tree, overall_status='UNSAT', mechanism_panels=[])
        page = self.run / 'relu_realtime.html'
        UnifiedRealtimeDashboard({}, page, embedded_pages={'relu-tree': tree}).write()
        harness = r'''<script>(async()=>{
const $=id=>document.getElementById(id),assert=(p,m)=>{if(!p)throw Error(m);};
const wait=async(p,label)=>{for(let i=0;i<350;i++){if(p())return;await fetch(new URL('api/info',location.href));await new Promise(r=>setTimeout(r,20));}throw Error('timeout '+label);};
const out=document.createElement('pre');out.id='tree-node-selection-test';
const chips=()=>[...$('llm-selected-items').querySelectorAll('span')];
try{
 await wait(()=>$('llm-connection').textContent.includes('gemini-'),'connection');
 const frame=$('relu-tree-dashboard');
 await wait(()=>frame.contentDocument?.querySelectorAll('#tree li > details > summary').length===2,'tree');
 const nodes=[...frame.contentDocument.querySelectorAll('#tree li > details > summary')];
 $('llm-question').value='이 두 노드를 비교해 주세요.';
 nodes[0].click();await wait(()=>chips().length===1,'first node');
 nodes[1].click();await wait(()=>chips().length===2,'second node');
 assert(chips().some(chip=>chip.textContent.includes('노드 #1'))&&chips().some(chip=>chip.textContent.includes('노드 #2')),'each chip visibly identifies its own node');
 assert(chips().every(chip=>chip.textContent.includes('R1')&&chip.textContent.includes('L1:N0')),'round and neuron remain visible');
 nodes[0].click();await wait(()=>chips().at(-1).textContent.includes('노드 #1'),'repeated node');
 assert(chips().length===2,'same node is not duplicated on expand/collapse');
 const firstLabel=nodes[0].querySelector('.meta');
 assert(firstLabel,'original tree node information remains');
 $('llm-consent').click();$('llm-start').click();
 await wait(()=>$('llm-result-tag').dataset.status==='complete','analysis');
 $('llm-multi-select').click();
 assert(chips().length===1&&chips()[0].textContent.includes('노드 #1'),'single mode retains last node');
 nodes[1].querySelector('.subtree-focus').click();
 await wait(()=>chips().length===1&&chips()[0].textContent.includes('노드 #2'),'branch focus button');
 assert($('llm-question').value==='이 두 노드를 비교해 주세요.','node selection stays outside question');
 $('llm-clear-selection').click();assert(chips().length===0,'node selections can be cleared');
 out.textContent='PASS';
}catch(error){out.textContent='FAIL '+error.message;}document.body.append(out);
})();</script>'''
        page.write_text(page.read_text(encoding='utf-8').replace('</body>', harness + '</body>'), encoding='utf-8')
        process = subprocess.run([chromium_path(), '--headless', '--no-sandbox', '--disable-gpu', '--no-first-run',
            f'--user-data-dir={self.root / "tree-node-profile"}', '--virtual-time-budget=15000', '--dump-dom',
            f'http://127.0.0.1:{self.server.server_port}{self.base}'], capture_output=True, timeout=30)
        dom = process.stdout.decode('utf-8', errors='replace')
        match = re.search(r'<pre id="tree-node-selection-test">([^<]+)</pre>', dom)
        self.assertIsNotNone(match, dom[-1500:])
        self.assertEqual(match[1], 'PASS')
        result = next(iter(self.app.results.values()))
        self.assertEqual(result['question'], '이 두 노드를 비교해 주세요.')
        self.assertEqual(len(result['selected_items']), 2)
        context = json.loads(self.clients[0].requests[0][0][0]['parts'][1]['text'])['visualization_selection']
        self.assertEqual(context, result['selected_items'])
        self.assertTrue(any('노드 #1' in item['text'] for item in context))
        self.assertTrue(any('노드 #2' in item['text'] for item in context))

    @unittest.skipUnless(chromium_path(), "Chromium is not installed")
    def test_browser_consent_persists_selection_reload_and_manual_opt_out(self):
        harness = r'''<script>(async()=>{
const $=id=>document.getElementById(id),assert=(p,m)=>{if(!p)throw Error(m);};
const wait=async p=>{for(let i=0;i<300;i++){if(p())return;await fetch(new URL('api/info',location.href));await new Promise(r=>setTimeout(r,20));}throw Error('connection timeout');};
const out=document.createElement('pre');out.id='consent-test-result';
try{
 await wait(()=>$('llm-connection').textContent.includes('gemini-'));
 const stage=sessionStorage.getItem('consent-test-stage');
 if(stage===null){
   assert(!$('llm-consent').checked&&$('llm-start').disabled,'fresh page starts unchecked');
   $('llm-consent').click();assert(!$('llm-start').disabled,'explicit opt-in enables analysis');
   $('llm-topic').value='mechanisms';$('llm-topic').dispatchEvent(new Event('change'));
   document.querySelector('[data-mechanism="numeric"]').click();
   assert($('llm-consent').checked,'topic and mechanism preserve consent');
   window.dispatchEvent(new CustomEvent('log-analysis-select',{detail:{scope:{topic:'simplex',call_id:'simplex-1',round:1},text:'첫 호출'}}));
   assert($('llm-consent').checked&&!$('llm-start').disabled,'chart selection preserves consent');
   $('llm-call').value='simplex-2';$('llm-call').dispatchEvent(new Event('input'));
   assert($('llm-consent').checked,'call edit preserves consent');
   sessionStorage.setItem('consent-test-stage','checked');location.reload();return;
 }
 if(stage==='checked'){
   assert($('llm-consent').checked&&!$('llm-start').disabled,'reload restores explicit consent');
   $('llm-consent').click();assert($('llm-start').disabled,'manual opt-out disables analysis');
   $('llm-topic').value='simplex';$('llm-topic').dispatchEvent(new Event('change'));
   assert(!$('llm-consent').checked,'selection cannot undo opt-out');
   sessionStorage.setItem('consent-test-stage','unchecked');location.reload();return;
 }
 assert(!$('llm-consent').checked&&$('llm-start').disabled,'reload preserves explicit opt-out');
 out.textContent='PASS';
}catch(error){out.textContent='FAIL '+error.message;}document.body.appendChild(out);
})();</script>'''
        (self.run / 'relu_realtime.html').write_text('<!doctype html><meta charset="utf-8"><main>'
            '<section><h2>Simplex</h2><img id="simplex-dashboard"></section>'
            '<section><h2>Mechanisms</h2><iframe id="solver-mechanisms-dashboard"></iframe></section>'
            '</main>' + harness, encoding='utf-8')
        result = subprocess.run([chromium_path(), '--headless', '--no-sandbox', '--disable-gpu', '--no-first-run',
            f'--user-data-dir={self.root / "consent-profile"}', '--virtual-time-budget=15000', '--dump-dom',
            f'http://127.0.0.1:{self.server.server_port}{self.base}'], capture_output=True, timeout=30)
        dom = result.stdout.decode('utf-8', errors='replace')
        match = re.search(r'<pre id="consent-test-result">([^<]+)</pre>', dom)
        self.assertIsNotNone(match, dom[-1000:])
        self.assertEqual(match[1], 'PASS')
        self.assertFalse(self.clients, 'checking consent must never call Gemini')
        self.assertFalse(self.app.jobs)

    @unittest.skipUnless(chromium_path(), "Chromium is not installed")
    def test_browser_selection_result_isolation_consent_and_no_html_injection(self):
        # Exercise the real panel with local mocked Gemini; never uses credentials/network.
        harness = r'''<script>(async()=>{
const wait=async(p,label)=>{for(let i=0;i<300;i++){if(p())return;await fetch(new URL('api/info',location.href));await new Promise(r=>setTimeout(r,20));}throw Error('timeout '+label+' '+document.getElementById('llm-progress')?.textContent+' '+document.getElementById('llm-result')?.textContent);};
const $=id=>document.getElementById(id),assert=(p,m)=>{if(!p)throw Error(m);};
const out=document.createElement('pre');out.id='test-result';
try{
 await wait(()=>$('llm-connection').textContent.includes('gemini-'),'connection');
 assert($('llm-start').disabled,'consent gate');
 assert($('llm-topic').value==='all','default selection is all');
 assert($('llm-mechanism-label').hidden,'mechanism tabs initially hidden');
 assert(![...$('llm-topic').options].some(option=>['bounds','backjump','numeric','underapprox'].includes(option.value)),'no separate top-level mechanisms');
 $('llm-topic').value='mechanisms';$('llm-topic').dispatchEvent(new Event('change'));
 assert(!$('llm-mechanism-label').hidden,'mechanism tabs shown next to topic');
 assert(document.querySelector('[data-mechanism="all"]').getAttribute('aria-checked')==='true','all mechanisms is default tab');
 const numericTab=document.querySelector('[data-mechanism="numeric"]');numericTab.click();
 assert(numericTab.getAttribute('aria-checked')==='true' && $('llm-layer-label').hidden,'numeric tab active without neuron filter');
 numericTab.dispatchEvent(new KeyboardEvent('keydown',{key:'ArrowRight',bubbles:true}));
 assert(document.querySelector('[data-mechanism="underapprox"]').getAttribute('aria-checked')==='true','tabs support arrow keys');
 const treeButton=document.querySelector('#tree-fixture h2 .llm-topic-button');
 assert(treeButton,'tree heading has an LLM analysis action');
 treeButton.click();
 assert($('llm-topic').value==='tree','tree action selects its own analysis topic');
 assert(!$('llm-layer-label').hidden && !$('llm-neuron-label').hidden,'tree layer/neuron selectors');
 $('llm-topic').value='simplex';$('llm-topic').dispatchEvent(new Event('change'));
 $('llm-call').value='simplex-1';$('llm-call').dispatchEvent(new Event('input'));
 $('llm-consent').checked=true;$('llm-consent').dispatchEvent(new Event('change'));$('llm-start').click();
 await wait(()=>$('llm-result').textContent.includes('선택 항목 분석'),'analysis');
 assert($('llm-result').querySelector('h2')?.textContent==='선택 항목 분석','rendered Markdown heading');
 assert($('llm-result').querySelector('strong')?.textContent==='관찰','rendered emphasis');
 assert($('llm-result').querySelector('.llm-citation')?.textContent==='솔버 진행 · 기록 3','human-readable evidence chip');
 assert($('llm-result-tag').textContent==='분석 완료','completion badge');
 assert(!$('llm-budget') && !$('llm-budget-meter') && !$('llm-progress').textContent.includes('API 1회'),'internal call count must not be displayed');
 assert($('llm-quota-link').href==='https://aistudio.google.com/rate-limit','provider quota link');
 $('llm-quota-link').click();
 await wait(()=>$('llm-quota-link').parentElement.querySelector('[data-quota-status]').textContent.includes('기본 브라우저에서 열었습니다'),'quota link');
 $('llm-topic').value='mechanisms';$('llm-topic').dispatchEvent(new Event('change'));
 document.querySelector('[data-mechanism="bounds"]').click();
 await wait(()=>$('llm-result').textContent.includes('저장된 분석이 없습니다'),'selection');
 assert(!$('llm-result').textContent.includes('선택 항목 분석'),'no other-topic result leak');
 assert($('llm-consent').checked&&!$('llm-start').disabled,'selection change preserves consent');
 $('llm-layer').value='1';$('llm-layer').dispatchEvent(new Event('change'));
 assert($('llm-neuron').options.length===3,'actual neurons in selected layer');
 $('llm-consent').checked=false;$('llm-consent').dispatchEvent(new Event('change'));
 $('llm-topic').value='tree';$('llm-topic').dispatchEvent(new Event('change'));
 assert(!$('llm-consent').checked&&$('llm-start').disabled,'manual opt-out stays unchecked');
 assert(!document.querySelector('#llm-result script'),'text-only rendering');
 out.textContent='PASS';
}catch(e){out.textContent='FAIL '+e.message;}document.body.appendChild(out);
})();</script>'''
        (self.run / "relu_realtime.html").write_text('<!doctype html><meta charset="utf-8"><main>'
            '<section id="tree-fixture"><h2>Interactive ReLU Search Tree</h2><iframe id="relu-tree-dashboard"></iframe></section>'
            '<section><h2>Simplex</h2><img id="simplex-dashboard"></section>'
            '<section><h2>Mechanisms</h2><iframe id="solver-mechanisms-dashboard"></iframe></section>'
            '</main>' + harness, encoding="utf-8")
        with patch("Automation.LogAnalysisServer.webbrowser.open", return_value=True) as opened:
            result = subprocess.run([chromium_path(), "--headless", "--no-sandbox", "--disable-gpu", "--no-first-run",
                                     f"--user-data-dir={self.root / 'profile'}", "--virtual-time-budget=18000", "--dump-dom",
                                     f"http://127.0.0.1:{self.server.server_port}{self.base}"], capture_output=True, timeout=30)
        dom = result.stdout.decode("utf-8", errors="replace")
        match = re.search(r'<pre id="test-result">([^<]+)</pre>', dom)
        self.assertIsNotNone(match, dom[-1000:])
        self.assertEqual(match[1], "PASS")
        opened.assert_called_once_with("https://aistudio.google.com/rate-limit", new=2)


if __name__ == "__main__":
    unittest.main()
