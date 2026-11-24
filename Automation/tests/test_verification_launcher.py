import gzip
from io import BytesIO
import json
from http.client import HTTPConnection
from pathlib import Path
import re
import subprocess
from tempfile import TemporaryDirectory
from threading import Event, Thread
from time import monotonic, sleep
import unittest
from unittest.mock import Mock, patch

from Automation.LogAnalysisServer import make_server
from Automation.LogAnalysisIndex import live_run_fingerprint
from Automation.BatchLogSnapshot import summarize_logs
from Automation.tests.test_reactive_tree_browser import chromium_path
from Automation.VerificationLauncher import LauncherPage, VerificationLauncher, _batch_metrics, _worker


class VerificationLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.workspace = Path(self.temp.name)
        self.root = self.workspace / "benchmarks"
        self.group = self.root / "acas"
        self.group.mkdir(parents=True)
        for name in ("network.onnx", "network.onnx.gz", "prop_3.vnnlib.gz"):
            (self.group / name).touch()
        with patch("Automation.AnalyzeLogs.load_gemini_api_key", return_value=""):
            self.app = LauncherPage()
        self.app.verification = VerificationLauncher(self.workspace, self.root)
        self.server = make_server(self.app)
        self.thread = Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self._close)

    def _close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(3)
        if self.app.verification.current:
            self.app.verification._close_log(self.app.verification.current["log"])

    def request(self, route, data=None, token=None, *, post=False):
        connection = HTTPConnection("127.0.0.1", self.server.server_port, timeout=5)
        try:
            headers = {} if data is None else {"Content-Type": "application/json", "X-Analysis-Token": token or ""}
            connection.request("POST" if post or data is not None else "GET", f"/s/{self.app.token}/{route}",
                               body=None if data is None else json.dumps(data), headers=headers)
            response = connection.getresponse()
            return response.status, response.read()
        finally:
            connection.close()

    @unittest.skipUnless(chromium_path(), "Chromium is not installed")
    def test_browser_can_ask_about_structure_without_running_verification(self):
        from Automation.AnalyzeLogs import DEFAULT_MODEL, GeminiClient

        self.app.batch_client = Mock(key="test-key", model=DEFAULT_MODEL)
        client = GeminiClient("test-key")
        saved_run = self.workspace / "visualization" / "outputs" / "old-xor" / "result.json"
        saved_run.parent.mkdir(parents=True)
        saved_run.write_text(json.dumps({"status": "COUNTEREXAMPLE", "model": {"name": "xor_network.bin"},
                                         "solver": {"status": "SAT"}}), encoding="utf-8")

        def answer(request):
            data = json.loads(request["contents"][0]["parts"][0]["text"])
            return {"candidates": [{"content": {"parts": [{"text": "답변 " + data["question"]}]}}]}

        harness = r'''<script>(async()=>{
const out=document.createElement('pre');out.id='overview-test';
async function wait(test){for(let i=0;i<200;i++){if(test())return;await new Promise(r=>setTimeout(r,40));}throw Error('timeout');}
try{
 const panel=document.getElementById('verify-batch-analysis');
 const model=panel.querySelector('[data-batch-model]'), analyze=panel.querySelector('[data-analyze]');
 const report=panel.querySelector('[data-report]'), question=panel.querySelector('textarea');
 const picker=panel.querySelector('[data-saved-runs-picker]'), savedRuns=picker.querySelector('select');
 await wait(()=>!panel.hidden&&!model.disabled);
 if(analyze.textContent!=='질문하기'||!analyze.disabled)throw Error('label or consent gate');
 await wait(()=>model.value);
 const consent=panel.querySelector('input[type=checkbox]');
 consent.checked=true;consent.dispatchEvent(new Event('change',{bubbles:true}));
 async function ask(text){
  await wait(()=>!analyze.disabled);
  question.value=text;analyze.click();
  await wait(()=>report.textContent.includes('답변 '+text));
 }
 picker.open=true;
 await wait(()=>savedRuns.options.length===1);
 savedRuns.selectedIndex=0;
 await ask('현재 검증기의 구조는?');
 savedRuns.dispatchEvent(new Event('change',{bubbles:true}));
 await ask('이 저장 실행을 설명해 주세요.');
 picker.open=false;
 await ask('그럼 GLPK의 역할은?');
 if(!document.querySelector('[data-runs]').hidden)throw Error('fake run appeared');
 out.textContent='PASS';
}catch(error){out.textContent='FAIL '+error.message;}document.body.append(out);
})();</script>'''
        page = self.app.dashboard().decode("utf-8")
        self.app.dashboard = lambda: page.replace("</body>", harness + "</body>").encode("utf-8")
        with patch("Automation.AnalyzeLogs.GeminiClient", return_value=client), \
                patch.object(client, "list_models", return_value={"analysis_models": [DEFAULT_MODEL]}), \
                patch.object(client, "_post_generation", side_effect=answer) as generate, \
                patch.object(self.app.verification, "_launch") as launch:
            browser = subprocess.run([
                chromium_path(), "--headless", "--no-sandbox", "--disable-gpu", "--no-first-run",
                f"--user-data-dir={self.workspace / 'overview-profile'}", "--virtual-time-budget=18000",
                "--dump-dom", f"http://127.0.0.1:{self.server.server_port}/s/{self.app.token}/",
            ], capture_output=True, timeout=30)
            launch.assert_not_called()
        dom = browser.stdout.decode("utf-8", errors="replace")
        match = re.search(r'<pre id="overview-test">([^<]+)</pre>', dom)
        self.assertIsNotNone(match, dom[-1000:])
        self.assertEqual(match[1], "PASS")
        sent = [json.loads(call.args[0]["contents"][0]["parts"][0]["text"]) for call in generate.call_args_list]
        self.assertEqual(sent[0]["runs"], [])
        self.assertEqual(sent[1]["runs"][0]["model"], "xor_network.bin")
        self.assertEqual(sent[1]["snapshot"]["run_source"], "saved_runs")
        self.assertNotIn("session_conversation", sent[1])
        self.assertEqual(sent[2]["runs"], [])
        self.assertTrue(sent[0]["project_structure"]["verification_flow"])
        self.assertEqual([turn["question"] for turn in sent[2]["session_conversation"]], ["현재 검증기의 구조는?"])

    def test_catalog_choice_is_local_and_starts_one_validated_process(self):
        status, page = self.request("")
        self.assertEqual(status, 200)
        self.assertIn(b'id="verify-picker"', page)
        _, raw = self.request("api/verify-catalog")
        catalog = json.loads(raw)
        self.assertIn("acas", {group["id"] for group in catalog["groups"]})
        self.assertEqual(self.request("api/verify-catalog?group=..")[0], 400)
        _, raw = self.request("api/verify-catalog?group=acas")
        choices = json.loads(raw)
        self.assertEqual([item["label"] for item in choices["models"]], ["network.onnx"])
        selection = {"group": "acas", "model": choices["models"][0]["id"],
                     "property": choices["properties"][0]["id"], "engine": "glpk", "timeout": 670}
        self.assertEqual(self.request("api/verify-start", post=True)[0], 403)
        self.assertEqual(self.request("api/verify-start", {**selection, "model": "../network.onnx"}, self.app.csrf)[0], 400)
        bad_settings = {**selection, "settings": {"panels": []}}
        self.assertEqual(self.request("api/verify-start", bad_settings, self.app.csrf)[0], 400)
        with patch("Automation.VerificationLauncher.subprocess.Popen") as launch:
            launch.return_value.poll.return_value = None
            status, raw = self.request("api/verify-start", selection, self.app.csrf)
            self.assertEqual(status, 202)
            job_id = json.loads(raw)["id"]
            self.assertEqual(json.loads(self.request("api/verify-job?id=" + job_id)[1])["status"], "running")
            self.assertEqual(self.request("api/verify-start", selection, self.app.csrf)[0], 409)
            argv = launch.call_args.args[0]
            self.assertEqual(argv[-4:], [str(self.group / "network.onnx"), str(self.group / "prop_3.vnnlib.gz"),
                                         "glpk", "670"])
            self.assertEqual(launch.call_args.kwargs["env"]["PYTHONHASHSEED"], "0")
            launch.assert_called_once()

    def test_launcher_forwards_worker_settings(self):
        choices = self.app.verification.catalog("acas")
        settings = {"visualization_mode": "off", "panels": [], "max_rounds": 10, "fix_python_hash_seed": False}
        selection = {"group": "acas", "model": choices["models"][0]["id"],
                     "property": choices["properties"][0]["id"], "engine": "glpk",
                     "timeout": 30, "settings": settings}
        with patch.dict("os.environ", {"PYTHONHASHSEED": "17"}), \
                patch("Automation.VerificationLauncher.subprocess.Popen") as launch:
            self.app.verification.start(selection)
        command = launch.call_args.args[0]
        self.assertEqual(json.loads(command[command.index("--worker-settings") + 1]), settings)
        self.assertNotIn("PYTHONHASHSEED", launch.call_args.kwargs["env"])

    def test_session_history_tracks_single_and_batch_without_visualization(self):
        files = self.app.verification.catalog("acas")
        selection = {"group": "acas", "model": files["models"][0]["id"],
                     "property": files["properties"][0]["id"], "engine": "native",
                     "timeout": 30, "settings": {"visualization_mode": "off"}}
        single = Mock()
        single.poll.return_value = None
        batch = Mock()
        batch.wait.return_value = 0
        decided_at = "2026-11-23T14:39:00+09:00"
        single_log = BytesIO(("VERIFICATION_RESULT " + json.dumps({
            "verdict": "VERIFIED", "verification_status": "UNSAT", "decided_at": decided_at,
            "metrics": {"solver": {"elapsed_seconds": 3.24}},
        })).encode())
        single_log.state_directory = TemporaryDirectory()
        launch_results = [
            (single, single_log),
            (batch, BytesIO(b'VERIFICATION_RESULT {"verdict":"UNKNOWN","metrics":{"solver":{"reason":"TIMEOUT"}}}')),
        ]

        def history():
            code, raw = self.request("api/verify-history")
            self.assertEqual(code, 200)
            return json.loads(raw)["runs"]

        self.assertEqual(history(), [])
        with patch.object(self.app.verification, "_launch", side_effect=launch_results):
            self.assertEqual(self.request("api/verify-start", selection, self.app.csrf)[0], 202)
            first = history()[0]
            self.assertEqual(first["status"], "running")
            self.assertEqual((first["model"], first["property"]), ("network.onnx", "prop_3.vnnlib.gz"))
            self.assertGreater(history()[0]["elapsed_seconds"], first["elapsed_seconds"])
            Path(single_log.state_directory.name, "state.json").write_text(json.dumps({
                "phase": "finalizing", "verification_seconds": 3.24,
            }), encoding="utf-8")
            pending = history()[0]
            self.assertTrue(pending["finalizing"])
            self.assertEqual(pending["elapsed_seconds"], 3.24)
            single.poll.return_value = 0
            complete = history()[0]
            self.assertEqual(complete["result"], "UNSAT")
            self.assertEqual(complete["decided_at"], decided_at)
            self.assertEqual(complete["elapsed_seconds"], pending["elapsed_seconds"])
            self.assertEqual(history()[0]["elapsed_seconds"], complete["elapsed_seconds"])
            self.assertEqual(self.request("api/verify-batch", {"runs": [selection]}, self.app.csrf)[0], 202)
            self.app.verification.batch_thread.join(5)
        self.assertEqual([(run["number"], run["result"]) for run in history()], [(1, "UNSAT"), (2, "TIMEOUT")])
        reopened = VerificationLauncher(self.workspace, self.root, restore_preferences=False)
        self.assertEqual(reopened.session_history(), {"runs": []})

    def test_analysis_compares_single_session_and_saved_runs_with_or_without_panels(self):
        verification = self.app.verification
        files = verification.catalog("acas")
        selection = {"group": "acas", "model": files["models"][0]["id"],
                     "property": files["properties"][0]["id"], "engine": "glpk", "timeout": 30}
        launches = []
        for mode in ("off", "realtime"):
            directory = self.workspace / "visualization" / "outputs" / mode
            directory.mkdir(parents=True)
            result_path = directory / "result.json"
            result = {"status": "COUNTEREXAMPLE", "model": {"name": "network.onnx"},
                      "vnnlib": {"property_name": "prop_3"}, "visualization": {"mode": mode},
                      "solver": {"status": "SAT", "rounds": 1, "round_details": [{"round": 1, "result": "THEORY_SAT"}],
                                 "theory_stats": {"LP_CALLS": 3, "RELU_SPLITS": 1, "GLPK_PIVOTS": 6}}}
            result_path.write_text(json.dumps(result), encoding="utf-8")
            if mode == "realtime":
                (directory / "solver_progress.jsonl").write_text(
                    '{"event":"round_end","round":1,"result":"THEORY_SAT"}\n', encoding="utf-8")
            outcome = {"verdict": result["status"], "result_path": str(result_path), "metrics": _batch_metrics(result)}
            process = Mock()
            process.poll.return_value = 0
            launches.append((process, BytesIO(("VERIFICATION_RESULT " + json.dumps(outcome)).encode())))

        with patch.object(verification, "_launch", side_effect=launches):
            for mode in ("off", "realtime"):
                verification.start({**selection, "settings": {"visualization_mode": mode}})

        def compare(launcher, body):
            client = Mock(model="gemini-3.5-flash-lite", requests_made=1)
            client.summarize_verification_batch.return_value = "두 실행을 비교했습니다."
            launched = launcher.start_batch_analysis({**body, "consent": True}, lambda: client)
            deadline = monotonic() + 4
            while monotonic() < deadline:
                state = launcher.batch_status(launched["id"])
                if state["analysis"]["status"] != "running":
                    break
                sleep(0.02)
            self.assertEqual(state["analysis"]["status"], "complete")
            self.assertFalse(client.summarize_verification_batch.call_args.kwargs["snapshot"].get("overview"))
            runs = {run["metrics"]["configuration"]["visualization_mode"]: run
                    for run in client.summarize_verification_batch.call_args.args[0]}
            self.assertEqual(set(runs), {"off", "realtime"})
            self.assertEqual(runs["off"]["metrics"]["recent_dpll_rounds"][0]["result"], "THEORY_SAT")
            self.assertEqual(runs["off"]["metrics"]["relu_splits"]["total_split_events"], 1)
            self.assertEqual(runs["off"]["metrics"]["reluplex"]["LP_CALLS"], 3)
            self.assertEqual(runs["realtime"]["recorded_logs"][0]["recorded_events"], {"round_end": 1})
            self.assertNotIn("result_path", runs["realtime"])

        compare(verification, {})
        code, raw = self.request("api/verify-saved-runs")
        self.assertEqual(code, 200)
        saved = [item["id"] for item in json.loads(raw)["runs"]]
        self.assertCountEqual(saved, ["off/result.json", "realtime/result.json"])
        reopened = VerificationLauncher(self.workspace, self.root, restore_preferences=False)
        self.assertEqual(reopened.session_history(), {"runs": []})
        compare(reopened, {"saved_runs": saved})
        with self.assertRaises(ValueError):
            reopened.start_batch_analysis({"consent": True, "saved_runs": ["../../result.json"]}, Mock())

    def test_catalog_keeps_glpk_selected_when_import_is_blocked(self):
        with patch("Automation.VerificationLauncher._glpk_import_error", return_value="DLL blocked"):
            code, raw = self.request("api/verify-catalog")
        self.assertEqual(code, 200)
        data = json.loads(raw)
        self.assertEqual(data["engines"], ["glpk", "native", "z3"])
        self.assertEqual(data["glpk_error"], "DLL blocked")
        with patch("Automation.VerificationLauncher._glpk_import_error", return_value=None):
            self.assertEqual(json.loads(self.request("api/verify-catalog")[1])["engines"][0], "glpk")

    def test_worker_load_error_finishes_dashboard_and_run_manifest(self):
        run = self.workspace / "visualization" / "outputs" / "failed-run"
        run.mkdir(parents=True)
        dashboard = run / "relu_realtime.html"
        dashboard.touch()
        manifest = run / "run_info.json"
        manifest.write_text('{"status":"RUNNING"}', encoding="utf-8")
        log = BytesIO((
            "VERIFICATION_OUTPUT " + json.dumps(str(run)) + "\n"
            "ImportError: DLL load failed while importing _swiglpk: DLL blocked\n"
        ).encode())
        launcher = self.app.verification
        with patch.object(launcher, "_worker_state", return_value={"dashboard": dashboard}):
            outcome = launcher._finish_worker(1, log, stopped=False)
        self.assertEqual(outcome["diagnostic"]["code"], "GLPK_LOAD_ERROR")
        self.assertEqual(json.loads(manifest.read_text(encoding="utf-8"))["status"], "ERROR")
        status_script = dashboard.with_suffix(".status.js").read_text(encoding="utf-8")
        self.assertIn('"ERROR"', status_script)
        self.assertIn("DLL blocked", status_script)

    def test_dashboard_can_be_reopened_during_verification(self):
        files = self.app.verification.catalog("acas")
        selection = {"group": "acas", "model": files["models"][0]["id"],
                     "property": files["properties"][0]["id"], "engine": "native",
                     "timeout": 30}
        dashboard = self.workspace / "visualization" / "outputs" / "run" / "relu_realtime.html"
        dashboard.parent.mkdir(parents=True)
        dashboard.touch()
        with patch("Automation.VerificationLauncher.subprocess.Popen") as launch, \
                patch("Automation.LogAnalysisLauncher.open_dashboard_with_analysis",
                      return_value={"browser_opened": True}) as open_page:
            launch.return_value.poll.return_value = None
            job = self.app.verification.start(selection)
            route = "api/verify-job?id=" + job["id"]
            self.assertFalse(json.loads(self.request(route)[1])["dashboard_ready"])
            command = launch.call_args.args[0]
            state_path = Path(command[command.index("--worker-state") + 1])
            state_path.write_text(json.dumps({"phase": "running", "dashboard": str(dashboard)}), encoding="utf-8")
            self.assertTrue(json.loads(self.request(route)[1])["dashboard_ready"])
            self.assertEqual(self.request("api/verify-open-dashboard", {"id": job["id"]}, self.app.csrf)[0], 200)
            open_page.assert_called_once_with(dashboard.parent, dashboard)
            manifest = dashboard.parent / "run_info.json"
            manifest.write_text('{"status":"RUNNING"}', encoding="utf-8")
            log = self.app.verification.current["log"]
            log.write(("VERIFICATION_OUTPUT " + json.dumps(str(dashboard.parent)) + "\n").encode())
            self.assertEqual(self.request("api/verify-stop", {"id": job["id"]}, self.app.csrf)[0], 202)
            launch.return_value.terminate.assert_called_once()
            launch.return_value.poll.return_value = 1
            self.assertEqual(json.loads(self.request(route)[1])["status"], "cancelled")
            self.assertEqual(json.loads(manifest.read_text(encoding="utf-8"))["status"], "CANCELLED")
            self.assertIn("CANCELLED", dashboard.with_suffix(".status.js").read_text(encoding="utf-8"))

    def test_stopping_a_batch_keeps_completed_results_and_skips_pending_runs(self):
        files = self.app.verification.catalog("acas")
        selection = {"group": "acas", "model": files["models"][0]["id"],
                     "property": files["properties"][0]["id"], "engine": "native", "timeout": 30}
        started, terminated = Event(), Event()
        finished = Mock()
        finished.wait.return_value = 0
        active = Mock()
        active.poll.return_value = None
        active.terminate.side_effect = terminated.set

        def wait_for_stop():
            started.set()
            if not terminated.wait(5):
                raise OSError("Stop not received")
            return 1

        active.wait.side_effect = wait_for_stop
        with patch.object(self.app.verification, "_launch", side_effect=[
                (finished, BytesIO(b'VERIFICATION_RESULT {"verdict":"VERIFIED"}')),
                (active, BytesIO())]) as launch:
            job_id = self.app.verification.start_batch({"runs": [selection] * 3})["id"]
            try:
                self.assertTrue(started.wait(5))
                self.assertEqual(self.request("api/verify-stop", post=True)[0], 403)
                self.assertEqual(self.request("api/verify-stop", {"id": job_id}, self.app.csrf)[0], 202)
            finally:
                terminated.set()
            deadline = monotonic() + 5
            while self.app.verification.batch_active and monotonic() < deadline:
                sleep(0.02)
            batch = self.app.verification.batch_status(job_id)
            self.assertEqual(batch["status"], "cancelled")
            self.assertEqual([run["status"] for run in batch["runs"]], ["complete", "cancelled", "cancelled"])
            self.assertEqual(batch["runs"][0]["verdict"], "VERIFIED")
            self.assertEqual(launch.call_count, 2)
            active.terminate.assert_called_once()
            history = self.app.verification.session_history()["runs"]
            self.assertEqual([run["status"] for run in history], ["complete", "cancelled"])

    def test_shutdown_reaps_the_active_worker_before_closing_server(self):
        choices = self.app.verification.catalog("acas")
        selection = {"group": "acas", "model": choices["models"][0]["id"],
                     "property": choices["properties"][0]["id"], "engine": "native", "timeout": 30}
        with patch("Automation.VerificationLauncher.subprocess.Popen") as launch:
            process = launch.return_value
            process.poll.return_value = None
            process.wait.return_value = 1
            self.app.verification.start(selection)
            self.assertEqual(self.request("api/verify-shutdown", post=True)[0], 403)
            code, payload = self.request("api/verify-shutdown", {}, self.app.csrf)
            self.assertEqual((code, json.loads(payload)), (200, {"status": "closed"}))
            self.assertTrue(self.app.shutdown_requested.wait(1))
            self.assertIsNone(self.app.verification.current)
            process.terminate.assert_called_once()
            process.wait.assert_called_once_with(timeout=5)

    def test_catalog_orders_property_and_network_numbers_numerically(self):
        for name in ("prop_10.vnnlib.gz", "prop_1.vnnlib.gz", "prop_2.vnnlib",
                     "network_2_10.onnx", "network_2_2.onnx", "network_2_1.onnx"):
            (self.group / name).touch()
        status, raw = self.request("api/verify-catalog?group=acas")
        self.assertEqual(status, 200)
        choices = json.loads(raw)
        self.assertEqual([entry["label"] for entry in choices["properties"]],
                         ["prop_1.vnnlib.gz", "prop_2.vnnlib", "prop_3.vnnlib.gz", "prop_10.vnnlib.gz"])
        self.assertEqual([entry["label"] for entry in choices["models"] if "_2_" in entry["label"]],
                         ["network_2_1.onnx", "network_2_2.onnx", "network_2_10.onnx"])

    def test_batch_runs_in_order_and_sends_all_summaries_only_after_consent(self):
        (self.group / "network_2.onnx").touch()
        _, raw = self.request("api/verify-catalog?group=acas")
        files = json.loads(raw)
        selection = {"group": "acas", "model": files["models"][0]["id"],
                     "property": files["properties"][0]["id"], "engine": "native", "timeout": 10}
        runs = [selection, {**selection, "model": files["models"][1]["id"]}]
        self.assertEqual(self.request("api/verify-batch", post=True)[0], 403)
        token = self.app.csrf
        self.assertEqual(self.request("api/verify-batch", {"runs": [selection, {**selection, "model": "bad"}]}, token)[0], 400)

        waiting, release = Event(), Event()
        def fail_first():
            waiting.set()
            if not release.wait(5):
                raise OSError("Test worker waited too long")
            return 1

        first = Mock()
        first.wait.side_effect = fail_first
        second = Mock()
        second.wait.return_value = 0
        result = {"verdict": "VERIFIED", "result_path": "LOCAL_ONLY",
                  "metrics": {"simplex": {"simplex_calls": 2}}}
        result_log = BytesIO(("VERIFICATION_RESULT " + json.dumps(result)).encode())
        failed_dir = self.workspace / "visualization" / "outputs" / "failed-run"
        failed_dir.mkdir(parents=True)
        (failed_dir / "run_info.json").write_text(json.dumps({
            "phase": "FINALIZING",
            "solver_result": {"status": "UNSAT", "rounds": 2, "theory_stats": {"LP_CALLS": 51}},
        }), encoding="utf-8")
        error_log = BytesIO((
            "VERIFICATION_OUTPUT " + json.dumps(str(failed_dir)) + "\n"
            "glp_free: memory allocation error\n"
            "Error detected in file ..\\src\\env\\alloc.c at line 70\n"
        ).encode())
        fake_client = Mock(model="gemini-3.5-flash-lite", requests_made=1)
        fake_client.summarize_verification_batch.return_value = "검증 결과 비교"
        self.app.batch_client = fake_client

        def wait_for(route, field):
            deadline = monotonic() + 5
            while monotonic() < deadline:
                _, payload = self.request(route)
                state = json.loads(payload)
                for part in field:
                    state = state[part]
                if state == "complete":
                    return json.loads(payload)
                sleep(0.02)
            self.fail("Batch did not complete")

        with patch.object(self.app.verification, "_launch", side_effect=[
                (first, error_log), (second, result_log)]) as launch, \
                patch("Automation.AnalyzeLogs.GeminiClient", return_value=fake_client):
            code, payload = self.request("api/verify-batch", {"runs": runs}, token)
            self.assertEqual(code, 202, payload)
            batch_id = json.loads(payload)["id"]
            route = "api/verify-batch?id=" + batch_id
            question = {"id": batch_id, "question": "차이를 설명해 주세요.", "consent": True}
            try:
                self.assertTrue(waiting.wait(5))
                self.assertEqual(launch.call_count, 1)
                self.assertEqual(self.request("api/verify-start", selection, token)[0], 409)
                self.assertEqual(self.request("api/verify-batch-analysis", question, token)[0], 409)
            finally:
                release.set()
            state = wait_for(route, ["status"])
            self.assertEqual([run["status"] for run in state["runs"]], ["error", "complete"])
            self.assertEqual([item.args[0].id for item in launch.call_args_list], [run["model"] for run in runs])
            self.assertEqual(state["runs"][1]["verdict"], "VERIFIED")
            self.assertEqual(self.request("api/verify-batch-analysis", {**question, "consent": False}, token)[0], 400)
            fake_client.summarize_verification_batch.assert_not_called()
            code, accepted = self.request("api/verify-batch-analysis", question, token)
            self.assertEqual(code, 202)
            self.assertEqual(json.loads(accepted)["request_budget"], {"used": 0, "limit": 2, "remaining": 2})
            state = wait_for(route, ["analysis", "status"])
            self.assertEqual(state["analysis"]["request_budget"], {"used": 1, "limit": 2, "remaining": 1})
            sent, asked = fake_client.summarize_verification_batch.call_args.args
            self.assertEqual(asked, question["question"])
            self.assertEqual([item["number"] for item in sent], [1, 2])
            self.assertNotIn("error", sent[0])
            self.assertNotIn("run_manifest_path", sent[0])
            self.assertEqual(sent[0]["diagnostic"], {
                "code": "GLPK_ALLOCATOR_ACCOUNTING_ERROR", "exit_code": 1, "last_saved_phase": "FINALIZING",
            })
            self.assertEqual(sent[0]["solver_checkpoint"]["solver"]["status"], "UNSAT")
            self.assertEqual(sent[0]["solver_checkpoint"]["reluplex"]["LP_CALLS"], 51)
            self.assertNotIn("result_path", sent[1])
            self.assertEqual(sent[1]["metrics"]["simplex"]["simplex_calls"], 2)
            archived = self.workspace / "Automation" / "Results" / "verification_batches" / batch_id / "batch.json"
            self.assertEqual(json.loads(archived.read_text(encoding="utf-8"))["analysis"], state["analysis"])
            reopened = VerificationLauncher(self.workspace, self.root)
            self.assertEqual(reopened.saved_analyses(batch_id)["analysis"]["report"], "검증 결과 비교")
            self.assertEqual(reopened.saved_analyses()["analyses"][0]["id"], batch_id)
            self.assertEqual(self.request("api/verify-saved-analyses?id=..")[0], 400)

    def test_picker_preferences_persist_across_server_instances(self):
        _, raw = self.request("api/verify-catalog?group=acas")
        choices = json.loads(raw)
        selection = {"group": "acas", "model": choices["models"][0]["id"],
                     "property": choices["properties"][0]["id"], "engine": "native", "timeout": 123}
        self.assertEqual(self.request("api/verify-preferences", selection, self.app.csrf)[0], 200)
        self.assertEqual(self.request("api/verify-preferences", {**selection, "model": "b" * 24}, self.app.csrf)[0], 400)
        reopened = VerificationLauncher(self.workspace, self.root)
        self.assertEqual(reopened.catalog()["preferences"], selection)
        self.assertIsNone(reopened.current)

    def test_batch_can_analyze_flushed_logs_while_the_first_run_is_still_running(self):
        files = self.app.verification.catalog("acas")
        selection = {"group": "acas", "model": files["models"][0]["id"],
                     "property": files["properties"][0]["id"], "engine": "native", "timeout": 30}
        directory = self.workspace / "visualization" / "outputs" / "active"
        directory.mkdir(parents=True)
        progress = directory / "solver_progress.jsonl"
        progress.write_text('{"event":"round_start","round":1}\n{"event":"simplex_end","call_number":1}', encoding="utf-8")
        log = BytesIO(b'VERIFICATION_RESULT {"verdict":"VERIFIED"}')
        log.state_directory = TemporaryDirectory()
        Path(log.state_directory.name, "state.json").write_text(
            json.dumps({"phase": "running", "directory": str(directory)}), encoding="utf-8")
        waiting, release, captured, parse_snapshot = Event(), Event(), Event(), Event()
        active = Mock()

        def wait_for_verification():
            waiting.set()
            release.wait(5)
            return 0

        def read_snapshot(limits):
            captured.set()
            parse_snapshot.wait(5)
            return summarize_logs(limits)

        active.wait.side_effect = wait_for_verification
        second = Mock()
        second.wait.return_value = 0
        client = Mock(model="gemini-3.5-flash-lite", requests_made=1)
        client.summarize_verification_batch.return_value = "부분 분석"
        self.app.batch_client = client
        with patch.object(self.app.verification, "_launch", side_effect=[
                (active, log), (second, BytesIO(b'VERIFICATION_RESULT {"verdict":"VERIFIED"}'))]), \
                patch("Automation.VerificationLauncher.summarize_logs", side_effect=read_snapshot), \
                patch("Automation.AnalyzeLogs.GeminiClient", return_value=client):
            batch_id = self.app.verification.start_batch({"runs": [selection, selection]})["id"]
            try:
                self.assertTrue(waiting.wait(5))
                self.assertTrue(self.app.verification.batch_status(batch_id)["analysis_ready"])
                code, _ = self.request("api/verify-batch-analysis", {"id": batch_id, "consent": True}, self.app.csrf)
                self.assertEqual(code, 202)
                self.assertTrue(captured.wait(5))
                with progress.open("a", encoding="utf-8") as stream:
                    stream.write('\n{"event":"round_end","round":1,"result":"UNSAT"}\n')
                parse_snapshot.set()
                deadline = monotonic() + 4
                while monotonic() < deadline:
                    state = self.app.verification.batch_status(batch_id)
                    if state["analysis"]["status"] == "complete":
                        break
                    sleep(0.02)
                self.assertEqual(state["analysis"]["status"], "complete")
                self.assertEqual(state["status"], "running")
                self.assertTrue(state["analysis"]["partial"])
                sent = client.summarize_verification_batch.call_args.args[0]
                self.assertEqual([run["status"] for run in sent], ["running", "waiting"])
                self.assertEqual(sent[0]["recorded_logs"][0]["recorded_events"], {"round_start": 1})
                self.assertNotIn("recorded_logs", sent[1])
            finally:
                parse_snapshot.set()
                release.set()
                self.app.verification.batch_thread.join(5)

    def test_reopened_launcher_analyzes_cancelled_logs_without_a_final_result(self):
        directory = self.workspace / "visualization" / "outputs" / "cancelled"
        directory.mkdir(parents=True)
        manifest = directory / "run_info.json"
        manifest.write_text('{"status":"CANCELLED"}', encoding="utf-8")
        progress = directory / "solver_progress.jsonl"
        records = [
            {"event": "simplex_start", "call_id": "simplex-1", "call_number": 1, "depth": 2},
            {"event": "simplex_end", "call_id": "simplex-1", "duration_seconds": 0.2},
            {"event": "simplex_start", "call_id": "simplex-2", "call_number": 2, "depth": 7},
            {"event": "simplex_end", "call_id": "simplex-2", "duration_seconds": 2.5, "iterations": 0},
        ]
        progress.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
        batch_id = "b" * 24
        self.app.verification._save_batch({
            "id": batch_id, "status": "cancelled", "analysis": {
                "status": "complete", "question": "이전 세션의 질문", "report": "이전 세션의 답변",
            },
            "runs": [{"number": 1, "status": "cancelled", "run_manifest_path": str(manifest)}],
        })
        self.assertIsNone(self.app.verification.catalog()["batch_id"])
        original_id = batch_id
        code, data = self.request("api/verify-followup", {"id": original_id}, self.app.csrf)
        self.assertEqual(code, 200)
        batch_id = json.loads(data)["id"]
        self.assertNotEqual(batch_id, original_id)
        self.assertEqual(self.app.verification.catalog()["batch_id"], batch_id)
        self.assertTrue(self.app.verification.batch_status(batch_id)["analysis_ready"])
        client = Mock(model="gemini-3.5-flash-lite", requests_made=1)
        client.summarize_verification_batch.return_value = "저장된 기록을 분석했습니다."
        def ask(question):
            self.app.verification.start_batch_analysis(
                {"id": batch_id, "consent": True, "question": question}, lambda: client)
            deadline = monotonic() + 4
            while monotonic() < deadline:
                state = self.app.verification.batch_status(batch_id)
                if state["analysis"]["status"] != "running":
                    return state
                sleep(0.02)
            self.fail("Analysis did not complete")

        state = ask("중단 전 기록을 설명해 주세요.")
        self.assertEqual(state["analysis"]["status"], "complete")
        previous = {"question": "이전 세션의 질문", "answer": "이전 세션의 답변"}
        self.assertEqual(client.summarize_verification_batch.call_args.kwargs["conversation"], [previous])
        run = client.summarize_verification_batch.call_args.args[0][0]
        self.assertEqual(run["status"], "cancelled")
        summary = run["recorded_logs"][0]
        self.assertEqual(summary["completed_simplex"]["calls"], 2)
        comparison = summary["simplex_timing_comparison"]
        self.assertEqual(comparison["later_sample"]["median_seconds"], 2.5)
        self.assertEqual(comparison["slowest_calls"][0]["iterations"], 0)
        self.assertEqual(comparison["slowest_calls"][0]["depth"], 7)
        self.assertFalse((directory / "result.json").exists())
        ask("그중 가장 오래 걸린 호출은요?")
        self.assertEqual(client.summarize_verification_batch.call_args.kwargs["conversation"], [
            previous,
            {"question": "중단 전 기록을 설명해 주세요.", "answer": "저장된 기록을 분석했습니다."},
        ])
        self.assertEqual(self.app.verification.saved_analyses(batch_id=original_id)["analysis"]["report"], "이전 세션의 답변")

    @unittest.skipUnless(chromium_path(), "Chromium is not installed")
    def test_browser_adds_two_entries_and_submits_in_order(self):
        (self.group / "network_2.onnx").touch()
        previous = self.app.verification.catalog("acas")
        self.app.verification.save_preferences({
            "group": "acas", "model": previous["models"][0]["id"],
            "property": previous["properties"][0]["id"], "engine": "native", "timeout": 12,
        })
        self.app.verification = VerificationLauncher(self.workspace, self.root, restore_preferences=False)
        client = Mock(model="gemini-3.6-flash", requests_made=1)
        client.list_models.return_value = {"analysis_models": ["gemini-3.5-flash-lite", "gemini-3.6-flash"]}
        client.summarize_verification_batch.return_value = "## 비교 결과\n\n**전체 2건**"
        self.app.batch_client = Mock(model="gemini-3.5-flash-lite", key="test-key")
        run_dir = self.workspace / "visualization" / "outputs" / "older_run"
        run_dir.mkdir(parents=True)
        (run_dir / "run_info.json").write_text('{"run_id":"older-id"}', encoding="utf-8")
        self.app.verification._save_batch({
            "id": "a" * 24, "runs": [
                {"number": 1, "result_path": str(run_dir / "result.json"),
                 "model": "network.onnx", "property": "prop3"},
                {"number": 2, "model": "network.onnx", "property": "prop_5.vnnlib.gz"},
                {"number": 3, "model": "third_network.onnx", "property": "prop_9.vnnlib.gz"},
            ],
            "analysis": {"status": "complete", "report": "이전 분석", "model": "gemini-3.5-flash-lite",
                         "question": "두 판정의 차이는 무엇인가요? 세부 수치도 설명해 주세요."},
        })
        individual = self.app.verification.analysis_directory / "20261116_090000_01234567"
        individual.mkdir(parents=True)
        (individual / "analysis.json").write_text(json.dumps({
            "status": "complete", "analysis": "개별 Simplex 답변", "model": "gemini-3.5-flash-lite",
            "question": "Simplex 호출이 왜 오래 걸렸나요? 구체적으로 설명해 주세요.",
            "scope": {"topic": "simplex", "round": 1}, "run_fingerprint": live_run_fingerprint(run_dir, "older-id"),
            "runs": [{"directory_name": "older_run", "run_context_only": {
                "model": {"source_path": "network.onnx"}, "vnnlib": {"property_name": "prop3"},
            }}],
        }), encoding="utf-8")
        document = self.app.dashboard().decode("utf-8")
        harness = r'''<script>(async()=>{
const out=document.createElement('pre');out.id='batch-test';
const wait=async fn=>{
 for(let i=0;i<150;i++){
  if(fn())return;
  await new Promise(resolve=>setTimeout(resolve,40));
 }
 throw Error('Wait timed out');
};
try{
 const form=document.querySelector('#verify-picker form'), f=form.elements;
 const add=form.querySelector('[data-queue]'), start=form.querySelector('[type=submit]');
 const select=(field,value)=>{
  field.value=value;
  field.dispatchEvent(new Event('change',{bubbles:true}));
 };
 const search=(name,text)=>{
   const input=form.querySelector('[data-catalog-search="'+name+'"]');
   input.value=text;
   input.dispatchEvent(new Event('input',{bubbles:true}));
   return input;
 };
 const enter=input=>input.dispatchEvent(new KeyboardEvent('keydown',{key:'Enter',bubbles:true,cancelable:true}));
 const matches=input=>[...document.getElementById(input.getAttribute('aria-controls')).querySelectorAll('[role="option"]')];
 await wait(()=>f.group.options.length>1);
 const panel=document.getElementById('verify-batch-analysis');
 if(f.group.value||f.model.value||f.property.value||panel.querySelector('textarea').value
   ||!form.querySelector('[data-runs]').hidden)throw Error('A fresh launcher restored an old selection or analysis');
 const questionTab=panel.querySelector('#verify-question-tab'), savedTab=panel.querySelector('#verify-saved-tab');
 if(panel.querySelector('#verify-question-panel').hidden||questionTab.getAttribute('aria-selected')!=='true'
   ||!panel.querySelector('#verify-saved-panel').hidden)throw Error('Question tab is not the default');
 if(f.timeout.value!=='600'||f.visualization_mode.value!=='realtime'||!f.fix_python_hash_seed.checked)
   throw Error('Verification defaults were not applied');
 const benchmarkSearch=search('group','AC_AS');
 if(matches(benchmarkSearch).length!==1||matches(benchmarkSearch)[0].textContent!=='acas')throw Error('Benchmark search did not filter');
 enter(benchmarkSearch);
 await wait(()=>f.model.options.length===3);
 const propertySearch=search('property','PROP3');
 if(!matches(propertySearch)[0]?.textContent.includes('prop_3'))throw Error('Property search did not match numeric suffix');
 enter(propertySearch);
 const modelCombo=search('model','no-match');
 if(matches(modelCombo).length)throw Error('Search showed nonmatching models');
 modelCombo.closest('.verify-lookup').querySelector('[data-catalog-toggle]').click();
 if(matches(modelCombo).length!==2)throw Error('Dropdown did not restore the full model list');
 matches(modelCombo)[0].click();
 if(modelCombo.value!=='network.onnx'||modelCombo.getAttribute('aria-expanded')!=='false')throw Error('Dropdown selection was not committed');
 select(f.visualization_mode,'feedback');
 const simplex=form.querySelector('[data-panel][value="simplex"]');
 simplex.checked=false;
 simplex.dispatchEvent(new Event('change',{bubbles:true}));
 select(f.max_rounds,'25');
 add.click();
 const modelSearch=search('model','Network 2');
 if(f.model.selectedOptions[0].text!=='network.onnx')throw Error('Searching discarded the selected model');
 enter(modelSearch);
 if(f.model.selectedOptions[0].text!=='network_2.onnx')throw Error('Model search did not select the result');
 select(f.visualization_mode,'off');
 select(f.max_rounds,'50');
 f.fix_python_hash_seed.click();
 add.click();
 if(form.querySelector('[data-draft]').children.length!==2)throw Error('Draft is missing entries');
 select(f.group,'');
 start.click();
 await wait(()=>form.querySelector('[data-runs]').children.length===2);
 const historyRows=form.querySelector('[data-history]').rows;
 await wait(()=>historyRows.length===2&&historyRows[1].cells[3].textContent==='VERIFIED');
 if(historyRows[0].cells[1].textContent!=='network.onnx'||historyRows[1].cells[1].textContent!=='network_2.onnx'
   ||!historyRows[1].cells[0].textContent||!historyRows[1].cells[4].textContent.endsWith('초'))
   throw Error('History must show timestamps and results for both modes in execution order');
 const models=panel.querySelector('[data-batch-model]');
 await wait(()=>!panel.querySelector('[data-new-analysis]').hidden&&models.options.length===2);
 const usage=panel.querySelector('.llm-status-details');
 if(usage.open||usage.querySelector('progress')||!usage.querySelector('#llm-batch-quota-link'))
   throw Error('The model usage section must start collapsed with the provider quota link');
 usage.querySelector('summary').click();
 if(!usage.open)throw Error('The model usage section did not open');
 usage.querySelector('#llm-batch-quota-link').click();
 await wait(()=>usage.querySelector('[data-quota-status]').textContent.includes('기본 브라우저에서 열었습니다'));
 select(models,'gemini-3.6-flash');
 const effort=panel.querySelector('[data-batch-effort]');
 if(![...effort.options].some(option=>option.value==='minimal'))throw Error('Model effort choices are missing');
 select(effort,'high');
 const analyze=panel.querySelector('[data-analyze]'), copy=panel.querySelector('[data-copy]');
 if(!analyze.disabled||!copy.disabled)throw Error('Analysis controls should start disabled');
 select(panel.querySelector('textarea'),'두 실행을 비교해 주세요.');
 const consent=panel.querySelector('input');
 consent.checked=true;
 consent.dispatchEvent(new Event('change',{bubbles:true}));
 analyze.click();
 await wait(()=>panel.querySelector('[data-analysis-tag]').dataset.status==='complete');
 if(usage.querySelector('progress')||usage.textContent.includes('사용 1/1'))
   throw Error('The completed request usage must remain hidden');
 usage.querySelector('summary').click();
 if(usage.open)throw Error('The model usage section did not collapse');
 if(panel.querySelector('[data-report] h2')?.textContent!=='비교 결과'
   ||panel.querySelector('[data-report] strong')?.textContent!=='전체 2건'
   ||copy.disabled)throw Error('Analysis report was not rendered');
 let copied='';
 Object.defineProperty(navigator,'clipboard',{value:{writeText:async text=>{copied=text;}}});
 copy.click();
 await wait(()=>copied.startsWith('## 비교 결과'));
 const saved=panel.querySelector('[data-view-saved]');
 savedTab.click();
 if(!panel.querySelector('#verify-question-panel').hidden)throw Error('Saved tab did not open');
 const history=panel.querySelector('[data-saved-analyses]');
 const networks=panel.querySelector('[data-saved-network]');
 const properties=panel.querySelector('[data-saved-property]');
 await wait(()=>networks.options.length===3);
 select(networks,'third_network.onnx');
 if(properties.options.length!==1||properties.value!=='prop_9'||history.value!=='a'.repeat(24))
   throw Error('Batch appears outside its recorded model and property');
 select(networks,'network.onnx');
 if(properties.options.length!==2||properties.options[0].value!=='prop_3')throw Error('Properties were not grouped');
 select(properties,'prop_3');
 await wait(()=>history.options.length===3);
 select(history,'a'.repeat(24));
 if(!history.selectedOptions[0].textContent.endsWith(' · 두 판정의 차이는 무엇인가요?'))throw Error('Batch list must show only date and question');
 saved.click();
 await wait(()=>panel.querySelector('[data-report]').textContent==='이전 분석');
 if(panel.querySelector('[data-saved-title]').textContent!=='두 판정의 차이는 무엇인가요?')throw Error('Batch question title is missing');
 const individual=history.querySelector('option[value^="run:"]');
 if(!individual?.textContent.endsWith(' · Simplex 호출이 왜 오래 걸렸나요?'))throw Error('Individual list must show only date and question');
 select(history,individual.value);
 saved.click();
 await wait(()=>panel.querySelector('[data-report]').textContent==='개별 Simplex 답변');
 if(panel.querySelector('[data-saved-title]').textContent!=='Simplex 호출이 왜 오래 걸렸나요?')throw Error('Individual question title is missing');
 questionTab.click();
 if(panel.querySelector('textarea').value!=='두 실행을 비교해 주세요.'
   ||panel.querySelector('[data-report] h2')?.textContent!=='비교 결과')throw Error('Question view was overwritten');
 savedTab.click();
 const followup=panel.querySelector('[data-followup-saved]');
 if(followup.previousElementSibling!==saved)throw Error('Followup must sit next to view saved');
 followup.click();
 await wait(()=>!panel.querySelector('[data-followup-source]').hidden&&panel.querySelector('[data-report]').textContent==='개별 Simplex 답변');
 if(panel.querySelector('#verify-question-panel').hidden||panel.querySelector('textarea').value
   ||!panel.querySelector('[data-followup-source]').textContent.includes('Simplex 호출이 왜 오래 걸렸나요?'))
   throw Error('Followup did not open the selected saved question');
 await wait(()=>!analyze.disabled);
 select(panel.querySelector('textarea'),'그때의 병목을 더 설명해 주세요.');
 analyze.click();
 await wait(()=>panel.querySelector('[data-report] h2')?.textContent==='비교 결과');
 form.querySelector('[data-shutdown]').click();
 await wait(()=>form.querySelector('[role="status"]').textContent==='검증기가 종료되었습니다.');
 out.textContent='PASS';
}catch(error){out.textContent='FAIL '+error.message;}document.body.append(out);
})();</script>'''
        self.app.dashboard = lambda: document.replace("</body>", harness + "</body>").encode("utf-8")
        worker = Mock()
        worker.wait.return_value = 0
        result = b'VERIFICATION_RESULT {"verdict":"VERIFIED"}'
        with patch.object(self.app.verification, "_launch", side_effect=[
                (worker, BytesIO(result)), (worker, BytesIO(result))]) as launch, \
                patch("Automation.AnalyzeLogs.GeminiClient", return_value=client) as gemini, \
                patch("Automation.LogAnalysisServer.webbrowser.open", return_value=True) as opened:
            process = subprocess.run([
                chromium_path(), "--headless", "--no-sandbox", "--disable-gpu", "--no-first-run",
                f"--user-data-dir={self.workspace / 'batch-profile'}", "--virtual-time-budget=12000", "--dump-dom",
                f"http://127.0.0.1:{self.server.server_port}/s/{self.app.token}/",
            ], capture_output=True, timeout=30)
        dom = process.stdout.decode("utf-8", errors="replace")
        self.assertIn('<pre id="batch-test">PASS</pre>', dom)
        opened.assert_called_once_with("https://aistudio.google.com/rate-limit", new=2)
        self.assertEqual([call.args[0].label for call in launch.call_args_list], ["network.onnx", "network_2.onnx"])
        settings = [call.args[4] for call in launch.call_args_list]
        self.assertEqual([item["visualization_mode"] for item in settings], ["feedback", "off"])
        self.assertEqual([item["max_rounds"] for item in settings], [25, 50])
        self.assertEqual([item.get("fix_python_hash_seed", True) for item in settings], [True, False])
        self.assertNotIn("simplex", settings[0]["panels"])
        calls = client.summarize_verification_batch.call_args_list
        self.assertEqual([call.args[1] for call in calls], ["두 실행을 비교해 주세요.", "그때의 병목을 더 설명해 주세요."])
        self.assertEqual(calls[0].kwargs["conversation"], [])
        self.assertEqual(calls[1].kwargs["conversation"], [
            {"question": "Simplex 호출이 왜 오래 걸렸나요? 구체적으로 설명해 주세요.", "answer": "개별 Simplex 답변"},
        ])
        gemini.assert_any_call("test-key", "gemini-3.6-flash", retries=0, thinking_level="high")

    @unittest.skipUnless(chromium_path(), "Chromium is not installed")
    def test_new_browser_server_restores_dropdowns_without_launching(self):
        _, raw = self.request("api/verify-catalog?group=acas")
        choices = json.loads(raw)
        selection = {"group": "acas", "model": choices["models"][0]["id"],
                     "property": choices["properties"][0]["id"], "engine": "z3", "timeout": 123,
                     "settings": {"visualization_mode": "realtime", "panels": ["simplex"],
                                  "open_dashboard": False, "max_recursion": 27}}
        self.assertEqual(self.request("api/verify-preferences", selection, self.app.csrf)[0], 200)
        other_app = LauncherPage()
        other_app.verification = VerificationLauncher(self.workspace, self.root)
        document = other_app.dashboard().decode("utf-8")
        harness = r'''<script>(async()=>{
const out=document.createElement('pre');out.id='restore-test';
try{
 const form=document.querySelector('#verify-picker form'), f=form.elements;
 const expected=EXPECTED_SELECTION;
 for(let i=0;i<260&&form.querySelector('[type=submit]').disabled;i++)await new Promise(r=>setTimeout(r,25));
 for(const name of ['group','model','property','engine','timeout']){
   if(f[name].value!==String(expected[name]))throw Error('wrong '+name+' '+f[name].value);
 }
 for(const name of ['group','model','property']){
   if(form.querySelector('[data-catalog-search="'+name+'"]').value!==f[name].selectedOptions[0].text)
     throw Error('Saved selection was not restored in the editable field');
 }
 const panels=[...form.querySelectorAll('[data-panel]:checked')].map(input=>input.value);
 if(panels.join(',')!=='simplex'||f.visualization_mode.value!=='realtime'
    ||f.namedItem('open_dashboard')||f.max_recursion.value!=='27')throw Error('wrong saved settings');
 out.textContent='PASS';
}catch(error){out.textContent='FAIL '+error.message;}document.body.append(out);
})();</script>'''.replace("EXPECTED_SELECTION", json.dumps(selection))
        other_app.dashboard = lambda: document.replace("</body>", harness + "</body>").encode("utf-8")
        with make_server(other_app) as server:
            thread = Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                process = subprocess.run([
                    chromium_path(), "--headless", "--no-sandbox", "--disable-gpu", "--no-first-run",
                    f"--user-data-dir={self.workspace / 'profile'}", "--virtual-time-budget=12000", "--dump-dom",
                    f"http://127.0.0.1:{server.server_port}/s/{other_app.token}/",
                ], capture_output=True, timeout=30)
            finally:
                server.shutdown()
                thread.join(3)
        dom = process.stdout.decode("utf-8", errors="replace")
        match = re.search(r'<pre id="restore-test">([^<]+)</pre>', dom)
        self.assertIsNotNone(match, dom[-1200:])
        self.assertEqual(match[1], "PASS")
        self.assertIsNone(other_app.verification.current)

    def test_compressed_model_is_staged_for_verification(self):
        model = self.group / "network.onnx.gz"
        model.write_bytes(gzip.compress(b"example ONNX bytes"))
        prop = self.group / "prop_3.vnnlib.gz"
        run = Mock(side_effect=lambda path, *a, **kw: {
            "status": "VERIFIED", "contents": Path(path).read_bytes(),
        })
        with patch("Automation.AutoVerify.run_vnnlib_verification", run):
            result = _worker(model, prop, "glpk", 670)
        self.assertEqual(result["contents"], b"example ONNX bytes")
        args, options = run.call_args
        self.assertEqual(Path(args[0]).name, "network.onnx")
        self.assertEqual(args[1], str(prop))
        self.assertEqual(options["visualization_mode"], "realtime")
        # 기본값 = 전남대_reluplex 최고 설정의 기법 ①②③ (엔진 쪽은 Reluplex 기본값 glpk-core-basis)
        from Reluplex import feature_options
        self.assertEqual(options["reluplex_options"],
                         {"lp_engine": "glpk", **feature_options("bound,backjump,numeric")})
        self.assertFalse(Path(args[0]).exists())

    def test_worker_automatically_opens_only_realtime_modes(self):
        for mode, opens in (("realtime", True), ("both", True), ("feedback", False), ("off", False)):
            with self.subTest(mode=mode), patch(
                    "Automation.AutoVerify.run_vnnlib_verification", return_value={"status": "VERIFIED"}) as run:
                _worker(self.group / "network.onnx", self.group / "prop_3.vnnlib.gz", "native", 12,
                        settings={"visualization_mode": mode, "open_dashboard": False})
                self.assertIs(run.call_args.kwargs["open_realtime_view"], opens)

    def test_managed_worker_hands_dashboard_to_owning_launcher(self):
        state_path = self.workspace / "state.json"
        dashboard = self.workspace / "visualization" / "outputs" / "run" / "relu_realtime.html"
        dashboard.parent.mkdir(parents=True)
        dashboard.touch()

        def verify(*_args, **options):
            options["dashboard_ready_callback"](dashboard)
            self.assertTrue(options["dashboard_opener"](dashboard.as_uri()))
            return {"status": "VERIFIED"}

        with patch("Automation.AutoVerify.run_vnnlib_verification", side_effect=verify):
            _worker(self.group / "network.onnx", self.group / "prop_3.vnnlib.gz", "native", 12,
                    state_path=state_path, managed_browser=True)
        browser, process, log = Mock(), Mock(), Mock()
        log.state_directory.name = str(self.workspace)
        self.app.verification._open_worker_dashboard(browser, process, log)
        browser.open.assert_called_once_with(dashboard.as_uri())

    def test_worker_applies_selected_panels_limits_and_features(self):
        with patch("Automation.AutoVerify.run_vnnlib_verification", return_value={"status": "VERIFIED"}) as run:
            _worker(self.group / "network.onnx", self.group / "prop_3.vnnlib.gz", "glpk", 12,
                    settings={"visualization_mode": "feedback", "panels": ["simplex"],
                              "simplex_max_iter": 25, "max_rounds": 9, "max_recursion": 7,
                              "lp_mode": "oracle", "features": ["underapprox"], "under_approx_eps": 0.03})
        settings = run.call_args.kwargs
        self.assertEqual([settings["simplex_max_iter"], settings["max_rounds"], settings["max_recursion"]],
                         [25, 9, 7])
        self.assertEqual(settings["realtime_panels"], ["simplex"])
        self.assertEqual(settings["visualization_mode"], "feedback")
        self.assertFalse(settings["open_realtime_view"])
        self.assertEqual(settings["reluplex_options"]["lp_mode"], "oracle")
        self.assertFalse(settings["reluplex_options"]["derive_bound"])
        self.assertEqual(settings["reluplex_options"]["under_approx"], 0.03)
