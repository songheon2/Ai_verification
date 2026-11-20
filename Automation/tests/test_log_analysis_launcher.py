"""Existing-command auto connection; dummy keys only, no external API requests."""
from http.client import HTTPConnection
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import MagicMock, patch
from urllib.parse import urlsplit

from Automation import LogAnalysisLauncher as launcher
from Automation.LogAnalysisServer import AnalysisApp, make_server, run_server_loop


class AutomaticAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        env_file = patch("Automation.AnalyzeLogs.GEMINI_ENV_FILE", self.root / ".env")
        env_file.start()
        self.addCleanup(env_file.stop)
        self.page = self.root / "custom_realtime.html"
        self.page.write_text('<!doctype html><main><h1>Existing visualization</h1></main>', encoding="utf-8")
        (self.root / "run_info.json").write_text(json.dumps({"run_id": "automatic-test", "status": "RUNNING"}), encoding="utf-8")
        self.key_patch = patch.dict(os.environ, {"GEMINI_API_KEY": "offline-test-key-never-uploaded"})
        self.key_patch.start()
        self.addCleanup(self.key_patch.stop)

    def test_no_key_or_invalid_key_never_starts_process(self):
        with patch("Automation.LogAnalysisLauncher.subprocess.Popen") as spawn:
            for key, reason in (("", "missing_key"), ("bad\nkey", "invalid_key")):
                with self.subTest(reason=reason), patch.dict(os.environ, {"GEMINI_API_KEY": key}):
                    self.assertEqual(launcher.start_analysis_bridge(self.root, self.page)["reason"], reason)
            spawn.assert_not_called()

    def test_custom_dashboard_outside_run_keeps_file_behavior(self):
        with patch("Automation.LogAnalysisLauncher.subprocess.Popen") as spawn:
            result = launcher.start_analysis_bridge(self.root, self.root.parent / "elsewhere.html")
        self.assertEqual(result["reason"], "external_dashboard")
        spawn.assert_not_called()

    def test_handshake_hides_window_and_never_contains_api_key(self):
        process = MagicMock()
        process.poll.return_value = None
        observed = {}
        def spawn(command, **kwargs):
            observed.update(command=command, kwargs=kwargs)
            ready = Path(command[command.index("--ready-file") + 1])
            ready.write_text(json.dumps({"status": "ready", "url": "http://127.0.0.1:12345/s/session/"}), encoding="utf-8")
            return process
        with patch("Automation.LogAnalysisLauncher.subprocess.Popen", side_effect=spawn), \
                patch("Automation.AnalyzeLogs.GeminiClient._request", side_effect=AssertionError("No network")), \
                patch.object(launcher, "_children", []):
            result = launcher.start_analysis_bridge(self.root, self.page)
        self.assertEqual(result["status"], "ready")
        self.assertNotIn(os.environ["GEMINI_API_KEY"], json.dumps(observed, default=str))
        self.assertEqual(observed["kwargs"]["creationflags"], getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self.assertTrue(observed["kwargs"]["close_fds"])
        self.assertEqual(observed["kwargs"]["stdin"], subprocess.DEVNULL)

    def test_startup_timeout_and_spawn_failure_are_nonfatal(self):
        process = MagicMock()
        process.poll.return_value = None
        with patch("Automation.LogAnalysisLauncher.subprocess.Popen", return_value=process):
            self.assertEqual(launcher.start_analysis_bridge(self.root, self.page, startup_timeout=0)["reason"], "startup_failed")
        process.terminate.assert_called_once()
        with patch("Automation.LogAnalysisLauncher.subprocess.Popen", side_effect=PermissionError("blocked")):
            self.assertEqual(launcher.start_analysis_bridge(self.root, self.page)["reason"], "startup_failed")

    def test_opens_only_one_http_page_or_falls_back_to_existing_file(self):
        with patch("Automation.LogAnalysisLauncher.webbrowser.open") as browser, \
                patch("Automation.LogAnalysisLauncher.start_analysis_bridge", return_value={"status": "ready", "url": "http://127.0.0.1:123/s/private/", "model": launcher.DEFAULT_MODEL}), \
                patch("builtins.print"):
            result = launcher.open_dashboard_with_analysis(self.root, self.page)
            browser.assert_called_once_with("http://127.0.0.1:123/s/private/", new=2)
            self.assertNotIn("url", result)  # Do not archive temporary session tokens in result.json.
        with patch("Automation.LogAnalysisLauncher.webbrowser.open") as browser, \
                patch.dict(os.environ, {"GEMINI_API_KEY": ""}), patch("builtins.print"):
            launcher.open_dashboard_with_analysis(self.root, self.page)
            browser.assert_called_once_with(self.page.resolve().as_uri(), new=2)

    def test_browser_error_does_not_interrupt_verification(self):
        with patch("Automation.LogAnalysisLauncher.webbrowser.open", side_effect=OSError("browser unavailable")), \
                patch.dict(os.environ, {"GEMINI_API_KEY": ""}), patch("builtins.print"):
            result = launcher.open_dashboard_with_analysis(self.root, self.page)
        self.assertFalse(result["browser_opened"])

    def test_real_child_serves_existing_dashboard_without_api_and_can_close(self):
        realtime = self.root / "realtime"
        realtime.mkdir()
        page_path = self.page.rename(realtime / self.page.name)
        (realtime / "panel.selection.js").write_text("window.panelSelection = [];", encoding="utf-8")
        result = launcher.start_analysis_bridge(self.root, page_path)
        self.assertEqual(result["status"], "ready", result)
        child = launcher._children[-1]
        self.addCleanup(lambda: launcher._stop_child(child))
        url = urlsplit(result["url"])
        connection = HTTPConnection(url.hostname, url.port, timeout=5)
        self.addCleanup(connection.close)
        connection.request("GET", url.path)
        response = connection.getresponse()
        page = response.read().decode("utf-8")
        self.assertEqual(response.status, 200)
        self.assertIn("Existing visualization", page)
        self.assertIn('id="llm-start"', page)
        self.assertNotIn(os.environ["GEMINI_API_KEY"], page)
        connection.request("GET", url.path + "panel.selection.js")
        response = connection.getresponse()
        self.assertEqual(response.status, 200)
        self.assertEqual(response.read(), b"window.panelSelection = [];")
        connection.request("GET", url.path + "api/info")
        response = connection.getresponse()
        info = json.loads(response.read())
        self.assertEqual(info["model"], launcher.DEFAULT_MODEL)
        self.assertEqual(info["state"]["status"], "RUNNING")
        self.assertEqual(list((self.root / "analysis").glob("**/*")), [])
        headers = {"Content-Type": "application/json", "X-Analysis-Token": "wrong"}
        connection.request("POST", url.path + "api/close", body="{}", headers=headers)
        response = connection.getresponse()
        response.read()
        self.assertEqual(response.status, 403)
        headers["X-Analysis-Token"] = info["csrf"]
        connection.request("POST", url.path + "api/close", body="{}", headers=headers)
        response = connection.getresponse()
        response.read()
        self.assertEqual(response.status, 200)
        self.assertEqual(child.wait(timeout=5), 0)

    def test_automatic_idle_loop_exits_without_processing_or_api(self):
        app = AnalysisApp(self.root, client_factory=lambda _: self.fail("No API"), dashboard=self.page.name)
        server = make_server(app)
        try:
            app.last_activity = 0
            with patch.object(server, "handle_request") as handle:
                run_server_loop(server, app, idle_seconds=0)
                handle.assert_not_called()
        finally:
            server.server_close()
            app.close()

    def test_existing_verification_opens_connected_dashboard_before_solver(self):
        from Automation.AutoVerify import run_vnnlib_verification
        from Automation.SolverStatus import SolverResult, SolverStatus
        prop = self.root / "property.vnnlib"
        prop.write_text('(declare-const X_0 Real) (declare-const X_1 Real) '
                        '(declare-const Y_0 Real) (assert (>= Y_0 0))', encoding="utf-8")
        run = self.root / "verification"
        def connected(root, dashboard):
            self.assertEqual(Path(root), run)
            self.assertTrue(Path(dashboard).is_file())
            self.assertTrue((run / "run_info.json").is_file())
            return {"status": "ready", "model": launcher.DEFAULT_MODEL}
        def solver(*args, **kwargs):
            open_page.assert_called_once()
            return SolverResult(status=SolverStatus.UNSAT, reason="BOOLEAN_UNSAT", rounds=1, elapsed_seconds=0)
        model = Path(__file__).resolve().parents[2] / "xor_network.bin"
        with patch("Automation.LogAnalysisLauncher.open_dashboard_with_analysis", side_effect=connected) as open_page, \
                patch("Automation.AutoVerify.dpll_t_detailed", side_effect=solver):
            result = run_vnnlib_verification(str(model), str(prop), visualization_mode="realtime",
                        realtime_panels=["mechanisms"], open_realtime_view=True, visualization_output_dir=str(run))
        self.assertEqual(result["visualization"]["realtime"]["llm_analysis"]["status"], "ready")
        self.assertEqual(result["status"], "VERIFIED")

    def test_no_browser_and_dry_run_do_not_launch_analysis(self):
        from Automation.AutoVerify import run_vnnlib_verification
        from Automation.SolverStatus import SolverResult, SolverStatus
        prop = self.root / "property.vnnlib"
        prop.write_text('(declare-const X_0 Real) (declare-const X_1 Real) '
                        '(declare-const Y_0 Real) (assert (>= Y_0 0))', encoding="utf-8")
        model = Path(__file__).resolve().parents[2] / "xor_network.bin"
        with patch("Automation.LogAnalysisLauncher.open_dashboard_with_analysis") as open_page, \
                patch("Automation.AutoVerify.dpll_t_detailed", return_value=SolverResult(
                    status=SolverStatus.UNSAT, reason="BOOLEAN_UNSAT", rounds=1, elapsed_seconds=0)):
            run_vnnlib_verification(str(model), str(prop), visualization_mode="off", visualization_output_dir=str(self.root / "off"))
            run_vnnlib_verification(str(model), str(prop), visualization_mode="realtime", dry_run=True, open_realtime_view=True)
            open_page.assert_not_called()


if __name__ == "__main__":
    unittest.main()
