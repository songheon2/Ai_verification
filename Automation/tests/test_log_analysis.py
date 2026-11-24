"""Offline tests only: never send logs or consume Gemini quota."""
import copy
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from Automation.AnalyzeLogs import (
    DEFAULT_MODEL, GeminiClient, NoRedirect, analyze, bounded, build_parser, main,
)
from Automation.LogAnalysisIndex import MAX_LINE_BYTES, RunIndex, safe_data


class FakeClient:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.requests = []

    def generate(self, contents, *, final=False):
        self.requests.append((copy.deepcopy(contents), final))
        value = next(self.responses)
        if isinstance(value, Exception):
            raise value
        return value


def answer(parts, finish="STOP"):
    return {"candidates": [{"content": {"role": "model", "parts": parts}, "finishReason": finish}],
            "usageMetadata": {"totalTokenCount": 12}}


class LogAnalysisTests(unittest.TestCase):
    def test_clicked_context_survives_run_summary_truncation(self):
        class LargeSummary:
            def summary(self):
                return {"large": "x" * 100000}
        client = FakeClient(answer([{"text": "선택 항목 분석입니다."}]))
        items = [{"scope": {"topic": "simplex", "call_id": "simplex-1"},
                  "text": "가" * 4000 + " SNAPSHOT_END"}]
        result = analyze({"run": LargeSummary()}, client, "왜 오래 걸렸나요?", selected_items=items)
        self.assertEqual(result["status"], "complete")
        parts = client.requests[0][0][0]["parts"]
        self.assertTrue(json.loads(parts[0]["text"])["truncated"])
        self.assertEqual(json.loads(parts[1]["text"])["visualization_selection"], items)

    def test_cli_and_automatic_dashboard_share_the_same_default_model(self):
        from inspect import signature
        from Automation.LogAnalysisLauncher import DEFAULT_MODEL as automatic_model
        from Automation.LogAnalysisServer import AnalysisApp
        self.assertEqual(DEFAULT_MODEL, "gemini-3.5-flash-lite")
        self.assertEqual(build_parser().parse_args([]).model, DEFAULT_MODEL)
        self.assertEqual(GeminiClient("offline-test-key").model, DEFAULT_MODEL)
        self.assertEqual(automatic_model, DEFAULT_MODEL)
        self.assertEqual(signature(AnalysisApp).parameters["model"].default, DEFAULT_MODEL)

    def test_generation_uses_visualization_terms_not_raw_log_report_instructions(self):
        client = GeminiClient("offline-test-key", "gemini-3.5-flash-lite")
        with patch.object(client, "_post_generation", return_value=answer([{"text": "관찰"}])) as send:
            client.generate([{"role": "user", "parts": [{"text": "분석"}]}])
            client.summarize_verification_batch([{"status": "complete"}], "비교")
            client.summarize_verification_batch([], "구조", snapshot={"overview": True})
        individual, batch, overview = [
            call.args[0]["systemInstruction"]["parts"][0]["text"] for call in send.call_args_list
        ]
        for instruction in (individual, batch, overview):
            self.assertIn("첫 문장에 질문의 결론", instruction)
            self.assertIn("문장마다 한 문단", instruction)
            self.assertIn("소수점, 수식, 표 구조", instruction)
            self.assertIn("내부 식별자", instruction)
            for term in ("selected atoms", "BOOLEAN_UNSAT", "GLPK 내부 반복", "ReLUplex", "수정 예산", "과소근사"):
                self.assertIn(term, instruction)
        for instruction in (individual, batch):
            self.assertIn("GLPK", instruction)
            self.assertIn("누락 항목은 미확인", instruction)
        for name in ("Theory Solver Result", "Repeated bound violations", "STABLE 확정 뉴런"):
            self.assertIn(name, individual)
        self.assertIn("기록된 순번", individual)
        self.assertIn("시각화 켬·끔", batch)
        self.assertIn("구조와 흐름", overview)

    def test_last_api_call_asks_for_final_answer_without_repeating_format_rules(self):
        client = FakeClient(answer([{"text": "현재 기록만으로는 원인을 확정할 수 없습니다."}]))
        result = analyze({"run": self.index()}, client, "왜 첫 호출이 오래 걸리나요?", 1)
        self.assertEqual(result["status"], "complete")
        contents, final = client.requests[0]
        self.assertTrue(final)
        instruction = contents[-1]["parts"][0]["text"]
        self.assertEqual(instruction, "확보한 근거로 최종 답변을 작성하세요.")

    def test_batch_retries_token_cutoff_and_preserves_the_configured_thinking_level(self):
        client = GeminiClient("offline-test-key", thinking_level="medium")
        client.max_requests = 2
        cutoff = answer([{"text": "internal", "thought": True}], "MAX_TOKENS")
        with patch.object(client, "_post_generation", side_effect=[cutoff, answer([{"text": "비교 결과"}])]) as send:
            self.assertEqual(client.summarize_verification_batch([{"status": "complete"}] * 10, "비교"), "비교 결과")
        configs = [call.args[0]["generationConfig"] for call in send.call_args_list]
        self.assertEqual([config["maxOutputTokens"] for config in configs], [16384, 32768])
        self.assertEqual(configs[0]["thinkingConfig"], configs[1]["thinkingConfig"])

    def test_batch_explains_missing_answer_and_respects_request_budget(self):
        client = GeminiClient("offline-test-key")
        client.max_requests = 1
        for response, reason in (({"promptFeedback": {"blockReason": "SAFETY"}}, "SAFETY"),
                                 (answer([{"text": "미완성"}], "MAX_TOKENS"), "MAX_TOKENS")):
            def send_once(_request):
                client.requests_made += 1
                return response

            client.requests_made = 0
            with self.subTest(reason=reason), patch.object(client, "_post_generation", side_effect=send_once) as send:
                with self.assertRaisesRegex(RuntimeError, reason):
                    client.summarize_verification_batch([], "질문")
                send.assert_called_once()

    def test_sentence_paragraphs_and_decimal_values_are_preserved_in_results(self):
        text = "현재 기록만으로 누적 오차를 확정할 수 없습니다.\n\n계수는 2.06이고 임계치는 2.71×10⁻¹²입니다."
        client = FakeClient(answer([{"text": text}]))
        result = analyze({"run": self.index()}, client, "수치 변화의 의미는 무엇인가요?", 1)
        self.assertEqual(result["analysis"], text)

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.run = self.root / "saved-run"
        self.run.mkdir()
        (self.run / "result.json").write_text(json.dumps({
            "status": "UNSAT", "model": {"layer_sizes": [2, 3, 1], "source_path": "C:\\private\\network.onnx"},
            "solver": {"elapsed_seconds": 15, "timeout_seconds": 30}}), encoding="utf-8")
        self.write_log("solver_progress.jsonl", [
            {"event": "round_end", "round": 1, "duration_seconds": 15},
            {"event": "simplex_end", "call_id": "simplex-1", "duration_seconds": 3, "pivots": 4},
            {"event": "simplex_call", "call_id": "simplex-2", "duration_seconds": 7, "pivots": 2},
        ])
        self.write_log("relu_events.jsonl", [
            {"event": "+", "layer": 1, "index": 2, "variable": "z1_2"},
            {"event": "-", "layer": 1, "index": 2, "variable": "z1_2"},
            {"event": "+", "layer": 1, "index": 2, "variable": "z1_2"},
        ])
        self.write_log("relu_events_branches.jsonl", [
            {"event": "branch_start", "branch_id": "b1", "round_index": 1,
             "time_ns": 100, "depth": 1, "layer": 1, "index": 2, "variable": "z1_2"},
            {"event": "branch_end", "branch_id": "b1", "time_ns": 200, "result": "UNSAT"},
        ])

    def write_log(self, name, records):
        (self.run / name).write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")

    def index(self, **kwargs):
        index = RunIndex(self.run, "run", **kwargs)
        self.addCleanup(index.close)
        return index

    def test_summary_counts_both_formats_without_nested_duration_double_count(self):
        summary = self.index().summary()
        self.assertEqual(summary["indexed_completed_simplex"], {"calls": 2, "seconds": 10, "pivots": 6})
        self.assertEqual(summary["result"]["model"]["source_path"], "network.onnx")
        self.assertFalse(any(f.get("truncated") for f in summary["files"]))

    def test_exact_rankings_and_evidence(self):
        index = self.index()
        self.assertEqual(index.rank({"metric": "slow_calls"})["ranking"][0]["data"]["call_id"], "simplex-2")
        self.assertEqual(index.rank({"metric": "pivot_calls"})["ranking"][0]["data"]["call_id"], "simplex-1")
        splits = index.rank({"metric": "split_neurons"})["ranking"]
        self.assertEqual(splits[0]["count"], 2)  # '-' is not another split
        self.assertEqual(splits[0]["example_evidence"], "run:relu_events.jsonl:1")
        self.assertEqual(index.rank({"metric": "branch_neurons"})["ranking"][0]["count"], 1)

    def test_query_filters_pagination_round_mapping_branch_and_time(self):
        index = self.index()
        rows = index.query({"file": "relu_events.jsonl", "layer": 1, "neuron": 2, "start_line": 3})["records"]
        self.assertEqual([r["evidence"] for r in rows], ["run:relu_events.jsonl:3"])
        self.assertEqual(len(index.query({"round": 1, "event": "branch_start"})["records"]), 1)
        self.assertEqual(len(index.query({"branch_id": "b1"})["records"]), 2)
        self.assertEqual(index.query({"after_ns": 150, "before_ns": 250})["records"][0]["data"]["event"], "branch_end")

    def test_queries_do_not_accept_sql_or_paths(self):
        index = self.index()
        self.assertEqual(index.query({"file": "../../secret.txt"})["records"], [])
        self.assertEqual(index.query({"event": "' OR 1=1 --"})["records"], [])
        with self.assertRaises(ValueError):
            index.rank({"metric": "DROP TABLE events"})
        with self.assertRaises(ValueError):
            index.query({"file": {"malicious": True}})

    def test_index_budget_missing_and_corrupt_logs_are_explicit(self):
        (self.run / "relu_events.jsonl").unlink()
        with (self.run / "solver_progress.jsonl").open("a", encoding="utf-8") as stream:
            stream.write("broken\n{}\n")
        summary = self.index().summary()
        self.assertEqual(summary["files"][0]["invalid_lines"], 1)
        self.assertTrue(summary["files"][2]["missing"])
        limited = self.index(max_bytes=10).summary()
        self.assertTrue(limited["files"][0]["truncated"])
        self.assertLessEqual(sum(f.get("indexed_bytes", 0) for f in limited["files"]), 10)

    def test_oversized_line_stops_without_unbounded_read(self):
        (self.run / "solver_progress.jsonl").write_bytes(b" " * (MAX_LINE_BYTES + 20))
        info = self.index().summary()["files"][0]
        self.assertTrue(info["truncated"])
        self.assertEqual(info["invalid_lines"], 1)

    def test_completed_result_is_required(self):
        (self.run / "result.json").unlink()
        with self.assertRaisesRegex(ValueError, "Completed"):
            self.index()

    def test_read_only_sources_and_cleanup(self):
        def hashes():
            return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in self.run.iterdir()}
        before = hashes()
        index = RunIndex(self.run, "run")
        cache = Path(index.temp.name)
        index.query({})
        index.close()
        self.assertFalse(cache.exists())
        self.assertEqual(hashes(), before)

    def test_symlink_outside_selected_run_is_rejected(self):
        (self.run / "solver_progress.jsonl").unlink()
        external = self.root / "outside.jsonl"
        external.write_text("{}\n", encoding="utf-8")
        try:
            (self.run / "solver_progress.jsonl").symlink_to(external)
        except OSError:
            self.skipTest("Windows symlink permission not enabled")
        with self.assertRaisesRegex(ValueError, "symlinks"):
            self.index()

    def test_tool_loop_preserves_signatures_ids_and_records_audit(self):
        parts = [{"functionCall": {"name": "rank_events", "args": {"run_id": "run", "metric": "slow_calls"}, "id": "c1"},
                  "thoughtSignature": "opaque-signature"}]
        client = FakeClient(answer(parts), answer([{"text": "관찰: [run:solver_progress.jsonl:3]"}]))
        result = analyze({"run": self.index()}, client, "분석", max_calls=3)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["api_calls"], 2)
        second = client.requests[1][0]
        self.assertEqual(second[1]["parts"], parts)
        self.assertEqual(second[2]["parts"][0]["functionResponse"]["id"], "c1")
        self.assertEqual(len(result["queries"]), 1)

    def test_unknown_tools_and_runs_cannot_execute_actions(self):
        client = FakeClient(answer([
            {"functionCall": {"name": "exec", "args": {"command": "delete"}}},
            {"functionCall": {"name": "run_summary", "args": {"run_id": "unselected"}}},
        ]), answer([{"text": "한계"}]))
        result = analyze({"run": self.index()}, client, "분석", 2)
        self.assertTrue(all("error" in q["response"] for q in result["queries"]))
        self.assertTrue(client.requests[-1][1])

    def test_calls_are_capped_and_final_call_has_no_tools(self):
        call = answer([{"functionCall": {"name": "run_summary", "args": {"run_id": "run"}}}])
        client = FakeClient(call, call)
        result = analyze({"run": self.index()}, client, "분석", 2)
        self.assertEqual(result["status"], "budget_exhausted")
        self.assertEqual(len(client.requests), 2)
        self.assertEqual([final for _, final in client.requests], [False, True])

    def test_api_error_retains_partial_evidence_without_retry(self):
        call = answer([{"functionCall": {"name": "run_summary", "args": {"run_id": "run"}}}])
        client = FakeClient(call, RuntimeError("Quota reached"))
        result = analyze({"run": self.index()}, client, "분석", 5)
        self.assertEqual(result["status"], "api_error")
        self.assertEqual(result["api_calls"], 2)
        self.assertEqual(len(result["queries"]), 1)

    def test_token_limit_is_incomplete_not_success(self):
        result = analyze({"run": self.index()}, FakeClient(answer([{"text": "partial"}], "MAX_TOKENS")), "분석")
        self.assertEqual(result["status"], "incomplete")

    def test_redaction_and_response_budget(self):
        clean = safe_data({"api_key": "private", "path": "C:\\user\\file.onnx", "value": float("inf"),
                           "text": "AIza" + "a" * 30})
        self.assertNotIn("private", str(clean))
        self.assertEqual(clean["path"], "file.onnx")
        self.assertNotIn("AIza", clean["text"])
        value = bounded({"data": "가" * 10000}, 1000)
        self.assertTrue(value["truncated"])
        self.assertLessEqual(len(json.dumps(value, ensure_ascii=False).encode()), 1000)
        escaped = bounded({"data": '"\\\n' * 10000}, 1000)
        self.assertLessEqual(len(json.dumps(escaped, ensure_ascii=False).encode()), 1000)

    def test_blocked_response_keeps_previous_queries(self):
        call = answer([{"functionCall": {"name": "run_summary", "args": {"run_id": "run"}}}])
        result = analyze({"run": self.index()}, FakeClient(call, {}), "분석", 3)
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(len(result["queries"]), 1)

    def test_prepare_only_no_credentials_network_and_new_output_files(self):
        output = self.root / "analysis-output"
        with patch.dict(os.environ, {}, clear=True), patch("Automation.AnalyzeLogs.GeminiClient") as client:
            code = main(["--run", str(self.run), "--compare", str(self.run),
                         "--prepare-only", "--output-dir", str(output)])
            client.assert_not_called()
        self.assertEqual(code, 0)
        result = json.loads(next(output.glob("*/analysis.json")).read_text(encoding="utf-8"))
        self.assertEqual(result["api_calls"], 0)
        self.assertEqual([r["run_id"] for r in result["runs"]], ["run", "compare"])

    def test_upload_requires_explicit_consent_and_key(self):
        with patch("Automation.AnalyzeLogs.GeminiClient") as client, patch("sys.stderr"):
            with self.assertRaises(SystemExit):
                main(["--run", str(self.run)])
            client.assert_not_called()
        with patch("Automation.AnalyzeLogs.load_gemini_api_key", return_value=""), patch("sys.stderr"):
            with self.assertRaises(SystemExit):
                main(["--run", str(self.run), "--allow-upload"])

    def test_transport_uses_header_limits_and_disables_redirects(self):
        client = GeminiClient("not-a-real-key")
        response = BytesIO(json.dumps(answer([{"text": "ok"}])).encode())
        with patch.object(client.opener, "open", return_value=response) as send:
            client.generate([{"role": "user", "parts": [{"text": "test"}]}], final=True)
            req = send.call_args.args[0]
            self.assertEqual(req.get_header("X-goog-api-key"), "not-a-real-key")
            self.assertNotIn("not-a-real-key", req.full_url)
            self.assertEqual(json.loads(req.data)["toolConfig"]["functionCallingConfig"]["mode"], "NONE")
        self.assertIsNone(NoRedirect().redirect_request(None, None, 302, "", {}, "https://other"))
        error = HTTPError("https://host", 429, "secret should not appear", {}, None)
        with patch.object(client.opener, "open", side_effect=error) as send:
            with self.assertRaisesRegex(RuntimeError, "no automatic retry"):
                client.generate([])
            self.assertEqual(send.call_count, 1)

    def test_transient_retry_consumes_total_budget_and_preserves_error_details(self):
        client = GeminiClient("private-key")
        client.max_requests = 2
        error = HTTPError("https://host", 503, "", {}, BytesIO(json.dumps({"error": {
            "status": "UNAVAILABLE", "message": "Temporarily overloaded private-key"}}).encode()))
        response = BytesIO(json.dumps(answer([{"text": "ok"}])).encode())
        progress = []
        counts = []
        client.progress = progress.append
        client.request_progress = lambda used, limit: counts.append((used, limit))
        with patch.object(client.opener, "open", side_effect=[error, response]) as send, patch("Automation.AnalyzeLogs.time.sleep"):
            result = analyze({"run": self.index()}, client, "분석", 2)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["api_calls"], 2)
        self.assertEqual(send.call_count, 2)
        self.assertEqual(counts, [(1, 2), (2, 2)])
        self.assertTrue(any("UNAVAILABLE" in p for p in progress))
        self.assertNotIn("private-key", str(progress))

    def test_retry_never_exceeds_request_budget(self):
        client = GeminiClient("private-key")
        error = HTTPError("https://host", 503, "", {}, BytesIO(b'{"error":{"message":"busy"}}'))
        with patch.object(client.opener, "open", side_effect=error) as send, patch("Automation.AnalyzeLogs.time.sleep") as sleep:
            result = analyze({"run": self.index()}, client, "분석", 1)
        self.assertEqual(result["api_calls"], 1)
        self.assertEqual(result["status"], "api_error")
        send.assert_called_once()
        sleep.assert_not_called()

    def test_timeout_is_not_retried_and_diagnostic_is_safe(self):
        client = GeminiClient("private-key")
        with patch.object(client.opener, "open", side_effect=URLError(TimeoutError("read timed out private-key"))) as send:
            result = analyze({"run": self.index()}, client, "분석", 8)
        self.assertEqual(result["status"], "api_error")
        self.assertEqual(result["api_calls"], 1)
        self.assertIn("응답 대기 시간(1200초)", result["analysis"])
        self.assertIn("추론 강도를 낮춰", result["analysis"])
        self.assertNotIn("private-key", result["analysis"])
        send.assert_called_once()

    def test_google_error_status_and_message_are_exposed_without_key(self):
        client = GeminiClient("private-key")
        error = HTTPError("https://host", 400, "", {}, BytesIO(json.dumps({"error": {
            "status": "INVALID_ARGUMENT", "message": "API key not valid: private-key"}}).encode()))
        with patch.object(client.opener, "open", side_effect=error):
            result = analyze({"run": self.index()}, client, "분석", 2)
        self.assertIn("INVALID_ARGUMENT", result["analysis"])
        self.assertIn("API key not valid", result["analysis"])
        self.assertNotIn("private-key", result["analysis"])

    def test_connection_check_lists_models_without_reading_logs(self):
        client = GeminiClient("private-key")
        response = BytesIO(json.dumps({"models": [
            {"name": "models/" + DEFAULT_MODEL, "supportedGenerationMethods": ["generateContent"]},
            {"name": "models/gemini-3.8-flash", "supportedGenerationMethods": ["generateContent"]},
            {"name": "models/gemini-3.5-flash", "supportedGenerationMethods": ["generateContent"]},
            {"name": "models/gemini-4-hydrogen", "supportedGenerationMethods": ["generateContent"]},
            {"name": "models/gemini-4-carbon", "supportedGenerationMethods": ["generateContent"]},
            {"name": "models/gemini-3.1-flash-lite", "supportedGenerationMethods": ["generateContent"]},
            {"name": "models/gemini-image", "supportedGenerationMethods": ["generateContent"]},
            {"name": "models/gemini-embedding", "supportedGenerationMethods": ["embedContent"]},
        ]}).encode())
        with patch.object(client.opener, "open", return_value=response) as send:
            listing = client.list_models()
        self.assertTrue(listing["selected_model_listed"])
        self.assertEqual(listing["analysis_models"], [DEFAULT_MODEL, "gemini-3.8-flash", "gemini-3.5-flash", "gemini-4-hydrogen", "gemini-4-carbon"])
        self.assertEqual(listing["flash_models"], [DEFAULT_MODEL, "gemini-3.8-flash", "gemini-3.5-flash"])
        self.assertEqual(send.call_args.args[0].get_method(), "GET")
        self.assertIsNone(send.call_args.args[0].data)
        with patch.dict(os.environ, {"GEMINI_API_KEY": "private-key"}), patch("Automation.AnalyzeLogs.GeminiClient", return_value=client), patch.object(client, "list_models", return_value=listing), patch("Automation.AnalyzeLogs.RunIndex") as index, patch("sys.stdout"):
            self.assertEqual(main(["--check-api"]), 0)
            index.assert_not_called()

    def test_pasted_key_quotes_are_normalized_invalid_characters_rejected(self):
        self.assertEqual(GeminiClient('  "private-key"  ').key, "private-key")
        for key in ("키 설정 완료", "private\nkey", "private\x00key"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                GeminiClient(key)

    def test_generation_probe_does_not_send_logs_system_instructions_or_tools(self):
        client = GeminiClient("private-key")
        response = BytesIO(json.dumps(answer([{"text": "OK"}])).encode())
        with patch.object(client.opener, "open", return_value=response) as send:
            result = client.check_generation()
        body = json.loads(send.call_args.args[0].data)
        self.assertTrue(result["generation_succeeded"])
        self.assertEqual(body["contents"][0]["parts"][0]["text"], "Reply with exactly OK.")
        self.assertNotIn("tools", body)
        self.assertNotIn("systemInstruction", body)
        self.assertEqual(body["generationConfig"]["thinkingConfig"], {"thinkingLevel": "LOW"})

    def test_generation_probe_cli_never_reads_run(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "private-key"}), patch("Automation.AnalyzeLogs.GeminiClient") as client, patch("Automation.AnalyzeLogs.RunIndex") as index, patch("sys.stdout"):
            client.return_value.check_generation.return_value = {"generation_succeeded": True}
            self.assertEqual(main(["--check-generation", "--model", "gemini-3.5-flash-lite", "--max-calls", "1"]), 0)
            self.assertEqual(client.return_value.max_requests, 1)
            index.assert_not_called()

    def test_thinking_configuration_respects_model_version_and_explicit_default(self):
        self.assertEqual(GeminiClient("key", thinking_level="default").generation_config(), {"maxOutputTokens": 4096})
        self.assertEqual(GeminiClient("key", model="gemini-2.5-flash").generation_config()["thinkingConfig"], {"thinkingBudget": 512})
        self.assertEqual(GeminiClient("key", thinking_level="high").generation_config()["thinkingConfig"], {"thinkingLevel": "HIGH"})
        self.assertEqual(GeminiClient("key", model="gemini-4-hydrogen", thinking_level="xhigh").generation_config()["thinkingConfig"], {"thinkingLevel": "xhigh"})
        with self.assertRaises(ValueError):
            GeminiClient("key", model="gemini-3.8-flash", thinking_level="minimal")

    def test_gemini_uses_one_default_timeout_without_retrying_a_read_timeout(self):
        class SlowResponse(BytesIO):
            def read(self, *_args):
                raise TimeoutError("The read operation timed out")

        client = GeminiClient("offline-key", model="gemini-4-neon")
        self.assertEqual(client.timeout, 1200)
        client.thinking_level = "xhigh"
        with patch.object(client.opener, "open", return_value=SlowResponse()) as send:
            with self.assertRaisesRegex(RuntimeError, r"응답 대기 시간\(1200초\)"):
                client.generate([{"role": "user", "parts": [{"text": "test"}]}], final=True)
        self.assertEqual(send.call_count, 1)
        self.assertEqual(send.call_args.kwargs["timeout"], 1200)
        explicit = GeminiClient("offline-key", model="gemini-4-neon", thinking_level="xhigh", timeout=20)
        self.assertEqual(explicit.timeout, 20)


if __name__ == "__main__":
    unittest.main()
