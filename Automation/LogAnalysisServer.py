"""Loopback-only dashboard bridge. No arbitrary paths, commands or browser API keys."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import os
from pathlib import Path
import secrets
import threading
import time
from urllib.parse import parse_qs, unquote, urlsplit
import webbrowser

from Automation.AnalyzeLogs import (DEFAULT_MODEL, GeminiClient, analysis_model_options, analyze, is_analysis_model,
                                    load_gemini_api_key, model_thinking_levels, validate_thinking_level)
from Automation.AnalysisConversation import AnalysisConversation
from Automation.LogAnalysisIndex import safe_data
from Automation.LiveLogAnalysis import LiveRunIndex
from Automation.LogAnalysisScope import ScopedIndex, normalize_scope, scope_key
from visualization.LogAnalysisPanel import install_panel
from visualization.Fonts import apply_web_font
from visualization.VerificationPicker import AI_STUDIO_QUOTA_URL
from visualization.VisualizationMode import REALTIME_DIRECTORY, resolve_visualization_output


class AnalysisApp:
    def __init__(self, run, *, model=DEFAULT_MODEL, max_calls=8, max_log_mb=256,
                 output_dir="Automation/Results/llm_analysis", client_factory=None, dashboard="relu_realtime.html",
                 model_client_factory=None, model_catalog=None, verification=None, batch_max_calls=2):
        self.root = resolve_visualization_output(run).resolve(strict=True)
        self.dashboard_path = (self.root / dashboard).resolve()
        if self.dashboard_path.parent not in (self.root, self.root / REALTIME_DIRECTORY) or self.dashboard_path.suffix.lower() != ".html":
            raise ValueError("Dashboard must be an HTML file in the run directory")
        if not self.dashboard_path.is_file() and self.dashboard_path.name == "relu_realtime.html":
            for parent in (self.root / REALTIME_DIRECTORY, self.root):
                for name in ("relu_realtime.html", "solver_realtime.html"):
                    candidate = parent / name
                    if candidate.is_file() and candidate.resolve().parent == parent:
                        self.dashboard_path = candidate
                        break
                if self.dashboard_path.is_file():
                    break
        if not is_analysis_model(model):
            raise ValueError("Gemini 3.5 이상 텍스트 모델을 선택해 주세요.")
        self.model, self.max_calls, self.max_log_mb = model, max_calls, max_log_mb
        self.output_dir = Path(output_dir)
        self.client_factory = client_factory
        self.model_client_factory = model_client_factory
        self.model_catalog = model_catalog
        self.model_lock = threading.Lock()
        self.available_models = None
        self.verification = verification
        self.batch_max_calls = batch_max_calls
        if client_factory is None:
            raise ValueError("A server-side client factory is required")
        self.index = LiveRunIndex(self.root, max_log_mb * 1024 * 1024)
        self.fingerprint, self.identity = self.index.fingerprint, self.index.identity
        self.token, self.csrf = secrets.token_urlsafe(24), secrets.token_urlsafe(24)
        self.lock = threading.Lock()
        self.jobs, self.results = {}, {}
        self.conversation = AnalysisConversation()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="log-analysis")
        self.active = None
        self.last_activity = time.monotonic()
        self.shutdown_requested = threading.Event()
        compatible = {self.fingerprint}
        legacy_result = self.root / "result.json"
        if legacy_result.is_file() and legacy_result.resolve().parent == self.root and legacy_result.stat().st_size <= 1024 * 1024:
            compatible.add(hashlib.sha256(str(self.root).encode() + legacy_result.read_bytes()).hexdigest())
        for path in sorted(self.output_dir.glob("*/analysis.json"), reverse=True)[:100]:
            try:
                if path.stat().st_size > 2 * 1024 * 1024:
                    continue
                result = json.loads(path.read_text(encoding="utf-8"))
                if result.get("run_fingerprint") in compatible and "scope" in result:
                    self.results.setdefault(scope_key(result["scope"]), result)
            except (OSError, ValueError, TypeError):
                continue

    def info(self):
        with self.lock:
            state = self.index.refresh()
            self._sync_identity()
            return {"model": self.model, "max_calls": self.max_calls, "run_name": self.root.name,
                    "options": self.index.options(), "state": state, "csrf": self.csrf,
                    "provider_quota_remaining": None, "selection_context": True,
                    "model_selection": self.model_catalog is not None,
                    "thinking_levels": model_thinking_levels(self.model),
                    "verification_launcher": self.verification is not None}

    def models(self):
        if self.model_catalog is None:
            return analysis_model_options([self.model])
        with self.model_lock:
            if self.available_models is None:
                listed = self.model_catalog()["analysis_models"]
                self.available_models = tuple(analysis_model_options(listed)["models"])
            return analysis_model_options(self.available_models)

    def request_budget(self, used=0):
        used = max(0, int(used))
        return {"used": used, "limit": self.max_calls, "remaining": max(0, self.max_calls - used)}

    def _sync_identity(self):
        if self.identity != self.index.identity:
            self.results.clear()
            self.conversation.clear()
            self.identity, self.fingerprint = self.index.identity, self.index.fingerprint

    def _analysis_model(self, request):
        model = request.get("model", self.model)
        if not is_analysis_model(model):
            raise ValueError("Gemini 3.5 이상 텍스트 모델을 선택해 주세요.")
        if self.model_catalog is not None and model not in (self.available_models or ()):
            raise ValueError("조회된 모델 목록에서 선택해 주세요.")
        if model != self.model and self.model_client_factory is None:
            raise ValueError("현재 분석 서버에서 지원하지 않는 모델입니다.")
        return model

    def _job_client(self, model, progress, thinking_level):
        client = (self.client_factory(progress) if model == self.model
                  else self.model_client_factory(model, progress))
        if thinking_level is not None:
            client.thinking_level = thinking_level
        return client

    batch_analysis_available = True

    def start_batch_analysis(self, request):
        if not isinstance(request, dict) or request.get("consent") is not True:
            raise ValueError("검증 묶음의 요약 전송에 동의해 주세요.")
        model = self._analysis_model(request)
        thinking_level = validate_thinking_level(model, request["thinking_level"]) if "thinking_level" in request else None

        return self.verification.start_batch_analysis(
            request, lambda: self._job_client(model, None, thinking_level), max_calls=self.batch_max_calls)

    def start(self, request):
        if not isinstance(request, dict) or request.get("consent") is not True:
            raise ValueError("선택 항목의 로그 전송에 동의해 주세요.")
        model = self._analysis_model(request)
        thinking_level = validate_thinking_level(model, request["thinking_level"]) if "thinking_level" in request else None
        scope = normalize_scope(request.get("scope"))
        question = request.get("question", "")
        if not isinstance(question, str) or len(question) > 2000:
            raise ValueError("질문은 2000자 이내로 입력해 주세요.")
        items = request.get("selected_items", [])
        if not isinstance(items, list) or len(items) > 20:
            raise ValueError("선택 항목은 최대 20개까지 전달할 수 있습니다.")
        selected_items = []
        for item in items:
            if not isinstance(item, dict) or set(item) != {"scope", "text"}:
                raise ValueError("선택 정보 형식이 올바르지 않습니다.")
            item_scope = normalize_scope(item["scope"])
            if "selections" in item_scope or not isinstance(item["text"], str) or len(item["text"]) > 5000:
                raise ValueError("선택 정보 형식 또는 크기가 올바르지 않습니다.")
            selected_items.append({"scope": item_scope, "text": item["text"]})
        if selected_items:
            expected = {scope_key(value) for value in scope.get("selections", [scope])}
            if {scope_key(item["scope"]) for item in selected_items} != expected:
                raise ValueError("선택 정보와 분석 범위가 일치하지 않습니다.")
        if len(json.dumps(selected_items, ensure_ascii=False).encode("utf-8")) > 32000:
            raise ValueError("선택 정보가 너무 큽니다. 일부 항목을 해제해 주세요.")
        with self.lock:
            if self.active:
                raise RuntimeError("다른 분석이 실행 중입니다. 완료 후 다시 요청해 주세요.")
            # Freeze before acknowledging the button press, not after a queued worker starts.
            index = self.index.freeze()
            self._sync_identity()
            job_id = secrets.token_hex(12)
            self.active = job_id
            fingerprint = self.fingerprint
            conversation = self.conversation.snapshot()
            budget = self.request_budget()
            self.jobs[job_id] = {"id": job_id, "scope": scope, "model": model, "status": "running",
                                 "snapshot": index.snapshot, "request_budget": budget,
                                 "progress": "선택 항목의 고정된 로그를 정리 중입니다."}
        self.executor.submit(self._work, job_id, scope, question, selected_items, index, fingerprint, model, conversation, thinking_level)
        return {"id": job_id, "snapshot": index.snapshot, "request_budget": budget, "model": model}

    def _work(self, job_id, scope, question, selected_items, index, fingerprint, model, conversation, thinking_level):
        def progress(message):
            with self.lock:
                self.jobs[job_id]["progress"] = message
        def request_progress(used, _limit):
            with self.lock:
                self.jobs[job_id]["request_budget"] = self.request_budget(used)
        try:
            selected = ScopedIndex(index, scope)
            summary = selected.summary()
            focus = ("선택 목록의 항목들을 함께 분석·비교하되 선택하지 않은 항목으로 논점을 바꾸지 마세요. " if "selections" in scope
                     else "이 항목에 답하고 다른 항목으로 논점을 바꾸지 마세요. ")
            prompt = (f"사용자 질문: {question.strip() or '선택 항목의 병목 원인과 그 근거를 분석해 주세요.'}\n"
                      f"분석 범위: {json.dumps(scope, ensure_ascii=False)}. {focus}"
                      "기록은 snapshot 시점에 고정되어 있습니다. 실행 중이면 최종 판정이나 완료 시간을 추측하지 마세요. "
                      "관련 기록이 없어도 저장 설정과 제공된 구현 참고 설명으로 답할 수 있는 부분은 답하세요. "
                      "원인 질문에는 수치 나열을 넘어 비교 지표와 구현 기전으로 설명하고 확인 사실과 가설을 구분하세요.")
            client = self._job_client(model, progress, thinking_level)
            client.request_progress = request_progress
            result = analyze({"run": selected}, client, prompt, self.max_calls, selected_items=selected_items,
                             conversation=conversation)
            result.update(scope=scope, question=question, selected_items=selected_items, model=model, runs=[summary],
                          run_fingerprint=fingerprint, snapshot=index.snapshot,
                          request_budget=self.request_budget(result.get("api_calls", 0)),
                          created_at=datetime.now(timezone.utc).isoformat(),
                          disclaimer="LLM 분석은 검증 판정/증명이 아닙니다. 선택 범위 밖 실행 정보는 배경 문맥입니다.")
            if thinking_level is not None:
                result["thinking_level"] = thinking_level
            result = safe_data(result)
            self._save(result)
            with self.lock:
                if self.identity == index.snapshot["identity"]:
                    self.results[scope_key(scope)] = result
                    if result["status"] in ("complete", "incomplete"):
                        self.conversation.remember(question, result["analysis"])
                if len(self.results) > 50:
                    self.results.pop(next(iter(self.results)))
                self.jobs[job_id].update(status="done", result=result, progress=result["status"],
                                         request_budget=result["request_budget"])
        except Exception as exc:
            with self.lock:
                self.jobs[job_id].update(status="error", error=f"분석을 완료하지 못했습니다: {type(exc).__name__}: {safe_data(str(exc))}")
        finally:
            if index is not None:
                index.close()
            with self.lock:
                self.active = None
                while len(self.jobs) > 20:
                    self.jobs.pop(next(iter(self.jobs)))

    def _save(self, result):
        directory = self.output_dir / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + secrets.token_hex(4))
        directory.mkdir(parents=True, exist_ok=False)
        with (directory / "analysis.json").open("x", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        with (directory / "analysis.txt").open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(result["scope"], ensure_ascii=False) + "\n\n" + result["analysis"])

    def dashboard(self):
        path = self.dashboard_path
        if path.is_file() and path.resolve().parent in (self.root, self.root / REALTIME_DIRECTORY) and path.stat().st_size <= 2 * 1024 * 1024:
            document = path.read_text(encoding="utf-8")
        else:
            document = '<!doctype html><html lang="ko"><meta charset="utf-8"><title>항목별 LLM 분석</title><body style="background:#0f172a;color:white;font-family:&quot;Noto Sans KR&quot;,sans-serif"><main style="max-width:1080px;margin:auto"></main></body></html>'
        document = apply_web_font(install_panel(document), path)
        return document.encode("utf-8")

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=True)
        with self.lock:
            self.index.close()


def make_server(app, port=0):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass  # Do not log capability URLs, user queries or credentials.

        def route(self):
            expected = f"127.0.0.1:{self.server.server_port}"
            if self.headers.get("Host") != expected:
                raise PermissionError("Invalid Host")
            origin = self.headers.get("Origin")
            if origin and origin != "http://" + expected:
                raise PermissionError("Cross-origin requests are not allowed")
            split = urlsplit(self.path)
            prefix = "/s/" + app.token + "/"
            if not split.path.startswith(prefix):
                raise PermissionError("Invalid session")
            app.last_activity = time.monotonic()
            return unquote(split.path[len(prefix):]), parse_qs(split.query)

        def send(self, value, status=200, content_type="application/json; charset=utf-8"):
            data = value if isinstance(value, bytes) else json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy", "frame-ancestors 'self'")
            try:
                self.end_headers()
                if self.command != "HEAD":
                    self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass

        def do_GET(self):
            try:
                path, query = self.route()
                if path == "api/models" and isinstance(app, AnalysisApp):
                    self.send(app.models())
                elif path == "api/verify-models" and app.verification is not None:
                    models = app.models()
                    if isinstance(app, AnalysisApp):
                        models = {**models, "default_model": app.model}
                    self.send(models)
                elif path == "api/verify-saved-analyses" and app.verification is not None:
                    self.send(app.verification.saved_analyses(query.get("id", [None])[0],
                                                              query.get("run_id", [None])[0]))
                elif path == "api/verify-catalog" and app.verification is not None:
                    self.send({**app.verification.catalog(query.get("group", [None])[0]), "csrf": app.csrf,
                               "batch_analysis_available": app.batch_analysis_available})
                elif path == "api/verify-job" and app.verification is not None:
                    self.send(app.verification.job(query.get("id", [""])[0]))
                elif path == "api/verify-history" and app.verification is not None:
                    self.send(app.verification.session_history())
                elif path == "api/verify-batch" and app.verification is not None:
                    self.send(app.verification.batch_status(query.get("id", [""])[0]))
                elif path == "api/info" and isinstance(app, AnalysisApp):
                    self.send(app.info())
                elif path == "api/result" and isinstance(app, AnalysisApp):
                    key = scope_key(json.loads(query.get("scope", ["{}"]) [0]))
                    with app.lock:
                        self.send({"result": app.results.get(key)})
                elif path == "api/job" and isinstance(app, AnalysisApp):
                    with app.lock:
                        job = app.jobs.get(query.get("id", [""])[0])
                        self.send(job if job else {"error": "Job not found"}, 200 if job else 404)
                elif path in ("", "index.html", "relu_realtime.html"):
                    self.send(app.dashboard(), content_type="text/html; charset=utf-8")
                else:
                    if app.root is None:
                        self.send({"error": "File not found"}, 404)
                        return
                    assets = app.dashboard_path.parent
                    file = (assets / path).resolve()
                    if not file.is_relative_to(assets) or file.suffix.lower() not in {".html", ".png", ".js", ".css", ".svg", ".ico", ".ttf"}:
                        raise PermissionError("File not allowed")
                    if not file.is_file():
                        self.send({"error": "File not found"}, 404)
                    else:
                        with file.open("rb") as stream:
                            stat = os.fstat(stream.fileno())
                            revalidated = file.suffix.lower() in {".png", ".svg", ".js", ".html"}
                            etag = f'"{stat.st_mtime_ns:x}-{stat.st_size:x}"' if revalidated else None
                            if etag and self.headers.get("If-None-Match") == etag:
                                self.send_response(304)
                                self.send_header("ETag", etag)
                                self.send_header("Cache-Control", "private, no-cache")
                                self.end_headers()
                                return
                            self.send_response(200)
                            self.send_header("Content-Type", mimetypes.guess_type(file.name)[0] or "application/octet-stream")
                            self.send_header("Content-Length", str(stat.st_size))
                            self.send_header("Cache-Control", "private, no-cache" if revalidated else "private, max-age=31536000, immutable" if file.suffix.lower() == ".ttf" else "no-store")
                            if etag:
                                self.send_header("ETag", etag)
                            self.send_header("Referrer-Policy", "no-referrer")
                            self.send_header("X-Content-Type-Options", "nosniff")
                            self.send_header("Content-Security-Policy", "frame-ancestors 'self'")
                            self.end_headers()
                            if self.command == "HEAD":
                                return
                            try:
                                while chunk := stream.read(64 * 1024):
                                    self.wfile.write(chunk)
                            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                                pass
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass
            except PermissionError as exc:
                self.send({"error": str(exc)}, 403)
            except RuntimeError as exc:
                self.send({"error": str(exc)}, 503)
            except (ValueError, TypeError, OSError):
                self.send({"error": "Invalid request or unavailable file"}, 400)

        def do_HEAD(self):
            self.do_GET()

        def do_POST(self):
            try:
                path, _ = self.route()
                if not (path == "api/open-ai-studio-quota"
                        or (isinstance(app, AnalysisApp) and path in ("api/analyze", "api/close"))
                        or (path == "api/verify-shutdown" and hasattr(app, "shutdown_launcher"))
                        or (path == "api/verify-followup" and hasattr(app, "shutdown_launcher"))
                        or (path in ("api/verify-start", "api/verify-preferences", "api/verify-batch",
                                     "api/verify-batch-analysis", "api/verify-open-dashboard", "api/verify-stop")
                            and app.verification is not None)):
                    self.send({"error": "Not found"}, 404)
                    return
                if not secrets.compare_digest(self.headers.get("X-Analysis-Token", ""), app.csrf):
                    raise PermissionError("Invalid CSRF token")
                if self.headers.get_content_type() != "application/json":
                    raise ValueError("JSON required")
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 128 * 1024:
                    raise ValueError("Request body too large or empty")
                body = json.loads(self.rfile.read(length))
                if path == "api/open-ai-studio-quota":
                    try:
                        opened = webbrowser.open(AI_STUDIO_QUOTA_URL, new=2)
                    except (OSError, webbrowser.Error):
                        opened = False
                    if opened:
                        self.send({"opened": True})
                    else:
                        self.send({"error": "기본 브라우저를 열 수 없습니다. AI Studio 주소를 기본 브라우저에 직접 붙여 넣어 주세요."}, 503)
                elif path == "api/close":
                    app.shutdown_requested.set()
                    self.send({"status": "closing"})
                elif path == "api/verify-shutdown":
                    app.shutdown_launcher()
                    self.send({"status": "closed"})
                    app.shutdown_requested.set()
                elif path == "api/verify-followup":
                    self.send(app.verification.follow_up_saved_analysis(body))
                elif path == "api/verify-start":
                    self.send(app.verification.start(body), 202)
                elif path == "api/verify-batch":
                    self.send(app.verification.start_batch(body), 202)
                elif path == "api/verify-batch-analysis":
                    self.send(app.start_batch_analysis(body), 202)
                elif path == "api/verify-open-dashboard":
                    self.send(app.verification.open_dashboard(body))
                elif path == "api/verify-stop":
                    self.send(app.verification.stop(body), 202)
                elif path == "api/verify-preferences":
                    app.verification.save_preferences(body)
                    self.send({"saved": True})
                else:
                    self.send(app.start(body), 202)
            except PermissionError as exc:
                self.send({"error": str(exc)}, 403)
            except (ValueError, TypeError) as exc:
                self.send({"error": str(exc)}, 400)
            except RuntimeError as exc:
                self.send({"error": str(exc)}, 409)
            except OSError:
                error = ("검증 프로세스를 시작하지 못했습니다." if path == "api/verify-start"
                         else "파일을 읽거나 쓰지 못했습니다. 다시 요청해 주세요.")
                self.send({"error": error}, 503)

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    return server


def run_server_loop(server, app, *, idle_seconds=None):
    """Also keep auto-launched bridges useful after the solver process exits."""
    server.timeout = 1
    while not app.shutdown_requested.is_set():
        if (idle_seconds is not None and not app.active
                and not (app.verification is not None and app.verification.batch_active)
                and time.monotonic() - app.last_activity > idle_seconds):
            break
        server.handle_request()


def serve(args, key):
    from Automation.VerificationLauncher import VerificationLauncher

    app = AnalysisApp(args.run, model=args.model, max_calls=args.max_calls, max_log_mb=args.max_log_mb,
                      output_dir=args.output_dir,
                      verification=VerificationLauncher(),
                      batch_max_calls=args.retries + 1,
                      model_catalog=lambda: GeminiClient(key, args.model, timeout=12, retries=0).list_models(),
                      model_client_factory=lambda model, progress: GeminiClient(
                          key, model, args.request_timeout, args.retries, progress, "default"),
                      client_factory=lambda progress: GeminiClient(key, args.model, args.request_timeout,
                                                                   args.retries, progress, args.thinking_level))
    server = None
    try:
        server = make_server(app, args.port)
        url = f"http://127.0.0.1:{server.server_port}/s/{app.token}/"
        print(f"항목별 LLM 분석 대시보드: {url}", flush=True)
        print("브라우저에서 항목과 범위를 선택하고 전송 동의 후 분석 버튼을 누르세요. 종료: Ctrl+C", flush=True)
        if args.open:
            webbrowser.open(url)
        run_server_loop(server, app)
    except KeyboardInterrupt:
        print("서버 종료 중입니다. 진행 중인 API 요청이 있다면 응답/시간 제한 후 종료됩니다.", flush=True)
    finally:
        if server is not None:
            server.server_close()
        app.close()
    return 0


def automatic_main(argv=None):
    """Internal child entry point; existing verification commands need no new flags."""
    import argparse
    from Automation.VerificationLauncher import VerificationLauncher
    from visualization.AtomicFile import write_atomic_text
    parser = argparse.ArgumentParser()
    parser.add_argument("--auto-run", type=Path, required=True)
    parser.add_argument("--dashboard", required=True)
    parser.add_argument("--ready-file", type=Path, required=True)
    args = parser.parse_args(argv)
    app = server = None
    try:
        client = GeminiClient(load_gemini_api_key(), DEFAULT_MODEL)
        app = AnalysisApp(args.auto_run, model=DEFAULT_MODEL, dashboard=args.dashboard,
                          verification=VerificationLauncher(),
                          model_catalog=lambda: GeminiClient(client.key, timeout=12, retries=0).list_models(),
                          model_client_factory=lambda model, progress: GeminiClient(
                              client.key, model, progress=progress, thinking_level="default"),
                          client_factory=lambda progress: GeminiClient(client.key, DEFAULT_MODEL, progress=progress))
        server = make_server(app)
        url = f"http://127.0.0.1:{server.server_port}/s/{app.token}/"
        write_atomic_text(args.ready_file, json.dumps({"status": "ready", "url": url}))
        # Browser polling keeps the bridge alive. Closed tabs leave no permanent process.
        run_server_loop(server, app, idle_seconds=30 * 60)
        return 0
    except Exception:
        # No exception text/credentials are passed through the launch handshake.
        write_atomic_text(args.ready_file, json.dumps({"status": "unavailable"}))
        return 1
    finally:
        if server is not None:
            server.server_close()
        if app is not None:
            app.close()


if __name__ == "__main__":
    raise SystemExit(automatic_main())
