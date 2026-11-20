"""Automatically prepare a private LLM bridge for the existing realtime command.

No API requests here. The child uses the same local key settings. The key never
appears in arguments, a readiness file or the dashboard. Failures fall back to file://.
"""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from urllib.parse import urlsplit
import webbrowser


from Automation.AnalyzeLogs import DEFAULT_MODEL, GeminiClient, load_gemini_api_key
from visualization.VisualizationMode import REALTIME_DIRECTORY
_children = []


def _stop_child(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)


def start_analysis_bridge(run, dashboard, *, startup_timeout=5):
    key = load_gemini_api_key()
    if not key.strip():
        return {"status": "unavailable", "reason": "missing_key"}
    try:
        GeminiClient(key, DEFAULT_MODEL)  # Validate locally, without contacting Google.
    except ValueError:
        return {"status": "unavailable", "reason": "invalid_key"}
    root, page = Path(run).resolve(), Path(dashboard).resolve()
    if page.parent not in (root, root / REALTIME_DIRECTORY):
        return {"status": "unavailable", "reason": "external_dashboard"}
    process = None
    try:
        with tempfile.TemporaryDirectory(prefix="verification-llm-start-") as temporary:
            ready = Path(temporary) / "ready.json"
            command = [sys.executable, "-m", "Automation.LogAnalysisServer", "--auto-run", str(root),
                       "--dashboard", str(page.relative_to(root)), "--ready-file", str(ready)]
            process = subprocess.Popen(
                command, cwd=str(Path(__file__).resolve().parent.parent),
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                close_fds=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            deadline = time.monotonic() + startup_timeout
            while time.monotonic() < deadline:
                if ready.is_file():
                    response = json.loads(ready.read_text(encoding="utf-8"))
                    url = urlsplit(response.get("url", ""))
                    if (response.get("status") == "ready" and url.scheme == "http"
                            and url.hostname == "127.0.0.1" and url.port
                            and url.path.startswith("/s/") and url.path.endswith("/")):
                        _children[:] = [child for child in _children if child.poll() is None]
                        _children.append(process)
                        return {"status": "ready", "url": response["url"], "model": DEFAULT_MODEL}
                    break
                if process.poll() is not None:
                    break
                time.sleep(0.05)
        _stop_child(process)
        return {"status": "unavailable", "reason": "startup_failed"}
    except (OSError, ValueError, subprocess.SubprocessError):
        if process is not None:
            _stop_child(process)
        return {"status": "unavailable", "reason": "startup_failed"}
    except BaseException:
        if process is not None:
            _stop_child(process)
        raise


def open_dashboard_with_analysis(run, dashboard, *, open_page=None):
    """Open exactly one existing dashboard, through HTTP when the key is ready."""
    connection = start_analysis_bridge(run, dashboard)
    target = connection.get("url") or Path(dashboard).resolve().as_uri()
    try:
        opened = bool(webbrowser.open(target, new=2) if open_page is None else open_page(target))
    except (OSError, webbrowser.Error):
        opened = False
    if not opened:
        print("시각화 탭을 자동으로 열지 못했습니다. 검증은 계속 진행합니다.", flush=True)
    elif connection["status"] == "ready":
        print(f"LLM 분석 자동 연결: {target} (분석 버튼을 누를 때만 전송)", flush=True)
    elif connection["reason"] == "missing_key":
        print("LLM 분석: GEMINI_API_KEY가 없어 기존 시각화로 엽니다.", flush=True)
    elif connection["reason"] == "invalid_key":
        print("LLM 분석: API 키 형식을 확인해 주세요. 검증과 기존 시각화는 계속 진행합니다.", flush=True)
    else:
        print("LLM 분석 자동 연결을 준비하지 못해 기존 시각화로 엽니다. 검증은 계속 진행합니다.", flush=True)
    return {**{key: value for key, value in connection.items() if key != "url"}, "browser_opened": opened}
