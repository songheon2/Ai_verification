"""Local ONNX/VNNLIB picker for the verification dashboard."""
from dataclasses import dataclass
from datetime import datetime
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory, TemporaryFile
from threading import Event, Lock, Thread
from time import monotonic, sleep
import webbrowser

from Reluplex import DEFAULT_FEATURES, FEATURES, feature_options
from Automation.AnalysisConversation import AnalysisConversation
from Automation.BatchLogSnapshot import capture_log_limits, summarize_logs
from visualization.AtomicFile import write_atomic_text
from visualization.VisualizationMode import REALTIME_DIRECTORY, VERIFICATION_PANELS, VisualizationMode, resolve_visualization_output

WORKSPACE = Path(__file__).resolve().parents[1]
ENGINES = ("glpk", "native", "z3")
NUMBER_PART = re.compile(r"([0-9]+)")
CATALOG_ID = re.compile(r"[0-9a-f]{24}")
RUN_ANALYSIS_ID = re.compile(r"[0-9]{8}_[0-9]{6}_[0-9a-f]{8}")
PREFERENCE_FIELDS = {"group", "model", "property", "engine", "timeout"}
MAX_BATCH_RUNS = 50
RUN_OUTPUT_MARKER = "VERIFICATION_OUTPUT "
DEFAULT_RUN_SETTINGS = {
    "visualization_mode": "both",
    "panels": list(VERIFICATION_PANELS),
    "simplex_max_iter": 1_000_000,
    "max_rounds": 10_000,
    "max_recursion": 500,
    "lp_mode": "core",
    "features": list(DEFAULT_FEATURES),
    "under_approx_eps": 0.1,
}


def _run_settings(custom):
    if not isinstance(custom, dict) or custom.keys() - DEFAULT_RUN_SETTINGS.keys() - {"open_dashboard"}:
        raise ValueError("지원하는 검증 설정만 선택해 주세요.")
    settings = {**DEFAULT_RUN_SETTINGS, **{key: value for key, value in custom.items() if key != "open_dashboard"}}
    mode = settings["visualization_mode"]
    if mode not in tuple(item.value for item in VisualizationMode):
        raise ValueError("시각화 모드를 선택해 주세요.")
    for key, allowed in (("panels", VERIFICATION_PANELS), ("features", FEATURES)):
        selected = settings[key]
        if (not isinstance(selected, list) or len(selected) > len(allowed)
                or any(item not in allowed for item in selected) or len(set(selected)) != len(selected)):
            raise ValueError("패널과 검증 기법은 목록에서 선택해 주세요.")
    if mode != "off" and not settings["panels"]:
        raise ValueError("시각화 패널을 하나 이상 선택해 주세요.")
    if settings["lp_mode"] not in ("core", "oracle"):
        raise ValueError("LP 모드 설정을 확인해 주세요.")
    for key in ("simplex_max_iter", "max_rounds", "max_recursion"):
        value = settings[key]
        if type(value) is not int or not 1 <= value <= 2_147_483_647:
            raise ValueError("반복·라운드·깊이 제한을 양의 정수로 입력해 주세요.")
    eps = settings["under_approx_eps"]
    if type(eps) not in (int, float) or not 0 < eps < float("inf"):
        raise ValueError("과소근사 ε는 0보다 큰 유한한 수로 입력해 주세요.")
    return settings


def _batch_metrics(result):
    solver = result.get("solver", {})
    theory = solver.get("theory_stats", {})
    return {
        "network_layers": result.get("model", {}).get("layer_sizes"),
        "solver": {key: solver[key] for key in ("status", "reason", "elapsed_seconds", "rounds")
                   if key in solver},
        "simplex": result.get("solver_progress_summary"),
        "relu_splits": result.get("split_summary"),
        "reluplex": {key: theory[key] for key in (
            "SEARCH_NODES", "UNSAT_LEAVES", "CONFLICT_LEAVES", "LP_CALLS", "LP_SECONDS", "REPAIR_STEPS",
            "DERIVE_FIXED_RELUS", "DERIVE_INFEASIBLE", "SEARCH_DERIVE_CALLS", "SEARCH_DERIVE_SECONDS",
            "SEARCH_DERIVE_FIXED", "BACKJUMPS", "SIMPLEX_CYCLES_DETECTED", "PIVOT_TIGHTENED",
            "SIMPLEX_FULL_TIGHTEN", "UA_FIXED", "UA_SECONDS", "UA_RESULT_SAT", "UA_RESULT_UNSAT",
        ) if key in theory},
    }


def _natural_name_key(name):
    folded = name.casefold()
    parts = NUMBER_PART.split(folded)
    return tuple(int(part) if index % 2 else part for index, part in enumerate(parts)), folded, name


def _saved_question_title(question, fallback):
    if not isinstance(question, str) or not question.strip():
        return fallback
    text = " ".join(question.split())
    first_sentence = re.split(r"(?<=[.!?。！？])\s+", text, maxsplit=1)[0]
    return first_sentence if len(first_sentence) <= 64 else first_sentence[:63].rstrip() + "…"


@dataclass(frozen=True)
class CatalogFile:
    path: Path
    root: Path
    label: str

    @property
    def id(self):
        return hashlib.sha256(str(self.path).encode("utf-8")).hexdigest()[:24]

    def public(self):
        return {"id": self.id, "label": self.label}


class VerificationLauncher:
    def __init__(self, workspace=WORKSPACE, benchmark_root=None, *, restore_preferences=True):
        self.workspace = Path(workspace).resolve()
        self.benchmark_root = (Path(benchmark_root) if benchmark_root is not None else
                               self.workspace.parent / "vnncomp2025_benchmarks" / "benchmarks").resolve()
        self.lock = Lock()
        self.current = None
        self.batch = None
        self.session_runs = []
        self.analysis_conversation = AnalysisConversation()
        self.browser = None
        self.closing = False
        self.batch_thread = None
        self.batch_log = None
        self.batch_process = None
        self.batch_worker_stopped = False
        self.batch_dashboards = {}
        self.group_files = {}
        self.session_preferences = None if restore_preferences else {}
        self.preferences_path = self.workspace / "Automation" / "Results" / "verification-picker.json"

    @staticmethod
    def _selection(body):
        if not isinstance(body, dict) or body.keys() - {"settings"} != PREFERENCE_FIELDS:
            raise ValueError("신경망과 속성을 목록에서 선택해 주세요.")
        group, model, prop, engine, timeout = (body[name] for name in (
            "group", "model", "property", "engine", "timeout"))
        if not all(isinstance(value, str) for value in (group, model, prop, engine)):
            raise ValueError("선택 값 형식이 올바르지 않습니다.")
        if (len(group) > 255 or any(value and not CATALOG_ID.fullmatch(value) for value in (model, prop))
                or (not group and (model or prop))):
            raise ValueError("신경망과 속성을 목록에서 선택해 주세요.")
        if engine not in ENGINES or type(timeout) is not int or not 1 <= timeout <= 43200:
            raise ValueError("지원 엔진과 1~43200초 범위의 제한 시간을 입력해 주세요.")
        _run_settings(body.get("settings", {}))
        return group, model, prop, engine, timeout

    def _read_preferences(self, groups):
        if self.session_preferences is not None:
            return self.session_preferences
        try:
            if not self.preferences_path.resolve().is_relative_to(self.workspace):
                return {}
            with self.preferences_path.open(encoding="utf-8") as stream:
                data = json.loads(stream.read(4096))
            group, *_ = self._selection(data)
        except (OSError, ValueError):
            return {}
        data.get("settings", {}).pop("open_dashboard", None)
        return data if not group or group in groups else {**data, "group": "", "model": "", "property": ""}

    def _write_preferences(self, data):
        if self.session_preferences is not None:
            self.session_preferences = data
            return
        if not self.preferences_path.resolve().is_relative_to(self.workspace):
            raise OSError("검증 선택 저장 위치가 작업공간 밖입니다.")
        write_atomic_text(self.preferences_path, json.dumps(data, ensure_ascii=False) + "\n")

    def save_preferences(self, body):
        group, model, prop, _, _ = self._selection(body)
        if group and group not in self._groups():
            raise ValueError("벤치마크 목록에서 선택해 주세요.")
        with self.lock:
            files = self.group_files.get(group, {})
            if ((model and model not in files.get("models", {}))
                    or (prop and prop not in files.get("properties", {}))):
                raise ValueError("선택한 파일이 목록에 없습니다.")
            self._write_preferences(body)

    def _groups(self):
        groups = {"local": (self.workspace / "Onnx", self.workspace / "Automation" / "SmokeTest",
                            self.workspace / "Vnnlib", self.workspace / "Properties")}
        if self.benchmark_root.is_dir():
            for folder in sorted(self.benchmark_root.iterdir(), key=lambda path: _natural_name_key(path.name)):
                if folder.is_dir() and folder.resolve().parent == self.benchmark_root:
                    groups[folder.name] = (folder,)
        return groups

    def _options(self, group, roots):
        options = {"models": {}, "properties": {}}
        for declared_root in roots:
            root = declared_root.resolve()
            scope = self.workspace if group == "local" else self.benchmark_root
            if not root.is_dir() or not root.is_relative_to(scope):
                continue
            for folder, dirs, names in os.walk(root):
                dirs[:] = sorted(d for d in dirs
                                  if (Path(folder) / d).resolve().is_relative_to(root)
                                  and not (Path(folder) / d).is_symlink())
                for name in sorted(names):
                    kind = ("models" if name.lower().endswith((".onnx", ".onnx.gz")) else
                            "properties" if name.lower().endswith((".vnnlib", ".vnnlib.gz")) else None)
                    if kind is None:
                        continue
                    source = Path(folder) / name
                    path = source.resolve()
                    if not path.is_relative_to(root) or not path.is_file():
                        continue
                    label = source.relative_to(root).as_posix()
                    if group == "local":
                        label = f"{declared_root.name}/{label}"
                    # Keep the unpacked file when the same relative file has both forms.
                    logical = label.removesuffix(".gz")
                    if logical not in options[kind] or not label.endswith(".gz"):
                        options[kind][logical] = CatalogFile(path, root, label)
        return {kind: {files[name].id: files[name] for name in sorted(files, key=_natural_name_key)}
                for kind, files in options.items()}

    def catalog(self, group=None):
        groups = self._groups()
        if group is None:
            with self.lock:
                batch_id = self.batch["id"] if self.batch is not None else None
                single_id = (self.current["id"] if self.current is not None
                             and self.current["process"].poll() is None else None)
            return {"groups": [{"id": key, "label": "로컬 파일" if key == "local" else key}
                               for key in groups], "engines": list(ENGINES),
                    "preferences": self._read_preferences(groups), "max_batch_runs": MAX_BATCH_RUNS,
                    "batch_id": batch_id, "single_id": single_id}
        if group not in groups:
            raise ValueError("벤치마크 목록에서 선택해 주세요.")
        files = self._options(group, groups[group])
        with self.lock:
            self.group_files[group] = files
        return {kind: [entry.public() for entry in entries.values()] for kind, entries in files.items()}

    def _resolve(self, body):
        group, model_id, prop_id, engine, timeout = self._selection(body)
        files = self.group_files.get(group, {})
        model = files.get("models", {}).get(model_id)
        prop = files.get("properties", {}).get(prop_id)
        for entry in (model, prop):
            if (entry is None or entry.path.resolve() != entry.path
                    or not entry.path.is_relative_to(entry.root) or not entry.path.is_file()):
                raise ValueError("선택한 파일이 목록에 없습니다. 목록을 다시 열어 주세요.")
        return model, prop, engine, timeout, body.get("settings", {})

    def _require_idle(self):
        if self.closing:
            raise RuntimeError("검증기를 종료하고 있습니다.")
        if (self.current is not None and self.current["process"].poll() is None
                or self.batch is not None and (self.batch["status"] in ("running", "stopping")
                                              or self.batch["analysis"]["status"] == "running")):
            raise RuntimeError("이 화면에서 시작한 검증 또는 통합 분석이 이미 실행 중입니다.")

    @property
    def batch_active(self):
        with self.lock:
            return self.batch is not None and (
                self.batch["status"] in ("running", "stopping") or self.batch["analysis"]["status"] == "running")

    def _launch(self, model, prop, engine, timeout, settings):
        command = [sys.executable, "-m", "Automation.VerificationLauncher"]
        log = TemporaryFile()
        try:
            log.state_directory = TemporaryDirectory(prefix="verification-worker-", ignore_cleanup_errors=True)
        except OSError:
            log.close()
            raise
        command.extend(("--worker-state", str(Path(log.state_directory.name) / "state.json")))
        browser = self.browser
        if browser is not None:
            command.append("--managed-browser")
        if settings:
            command.extend(("--worker-settings", json.dumps(settings)))
        command.extend(("--worker", str(model.path), str(prop.path), engine, str(timeout)))
        config = dict(os.environ, PYTHONIOENCODING="utf-8")
        try:
            process = subprocess.Popen(command, cwd=self.workspace, env=config,
                                       stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                       close_fds=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except OSError:
            self._close_log(log)
            raise
        if browser is not None:
            log.browser_thread = Thread(target=self._open_worker_dashboard, args=(browser, process, log),
                                        name="verification-tab", daemon=True)
            log.browser_thread.start()
        return process, log

    def _open_worker_dashboard(self, browser, process, log):
        # The worker publishes its URL after the realtime page and its optional
        # LLM connection are ready. The launcher owns the resulting browser tab.
        while True:
            exited = process.poll() is not None
            state = self._worker_state(log)
            if state.get("dashboard") and state.get("dashboard_url"):
                browser.open(state["dashboard_url"])
                return
            if exited:
                return
            sleep(.1)

    @staticmethod
    def _close_log(log):
        if hasattr(log, "browser_thread"):
            log.browser_thread.join(timeout=3)
        log.close()
        if hasattr(log, "state_directory"):
            log.state_directory.cleanup()

    def _worker_state(self, log):
        if log is None or not hasattr(log, "state_directory"):
            return {}
        try:
            path = Path(log.state_directory.name) / "state.json"
            state = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(state, dict):
                return {}
            root = (self.workspace / "visualization" / "outputs").resolve()
            if state.get("directory"):
                directory = Path(state["directory"]).resolve()
                if directory.is_dir() and root.is_relative_to(self.workspace) and directory.is_relative_to(root):
                    state["directory"] = directory
                else:
                    state.pop("directory")
            if state.get("dashboard"):
                dashboard = Path(state["dashboard"]).resolve()
                if (root.is_relative_to(self.workspace) and dashboard.is_relative_to(root)
                        and dashboard.suffix.lower() == ".html" and dashboard.is_file()):
                    return {**state, "dashboard": dashboard}
                state.pop("dashboard")
            return state
        except (OSError, ValueError):
            return {}

    def _outcome(self, code, stream, *, stopped=False):
        size = stream.seek(0, os.SEEK_END)
        stream.seek(max(0, size - 8192))
        lines = stream.read(8192).decode("utf-8", errors="replace").splitlines()
        if code != 0:
            diagnostic = {"code": "WORKER_EXIT", "exit_code": code}
            message = lines[-1][:500] if lines else f"Exit code {code}"
            if any("alloc.c at line 70" in line and "env" in line for line in lines):
                diagnostic["code"] = "GLPK_ALLOCATOR_ACCOUNTING_ERROR"
                message = "GLPK 메모리 관리 오류 (alloc.c:70)"
            unsupported = re.match(
                r"ValueError: (?:hidden activation |unsupported ONNX operation '|"
                r"VNNLIB direct verification does not support a removed final activation: )([A-Za-z0-9_]+)",
                message,
            )
            if unsupported:
                diagnostic.update(code="UNSUPPORTED_ONNX_OPERATION", operation=unsupported[1])
            outcome = ({"status": "cancelled", "diagnostic": {"code": "USER_CANCELLED"}}
                       if stopped else {"status": "error", "error": message, "diagnostic": diagnostic})
            stream.seek(0)
            header = stream.read(65536).decode("utf-8", errors="replace")
            directory = next((line.removeprefix(RUN_OUTPUT_MARKER) for line in header.splitlines()
                              if line.startswith(RUN_OUTPUT_MARKER)), "")
            if directory:
                self._load_failed_checkpoint(directory, outcome)
            return outcome
        for line in reversed(lines):
            if line.startswith("VERIFICATION_RESULT "):
                try:
                    return {"status": "complete", **json.loads(line.removeprefix("VERIFICATION_RESULT "))}
                except ValueError:
                    break
        return {"status": "error", "error": "검증 결과를 읽을 수 없습니다."}

    def _finish_worker(self, code, log, stopped):
        outcome = self._outcome(code, log, stopped=stopped)
        if outcome["status"] != "cancelled":
            return outcome
        from visualization.UnifiedRealtimeDashboard import publish_dashboard_status

        try:
            manifest_path = outcome.get("run_manifest_path")
            if manifest_path:
                path = Path(manifest_path)
                manifest = json.loads(path.read_text(encoding="utf-8"))
                manifest.update(status="CANCELLED", phase="CANCELLED",
                                finished_at=datetime.now().astimezone().isoformat())
                write_atomic_text(path, json.dumps(manifest, ensure_ascii=False))
            dashboard = self._worker_state(log).get("dashboard")
            if dashboard:
                publish_dashboard_status(dashboard, "CANCELLED")
        except (OSError, ValueError):
            pass
        return outcome

    def _load_failed_checkpoint(self, directory, outcome):
        try:
            target = (Path(json.loads(directory)) / "run_info.json").resolve()
            root = (self.workspace / "visualization" / "outputs").resolve()
            if not target.is_relative_to(root) or not root.is_relative_to(self.workspace):
                return
            with target.open(encoding="utf-8") as stream:
                manifest = json.loads(stream.read(1024 * 1024))
        except (OSError, TypeError, ValueError):
            return
        if not isinstance(manifest, dict):
            return
        phase = manifest.get("phase")
        if phase:
            outcome["diagnostic"]["last_saved_phase"] = phase
        outcome["run_manifest_path"] = str(target)
        if isinstance(manifest.get("solver_result"), dict):
            outcome["solver_checkpoint"] = _batch_metrics({
                "model": manifest.get("model", {}), "solver": manifest["solver_result"],
            })

    def start(self, body):
        with self.lock:
            self._require_idle()
            entry = self._resolve(body)
            try:
                self._write_preferences(body)
            except OSError:
                saved = False
            else:
                saved = True
            if self.current is not None:
                self._refresh_single_outcome(self.current)
            model, prop, *_ = entry
            session_run = self._start_session_run(model.label, prop.label)
            try:
                process, log = self._launch(*entry)
            except (OSError, ValueError, subprocess.SubprocessError):
                self._finish_session_run(session_run, {"status": "error"})
                raise
            if self.current is not None:
                self._close_log(self.current["log"])
            job_id = secrets.token_hex(12)
            self.current = {"id": job_id, "process": process, "log": log, "session_run": session_run}
            return {"id": job_id, "status": "running", "preferences_saved": saved}

    def _start_session_run(self, model, prop):
        run = {"number": len(self.session_runs) + 1, "model": model, "property": prop,
               "started_at": datetime.now().astimezone().isoformat(), "started": monotonic(),
               "status": "running", "result": None}
        self.session_runs.append(run)
        return run

    @staticmethod
    def _finish_session_run(run, outcome):
        observed = monotonic()
        finished = outcome.pop("finished_monotonic", observed)
        if not isinstance(finished, (int, float)) or not run["started"] <= finished <= observed:
            finished = observed
        solver = (outcome.get("metrics") or {}).get("solver") or {}
        result = "TIMEOUT" if solver.get("reason") == "TIMEOUT" else outcome.get("verdict")
        run.update(status=outcome["status"], result=result, finished=finished)

    def _finish_single(self, job, code):
        if "outcome" not in job:
            outcome = self._finish_worker(code, job["log"], job.get("stopped", False))
            self._finish_session_run(job["session_run"], outcome)
            job["outcome"] = outcome

    def _refresh_single_outcome(self, job):
        code = job["process"].poll()
        if code is not None:
            self._finish_single(job, code)
        return code

    def session_history(self):
        with self.lock:
            if self.current is not None:
                self._refresh_single_outcome(self.current)
            now = monotonic()
            return {"runs": [{**{key: run[key] for key in ("number", "model", "property", "started_at", "status", "result")},
                              "elapsed_seconds": max(0.0, run.get("finished", now) - run["started"])}
                             for run in self.session_runs]}

    def job(self, job_id):
        with self.lock:
            if self.current is None or self.current["id"] != job_id:
                raise ValueError("검증 실행을 찾을 수 없습니다.")
            log = self.current["log"]
            state = self._worker_state(log)
            progress = {"phase": state.get("phase", "starting"),
                        "dashboard_ready": bool(state.get("dashboard"))}
            code = self._refresh_single_outcome(self.current)
            if code is None:
                return {"status": "stopping" if self.current.get("stopped") else "running", **progress}
            return {**self.current["outcome"], **progress}

    @staticmethod
    def _terminate(process):
        if process is None or process.poll() is not None:
            return False
        try:
            process.terminate()
        except OSError:
            if process.poll() is None:
                raise RuntimeError("실행 중인 검증을 중단하지 못했습니다.")
            return False
        return True

    def stop(self, body):
        if not isinstance(body, dict) or not isinstance(body.get("id"), str):
            raise ValueError("실행 정보를 찾을 수 없습니다.")
        job_id = body["id"]
        with self.lock:
            if self.current is not None and self.current["id"] == job_id:
                self._stop_single()
                return {"status": "stopping" if self.current.get("stopped") else "complete"}
            batch = self._get_batch(job_id)
            self._stop_batch(batch)
            return {"status": batch["status"]}

    def _stop_single(self):
        if self.current is not None and not self.current.get("stopped"):
            self.current["stopped"] = self._terminate(self.current["process"])

    def _stop_batch(self, batch):
        if batch["status"] == "running":
            self.batch_worker_stopped = self._terminate(self.batch_process)
            batch["status"] = "stopping"
            for run in batch["runs"]:
                if run["status"] == "waiting":
                    run["status"] = "cancelled"
            self._save_batch(batch)

    def shutdown(self):
        with self.lock:
            self._stop_single()
            if self.batch is not None:
                self._stop_batch(self.batch)
                if self.batch["analysis"]["status"] == "running":
                    self.batch["analysis"] = {"status": "cancelled"}
                    self._save_batch(self.batch)
            self.closing = True
            single, batch_thread = self.current, self.batch_thread

        if batch_thread is not None:
            batch_thread.join(timeout=5)
            if batch_thread.is_alive():
                raise RuntimeError("대기열을 정리하고 있습니다. 잠시 후 다시 종료해 주세요.")
        if single is not None:
            try:
                code = single["process"].wait(timeout=5)
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError("검증을 중단하고 있습니다. 잠시 후 다시 종료해 주세요.") from exc
            with self.lock:
                if self.current is single:
                    self._finish_single(single, code)
                    self._close_log(single["log"])
                    self.current = None

    def start_batch(self, body):
        selections = body.get("runs") if isinstance(body, dict) else None
        if not isinstance(selections, list) or not 1 <= len(selections) <= MAX_BATCH_RUNS:
            raise ValueError(f"대기열에는 검증을 1~{MAX_BATCH_RUNS}개 담을 수 있습니다.")
        with self.lock:
            self._require_idle()
            entries = [self._resolve(selection) for selection in selections]
            batch = {
                "id": secrets.token_hex(12), "status": "running", "analysis": {"status": "idle"},
                "runs": [{"number": number, "status": "waiting", "benchmark": selection["group"],
                          "model": model.label, "property": prop.label, "engine": engine,
                          "timeout_seconds": timeout, "settings": _run_settings(settings)}
                         for number, ((model, prop, engine, timeout, settings), selection)
                         in enumerate(zip(entries, selections), 1)],
            }
            self.batch = batch
            self.batch_dashboards.clear()
            self._save_batch(batch)
            self.batch_thread = Thread(target=self._run_batch, args=(batch, entries),
                                       name="verification-batch", daemon=True)
            self.batch_thread.start()
            return {"id": batch["id"], "status": batch["status"]}

    @property
    def batch_directory(self):
        return self.workspace / "Automation" / "Results" / "verification_batches"

    def _batch_path(self, batch_id):
        if not CATALOG_ID.fullmatch(batch_id):
            raise ValueError("저장된 검증 묶음을 찾을 수 없습니다.")
        path = (self.batch_directory / batch_id / "batch.json").resolve()
        if (not path.is_relative_to(self.workspace)
                or path.parent != self.batch_directory.resolve() / batch_id or path.name != "batch.json"):
            raise OSError("Batch report directory is outside the workspace")
        return path

    def _save_batch(self, batch):
        try:
            path = self._batch_path(batch["id"])
            write_atomic_text(path, json.dumps(batch, ensure_ascii=False, indent=2) + "\n")
        except OSError:
            batch["save_error"] = "통합 결과를 파일에 저장하지 못했습니다."
        else:
            batch.pop("save_error", None)

    def _read_batch(self, batch_id):
        try:
            path = self._batch_path(batch_id)
            with path.open(encoding="utf-8") as stream:
                data = json.loads(stream.read(2 * 1024 * 1024))
            modified = datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat()
        except (OSError, ValueError):
            return None
        if not isinstance(data, dict) or data.get("id") != batch_id:
            return None
        output_root = self.workspace / "visualization" / "outputs"
        runs = data.get("runs")
        for run in runs if isinstance(runs, list) else ():
            if not isinstance(run, dict):
                continue
            for key in ("result_path", "run_manifest_path", "run_log_directory"):
                value = run.get(key)
                if isinstance(value, str) and value and not Path(value).exists():
                    resolved = resolve_visualization_output(value, output_root)
                    if resolved.exists():
                        run[key] = str(resolved)
        return data, modified

    def follow_up_saved_analysis(self, body):
        selection = body.get("id") if isinstance(body, dict) else None
        if not isinstance(selection, str):
            raise ValueError("후속 질문의 기준이 될 저장 결과를 선택해 주세요.")
        overview = False
        if selection.startswith("run:"):
            report = self._read_run_analysis(selection[4:])
            if report is None:
                raise ValueError("저장된 개별 분석을 찾을 수 없습니다.")
            runs = []
            sources = [{"network": report["network"], "property": report["property"]}]
            status = "complete"
        else:
            loaded = self._read_batch(selection)
            report = self._saved_batch_analysis(*loaded) if loaded else None
            if report is None:
                raise ValueError("저장된 통합 분석을 찾을 수 없습니다.")
            batch, _ = loaded
            if batch.get("status") not in ("complete", "cancelled") or not isinstance(batch.get("runs"), list):
                raise ValueError("종료된 검증 묶음의 분석을 선택해 주세요.")
            runs, sources, status = batch["runs"], report["sources"], batch["status"]
            overview = bool(batch.get("overview"))
        source = {"title": report["title"], "sources": sources}
        prior = {key: report[key] for key in ("question", "report", "model", "saved_at") if key in report}
        prior["status"] = "complete"
        with self.lock:
            self._require_idle()
            self.analysis_conversation.clear()
            question = prior.get("question")
            self.analysis_conversation.remember(question if isinstance(question, str) else "", prior["report"])
            self.batch_dashboards.clear()
            self.batch = {"id": secrets.token_hex(12), "status": status, "runs": runs,
                          "analysis": prior, "followup_source": source, "overview": overview}
            return {"id": self.batch["id"]}

    @staticmethod
    def _saved_batch_analysis(data, modified):
        result = data.get("saved_analysis") or data.get("analysis")
        if not isinstance(result, dict) or result.get("status") != "complete" or not isinstance(result.get("report"), str):
            return None
        runs = data.get("runs")
        sources = []
        if isinstance(runs, list):
            for run in runs:
                if not isinstance(run, dict):
                    continue
                network = Path(run.get("model") or "").name
                prop = Path(run.get("property") or "").name
                source = {"network": network, "property": prop}
                if (network or prop) and source not in sources:
                    sources.append(source)
        if not sources and isinstance(data.get("followup_source"), dict):
            sources = data["followup_source"].get("sources", [])
        if not sources and data.get("overview"):
            sources = [{"network": "프로젝트 구조", "property": "전체"}]
        return {**result, "id": data["id"], "saved_at": result.get("saved_at") or modified,
                "title": _saved_question_title(result.get("question"), "프로젝트 구조" if data.get("overview") else "전체 실행 비교"),
                "sources": sources,
                "run_count": len(runs) if isinstance(runs, list) else 0}

    @property
    def analysis_directory(self):
        return self.workspace / "Automation" / "Results" / "llm_analysis"

    def _read_run_analysis(self, analysis_id):
        from Automation.LogAnalysisScope import MECHANISMS, TOPICS, normalize_scope

        if not RUN_ANALYSIS_ID.fullmatch(analysis_id):
            return None
        try:
            root = self.analysis_directory.resolve()
            path = (root / analysis_id / "analysis.json").resolve()
            if (not root.is_relative_to(self.workspace)
                    or path != root / analysis_id / "analysis.json"):
                return None
            with path.open(encoding="utf-8") as stream:
                data = json.loads(stream.read(2 * 1024 * 1024))
            modified = datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat()
        except (OSError, ValueError):
            return None
        if (not isinstance(data, dict) or data.get("status") not in ("complete", "incomplete")
                or not isinstance(data.get("scope"), dict) or not isinstance(data.get("run_fingerprint"), str)
                or not isinstance(data.get("analysis"), str) or not data["analysis"].strip()):
            return None
        summaries = data.get("runs")
        summary = summaries[0] if isinstance(summaries, list) and summaries and isinstance(summaries[0], dict) else {}
        context = summary.get("run_context_only") or summary.get("result") or {}
        if not isinstance(context, dict):
            context = {}
        network = context.get("model") or {}
        prop = context.get("vnnlib") or {}
        try:
            scope = normalize_scope(data["scope"])
        except ValueError:
            return None
        if "selections" in scope and len(scope["selections"]) > 1:
            scope_label = f"선택 항목 {len(scope['selections'])}개"
        else:
            item = scope["selections"][0] if "selections" in scope else scope
            parts = [TOPICS[item["topic"]]]
            if item.get("mechanism") not in (None, "all"):
                parts.append(MECHANISMS[item["mechanism"]])
            if "round" in item:
                parts.append(f"R{item['round']}")
            if "call_id" in item:
                parts.append(item["call_id"])
            scope_label = " · ".join(parts)
        return {"id": analysis_id, "saved_at": data.get("created_at") or modified,
                "report": data["analysis"], "status": data["status"], "model": data.get("model"),
                "question": data.get("question") if isinstance(data.get("question"), str) else "",
                "title": _saved_question_title(data.get("question"), scope_label + " 분석"),
                "network": Path(network.get("source_path") or network.get("name") or "").name,
                "property": prop.get("property_name") or Path(prop.get("path") or "").name,
                "run_name": summary.get("directory_name"), "scope": scope_label,
                "fingerprint": data["run_fingerprint"]}

    def _batch_run_sources(self, batches, fingerprints):
        from Automation.LogAnalysisIndex import live_run_fingerprint

        matches = {}
        root = (self.workspace / "visualization" / "outputs").resolve()
        if not fingerprints or not root.is_relative_to(self.workspace):
            return matches
        for batch, _ in batches:
            runs = batch.get("runs")
            if not isinstance(runs, list):
                continue
            for run in runs:
                if not isinstance(run, dict):
                    continue
                source = run.get("result_path") or run.get("run_manifest_path")
                if not source:
                    continue
                manifest = (Path(source).parent / "run_info.json").resolve()
                if not manifest.is_relative_to(root):
                    continue
                try:
                    with manifest.open(encoding="utf-8") as stream:
                        info = json.loads(stream.read(1024 * 1024))
                except (OSError, ValueError):
                    continue
                if not isinstance(info, dict):
                    continue
                key = live_run_fingerprint(manifest.parent, info.get("run_id"))
                if key in fingerprints:
                    matches[key] = {"batch_id": batch["id"], "number": run.get("number")}
        return matches

    def saved_analyses(self, batch_id=None, run_id=None):
        if run_id is not None:
            result = self._read_run_analysis(run_id)
            if result is None:
                raise ValueError("저장된 개별 분석을 찾을 수 없습니다.")
            return {"analysis": {key: value for key, value in result.items() if key != "fingerprint"}}
        if batch_id is not None:
            batch = self._read_batch(batch_id)
            result = self._saved_batch_analysis(*batch) if batch else None
            if result is None:
                raise ValueError("저장된 통합 분석을 찾을 수 없습니다.")
            return {"analysis": result}
        batches = []
        if self.batch_directory.resolve().is_relative_to(self.workspace):
            for path in self.batch_directory.glob("*/batch.json"):
                if batch := self._read_batch(path.parent.name):
                    batches.append(batch)
        summaries = []
        for batch in batches:
            result = self._saved_batch_analysis(*batch)
            if result is not None:
                summaries.append({key: result[key] for key in ("id", "saved_at", "title", "sources", "run_count", "model")
                                  if key in result})
        summaries.sort(key=lambda item: item["saved_at"], reverse=True)
        individual = []
        if self.analysis_directory.resolve().is_relative_to(self.workspace):
            for path in self.analysis_directory.glob("*/analysis.json"):
                if result := self._read_run_analysis(path.parent.name):
                    individual.append(result)
        sources = self._batch_run_sources(batches, {report["fingerprint"] for report in individual})
        individual.sort(key=lambda item: item["saved_at"], reverse=True)
        return {"analyses": summaries, "run_analyses": [
            {**{key: value for key, value in report.items() if key not in ("report", "fingerprint")},
             **sources.get(report["fingerprint"], {})} for report in individual
        ]}

    def _run_batch(self, batch, entries):
        for run, entry in zip(batch["runs"], entries):
            log = None
            try:
                with self.lock:
                    if batch["status"] == "stopping":
                        break
                    run["status"] = "running"
                    session_run = self._start_session_run(run["model"], run["property"])
                    self._save_batch(batch)
                    process, log = self._launch(*entry)
                    self.batch_process = process
                    self.batch_log = log
                code = process.wait()
                with self.lock:
                    result = self._finish_worker(code, log, self.batch_worker_stopped)
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                result = {"status": "error", "error": str(exc)[:500], "diagnostic": {"code": "WORKER_LAUNCH_ERROR"}}
            with self.lock:
                state = self._worker_state(log)
                if state.get("dashboard"):
                    self.batch_dashboards[run["number"]] = state["dashboard"]
                if state.get("directory") and result["status"] in ("cancelled", "error"):
                    result["run_log_directory"] = str(state["directory"])
                self.batch_log = None
                self.batch_process = None
                self.batch_worker_stopped = False
                self._finish_session_run(session_run, result)
                run.update(result)
                self._save_batch(batch)
            if log is not None:
                self._close_log(log)
        with self.lock:
            batch["status"] = "cancelled" if batch["status"] == "stopping" else "complete"
            self._save_batch(batch)

    def _get_batch(self, batch_id):
        if self.batch is None or self.batch["id"] != batch_id:
            raise ValueError("검증 묶음을 찾을 수 없습니다.")
        return self.batch

    def batch_status(self, batch_id):
        with self.lock:
            batch = self._get_batch(batch_id)
            state = self._worker_state(self.batch_log)
            runs = [{**run, "dashboard_ready": (run["number"] in self.batch_dashboards
                     or run["status"] == "running" and bool(state.get("dashboard"))),
                     "phase": state.get("phase", "starting") if run["status"] == "running" else None}
                    for run in batch["runs"]]
            analysis_ready = bool(batch.get("followup_source") or batch.get("overview")) or any(
                self._has_analysis_evidence(run) or self._batch_log_limits(run, state) for run in runs)
            return {**{key: value for key, value in batch.items() if key != "saved_analysis"},
                    "runs": runs, "analysis_ready": analysis_ready}

    def open_dashboard(self, body):
        from Automation.LogAnalysisLauncher import open_dashboard_with_analysis

        if not isinstance(body, dict):
            raise ValueError("실행 정보를 찾을 수 없습니다.")
        run_id, number = body.get("id"), body.get("number")
        with self.lock:
            if number is None and self.current is not None and self.current["id"] == run_id:
                dashboard = self._worker_state(self.current["log"]).get("dashboard")
            else:
                batch = self._get_batch(run_id)
                if type(number) is not int or not 1 <= number <= len(batch["runs"]):
                    raise ValueError("실행 번호를 찾을 수 없습니다.")
                dashboard = self.batch_dashboards.get(number)
                if dashboard is None and batch["runs"][number - 1]["status"] == "running":
                    dashboard = self._worker_state(self.batch_log).get("dashboard")
        if dashboard is None or not dashboard.is_file():
            raise ValueError("대시보드가 아직 준비되지 않았습니다.")
        options = {"open_page": self.browser.open} if self.browser is not None else {}
        run_directory = dashboard.parent
        if run_directory.name == REALTIME_DIRECTORY:
            run_directory = run_directory.parent
        result = open_dashboard_with_analysis(run_directory, dashboard, **options)
        if not result["browser_opened"]:
            raise RuntimeError("브라우저를 열지 못했습니다. 실행 폴더의 HTML을 직접 열 수 있습니다.")
        return {"opened": True}

    @staticmethod
    def _has_analysis_evidence(run):
        return any(run.get(key) for key in ("verdict", "metrics", "diagnostic", "solver_checkpoint"))

    def _batch_log_limits(self, run, state):
        if run["status"] == "running" and state.get("directory"):
            return capture_log_limits(state["directory"])
        if run["status"] in ("error", "cancelled"):
            if run.get("run_log_directory"):
                directory = Path(run["run_log_directory"]).resolve()
            elif run.get("run_manifest_path"):
                directory = Path(run["run_manifest_path"]).resolve().parent
            else:
                return []
            root = (self.workspace / "visualization" / "outputs").resolve()
            if root.is_relative_to(self.workspace) and directory.is_relative_to(root):
                return capture_log_limits(directory)
        return []

    def start_batch_analysis(self, body, client_factory, *, max_calls=1):
        if not isinstance(body, dict) or body.get("consent") is not True:
            raise ValueError("검증 묶음의 요약 전송에 동의해 주세요.")
        question = body.get("question", "")
        if not isinstance(question, str) or len(question) > 2000:
            raise ValueError("질문은 2000자 이내로 입력해 주세요.")
        with self.lock:
            if self.closing:
                raise RuntimeError("검증기를 종료하고 있습니다.")
            batch_id = body.get("id")
            if batch_id is None and self.batch is None:
                self._require_idle()
                self.batch = {"id": secrets.token_hex(12), "status": "complete", "runs": [],
                              "analysis": {"status": "idle"}, "overview": True}
            elif batch_id is None and not self.batch.get("overview"):
                raise ValueError("현재 검증 묶음을 먼저 불러와 주세요.")
            batch = self._get_batch(batch_id) if batch_id is not None else self.batch
            if batch["analysis"]["status"] == "running":
                raise RuntimeError("통합 분석이 진행 중입니다.")
            if batch.get("overview") and self.current is not None and self.current["process"].poll() is None:
                raise RuntimeError("검증이 끝난 뒤 프로젝트 구조를 질문해 주세요.")
            state = self._worker_state(self.batch_log)
            limits = [self._batch_log_limits(run, state) for run in batch["runs"]]
            runs = [{key: value for key, value in run.items()
                     if key not in ("result_path", "run_manifest_path", "run_log_directory", "error")}
                    for run in batch["runs"]]
            if not (batch.get("followup_source") or batch.get("overview")) and not any(limits) and not any(self._has_analysis_evidence(run) for run in runs):
                raise RuntimeError("아직 분석할 검증 로그가 없습니다.")
            snapshot = {"snapshot_at": datetime.now().astimezone().isoformat(),
                        "partial": any(run["status"] != "complete" for run in runs)}
            if batch.get("followup_source"):
                snapshot["followup_source"] = batch["followup_source"]
            if batch.get("overview"):
                snapshot["overview"] = True
            conversation = self.analysis_conversation.snapshot()
            if batch["analysis"]["status"] == "complete":
                batch["saved_analysis"] = batch["analysis"]
            budget = {"used": 0, "limit": max_calls, "remaining": max_calls}
            batch["analysis"] = {"status": "running", "request_budget": budget, **snapshot}
            Thread(target=self._analyze_batch,
                   args=(batch, runs, limits, snapshot, question, conversation, client_factory, max_calls),
                   name="verification-batch-analysis", daemon=True).start()
        return {"id": batch["id"], "status": "running", "request_budget": budget}

    def _analyze_batch(self, batch, runs, limits, snapshot, question, conversation, client_factory, max_calls):
        from Automation.LogAnalysisIndex import safe_data

        def budget(used):
            return {"used": used, "limit": max_calls, "remaining": max(0, max_calls - used)}

        def report_usage(used, _limit):
            with self.lock:
                if batch["analysis"]["status"] == "running":
                    batch["analysis"] = {**batch["analysis"], "request_budget": budget(used)}

        client = None
        try:
            for run, sources in zip(runs, limits):
                if observations := summarize_logs(sources):
                    run["recorded_logs"] = observations
            if not (snapshot.get("followup_source") or snapshot.get("overview")) and not any(
                self._has_analysis_evidence(run) or run.get("recorded_logs") for run in runs
            ):
                raise ValueError("요청 시점에 읽을 수 있는 검증 기록이 없습니다.")
            client = client_factory()
            client.max_requests = max_calls
            client.request_progress = report_usage
            report = client.summarize_verification_batch(runs, question, snapshot=snapshot, conversation=conversation)
            result = {"status": "complete", "model": client.model, "report": report,
                      "api_calls": client.requests_made, "question": question,
                      "saved_at": datetime.now().astimezone().isoformat(), **snapshot}
        except Exception as exc:
            key = getattr(client, "key", None)
            message = str(exc).replace(key, "[REDACTED]") if isinstance(key, str) and key else str(exc)
            result = {"status": "error", "error": safe_data(message[:500])}
        result["request_budget"] = budget(client.requests_made if client is not None else 0)
        with self.lock:
            if batch["analysis"]["status"] == "running":
                batch["analysis"] = result
                if result["status"] == "complete":
                    self.analysis_conversation.remember(question, result["report"])
                    batch["saved_analysis"] = result
                self._save_batch(batch)


def _worker(model, prop, engine, timeout, *, settings=None, state_path=None, managed_browser=False):
    from Automation.AutoVerify import run_vnnlib_verification

    settings = _run_settings({} if settings is None else settings)
    mode = VisualizationMode(settings["visualization_mode"])
    # Record the selected features explicitly; glpk_extract etc. follow Reluplex defaults (best setting).
    solver_settings = {"lp_engine": engine,
                       **feature_options(settings["features"], settings["under_approx_eps"])}
    if engine != "native" and settings["lp_mode"] != "core":
        solver_settings["lp_mode"] = settings["lp_mode"]

    state = {"phase": "starting"}

    def publish(**values):
        state.update(values)
        if state_path is not None:
            try:
                write_atomic_text(state_path, json.dumps(state))
            except OSError:
                # A launcher status-write failure must not stop verification.
                pass

    def run_started(directory):
        print(RUN_OUTPUT_MARKER + json.dumps(str(directory), ensure_ascii=True), flush=True)
        publish(phase="preparing" if mode.realtime_enabled else "running", directory=str(directory))

    def request_dashboard_tab(url):
        publish(dashboard_url=url)
        return True

    def run(path):
        result = run_vnnlib_verification(
            str(path), str(prop), allow_large_model=True, timeout_seconds=timeout,
            simplex_max_iter=settings["simplex_max_iter"], max_rounds=settings["max_rounds"],
            max_recursion=settings["max_recursion"], visualization_mode=mode.value,
            realtime_panels=settings["panels"],
            open_realtime_view=mode.realtime_enabled,
            reluplex_options=solver_settings,
            run_started_callback=run_started,
            dashboard_ready_callback=(
                lambda path: publish(phase="running", dashboard=str(path))
            ) if state_path is not None and mode.realtime_enabled else None,
            dashboard_opener=request_dashboard_tab if managed_browser and state_path is not None else None,
        )
        print("VERIFICATION_RESULT " + json.dumps({
            "verdict": result["status"], "result_path": result.get("result_path"),
            "metrics": _batch_metrics(result), "finished_monotonic": monotonic(),
        }, ensure_ascii=True), flush=True)
        return result

    if model.name.lower().endswith(".onnx.gz"):
        with TemporaryDirectory(prefix="verification-onnx-") as directory:
            unpacked = Path(directory) / model.name[:-3]
            with gzip.open(model, "rb") as source, unpacked.open("wb") as target:
                shutil.copyfileobj(source, target)
            return run(unpacked)
    return run(model)


class LauncherPage:
    batch_max_calls = 1
    root = None
    active = False

    def __init__(self):
        from Automation.AnalyzeLogs import DEFAULT_MODEL, GeminiClient, load_gemini_api_key

        self.verification = VerificationLauncher(restore_preferences=False)
        self.batch_client = None
        self.model_lock = Lock()
        self.available_models = None
        try:
            self.batch_client = GeminiClient(load_gemini_api_key(), DEFAULT_MODEL, retries=0)
        except ValueError:
            pass
        self.token, self.csrf = secrets.token_urlsafe(24), secrets.token_urlsafe(24)
        self.shutdown_requested = Event()
        self.last_activity = monotonic()

    @property
    def batch_analysis_available(self):
        return self.batch_client is not None

    def models(self):
        from Automation.AnalyzeLogs import GeminiClient, analysis_model_options

        if self.batch_client is None:
            return {"models": []}
        with self.model_lock:
            if self.available_models is None:
                listed = GeminiClient(self.batch_client.key, self.batch_client.model, timeout=8, retries=0).list_models()
                self.available_models = tuple(analysis_model_options(listed["analysis_models"])["models"])
            return {**analysis_model_options(self.available_models), "default_model": self.batch_client.model}

    def start_batch_analysis(self, body):
        from Automation.AnalyzeLogs import GeminiClient, is_analysis_model, validate_thinking_level

        if self.batch_client is None:
            raise ValueError("Gemini API 키를 설정한 후 이 실행 화면을 다시 열어 주세요.")
        if not isinstance(body, dict) or body.get("consent") is not True:
            raise ValueError("검증 묶음의 요약 전송에 동의해 주세요.")
        model = body.get("model", self.batch_client.model)
        if not is_analysis_model(model) or ("model" in body and model not in self.models()["models"]):
            raise ValueError("조회된 모델 목록에서 선택해 주세요.")
        thinking_level = validate_thinking_level(model, body["thinking_level"]) if "thinking_level" in body else None
        options = {"thinking_level": thinking_level} if thinking_level is not None else {}
        return self.verification.start_batch_analysis(
            body, lambda: GeminiClient(self.batch_client.key, model, retries=0, **options), max_calls=self.batch_max_calls)

    def shutdown_launcher(self):
        self.verification.shutdown()

    def dashboard(self):
        from visualization.VerificationPicker import picker_markup
        return ('<!doctype html><html lang="ko"><meta charset="utf-8">'
                '<meta name="viewport" content="width=device-width,initial-scale=1">'
                '<title>신경망 검증 실행</title><body style="margin:24px;background:#0f172a;'
                'color:#e2e8f0;font-family:system-ui"><main style="max-width:660px;margin:auto">'
                '<h1>신경망 검증 실행</h1>' + picker_markup(standalone=True) + '</main></body></html>').encode("utf-8")


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="로컬 신경망과 VNNLIB 속성을 화면에서 선택합니다.")
    parser.add_argument("--worker", nargs=4, metavar=("MODEL", "PROPERTY", "ENGINE", "TIMEOUT"),
                        help=argparse.SUPPRESS)
    parser.add_argument("--worker-settings", default="{}", help=argparse.SUPPRESS)
    parser.add_argument("--worker-state", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--managed-browser", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        model, prop, engine, timeout = args.worker
        try:
            _worker(Path(model), Path(prop), engine, int(timeout), settings=json.loads(args.worker_settings),
                    state_path=args.worker_state, managed_browser=args.managed_browser)
        except Exception as exc:
            print(f"{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            return 1
        return 0
    from Automation.LogAnalysisServer import make_server, run_server_loop
    from Automation.VerificationBrowser import VerificationBrowser, chromium_executables

    app = LauncherPage()
    with make_server(app) as server:
        url = f"http://127.0.0.1:{server.server_port}/s/{app.token}/"
        print(f"신경망 검증 실행: {url}", flush=True)
        for executable in chromium_executables():
            browser = VerificationBrowser(executable)
            if browser.start(url):
                app.verification.browser = browser
                break
        else:
            try:
                opened = webbrowser.open(url, new=2)
            except (OSError, webbrowser.Error):
                opened = False
            if opened:
                print("기본 브라우저에서 열었습니다. 검증기를 종료한 뒤 해당 탭은 직접 닫아 주세요.", flush=True)
            else:
                print("브라우저 자동 열기에 실패했습니다. 위 주소를 브라우저에 붙여 넣어 주세요.", flush=True)
        try:
            run_server_loop(server, app)
        except KeyboardInterrupt:
            pass
        finally:
            if app.verification.browser is not None:
                app.verification.browser.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
