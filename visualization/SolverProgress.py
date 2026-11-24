"""DPLL(T) round·theory·Simplex 세부 계측과 visualization.

계측 JSONL은 실시간 dashboard와 종료 후 feedback의 공통 원본이다.
solver는 이 모듈의 logger method만 호출하고, 렌더러와 집계기가
같은 이벤트를 독립적으로 소비한다.
"""
from __future__ import annotations

import json
import inspect
import math
import threading
import warnings
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from time import monotonic, sleep, time_ns
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, TextIO
from Automation.LpTelemetry import METRIC_META, metric_available

import matplotlib

matplotlib.use("Agg")

from visualization.RenderLock import render_turn
from visualization.AppendLogReader import AppendLogReader
from visualization.RunLabels import network_display_name
from visualization.RenderProcess import (
    REALTIME_RENDER_INTERVAL_SECONDS,
    submit_solver_render,
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "NaN"
        return "Infinity" if value > 0 else "-Infinity"
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value

def _parse_progress_line(log_path, line_no, line):
    line = line.strip()
    if not line:
        return None
    try:
        value = json.loads(line)
        if isinstance(value, dict) and "event" in value:
            return value
    except (TypeError, ValueError):
        pass
    warnings.warn(f"{log_path}:{line_no}의 잘못된 progress를 건너뜁니다")
    return None


def read_solver_progress(path: str | Path) -> List[Dict[str, Any]]:
    log_path = Path(path)
    if not log_path.exists():
        return []
    events: List[Dict[str, Any]] = []
    with log_path.open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, start=1):
            if (value := _parse_progress_line(log_path, line_no, line)) is not None:
                events.append(value)
    return events


class SolverProgressLogger:
    """thread-safe solver progress JSONL logger."""

    def __init__(
        self,
        path: str | Path,
        update_callback: Optional[Callable[[], None]] = None,
        *,
        reset_log: bool = False,
        refresh_interval_seconds: float = 0.3,
        enabled_panels: Optional[Iterable[str]] = None,
        simplex_detail_stride: int = 1,
        compact_simplex_calls: bool = False,
        flush_interval_seconds: float = 1.0,
        flush_bytes: int = 256 * 1024,
        witness_context: Optional[Dict[str, Any]] = None,
    ) -> None:
        allowed_panels = {"dpll_theory", "simplex"}
        selected_panels = (
            allowed_panels if enabled_panels is None else set(enabled_panels)
        )
        unknown_panels = selected_panels - allowed_panels
        if unknown_panels:
            names = ", ".join(sorted(unknown_panels))
            raise ValueError(f"지원하지 않는 solver progress panel: {names}")
        if simplex_detail_stride <= 0:
            raise ValueError("simplex_detail_stride must be positive")
        if flush_interval_seconds <= 0:
            raise ValueError("flush_interval_seconds must be positive")
        if flush_bytes <= 0:
            raise ValueError("flush_bytes must be positive")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream: Optional[TextIO] = self.path.open(
            "w" if reset_log else "a",
            encoding="utf-8",
            buffering=1024 * 1024,
        )
        self.update_callback = update_callback
        self.witness_context = dict(witness_context or {})
        self._contextual_update_callback = False
        if update_callback is not None:
            try:
                inspect.signature(update_callback).bind_partial(
                    panel="dpll_theory", event="round_start"
                )
                self._contextual_update_callback = True
            except (TypeError, ValueError):
                # 기존의 인자 없는 callback API도 계속 지원한다.
                pass
        self.refresh_interval_seconds = refresh_interval_seconds
        self.enabled_panels = frozenset(selected_panels)
        self.simplex_detail_stride = simplex_detail_stride
        self.compact_simplex_calls = compact_simplex_calls
        self.flush_interval_seconds = flush_interval_seconds
        self.flush_bytes = flush_bytes
        self._lock = threading.RLock()
        self._last_update = 0.0
        self._last_flush = monotonic()
        self._unflushed_bytes = 0
        self._round_started: Dict[int, int] = {}
        self._round_simplex_calls: Counter[int] = Counter()
        self._active_round: Optional[int] = None
        self._simplex_started: Dict[str, int] = {}
        self._simplex_numbers: Dict[str, int] = {}
        self._simplex_rounds: Dict[str, Optional[int]] = {}
        self._simplex_metadata: Dict[str, Dict[str, Any]] = {}
        self._simplex_cycle_counts: Counter[str] = Counter()
        self._simplex_sequence = 0
        self._constraint_details: Dict[tuple[int, str], Dict[str, Any]] = {}
        self._emitted_constraint_details: set[tuple[str, bool]] = set()
        if self.witness_context.get("model_name"):
            self._append("run_context", witness_context=self.witness_context,
                         force_flush=True)

    def _flush_locked(self) -> None:
        if self._stream is None:
            return
        self._stream.flush()
        self._unflushed_bytes = 0
        self._last_flush = monotonic()

    def flush(self) -> None:
        """Make every buffered record visible to readers without closing the log."""
        with self._lock:
            self._flush_locked()

    def close(self) -> None:
        """Flush and close the persistent JSONL stream. Safe to call repeatedly."""
        with self._lock:
            if self._stream is None:
                return
            self._flush_locked()
            self._stream.close()
            self._stream = None

    def __enter__(self) -> "SolverProgressLogger":
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            # Interpreter shutdown may have already torn down I/O internals.
            pass

    def _append(
        self,
        event: str,
        *,
        force_update: bool = False,
        force_flush: bool = False,
        **payload: Any,
    ) -> None:
        panel = "simplex" if event.startswith("simplex_") else "dpll_theory"
        if event != "solver_end" and panel not in self.enabled_panels:
            return
        record = {
            "timestamp": datetime.now().astimezone().isoformat(timespec="microseconds"),
            "time_ns": time_ns(),
            "event": event,
            **_json_safe(payload),
        }
        encoded = json.dumps(
            record,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ) + "\n"
        with self._lock:
            if self._stream is None:
                raise RuntimeError("solver progress log is closed")
            self._stream.write(encoded)
            # Exact byte accounting would encode the JSON a second time. The
            # character count is a cheap conservative-enough flush trigger for
            # the overwhelmingly ASCII progress schema.
            self._unflushed_bytes += len(encoded)
            now = monotonic()
            if (
                force_flush
                or force_update
                or self._unflushed_bytes >= self.flush_bytes
                or now - self._last_flush >= self.flush_interval_seconds
                or (
                    self.update_callback is not None
                    and now - self._last_flush >= self.refresh_interval_seconds
                )
            ):
                self._flush_locked()
            self.request_update(force=force_update, panel=panel, event=event)

    def request_update(
        self,
        *,
        force: bool = True,
        panel: Optional[str] = None,
        event: str = "manual",
    ) -> None:
        if self.update_callback is None:
            return
        now = monotonic()
        if not force and now - self._last_update < self.refresh_interval_seconds:
            return
        try:
            if self._contextual_update_callback:
                self.update_callback(panel=panel, event=event)
            else:
                self.update_callback()
            self._last_update = monotonic()
        except Exception as exc:
            warnings.warn(f"solver progress visualizer 갱신 실패: {exc}")

    def round_start(self, round_index: int) -> None:
        with self._lock:
            started = time_ns()
            self._round_started[round_index] = started
            self._active_round = round_index
        self._append("round_start", round=round_index, force_update=True)

    def round_end(self, round_index: int, result: str, **meta: Any) -> None:
        with self._lock:
            started = self._round_started.pop(round_index, time_ns())
            if self._active_round == round_index:
                self._active_round = None
            simplex_calls = self._round_simplex_calls[round_index]
        self._append(
            "round_end",
            round=round_index,
            result=result,
            duration_seconds=max(0.0, (time_ns() - started) / 1_000_000_000),
            simplex_calls=simplex_calls,
            # round 종료는 duration/result를 확정하는 프레임이므로 throttle로
            # 버리면 안 된다. 다음 round_start와 별도 프레임으로 보존한다.
            force_update=True,
            **meta,
        )

    def abort_active_round(self, reason: str) -> None:
        with self._lock:
            round_index = self._active_round
        if round_index is not None:
            self.round_end(round_index, reason)

    def theory_selection(self, round_index: int, atoms: List[Mapping[str, Any]]) -> None:
        compact_atoms = []
        with self._lock:
            for raw_atom in atoms:
                atom = dict(raw_atom)
                detail = atom.pop("constraint_detail", None)
                slack_name = atom.get("slack_name")
                if detail is not None and slack_name is not None:
                    self._constraint_details[(round_index, str(slack_name))] = {
                        "round": round_index,
                        "slack_name": str(slack_name),
                        "constraint_id": atom.get("constraint_id", atom.get("atom_id")),
                        "atom_id": atom.get("atom_id"),
                        "polarity": atom.get("polarity"),
                        **dict(detail),
                    }
                compact_atoms.append(atom)
        self._append(
            "theory_selection",
            round=round_index,
            selection_id=f"round-{round_index}",
            atom_count=len(compact_atoms),
            atoms=compact_atoms,
            force_update=False,
        )

    def theory_result(self, round_index: int, result: str, **meta: Any) -> None:
        self._append(
            "theory_result",
            round=round_index,
            result=result,
            force_update=False,
            **meta,
        )

    def simplex_start(
        self,
        *,
        round_index: Optional[int],
        depth: Optional[int],
        origin: str,
        row_count: int,
        variable_count: int,
        theory_atom_ids: Iterable[str] = (),
        selection_id: Optional[str] = None,
        **meta: Any,
    ) -> str:
        atom_ids = [] if self.compact_simplex_calls else list(theory_atom_ids)
        with self._lock:
            self._simplex_sequence += 1
            call_number = self._simplex_sequence
            # A monotonic number is already unique within one reset log and is
            # substantially smaller than a UUID in million-call feedback runs.
            call_id = f"simplex-{call_number}"
            started = time_ns()
            self._simplex_started[call_id] = started
            self._simplex_numbers[call_id] = call_number
            self._simplex_rounds[call_id] = round_index
            if self.compact_simplex_calls:
                self._simplex_metadata[call_id] = {
                    "round": round_index,
                    "depth": depth,
                    "origin": origin,
                    "selection_id": selection_id,
                    # theory atoms are already stored once by theory_selection;
                    # selection_id links the call without repeating the same
                    # potentially large id list hundreds of thousands of times.
                    "row_count": row_count,
                    "variable_count": variable_count,
                    **meta,
                }
            if round_index is not None:
                self._round_simplex_calls[round_index] += 1
        if not self.compact_simplex_calls:
            self._append(
                "simplex_start",
                call_id=call_id,
                call_number=call_number,
                round=round_index,
                depth=depth,
                origin=origin,
                selection_id=selection_id,
                **(
                    {"theory_atom_ids": atom_ids} if atom_ids else {}
                ),
                row_count=row_count,
                variable_count=variable_count,
                **meta,
                force_update=False,
            )
        return call_id

    def simplex_iteration(
        self,
        call_id: str,
        iteration: int,
        *,
        violated_var: Optional[str],
        value: Optional[float] = None,
        lower: Optional[float] = None,
        upper: Optional[float] = None,
    ) -> None:
        if self.compact_simplex_calls and violated_var is None:
            # The completed call record already carries the exact iteration
            # total; an extra terminal iteration record adds no feedback data.
            return
        if violated_var is not None and iteration % self.simplex_detail_stride != 0:
            return
        with self._lock:
            call_number = self._simplex_numbers.get(call_id)
            round_index = self._simplex_rounds.get(call_id)
            detail_key = (
                (round_index, str(violated_var))
                if round_index is not None and violated_var is not None
                else None
            )
            detail = self._constraint_details.get(detail_key) if detail_key else None
            definition_key = (
                (str(detail.get("constraint_id")), bool(detail.get("polarity")))
                if detail is not None
                else None
            )
            should_emit_detail = (
                definition_key is not None
                and detail is not None
                and definition_key not in self._emitted_constraint_details
            )
            if should_emit_detail:
                self._emitted_constraint_details.add(definition_key)
        if should_emit_detail:
            self._append(
                "constraint_detail",
                force_update=False,
                **{
                    key: value
                    for key, value in detail.items()
                    if key not in {"round", "slack_name"}
                },
            )
        self._append(
            "simplex_iteration",
            call_id=call_id,
            call_number=call_number,
            iteration=iteration,
            violated_var=violated_var,
            value=value,
            lower=lower,
            upper=upper,
            force_update=False,
        )

    def simplex_pivot(
        self,
        call_id: str,
        iteration: int,
        *,
        entering: str,
        leaving: str,
        row: str,
    ) -> None:
        if iteration % self.simplex_detail_stride != 0:
            return
        with self._lock:
            call_number = self._simplex_numbers.get(call_id)
        self._append(
            "simplex_pivot",
            call_id=call_id,
            call_number=call_number,
            iteration=iteration,
            entering=entering,
            leaving=leaving,
            row=row,
            force_update=False,
        )

    def simplex_cycle(self, call_id: str, *, iteration: int, pivot: int, random_steps: int) -> None:
        with self._lock:
            self._simplex_cycle_counts[call_id] += 1
        self._append("simplex_cycle", call_id=call_id, iteration=iteration,
                     pivot=pivot, random_steps=random_steps)

    def simplex_bound_observations(self, call_id: str, observations, *, source="call_boundary") -> None:
        # Batch once-per-call snapshots separately from sampled internal iterations.
        # Chunking retains all observations without oversized JSONL records.
        observations = list(observations)
        for offset in range(0, len(observations), 128):
            with self._lock:
                number = self._simplex_numbers.get(call_id)
                round_index = self._simplex_rounds.get(call_id)
            self._append("simplex_bound_observation", call_id=call_id, call_number=number,
                         round=round_index, observation_source=source,
                         observations=observations[offset:offset + 128], force_update=False)

    def simplex_end(
        self,
        call_id: str,
        result: str,
        *,
        iterations: Optional[int],
        pivots: Optional[int],
        **meta: Any,
    ) -> None:
        with self._lock:
            # Active-call membership is sufficient for idempotence and avoids
            # retaining every completed call id during week-long runs.
            if call_id not in self._simplex_started:
                return
            started = self._simplex_started.pop(call_id)
            call_number = self._simplex_numbers.pop(call_id, None)
            self._simplex_rounds.pop(call_id, None)
            metadata = self._simplex_metadata.pop(call_id, {})
            cycles = meta.pop("cycles", self._simplex_cycle_counts.pop(call_id, 0))
        terminal_event = "simplex_call" if self.compact_simplex_calls else "simplex_end"
        self._append(
            terminal_event,
            call_id=call_id,
            call_number=call_number,
            **({**metadata, "start_ns": started, **meta} if self.compact_simplex_calls else meta),
            result=result,
            duration_seconds=max(0.0, (time_ns() - started) / 1_000_000_000),
            iterations=iterations,
            pivots=pivots,
            cycles=cycles,
            force_flush=result not in {"SAT", "UNSAT"},
            force_update=False,
        )

    def solver_end(
        self, status: str, reason: Optional[str], rounds: int,
        *, model: Optional[Dict[str, float]] = None,
    ) -> None:
        try:
            self._append(
                "solver_end",
                status=status,
                reason=reason,
                rounds=rounds,
                model=model if status == "SAT" else None,
                witness_context=self.witness_context,
                force_update=True,
                force_flush=True,
            )
        finally:
            self.close()


class _SolverFeedback:
    def __init__(self):
        self.rounds: Dict[int, Dict[str, Any]] = {}
        self.calls: Dict[str, Dict[str, Any]] = {}
        self.violations: Counter[str] = Counter()
        self.entering: Counter[str] = Counter()
        self.leaving: Counter[str] = Counter()
        self.pivot_rows: Counter[str] = Counter()
        self.violation_refs: Counter[tuple[Optional[int], str]] = Counter()
        self.observed_refs: Counter[tuple[Optional[int], str]] = Counter()
        self.constraint_details: Dict[tuple[str, bool], Dict[str, Any]] = {}
        self.slack_references: Dict[tuple[int, str], Dict[str, Any]] = {}
        self.theory_links: List[Dict[str, Any]] = []
        self.solver_result: Dict[str, Any] = {}

    def add(self, events: Iterable[Mapping[str, Any]]) -> None:
        for raw in events:
            event = dict(raw)
            kind = event.get("event")
            if kind == "run_context":
                self.solver_result["witness_context"] = event.get("witness_context", {})
            elif kind == "round_start":
                index = int(event["round"])
                self.rounds.setdefault(index, {"round": index}).update({"start_time_ns": event.get("time_ns")})
            elif kind == "round_end":
                index = int(event["round"])
                self.rounds.setdefault(index, {"round": index}).update(
                    {
                        "result": event.get("result"),
                        "duration_seconds": float(event.get("duration_seconds", 0.0)),
                        "simplex_calls": int(event.get("simplex_calls", 0)),
                    }
                )
            elif kind == "theory_selection":
                index = int(event["round"])
                atoms = list(event.get("atoms", []))
                self.rounds.setdefault(index, {"round": index})["theory_atoms"] = atoms
                for atom in atoms:
                    slack_name = atom.get("slack_name")
                    if slack_name is not None:
                        self.slack_references[(index, str(slack_name))] = {
                            "constraint_id": atom.get(
                                "constraint_id", atom.get("atom_id")
                            ),
                            "atom_id": atom.get("atom_id"),
                            "polarity": atom.get("polarity"),
                        }
            elif kind == "theory_result":
                index = int(event["round"])
                result = event.get("result")
                self.rounds.setdefault(index, {"round": index})["theory_result"] = result
                self.theory_links.append({"round": index, "result": result, **{k: v for k, v in event.items() if k not in {"event", "timestamp", "time_ns", "round", "result"}}})
            elif kind == "simplex_start":
                call_id = str(event["call_id"])
                self.calls[call_id] = {
                    "call_id": call_id,
                    "call_number": event.get("call_number"),
                    "start_timestamp": event.get("timestamp"),
                    "start_time_ns": event.get("time_ns"),
                    "round": event.get("round"),
                    "depth": event.get("depth"),
                    "origin": event.get("origin"),
                    "selection_id": event.get("selection_id"),
                    "theory_atom_ids": list(event.get("theory_atom_ids", [])),
                    "row_count": event.get("row_count"),
                    "variable_count": event.get("variable_count"),
                    **{key: event[key] for key in METRIC_META if key in event},
                }
            elif kind == "simplex_iteration":
                call_id = str(event.get("call_id"))
                self.calls.setdefault(call_id, {"call_id": call_id})["iterations"] = max(
                    int(self.calls.get(call_id, {}).get("iterations", 0)),
                    int(event.get("iteration", -1)) + 1,
                )
                variable = event.get("violated_var")
                if variable:
                    self.violations[str(variable)] += 1
                    self.violation_refs[(self.calls.get(call_id, {}).get("round"), str(variable))] += 1
            elif kind == "constraint_detail":
                constraint_id = event.get("constraint_id", event.get("atom_id"))
                if constraint_id is not None:
                    self.constraint_details[(str(constraint_id), bool(event.get("polarity")))] = {
                        key: value
                        for key, value in event.items()
                        if key not in {"event", "timestamp", "time_ns"}
                    }
            elif kind == "simplex_bound_observation":
                round_index = event.get("round", self.calls.get(str(event.get("call_id")), {}).get("round"))
                for observation in event.get("observations", []):
                    variable = observation.get("variable")
                    if isinstance(variable, str):
                        self.observed_refs[(round_index, variable)] += 1
            elif kind == "simplex_cycle":
                call = self.calls.setdefault(str(event["call_id"]), {"call_id": str(event["call_id"])})
                call["cycles"] = int(call.get("cycles", 0)) + 1
                if len(call.setdefault("cycle_events", [])) < 32:
                    call["cycle_events"].append({key: event.get(key) for key in ("iteration", "pivot", "random_steps")})
            elif kind == "simplex_pivot":
                call_id = str(event.get("call_id"))
                call = self.calls.setdefault(call_id, {"call_id": call_id})
                call["pivots"] = int(call.get("pivots", 0)) + 1
                self.entering[str(event.get("entering"))] += 1
                self.leaving[str(event.get("leaving"))] += 1
                self.pivot_rows[str(event.get("row"))] += 1
            elif kind in {"simplex_end", "simplex_call"}:
                call_id = str(event["call_id"])
                completed = {
                    "result": event.get("result"),
                    "call_number": event.get(
                        "call_number", self.calls.get(call_id, {}).get("call_number")
                    ),
                    "end_timestamp": event.get("timestamp"),
                    "duration_seconds": float(event.get("duration_seconds", 0.0)),
                    "iterations": int(event.get("iterations") or 0),
                    "pivots": int(event.get("pivots") or 0),
                    "cycles": int(event.get("cycles", self.calls.get(call_id, {}).get("cycles", 0)) or 0),
                    **{key: event[key] for key in METRIC_META if key in event},
                }
                if kind == "simplex_call":
                    completed.update(
                        {
                            "round": event.get("round"),
                            "depth": event.get("depth"),
                            "origin": event.get("origin"),
                            "selection_id": event.get("selection_id"),
                            "theory_atom_ids": list(event.get("theory_atom_ids", [])),
                            "row_count": event.get("row_count"),
                            "variable_count": event.get("variable_count"),
                        }
                    )
                self.calls.setdefault(call_id, {"call_id": call_id}).update(completed)
            elif kind == "solver_end":
                self.solver_result = {
                    "status": event.get("status"),
                    "reason": event.get("reason"),
                    "rounds": event.get("rounds"),
                    "model": event.get("model"),
                    "witness_context": event.get("witness_context", {}),
                }

    def snapshot(self) -> Dict[str, Any]:
        ordered_rounds = [dict(self.rounds[index]) for index in sorted(self.rounds)]
        ordered_calls = [dict(call) for call in self.calls.values()]
        round_call_counts = Counter(call.get("round") for call in ordered_calls)
        now = time_ns()
        for round_item in ordered_rounds:
            if "duration_seconds" not in round_item and round_item.get("start_time_ns"):
                round_item["duration_seconds"] = max(
                    0.0, (now - int(round_item["start_time_ns"])) / 1_000_000_000
                )
                round_item.setdefault("result", "RUNNING")
            round_item.setdefault(
                "simplex_calls",
                round_call_counts[round_item["round"]],
            )
        for call in ordered_calls:
            if "duration_seconds" not in call and call.get("start_time_ns"):
                call["duration_seconds"] = max(
                    0.0, (now - int(call["start_time_ns"])) / 1_000_000_000
                )
                call.setdefault("result", "RUNNING")
                call.setdefault("iterations", 0)
                call.setdefault("pivots", 0)
        resolved_by_constraint: Dict[tuple[Any, ...], Dict[str, Any]] = {}
        scoped_refs = [(key, count, "solver_iteration") for key, count in self.violation_refs.items()]
        scoped_refs.extend((key, count, "call_boundary") for key, count in self.observed_refs.items())
        for (round_index, slack_name), count, observation_scope in scoped_refs:
            reference = (
                self.slack_references.get((int(round_index), slack_name), {})
                if round_index is not None
                else {}
            )
            constraint_id = reference.get(
                "constraint_id", f"round-{round_index}:{slack_name}"
            )
            polarity = bool(reference.get("polarity"))
            detail = self.constraint_details.get((str(constraint_id), polarity), {})
            key = (constraint_id, polarity, observation_scope)
            item = resolved_by_constraint.setdefault(
                key,
                {
                    "constraint_id": constraint_id,
                    "atom_id": reference.get("atom_id", detail.get("atom_id")),
                    "polarity": reference.get("polarity", detail.get("polarity")),
                    "slack_name": slack_name,
                    "observation_scope": observation_scope,
                    "count": 0,
                    "rounds": [],
                    "occurrences": [],
                    **{
                        name: detail[name]
                        for name in ("original", "effective")
                        if name in detail
                    },
                },
            )
            item["count"] += count
            if round_index not in item["rounds"]:
                item["rounds"].append(round_index)
            item["occurrences"].append(
                {"round": round_index, "slack_name": slack_name, "count": count}
            )
        resolved_constraints = sorted(
            resolved_by_constraint.values(),
            key=lambda item: (-int(item["count"]), str(item["constraint_id"])),
        )

        def total(field):
            measured = [int(call.get(field, 0)) for call in ordered_calls if metric_available(call, field)]
            return sum(measured) if measured or not ordered_calls else None

        return {
            "solver": dict(self.solver_result),
            "rounds": ordered_rounds,
            "simplex_calls": ordered_calls,
            "counts": {
                "rounds": len(ordered_rounds),
                "simplex_calls": len(ordered_calls),
                "simplex_iterations": total("iterations"),
                "simplex_pivots": total("pivots"),
                "pivot_measured_calls": sum(metric_available(call, "pivots") for call in ordered_calls),
            },
            "top_violated_variables": dict(self.violations.most_common(20)),
            "top_violated_constraints": [item for item in resolved_constraints if item["observation_scope"] == "solver_iteration"][:20],
            "top_observed_bound_constraints": [item for item in resolved_constraints if item["observation_scope"] == "call_boundary"][:20],
            "top_entering_variables": dict(self.entering.most_common(20)),
            "top_leaving_variables": dict(self.leaving.most_common(20)),
            "top_pivot_rows": dict(self.pivot_rows.most_common(20)),
            "theory_links": list(self.theory_links),
        }


def build_solver_feedback(events: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    feedback = _SolverFeedback()
    feedback.add(events)
    return feedback.snapshot()


class SolverProgressReader:
    """Accumulate the renderer's view of a growing progress log."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._log = AppendLogReader(self.path)
        self._feedback = _SolverFeedback()

    def read(self) -> Dict[str, Any]:
        with self._log.read() as (reset, lines):
            if reset:
                self._feedback = _SolverFeedback()
            events = (_parse_progress_line(self.path, number, line) for number, line in lines)
            self._feedback.add(event for event in events if event is not None)
        return self._feedback.snapshot()


def _result_color(result: Optional[str]) -> str:
    if result in {"SAT", "THEORY_SAT", "THEORY_TRIVIAL"}:
        return "#16a34a"
    if result in {"UNSAT", "THEORY_UNSAT", "BOOLEAN_UNSAT"}:
        return "#dc2626"
    if result in {"CONFLICT", "THEORY_CONFLICT"}:
        return "#9333ea"
    if result in {"TIMEOUT", "THEORY_TIMEOUT"}:
        return "#ea580c"
    if result in {None, "RUNNING", "THEORY_RUNNING"}:
        return "#2563eb"
    if result in {
        "ITERATION_LIMIT",
        "SIMPLEX_ITERATION_LIMIT",
        "STALLED",
        "SIMPLEX_STALLED",
        "NUMERICAL_FAILURE",
        "SIMPLEX_NUMERICAL_FAILURE",
        "VALIDATION_INCONCLUSIVE",
        "SIMPLEX_VALIDATION_INCONCLUSIVE",
        "RELUPLEX_VALIDATION_INCONCLUSIVE",
        "STRICT_INEQUALITY_INCONCLUSIVE",
    }:
        return "#f59e0b"
    return "#64748b"


def _simplex_result_legend(axis, calls) -> None:
    from matplotlib.lines import Line2D

    results = ["SAT", "UNSAT"]
    results.extend(sorted({str(call.get("result", "RUNNING")) for call in calls} - set(results)))
    handles = [Line2D([], [], color=_result_color(result), linewidth=3, label=result)
               for result in results]
    extra_handles, _ = axis.get_legend_handles_labels()
    axis.legend(handles=handles + extra_handles, loc="upper left",
                bbox_to_anchor=(1.02, 1.0), borderaxespad=0, ncol=1, fontsize=9)


def _set_integer_ticks(axis, direction: str) -> None:
    """호출·반복·선택 횟수처럼 이산적인 축에 정수 눈금만 사용한다."""
    from matplotlib.ticker import MaxNLocator

    target = axis.xaxis if direction == "x" else axis.yaxis
    target.set_major_locator(MaxNLocator(integer=True, min_n_ticks=1))


def _set_duration_ylim(axis, durations: Iterable[float]) -> None:
    """시간축을 실제 최장 duration에 맞추고 자동 눈금의 과도한 여백을 막는다."""
    from matplotlib.ticker import LinearLocator

    maximum = max(
        (float(value) for value in durations if math.isfinite(float(value))),
        default=0.0,
    )
    axis.set_ylim(0.0, maximum if maximum > 0.0 else 1.0)
    axis.yaxis.set_major_locator(LinearLocator(numticks=6))


def _completed_call_durations(calls: Iterable[Mapping[str, Any]]) -> List[float]:
    """Return final durations only, so a running call cannot rescale every frame."""
    return [
        max(0.0, float(item.get("duration_seconds", 0.0)))
        for item in calls
        if item.get("result") not in (None, "RUNNING")
    ]


def _constraint_label(item: Mapping[str, Any]) -> str:
    """Return a compact one-line label; expressions remain in feedback JSON."""
    slack = str(item.get("slack_name", "?"))
    raw_constraint_id = item.get("constraint_id", item.get("atom_id"))
    if raw_constraint_id is None or str(raw_constraint_id).startswith("round-"):
        return slack
    constraint_id = str(raw_constraint_id)
    rounds = item.get("rounds") or [item.get("round")]
    round_text = ",".join(str(value) for value in rounds if value is not None)
    label = f"{slack} · {constraint_id}"
    if round_text:
        label += f" (R{round_text})"
    return label


def _annotate_violation_variable_types(axis) -> None:
    """Explain generated variable names without competing with the chart data."""
    axis.text(
        1.0,
        -0.21,
        "h: ReLU output  ·  z: value before ReLU  ·  ineq_slack: inequality auxiliary variable",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=7,
        color="#64748b",
        alpha=0.8,
        clip_on=False,
    )


def _theory_outcome(round_item: Mapping[str, Any]) -> Optional[str]:
    """실제 theory 호출 결과만 UI용 상태로 정규화한다."""
    raw_result = round_item.get("theory_result")
    round_result = str(round_item.get("result", "RUNNING"))
    # 이전/부분 로그에는 theory_result 이벤트 없이 round_end에만
    # THEORY_* 결과가 남아 있을 수 있으므로 그 경우도 theory 호출로 인정한다.
    if raw_result is None and round_result.startswith("THEORY_"):
        raw_result = round_result
    if raw_result is not None:
        return {
            "THEORY_CONFLICT": "CONFLICT",
            "THEORY_SAT": "SAT",
            "THEORY_TRIVIAL": "SAT",
            "THEORY_UNSAT": "UNSAT",
        }.get(str(raw_result), str(raw_result).removeprefix("THEORY_"))

    # DPLL 단계에서 끝난 BOOLEAN_UNSAT round는 theory 결과가 아니다.
    if "theory_atoms" not in round_item:
        return None
    if round_result == "RUNNING":
        return "RUNNING"
    return "UNKNOWN"


def solver_panel_paths(base_path: str | Path) -> Dict[str, Path]:
    base = Path(base_path)
    return {
        name: base.with_name(f"{base.stem}_{name}{base.suffix}")
        for name in ("dpll_theory", "simplex")
    }


def _feedback_from(events_or_path) -> Dict[str, Any]:
    if isinstance(events_or_path, dict) and "simplex_calls" in events_or_path:
        return events_or_path
    events = (
        read_solver_progress(events_or_path)
        if isinstance(events_or_path, (str, Path))
        else [dict(event) for event in events_or_path]
    )
    return build_solver_feedback(events)


def _compact_theory_flow_lines(
    feedback: Mapping[str, Any], rounds: Iterable[Mapping[str, Any]]
) -> List[str]:
    """긴 수식 대신 round별 핵심 전달 흐름만 짧게 만든다."""
    calls = feedback["simplex_calls"]
    call_number_by_id = {
        str(call.get("call_id")): index for index, call in enumerate(calls, start=1)
    }
    calls_by_round = defaultdict(list)
    for call in calls:
        calls_by_round[call.get("round")].append(call)
    lines: List[str] = []
    for item in rounds:
        round_index = item.get("round")
        atoms = list(item.get("theory_atoms", []))
        kind_counts = Counter(str(atom.get("kind", "Other")) for atom in atoms)
        kind_parts = [
            f"{kind}:{count}" for kind, count in kind_counts.most_common(3)
        ]
        if len(kind_counts) > 3:
            kind_parts.append(f"+{len(kind_counts) - 3} types")
        atom_summary = ", ".join(kind_parts) or "none"

        round_calls = calls_by_round[round_index]
        call_labels = [
            f"S{call_number_by_id.get(str(call.get('call_id')), '?')}:{call.get('result', 'RUNNING')}"
            for call in round_calls[:4]
        ]
        if len(round_calls) > 4:
            call_labels.append(f"+{len(round_calls) - 4} more")
        call_summary = ", ".join(call_labels) or "none"
        theory_result = _theory_outcome(item) or "NOT RUN"
        lines.append(
            f"R{round_index} | selected atoms {len(atoms)} ({atom_summary}) | Simplex {len(round_calls)}\n"
            f"   {call_summary} | {theory_result}"
        )
    return lines


def _sat_witness_lines(feedback: Dict[str, Any]) -> List[str]:
    """Only the selected output conditions of a final SAT branch."""
    solver = feedback.get("solver", {})
    if solver.get("status") != "SAT":
        return []
    outputs = solver.get("witness_context", {}).get("outputs", {})
    aliases = {var: name for name, var in outputs.items()}
    links = [item for item in feedback.get("theory_links", [])
             if item.get("result") == "THEORY_SAT"]
    if not links or not aliases:
        return []
    conditions = []
    for item in links[-1].get("inequalities", []):
        coeffs = item.get("coeffs", {})
        if not coeffs or not set(coeffs) <= set(aliases):
            continue
        terms = []
        for var, coefficient in sorted(coeffs.items()):
            sign = "-" if coefficient < 0 else "+"
            magnitude = abs(coefficient)
            term = aliases[var] if magnitude == 1 else f"{magnitude:.7g}*{aliases[var]}"
            terms.append((sign, term))
        expression = " ".join(
            (("-" if sign == "-" else "") + term) if index == 0 else f"{sign} {term}"
            for index, (sign, term) in enumerate(terms)
        )
        conditions.append(f"{expression} >= {item['lower']:.7g}")
    return list(dict.fromkeys(conditions))


def _sat_solution_values(feedback: Dict[str, Any]) -> Dict[str, str]:
    """Format input/output values of the final counterexample by section."""
    solver = feedback.get("solver", {})
    model = solver.get("model")
    if solver.get("status") != "SAT" or not isinstance(model, dict):
        return {}
    context = solver.get("witness_context", {})

    inputs = []
    for name, expression in context.get("inputs", {}).items():
        coeffs = expression.get("coeffs", {})
        if all(var in model for var in coeffs):
            value = math.fsum(coefficient * model[var]
                              for var, coefficient in coeffs.items())
            value += expression.get("constant", 0.0)
            inputs.append(f"{name} = {value:.7g}")

    outputs = [
        f"{name} = {model[var]:.7g}"
        for name, var in context.get("outputs", {}).items()
        if var in model
    ]
    return {
        "inputs": ", ".join(inputs),
        "outputs": ", ".join(outputs),
    }


def draw_dpll_theory_panel(
    events_or_path, *, title: str = "DPLL Rounds & Theory Flow"
):
    """DPLL round 진행과 theory 전달 흐름을 하나의 패널에 표시한다."""
    import matplotlib.pyplot as plt

    feedback = _feedback_from(events_or_path)
    rounds = feedback["rounds"][-12:]
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(min(15, 10.5 + max(0, len(rounds) - 1) * 0.5), 7.2),
        gridspec_kw={"height_ratios": [1, 1.25], "width_ratios": [1, 1.7]},
    )
    fig.suptitle(title, fontsize=16, fontweight="bold")
    ax_duration, ax_calls, ax_results, ax_links = axes.flat

    round_ids = [int(item["round"]) for item in rounds]
    durations = [
        max(0.0, float(item.get("duration_seconds", 0.0))) for item in rounds
    ]
    colors = [_result_color(item.get("result")) for item in rounds]
    ax_duration.bar(round_ids, durations, color=colors)
    ax_duration.set_xticks(round_ids)
    for x_pos, duration, item in zip(round_ids, durations, rounds):
        ax_duration.annotate(
            f"{item.get('result', 'RUNNING')}\n{duration:.3f} s",
            (x_pos, duration / 2),
            xytext=(0, 0),
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=7,
            multialignment="center",
            linespacing=1.3,
            bbox={"boxstyle": "square,pad=0.45", "facecolor": "white",
                  "edgecolor": "none", "alpha": 0.85},
        )
    ax_duration.set_title("Round duration and result")
    ax_duration.set_xlabel("DPLL round")
    ax_duration.set_ylabel("Duration (s)")
    ax_duration.set_ylim(bottom=0)
    ax_duration.grid(axis="y", alpha=0.3)

    call_counts = [int(item.get("simplex_calls", 0)) for item in rounds]
    call_bars = ax_calls.bar(round_ids, call_counts, color="#2563eb")
    ax_calls.set_xticks(round_ids)
    ax_calls.set_title("Simplex calls per round")
    ax_calls.set_xlabel("DPLL round")
    ax_calls.set_ylabel("Calls")
    _set_integer_ticks(ax_calls, "y")
    ax_calls.grid(axis="y", alpha=0.3)
    for x_pos, count in zip(round_ids, call_counts):
        ax_calls.annotate(str(count), (x_pos, count / 2), xytext=(0, 0),
                          textcoords="offset points", ha="center", va="center", fontsize=9,
                          bbox={"boxstyle": "square,pad=0.45", "facecolor": "white",
                                "edgecolor": "none", "alpha": 0.85})

    results = Counter(
        outcome for item in rounds if (outcome := _theory_outcome(item)) is not None
    )
    labels = list(results)
    values = [results[label] for label in labels]
    ax_results.barh(labels, values, color=[_result_color(label) for label in labels])
    ax_results.set_title("Theory Solver Result")
    ax_results.set_xlabel("Count")
    _set_integer_ticks(ax_results, "x")
    ax_results.set_xlim(0, max(values, default=1) * 1.2)
    for row, count in enumerate(values):
        ax_results.annotate(str(count), (count, row), xytext=(5, 0), textcoords="offset points", va="center")

    link_lines = _compact_theory_flow_lines(feedback, rounds)
    ax_links.axis("off")
    ax_links.set_title("Round | selected atoms | Simplex | theory result", loc="left")
    flow_text = ax_links.text(
        0.01,
        0.96,
        "\n".join(link_lines) if link_lines else "No theory selection yet",
        transform=ax_links.transAxes,
        va="top",
        family="monospace",
        fontsize=8.5,
        clip_on=True,
    )
    solver = feedback.get("solver", {})
    fig.text(
        0.09,
        0.01,
        f"rounds={feedback['counts']['rounds']} | solver={solver.get('status', 'RUNNING')} / {solver.get('reason', '-')}",
        fontsize=9,
    )
    model_name = network_display_name(solver.get("witness_context", {}))
    if model_name:
        fig.text(0.91, 0.055, f"Network: {model_name}",
                 ha="right", va="bottom", fontsize=11, color="#111827")
        layer_sizes = solver.get("witness_context", {}).get("model_layer_sizes", [])
        if layer_sizes:
            structure = " -> ".join(str(size) for size in layer_sizes)
            fig.text(0.91, 0.035,
                     f"Structure (input -> hidden -> output): {structure} | ReLU: {sum(layer_sizes[1:-1])}",
                     ha="right", va="bottom", fontsize=10, color="#374151")
    witness_lines = _sat_witness_lines(feedback)
    if witness_lines:
        import textwrap
        formula = " AND ".join(f"({line})" for line in witness_lines)
        solution = _sat_solution_values(feedback)
        sections = [
            ("VNNLIB", formula),
            ("Counter example input", solution.get("inputs", "unavailable")),
            ("Network output", solution.get("outputs", "unavailable")),
        ]
        cursor = max(0.18, 0.90 - 0.065 * len(link_lines))
        for label, value in sections:
            value_lines = textwrap.wrap(value, width=75) or ["unavailable"]
            # Keep long counterexamples inside the existing flow area.
            value_lines = value_lines[:4]
            label_artist = ax_links.text(
                0.01, cursor, label, transform=ax_links.transAxes,
                va="top", family="monospace", fontsize=9,
                color="#111827", fontweight="bold",
                clip_on=True,
            )
            cursor -= 0.052
            value_artist = ax_links.text(
                0.01, cursor, "\n".join(value_lines),
                transform=ax_links.transAxes, va="top",
                family="monospace", fontsize=9, color="#166534",
                clip_on=True,
            )
            from visualization.AnalysisSelection import select_text
            for artist in (label_artist, value_artist):
                select_text(fig, artist, {"scope": {"topic": "dpll"}, "text": f"{label}\n{value}"}, label="DPLL · 반례 해")
            cursor -= 0.052 * len(value_lines) + 0.035
    fig.subplots_adjust(left=0.09, right=0.91, top=0.91, bottom=0.14, hspace=0.38, wspace=0.5)
    flow_line_count = max(1, sum(line.count("\n") + 1 for line in link_lines))
    available_points = ax_links.get_position().height * fig.get_figheight() * 72 * 0.90
    flow_text.set_fontsize(min(8.5, available_points / (1.2 * flow_line_count)))
    renderer = fig.canvas.get_renderer()
    # Font ascenders/descenders differ: fit measured text, not a fixed line ratio.
    text_height = flow_text.get_window_extent(renderer).height
    available_height = ax_links.get_window_extent(renderer).height * 0.90
    if text_height > available_height:
        flow_text.set_fontsize(flow_text.get_fontsize() * available_height / text_height)
    for bars, axis, values in ((call_bars, ax_calls, call_counts),):
        for bar, label, x_pos, value in zip(bars, axis.texts, round_ids, values):
            label.update_positions(renderer)
            label.update_bbox_position_size(renderer)
            box = label.get_bbox_patch().get_window_extent(renderer)
            area = bar.get_window_extent(renderer)
            if area.height < box.height + 8 or area.width < box.width + 8:
                label.xy = (x_pos, value)
                label.set_verticalalignment("bottom")
                label.set_position((0, 6))
    from visualization.AnalysisSelection import solver_selection
    solver_selection(fig, axes, feedback, simplex=False, flow_lines=link_lines)
    return fig, axes


def _annotate_simplex_rounds(axes, calls):
    if len({call.get("round") for call in calls if call.get("round") is not None}) < 2:
        return
    previous = None
    for number, call in enumerate(calls, 1):
        round_index = call.get("round")
        if round_index is not None and round_index != previous:
            for ax in axes:
                # Boundary before the first call belonging to the new round.
                ax.axvline(number - 0.5, color="#374151", linestyle="--",
                           linewidth=1.1, zorder=4)
                ax.annotate(f"D{round_index}\ncall {number}",
                            xy=(number - 0.5, 0.98),
                            xycoords=ax.get_xaxis_transform(),
                            xytext=(3, 0), textcoords="offset points",
                            ha="left", va="top", fontsize=8, color="#111827",
                            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8},
                            zorder=5)
        previous = round_index


LP_ENGINE_LABELS = {"native": "Python Simplex", "simplex": "Python Simplex Oracle",
                    "glpk": "GLPK", "z3": "Z3"}


def _simplex_panel_engine(feedback):
    calls = feedback.get("simplex_calls", [])
    configured = feedback.get("solver", {}).get("witness_context", {}).get("lp_engine")
    engines = {call.get("lp_engine") or configured for call in calls}
    engines.discard(None)
    if not calls:
        return configured
    return next(iter(engines)) if len(engines) == 1 else "mixed" if engines else None


def _simplex_work_labels(engine, calls):
    if engine == "glpk":
        return "iterations", "GLPK iterations per call", "GLPK iterations", "Internal steps counted during each call"
    if engine == "z3":
        return "pivots", "Z3 arithmetic pivots per call", "Arithmetic pivots", "From Z3 solver statistics"
    if engine == "mixed":
        return None, "LP work per call", "", ""
    if engine in ("native", "simplex") or any(metric_available(call, "pivots") for call in calls):
        return "pivots", "Pivots per call", "Pivots within call", ""
    if any(metric_available(call, "iterations") for call in calls):
        return "iterations", "Iterations per call", "Reported iterations", ""
    return None, "LP work per call", "", ""


def draw_simplex_panel(
    events_or_path, *, title: str = "Simplex Progress", show_round_boundaries: bool = False,
):
    import matplotlib.pyplot as plt
    feedback = _feedback_from(events_or_path)
    calls = feedback["simplex_calls"]
    engine = _simplex_panel_engine(feedback)
    engine_label = LP_ENGINE_LABELS.get(engine, "Mixed LP engines" if engine == "mixed" else engine)
    external = engine is not None and engine != "native"
    call_order_label = "LP call order" if external else "Simplex call order"
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2))
    ax_duration, ax_work, ax_results, ax_violations = axes.flat
    fig.suptitle(f"{title} · {engine_label}" if engine_label else title, fontsize=16, fontweight="bold")
    call_numbers = list(range(1, len(calls) + 1))
    durations = [
        max(0.0, float(item.get("duration_seconds", 0.0))) for item in calls
    ]
    # One collection retains every call without thousands of Rectangle artists.
    ax_duration.vlines(call_numbers, 0, durations,
                       colors=[_result_color(item.get("result")) for item in calls],
                       linewidth=1.0)
    _set_integer_ticks(ax_duration, "x")
    ax_duration.set_xlim(0.5, max(1.5, len(calls) + 0.5))
    ax_duration.set_title("LP call duration" if external else "Call duration (all calls)")
    ax_duration.set_xlabel(call_order_label)
    ax_duration.set_ylabel("Seconds")
    _set_duration_ylim(ax_duration, _completed_call_durations(calls))
    ax_duration.grid(axis="y", alpha=0.3)

    metric, work_title, work_unit, work_source = _simplex_work_labels(engine, calls)
    measured = [int(item.get(metric, 0)) if metric and metric_available(item, metric) else None
                for item in calls]
    reported = [value for value in measured if value is not None]
    # 수백~수천 call에서도 선 자체는 모든 값을 연결하되 마커는 최대 약
    # 80개만 보이게 한다.
    marker_stride = max(1, (len(call_numbers) + 79) // 80)
    ax_work.plot(
        call_numbers,
        [value if value is not None else float("nan") for value in measured],
        marker="s",
        markevery=marker_stride,
        markersize=2.0,
        markeredgewidth=0,
        linewidth=0.75,
        alpha=0.9,
        color="#ff7f0e",
    )
    _set_integer_ticks(ax_work, "x")
    ax_work.set_xlim(0.5, max(1.5, len(calls) + 0.5))
    _set_integer_ticks(ax_work, "y")
    ax_work.set_title(work_title)
    if engine == "mixed":
        ax_work.text(0.5, 0.5, "Work counts use different units by engine",
                     transform=ax_work.transAxes, ha="center", va="center", fontsize=9, color="#6b7280")
    elif not reported:
        waiting = not calls or all(call.get("result", "RUNNING") == "RUNNING" for call in calls)
        ax_work.text(0.5, 0.5, "Waiting for LP calls" if not calls else
                     "Waiting for call results" if waiting else "No counts reported",
                     transform=ax_work.transAxes, ha="center", va="center", fontsize=9, color="#6b7280")
    elif len(reported) < len(calls):
        ax_work.text(0.02, 0.96, "Gaps: count not reported", transform=ax_work.transAxes, va="top", fontsize=8)
    ax_work.set_xlabel(f"{call_order_label}\n{work_source}" if work_source else call_order_label)
    ax_work.set_ylabel(work_unit)
    ax_work.grid(alpha=0.3)

    cycle_calls = [(number, call) for number, call in zip(call_numbers, calls)
                   if call.get("cycles", 0)]
    if cycle_calls:
        for axis, field in ((ax_duration, "duration_seconds"), (ax_work, metric)):
            points = [(number, call.get(field, 0) if axis is ax_duration else measured[number - 1])
                      for number, call in cycle_calls]
            points = [(number, value) for number, value in points if value is not None]
            if not points:
                continue
            x, y = zip(*points)
            axis.scatter(x, y,
                         marker="^", s=38, color="#7c3aed", zorder=5,
                         label="Cycle detected")
            axis.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0),
                        borderaxespad=0, ncol=1, fontsize=9)
        ax_work.text(0.02, 0.86,
                     f"Cycles: {sum(call['cycles'] for _, call in cycle_calls)} "
                     f"in {len(cycle_calls)} calls", transform=ax_work.transAxes,
                     va="top", fontsize=10, color="#6d28d9")

    if show_round_boundaries:
        _annotate_simplex_rounds((ax_duration, ax_work), calls)

    _simplex_result_legend(ax_duration, calls)

    result_counts = Counter(item.get("result", "RUNNING") for item in calls)
    result_labels = list(result_counts)
    ax_results.bar(
        result_labels,
        [result_counts[label] for label in result_labels],
        color=[_result_color(label) for label in result_labels],
    )
    ax_results.set_title("LP call results" if external else "Call results (all calls)")
    _set_integer_ticks(ax_results, "y")
    ax_results.tick_params(axis="x", labelrotation=20)

    violated, violation_scope = bound_violation_items(feedback)
    violated = violated[:8]
    if violated:
        ax_violations.barh(
            [_constraint_label(item) for item in reversed(violated)],
            [int(item.get("count", 0)) for item in reversed(violated)],
            color="#ef4444",
        )
        ax_violations.tick_params(axis="y", labelsize=7)
    internal = violation_scope == "solver_iteration"
    ax_violations.set_title("Repeated bound violations" if internal else "Bound violations at LP call entry",
                            fontsize=10)
    ax_violations.set_xlabel("Internal check samples" if internal else "LP call-entry observations")
    if not violated:
        ax_violations.text(.5, .5, "No sampled violations" if internal
                           else "No call-entry violations recorded", transform=ax_violations.transAxes,
                           ha="center", va="center", fontsize=9, color="#6b7280")
    _annotate_violation_variable_types(ax_violations)
    _set_integer_ticks(ax_violations, "x")
    ax_violations.grid(axis="x", alpha=0.3)

    summary = f"Calls: {len(calls)}"
    if reported:
        summary += f" | {work_unit}: {sum(reported)}"
        if len(reported) < len(calls):
            summary += f" (reported for {len(reported)}/{len(calls)} calls)"
    fig.text(0.01, 0.01, summary, fontsize=9)
    # Keep useful plot area in each quadrant; long legend labels must not
    # reserve the width of another graph.
    import textwrap
    for axis in axes.flat:
        legend = axis.get_legend()
        if legend is not None:
            for label in legend.get_texts():
                label.set_text(textwrap.fill(label.get_text(), width=18))
                label.set_fontsize(8)
    fig.subplots_adjust(left=0.07, right=0.93, top=0.91, bottom=0.09, hspace=0.48, wspace=0.50)
    if ax_work.get_legend() is not None:
        position = ax_work.get_position()
        ax_work.set_position([position.x0, position.y0, position.width - 0.13, position.height])
    from visualization.AnalysisSelection import solver_selection
    solver_selection(fig, axes, feedback, simplex=True)
    return fig, axes


def bound_violation_items(feedback):
    internal = feedback.get("top_violated_constraints", [])
    if not feedback.get("simplex_calls") and _simplex_panel_engine(feedback) in ("native", "simplex"):
        return internal, "solver_iteration"
    if internal or any(call.get("bound_violation_scope", "solver_iteration" if call.get("work_counts_available", True)
                                else "call_boundary") == "solver_iteration"
                       for call in feedback.get("simplex_calls", [])):
        return internal, "solver_iteration"
    return feedback.get("top_observed_bound_constraints", []), "call_boundary"


class SolverProgressPanelVisualizer:
    """Render the current DPLL/Theory and Simplex panels in a separate process.

    Coalesce updates to the latest state; keep refreshing during an active round.
    Browser HTML and polling belong to UnifiedRealtimeDashboard, not this worker.
    """

    def __init__(
        self,
        log_path: str | Path,
        base_output_path: str | Path,
        *,
        enabled_panels: Optional[Iterable[str]] = None,
    ) -> None:
        self.log_path = Path(log_path)
        self._state_lock = threading.Lock()
        self._worker: Optional[threading.Thread] = None
        allowed_panels = {"dpll_theory", "simplex"}
        selected_panels = (
            allowed_panels if enabled_panels is None else set(enabled_panels)
        )
        unknown_panels = selected_panels - allowed_panels
        if unknown_panels:
            names = ", ".join(sorted(unknown_panels))
            raise ValueError(f"지원하지 않는 solver progress panel: {names}")
        all_paths = solver_panel_paths(base_output_path)
        self.panel_paths = {
            name: all_paths[name] for name in sorted(selected_panels)
        }
        self._pending_panels: set[str] = set()
        self._round_active = False

    def request_update(
        self, *, panel: Optional[str] = None, event: Optional[str] = None
    ) -> None:
        selected = set(self.panel_paths) if panel is None else {panel}
        selected &= set(self.panel_paths)
        # Terminal frames include every panel, even if a callback was throttled.
        if event == "solver_end":
            selected = set(self.panel_paths)
        if not selected:
            return
        with self._state_lock:
            if event == "round_start":
                self._round_active = True
            elif event in {"round_end", "solver_end"}:
                self._round_active = False
            # Keep only the latest request; never parse the growing log on the
            # solver thread or build a backlog of obsolete round snapshots.
            self._pending_panels.update(selected)
            if self._worker is not None:
                return
            self._worker = threading.Thread(
                target=self._run_updates,
                name="solver-progress-panel-renderer",
                daemon=True,
            )
            self._worker.start()

    def _run_updates(self) -> None:
        try:
            while True:
                started = time_ns()
                with self._state_lock:
                    selected = set(self._pending_panels)
                    self._pending_panels.clear()
                    if self._round_active:
                        selected.update(self.panel_paths)
                    if not selected:
                        self._worker = None
                        return
                try:
                    submit_solver_render(
                        self.log_path, self.panel_paths, tuple(sorted(selected))
                    ).result()
                except Exception as exc:
                    # A temporarily locked PNG must not kill the round's
                    # heartbeat when the LP engine emits no new events.
                    warnings.warn(f"solver progress background render 실패; 다음 갱신에서 재시도합니다: {exc}")
                    with self._state_lock:
                        if self._round_active:
                            self._pending_panels.update(selected)
                with self._state_lock:
                    more = bool(self._pending_panels) or self._round_active
                    if not more:
                        self._worker = None
                        return
                # Browser polling is independent of expensive PNG generation.
                # A real cooldown applies even under continuous solver events.
                elapsed = (time_ns() - started) / 1_000_000_000
                sleep(max(0.0, REALTIME_RENDER_INTERVAL_SECONDS - elapsed))
        except Exception as exc:
            warnings.warn(f"solver progress background render 실패: {exc}")
            with self._state_lock:
                self._worker = None

    def flush(self) -> None:
        while True:
            with self._state_lock:
                worker = self._worker
            if worker is None:
                return
            worker.join()



def write_solver_feedback_panels(
    log_path: str | Path,
    base_image_path: str | Path,
    json_path: Optional[str | Path] = None,
    *,
    enabled_panels: Optional[Iterable[str]] = None,
) -> tuple[Dict[str, Any], Dict[str, Path]]:
    import matplotlib.pyplot as plt

    events = read_solver_progress(log_path)
    feedback = build_solver_feedback(events)
    allowed_panels = {"dpll_theory", "simplex"}
    selected_panels = (
        allowed_panels if enabled_panels is None else set(enabled_panels)
    )
    unknown_panels = selected_panels - allowed_panels
    if unknown_panels:
        names = ", ".join(sorted(unknown_panels))
        raise ValueError(f"지원하지 않는 solver feedback panel: {names}")
    all_paths = solver_panel_paths(base_image_path)
    paths = {name: all_paths[name] for name in sorted(selected_panels)}
    renderers = {
        "dpll_theory": draw_dpll_theory_panel,
        "simplex": draw_simplex_panel,
    }
    titles = {
        "dpll_theory": "DPLL Rounds & Theory Flow Feedback",
        "simplex": "Simplex Feedback",
    }
    with render_turn():
        for name in paths:
            renderer = renderers[name]
            output = paths[name]
            output.parent.mkdir(parents=True, exist_ok=True)
            options = {"show_round_boundaries": True} if name == "simplex" else {}
            fig, _ = renderer(feedback, title=titles[name], **options)
            fig.savefig(output, dpi=150, bbox_inches="tight")
            from visualization.AnalysisSelection import write_selection_map
            write_selection_map(fig, output, cropped=True)
            plt.close(fig)
    resolved_json = (
        Path(json_path) if json_path is not None else Path(base_image_path).with_suffix(".json")
    )
    resolved_json.parent.mkdir(parents=True, exist_ok=True)
    resolved_json.write_text(
        json.dumps(feedback, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return feedback, paths
