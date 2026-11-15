"""DPLL(T) round·theory·Simplex 세부 계측과 visualization.

계측 JSONL은 실시간 dashboard와 종료 후 feedback의 공통 원본이다.
solver는 이 모듈의 logger method만 호출하고, 렌더러와 집계기가
같은 이벤트를 독립적으로 소비한다.
"""
from __future__ import annotations

import json
import inspect
import math
import os
import threading
import warnings
import webbrowser
from collections import Counter, deque
from datetime import datetime
from pathlib import Path
from time import monotonic, sleep, time_ns
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, TextIO

import matplotlib

matplotlib.use("Agg")

from visualization.RenderLock import render_turn
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

def read_solver_progress(path: str | Path) -> List[Dict[str, Any]]:
    log_path = Path(path)
    if not log_path.exists():
        return []
    events: List[Dict[str, Any]] = []
    with log_path.open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
                if not isinstance(value, dict) or "event" not in value:
                    raise ValueError
                events.append(value)
            except (TypeError, ValueError, json.JSONDecodeError):
                warnings.warn(f"{log_path}:{line_no}의 잘못된 progress를 건너뜁니다")
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

    def simplex_end(
        self,
        call_id: str,
        result: str,
        *,
        iterations: int,
        pivots: int,
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
            cycles = self._simplex_cycle_counts.pop(call_id, 0)
        terminal_event = "simplex_call" if self.compact_simplex_calls else "simplex_end"
        self._append(
            terminal_event,
            call_id=call_id,
            call_number=call_number,
            **(metadata if self.compact_simplex_calls else {}),
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


def build_solver_feedback(events: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    rounds: Dict[int, Dict[str, Any]] = {}
    calls: Dict[str, Dict[str, Any]] = {}
    violations: Counter[str] = Counter()
    entering: Counter[str] = Counter()
    leaving: Counter[str] = Counter()
    pivot_rows: Counter[str] = Counter()
    violation_refs: Counter[tuple[Optional[int], str]] = Counter()
    constraint_details: Dict[tuple[str, bool], Dict[str, Any]] = {}
    slack_references: Dict[tuple[int, str], Dict[str, Any]] = {}
    theory_links: List[Dict[str, Any]] = []
    solver_result: Dict[str, Any] = {}

    for raw in events:
        event = dict(raw)
        kind = event.get("event")
        if kind == "run_context":
            solver_result["witness_context"] = event.get("witness_context", {})
        elif kind == "round_start":
            index = int(event["round"])
            rounds.setdefault(index, {"round": index}).update({"start_time_ns": event.get("time_ns")})
        elif kind == "round_end":
            index = int(event["round"])
            rounds.setdefault(index, {"round": index}).update(
                {
                    "result": event.get("result"),
                    "duration_seconds": float(event.get("duration_seconds", 0.0)),
                    "simplex_calls": int(event.get("simplex_calls", 0)),
                }
            )
        elif kind == "theory_selection":
            index = int(event["round"])
            atoms = list(event.get("atoms", []))
            rounds.setdefault(index, {"round": index})["theory_atoms"] = atoms
            for atom in atoms:
                slack_name = atom.get("slack_name")
                if slack_name is not None:
                    slack_references[(index, str(slack_name))] = {
                        "constraint_id": atom.get(
                            "constraint_id", atom.get("atom_id")
                        ),
                        "atom_id": atom.get("atom_id"),
                        "polarity": atom.get("polarity"),
                    }
        elif kind == "theory_result":
            index = int(event["round"])
            result = event.get("result")
            rounds.setdefault(index, {"round": index})["theory_result"] = result
            theory_links.append({"round": index, "result": result, **{k: v for k, v in event.items() if k not in {"event", "timestamp", "time_ns", "round", "result"}}})
        elif kind == "simplex_start":
            call_id = str(event["call_id"])
            calls[call_id] = {
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
            }
        elif kind == "simplex_iteration":
            call_id = str(event.get("call_id"))
            calls.setdefault(call_id, {"call_id": call_id})["iterations"] = max(
                int(calls.get(call_id, {}).get("iterations", 0)),
                int(event.get("iteration", -1)) + 1,
            )
            variable = event.get("violated_var")
            if variable:
                violations[str(variable)] += 1
                violation_refs[(calls.get(call_id, {}).get("round"), str(variable))] += 1
        elif kind == "constraint_detail":
            constraint_id = event.get("constraint_id", event.get("atom_id"))
            if constraint_id is not None:
                constraint_details[(str(constraint_id), bool(event.get("polarity")))] = {
                    key: value
                    for key, value in event.items()
                    if key not in {"event", "timestamp", "time_ns"}
                }
        elif kind == "simplex_cycle":
            call = calls.setdefault(str(event["call_id"]), {"call_id": str(event["call_id"])})
            call["cycles"] = int(call.get("cycles", 0)) + 1
            if len(call.setdefault("cycle_events", [])) < 32:
                call["cycle_events"].append({key: event.get(key) for key in ("iteration", "pivot", "random_steps")})
        elif kind == "simplex_pivot":
            call_id = str(event.get("call_id"))
            call = calls.setdefault(call_id, {"call_id": call_id})
            call["pivots"] = int(call.get("pivots", 0)) + 1
            entering[str(event.get("entering"))] += 1
            leaving[str(event.get("leaving"))] += 1
            pivot_rows[str(event.get("row"))] += 1
        elif kind in {"simplex_end", "simplex_call"}:
            call_id = str(event["call_id"])
            completed = {
                "result": event.get("result"),
                "call_number": event.get(
                    "call_number", calls.get(call_id, {}).get("call_number")
                ),
                "end_timestamp": event.get("timestamp"),
                "duration_seconds": float(event.get("duration_seconds", 0.0)),
                "iterations": int(event.get("iterations", 0)),
                "pivots": int(event.get("pivots", 0)),
                "cycles": int(event.get("cycles", calls.get(call_id, {}).get("cycles", 0))),
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
            calls.setdefault(call_id, {"call_id": call_id}).update(completed)
        elif kind == "solver_end":
            solver_result = {
                "status": event.get("status"),
                "reason": event.get("reason"),
                "rounds": event.get("rounds"),
                "model": event.get("model"),
                "witness_context": event.get("witness_context", {}),
            }

    ordered_rounds = [rounds[index] for index in sorted(rounds)]
    ordered_calls = list(calls.values())
    now = time_ns()
    for round_item in ordered_rounds:
        if "duration_seconds" not in round_item and round_item.get("start_time_ns"):
            round_item["duration_seconds"] = max(
                0.0, (now - int(round_item["start_time_ns"])) / 1_000_000_000
            )
            round_item.setdefault("result", "RUNNING")
        round_item.setdefault(
            "simplex_calls",
            sum(call.get("round") == round_item.get("round") for call in ordered_calls),
        )
    for call in ordered_calls:
        if "duration_seconds" not in call and call.get("start_time_ns"):
            call["duration_seconds"] = max(
                0.0, (now - int(call["start_time_ns"])) / 1_000_000_000
            )
            call.setdefault("result", "RUNNING")
            call.setdefault("iterations", 0)
            call.setdefault("pivots", 0)
    resolved_by_constraint: Dict[tuple[Any, Any], Dict[str, Any]] = {}
    for (round_index, slack_name), count in violation_refs.items():
        reference = (
            slack_references.get((int(round_index), slack_name), {})
            if round_index is not None
            else {}
        )
        constraint_id = reference.get(
            "constraint_id", f"round-{round_index}:{slack_name}"
        )
        polarity = bool(reference.get("polarity"))
        detail = constraint_details.get((str(constraint_id), polarity), {})
        key = (constraint_id, polarity)
        item = resolved_by_constraint.setdefault(
            key,
            {
                "constraint_id": constraint_id,
                "atom_id": reference.get("atom_id", detail.get("atom_id")),
                "polarity": reference.get("polarity", detail.get("polarity")),
                "slack_name": slack_name,
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
    )[:20]

    return {
        "solver": solver_result,
        "rounds": ordered_rounds,
        "simplex_calls": ordered_calls,
        "counts": {
            "rounds": len(ordered_rounds),
            "simplex_calls": len(ordered_calls),
            "simplex_iterations": sum(int(call.get("iterations", 0)) for call in ordered_calls),
            "simplex_pivots": sum(int(call.get("pivots", 0)) for call in ordered_calls),
        },
        "top_violated_variables": dict(violations.most_common(20)),
        "top_violated_constraints": resolved_constraints,
        "top_entering_variables": dict(entering.most_common(20)),
        "top_leaving_variables": dict(leaving.most_common(20)),
        "top_pivot_rows": dict(pivot_rows.most_common(20)),
        "theory_links": theory_links,
    }


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
        -0.16,
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


def draw_solver_dashboard(
    events_or_path: Iterable[Mapping[str, Any]] | str | Path,
    *,
    title: str = "DPLL(T) / Simplex Progress",
):
    import matplotlib.pyplot as plt

    events = (
        read_solver_progress(events_or_path)
        if isinstance(events_or_path, (str, Path))
        else [dict(event) for event in events_or_path]
    )
    feedback = build_solver_feedback(events)
    rounds = feedback["rounds"][-20:]
    # Simplex 통계는 최근 일부 호출만 자르지 않고 실행 시작 이후의
    # 모든 call을 누적해서 보여준다.
    calls = feedback["simplex_calls"]

    fig = plt.figure(figsize=(15, 10.5))
    grid = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.9])
    ax_round = fig.add_subplot(grid[0, 0])
    ax_duration = fig.add_subplot(grid[0, 1])
    ax_work = fig.add_subplot(grid[1, 0])
    ax_vars = fig.add_subplot(grid[1, 1])
    ax_links = fig.add_subplot(grid[2, :])
    axes = [ax_round, ax_duration, ax_work, ax_vars, ax_links]
    fig.suptitle(title, fontsize=16, fontweight="bold")

    round_ids = [int(item["round"]) for item in rounds]
    round_durations = [
        max(0.0, float(item.get("duration_seconds", 0.0))) for item in rounds
    ]
    round_colors = [_result_color(item.get("result")) for item in rounds]
    ax_round.bar(round_ids, round_durations, color=round_colors)
    ax_round.set_xticks(round_ids)
    for x_pos, duration, item in zip(round_ids, round_durations, rounds):
        ax_round.annotate(
            f"{item.get('result', 'RUNNING')}\nS={item.get('simplex_calls', 0)}",
            (x_pos, duration),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    ax_round.set_title("DPLL rounds (latest 20)")
    ax_round.set_xlabel("Round")
    ax_round.set_ylabel("Duration (s)")
    ax_round.set_ylim(bottom=0)
    ax_round.grid(axis="y", alpha=0.3)

    call_numbers = list(range(max(1, len(feedback["simplex_calls"]) - len(calls) + 1), len(feedback["simplex_calls"]) + 1))
    call_durations = [
        max(0.0, float(item.get("duration_seconds", 0.0))) for item in calls
    ]
    call_colors = [_result_color(item.get("result")) for item in calls]
    ax_duration.bar(call_numbers, call_durations, color=call_colors)
    ax_duration.set_xticks(call_numbers)
    ax_duration.set_xticklabels(
        [f"{number}\n{item.get('result', 'RUNNING')}" for number, item in zip(call_numbers, calls)],
        fontsize=7,
    )
    ax_duration.set_title("Simplex call duration (all calls)")
    ax_duration.set_xlabel("Simplex call order")
    ax_duration.set_ylabel("Duration (s)")
    # Keep the cumulative completed-call maximum.  A RUNNING call's elapsed
    # time changes on every refresh, but its final duration is not known yet.
    _set_duration_ylim(ax_duration, _completed_call_durations(calls))
    ax_duration.grid(axis="y", alpha=0.3)

    pivots = [int(item.get("pivots", 0)) for item in calls]
    marker_stride = max(1, (len(call_numbers) + 79) // 80)
    ax_work.plot(
        call_numbers, pivots, marker="s", markevery=marker_stride,
        markersize=2.0, markeredgewidth=0, linewidth=0.75, alpha=0.9,
        color="#9333ea",
    )
    ax_work.set_xticks(call_numbers)
    _set_integer_ticks(ax_work, "y")
    ax_work.set_title("Simplex pivots per call")
    ax_work.set_xlabel("Simplex call order")
    ax_work.set_ylabel("Pivots within call")
    ax_work.grid(alpha=0.3)

    cycle_calls = [(number, call) for number, call in zip(call_numbers, calls)
                   if call.get("cycles", 0)]
    if cycle_calls:
        for axis, field in ((ax_duration, "duration_seconds"), (ax_work, "pivots")):
            axis.scatter([number for number, _ in cycle_calls],
                         [call.get(field, 0) for _, call in cycle_calls],
                         marker="^", s=38, color="#7c3aed", zorder=5,
                         label="Cycle detected / random pivot")
            axis.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0),
                        borderaxespad=0, ncol=1, fontsize=9)
        ax_work.text(0.02, 0.96,
                     f"Cycles: {sum(call['cycles'] for _, call in cycle_calls)} "
                     f"in {len(cycle_calls)} calls", transform=ax_work.transAxes,
                     va="top", fontsize=10, color="#6d28d9")

    _simplex_result_legend(ax_duration, calls)
    selected = feedback.get("top_violated_constraints", [])[:8]
    if selected:
        labels = [_constraint_label(item) for item in reversed(selected)]
        values = [int(item.get("count", 0)) for item in reversed(selected)]
        ax_vars.barh(labels, values, color="#ef4444")
        ax_vars.tick_params(axis="y", labelsize=7)
    ax_vars.set_title("Repeated bound violations")
    ax_vars.set_xlabel("Selections")
    _annotate_violation_variable_types(ax_vars)
    _set_integer_ticks(ax_vars, "x")
    ax_vars.grid(axis="x", alpha=0.3)

    ax_links.axis("off")
    ax_links.set_title("DPLL | theory atoms | Simplex calls | theory result", loc="left")
    all_calls = feedback["simplex_calls"]
    call_number_by_id = {
        str(call.get("call_id")): index for index, call in enumerate(all_calls, start=1)
    }
    link_lines = []
    for round_item in feedback["rounds"][-6:]:
        round_index = round_item.get("round")
        atom_labels = []
        for atom in round_item.get("theory_atoms", [])[:5]:
            sign = "+" if atom.get("polarity") else "-"
            atom_labels.append(f"{sign}{atom.get('atom_id')}:{atom.get('kind')}")
        round_calls = [call for call in all_calls if call.get("round") == round_index]
        call_labels = [
            f"S{call_number_by_id.get(str(call.get('call_id')), '?')}"
            f"[{call.get('origin', '?')},{call.get('result', 'RUNNING')}]"
            for call in round_calls
        ]
        link_lines.append(
            f"R{round_index}  atoms=({', '.join(atom_labels) or '-'})"
            f"  |  {', '.join(call_labels) or 'no Simplex'}"
            f"  |  {round_item.get('theory_result', round_item.get('result', 'RUNNING'))}"
        )
    ax_links.text(
        0.01,
        0.9,
        "\n".join(link_lines) if link_lines else "No round progress yet",
        transform=ax_links.transAxes,
        va="top",
        family="monospace",
        fontsize=9,
    )

    latest_round = rounds[-1] if rounds else {}
    solver = feedback.get("solver", {})
    model_name = solver.get("witness_context", {}).get("model_name")
    if model_name:
        fig.text(0.98, 0.055, f"Network: {model_name}",
                 ha="right", va="bottom", fontsize=11, color="#111827")
        layer_sizes = solver.get("witness_context", {}).get("model_layer_sizes", [])
        if layer_sizes:
            structure = " -> ".join(str(size) for size in layer_sizes)
            fig.text(0.98, 0.035,
                     f"Structure (input -> hidden -> output): {structure} | ReLU: {sum(layer_sizes[1:-1])}",
                     ha="right", va="bottom", fontsize=10, color="#374151")
    fig.text(
        0.01,
        0.01,
        " | ".join(
            [
                f"rounds={feedback['counts']['rounds']}",
                f"simplex calls={feedback['counts']['simplex_calls']}",
                f"pivots={feedback['counts']['simplex_pivots']}",
                f"latest round result={latest_round.get('result', 'RUNNING')}",
                f"solver={solver.get('status', 'RUNNING')} / {solver.get('reason', '-')}",
            ]
        ),
        fontsize=9,
    )
    fig.subplots_adjust(
        left=0.08,
        right=0.80,
        top=0.94,
        bottom=0.08,
        hspace=0.58,
        wspace=1.05,
    )
    return fig, axes


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

        round_calls = [call for call in calls if call.get("round") == round_index]
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


def draw_dpll_panel(events_or_path, *, title: str = "DPLL Round Progress"):
    import matplotlib.pyplot as plt

    feedback = _feedback_from(events_or_path)
    rounds = feedback["rounds"][-20:]
    fig, (ax_duration, ax_calls) = plt.subplots(1, 2, figsize=(13, 4.2))
    fig.suptitle(title, fontsize=16, fontweight="bold")
    round_ids = [int(item["round"]) for item in rounds]
    durations = [
        max(0.0, float(item.get("duration_seconds", 0.0))) for item in rounds
    ]
    colors = [_result_color(item.get("result")) for item in rounds]
    ax_duration.bar(round_ids, durations, color=colors)
    ax_duration.set_xticks(round_ids)
    for x_pos, duration, item in zip(round_ids, durations, rounds):
        ax_duration.annotate(
            str(item.get("result", "RUNNING")),
            (x_pos, duration),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=7,
        )
    ax_duration.set_title("Round duration and result")
    ax_duration.set_xlabel("DPLL round")
    ax_duration.set_ylabel("Duration (s)")
    ax_duration.set_ylim(bottom=0)
    ax_duration.grid(axis="y", alpha=0.3)

    call_counts = [int(item.get("simplex_calls", 0)) for item in rounds]
    ax_calls.bar(round_ids, call_counts, color="#2563eb")
    ax_calls.set_xticks(round_ids)
    ax_calls.set_title("Simplex calls per round")
    ax_calls.set_xlabel("DPLL round")
    ax_calls.set_ylabel("Calls")
    _set_integer_ticks(ax_calls, "y")
    ax_calls.grid(axis="y", alpha=0.3)
    for x_pos, count in zip(round_ids, call_counts):
        ax_calls.annotate(str(count), (x_pos, count), ha="center", va="bottom")

    solver = feedback.get("solver", {})
    fig.text(
        0.01,
        0.01,
        f"rounds={feedback['counts']['rounds']} | solver={solver.get('status', 'RUNNING')} / {solver.get('reason', '-')}",
        fontsize=9,
    )
    fig.subplots_adjust(left=0.07, right=0.98, top=0.84, bottom=0.18, wspace=0.25)
    return fig, (ax_duration, ax_calls)


def draw_theory_panel(events_or_path, *, title: str = "DPLL ↔ Theory Flow"):
    import matplotlib.pyplot as plt

    feedback = _feedback_from(events_or_path)
    rounds = feedback["rounds"][-12:]
    fig, (ax_results, ax_links) = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={"width_ratios": [1, 2]})
    fig.suptitle(title, fontsize=16, fontweight="bold")
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
    ax_links.text(
        0.01,
        0.96,
        "\n".join(link_lines) if link_lines else "No theory selection yet",
        transform=ax_links.transAxes,
        va="top",
        family="monospace",
        fontsize=8.5,
    )
    fig.subplots_adjust(left=0.08, right=0.98, top=0.84, bottom=0.12, wspace=0.24)
    return fig, (ax_results, ax_links)


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
    round_bars = ax_duration.bar(round_ids, durations, color=colors)
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
    model_name = solver.get("witness_context", {}).get("model_name")
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
            ax_links.text(
                0.01, cursor, label, transform=ax_links.transAxes,
                va="top", family="monospace", fontsize=9,
                color="#111827", fontweight="bold",
                clip_on=True,
            )
            cursor -= 0.052
            ax_links.text(
                0.01, cursor, "\n".join(value_lines),
                transform=ax_links.transAxes, va="top",
                family="monospace", fontsize=9, color="#166534",
                clip_on=True,
            )
            cursor -= 0.052 * len(value_lines) + 0.035
    fig.subplots_adjust(left=0.09, right=0.91, top=0.91, bottom=0.14, hspace=0.38, wspace=0.5)
    flow_line_count = max(1, sum(line.count("\n") + 1 for line in link_lines))
    available_points = ax_links.get_position().height * fig.get_figheight() * 72 * 0.90
    flow_text.set_fontsize(min(8.5, available_points / (1.2 * flow_line_count)))
    renderer = fig.canvas.get_renderer()
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


def draw_simplex_panel(
    events_or_path, *, title: str = "Simplex Progress", show_round_boundaries: bool = False,
):
    import matplotlib.pyplot as plt

    feedback = _feedback_from(events_or_path)
    calls = feedback["simplex_calls"]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2))
    ax_duration, ax_work, ax_results, ax_violations = axes.flat
    fig.suptitle(title, fontsize=16, fontweight="bold")
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
    ax_duration.set_title("Call duration (all calls)")
    ax_duration.set_xlabel("Simplex call order")
    ax_duration.set_ylabel("Seconds")
    _set_duration_ylim(ax_duration, _completed_call_durations(calls))
    ax_duration.grid(axis="y", alpha=0.3)

    pivots = [int(item.get("pivots", 0)) for item in calls]
    # 수백~수천 call에서도 선 자체는 모든 값을 연결하되 마커는 최대 약
    # 80개만 보이게 한다.
    marker_stride = max(1, (len(call_numbers) + 79) // 80)
    ax_work.plot(
        call_numbers,
        pivots,
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
    ax_work.set_title("Pivots per call")
    ax_work.set_xlabel("Simplex call order")
    ax_work.set_ylabel("Pivots within call")
    ax_work.grid(alpha=0.3)

    cycle_calls = [(number, call) for number, call in zip(call_numbers, calls)
                   if call.get("cycles", 0)]
    if cycle_calls:
        for axis, field in ((ax_duration, "duration_seconds"), (ax_work, "pivots")):
            axis.scatter([number for number, _ in cycle_calls],
                         [call.get(field, 0) for _, call in cycle_calls],
                         marker="^", s=38, color="#7c3aed", zorder=5,
                         label="Cycle detected / random pivot")
            axis.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0),
                        borderaxespad=0, ncol=1, fontsize=9)
        ax_work.text(0.02, 0.96,
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
    ax_results.set_title("Call results (all calls)")
    _set_integer_ticks(ax_results, "y")
    ax_results.tick_params(axis="x", labelrotation=20)

    violated = feedback.get("top_violated_constraints", [])[:8]
    if violated:
        ax_violations.barh(
            [_constraint_label(item) for item in reversed(violated)],
            [int(item.get("count", 0)) for item in reversed(violated)],
            color="#ef4444",
        )
        ax_violations.tick_params(axis="y", labelsize=7)
    ax_violations.set_title("Repeated bound violations")
    _annotate_violation_variable_types(ax_violations)
    _set_integer_ticks(ax_violations, "x")
    ax_violations.grid(axis="x", alpha=0.3)

    fig.text(
        0.01,
        0.01,
        f"calls={feedback['counts']['simplex_calls']} | pivots={feedback['counts']['simplex_pivots']}",
        fontsize=9,
    )
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
    return fig, axes


class SolverProgressVisualizer:
    def __init__(
        self,
        log_path: str | Path,
        output_path: str | Path,
        *,
        live_view: bool = False,
        open_live_view: bool = False,
        refresh_interval_ms: int = 500,
    ) -> None:
        self.log_path = Path(log_path)
        self.output_path = Path(output_path)
        self.live_view_path = self.output_path.with_suffix(".html")
        self.live_view = live_view or open_live_view
        self.open_live_view = open_live_view
        self.refresh_interval_ms = refresh_interval_ms
        self._view_opened = False
        self._state_lock = threading.Lock()
        self._pending = False
        self._worker: Optional[threading.Thread] = None

    def _write_live_view(self) -> None:
        image_name = json.dumps(self.output_path.name)
        html = f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Solver progress</title><style>
body{{margin:0;background:#111827;color:#f9fafb;font-family:sans-serif}}main{{max-width:1500px;margin:auto;padding:18px}}
h1{{margin:0 0 4px}}p{{margin:0 0 12px;color:#9ca3af}}img{{width:100%;display:block;background:white;border-radius:8px}}
</style></head><body><main><h1>DPLL(T) / Simplex progress</h1>
<p>이미지 약 {REALTIME_RENDER_INTERVAL_SECONDS:g}초 주기 생성 · 브라우저 {self.refresh_interval_ms}ms 간격 확인</p><img id="dashboard" alt="solver progress dashboard"></main>
<script>const name={image_name},img=document.getElementById('dashboard');
function refresh(){{img.src=name+'?t='+Date.now()}}refresh();setInterval(refresh,{self.refresh_interval_ms});</script>
</body></html>"""
        self.live_view_path.parent.mkdir(parents=True, exist_ok=True)
        self.live_view_path.write_text(html, encoding="utf-8")

    def _render_once(self) -> None:
        import matplotlib.pyplot as plt

        with render_turn():
            fig, _ = draw_solver_dashboard(self.log_path)
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.output_path.with_name(f"{self.output_path.stem}.tmp{self.output_path.suffix}")
            try:
                fig.savefig(temporary, dpi=110, bbox_inches="tight")
                os.replace(temporary, self.output_path)
            finally:
                plt.close(fig)
                if temporary.exists():
                    temporary.unlink()
        if self.live_view:
            if not self.live_view_path.exists():
                self._write_live_view()
            if self.open_live_view and not self._view_opened:
                self._view_opened = True
                webbrowser.open(self.live_view_path.resolve().as_uri())

    def _run_updates(self) -> None:
        try:
            while True:
                with self._state_lock:
                    self._pending = False
                self._render_once()
                with self._state_lock:
                    if self._pending:
                        continue
                    self._worker = None
                    return
        except Exception as exc:
            warnings.warn(f"solver progress background render 실패: {exc}")
            with self._state_lock:
                self._worker = None

    def request_update(
        self, *, panel: Optional[str] = None, event: Optional[str] = None
    ) -> None:
        with self._state_lock:
            self._pending = True
            if self._worker is not None:
                return
            self._worker = threading.Thread(
                target=self._run_updates,
                name="solver-progress-renderer",
                daemon=True,
            )
            self._worker.start()

    def flush(self) -> None:
        while True:
            with self._state_lock:
                worker = self._worker
            if worker is None:
                return
            worker.join()


class SolverProgressPanelVisualizer(SolverProgressVisualizer):
    """DPLL/Theory와 Simplex를 서로 다른 PNG로 비동기 렌더링한다.

    Simplex의 많은 이벤트는 최신 상태 하나로 합치지만, round 시작/종료 프레임은
    FIFO에 보존한다. 따라서 PNG 생성 중 다음 round가 끝나도 브라우저가 각 round
    상태를 최소 한 번은 볼 수 있다.
    """

    def __init__(
        self,
        log_path: str | Path,
        base_output_path: str | Path,
        *,
        enabled_panels: Optional[Iterable[str]] = None,
        refresh_interval_ms: int = 500,
    ) -> None:
        super().__init__(
            log_path,
            base_output_path,
            refresh_interval_ms=refresh_interval_ms,
        )
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
        self._preserved_frames = deque()
        self._round_active = False

    def _render_once(
        self,
        panels: Optional[Iterable[str]] = None,
        events: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        import matplotlib.pyplot as plt

        renderers = {
            "dpll_theory": draw_dpll_theory_panel,
            "simplex": draw_simplex_panel,
        }
        with render_turn():
            selected = self.panel_paths if panels is None else panels
            feedback = build_solver_feedback(
                read_solver_progress(self.log_path) if events is None else events
            )
            for name in selected:
                renderer = renderers[name]
                output = self.panel_paths[name]
                output.parent.mkdir(parents=True, exist_ok=True)
                temporary = output.with_name(f"{output.stem}.tmp{output.suffix}")
                fig, _ = renderer(feedback)
                try:
                    fig.savefig(temporary, dpi=100)
                    os.replace(temporary, output)
                finally:
                    plt.close(fig)
                    if temporary.exists():
                        temporary.unlink()

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
                submit_solver_render(
                    self.log_path, self.panel_paths, tuple(sorted(selected))
                ).result()
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


def write_solver_feedback(
    log_path: str | Path,
    image_path: str | Path,
    json_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    import matplotlib.pyplot as plt

    events = read_solver_progress(log_path)
    feedback = build_solver_feedback(events)
    image = Path(image_path)
    image.parent.mkdir(parents=True, exist_ok=True)
    with render_turn():
        fig, _ = draw_solver_dashboard(events, title="DPLL(T) / Simplex Feedback")
        fig.savefig(image, dpi=150, bbox_inches="tight")
        plt.close(fig)
    resolved_json = Path(json_path) if json_path is not None else image.with_suffix(".json")
    resolved_json.parent.mkdir(parents=True, exist_ok=True)
    resolved_json.write_text(
        json.dumps(feedback, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return feedback


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
