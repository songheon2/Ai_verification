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
import uuid
import warnings
import webbrowser
from collections import Counter, deque
from datetime import datetime
from pathlib import Path
from time import monotonic, sleep, time_ns
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional

import matplotlib

matplotlib.use("Agg")

_RENDER_LOCK = threading.Lock()


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
    ) -> None:
        allowed_panels = {"dpll_theory", "simplex"}
        selected_panels = (
            allowed_panels if enabled_panels is None else set(enabled_panels)
        )
        unknown_panels = selected_panels - allowed_panels
        if unknown_panels:
            names = ", ".join(sorted(unknown_panels))
            raise ValueError(f"지원하지 않는 solver progress panel: {names}")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if reset_log:
            self.path.write_text("", encoding="utf-8")
        else:
            self.path.touch(exist_ok=True)
        self.update_callback = update_callback
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
        self._lock = threading.RLock()
        self._last_update = 0.0
        self._round_started: Dict[int, int] = {}
        self._round_simplex_calls: Counter[int] = Counter()
        self._active_round: Optional[int] = None
        self._simplex_started: Dict[str, int] = {}
        self._closed_simplex: set[str] = set()

    def _append(self, event: str, *, force_update: bool = False, **payload: Any) -> None:
        panel = "simplex" if event.startswith("simplex_") else "dpll_theory"
        if event != "solver_end" and panel not in self.enabled_panels:
            return
        record = {
            "timestamp": datetime.now().astimezone().isoformat(timespec="microseconds"),
            "time_ns": time_ns(),
            "event": event,
            **_json_safe(payload),
        }
        with self._lock:
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    + "\n"
                )
                stream.flush()
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
        self._append(
            "theory_selection",
            round=round_index,
            atom_count=len(atoms),
            atoms=[dict(atom) for atom in atoms],
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
        theory_atom_ids: Iterable[str],
        row_count: int,
        variable_count: int,
    ) -> str:
        call_id = f"simplex-{uuid.uuid4().hex}"
        with self._lock:
            self._simplex_started[call_id] = time_ns()
            if round_index is not None:
                self._round_simplex_calls[round_index] += 1
        self._append(
            "simplex_start",
            call_id=call_id,
            round=round_index,
            depth=depth,
            origin=origin,
            theory_atom_ids=list(theory_atom_ids),
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
        self._append(
            "simplex_iteration",
            call_id=call_id,
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
        self._append(
            "simplex_pivot",
            call_id=call_id,
            iteration=iteration,
            entering=entering,
            leaving=leaving,
            row=row,
            force_update=False,
        )

    def simplex_end(
        self,
        call_id: str,
        result: str,
        *,
        iterations: int,
        pivots: int,
    ) -> None:
        with self._lock:
            if call_id in self._closed_simplex:
                return
            self._closed_simplex.add(call_id)
            started = self._simplex_started.pop(call_id, time_ns())
        self._append(
            "simplex_end",
            call_id=call_id,
            result=result,
            duration_seconds=max(0.0, (time_ns() - started) / 1_000_000_000),
            iterations=iterations,
            pivots=pivots,
            force_update=False,
        )

    def solver_end(self, status: str, reason: Optional[str], rounds: int) -> None:
        self._append(
            "solver_end",
            status=status,
            reason=reason,
            rounds=rounds,
            force_update=True,
        )


def build_solver_feedback(events: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    rounds: Dict[int, Dict[str, Any]] = {}
    calls: Dict[str, Dict[str, Any]] = {}
    violations: Counter[str] = Counter()
    entering: Counter[str] = Counter()
    leaving: Counter[str] = Counter()
    pivot_rows: Counter[str] = Counter()
    theory_links: List[Dict[str, Any]] = []
    solver_result: Dict[str, Any] = {}

    for raw in events:
        event = dict(raw)
        kind = event.get("event")
        if kind == "round_start":
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
        elif kind == "theory_result":
            index = int(event["round"])
            result = event.get("result")
            rounds.setdefault(index, {"round": index})["theory_result"] = result
            theory_links.append({"round": index, "result": result, **{k: v for k, v in event.items() if k not in {"event", "timestamp", "time_ns", "round", "result"}}})
        elif kind == "simplex_start":
            call_id = str(event["call_id"])
            calls[call_id] = {
                "call_id": call_id,
                "start_time_ns": event.get("time_ns"),
                "round": event.get("round"),
                "depth": event.get("depth"),
                "origin": event.get("origin"),
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
        elif kind == "simplex_pivot":
            call_id = str(event.get("call_id"))
            call = calls.setdefault(call_id, {"call_id": call_id})
            call["pivots"] = int(call.get("pivots", 0)) + 1
            entering[str(event.get("entering"))] += 1
            leaving[str(event.get("leaving"))] += 1
            pivot_rows[str(event.get("row"))] += 1
        elif kind == "simplex_end":
            call_id = str(event["call_id"])
            calls.setdefault(call_id, {"call_id": call_id}).update(
                {
                    "result": event.get("result"),
                    "duration_seconds": float(event.get("duration_seconds", 0.0)),
                    "iterations": int(event.get("iterations", 0)),
                    "pivots": int(event.get("pivots", 0)),
                }
            )
        elif kind == "solver_end":
            solver_result = {
                "status": event.get("status"),
                "reason": event.get("reason"),
                "rounds": event.get("rounds"),
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
    if result in {"TIMEOUT", "ITERATION_LIMIT", "SIMPLEX_ITERATION_LIMIT"}:
        return "#f59e0b"
    return "#2563eb"


def _set_integer_ticks(axis, direction: str) -> None:
    """호출·반복·선택 횟수처럼 이산적인 축에 정수 눈금만 사용한다."""
    from matplotlib.ticker import MaxNLocator

    target = axis.xaxis if direction == "x" else axis.yaxis
    target.set_major_locator(MaxNLocator(integer=True, min_n_ticks=1))


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
    calls = feedback["simplex_calls"][-20:]

    fig = plt.figure(figsize=(15, 13))
    grid = fig.add_gridspec(4, 2, height_ratios=[1, 1, 1, 0.9])
    ax_round = fig.add_subplot(grid[0, 0])
    ax_duration = fig.add_subplot(grid[0, 1])
    ax_work = fig.add_subplot(grid[1, 0])
    ax_vars = fig.add_subplot(grid[1, 1])
    ax_choices = fig.add_subplot(grid[2, 0])
    ax_rows = fig.add_subplot(grid[2, 1])
    ax_links = fig.add_subplot(grid[3, :])
    axes = [ax_round, ax_duration, ax_work, ax_vars, ax_choices, ax_rows, ax_links]
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
    ax_duration.set_title("Simplex call duration (latest 20)")
    ax_duration.set_xlabel("Simplex call")
    ax_duration.set_ylabel("Duration (s)")
    ax_duration.set_ylim(bottom=0)
    ax_duration.grid(axis="y", alpha=0.3)

    iterations = [int(item.get("iterations", 0)) for item in calls]
    pivots = [int(item.get("pivots", 0)) for item in calls]
    ax_work.plot(call_numbers, iterations, marker="o", label="iterations", color="#2563eb")
    ax_work.plot(call_numbers, pivots, marker="s", label="pivots", color="#9333ea")
    ax_work.set_xticks(call_numbers)
    _set_integer_ticks(ax_work, "y")
    ax_work.set_title("Simplex work per call")
    ax_work.set_xlabel("Simplex call")
    ax_work.set_ylabel("Count")
    ax_work.legend()
    ax_work.grid(alpha=0.3)

    violated = Counter(feedback["top_violated_variables"])
    selected = violated.most_common(8)
    if selected:
        labels = [name for name, _ in reversed(selected)]
        values = [value for _, value in reversed(selected)]
        ax_vars.barh(labels, values, color="#ef4444")
    ax_vars.set_title("Repeated bound violations")
    ax_vars.set_xlabel("Selections")
    _set_integer_ticks(ax_vars, "x")
    ax_vars.grid(axis="x", alpha=0.3)

    entering_counts = Counter(feedback["top_entering_variables"])
    leaving_counts = Counter(feedback["top_leaving_variables"])
    choice_names = [
        name for name, _ in (entering_counts + leaving_counts).most_common(8)
    ]
    choice_positions = list(range(len(choice_names)))
    ax_choices.barh(
        [position - 0.2 for position in choice_positions],
        [entering_counts[name] for name in choice_names],
        height=0.4,
        color="#16a34a",
        label="entering",
    )
    ax_choices.barh(
        [position + 0.2 for position in choice_positions],
        [leaving_counts[name] for name in choice_names],
        height=0.4,
        color="#f97316",
        label="leaving",
    )
    ax_choices.set_yticks(choice_positions)
    ax_choices.set_yticklabels(choice_names)
    ax_choices.set_title("Entering / leaving variable frequency")
    ax_choices.set_xlabel("Selections")
    _set_integer_ticks(ax_choices, "x")
    ax_choices.legend()
    ax_choices.grid(axis="x", alpha=0.3)

    pivot_rows = Counter(feedback["top_pivot_rows"]).most_common(8)
    if pivot_rows:
        row_labels = [name for name, _ in reversed(pivot_rows)]
        row_values = [value for _, value in reversed(pivot_rows)]
        ax_rows.barh(row_labels, row_values, color="#9333ea")
    ax_rows.set_title("Repeatedly pivoted rows")
    ax_rows.set_xlabel("Pivots")
    _set_integer_ticks(ax_rows, "x")
    ax_rows.grid(axis="x", alpha=0.3)

    ax_links.axis("off")
    ax_links.set_title("DPLL → theory atoms → Simplex calls → theory result", loc="left")
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
            f"  →  {', '.join(call_labels) or 'no Simplex'}"
            f"  →  {round_item.get('theory_result', round_item.get('result', 'RUNNING'))}"
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
    fig.text(
        0.01,
        0.01,
        " | ".join(
            [
                f"rounds={feedback['counts']['rounds']}",
                f"simplex calls={feedback['counts']['simplex_calls']}",
                f"iterations={feedback['counts']['simplex_iterations']}",
                f"pivots={feedback['counts']['simplex_pivots']}",
                f"latest round result={latest_round.get('result', 'RUNNING')}",
                f"solver={solver.get('status', 'RUNNING')} / {solver.get('reason', '-')}",
            ]
        ),
        fontsize=9,
    )
    fig.subplots_adjust(
        left=0.08,
        right=0.98,
        top=0.94,
        bottom=0.08,
        hspace=0.58,
        wspace=0.24,
    )
    return fig, axes


def solver_panel_paths(base_path: str | Path) -> Dict[str, Path]:
    base = Path(base_path)
    return {
        name: base.with_name(f"{base.stem}_{name}{base.suffix}")
        for name in ("dpll_theory", "simplex")
    }


def _feedback_from(events_or_path) -> Dict[str, Any]:
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
            f"   {call_summary} → {theory_result}"
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
    ax_results.set_xticks([])
    ax_results.set_title("Theory results")
    ax_results.set_xlabel("")

    link_lines = _compact_theory_flow_lines(feedback, rounds)
    ax_links.axis("off")
    ax_links.set_title("Round → selected atoms → Simplex → theory result", loc="left")
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
        figsize=(15, 9),
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

    results = Counter(
        outcome for item in rounds if (outcome := _theory_outcome(item)) is not None
    )
    labels = list(results)
    values = [results[label] for label in labels]
    ax_results.barh(labels, values, color=[_result_color(label) for label in labels])
    ax_results.set_xticks([])
    ax_results.set_title("Theory results")
    ax_results.set_xlabel("")

    link_lines = _compact_theory_flow_lines(feedback, rounds)
    ax_links.axis("off")
    ax_links.set_title("Round → selected atoms → Simplex → theory result", loc="left")
    ax_links.text(
        0.01,
        0.96,
        "\n".join(link_lines) if link_lines else "No theory selection yet",
        transform=ax_links.transAxes,
        va="top",
        family="monospace",
        fontsize=8.5,
    )
    solver = feedback.get("solver", {})
    fig.text(
        0.01,
        0.01,
        f"rounds={feedback['counts']['rounds']} | solver={solver.get('status', 'RUNNING')} / {solver.get('reason', '-')}",
        fontsize=9,
    )
    fig.subplots_adjust(left=0.07, right=0.98, top=0.91, bottom=0.08, hspace=0.38, wspace=0.28)
    return fig, axes


def draw_simplex_panel(events_or_path, *, title: str = "Simplex Progress"):
    import matplotlib.pyplot as plt

    feedback = _feedback_from(events_or_path)
    calls = feedback["simplex_calls"][-20:]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    fig.suptitle(title, fontsize=16, fontweight="bold")
    ax_duration, ax_work, ax_results, ax_violations, ax_choices, ax_rows = axes.flat
    call_numbers = list(
        range(
            max(1, len(feedback["simplex_calls"]) - len(calls) + 1),
            len(feedback["simplex_calls"]) + 1,
        )
    )
    durations = [
        max(0.0, float(item.get("duration_seconds", 0.0))) for item in calls
    ]
    ax_duration.bar(call_numbers, durations, color=[_result_color(item.get("result")) for item in calls])
    ax_duration.set_xticks(call_numbers)
    ax_duration.set_title("Call duration")
    ax_duration.set_xlabel("Simplex call")
    ax_duration.set_ylabel("Seconds")
    ax_duration.set_ylim(bottom=0)
    ax_duration.grid(axis="y", alpha=0.3)

    iterations = [int(item.get("iterations", 0)) for item in calls]
    pivots = [int(item.get("pivots", 0)) for item in calls]
    ax_work.plot(call_numbers, iterations, marker="o", label="iterations")
    ax_work.plot(call_numbers, pivots, marker="s", label="pivots")
    ax_work.set_xticks(call_numbers)
    _set_integer_ticks(ax_work, "y")
    ax_work.set_title("Iterations / pivots per call")
    ax_work.legend()
    ax_work.grid(alpha=0.3)

    result_counts = Counter(item.get("result", "RUNNING") for item in calls)
    result_labels = list(result_counts)
    ax_results.bar(
        result_labels,
        [result_counts[label] for label in result_labels],
        color=[_result_color(label) for label in result_labels],
    )
    ax_results.set_title("Call results")
    _set_integer_ticks(ax_results, "y")
    ax_results.tick_params(axis="x", labelrotation=20)

    violated = Counter(feedback["top_violated_variables"]).most_common(8)
    if violated:
        ax_violations.barh(
            [name for name, _ in reversed(violated)],
            [value for _, value in reversed(violated)],
            color="#ef4444",
        )
    ax_violations.set_title("Repeated bound violations")
    _set_integer_ticks(ax_violations, "x")
    ax_violations.grid(axis="x", alpha=0.3)

    entering = Counter(feedback["top_entering_variables"])
    leaving = Counter(feedback["top_leaving_variables"])
    names = [name for name, _ in (entering + leaving).most_common(8)]
    positions = list(range(len(names)))
    ax_choices.barh(
        [position - 0.2 for position in positions],
        [entering[name] for name in names],
        height=0.4,
        label="entering",
        color="#16a34a",
    )
    ax_choices.barh(
        [position + 0.2 for position in positions],
        [leaving[name] for name in names],
        height=0.4,
        label="leaving",
        color="#f97316",
    )
    ax_choices.set_yticks(positions)
    ax_choices.set_yticklabels(names)
    ax_choices.set_title("Entering / leaving frequency")
    _set_integer_ticks(ax_choices, "x")
    ax_choices.legend()
    ax_choices.grid(axis="x", alpha=0.3)

    rows = Counter(feedback["top_pivot_rows"]).most_common(8)
    if rows:
        ax_rows.barh(
            [name for name, _ in reversed(rows)],
            [value for _, value in reversed(rows)],
            color="#9333ea",
        )
    ax_rows.set_title("Repeatedly pivoted rows")
    _set_integer_ticks(ax_rows, "x")
    ax_rows.grid(axis="x", alpha=0.3)
    fig.text(
        0.01,
        0.01,
        f"calls={feedback['counts']['simplex_calls']} | iterations={feedback['counts']['simplex_iterations']} | pivots={feedback['counts']['simplex_pivots']}",
        fontsize=9,
    )
    fig.subplots_adjust(left=0.08, right=0.98, top=0.91, bottom=0.08, hspace=0.38, wspace=0.3)
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
<p>{self.refresh_interval_ms}ms 간격 자동 갱신</p><img id="dashboard" alt="solver progress dashboard"></main>
<script>const name={image_name},img=document.getElementById('dashboard');
function refresh(){{img.src=name+'?t='+Date.now()}}refresh();setInterval(refresh,{self.refresh_interval_ms});</script>
</body></html>"""
        self.live_view_path.parent.mkdir(parents=True, exist_ok=True)
        self.live_view_path.write_text(html, encoding="utf-8")

    def _render_once(self) -> None:
        import matplotlib.pyplot as plt

        with _RENDER_LOCK:
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
        with _RENDER_LOCK:
            selected = self.panel_paths if panels is None else panels
            for name in selected:
                renderer = renderers[name]
                output = self.panel_paths[name]
                output.parent.mkdir(parents=True, exist_ok=True)
                temporary = output.with_name(f"{output.stem}.tmp{output.suffix}")
                fig, _ = renderer(self.log_path if events is None else events)
                try:
                    fig.savefig(temporary, dpi=110, bbox_inches="tight")
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
        if not selected:
            return

        preserve = panel == "dpll_theory" and event in {
            "round_start",
            "round_end",
            "solver_end",
        }
        snapshot = read_solver_progress(self.log_path) if preserve else None
        with self._state_lock:
            if preserve:
                self._preserved_frames.append((tuple(selected), snapshot))
            else:
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
                preserved = False
                with self._state_lock:
                    if self._preserved_frames:
                        panels, events = self._preserved_frames.popleft()
                        preserved = True
                    elif self._pending_panels:
                        panels = tuple(sorted(self._pending_panels))
                        self._pending_panels.clear()
                        events = None
                    else:
                        self._worker = None
                        return
                self._render_once(panels, events)
                if preserved:
                    # polling 직전에 파일이 바뀐 경우에도 다음 polling에서 이
                    # 프레임을 읽을 수 있도록 한 주기보다 조금 오래 유지한다.
                    sleep(self.refresh_interval_ms / 1000.0 * 1.25)
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
    with _RENDER_LOCK:
        fig, _ = draw_solver_dashboard(events, title="DPLL(T) / Simplex Feedback")
        fig.savefig(image, dpi=150, bbox_inches="tight")
        plt.close(fig)
    resolved_json = Path(json_path) if json_path is not None else image.with_suffix(".json")
    resolved_json.parent.mkdir(parents=True, exist_ok=True)
    resolved_json.write_text(json.dumps(feedback, ensure_ascii=False, indent=2), encoding="utf-8")
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
    with _RENDER_LOCK:
        for name in paths:
            renderer = renderers[name]
            output = paths[name]
            output.parent.mkdir(parents=True, exist_ok=True)
            fig, _ = renderer(events, title=titles[name])
            fig.savefig(output, dpi=150, bbox_inches="tight")
            plt.close(fig)
    resolved_json = (
        Path(json_path) if json_path is not None else Path(base_image_path).with_suffix(".json")
    )
    resolved_json.parent.mkdir(parents=True, exist_ok=True)
    resolved_json.write_text(
        json.dumps(feedback, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return feedback, paths
