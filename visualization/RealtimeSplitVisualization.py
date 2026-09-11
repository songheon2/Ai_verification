"""실시간 ReLU split 이벤트 로그와 layer/index 기반 visualizer.

로그는 append-only JSON Lines 형식이다. 같은 ``split_id``의 ``+``와 ``-``를
짝지을 수 있으므로 현재 열린 split 수뿐 아니라 사후 처리시간도 계산할 수 있다.
"""
from __future__ import annotations

import json
import os
import threading
import uuid
import warnings
import webbrowser
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import sleep, time_ns
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from visualization.RenderLock import serialized_render
from visualization.RenderProcess import (
    REALTIME_RENDER_INTERVAL_SECONDS,
    submit_relu_render,
)

DEFAULT_SPLIT_LOG_PATH = Path(__file__).resolve().parent / "outputs" / "log.txt"


def split_metadata_path(path: str | Path) -> Path:
    """split JSONL과 함께 보관하는 실행/모델 메타데이터 경로를 반환한다."""
    log_path = Path(path)
    return log_path.with_name(f"{log_path.stem}_meta.json")


def read_split_metadata(path: str | Path) -> Dict:
    metadata_path = split_metadata_path(path)
    if not metadata_path.exists():
        return {}
    try:
        value = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        warnings.warn(f"{metadata_path}의 split 메타데이터를 읽을 수 없습니다")
        return {}
    return value if isinstance(value, dict) else {}


@dataclass(frozen=True)
class SplitLogEvent:
    timestamp: str
    time_ns: int
    split_id: str
    layer: Optional[int]
    index: Optional[int]
    variable: str
    event: str

    @classmethod
    def from_dict(cls, value: Dict) -> "SplitLogEvent":
        event = str(value["event"])
        if event not in ("+", "-"):
            raise ValueError(f"지원하지 않는 split event: {event!r}")
        return cls(
            timestamp=str(value["timestamp"]),
            time_ns=int(value["time_ns"]),
            split_id=str(value["split_id"]),
            layer=None if value.get("layer") is None else int(value["layer"]),
            index=None if value.get("index") is None else int(value["index"]),
            variable=str(value.get("variable", "")),
            event=event,
        )


def read_split_events(path: str | Path) -> List[SplitLogEvent]:
    """손상되거나 작성 중인 마지막 줄은 건너뛰고 유효한 이벤트만 읽는다."""
    log_path = Path(path)
    if not log_path.exists():
        return []
    events: List[SplitLogEvent] = []
    with log_path.open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                events.append(SplitLogEvent.from_dict(json.loads(line)))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                warnings.warn(f"{log_path}:{line_no}의 잘못된 split 로그를 건너뜁니다")
    return events


def split_events_from_trace(trace) -> List[SplitLogEvent]:
    """feedback-only 실행의 in-memory split span을 로그 이벤트로 변환한다."""
    events: List[SplitLogEvent] = []
    sequence = 0
    for trace_event in trace.events:
        if trace_event.component != "reluplex_split":
            continue
        sequence += 1
        split_id = f"trace-{sequence}"
        layer = trace_event.meta.get("layer")
        index = trace_event.meta.get("index")
        common = {
            "timestamp": "",
            "split_id": split_id,
            "layer": None if layer is None else int(layer),
            "index": None if index is None else int(index),
            "variable": str(trace_event.branch_x or ""),
        }
        events.append(SplitLogEvent(
            time_ns=int(trace_event.t_start * 1_000_000_000),
            event="+",
            **common,
        ))
        if trace_event.t_end is not None:
            events.append(SplitLogEvent(
                time_ns=int(trace_event.t_end * 1_000_000_000),
                event="-",
                **common,
            ))
    return sorted(events, key=lambda event: event.time_ns)


def active_split_events(events: Iterable[SplitLogEvent]) -> List[SplitLogEvent]:
    """마지막 상태가 ``+``인 split을 시작 순서대로 반환한다."""
    active: Dict[str, SplitLogEvent] = {}
    for event in events:
        if event.event == "+":
            active[event.split_id] = event
        else:
            active.pop(event.split_id, None)
    return sorted(active.values(), key=lambda event: event.time_ns)


def split_neuron_counts(
    events: Iterable[SplitLogEvent],
) -> Tuple[Dict[Tuple[int, int], int], Dict[Tuple[int, int], int], int]:
    """``(layer, index)``별 현재 활성 수와 누적 시작 수를 계산한다.

    반환값은 ``(active_counts, total_counts, unknown_active)``다. 메타데이터가
    없는 이벤트도 전체 활성 수에서는 빠지지 않도록 ``unknown_active``로 센다.
    """
    event_list = list(events)
    totals: Counter[Tuple[int, int]] = Counter()
    for event in event_list:
        if event.event == "+" and event.layer is not None and event.index is not None:
            totals[(event.layer, event.index)] += 1

    active: Counter[Tuple[int, int]] = Counter()
    unknown_active = 0
    for event in active_split_events(event_list):
        if event.layer is None or event.index is None:
            unknown_active += 1
        else:
            active[(event.layer, event.index)] += 1
    return dict(active), dict(totals), unknown_active


class SplitEventLogger:
    """thread-safe append logger이자 프로세스 내 ``global_split_count`` 소유자."""

    def __init__(
        self,
        path: str | Path = DEFAULT_SPLIT_LOG_PATH,
        update_callback: Optional[Callable[[], None]] = None,
        *,
        reset_log: bool = False,
        model_layer_sizes: Optional[Sequence[int]] = None,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if reset_log:
            self.path.write_text("", encoding="utf-8")
            metadata_path = split_metadata_path(self.path)
            if model_layer_sizes is None and metadata_path.exists():
                metadata_path.unlink()
        else:
            self.path.touch(exist_ok=True)
        if model_layer_sizes is not None:
            normalized_sizes = [int(size) for size in model_layer_sizes]
            split_metadata_path(self.path).write_text(
                json.dumps(
                    {"model_layer_sizes": normalized_sizes},
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        self.update_callback = update_callback
        self.global_split_count = 0
        self._sequence = 0
        self._lock = threading.Lock()

    def begin(self, variable: str, layer: Optional[int], index: Optional[int]) -> str:
        with self._lock:
            self._sequence += 1
            split_id = f"{uuid.uuid4().hex}-{self._sequence}"
            self.global_split_count += 1
            self._append(split_id, variable, layer, index, "+")
            return split_id

    def round_start(self, round_index: int) -> None:
        """Record boundaries separately from neuron +/- events."""
        with self._lock:
            metadata = read_split_metadata(self.path)
            rounds = metadata.setdefault("round_starts", [])
            rounds.append({"round": round_index, "time_ns": time_ns()})
            destination = split_metadata_path(self.path)
            temporary = destination.with_suffix(".tmp")
            temporary.write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")
            os.replace(temporary, destination)
        self.request_update()

    def end(self, split_id: str, variable: str, layer: Optional[int], index: Optional[int]) -> None:
        with self._lock:
            self.global_split_count = max(0, self.global_split_count - 1)
            self._append(split_id, variable, layer, index, "-")

    def _append(
        self,
        split_id: str,
        variable: str,
        layer: Optional[int],
        index: Optional[int],
        event: str,
    ) -> None:
        now_ns = time_ns()
        record = {
            "timestamp": datetime.now().astimezone().isoformat(timespec="microseconds"),
            "time_ns": now_ns,
            "split_id": split_id,
            "layer": layer,
            "index": index,
            "variable": variable,
            "event": event,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            stream.flush()
        self.request_update()

    def request_update(self) -> None:
        """현재 로그 상태를 visualizer에 알린다. callback 오류는 solver와 격리한다."""
        if self.update_callback is not None:
            try:
                self.update_callback()
            except Exception as exc:  # 시각화 실패가 solver 결과를 바꾸면 안 된다.
                warnings.warn(f"실시간 split visualizer 업데이트 실패: {exc}")


def _draw_round_boundaries(ax, rounds, origin_ns, unit_seconds=1.0):
    """Draw full-height boundaries only after more than one round began."""
    rounds = sorted({item["round"]: item for item in rounds}.values(),
                    key=lambda item: item["time_ns"])
    if len(rounds) < 2:
        return
    left, right = ax.get_xlim()
    for index, item in enumerate(rounds):
        x = (item["time_ns"] - origin_ns) / 1e9 / unit_seconds
        end = ((rounds[index + 1]["time_ns"] - origin_ns) / 1e9 / unit_seconds
               if index + 1 < len(rounds) else right)
        if left <= x <= right:
            ax.axvline(x, ymin=0, ymax=1, color="#374151",
                       linestyle="--", linewidth=1.1, zorder=4)
        if end > left and x < right:
            ax.text(max(left, x) + (right - left) * 0.006, 0.98,
                    f"D{item['round']}", transform=ax.get_xaxis_transform(),
                    ha="left", va="top", fontsize=8, color="#111827",
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8},
                    clip_on=True, zorder=5)


def _draw_realtime_split_history(
    events: List[SplitLogEvent],
    *,
    ax,
    title: Optional[str] = None,
    window_seconds: float = 300.0,
    current_time_ns: Optional[int] = None,
    round_starts=(),
):
    if window_seconds <= 0:
        raise ValueError("window_seconds는 양수여야 합니다")
    from matplotlib.ticker import MaxNLocator

    active = active_split_events(events)
    started_count = sum(event.event == "+" for event in events)
    completed_count = sum(event.event == "-" for event in events)
    ax.clear()

    now_ns = time_ns() if current_time_ns is None else int(current_time_ns)
    window_ns = int(window_seconds * 1_000_000_000)
    window_start_ns = now_ns - window_ns
    count = 0
    for event in events:
        if event.time_ns >= window_start_ns:
            break
        if event.time_ns <= now_ns:
            count = count + 1 if event.event == "+" else max(0, count - 1)

    xs = [-window_seconds]
    ys = [count]
    for event in events:
        if event.time_ns < window_start_ns:
            continue
        if event.time_ns > now_ns:
            break
        count = count + 1 if event.event == "+" else max(0, count - 1)
        xs.append((event.time_ns - now_ns) / 1_000_000_000)
        ys.append(count)
    xs.append(0.0)
    ys.append(count)

    line_color = "#2563eb"
    ax.step(xs, ys, where="post", linewidth=2.2, color=line_color)
    ax.fill_between(xs, ys, step="post", color=line_color, alpha=0.16)
    max_count = max(ys, default=0)
    ax.set_xlim(-window_seconds, 0)
    ax.set_ylim(0, max(1, max_count + 1))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    tick_count = 6
    fixed_ticks = [
        -window_seconds + window_seconds * index / tick_count
        for index in range(tick_count + 1)
    ]
    ax.set_xticks(fixed_ticks)
    ax.set_xticklabels([f"{tick:g}" for tick in fixed_ticks])
    ax.grid(True, color="#d1d5db", linewidth=0.7, alpha=0.7)
    ax.set_ylabel("Active ReLU splits")
    ax.set_xlabel(
        f"Time from now (s), fixed {window_seconds:g}s window  |  "
        f"started: {started_count}  |  completed: {completed_count}"
    )
    ax.set_title(title or "Realtime ReLU splits")
    ax.text(
        0.985,
        0.93,
        f"ACTIVE  {len(active)}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=12,
        fontweight="bold",
        color=line_color,
    )
    return ax


def _draw_full_split_history(
    events: List[SplitLogEvent],
    *,
    ax,
    title: Optional[str] = None,
    round_starts=(),
):
    """첫 split부터 마지막 split까지 잘리지 않는 사후 시간 이력을 그린다."""
    from matplotlib.ticker import MaxNLocator

    ordered = sorted(events, key=lambda event: event.time_ns)
    started_count = sum(event.event == "+" for event in ordered)
    completed_count = sum(event.event == "-" for event in ordered)
    ax.clear()

    if ordered:
        first_ns = ordered[0].time_ns
        last_ns = ordered[-1].time_ns
        duration_seconds = max(0.0, (last_ns - first_ns) / 1_000_000_000)
    else:
        first_ns = 0
        duration_seconds = 0.0

    if duration_seconds >= 2 * 24 * 60 * 60:
        unit_seconds, unit_label = 24 * 60 * 60, "days"
    elif duration_seconds >= 2 * 60 * 60:
        unit_seconds, unit_label = 60 * 60, "hours"
    elif duration_seconds >= 2 * 60:
        unit_seconds, unit_label = 60, "minutes"
    else:
        unit_seconds, unit_label = 1, "seconds"

    count = 0
    xs = [0.0]
    ys = [0]
    for event in ordered:
        count = count + 1 if event.event == "+" else max(0, count - 1)
        xs.append(
            (event.time_ns - first_ns) / 1_000_000_000 / unit_seconds
        )
        ys.append(count)

    duration_units = duration_seconds / unit_seconds
    xs.append(duration_units)
    ys.append(count)

    line_color = "#2563eb"
    ax.step(xs, ys, where="post", linewidth=2.2, color=line_color)
    ax.fill_between(xs, ys, step="post", color=line_color, alpha=0.16)
    max_count = max(ys, default=0)
    ax.set_xlim(0, max(1.0, duration_units))
    ax.set_ylim(0, max(1, max_count + 1))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, color="#d1d5db", linewidth=0.7, alpha=0.7)
    ax.set_ylabel("Active ReLU splits")
    ax.set_xlabel(
        f"Elapsed time from first split ({unit_label}), full period  |  "
        f"started: {started_count}  |  completed: {completed_count}"
    )
    ax.set_title(title or "Full active split history")
    _draw_round_boundaries(ax, round_starts, first_ns, unit_seconds)
    ax.text(
        0.985,
        0.93,
        f"ACTIVE  {count}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=12,
        fontweight="bold",
        color=line_color,
    )
    return ax


def draw_realtime_split_history(
    path: str | Path,
    *,
    ax=None,
    title: Optional[str] = None,
    window_seconds: float = 300.0,
    current_time_ns: Optional[int] = None,
):
    """현재를 오른쪽 0에 둔 고정 폭 시간창에 활성 split 수를 그린다."""
    import matplotlib.pyplot as plt

    events = read_split_events(path)
    own_figure = ax is None
    if own_figure:
        fig, ax = plt.subplots(figsize=(8, 2.4))
    else:
        fig = ax.figure
    _draw_realtime_split_history(
        events,
        ax=ax,
        title=title,
        window_seconds=window_seconds,
        current_time_ns=current_time_ns,
    )
    return (fig, ax) if own_figure else ax


def draw_realtime_split_dashboard(
    path: str | Path,
    *,
    title: str = "Realtime ReLU Split Dashboard",
    window_seconds: float = 300.0,
    events: Optional[List[SplitLogEvent]] = None,
    playback: bool = False,
    full_history: bool = False,
    current_time_ns: Optional[int] = None,
    model_layer_sizes: Optional[Sequence[int]] = None,
    round_starts=None,
):
    """시간 이력과 ``(layer, index)``별 활성/누적 split을 함께 그린다."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    if events is None:
        events = read_split_events(path)
    if full_history and round_starts is None:
        round_starts = read_split_metadata(path).get("round_starts", [])
    active_counts, total_counts, unknown_active = split_neuron_counts(events)
    fig = plt.figure(figsize=(10.5, 7.2))
    grid = fig.add_gridspec(2, 1, height_ratios=(1.0, 1.65), hspace=0.42)
    history_ax = fig.add_subplot(grid[0, 0])
    neuron_ax = fig.add_subplot(grid[1, 0])
    if full_history:
        _draw_full_split_history(
            events,
            ax=history_ax,
            title="Full active split history",
            round_starts=round_starts,
        )
    else:
        _draw_realtime_split_history(
            events,
            ax=history_ax,
            title="Active split history",
            window_seconds=window_seconds,
            current_time_ns=current_time_ns,
            round_starts=round_starts,
        )

    observed = sorted(total_counts)
    if observed:
        layers = [layer for layer, _ in observed]
        indices = [index for _, index in observed]
        totals = [total_counts[key] for key in observed]
        active = [active_counts.get(key, 0) for key in observed]
        edge_colors = ["#2563eb" if count else "#9ca3af" for count in active]
        edge_widths = [3.0 if count else 0.8 for count in active]
        marker_sizes = [90.0 + 45.0 * min(count, 5) for count in active]
        points = neuron_ax.scatter(
            layers,
            indices,
            c=totals,
            s=marker_sizes,
            cmap="Reds",
            vmin=0,
            vmax=max(1, max(totals)),
            marker="s",
            edgecolors=edge_colors,
            linewidths=edge_widths,
        )
        for (layer, index), active_count in zip(observed, active):
            if active_count:
                neuron_ax.annotate(
                    str(index),
                    (layer, index),
                    xytext=(8, 0),
                    textcoords="offset points",
                    ha="left",
                    va="center",
                    fontsize=8,
                    fontweight="bold",
                    color="#1e3a8a",
                    bbox={
                        "boxstyle": "round,pad=0.15",
                        "fc": "white",
                        "ec": "none",
                        "alpha": 0.8,
                    },
                )
        colorbar = fig.colorbar(points, ax=neuron_ax, fraction=0.035, pad=0.03)
        colorbar.set_label("Cumulative split starts")
        neuron_ax.set_xticks(sorted(set(layers)))
        neuron_ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        neuron_ax.invert_yaxis()
        neuron_ax.legend(
            handles=[
                Line2D(
                    [0], [0], marker="s", linestyle="", markerfacecolor="#fecaca",
                    markeredgecolor="#2563eb", markeredgewidth=3,
                    markersize=10, label="currently active",
                )
            ],
            loc="upper right",
        )
    else:
        neuron_ax.text(
            0.5,
            0.5,
            "No layer/index split events yet",
            transform=neuron_ax.transAxes,
            ha="center",
            va="center",
            color="#6b7280",
            fontsize=12,
        )
        neuron_ax.set_xlim(0, 1)
        neuron_ax.set_ylim(1, 0)

    active_total = sum(active_counts.values()) + unknown_active
    active_layers = len({layer for layer, _ in active_counts})
    neuron_ax.set_title(
        f"Observed ReLU neurons  |  active={active_total}  |  "
        f"active layers={active_layers}  |  metadata-missing active={unknown_active}",
        loc="left",
    )
    neuron_ax.set_xlabel("Layer")
    neuron_ax.set_ylabel("Neuron index")
    neuron_ax.grid(True, color="#d1d5db", linewidth=0.6, alpha=0.55)
    if model_layer_sizes:
        sizes = [int(size) for size in model_layer_sizes]
        layer_parts = []
        for layer, size in enumerate(sizes):
            if layer == 0:
                role = "Input"
            elif layer == len(sizes) - 1:
                role = "Output"
            else:
                role = "Hidden"
            layer_parts.append(f"{role} L{layer}: {size}")
        relu_total = sum(sizes[1:-1])
        fig.text(
            0.5,
            0.018,
            "Network  |  " + "  |  ".join(layer_parts) + f"  |  Total ReLU: {relu_total}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#374151",
        )
    fig.suptitle(title, fontsize=16, fontweight="bold")
    fig.subplots_adjust(
        left=0.09,
        right=0.94,
        top=0.92,
        bottom=0.12 if model_layer_sizes else 0.09,
    )
    return fig, {"history": history_ax, "neurons": neuron_ax}


def draw_active_split_dots(path: str | Path, *, ax=None, title: Optional[str] = None):
    """이전 API 호환용 별칭. 현재는 시간 이력 선 그래프를 그린다."""
    return draw_realtime_split_history(path, ax=ax, title=title)


class RealtimeSplitVisualizer:
    """logger의 매 append callback으로 PNG와 live view를 갱신한다.

    ``live_view=True``면 PNG와 같은 이름의 HTML을 만들고, 브라우저가
    PNG를 주기적으로 다시 읽게 한다. solver thread에서 GUI event loop를
    돌리지 않으므로 검증 진행을 막지 않는다.
    """

    def __init__(
        self,
        log_path: str | Path,
        output_path: str | Path,
        *,
        live_view: bool = False,
        open_live_view: bool = False,
        refresh_interval_ms: int = 500,
        playback_interval_ms: int = 0,
        window_seconds: float = 300.0,
        model_layer_sizes: Optional[Sequence[int]] = None,
    ) -> None:
        if refresh_interval_ms <= 0:
            raise ValueError("refresh_interval_ms는 양수여야 합니다")
        if playback_interval_ms < 0:
            raise ValueError("playback_interval_ms는 음수일 수 없습니다")
        if window_seconds <= 0:
            raise ValueError("window_seconds는 양수여야 합니다")
        self.log_path = Path(log_path)
        self.output_path = Path(output_path)
        self.live_view = live_view or open_live_view
        self.open_live_view = open_live_view
        self.refresh_interval_ms = refresh_interval_ms
        self.playback_interval_ms = playback_interval_ms
        self.window_seconds = float(window_seconds)
        self.model_layer_sizes = (
            [int(size) for size in model_layer_sizes]
            if model_layer_sizes is not None
            else None
        )
        self.live_view_path = self.output_path.with_suffix(".html")
        self._view_opened = False
        self._state_lock = threading.Lock()
        self._pending = False
        self._worker: Optional[threading.Thread] = None
        self._rendered_event_count = 0
        self._initial_frame_rendered = False

    def _write_live_view(self) -> None:
        image_name = json.dumps(self.output_path.name)
        html = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Realtime ReLU splits</title>
  <style>
    body {{ margin: 0; background: #111827; color: #f9fafb; font-family: sans-serif; }}
    main {{ max-width: 1100px; margin: 0 auto; padding: 20px; }}
    h1 {{ margin: 0 0 6px; font-size: 22px; }}
    p {{ margin: 0 0 14px; color: #9ca3af; }}
    img {{ display: block; width: 100%; background: white; border-radius: 8px; }}
  </style>
</head>
<body>
  <main>
    <h1>Realtime ReLU splits</h1>
    <p>이미지 약 {REALTIME_RENDER_INTERVAL_SECONDS:g}초 주기 생성 · 브라우저 {self.refresh_interval_ms}ms 간격 확인</p>
    <img id="split-view" alt="Realtime ReLU split visualization">
  </main>
  <script>
    const imageName = {image_name};
    const image = document.getElementById("split-view");
    function refresh() {{
      image.src = imageName + "?t=" + Date.now();
    }}
    refresh();
    setInterval(refresh, {self.refresh_interval_ms});
  </script>
</body>
</html>
"""
        self.live_view_path.parent.mkdir(parents=True, exist_ok=True)
        self.live_view_path.write_text(html, encoding="utf-8")

    @serialized_render
    def _render_once(
        self,
        events: Optional[List[SplitLogEvent]] = None,
        *,
        current_time_ns: Optional[int] = None,
    ) -> None:
        import matplotlib.pyplot as plt

        layer_sizes = self.model_layer_sizes
        if layer_sizes is None:
            metadata = read_split_metadata(self.log_path)
            raw_sizes = metadata.get("model_layer_sizes")
            if isinstance(raw_sizes, list):
                layer_sizes = [int(size) for size in raw_sizes]
        fig, _ = draw_realtime_split_dashboard(
            self.log_path,
            events=events,
            playback=self.playback_interval_ms > 0,
            window_seconds=self.window_seconds,
            current_time_ns=current_time_ns,
            model_layer_sizes=layer_sizes,
        )
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_output = self.output_path.with_name(
            f"{self.output_path.stem}.tmp{self.output_path.suffix}"
        )
        try:
            # 실시간 화면은 정적 보고서보다 새 프레임 도착 속도가 중요하다.
            # 낮은 DPI와 고정 canvas를 사용해 tight-bbox 재계산 비용도 피한다.
            fig.savefig(temporary_output, dpi=100)
            os.replace(temporary_output, self.output_path)
        finally:
            plt.close(fig)
            if temporary_output.exists():
                temporary_output.unlink()

        if self.live_view:
            if not self.live_view_path.exists():
                self._write_live_view()
            if self.open_live_view and not self._view_opened:
                self._view_opened = True
                webbrowser.open(self.live_view_path.resolve().as_uri())

    def _run_live_updates(self) -> None:
        while True:
            started = time_ns()
            with self._state_lock:
                self._pending = False
            submit_relu_render(
                self.log_path,
                self.output_path,
                self.window_seconds,
                self.model_layer_sizes,
            ).result()
            with self._state_lock:
                if not self._pending:
                    self._worker = None
                    return
            # At most one new frame every three seconds. Rendering time counts
            # toward the interval rather than being added on top of it.
            elapsed = (time_ns() - started) / 1_000_000_000
            sleep(max(0.0, REALTIME_RENDER_INTERVAL_SECONDS - elapsed))

    def _run_playback_updates(self) -> None:
        """로그 이벤트를 하나씩 재생하되 solver thread는 지연시키지 않는다."""
        while True:
            events = read_split_events(self.log_path)
            if self._rendered_event_count < len(events):
                with self._state_lock:
                    self._pending = False
                self._rendered_event_count += 1
                self._initial_frame_rendered = True
                frame_events = events[: self._rendered_event_count]
                self._render_once(
                    frame_events,
                    current_time_ns=frame_events[-1].time_ns,
                )
                sleep(self.playback_interval_ms / 1000.0)
                continue

            if not self._initial_frame_rendered:
                with self._state_lock:
                    self._pending = False
                self._initial_frame_rendered = True
                self._render_once([])

            with self._state_lock:
                if self._pending:
                    self._pending = False
                    continue
                self._worker = None
                return

    def _run_updates(self) -> None:
        try:
            if self.playback_interval_ms > 0:
                self._run_playback_updates()
            else:
                self._run_live_updates()
        except Exception as exc:
            warnings.warn(f"실시간 split background render 실패: {exc}")
            with self._state_lock:
                self._worker = None

    def request_update(self) -> None:
        """렌더 요청을 비동기로 합쳐 solver가 PNG 저장을 기다리지 않게 한다."""
        with self._state_lock:
            self._pending = True
            if self._worker is not None:
                return
            self._worker = threading.Thread(
                target=self._run_updates,
                name="relu-split-renderer",
                daemon=True,
            )
            self._worker.start()

    def flush(self) -> None:
        """대기 중인 최종 실시간 화면 렌더가 끝날 때까지 기다린다."""
        while True:
            with self._state_lock:
                worker = self._worker
            if worker is None:
                return
            worker.join()


def replay_split_log(
    log_path: str | Path,
    output_path: str | Path,
    *,
    interval_ms: int = 250,
    window_seconds: float = 300.0,
    open_live_view: bool = False,
) -> Tuple[Path, Path]:
    """저장된 전체 split JSONL을 첫 이벤트부터 순서대로 다시 그린다."""
    if interval_ms <= 0:
        raise ValueError("interval_ms는 양수여야 합니다")
    visualizer = RealtimeSplitVisualizer(
        log_path,
        output_path,
        live_view=True,
        open_live_view=open_live_view,
        playback_interval_ms=interval_ms,
        window_seconds=window_seconds,
    )
    visualizer.request_update()
    visualizer.flush()
    return visualizer.output_path, visualizer.live_view_path
