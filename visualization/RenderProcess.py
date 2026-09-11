"""Shared low-priority process for realtime Matplotlib rendering."""
from __future__ import annotations

from concurrent.futures import Future, ProcessPoolExecutor
from pathlib import Path
import os
import threading
import uuid


_executor = None
_executor_lock = threading.Lock()
REALTIME_RENDER_INTERVAL_SECONDS = 3.0


def _pool() -> ProcessPoolExecutor:
    global _executor
    with _executor_lock:
        if _executor is None:
            _executor = ProcessPoolExecutor(max_workers=1)
        return _executor


def _lower_priority() -> None:
    if os.name != "nt":
        return
    try:
        import ctypes
        # Keep verification responsive when the renderer and solver need the
        # same CPU. Failure is harmless on restricted Windows installations.
        ctypes.windll.kernel32.SetPriorityClass(
            ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000
        )
    except (AttributeError, OSError):
        pass


def _replace_figure(fig, output_path: str, dpi: int) -> None:
    import matplotlib.pyplot as plt

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(
        f"{output.stem}.{uuid.uuid4().hex}.tmp{output.suffix}"
    )
    try:
        fig.savefig(temporary, dpi=dpi)
        os.replace(temporary, output)
    finally:
        plt.close(fig)
        if temporary.exists():
            temporary.unlink()


def _render_solver(log_path: str, panel_paths: dict[str, str], panels: tuple[str, ...]) -> None:
    _lower_priority()
    from visualization.SolverProgress import (
        build_solver_feedback,
        draw_dpll_theory_panel,
        draw_simplex_panel,
        read_solver_progress,
    )

    feedback = build_solver_feedback(read_solver_progress(log_path))
    renderers = {
        "dpll_theory": draw_dpll_theory_panel,
        "simplex": draw_simplex_panel,
    }
    for panel in panels:
        _replace_figure(renderers[panel](feedback)[0], panel_paths[panel], 100)


def _render_relu(
    log_path: str,
    output_path: str,
    window_seconds: float,
    model_layer_sizes,
) -> None:
    _lower_priority()
    from visualization.RealtimeSplitVisualization import (
        draw_realtime_split_dashboard,
        read_split_metadata,
    )

    layer_sizes = model_layer_sizes
    if layer_sizes is None:
        raw_sizes = read_split_metadata(log_path).get("model_layer_sizes")
        if isinstance(raw_sizes, list):
            layer_sizes = [int(size) for size in raw_sizes]
    fig, _ = draw_realtime_split_dashboard(
        log_path,
        playback=False,
        window_seconds=window_seconds,
        model_layer_sizes=layer_sizes,
    )
    _replace_figure(fig, output_path, 100)


def submit_solver_render(
    log_path: Path, panel_paths: dict[str, Path], panels
) -> Future:
    return _pool().submit(
        _render_solver,
        str(log_path),
        {name: str(path) for name, path in panel_paths.items()},
        tuple(panels),
    )


def submit_relu_render(
    log_path: Path,
    output_path: Path,
    window_seconds: float,
    model_layer_sizes,
) -> Future:
    return _pool().submit(
        _render_relu,
        str(log_path),
        str(output_path),
        window_seconds,
        model_layer_sizes,
    )
