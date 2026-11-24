"""Shared low-priority process for realtime Matplotlib rendering."""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import multiprocessing
from pathlib import Path
import os
import threading
import uuid


_executor = None
_executor_lock = threading.Lock()
REALTIME_RENDER_INTERVAL_SECONDS = 3.0
RENDER_TIMEOUT_SECONDS = 30.0
_renderer = None


def _pool() -> ThreadPoolExecutor:
    global _executor
    with _executor_lock:
        if _executor is None:
            _executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="realtime-render-ipc")
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
    from visualization.AtomicFile import replace_atomic_file

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(
        f"{output.stem}.{uuid.uuid4().hex}.tmp{output.suffix}"
    )
    try:
        fig.savefig(temporary, dpi=dpi, pil_kwargs={"compress_level": 2})
        from visualization.ReluNeuronHover import write_relu_hover_map
        write_relu_hover_map(fig, output)
        from visualization.AnalysisSelection import write_selection_map
        write_selection_map(fig, output)
        replace_atomic_file(temporary, output)
    finally:
        plt.close(fig)
        if temporary.exists():
            temporary.unlink()


def _render_solver(reader, panel_paths: dict[str, str], panels: tuple[str, ...]) -> None:
    _lower_priority()
    from visualization.SolverProgress import (
        draw_dpll_theory_panel,
        draw_simplex_panel,
    )

    feedback = reader.read()
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
    *,
    split_log=None,
) -> None:
    _lower_priority()
    from visualization.RealtimeSplitVisualization import (
        draw_realtime_split_dashboard,
        read_split_metadata,
    )

    metadata = read_split_metadata(log_path)
    layer_sizes = model_layer_sizes
    if layer_sizes is None:
        raw_sizes = metadata.get("model_layer_sizes")
        if isinstance(raw_sizes, list):
            layer_sizes = [int(size) for size in raw_sizes]
    if split_log is not None:
        split_log.read()
    fig, _ = draw_realtime_split_dashboard(
        log_path,
        playback=False,
        window_seconds=window_seconds,
        model_layer_sizes=layer_sizes,
        round_starts=metadata.get("round_starts", []),
        split_log=split_log,
    )
    _replace_figure(fig, output_path, 100)


def _serve_renderer(connection) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        _lower_priority()
        solver_reader = None
        split_log = None
        tree_writers = {}
        while True:
            action, args = connection.recv()
            try:
                if action == "tree":
                    config, status, reason, stats, version = args
                    if config not in tree_writers:
                        from visualization.ReluSearchTree import TreeSnapshotWriter
                        tree_writers[config] = TreeSnapshotWriter(*config)
                    tree_writers[config].write(status, reason, stats, version)
                    if status != "RUNNING":
                        del tree_writers[config]
                elif action == "solver":
                    log_path, panel_paths, panels = args
                    if solver_reader is None or solver_reader.path != Path(log_path):
                        from visualization.SolverProgress import SolverProgressReader
                        solver_reader = SolverProgressReader(log_path)
                    _render_solver(solver_reader, panel_paths, panels)
                elif action == "relu":
                    log_path = Path(args[0])
                    if split_log is None or split_log.path != log_path:
                        from visualization.RealtimeSplitVisualization import RealtimeSplitLog
                        split_log = RealtimeSplitLog(log_path)
                    _render_relu(*args, split_log=split_log)
                else:
                    raise ValueError(f"Unknown render action {action!r}")
            except Exception as exc:
                connection.send(f"{type(exc).__name__}: {exc}")
            else:
                connection.send(None)
    except (EOFError, BrokenPipeError):
        pass
    finally:
        connection.close()


def _stop_renderer() -> None:
    global _renderer
    if _renderer is None:
        return
    process, connection = _renderer
    _renderer = None
    connection.close()
    process.join(timeout=.2)
    if process.is_alive():
        process.terminate()
        process.join(timeout=2)
    if process.is_alive():
        process.kill()
        process.join(timeout=2)
    if not process.is_alive():
        process.close()


def _execute_render(action: str, *args) -> None:
    """Run on the serial IPC thread and replace an unresponsive child."""
    global _renderer
    if _renderer is None:
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe()
        process = context.Process(target=_serve_renderer, args=(child,),
                                  name="visualization-renderer", daemon=True)
        try:
            process.start()
        except BaseException:
            parent.close()
            raise
        finally:
            child.close()
        _renderer = process, parent
    _process, connection = _renderer
    try:
        connection.send((action, args))
        if not connection.poll(RENDER_TIMEOUT_SECONDS):
            raise TimeoutError(f"{action} render did not respond within {RENDER_TIMEOUT_SECONDS:g}s")
        error = connection.recv()
    except (OSError, EOFError, TimeoutError):
        _stop_renderer()
        raise
    if error is not None:
        raise RuntimeError(error)


def submit_solver_render(
    log_path: Path, panel_paths: dict[str, Path], panels
) -> Future:
    return _pool().submit(
        _execute_render, "solver",
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
        _execute_render, "relu",
        str(log_path),
        str(output_path),
        window_seconds,
        model_layer_sizes,
    )


def submit_tree_render(config, status, reason, stats, version) -> Future:
    return _pool().submit(_execute_render, "tree", config, status, reason, stats, version)
