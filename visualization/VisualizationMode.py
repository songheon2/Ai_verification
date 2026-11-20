"""실시간/사후 피드백 visualization 조합과 기본 출력 경로."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import json
from pathlib import Path
import re
from typing import Any, Mapping

VERIFICATION_PANELS = ("dpll-theory", "simplex", "relu", "mechanisms")
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs"
RUN_RELOCATIONS = ".run-relocations.json"
REALTIME_DIRECTORY = "realtime"
FEEDBACK_DIRECTORY = "feedback"
LOG_DIRECTORY = "logs"
INPUT_DIRECTORY = "inputs"
ARTIFACT_DIRECTORIES = (REALTIME_DIRECTORY, FEEDBACK_DIRECTORY, LOG_DIRECTORY, INPUT_DIRECTORY)


def _organized_artifact(path: Path, root: Path) -> Path:
    if path.exists() or not path.is_relative_to(root):
        return path
    matches = [(path.parent / group / path.name).resolve() for group in ARTIFACT_DIRECTORIES]
    existing = [candidate for candidate in matches if candidate.is_relative_to(root) and candidate.is_file()]
    return existing[0] if len(existing) == 1 else path


def resolve_visualization_output(path: str | Path, output_root: str | Path = DEFAULT_OUTPUT_ROOT) -> Path:
    """Follow an exact recorded move for a legacy run directory or artifact."""
    target = Path(path).resolve()
    root = Path(output_root).resolve()
    if target.exists() or not target.is_relative_to(root):
        return target
    mapping = root / RUN_RELOCATIONS
    if mapping.resolve().parent != root:
        return target
    try:
        with mapping.open(encoding="utf-8") as stream:
            moves = json.loads(stream.read(2 * 1024 * 1024))
    except (OSError, ValueError):
        return _organized_artifact(target, root)
    if not isinstance(moves, dict):
        return _organized_artifact(target, root)
    for parent in (target, *target.parents):
        if parent == root:
            break
        destination = moves.get(parent.relative_to(root).as_posix())
        if isinstance(destination, str):
            candidate = (root / destination / target.relative_to(parent)).resolve()
            if candidate.is_relative_to(root) and candidate.exists():
                return candidate
            moved = _organized_artifact(candidate, root)
            if moved != candidate:
                return moved
    return _organized_artifact(target, root)


class VisualizationMode(str, Enum):
    OFF = "off"
    REALTIME = "realtime"
    FEEDBACK = "feedback"
    BOTH = "both"

    @classmethod
    def parse(cls, value: str | "VisualizationMode") -> "VisualizationMode":
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).lower())
        except ValueError as exc:
            choices = ", ".join(mode.value for mode in cls)
            raise ValueError(f"visualization_mode는 {choices} 중 하나여야 합니다") from exc

    @property
    def realtime_enabled(self) -> bool:
        return self in (self.REALTIME, self.BOTH)

    @property
    def feedback_enabled(self) -> bool:
        return self in (self.FEEDBACK, self.BOTH)


@dataclass(frozen=True)
class VisualizationPaths:
    realtime_log: Path
    realtime_image: Path
    feedback_heatmap: Path
    feedback_history: Path
    feedback_search_tree: Path
    solver_progress_log: Path
    solver_realtime_image: Path
    solver_feedback_image: Path


def _artifact_stem(path: str | Path) -> str:
    name = Path(path).name
    for suffix in (".gz", ".vnnlib", ".onnx", ".custom", ".bin", ".txt"):
        if name.lower().endswith(suffix):
            name = name[: -len(suffix)]
    return name or "visualization"


def _safe_folder_component(value: str, *, max_length: int = 96) -> str:
    value = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", value)
    value = re.sub(r"\s+", " ", value).strip(" .")
    return (value or "network")[:max_length].rstrip(" .")


def property_folder_name(property_path: str | Path | None, model_path: str | Path) -> str:
    if property_path is None:
        return "property-unknown"
    label = _artifact_stem(property_path)
    numbered = re.fullmatch(r"prop[_\-\s]*(\d+)", label, re.IGNORECASE)
    if numbered:
        return "prop_" + (numbered[1].lstrip("0") or "0")
    if label.casefold() == _artifact_stem(model_path).casefold():
        return "default"
    return _safe_folder_component(label)


def benchmark_folder_name(model_path: str | Path, property_path: str | Path | None = None) -> str:
    """Use the benchmark collection before falling back to well-known model names."""
    for source in (model_path, property_path):
        if source is None:
            continue
        parts = re.split(r"[/\\]+", str(source))
        for index, part in enumerate(parts[:-2]):
            if part.casefold() == "benchmarks":
                return _safe_folder_component(parts[index + 1])
    stem = _artifact_stem(model_path).casefold()
    if stem.startswith("acasxu_"):
        return "acasxu_2023"
    if re.fullmatch(r"(?:un)?sat_v\d+_c\d+", stem):
        return "sat_relu"
    return "local"


def solver_variant_name(options: Mapping[str, Any] | None = None) -> str:
    """Effective LP implementation, including implicit engine mode defaults."""
    options = options or {}
    engine = options.get("lp_engine", "native")
    if engine == "native":
        return "simplex-original"
    mode = options.get("lp_mode") or ("oracle" if engine == "simplex" else "core")
    name = f"{engine}-{mode}"
    if engine == "glpk" and mode == "core":
        extract = options.get("glpk_extract", "full")
        if extract != "full":
            name += f"-{extract}"
        row_error = options.get("glpk_row_error", "auto")
        if row_error != "auto":
            name += f"-row-error-{row_error}"
    return _safe_folder_component(name)


def create_visualization_run_directory(
    model_path: str | Path,
    *,
    property_path: str | Path | None = None,
    started_at: datetime,
    output_root: str | Path | None = None,
    stable_path: bool = False,
    solver_options: Mapping[str, Any] | None = None,
) -> Path:
    """Create ``outputs/<benchmark>/<property>/<model>/<solver variant>/<timestamp run>``."""
    root = (
        Path(output_root)
        if output_root is not None
        else DEFAULT_OUTPUT_ROOT
    )
    root.mkdir(parents=True, exist_ok=True)
    local_started_at = started_at.astimezone()
    start_text = local_started_at.strftime("%Y%m%d_%H%M%S")
    model_name = _safe_folder_component(_artifact_stem(model_path))
    model_directory = (root / benchmark_folder_name(model_path, property_path)
                       / property_folder_name(property_path, model_path)
                       / model_name / solver_variant_name(solver_options))
    model_directory.mkdir(parents=True, exist_ok=True)
    base_name = start_text if stable_path else f"{start_text}~running"
    suffix = 1
    while True:
        run_directory = model_directory / (base_name if suffix == 1 else f"{base_name}_{suffix}")
        try:
            run_directory.mkdir()
            return run_directory
        except FileExistsError:
            suffix += 1


def finalize_visualization_run_directory(
    staging_directory: str | Path,
    model_path: str | Path,
    *,
    started_at: datetime,
    finished_at: datetime,
) -> Path:
    """실행 폴더를 같은 속성·엔진 폴더 안에서 시작~종료 시간 이름으로 확정한다."""
    staging = Path(staging_directory)
    start_text = started_at.astimezone().strftime("%Y%m%d_%H%M%S")
    finish_text = finished_at.astimezone().strftime("%Y%m%d_%H%M%S")
    base_name = f"{start_text}~{finish_text}"
    destination = staging.parent / base_name
    suffix = 2
    while destination.exists():
        destination = staging.parent / f"{base_name}_{suffix}"
        suffix += 1
    staging.rename(destination)
    return destination


def default_visualization_paths(
    model_path: str | Path,
    property_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    organized: bool = False,
    realtime: bool = False,
) -> VisualizationPaths:
    resolved_output_dir = (
        Path(output_dir)
        if output_dir is not None
        else DEFAULT_OUTPUT_ROOT
    )
    if output_dir is not None:
        live = resolved_output_dir / REALTIME_DIRECTORY if organized else resolved_output_dir
        feedback = resolved_output_dir / FEEDBACK_DIRECTORY if organized else resolved_output_dir
        logs = resolved_output_dir / LOG_DIRECTORY if organized else resolved_output_dir
        return VisualizationPaths(
            realtime_log=logs / "relu_events.jsonl",
            realtime_image=live / "relu_realtime.png",
            feedback_heatmap=feedback / "relu_feedback_heatmap.png",
            feedback_history=feedback / "relu_feedback_history.png",
            feedback_search_tree=(live if realtime else feedback) / "relu_search_tree.html",
            solver_progress_log=logs / "solver_progress.jsonl",
            solver_realtime_image=live / "solver_realtime.png",
            solver_feedback_image=feedback / "solver_feedback.png",
        )
    stem = f"{_artifact_stem(model_path)}__{_artifact_stem(property_path)}"
    return VisualizationPaths(
        realtime_log=resolved_output_dir / f"{stem}_realtime.jsonl",
        realtime_image=resolved_output_dir / f"{stem}_realtime.png",
        feedback_heatmap=resolved_output_dir / f"{stem}_feedback.png",
        feedback_history=resolved_output_dir / f"{stem}_feedback_history.png",
        feedback_search_tree=resolved_output_dir / f"{stem}_search_tree.html",
        solver_progress_log=resolved_output_dir / f"{stem}_solver_progress.jsonl",
        solver_realtime_image=resolved_output_dir / f"{stem}_solver_realtime.png",
        solver_feedback_image=resolved_output_dir / f"{stem}_solver_feedback.png",
    )


def combined_mode(*, realtime: bool, feedback: bool) -> VisualizationMode:
    if realtime and feedback:
        return VisualizationMode.BOTH
    if realtime:
        return VisualizationMode.REALTIME
    if feedback:
        return VisualizationMode.FEEDBACK
    return VisualizationMode.OFF
