"""실시간/사후 피드백 visualization 조합과 기본 출력 경로."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
import re


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


def create_visualization_run_directory(
    model_path: str | Path,
    *,
    started_at: datetime,
    output_root: str | Path | None = None,
    stable_path: bool = False,
) -> Path:
    """Create a run folder; browser sessions use a stable, status-neutral URL."""
    root = (
        Path(output_root)
        if output_root is not None
        else Path(__file__).resolve().parent / "outputs"
    )
    root.mkdir(parents=True, exist_ok=True)
    start_text = started_at.astimezone().strftime("%Y%m%d_%H%M%S")
    model_name = _safe_folder_component(_artifact_stem(model_path))
    base_name = start_text if stable_path else f"{start_text}~running"
    group_directory = root / base_name
    suffix = 2
    while group_directory.exists():
        group_directory = root / f"{base_name}_{suffix}"
        suffix += 1
    group_directory.mkdir()
    run_directory = group_directory / model_name
    run_directory.mkdir()
    return run_directory


def finalize_visualization_run_directory(
    staging_directory: str | Path,
    model_path: str | Path,
    *,
    started_at: datetime,
    finished_at: datetime,
) -> Path:
    """상위 시간 폴더를 ``시작시간~종료시간/모델/``으로 확정한다."""
    staging = Path(staging_directory)
    staging_group = staging.parent
    start_text = started_at.astimezone().strftime("%Y%m%d_%H%M%S")
    finish_text = finished_at.astimezone().strftime("%Y%m%d_%H%M%S")
    base_name = f"{start_text}~{finish_text}"
    destination_group = staging_group.parent / base_name
    suffix = 2
    while destination_group.exists():
        destination_group = staging_group.parent / f"{base_name}_{suffix}"
        suffix += 1
    staging_group.rename(destination_group)
    return destination_group / staging.name


def default_visualization_paths(
    model_path: str | Path,
    property_path: str | Path,
    *,
    output_dir: str | Path | None = None,
) -> VisualizationPaths:
    resolved_output_dir = (
        Path(output_dir)
        if output_dir is not None
        else Path(__file__).resolve().parent / "outputs"
    )
    if output_dir is not None:
        return VisualizationPaths(
            realtime_log=resolved_output_dir / "relu_events.jsonl",
            realtime_image=resolved_output_dir / "relu_realtime.png",
            feedback_heatmap=resolved_output_dir / "relu_feedback_heatmap.png",
            feedback_history=resolved_output_dir / "relu_feedback_history.png",
            solver_progress_log=resolved_output_dir / "solver_progress.jsonl",
            solver_realtime_image=resolved_output_dir / "solver_realtime.png",
            solver_feedback_image=resolved_output_dir / "solver_feedback.png",
        )
    stem = f"{_artifact_stem(model_path)}__{_artifact_stem(property_path)}"
    return VisualizationPaths(
        realtime_log=resolved_output_dir / f"{stem}_realtime.jsonl",
        realtime_image=resolved_output_dir / f"{stem}_realtime.png",
        feedback_heatmap=resolved_output_dir / f"{stem}_feedback.png",
        feedback_history=resolved_output_dir / f"{stem}_feedback_history.png",
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
