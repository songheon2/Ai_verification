"""실시간/사후 피드백 visualization 조합과 기본 출력 경로."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path


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
    solver_progress_log: Path
    solver_realtime_image: Path
    solver_feedback_image: Path


def _artifact_stem(path: str | Path) -> str:
    name = Path(path).name
    for suffix in (".gz", ".vnnlib", ".onnx", ".custom", ".bin", ".txt"):
        if name.lower().endswith(suffix):
            name = name[: -len(suffix)]
    return name or "visualization"


def default_visualization_paths(
    model_path: str | Path,
    property_path: str | Path,
) -> VisualizationPaths:
    output_dir = Path(__file__).resolve().parent / "outputs"
    stem = f"{_artifact_stem(model_path)}__{_artifact_stem(property_path)}"
    return VisualizationPaths(
        realtime_log=output_dir / f"{stem}_realtime.jsonl",
        realtime_image=output_dir / f"{stem}_realtime.png",
        feedback_heatmap=output_dir / f"{stem}_feedback.png",
        solver_progress_log=output_dir / f"{stem}_solver_progress.jsonl",
        solver_realtime_image=output_dir / f"{stem}_solver_realtime.png",
        solver_feedback_image=output_dir / f"{stem}_solver_feedback.png",
    )


def combined_mode(*, realtime: bool, feedback: bool) -> VisualizationMode:
    if realtime and feedback:
        return VisualizationMode.BOTH
    if realtime:
        return VisualizationMode.REALTIME
    if feedback:
        return VisualizationMode.FEEDBACK
    return VisualizationMode.OFF
