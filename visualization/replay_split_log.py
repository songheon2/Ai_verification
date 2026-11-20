"""저장된 ReLU split JSONL 전체를 실시간 그래프처럼 다시 재생한다."""
from __future__ import annotations

import argparse
from pathlib import Path

from visualization.RealtimeSplitVisualization import replay_split_log
from visualization.VisualizationMode import (
    DEFAULT_OUTPUT_ROOT,
    LOG_DIRECTORY,
    REALTIME_DIRECTORY,
    resolve_visualization_output,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", help="재생할 split JSONL 로그")
    parser.add_argument(
        "--output",
        help="갱신할 PNG 경로 (기본: 로그 옆 *_replay.png)",
    )
    parser.add_argument(
        "--interval-ms",
        type=int,
        default=250,
        help="이벤트 한 건당 재생 간격 ms (기본: 250)",
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=300.0,
        help="현재 기준으로 표시할 고정 시간 폭 (기본: 300초/5분)",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="자동 갱신 HTML을 기본 브라우저로 열기",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    log_path = resolve_visualization_output(Path(args.log))
    image_directory = log_path.parent
    if image_directory.name == LOG_DIRECTORY and log_path.resolve().is_relative_to(DEFAULT_OUTPUT_ROOT.resolve()):
        image_directory = image_directory.parent / REALTIME_DIRECTORY
    output_path = Path(args.output) if args.output else image_directory / f"{log_path.stem}_replay.png"
    image_path, live_view_path = replay_split_log(
        log_path,
        output_path,
        interval_ms=args.interval_ms,
        window_seconds=args.window_seconds,
        open_live_view=args.open,
    )
    print(f"Replay image: {image_path}")
    print(f"Live view   : {live_view_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
