"""자동 갱신 ReLU split 화면을 눈으로 확인하는 제어된 데모.

실제 검증 결과를 만드는 solver 벤치마크가 아니라, SplitEventLogger에
여러 레이어의 +/- 이벤트를 일정한 간격으로 보내 live view의
활성 split 수 선 그래프와 누적 진행 수치가 갱신되는지 보여 준다.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from time import sleep

from visualization.RealtimeSplitVisualization import (
    RealtimeSplitVisualizer,
    SplitEventLogger,
)


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def run_demo(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    split_count: int = 6,
    interval_seconds: float = 0.8,
    hold_seconds: float = 2.0,
    open_live_view: bool = True,
) -> tuple[Path, Path, Path]:
    if split_count <= 0:
        raise ValueError("split_count는 양수여야 합니다")
    if interval_seconds < 0 or hold_seconds < 0:
        raise ValueError("시간 간격은 음수일 수 없습니다")

    root = Path(output_dir)
    log_path = root / "realtime_split_demo.jsonl"
    image_path = root / "realtime_split_demo.png"
    visualizer = RealtimeSplitVisualizer(
        log_path,
        image_path,
        live_view=True,
        open_live_view=open_live_view,
    )
    logger = SplitEventLogger(
        log_path,
        update_callback=visualizer.request_update,
        reset_log=True,
    )
    logger.request_update()

    opened = []
    for index in range(split_count):
        layer = 1 + index // 3
        neuron = index % 3
        split_id = logger.begin(f"z{layer}_{neuron}", layer, neuron)
        opened.append((split_id, layer, neuron))
        sleep(interval_seconds)

    sleep(hold_seconds)

    for split_id, layer, neuron in reversed(opened):
        logger.end(split_id, f"z{layer}_{neuron}", layer, neuron)
        sleep(interval_seconds)

    visualizer.flush()
    return log_path, image_path, visualizer.live_view_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="활성 ReLU split 선이 올라갔다 내려가는 live view 데모"
    )
    parser.add_argument("--count", type=int, default=6, help="최대 활성 split 수 (기본 6)")
    parser.add_argument(
        "--interval", type=float, default=0.8, help="이벤트 간격 초 (기본 0.8)"
    )
    parser.add_argument(
        "--hold-seconds", type=float, default=2.0, help="모든 점을 유지할 초 (기본 2)"
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--no-open", action="store_true", help="브라우저를 자동으로 열지 않음")
    args = parser.parse_args()

    log_path, image_path, live_path = run_demo(
        output_dir=args.output_dir,
        split_count=args.count,
        interval_seconds=args.interval,
        hold_seconds=args.hold_seconds,
        open_live_view=not args.no_open,
    )
    print(f"Realtime log : {log_path}")
    print(f"Realtime PNG : {image_path}")
    print(f"Live view    : {live_path}")


if __name__ == "__main__":
    main()
