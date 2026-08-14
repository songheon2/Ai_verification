"""DPLL round·Simplex·theory dashboard의 제어된 live demo."""
from __future__ import annotations

import argparse
from pathlib import Path
from time import sleep

from visualization.RealtimeSplitVisualization import (
    RealtimeSplitVisualizer,
    SplitEventLogger,
)
from visualization.SolverProgress import (
    SolverProgressLogger,
    SolverProgressPanelVisualizer,
    write_solver_feedback_panels,
)
from visualization.UnifiedRealtimeDashboard import UnifiedRealtimeDashboard


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def run_demo(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    interval_seconds: float = 0.2,
    open_live_view: bool = True,
) -> dict[str, Path]:
    if interval_seconds < 0:
        raise ValueError("interval_seconds는 음수일 수 없습니다")
    root = Path(output_dir)
    log_path = root / "solver_progress_demo.jsonl"
    realtime_base = root / "solver_progress_demo_realtime.png"
    feedback_path = root / "solver_progress_demo_feedback.png"
    split_log_path = root / "solver_progress_demo_split.jsonl"
    split_image_path = root / "solver_progress_demo_split.png"
    live_view_path = root / "solver_progress_demo_realtime.html"
    visualizer = SolverProgressPanelVisualizer(
        log_path,
        realtime_base,
    )
    split_visualizer = RealtimeSplitVisualizer(split_log_path, split_image_path)
    split_logger = SplitEventLogger(
        split_log_path,
        update_callback=split_visualizer.request_update,
        reset_log=True,
    )
    split_logger.request_update()
    UnifiedRealtimeDashboard(
        split_image_path,
        visualizer.panel_paths["dpll_theory"],
        visualizer.panel_paths["simplex"],
        live_view_path,
    ).write(open_browser=open_live_view)
    logger = SolverProgressLogger(
        log_path,
        update_callback=visualizer.request_update,
        reset_log=True,
        refresh_interval_seconds=0.3,
    )

    round_results = ["THEORY_CONFLICT", "THEORY_CONFLICT", "THEORY_SAT"]
    simplex_results = ["SAT", "UNSAT", "ITERATION_LIMIT", "SAT"]
    variables = ["ineq_slack_0", "z1_3", "ineq_slack_2", "h2_1"]
    call_number = 0
    for round_index, round_result in enumerate(round_results, start=1):
        logger.round_start(round_index)
        split_id = split_logger.begin(
            f"z{round_index}_0", round_index, 0
        )
        atoms = [
            {
                "atom_id": f"a{round_index * 2 - 1}",
                "polarity": True,
                "kind": "InequProp",
                "expression": f"demo inequality for round {round_index}",
            },
            {
                "atom_id": f"a{round_index * 2}",
                "polarity": round_index % 2 == 1,
                "kind": "ReLUProp",
                "expression": f"demo ReLU for round {round_index}",
            },
        ]
        logger.theory_selection(round_index, atoms)
        for local_call in range(1 + (round_index % 2)):
            result = simplex_results[call_number % len(simplex_results)]
            call_number += 1
            call_id = logger.simplex_start(
                round_index=round_index,
                depth=local_call,
                origin="reluplex" if local_call == 0 else "relu_repair",
                theory_atom_ids=[str(atom["atom_id"]) for atom in atoms],
                row_count=3 + round_index,
                variable_count=6 + round_index,
            )
            pivot_count = 0
            iteration_count = 3 + round_index + local_call
            for iteration in range(iteration_count):
                violated = variables[(iteration + round_index) % len(variables)]
                logger.simplex_iteration(
                    call_id,
                    iteration,
                    violated_var=violated,
                    value=-1.0 - iteration,
                    lower=0.0,
                    upper=10.0,
                )
                if iteration < iteration_count - 1:
                    entering = f"x{(iteration + local_call) % 3}"
                    logger.simplex_pivot(
                        call_id,
                        iteration,
                        entering=entering,
                        leaving=violated,
                        row=violated,
                    )
                    pivot_count += 1
                sleep(interval_seconds)
            logger.simplex_end(
                call_id,
                result,
                iterations=iteration_count,
                pivots=pivot_count,
            )
        logger.theory_result(round_index, round_result)
        logger.round_end(round_index, round_result)
        split_logger.end(split_id, f"z{round_index}_0", round_index, 0)

    logger.solver_end("SAT", "THEORY_SAT", len(round_results))
    visualizer.flush()
    _feedback, feedback_panels = write_solver_feedback_panels(
        log_path, feedback_path
    )
    return {
        "log": log_path,
        "dpll_theory_realtime": visualizer.panel_paths["dpll_theory"],
        "simplex_realtime": visualizer.panel_paths["simplex"],
        "relu_realtime": split_image_path,
        "live_view": live_view_path,
        "dpll_theory_feedback": feedback_panels["dpll_theory"],
        "simplex_feedback": feedback_panels["simplex"],
        "feedback_json": feedback_path.with_suffix(".json"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DPLL round, Simplex call/iteration/pivot, theory link live dashboard demo"
    )
    parser.add_argument("--interval", type=float, default=0.2, help="iteration 간격 초")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()
    outputs = run_demo(
        output_dir=args.output_dir,
        interval_seconds=args.interval,
        open_live_view=not args.no_open,
    )
    for name, path in outputs.items():
        print(f"{name:16s}: {path}")


if __name__ == "__main__":
    main()
