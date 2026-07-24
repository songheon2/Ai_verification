"""
SolveTrace(Automation/SolveTrace.py)의 reluplex_split 이벤트를 모아 (layer,
neuron)별 누적 split 횟수를 계산하고, NetworkLayout.draw_network의 threshold
모드로 "어느 뉴런이 자주 split됐는지" 히트맵을 그리는 도구.

여러 검증 인스턴스(idx, eps 등)를 하나의 SolveTrace에 누적해서 넘기면, 그
전체에 대한 합산 히트맵이 나온다 — 인스턴스 하나짜리 trace를 넘기면 그
인스턴스만의 히트맵이 된다.

"현재 진행 중인 split 스택" 애니메이션(원 계획의 실시간 스냅샷 GIF)은 이
모듈에 포함하지 않았다 — 여기서는 정적인 누적 히트맵까지만 다룬다.

다른 모듈에서 쓰는 법 (주 용도)
------------------------------
    from Automation.SolveTrace import SolveTrace
    from DPLL_T import dpll_t_detailed
    from SplitHeatmap import split_counts_from_trace, draw_split_heatmap

    trace = SolveTrace()
    dpll_t_detailed(formula, trace=trace)          # 여러 인스턴스를 같은 trace에 누적 가능
    counts = split_counts_from_trace(trace)         # {(layer, neuron): 누적 split 횟수}
    fig, ax = draw_split_heatmap(model, counts, threshold=1)
    fig.savefig("split_heatmap.png")

단독 실행 (데모/스모크 테스트용)
--------------------------------
    python SplitHeatmap.py [--model model.bin] [--output output.png]
        [--threshold N] [--cap N]

    --model, -m    : 커스텀 바이너리 신경망(.bin) 경로 (생략 시 xor_network.bin)
    --output, -o   : 출력 이미지 경로 (생략 시 <model파일명>_split_heatmap.png)
    --threshold, -t: 이 횟수 이상 split된 뉴런만 표시 (생략 시 1, 즉 한 번이라도 split되면 표시)
    --cap          : 레이어당 최대 표시 뉴런 수 (생략 시 20)

    xor_network.bin에 대해 (case, eps) 여러 조합을 순차로 실제 검증하면서
    같은 SolveTrace에 이벤트를 누적한 뒤 히트맵을 그린다 (개별 인스턴스 하나로는
    split이 거의 안 일어나서 표시할 게 없기 때문 — 여러 인스턴스를 모아야
    "어디가 자주 split되는지"가 의미 있게 나온다).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from Automation.SolveTrace import SolveTrace
from GenericNNEncoding import NNModel
from NetworkLayout import draw_network, parse_neuron_var


def split_counts_from_trace(trace: SolveTrace) -> Dict[Tuple[int, int], int]:
    """trace.events 중 component=="reluplex_split"인 것만 골라, branch_x를
    (layer, neuron)으로 파싱해서 누적 횟수를 센다. branch_x가 GenericNNEncoding.py
    명명 규칙과 안 맞는 이벤트(파싱 실패)는 조용히 건너뛴다."""
    counts: Dict[Tuple[int, int], int] = {}
    for ev in trace.events:
        if ev.component != "reluplex_split" or ev.branch_x is None:
            continue
        key = parse_neuron_var(ev.branch_x)
        if key is None:
            continue
        counts[key] = counts.get(key, 0) + 1
    return counts


def draw_split_heatmap(
    model: NNModel,
    split_counts: Dict[Tuple[int, int], int],
    *,
    threshold: float = 1,
    cap: int = 20,
    title: Optional[str] = None,
    **draw_network_kwargs,
):
    """split_counts_from_trace()가 만든 (layer, neuron) -> 횟수 딕셔너리를
    NetworkLayout.draw_network의 threshold 모드로 그린다. threshold 미만인
    뉴런은 숨겨지고 숨김 구간마다 "⋮"로 표시된다 (자세한 규칙은
    NetworkLayout.compute_layout 참고)."""
    return draw_network(
        model, split_counts,
        select_mode="threshold", threshold=threshold, cap=cap,
        title=title, cmap_name="Reds",
        **draw_network_kwargs,
    )


def _run_demo_sweep(model_path: str, trace: SolveTrace) -> None:
    """xor_network.bin 전용 데모: (case, eps) 여러 조합을 순차 검증해서 하나의
    trace에 split 이벤트를 누적한다."""
    from DPLL import AndProp, NotProp
    from DPLL_T import dpll_t_detailed
    from GenericNNEncoding import load_nn_model, encode_nn
    from PreciseEncoding import out_zero_logit, out_one_logit
    from Robustness import make_precondition_linf_box
    from XOREncoding import FreshGen

    model = load_nn_model(model_path)
    xor_cases = [((0.0, 0.0), 0), ((0.0, 1.0), 1), ((1.0, 0.0), 1), ((1.0, 1.0), 0)]
    eps_values = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]

    n_run = 0
    for eps in eps_values:
        for (c1, c2), expected in xor_cases:
            fg = FreshGen(prefix=f"splitdemo_eps{int(eps * 1000)}_c{c1:.0f}{c2:.0f}_")
            x_vars = ["x0", "x1"]
            pre = make_precondition_linf_box(x_vars, (c1, c2), eps=eps, clamp_01=False)
            nn_prop, out_vars, _ = encode_nn(model, x_vars, gen=fg)
            s_var = out_vars[0]
            post = out_zero_logit(s_var) if expected == 0 else out_one_logit(s_var)
            phi = AndProp(pre, AndProp(nn_prop, NotProp(post)))

            dpll_t_detailed(phi, trace=trace, timeout_seconds=10)
            n_run += 1

    print(f"데모 sweep 완료: {n_run}개 인스턴스 실행")


def main() -> None:
    import argparse
    import os

    from GenericNNEncoding import load_nn_model

    default_model = os.path.join(os.path.dirname(os.path.abspath(__file__)), "xor_network.bin")

    parser = argparse.ArgumentParser(description="SolveTrace 누적 split 히트맵 데모/스모크 테스트")
    parser.add_argument("--model", "-m", default=default_model, help="커스텀 바이너리 신경망(.bin) 경로")
    parser.add_argument("--output", "-o", default=None, help="출력 이미지 경로 (생략 시 <model파일명>_split_heatmap.png)")
    parser.add_argument("--threshold", "-t", type=float, default=1, help="이 횟수 이상 split된 뉴런만 표시")
    parser.add_argument("--cap", type=int, default=20, help="레이어당 최대 표시 뉴런 수")
    args = parser.parse_args()

    output_path = args.output or (os.path.splitext(args.model)[0] + "_split_heatmap.png")

    trace = SolveTrace()
    _run_demo_sweep(args.model, trace)

    counts = split_counts_from_trace(trace)
    print(f"split 이벤트: {sum(1 for e in trace.events if e.component == 'reluplex_split')}개, "
          f"누적된 (layer,neuron) 종류: {len(counts)}개")
    for key, c in sorted(counts.items()):
        print(f"  layer={key[0]} neuron={key[1]}: {c}회")

    model = load_nn_model(args.model)
    fig, ax = draw_split_heatmap(
        model, counts, threshold=args.threshold, cap=args.cap,
        title=f"Split heatmap: {os.path.basename(args.model)}",
    )
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"저장 완료: {output_path}")


if __name__ == "__main__":
    main()
