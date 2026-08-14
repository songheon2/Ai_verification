"""
SolveTrace(visualization/SolveTrace.py)의 reluplex_split 이벤트를 모아 (layer,
neuron)별 누적 split 횟수를 계산하고, 모델의 모든 노드를 픽셀 셀로 그려
"어느 뉴런이 자주 split됐는지" 보여 주는 히트맵 도구.

여러 검증 인스턴스(idx, eps 등)를 하나의 SolveTrace에 누적해서 넘기면, 그
전체에 대한 합산 히트맵이 나온다 — 인스턴스 하나짜리 trace를 넘기면 그
인스턴스만의 히트맵이 된다.

"현재 진행 중인 split 스택" 애니메이션(원 계획의 실시간 스냅샷 GIF)은 이
모듈에 포함하지 않았다 — 여기서는 정적인 누적 히트맵까지만 다룬다.

다른 모듈에서 쓰는 법 (주 용도)
------------------------------
    from visualization.SolveTrace import SolveTrace
    from DPLL_T import dpll_t_detailed
    from visualization.SplitHeatmap import split_counts_from_trace, draw_split_heatmap

    trace = SolveTrace()
    dpll_t_detailed(formula, trace=trace)          # 여러 인스턴스를 같은 trace에 누적 가능
    counts = split_counts_from_trace(trace)         # {(layer, neuron): 누적 split 횟수}
    fig, ax = draw_split_heatmap(model, counts, threshold=1)
    fig.savefig("visualization/outputs/split_heatmap.png")

단독 실행 (범용 구조 시각화)
----------------------------
    python -m visualization.SplitHeatmap [--model model.bin|model.onnx] [--output output.png]
        [--demo-sweep] [--threshold N] [--cap N]

    --model, -m    : Custom(.bin/.txt/.custom) 또는 ONNX 신경망 경로
    --output, -o   : 출력 이미지 경로 (생략 시 <model파일명>_split_heatmap.png)
    --demo-sweep   : xor_network.bin 전용 split 검증 sweep 실행
    --threshold, -t: 호환성을 위해 남겨 둔 옵션 (노드 표시에 영향 없음)
    --cap          : 호환성을 위해 남겨 둔 옵션 (노드 표시에 영향 없음)

    기본 동작은 검증을 실행하지 않고 입력 모델의 모든 노드를 값 0인 셀로 그려
    구조를 보여 준다. --demo-sweep를 명시한 경우에만 xor_network.bin에 대해
    (case, eps) 여러 조합을 검증하고 실제 split 횟수로 색칠한다. 임의 모델의
    실제 split 히트맵은 모델에 맞는 VNNLIB와 Automation/AutoVerify.py를 사용한다.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from visualization.SolveTrace import SolveTrace
from GenericNNEncoding import NNModel
from visualization.NetworkLayout import parse_neuron_var


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
    """모델의 모든 노드를 하나씩 보존하는 split 횟수 히트맵을 그린다.

    각 열은 레이어, 각 행은 그 레이어의 실제 neuron index다. split 기록이
    없는 노드도 값 0인 셀로 반드시 표시하며, 짧은 레이어 아래의 회색 영역만
    실제 노드가 아닌 padding이다. 따라서 노드 수가 많아도 ``cap``이나
    ``threshold`` 때문에 생략되는 노드는 없다.

    ``threshold``와 ``cap``은 기존 호출 코드를 깨지 않기 위해 인자로만
    유지한다. 추가 matplotlib 옵션으로 ``ax``와 ``figsize``를 지원한다.
    그 밖의 알 수 없는 옵션은 조용히 무시하지 않고 ``TypeError``를 낸다.
    """
    # 이전 구현에서 쓰던 인자를 명시적으로 소비한다. 이제 이 값들은 색이나
    # 노드 선택에 관여하지 않는다: 히트맵의 핵심 불변식은 '전체 노드 표시'다.
    del threshold, cap

    ax = draw_network_kwargs.pop("ax", None)
    figsize = draw_network_kwargs.pop("figsize", None)
    if draw_network_kwargs:
        names = ", ".join(sorted(draw_network_kwargs))
        raise TypeError(f"지원하지 않는 draw_split_heatmap 옵션: {names}")

    layer_sizes = list(model.layer_sizes)
    if not layer_sizes or any(n <= 0 for n in layer_sizes):
        raise ValueError(f"layer_sizes는 양의 노드 수를 가져야 합니다: {layer_sizes!r}")

    n_layers = len(layer_sizes)
    max_nodes = max(layer_sizes)
    values = np.full((max_nodes, n_layers), np.nan, dtype=float)
    for layer, n_nodes in enumerate(layer_sizes):
        # 기록이 없는 노드까지 0으로 채우는 것이 중요하다.
        values[:n_nodes, layer] = [
            float(split_counts.get((layer, neuron), 0))
            for neuron in range(n_nodes)
        ]

    valid_values = values[~np.isnan(values)]
    if np.any(valid_values < 0):
        raise ValueError("split 횟수는 음수일 수 없습니다")

    own_fig = ax is None
    if own_fig:
        if figsize is None:
            # 기본 150 dpi 저장 시 보통 노드당 수 픽셀이 확보된다. 지나치게 큰
            # figure가 되지 않도록 제한하되 데이터 셀 자체는 절대 샘플링하지 않는다.
            figsize = (max(6.0, n_layers * 1.25 + 2.0),
                       max(5.0, min(18.0, max_nodes / 40.0)))
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    vmax = float(valid_values.max()) if valid_values.size else 0.0
    norm = mcolors.Normalize(vmin=0.0, vmax=max(1.0, vmax))
    cmap = matplotlib.colormaps["Reds"].copy()
    cmap.set_bad("#eeeeee")
    image = ax.imshow(
        np.ma.masked_invalid(values),
        origin="upper",
        interpolation="nearest",
        aspect="auto",
        cmap=cmap,
        norm=norm,
    )

    ax.set_xticks(range(n_layers))
    ax.set_xticklabels([f"L{layer}\n({n} nodes)" for layer, n in enumerate(layer_sizes)])
    ax.set_xlabel("Layer")
    ax.set_ylabel("Neuron index")
    ax.set_xlim(-0.5, n_layers - 0.5)
    ax.set_ylim(max_nodes - 0.5, -0.5)
    for boundary in np.arange(0.5, n_layers - 0.5, 1.0):
        ax.axvline(boundary, color="white", linewidth=1.5)
    if title:
        ax.set_title(title)

    colorbar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.03)
    colorbar.set_label("Split count")

    if own_fig:
        return fig, ax
    return ax


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

    from Automation.ModelInspector import load_model_for_verification

    visualization_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(visualization_dir)
    output_dir = os.path.join(visualization_dir, "outputs")
    default_model = os.path.join(project_dir, "xor_network.bin")

    parser = argparse.ArgumentParser(description="신경망 전체 노드 구조/누적 split 히트맵")
    parser.add_argument(
        "--model", "-m", default=default_model,
        help="Custom(.bin/.txt/.custom) 또는 ONNX 신경망 경로",
    )
    parser.add_argument("--output", "-o", default=None, help="출력 이미지 경로 (생략 시 visualization/outputs/<model파일명>_split_heatmap.png)")
    parser.add_argument(
        "--demo-sweep", action="store_true",
        help="xor_network.bin 전용 검증 sweep를 실행해 실제 split 횟수로 색칠",
    )
    parser.add_argument(
        "--threshold", "-t", type=float, default=1,
        help="이전 버전과의 CLI 호환용(현재는 0회 노드를 포함해 모든 노드를 표시)",
    )
    parser.add_argument(
        "--cap", type=int, default=20,
        help="이전 버전과의 CLI 호환용(현재는 레이어별 노드 수를 제한하지 않음)",
    )
    args = parser.parse_args()
    output_path = args.output or os.path.join(
        output_dir,
        os.path.splitext(os.path.basename(args.model))[0] + "_split_heatmap.png",
    )
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    model, model_info = load_model_for_verification(args.model)
    print(f"모델 로드 완료: 형식={model_info.source_format}, 구조={model.layer_sizes}, "
          f"전체 노드={sum(model.layer_sizes)}개")

    counts: Dict[Tuple[int, int], int] = {}
    if args.demo_sweep:
        if os.path.abspath(args.model) != os.path.abspath(default_model):
            parser.error("--demo-sweep는 기본 xor_network.bin 모델에서만 사용할 수 있습니다")
        trace = SolveTrace()
        _run_demo_sweep(args.model, trace)
        counts = split_counts_from_trace(trace)
        print(f"split 이벤트: {sum(1 for e in trace.events if e.component == 'reluplex_split')}개, "
              f"누적된 (layer,neuron) 종류: {len(counts)}개")
        for key, count in sorted(counts.items()):
            print(f"  layer={key[0]} neuron={key[1]}: {count}회")

    image_kind = "Split heatmap" if args.demo_sweep else "Network structure"
    fig, ax = draw_split_heatmap(
        model, counts, threshold=args.threshold, cap=args.cap,
        title=f"{image_kind}: {os.path.basename(args.model)}",
    )
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"저장 완료: {output_path}")


if __name__ == "__main__":
    main()
