"""
GenericNNEncoding.NNModel을 "교재 스타일" 완전연결망 다이어그램(레이어를 세로
한 줄로 세우고 인접 레이어의 모든 노드 쌍을 선으로 잇는 고전적인 신경망 그림)
으로 그리는 공용 모듈.

레이어 하나가 노드를 너무 많이 가지고 있으면(실제 모델은 256/491/491/16개
수준이라 다 그리면 선이 뭉개져서 못 알아봄) 일부만 그리고 나머지는 "⋮"로
생략한다 (교재에서 x1, x2, x3, ⋮, xp로 표시하는 것과 같은 방식). 어떤 노드를
그릴지 고르는 방식은 두 가지(select_mode)를 지원한다.

  select_mode="index" (기본값)
    구조만 볼 때 씀. 위 (max_display-2)개 + 아래 2개만 그리고 중간 하나를
    "⋮"로 생략한다 — 생략 구간이 항상 1개.

  select_mode="threshold"
    node_values[(layer, neuron)] >= threshold인 뉴런만 index 순서 그대로
    그리고, 나머지는 숨긴다. 숨김 구간은 연속으로 몇 개가 이어지든 "⋮" 1개로
    접이므로, 레이어 안에서 표시-숨김이 여러 번 반복되면 "⋮"도 여러 번 나올
    수 있다 (예: 뉴런 40번만 split이 잦고 41~199번은 안 잦고 200번이 또
    잦으면 "...40...⋮...200..." 처럼). threshold를 넘는 뉴런이 cap개보다
    많으면 값이 큰 순으로 cap개만 남긴다 (그래도 선이 너무 빽빽해지지 않게).
    ReLU split 횟수처럼 "어디가 핫스팟인지"를 보여줄 때 이 모드를 쓴다.
    단, 레이어 노드 수가 max_display 이하면 threshold를 무시하고 전부 그린다
    (예: 입력/출력층처럼 애초에 node_values가 없는 작은 레이어가 통째로
    "⋮" 하나로 사라지는 것을 막음 — 다 그려도 안 빽빽하니 굳이 숨길 이유가 없음).

기본 색은 역할별(input=초록, hidden=보라, output=분홍)로 칠하고, node_values를
넘기면 그 값이 있는 노드만 색 농도(cmap)로 덮어 칠한다. 생략(⋮)된 노드는
애초에 그려지지 않으므로 그 노드의 값은 화면에 안 나타난다.

변수명 <-> (layer, neuron) 변환
--------------------------------
GenericNNEncoding.py의 명명 규칙(은닉층 pre-activation z{layer}_{neuron},
post-activation h{layer}_{neuron})과 정확히 맞춘 parse_neuron_var() /
neuron_var_name()을 제공한다. 이후 Reluplex의 재귀 분기 변수(branch_x)
같은 문자열을 바로 (layer, neuron) 좌표로 매핑하는 데 쓸 수 있다.

다른 모듈에서 쓰는 법 (주 용도)
------------------------------
    from GenericNNEncoding import load_nn_model
    from visualization.NetworkLayout import draw_network, parse_neuron_var

    model = load_nn_model("Custom/model_custom.bin")
    node_values = {(1, 5): 3.0, (1, 12): 1.0}   # (layer, neuron) -> 값
    fig, ax = draw_network(model, node_values, title="예시")
    fig.savefig("out.png")

단독 실행 (데모/스모크 테스트용)
--------------------------------
    python -m visualization.NetworkLayout [--model model.bin] [--output output.png]
        [--select-mode index|threshold] [--max-display N] [--threshold T] [--cap N]

    --model, -m       : 커스텀 바이너리 신경망(.bin) 경로 (생략 시 xor_network.bin)
    --output, -o      : 출력 이미지 경로 (생략 시 <model파일명>_layout_demo_<mode>.png)
    --select-mode     : "index"(기본) 또는 "threshold"
    --max-display, -n : select-mode=index일 때 레이어당 최대 표시 노드 수 (기본 8)
    --threshold, -t   : select-mode=threshold일 때 표시 기준값 (기본 5.0)
    --cap             : select-mode=threshold일 때 레이어당 최대 표시 노드 수 (기본 20)

    옵션이라 순서 상관없이 넘길 수 있다. 예:
    python -m visualization.NetworkLayout --select-mode threshold --threshold 5 --model "Custom/model.bin"

    index 모드는 (레이어 내 실제 인덱스)를, threshold 모드는 40개 뉴런마다
    하나 + 몇 개를 이웃하게 배치한 가짜 "split 횟수"를 더미 값으로 채워서,
    실제 값을 아직 연결하지 않고도 레이아웃/색칠/생략 구간이 제대로 동작하는지
    (특히 threshold 모드에서 "⋮"가 여러 번 나오는지) 눈으로 확인할 수 있다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patches as patches

from GenericNNEncoding import NNModel

_NEURON_VAR_RE = re.compile(r"^[zh](\d+)_(\d+)(?:_.*)?$")

_ROLE_COLORS = {
    "input": "#8fd19e",
    "hidden": "#7b6fc9",
    "output": "#f2a6b0",
}


def parse_neuron_var(name: str) -> Optional[Tuple[int, int]]:
    """'z2_5' 또는 'h2_5' -> (layer=2, neuron=5). GenericNNEncoding.py의
    encode_nn()은 FreshGen을 거치면서 'z2_5_<prefix><n>' 처럼 뒤에 유일성
    suffix가 덧붙으므로, 접두사(z2_5) 뒤에 '_'로 시작하는 나머지는 무시하고
    매칭한다. 그 형식 자체와 안 맞는 이름이면 None을 반환한다
    (예: ineq_slack_3, relu_slack_x_pos_1 등)."""
    m = _NEURON_VAR_RE.match(name)
    if m is None:
        return None
    return int(m.group(1)), int(m.group(2))


def neuron_var_name(layer: int, neuron: int, kind: str = "z") -> str:
    """(layer, neuron) -> 'z{layer}_{neuron}' 또는 'h{layer}_{neuron}'."""
    if kind not in ("z", "h"):
        raise ValueError(f"kind는 'z' 또는 'h'여야 합니다: {kind!r}")
    return f"{kind}{layer}_{neuron}"


@dataclass(frozen=True)
class NodePosition:
    layer: int
    neuron: int
    x: float
    y: float


@dataclass(frozen=True)
class LayerLayout:
    layer: int
    n_total: int
    positions: List[NodePosition] = field(default_factory=list)
    truncated: bool = False
    ellipsis_ys: List[float] = field(default_factory=list)


def _role_of(layer: int, num_layers: int) -> str:
    if layer == 0:
        return "input"
    if layer == num_layers:
        return "output"
    return "hidden"


def _select_by_index(n: int, max_display: int) -> List[int]:
    """위 (max_display-2)개 + 아래 2개만 남기는 기존 방식 (구간이 하나뿐인 truncation)."""
    if n <= max_display:
        return list(range(n))
    top_k = max(1, max_display - 2)
    bottom_k = max(1, max_display - top_k)
    return list(range(top_k)) + list(range(n - bottom_k, n))


def _select_by_threshold(
    li: int, n: int, node_values: Dict[Tuple[int, int], float], threshold: float, cap: int
) -> List[int]:
    """threshold를 넘는 뉴런만 index 오름차순으로 골라낸다. 넘는 개수가 cap을
    초과하면 값이 큰 순으로 cap개만 남긴다(동점이면 index 낮은 쪽 우선)."""
    candidates = [
        (j, node_values[(li, j)])
        for j in range(n)
        if node_values.get((li, j), float("-inf")) >= threshold
    ]
    if len(candidates) > cap:
        candidates.sort(key=lambda pair: (-pair[1], pair[0]))
        candidates = candidates[:cap]
    return sorted(j for j, _ in candidates)


def _layout_column(li: int, n: int, shown: List[int], x: float, node_gap: float) -> LayerLayout:
    """shown(오름차순 index 목록)을 세로로 배치한다. shown에 없는 index가
    연속으로 이어지는 구간(앞/사이/뒤 어디든)마다 "⋮" 하나씩을 끼워 넣는다
    (숨김 구간 길이는 상관없이 마커 1개), index 순서는 항상 유지된다."""
    if not shown:
        return LayerLayout(layer=li, n_total=n, positions=[], truncated=n > 0, ellipsis_ys=[0.0])

    slots: List[Tuple[str, Optional[int]]] = []
    if shown[0] > 0:
        slots.append(("gap", None))
    slots.append(("node", shown[0]))
    for prev, cur in zip(shown, shown[1:]):
        if cur > prev + 1:
            slots.append(("gap", None))
        slots.append(("node", cur))
    if shown[-1] < n - 1:
        slots.append(("gap", None))

    total_height = (len(slots) - 1) * node_gap
    positions: List[NodePosition] = []
    ellipsis_ys: List[float] = []
    for i, (kind, j) in enumerate(slots):
        y = total_height / 2.0 - i * node_gap
        if kind == "node":
            positions.append(NodePosition(layer=li, neuron=j, x=x, y=y))
        else:
            ellipsis_ys.append(y)

    return LayerLayout(
        layer=li, n_total=n, positions=positions, truncated=len(shown) < n, ellipsis_ys=ellipsis_ys
    )


def compute_layout(
    model: NNModel,
    *,
    select_mode: str = "index",
    layer_gap: float = 3.0,
    node_gap: float = 1.0,
    max_display: int = 8,
    node_values: Optional[Dict[Tuple[int, int], float]] = None,
    threshold: Optional[float] = None,
    cap: int = 20,
) -> Dict[int, LayerLayout]:
    """레이어마다 노드를 세로 한 줄로 배치한다 (교재 스타일 다이어그램용).

    select_mode="index" (기본값)
        레이어의 노드 수 n이 max_display 이하면 전부 그리고, 넘으면 위쪽
        (max_display-2)개 + 아래쪽 2개만 그리고 중간을 "⋮" 하나로 생략한다.
        구조만 보여줄 때 쓴다 (값이 없어도 동작).

    select_mode="threshold"
        node_values[(layer, neuron)] >= threshold인 뉴런만 index 순서 그대로
        골라 그리고, 나머지는 숨겨서 연속 구간마다 "⋮"로 접는다. 넘는 뉴런이
        cap개보다 많으면 값이 큰 순으로 cap개만 남긴다. 예: split 횟수가 높은
        뉴런만 보이게 하고 싶을 때 node_values=split_counts, threshold=5 처럼 쓴다.
        단, 레이어의 노드 수 n이 max_display 이하면 threshold를 무시하고 전부
        그린다 — 입력/출력층처럼 애초에 node_values가 없는 작은 레이어가
        전부 숨겨져서 "⋮" 하나만 남는 것을 막기 위함(어차피 다 그려도 안 빽빽함).

    반환: {layer: LayerLayout}  (LayerLayout.ellipsis_ys는 레이어 안에서 숨겨진
    구간마다 하나씩 들어가므로, threshold 모드에서는 여러 개일 수 있다)
    """
    if select_mode not in ("index", "threshold"):
        raise ValueError(f"select_mode는 'index' 또는 'threshold'여야 합니다: {select_mode!r}")
    if select_mode == "threshold" and threshold is None:
        raise ValueError("select_mode='threshold'면 threshold를 지정해야 합니다")

    layouts: Dict[int, LayerLayout] = {}
    x_cursor = 0.0

    for li, n in enumerate(model.layer_sizes):
        if select_mode == "index":
            shown = _select_by_index(n, max_display)
        elif n <= max_display:
            shown = list(range(n))
        else:
            shown = _select_by_threshold(li, n, node_values or {}, threshold, cap)

        layouts[li] = _layout_column(li, n, shown, x_cursor, node_gap)
        x_cursor += layer_gap

    return layouts


def draw_network(
    model: NNModel,
    node_values: Optional[Dict[Tuple[int, int], float]] = None,
    *,
    title: Optional[str] = None,
    cmap_name: str = "Reds",
    node_radius: float = 0.28,
    select_mode: str = "index",
    max_display: int = 8,
    threshold: Optional[float] = None,
    cap: int = 20,
    layer_gap: float = 3.0,
    node_gap: float = 1.0,
    show_edges: bool = True,
    edge_color: str = "#999999",
    edge_alpha: float = 0.35,
    show_layer_titles: bool = True,
    show_weight_labels: bool = True,
    show_activation_labels: bool = True,
    show_io_var_labels: bool = True,
    ax=None,
):
    """레이어를 세로 한 줄로 세우고 인접 레이어를 모두 잇는 완전연결망 다이어그램을 그린다.

    node_values가 주어지면 {(layer, neuron): 값}에 있는 노드만 그 값의 크기에
    비례해 cmap_name 색 농도로 덮어 칠하고, 나머지는 역할별 기본색(input=초록,
    hidden=보라, output=분홍)으로 그린다. node_values를 안 주면 전부 기본색이라
    순수 구조 다이어그램으로 쓸 수 있다.

    select_mode="threshold"면 node_values를 색칠뿐 아니라 어떤 노드를 그릴지
    고르는 데도 쓴다 (threshold를 넘는 노드만, index 순서 유지, 숨김 구간마다
    "⋮" — compute_layout 참고). max_display/threshold/cap 의미는 compute_layout과 동일.

    ax를 넘기면 그 축에 그리고 ax를 반환하며, 넘기지 않으면 새 figure를
    만들어 (fig, ax)를 반환한다.
    """
    layouts = compute_layout(
        model, select_mode=select_mode, layer_gap=layer_gap, node_gap=node_gap,
        max_display=max_display, node_values=node_values, threshold=threshold, cap=cap,
    )
    layer_indices = sorted(layouts)

    # 실제 데이터 범위(특히 threshold 모드는 레이어마다 세로 길이가 크게 다를
    # 수 있음)를 먼저 재고, 그에 비례하는 figsize를 잡는다. 고정 크기를 쓰면
    # aspect='equal' 때문에 세로로 긴 데이터가 넓고 낮은 figure 안에 눌려
    # 들어가면서 열들이 한쪽으로 뭉치고 라벨이 겹치는 문제가 생긴다.
    all_ys = (
        [p.y for layout in layouts.values() for p in layout.positions]
        + [ey for layout in layouts.values() for ey in layout.ellipsis_ys]
    )
    top_y = max(all_ys) if all_ys else 0.0
    bottom_y = min(all_ys) if all_ys else 0.0
    xs = [li * layer_gap for li in layer_indices]

    own_fig = ax is None
    if own_fig:
        unit_scale = 0.4  # inch per data unit
        width_units = (max(xs) - min(xs)) if xs else 1.0
        height_units = (top_y - bottom_y) + node_radius * 8  # 위/아래 라벨 자리 여유
        fig_w = max(6.0, (width_units + 3.2) * unit_scale)
        fig_h = max(4.0, height_units * unit_scale)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    node_values = node_values or {}
    vals = [v for v in node_values.values() if v is not None]
    has_values = bool(vals)
    if has_values:
        vmin, vmax = min(vals), max(vals)
        if vmin == vmax:
            vmax = vmin + 1.0
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cmap = matplotlib.colormaps[cmap_name]

    if show_edges:
        for li, lj in zip(layer_indices, layer_indices[1:]):
            for p1 in layouts[li].positions:
                for p2 in layouts[lj].positions:
                    ax.plot([p1.x, p2.x], [p1.y, p2.y],
                            color=edge_color, linewidth=0.4, alpha=edge_alpha, zorder=1)

    for li in layer_indices:
        layout = layouts[li]
        role = _role_of(li, model.num_layers)
        base_color = _ROLE_COLORS[role]

        for pos in layout.positions:
            value = node_values.get((pos.layer, pos.neuron))
            color = cmap(norm(value)) if (has_values and value is not None) else base_color
            ax.add_patch(patches.Circle((pos.x, pos.y), node_radius,
                                         facecolor=color, edgecolor="black", linewidth=0.6, zorder=2))

        for ey in layout.ellipsis_ys:
            ax.text(li * layer_gap, ey, "⋮", fontsize=14, ha="center", va="center", zorder=2)

        if show_io_var_labels and role in ("input", "output") and layout.positions:
            var_letter = "x" if role == "input" else "y"
            dx = -node_radius * 2.2 if role == "input" else node_radius * 2.2
            ha = "right" if role == "input" else "left"
            for pos in layout.positions:
                label_idx = pos.neuron + 1 if role == "input" else pos.neuron
                ax.text(pos.x + dx, pos.y, f"${var_letter}_{{{label_idx}}}$",
                         fontsize=9, ha=ha, va="center", zorder=2)

    if show_layer_titles:
        for li in layer_indices:
            role = _role_of(li, model.num_layers)
            role_label = {"input": "Input", "hidden": "Hidden", "output": "Output"}[role]
            ax.text(li * layer_gap, top_y + node_radius * 4, f"{role_label}\nlayer $L_{{{li + 1}}}$",
                     fontsize=10, ha="center", va="bottom")

    label_y = bottom_y - node_radius * 4
    if show_activation_labels:
        for li in layer_indices:
            if li == 0:
                continue  # 입력층은 a^(i) 라벨 없음
            ax.text(li * layer_gap, label_y, f"$a^{{({li + 1})}}$", fontsize=10, ha="center", va="top")

    if show_weight_labels:
        for i, (li, lj) in enumerate(zip(layer_indices, layer_indices[1:]), start=1):
            xm = (li * layer_gap + lj * layer_gap) / 2
            ax.text(xm, label_y, f"$W^{{({i})}}$", fontsize=10, ha="center", va="top")

    pad_x = 1.6 if show_io_var_labels else 1.0
    ax.set_xlim(min(xs) - pad_x, max(xs) + pad_x)
    ax.set_ylim(label_y - 0.6, top_y + node_radius * 4 + 1.0)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=13, style="italic")

    if has_values:
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)

    if own_fig:
        return fig, ax
    return ax


def main() -> None:
    import argparse
    import os
    from GenericNNEncoding import load_nn_model

    visualization_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(visualization_dir)
    output_dir = os.path.join(visualization_dir, "outputs")
    default_model = os.path.join(project_dir, "xor_network.bin")

    parser = argparse.ArgumentParser(description="NetworkLayout 데모/스모크 테스트")
    parser.add_argument("--model", "-m", default=default_model, help="커스텀 바이너리 신경망(.bin) 경로")
    parser.add_argument("--output", "-o", default=None, help="출력 이미지 경로 (생략 시 visualization/outputs/<model파일명>_layout_demo.png)")
    parser.add_argument("--select-mode", choices=["index", "threshold"], default="index",
                         help="'index': 위/아래 max-display개만. 'threshold': 값이 threshold 넘는 노드만")
    parser.add_argument("--max-display", "-n", type=int, default=8, help="select-mode=index일 때 레이어당 최대 표시 노드 수")
    parser.add_argument("--threshold", "-t", type=float, default=5.0, help="select-mode=threshold일 때 표시 기준값")
    parser.add_argument("--cap", type=int, default=20, help="select-mode=threshold일 때 레이어당 최대 표시 노드 수")
    args = parser.parse_args()

    model_path = args.model
    output_path = args.output or os.path.join(
        output_dir,
        os.path.splitext(os.path.basename(model_path))[0] + f"_layout_demo_{args.select_mode}.png",
    )
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    model = load_nn_model(model_path)
    print(f"모델 로드 완료: 가중치 레이어 {model.num_layers}개, 크기 {model.layer_sizes}")

    if args.select_mode == "index":
        # 데모용 더미 값: 은닉/출력층 노드 전부에 (레이어 내 실제 인덱스)를
        # 값으로 채워서, 색이 인덱스에 따라 달라지는지 확인.
        demo_values = {
            (li, j): float(j)
            for li, n in enumerate(model.layer_sizes) if li > 0
            for j in range(n)
        }
    else:
        # 데모용 더미 값: split 횟수를 흉내내서, 40개마다 하나씩 + 몇 개는
        # 이웃끼리 붙여서 "표시-숨김-표시-숨김..."이 여러 번 반복되는지,
        # 즉 레이어당 ⋮가 여러 번 나오는지 눈으로 확인한다.
        demo_values = {}
        for li, n in enumerate(model.layer_sizes):
            hot = set(range(0, n, 40)) | {j for j in range(2, min(5, n))}
            for j in range(n):
                demo_values[(li, j)] = 10.0 if j in hot else 0.0

    layouts = compute_layout(model, select_mode=args.select_mode, max_display=args.max_display,
                              node_values=demo_values, threshold=args.threshold, cap=args.cap)
    shown = sum(len(l.positions) for l in layouts.values())
    n_ellipses = sum(len(l.ellipsis_ys) for l in layouts.values())
    print(f"레이아웃 계산 완료: 표시 노드 {shown}개 / 전체 {sum(model.layer_sizes)}개, ⋮ {n_ellipses}개")

    fig, ax = draw_network(model, demo_values, title=f"NetworkLayout demo ({args.select_mode}): {os.path.basename(model_path)}",
                            select_mode=args.select_mode, max_display=args.max_display,
                            threshold=args.threshold, cap=args.cap)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"저장 완료: {output_path}")


if __name__ == "__main__":
    main()
