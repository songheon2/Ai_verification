"""
커스텀 바이너리 신경망(.bin)의 프루닝(가중치/바이어스 0) 여부를 시각화.

두 가지 모드:

  --mode node (기본)
    레이어의 각 출력 노드마다 작은 박스 2개를 그린다:
      박스 1 (weight) : 그 노드로 들어오는 가중치가 전부 0(허용오차 이내)이면
                        빨강, 하나라도 0이 아니면 녹색
      박스 2 (bias)   : 그 노드의 바이어스가 0(허용오차 이내)이면 빨강, 아니면 녹색

  --mode element
    가중치 행렬 W(n_out x n_in)의 스칼라 원소 하나하나를 픽셀 1개로 찍는다
    (0이면 빨강, 아니면 녹색). bias는 각 행(출력 뉴런) 오른쪽에 얇은 열을
    하나 더 붙여서 같은 방식으로 표시한다. 원소 수가 매우 많아(레이어당
    최대 수십만 개) 도형(patch) 대신 numpy 배열을 직접 픽셀로 저장하며,
    --px-scale로 셀 하나를 NxN 픽셀로 확대할 수 있다 (기본 1:1, 리샘플링/
    안티에일리어싱으로 인한 손실 없음). 레이어마다 별도 PNG로 저장한다.

부동소수점 비교는 정확히 `== 0.0`으로 하지 않고, 절댓값이 허용오차(tol)
이하이면 0으로 간주한다. tol은 --tol 인자로 지정하며 기본값은 1e-7이다.

입력층(가중치/바이어스가 없는 레이어 0)은 그리지 않고, 가중치 레이어
1..m(각 레이어의 출력 노드 기준)만 그린다.

사용법:
  python VisualizeSparsity.py <custom.bin> [output] [--tol <threshold>] \
      [--mode node|element] [--px-scale N]
"""

from __future__ import annotations
import math
import os
import sys
import platform

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import to_rgb

from CustomBinary import read_custom

GREEN = "#2ca02c"
RED = "#d62728"


def _setup_font():
    if platform.system() == "Windows":
        plt.rcParams["font.family"] = "Malgun Gothic"
    elif platform.system() == "Darwin":
        plt.rcParams["font.family"] = "AppleGothic"
    else:
        plt.rcParams["font.family"] = "NanumGothic"
    plt.rcParams["axes.unicode_minus"] = False


def _layer_status(W: np.ndarray, b: np.ndarray, zero_tol: float):
    """반환: (w_nonzero[j], b_nonzero[j])  — j번째 출력 노드 기준

    절댓값이 zero_tol 이하이면 0으로 간주한다 (부동소수점 == 비교 회피).
    """
    w_nonzero = np.any(np.abs(W) > zero_tol, axis=1)
    b_nonzero = np.abs(b) > zero_tol
    return w_nonzero, b_nonzero


def _draw_layer(ax, n_in: int, n_out: int, w_nonzero: np.ndarray, b_nonzero: np.ndarray, title: str):
    cols = max(1, math.ceil(math.sqrt(n_out)))
    rows = math.ceil(n_out / cols)

    box_w, box_h = 0.4, 0.8
    gap_x, gap_y = 0.15, 0.25
    cell_w = 2 * box_w + gap_x
    cell_h = box_h + gap_y

    for j in range(n_out):
        r, c = divmod(j, cols)
        x0 = c * (cell_w + gap_x)
        y0 = (rows - 1 - r) * (cell_h + gap_y)

        w_color = GREEN if w_nonzero[j] else RED
        b_color = GREEN if b_nonzero[j] else RED

        ax.add_patch(patches.Rectangle((x0, y0), box_w, box_h,
                                        facecolor=w_color, edgecolor="black", linewidth=0.15))
        ax.add_patch(patches.Rectangle((x0 + box_w, y0), box_w, box_h,
                                        facecolor=b_color, edgecolor="black", linewidth=0.15))

    ax.set_xlim(-gap_x, cols * (cell_w + gap_x))
    ax.set_ylim(-gap_y, rows * (cell_h + gap_y))
    ax.set_aspect("equal")
    ax.axis("off")

    w_pruned = int(np.sum(~w_nonzero))
    b_pruned = int(np.sum(~b_nonzero))
    ax.set_title(
        f"{title}  ({n_in}→{n_out})\n"
        f"weight all-zero: {w_pruned}/{n_out}   bias zero: {b_pruned}/{n_out}",
        fontsize=10,
    )


def visualize_node(filepath: str, output_path: str | None = None, title: str | None = None,
                    zero_tol: float = 1e-7) -> str:
    sizes, weights, biases = read_custom(filepath)
    m = len(weights)

    _setup_font()

    fig, axes = plt.subplots(1, m, figsize=(6 * m, 8))
    if m == 1:
        axes = [axes]

    for li in range(m):
        W = np.array(weights[li])
        b = np.array(biases[li])
        w_nonzero, b_nonzero = _layer_status(W, b, zero_tol)
        _draw_layer(axes[li], sizes[li], sizes[li + 1], w_nonzero, b_nonzero, f"Layer {li + 1}")

    legend_elems = [
        patches.Patch(facecolor=GREEN, edgecolor="black", label="nonzero"),
        patches.Patch(facecolor=RED, edgecolor="black", label="zero (pruned)"),
    ]
    plt.tight_layout(rect=(0, 0, 1, 0.90))
    fig.suptitle(title or os.path.basename(filepath), fontsize=14, x=0.02, y=0.99, ha="left")
    fig.legend(handles=legend_elems, loc="upper right", bbox_to_anchor=(0.99, 0.99), ncol=2)

    if output_path is None:
        base = os.path.splitext(os.path.basename(filepath))[0]
        output_path = os.path.join(os.path.dirname(os.path.abspath(filepath)), base + "_sparsity.png")

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"저장 완료: {output_path}")
    return output_path


def _layer_element_image(W: np.ndarray, b: np.ndarray, zero_tol: float,
                          gap: int = 2) -> tuple[np.ndarray, int, int]:
    """W(n_out,n_in) + gap(흰색) + b(n_out,1)를 이어붙인 (n_out, n_in+gap+1, 3) uint8 RGB 배열.

    반환값에는 (image, w_zero_count, b_zero_count)도 함께 담는다.
    """
    green_rgb = tuple(int(round(c * 255)) for c in to_rgb(GREEN))
    red_rgb = tuple(int(round(c * 255)) for c in to_rgb(RED))

    n_out, n_in = W.shape
    w_mask = np.abs(W) > zero_tol            # (n_out, n_in)
    b_mask = np.abs(b) > zero_tol             # (n_out,)

    img = np.empty((n_out, n_in + gap + 1, 3), dtype=np.uint8)
    img[:, :n_in] = np.where(w_mask[:, :, None], green_rgb, red_rgb)
    img[:, n_in:n_in + gap] = 255
    img[:, n_in + gap:] = np.where(b_mask[:, None, None], green_rgb, red_rgb)

    return img, int(np.sum(~w_mask)), int(np.sum(~b_mask))


def visualize_elementwise(filepath: str, output_dir: str | None = None,
                           zero_tol: float = 1e-7, px_scale: int = 1) -> list[str]:
    """레이어마다 가중치 원소 1개 = 픽셀 1개짜리 PNG를 저장한다 (bias는 오른쪽 열).

    output_dir을 지정하지 않으면 입력 파일과 같은 폴더에 저장한다.
    px_scale > 1이면 각 픽셀을 NxN 블록으로 확대한다 (nearest, 리샘플링 손실 없음).
    """
    _, weights, biases = read_custom(filepath)
    m = len(weights)

    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(filepath))
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(filepath))[0]

    output_paths = []
    for li in range(m):
        W = np.array(weights[li])
        b = np.array(biases[li])
        img, w_zero, b_zero = _layer_element_image(W, b, zero_tol)

        if px_scale > 1:
            img = np.repeat(np.repeat(img, px_scale, axis=0), px_scale, axis=1)

        out_path = os.path.join(output_dir, f"{base}_L{li + 1}_element.png")
        plt.imsave(out_path, img)
        output_paths.append(out_path)

        n_out, n_in = W.shape
        print(f"Layer {li + 1} ({n_in}->{n_out}): "
              f"weight zero {w_zero}/{W.size} ({w_zero / W.size * 100:.1f}%)   "
              f"bias zero {b_zero}/{n_out} ({b_zero / n_out * 100:.1f}%)   -> {out_path}")

    print("범례: 초록=nonzero, 빨강=zero(pruned) / 각 레이어 이미지 오른쪽 얇은 열이 bias")
    return output_paths


if __name__ == "__main__":
    argv = sys.argv[1:]

    zero_tol = 1e-7
    if "--tol" in argv:
        idx = argv.index("--tol")
        zero_tol = float(argv[idx + 1])
        del argv[idx:idx + 2]

    mode = "node"
    if "--mode" in argv:
        idx = argv.index("--mode")
        mode = argv[idx + 1]
        del argv[idx:idx + 2]

    px_scale = 1
    if "--px-scale" in argv:
        idx = argv.index("--px-scale")
        px_scale = int(argv[idx + 1])
        del argv[idx:idx + 2]

    if len(argv) < 1:
        print("사용법: python VisualizeSparsity.py <custom.bin> [output] [--tol <t>] "
              "[--mode node|element] [--px-scale N]")
        sys.exit(1)

    input_path = argv[0]
    output_arg = argv[1] if len(argv) >= 2 else None

    if mode == "element":
        visualize_elementwise(input_path, output_arg, zero_tol=zero_tol, px_scale=px_scale)
    elif mode == "node":
        visualize_node(input_path, output_arg, zero_tol=zero_tol)
    else:
        print(f"알 수 없는 --mode 값: {mode} (node 또는 element만 지원)")
        sys.exit(1)
