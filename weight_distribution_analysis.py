"""
mMIMO 안테나 선택 신경망(256 -> 491 -> 491 -> 16, ReLU/ReLU/linear)의
weight 스칼라 분포를 threshold(eps) 기준 5개 카테고리로 분류/분석한다.

weight 로딩은 GenericNNEncoding.py의 load_nn_model()을 그대로 재사용한다
(bias는 이번 분석 대상에서 완전히 제외 — model.weights[0..2]만 사용).

카테고리 정의 (5-way, 상호배타 · 전체포괄, eps=1e-7 기본값)
    C0: w == 0
    C1: 0 < w <= eps
    C2: -eps <= w < 0
    C3: w > eps
    C4: w < -eps

출력
    1) 텍스트 위치 리포트 (콘솔)
       - C0: 레이어별 개수/비율만 (좌표 리스트 생략)
       - C1/C2: 전체 좌표 (output_idx, input_idx, w) 나열
       - C3/C4: |w| 내림차순 상위 N개만 나열 (--top-n)
    2) 분포 바차트 PNG (W1/W2/W3/전체 x C0..C4 grouped bar)
    3) 특정 노드 조회 (--layer, --node-idx) — 요약/상세/랭킹

사용법:
    python weight_distribution_analysis.py [--model PATH] [--eps 1e-7]
        [--top-n 20] [--rank-n 10] [--pct] [--output weight_distribution.png]
        [--layer 1|2|3 --node-idx N]
"""

from __future__ import annotations

import argparse
import os
import platform
import sys
from dataclasses import dataclass
from typing import List

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Windows 콘솔 기본 코드페이지(cp949)는 한글은 되지만 em-dash(—) 등 일부 유니코드
# 문자는 인코딩하지 못해 UnicodeEncodeError로 죽는다. UTF-8로 강제 전환한다.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

from GenericNNEncoding import load_nn_model

# ---------------------------------------------------------------------------
# 상수
# ---------------------------------------------------------------------------

EPS = 1e-7

C0, C1, C2, C3, C4 = range(5)
N_CATEGORIES = 5

CATEGORY_LABELS = {
    C0: "C0 (w == 0)",
    C1: "C1 (0 < w <= eps)",
    C2: "C2 (-eps <= w < 0)",
    C3: "C3 (w > eps)",
    C4: "C4 (w < -eps)",
}

# C0=빨강 / C1,C2=주황·노랑 계열(경계값, 밝음=+/어두움=-) / C3,C4=녹색·청록 계열(유의미, 밝음=+/어두움=-)
CATEGORY_COLORS = {
    C0: "#d62728",  # red
    C1: "#ffcc00",  # bright amber  (0 < w <= eps, +)
    C2: "#b36b00",  # dark amber    (-eps <= w < 0, -)
    C3: "#2ca02c",  # bright green  (w > eps, +)
    C4: "#0b6e4f",  # dark teal     (w < -eps, -)
}

DEFAULT_MODEL_PATH = os.path.join(
    _SCRIPT_DIR, "..", "models", "Custom",
    "Baseline mMIMO FC H hard short 80 HTHNN_LAY2_491 RELU 20241018 PRUNED 0.93_NO_SIGMOID_custom.bin",
)
DEFAULT_OUTPUT_PNG = os.path.join(_SCRIPT_DIR, "weight_distribution.png")

EXPECTED_LAYER_SHAPES = [(491, 256), (491, 491), (16, 491)]  # (out, in) — 문서상 기대 구조


def _setup_font():
    if platform.system() == "Windows":
        plt.rcParams["font.family"] = "Malgun Gothic"
    elif platform.system() == "Darwin":
        plt.rcParams["font.family"] = "AppleGothic"
    else:
        plt.rcParams["font.family"] = "NanumGothic"
    plt.rcParams["axes.unicode_minus"] = False


# ---------------------------------------------------------------------------
# 데이터 구조
# ---------------------------------------------------------------------------

@dataclass
class LayerStats:
    layer_id: int  # 1-indexed
    W: np.ndarray  # (n_out, n_in)
    cat: np.ndarray  # (n_out, n_in), int8, values in {C0..C4}
    counts: np.ndarray  # (5,) int64
    ratios: np.ndarray  # (5,) float64
    node_counts: np.ndarray  # (n_out, 5) int64 — 노드(출력)별 카테고리 카운트

    @property
    def n_out(self) -> int:
        return self.W.shape[0]

    @property
    def n_in(self) -> int:
        return self.W.shape[1]

    @property
    def size(self) -> int:
        return self.W.size


# ---------------------------------------------------------------------------
# 1. 로딩 + 카테고리 분류 파이프라인
# ---------------------------------------------------------------------------

def load_weights(model_path: str) -> List[np.ndarray]:
    """GenericNNEncoding.load_nn_model()을 재사용해 W1, W2, W3만 numpy 배열로 반환한다 (bias 제외).

    NNModel.weights[i]는 (nested) Python list이므로 여기서 numpy 배열로 변환한다.
    """
    model = load_nn_model(model_path)
    weights = [np.asarray(W, dtype=float) for W in model.weights]
    if len(weights) != 3:
        raise ValueError(f"레이어 수가 3이 아닙니다 (m={len(weights)}) — 이 스크립트는 256->491->491->16 구조를 가정합니다.")
    for li, (W, expected) in enumerate(zip(weights, EXPECTED_LAYER_SHAPES), start=1):
        if tuple(W.shape) != expected:
            print(f"[경고] Layer{li} shape={W.shape}, 기대값={expected} (문서상 구조와 다름 — 실제 shape 기준으로 계속 진행)")
    return weights


def categorize(W: np.ndarray, eps: float = EPS) -> np.ndarray:
    """(out, in) weight 행렬을 벡터화 연산으로 C0..C4 카테고리 배열(int8)로 변환한다."""
    conditions = [
        W == 0.0,
        (W > 0.0) & (W <= eps),
        (W >= -eps) & (W < 0.0),
        W > eps,
        W < -eps,
    ]
    choices = [C0, C1, C2, C3, C4]
    cat = np.select(conditions, choices, default=-1)
    if np.any(cat == -1):
        raise AssertionError("categorize: 5개 카테고리 중 어디에도 속하지 않는 값이 있습니다 (NaN/Inf 의심)")
    return cat.astype(np.int8)


def compute_node_counts(cat: np.ndarray) -> np.ndarray:
    """(n_out, n_in) 카테고리 배열 -> (n_out, 5) 노드별 카테고리 카운트 (벡터화, for-loop 없음)."""
    n_out = cat.shape[0]
    counts = np.empty((n_out, N_CATEGORIES), dtype=np.int64)
    for c in range(N_CATEGORIES):
        counts[:, c] = np.sum(cat == c, axis=1)
    return counts


def build_layer_stats(layer_id: int, W: np.ndarray, eps: float = EPS) -> LayerStats:
    cat = categorize(W, eps)
    counts = np.bincount(cat.ravel(), minlength=N_CATEGORIES).astype(np.int64)

    # sanity check: C0+C1+C2+C3+C4 == out_dim * in_dim
    assert counts.sum() == W.size, (
        f"Layer{layer_id} sanity check 실패: sum(counts)={counts.sum()} != W.size={W.size}"
    )

    ratios = counts / W.size
    node_counts = compute_node_counts(cat)

    # sanity check: 노드별 합계도 n_in과 일치해야 함
    assert np.all(node_counts.sum(axis=1) == W.shape[1]), (
        f"Layer{layer_id} 노드별 sanity check 실패: 일부 노드의 카테고리 합계가 n_in과 다릅니다"
    )

    return LayerStats(layer_id=layer_id, W=W, cat=cat, counts=counts, ratios=ratios, node_counts=node_counts)


def build_all_stats(weights: List[np.ndarray], eps: float = EPS) -> List[LayerStats]:
    return [build_layer_stats(li, W, eps) for li, W in enumerate(weights, start=1)]


def total_counts(stats: List[LayerStats]) -> np.ndarray:
    counts = np.zeros(N_CATEGORIES, dtype=np.int64)
    for s in stats:
        counts += s.counts
    return counts


# ---------------------------------------------------------------------------
# 2. 출력 1 — 텍스트 위치 리포트
# ---------------------------------------------------------------------------

def print_position_report(stats: List[LayerStats], top_n: int = 20) -> None:
    print("\n" + "=" * 78)
    print("출력 1: 카테고리별 텍스트 위치 리포트")
    print("=" * 78)

    for s in stats:
        print(f"\n--- Layer{s.layer_id} ({s.n_in} -> {s.n_out}, total={s.size}) ---")

        # C0: 개수/비율만
        c0_count, c0_ratio = s.counts[C0], s.ratios[C0]
        print(f"[{CATEGORY_LABELS[C0]}] count={c0_count} ({c0_ratio * 100:.2f}%)  (좌표 목록 생략)")

        # C1, C2: 전체 좌표 리스트
        for c in (C1, C2):
            count, ratio = s.counts[c], s.ratios[c]
            print(f"[{CATEGORY_LABELS[c]}] count={count} ({ratio * 100:.4f}%)")
            if count == 0:
                print("  (해당 없음)")
                continue
            rows, cols = np.nonzero(s.cat == c)
            for r, col in zip(rows.tolist(), cols.tolist()):
                print(f"  (output_idx={r}, input_idx={col}, w={s.W[r, col]:.10e})")

        # C3, C4: |w| 내림차순 상위 N개
        for c in (C3, C4):
            count, ratio = s.counts[c], s.ratios[c]
            print(f"[{CATEGORY_LABELS[c]}] count={count} ({ratio * 100:.2f}%)  — 상위 {min(top_n, count)}개 (|w| 내림차순)")
            if count == 0:
                print("  (해당 없음)")
                continue
            rows, cols = np.nonzero(s.cat == c)
            vals = s.W[rows, cols]
            order = np.argsort(-np.abs(vals))[:top_n]
            for r, col, w in zip(rows[order].tolist(), cols[order].tolist(), vals[order].tolist()):
                print(f"  (output_idx={r}, input_idx={col}, w={w:.6f})")


# ---------------------------------------------------------------------------
# 3. 출력 2 — 분포 바차트
# ---------------------------------------------------------------------------

def plot_distribution(stats: List[LayerStats], output_path: str, show_pct: bool = False) -> None:
    _setup_font()

    group_names = [f"W{s.layer_id}" for s in stats] + ["전체(합산)"]
    group_counts = [s.counts for s in stats] + [total_counts(stats)]
    group_totals = [s.size for s in stats] + [sum(s.size for s in stats)]

    n_groups = len(group_names)
    x = np.arange(n_groups)
    bar_w = 0.15

    fig, ax = plt.subplots(figsize=(11, 6.5))

    for i, c in enumerate(range(N_CATEGORIES)):
        counts_c = np.array([counts[c] for counts in group_counts], dtype=np.int64)
        totals_c = np.array(group_totals, dtype=np.int64)
        pct_c = counts_c / totals_c * 100

        y_vals = pct_c if show_pct else counts_c
        offset = (i - (N_CATEGORIES - 1) / 2) * bar_w
        bars = ax.bar(x + offset, y_vals, width=bar_w, color=CATEGORY_COLORS[c], label=CATEGORY_LABELS[c])

        for bar, cnt, pct in zip(bars, counts_c, pct_c):
            ax.annotate(
                f"{cnt}\n({pct:.1f}%)",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center", va="bottom", fontsize=6.5, rotation=90,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(group_names)
    ax.set_ylabel("비율 (%)" if show_pct else "개수 (count)")
    ax.set_title("mMIMO 안테나 선택 신경망 — Layer별 weight 카테고리 분포 (eps=%.0e)" % EPS)
    ax.legend(loc="upper right", fontsize=9)
    ax.margins(y=0.18)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"\n바차트 저장 완료: {output_path}")

    overall_c0_ratio = total_counts(stats)[C0] / sum(s.size for s in stats) * 100
    close_to_93 = abs(overall_c0_ratio - 93.0) <= 2.0
    verdict = "예상(약 93%)과 일치" if close_to_93 else "예상(약 93%)과 차이 있음 — 확인 필요"
    print(f"[Sanity] 전체 C0(=0) 비율 = {overall_c0_ratio:.2f}%  -> {verdict}")


# ---------------------------------------------------------------------------
# 4. 출력 3 — 특정 노드 조회
# ---------------------------------------------------------------------------

def report_node(stats: List[LayerStats], layer_id: int, node_idx: int) -> None:
    s = next((s for s in stats if s.layer_id == layer_id), None)
    if s is None:
        raise ValueError(f"layer_id={layer_id} 를 찾을 수 없습니다 (1..{len(stats)} 범위여야 함)")
    if not (0 <= node_idx < s.n_out):
        raise ValueError(f"node_idx={node_idx} 범위 초과 (Layer{layer_id}의 n_out={s.n_out})")

    counts = s.node_counts[node_idx]
    ratios = counts / s.n_in

    # (1) 요약 레벨
    print(f"\n[Layer{layer_id} - Node {node_idx}] (input_dim={s.n_in})")
    parts = [f"{CATEGORY_LABELS[c].split()[0]}={counts[c]} ({ratios[c] * 100:.2f}%)" for c in range(N_CATEGORIES)]
    print("  " + "  ".join(parts))
    total = int(counts.sum())
    check = "✓" if total == s.n_in else "✗"
    print(f"  합계 검증: {'+'.join(str(c) for c in counts.tolist())} = {total} (input_dim={s.n_in}) {check}")
    assert total == s.n_in, "노드 합계가 input_dim과 일치하지 않습니다"

    # (2) 상세 레벨
    print(f"\n  상세 (0이 아닌 weight, |w| 내림차순):")
    print(f"    C0: 전부 0 ({counts[C0]}개, 목록 생략)")
    row_cat = s.cat[node_idx]
    row_w = s.W[node_idx]
    nonzero_idx = np.nonzero(row_cat != C0)[0]
    if nonzero_idx.size == 0:
        print("    (0이 아닌 weight 없음)")
    else:
        order = nonzero_idx[np.argsort(-np.abs(row_w[nonzero_idx]))]
        for idx in order.tolist():
            c = int(row_cat[idx])
            print(f"    input_idx={idx:>4}  w={row_w[idx]: .10f}  category={CATEGORY_LABELS[c]}")


def report_node_ranking(stats: List[LayerStats], rank_n: int = 10) -> None:
    print("\n" + "=" * 78)
    print(f"출력 3-(3): 노드 랭킹 뷰 (C0 비율=프루닝 비율 기준, top-{rank_n})")
    print("=" * 78)

    rows = []
    for s in stats:
        c0_ratio = s.node_counts[:, C0] / s.n_in
        for idx in range(s.n_out):
            rows.append((s.layer_id, idx, float(c0_ratio[idx]), s.node_counts[idx].copy()))

    rows.sort(key=lambda r: r[2], reverse=True)
    most_pruned = rows[:rank_n]
    least_pruned = sorted(rows, key=lambda r: r[2])[:rank_n]

    def _print_table(title: str, items):
        print(f"\n-- {title} --")
        print(f"{'rank':>4}  {'layer':>5}  {'node':>5}  {'C0 ratio':>9}  {'C0':>6} {'C1':>4} {'C2':>4} {'C3':>6} {'C4':>6}")
        for rank, (layer_id, idx, c0_ratio, cnt) in enumerate(items, start=1):
            print(f"{rank:>4}  {layer_id:>5}  {idx:>5}  {c0_ratio * 100:>8.2f}%  "
                  f"{cnt[C0]:>6} {cnt[C1]:>4} {cnt[C2]:>4} {cnt[C3]:>6} {cnt[C4]:>6}")

    _print_table(f"가장 많이 프루닝된 노드 top-{rank_n} (거의 죽은 노드 후보)", most_pruned)
    _print_table(f"가장 덜 프루닝된 노드 top-{rank_n}", least_pruned)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="커스텀 바이너리(.bin) 모델 경로")
    parser.add_argument("--eps", type=float, default=EPS, help="threshold epsilon (기본 1e-7)")
    parser.add_argument("--top-n", type=int, default=20, help="C3/C4 좌표 리스트 상위 개수 (기본 20)")
    parser.add_argument("--rank-n", type=int, default=10, help="노드 랭킹 뷰 top-N (기본 10)")
    parser.add_argument("--pct", action="store_true", help="바차트 y축을 개수 대신 비율(%)로 표시")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PNG, help="바차트 PNG 저장 경로")
    parser.add_argument("--layer", type=int, default=None, help="노드 조회: layer id (1, 2, 3)")
    parser.add_argument("--node-idx", type=int, default=None, help="노드 조회: output node 인덱스")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)

    print(f"모델 로드: {args.model}")
    weights = load_weights(args.model)
    stats = build_all_stats(weights, eps=args.eps)

    for s in stats:
        print(f"Layer{s.layer_id}: shape={s.W.shape}, total={s.size}  (sanity check 통과)")

    total = total_counts(stats)
    grand_total = sum(s.size for s in stats)
    print(f"\n전체 weight 수 = {' + '.join(str(s.size) for s in stats)} = {grand_total}")
    for c in range(N_CATEGORIES):
        print(f"  {CATEGORY_LABELS[c]}: {total[c]} ({total[c] / grand_total * 100:.2f}%)")

    print_position_report(stats, top_n=args.top_n)
    plot_distribution(stats, args.output, show_pct=args.pct)
    report_node_ranking(stats, rank_n=args.rank_n)

    if args.layer is not None and args.node_idx is not None:
        report_node(stats, args.layer, args.node_idx)
    elif args.layer is not None or args.node_idx is not None:
        print("\n[알림] 특정 노드 조회를 하려면 --layer와 --node-idx를 함께 지정해야 합니다.")


if __name__ == "__main__":
    main()
