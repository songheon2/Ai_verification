from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from DPLL import Prop, ReLUProp
from XOREncoding import conj, eq_lin, FreshGen


# ============================================================
# 데이터 구조
# ============================================================

@dataclass
class NNModel:
    """
    파일에서 파싱한 완전연결 신경망 모델.

    num_layers  : 가중치 레이어 수 m
    layer_sizes : 각 레이어의 노드 수 [n_0, n_1, ..., n_m]
    weights     : weights[i][j][k]  = W_{i+1}[j, k]  (레이어 i+1, 출력 뉴런 j, 입력 k)
    biases      : biases[i][j]      = b_{i+1}[j]
    """
    num_layers: int
    layer_sizes: List[int]
    weights: List[List[List[float]]]
    biases: List[List[float]]


# ============================================================
# 모델 파일 파싱
# ============================================================

def load_nn_model(filepath: str) -> NNModel:
    """
    표준 텍스트 형식의 신경망 파일을 읽어 NNModel로 반환한다.

    파일 형식
    ---------
    Line 1        : m  (가중치 레이어 수)
    Lines 2..m+2  : 각 레이어의 노드 수 n_0, n_1, ..., n_m  (한 줄에 하나)
    이후 m개 블록 :
        n_{i+1} 행 x n_i 열의 가중치 행렬  (행 단위로 공백 구분)
        n_{i+1} 개의 바이어스 값  (한 줄에 공백 구분)

    토큰 단위로 파싱하므로 줄바꿈 위치는 유연하게 허용된다.
    """
    with open(filepath, "r") as f:
        tokens = f.read().split()

    pos = 0

    def read_int() -> int:
        nonlocal pos
        v = int(tokens[pos])
        pos += 1
        return v

    def read_float() -> float:
        nonlocal pos
        v = float(tokens[pos])
        pos += 1
        return v

    m = read_int()
    sizes = [read_int() for _ in range(m + 1)]

    weights: List[List[List[float]]] = []
    biases: List[List[float]] = []

    for i in range(m):
        n_in, n_out = sizes[i], sizes[i + 1]
        W = [[read_float() for _ in range(n_in)] for _ in range(n_out)]
        b = [read_float() for _ in range(n_out)]
        weights.append(W)
        biases.append(b)

    return NNModel(num_layers=m, layer_sizes=sizes, weights=weights, biases=biases)


# ============================================================
# 단일 입력 경로 인코딩
# ============================================================

def encode_nn(
    model: NNModel,
    input_vars: List[str],
    gen: Optional[FreshGen] = None,
) -> Tuple[Prop, List[str], Dict[str, List[str]]]:
    """
    완전연결 신경망을 Prop 제약식의 conjunction으로 인코딩한다.

    활성화 함수
    -----------
    - 은닉층 : ReLU  →  pre-activation z, post-activation h = relu(z)
    - 출력층 : 항등 (logit)  →  sigmoid(logit) > 0.5  ⟺  logit > 0

    Parameters
    ----------
    model      : load_nn_model로 불러온 NNModel
    input_vars : 입력 뉴런 변수 이름 리스트 (길이 == layer_sizes[0])
    gen        : 보조 변수 이름 생성기; None이면 내부에서 생성

    Returns
    -------
    phi         : Prop  — 네트워크 전체 제약식의 conjunction
    output_vars : 출력 logit 변수 이름 리스트 (길이 == layer_sizes[-1])
    aux         : {"L1": [...], "L2": [...], ...}  — 레이어별 보조 변수 이름
    """
    if len(input_vars) != model.layer_sizes[0]:
        raise ValueError(
            f"input_vars 길이 {len(input_vars)} != 모델 입력 크기 {model.layer_sizes[0]}"
        )

    if gen is None:
        gen = FreshGen("nn")

    constraints: List[Prop] = []
    aux: Dict[str, List[str]] = {}
    cur_vars = list(input_vars)

    for li in range(model.num_layers):
        W = model.weights[li]
        b = model.biases[li]
        n_out = model.layer_sizes[li + 1]
        is_last = (li == model.num_layers - 1)
        layer_out: List[str] = []

        for j in range(n_out):
            z = gen.fresh(f"z{li+1}_{j}")

            # affine: z == sum_k W[j][k] * cur_vars[k] + b[j]
            # eq_lin 형식으로:  z - sum_k W[j][k]*cur_vars[k] == b[j]
            terms: Dict[str, float] = {z: 1.0}
            for k, xk in enumerate(cur_vars):
                terms[xk] = terms.get(xk, 0.0) - W[j][k]
            constraints.append(eq_lin(terms, b[j]))

            if is_last:
                # 출력층: logit 변수만 저장
                layer_out.append(z)
            else:
                # 은닉층: ReLU 제약 추가
                h = gen.fresh(f"h{li+1}_{j}")
                constraints.append(ReLUProp(x=z, y=h))
                layer_out.append(h)

        aux[f"L{li+1}"] = layer_out
        cur_vars = layer_out

    return conj(constraints), cur_vars, aux


# ============================================================
# 테스트
# ============================================================

def main():
    import os
    import platform
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator
    from tqdm import tqdm
    from DPLL import AndProp, NotProp
    from DPLL_T import dpll_t
    from Robustness import make_precondition_linf_box
    from PreciseEncoding import out_zero_logit, out_one_logit, xor_nn_sx

    if platform.system() == 'Windows':
        plt.rcParams['font.family'] = 'Malgun Gothic'
    elif platform.system() == 'Darwin':
        plt.rcParams['font.family'] = 'AppleGothic'
    else:
        plt.rcParams['font.family'] = 'NanumGothic'
    plt.rcParams['axes.unicode_minus'] = False

    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "xor_network.txt")
    model = load_nn_model(model_path)
    print(f"모델 로드 완료: 가중치 레이어 {model.num_layers}개, 크기 {model.layer_sizes}")

    eps_values = np.arange(0.0, 0.505, 0.005)
    sat_counts = []
    milestones = {}
    prev_sat_count = 0

    xor_cases = [
        ((0.0, 0.0), 0), ((0.0, 1.0), 1),
        ((1.0, 0.0), 1), ((1.0, 1.0), 0)
    ]

    print("=== 구간별 다중 반례 수집 Sweep 시작 ===")

    for eps in tqdm(eps_values, desc="검증 진행 중"):
        current_sat = 0
        current_cexs = []

        for (c1, c2), expected in xor_cases:
            fg = FreshGen(prefix=f"adv_eps{int(eps*1000)}_")
            x_vars = ["x0", "x1"]

            pre = make_precondition_linf_box(x_vars, (c1, c2), eps=eps, clamp_01=False)
            nn_prop, out_vars, _ = encode_nn(model, x_vars, gen=fg)
            s_var = out_vars[0]

            post = out_zero_logit(s_var) if expected == 0 else out_one_logit(s_var)
            phi = AndProp(pre, AndProp(nn_prop, NotProp(post)))

            assignment, sat = dpll_t(phi, debug=False)

            if sat:
                current_sat += 1
                current_cexs.append({
                    'case': (c1, c2),
                    'expected': expected,
                    'x0': assignment.get('x0'),
                    'x1': assignment.get('x1')
                })

        sat_counts.append(current_sat)

        if current_sat > prev_sat_count:
            milestones[current_sat] = {'eps': eps, 'cexs': current_cexs}
            prev_sat_count = current_sat

    # 그래프
    plt.figure(figsize=(12, 7))
    plt.plot(eps_values, sat_counts, marker='o', color='royalblue',
             drawstyle='steps-post', label='반례 발생 케이스 수 (SAT)')

    ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(0.05))
    ax.xaxis.set_minor_locator(MultipleLocator(0.01))

    plt.grid(True, which='both', linestyle='--', alpha=0.4)
    plt.axvline(x=0.5, color='orange', linestyle='--', label=r'이론적 경계 ($\epsilon=0.5$)')
    plt.title('Epsilon 변화에 따른 XOR 신경망 강건성 분석 (구간별 반례 추적)')
    plt.xlabel(r'섭동 범위 ($\epsilon$)')
    plt.ylabel('SAT 발생 수 (4개 케이스 중)')
    plt.ylim(-0.2, 4.2)
    plt.legend()
    plt.show(block=False)
    plt.pause(1)

    if milestones:
        print("\n\n" + "="*70)
        print("[자동 검산] 각 고장 구간(Jump)별 발견된 모든 반례 정밀 검증")
        print("="*70)

        for sat_count in sorted(milestones.keys()):
            data = milestones[sat_count]
            eps = data['eps']
            cexs = data['cexs']

            print(f"\n[새로운 고장 구간 진입!] 총 {sat_count}개 케이스 고장 (epsilon = {eps:.3f})")
            print("-" * 70)

            for i, cex in enumerate(cexs, start=1):
                c1, c2 = cex['case']
                expected = cex['expected']
                x0_cex = cex['x0']
                x1_cex = cex['x1']

                print(f"  [{i}/{len(cexs)}] 케이스: ({c1}, {c2}) -> 기대값 {expected}")
                print(f"     솔버가 찾은 입력 : x0 = {x0_cex:.8f}, x1 = {x1_cex:.8f}")

                logit_val = xor_nn_sx(x0_cex, x1_cex)
                prediction = 1 if logit_val >= 0 else 0

                if prediction != expected:
                    print(f"     기본 검증 완료 : Logit = {logit_val:.8f} (오답 발생 확인!)")
                else:
                    print(f"     수치 오차 감지 (Logit = {logit_val:.8f}). 미세 밀어내기 적용 중...")
                    shift = 0.001
                    x0_pushed = x0_cex + (shift if x0_cex > c1 else -shift)
                    x1_pushed = x1_cex + (shift if x1_cex > c2 else -shift)
                    logit_pushed = xor_nn_sx(x0_pushed, x1_pushed)
                    pred_pushed = 1 if logit_pushed >= 0 else 0

                    if pred_pushed != expected:
                        print(f"     밀어내기 성공 : x0={x0_pushed:.5f}, x1={x1_pushed:.5f}")
                        print(f"                     Logit = {logit_pushed:.8f} (오답 확인!)")
                    else:
                        print(f"     밀어내기 실패 : 더 강한 밀어내기가 필요합니다.")
                print("")
    else:
        print("\n반례가 발견되지 않았습니다.")

    plt.show()


if __name__ == "__main__":
    main()
