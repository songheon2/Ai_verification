"""
ACAS Xu Neural Network  —  Prop Encoding Verification  (Method 3)

검증 전략: 뉴런별 dpll_t
  dpll_t 는 ReLU 10개 이상을 동시에 풀지 못하므로
  각 뉴런을 독립적으로 1-ReLU 서브모델로 검증한다.

  Neuron j in layer l:
    phi_j  ∧  (layer-l 입력 변수 고정)  →  dpll_t  →  z_j 비교
  모든 뉴런의 z_j 값이 NumPy 순전파 결과와 일치하면 인코딩 정확.

Usage
-----
  python ACASVerification.py
  python ACASVerification.py --input 10000 0 0 500 500
"""

from __future__ import annotations
import os, sys
import numpy as np
from typing import List

from GenericNNEncoding import load_nn_model, encode_nn, NNModel
from XOREncoding import FreshGen, eq_lin, conj
from DPLL import AndProp
from DPLL_T import dpll_t

# ACAS Xu 입력 정규화 파라미터 (nnet 파일 norm2/norm3)
ACAS_MEAN  = np.array([19791.091, 0.0,           0.0,           650.0, 600.0])
ACAS_RANGE = np.array([60261.0,   6.28318530718, 6.28318530718, 1100.0, 1200.0])

def normalize_input(x: List[float]) -> np.ndarray:
    return (np.array(x, dtype=float) - ACAS_MEAN) / ACAS_RANGE


# ─── NumPy 레이어별 순전파 ────────────────────────────────────────

def numpy_layer_activations(model: NNModel, x: List[float]):
    """activations[i] = i번째 레이어의 입력값 (z, pre-relu 포함)"""
    acts = [np.array(x, dtype=float)]
    z_vals = []
    for i in range(model.num_layers):
        W = np.array(model.weights[i])
        b = np.array(model.biases[i])
        z = W @ acts[-1] + b
        z_vals.append(z)
        h = np.maximum(0.0, z) if i < model.num_layers - 1 else z
        acts.append(h)
    return acts, z_vals   # acts[i]=i번 레이어 입력, z_vals[i]=i번 레이어 pre-activation


# ─── 뉴런별 개별 검증 ────────────────────────────────────────────

def verify_neuron(
    layer_idx: int, neuron_idx: int,
    model: NNModel, in_vars: List[str], np_input: np.ndarray,
    np_z: float, tol: float = 1e-4
) -> bool:
    """
    레이어 layer_idx 의 neuron_idx 번 뉴런 인코딩을 검증.
    1-출력 서브모델로 encode_nn 후 dpll_t → z 값 비교.
    """
    sub = NNModel(
        num_layers=1,
        layer_sizes=[len(in_vars), 1],
        weights=[model.weights[layer_idx][neuron_idx:neuron_idx+1]],
        biases=[model.biases[layer_idx][neuron_idx:neuron_idx+1]],
    )
    phi, out_vars, _ = encode_nn(sub, in_vars, gen=FreshGen(f"l{layer_idx}n{neuron_idx}"))
    input_fix = conj([eq_lin({v: 1.0}, float(val)) for v, val in zip(in_vars, np_input)])
    asgn, sat = dpll_t(AndProp(input_fix, phi), debug=False)

    if not sat:
        return False
    sol = asgn.get(out_vars[0])
    if sol is None:
        return False
    return abs(float(sol) - float(np_z)) <= tol


# ─── 전체 검증 ────────────────────────────────────────────────────

W_LINE = 68
SEP  = "=" * W_LINE
SEP2 = "-" * W_LINE
ACAS_LABELS = ["COC", "WL", "WR", "SL", "SR"]


def verify(model: NNModel, x_input: List[float]) -> bool:
    x_norm = normalize_input(x_input).tolist()
    acts, z_vals = numpy_layer_activations(model, x_norm)

    print(SEP)
    print(f"  Input (raw)  = {x_input}")
    x_norm = normalize_input(x_input).tolist()
    print(f"  Input (norm) = {[round(v,5) for v in x_norm]}")
    print(SEP2)
    print("  [NumPy output logits  (on normalized input)]")
    for val, label in zip(acts[-1], ACAS_LABELS):
        print(f"    {label}: {val:.6f}")
    print()

    all_ok = True

    for li in range(model.num_layers):
        n_in  = model.layer_sizes[li]
        n_out = model.layer_sizes[li + 1]
        is_last = li == model.num_layers - 1
        kind = "Output/Identity" if is_last else "Hidden/ReLU"

        np_input = acts[li]
        in_vars  = [f"L{li+1}_in{k}" for k in range(n_in)]
        np_z     = z_vals[li]

        pass_cnt = 0
        fail_neurons = []

        for j in range(n_out):
            ok = verify_neuron(li, j, model, in_vars, np_input, np_z[j])
            if ok:
                pass_cnt += 1
            else:
                fail_neurons.append(j)

        status = "PASS" if not fail_neurons else f"FAIL (neurons: {fail_neurons[:5]}{'...' if len(fail_neurons)>5 else ''})"
        print(f"  Layer {li+1}  [{kind}]  {n_in} -> {n_out}  "
              f"{pass_cnt}/{n_out}  {status}")

        if fail_neurons:
            all_ok = False

    print()
    print(SEP2)
    result = "PASS - all neurons match" if all_ok else "FAIL - mismatch detected"
    print(f"  Result : {result}")
    print(SEP)
    return all_ok


# ─── 메인 ────────────────────────────────────────────────────────

def main():
    if "--model" in sys.argv:
        idx = sys.argv.index("--model")
        model_path = sys.argv[idx + 1]
    else:
        model_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "Custom", "output.bin"
        )
    model = load_nn_model(model_path)
    print(f"Model: {model.num_layers} layers  "
          f"{' -> '.join(map(str, model.layer_sizes))}")
    print()

    if "--input" in sys.argv:
        idx = sys.argv.index("--input")
        x = [float(v) for v in sys.argv[idx+1: idx+1+model.layer_sizes[0]]]
    else:
        x = [10000.0, 0.0, 0.0, 500.0, 500.0]

    verify(model, x)


if __name__ == "__main__":
    main()
