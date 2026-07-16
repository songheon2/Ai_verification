"""
ONNX (FC+ReLU 신경망) → 커스텀 형식 변환기 (바이너리)

사용법
------
    python FormatConverters/OnnxToCustom.py <input.onnx> [output.bin]

    input.onnx  : 변환할 ONNX 모델 경로 (필수)
    output.bin  : 출력 경로 (생략 시 Custom/<input파일명>_custom.bin 에 저장)

예제
----
    python FormatConverters/OnnxToCustom.py "Onnx/ACASXU_experimental_v2a_1_1.onnx"
    → Custom/ACASXU_experimental_v2a_1_1_custom.bin 생성

    python FormatConverters/OnnxToCustom.py "Onnx/model.onnx" "Custom/my_model.bin"
    → 출력 경로 직접 지정

NnetToCustom.py와 동일한 커스텀 바이너리 형식을 사용한다 (정의는 CustomBinary.py 참조).

ONNX 그래프를 입력에서 출력까지 순차적으로 따라가며, Gemm/MatMul/Add/Mul/Sub로
이어지는 연속된 아핀(affine) 연산들을 하나의 FC 레이어로 접고(fold), Relu 등의
비선형 활성화를 만나면 그 지점에서 레이어 경계를 끊는다. 이렇게 하면 순수 Gemm
체인뿐 아니라, BatchNorm이 Mul+Add로 접혀 export된 MatMul+BatchNorm+Relu 체인도
동일하게 FC 레이어로 추출할 수 있다.

입력 정규화(Clip/Max/Min 등 비아핀 전처리 노드)는 첫 아핀 연산 이전까지는
건너뛴다 (nnet의 정규화 블록을 무시하는 NnetToCustom.py와 동일한 취급).
마지막 레이어 뒤에 Relu가 아닌 활성화(예: Sigmoid)가 있으면 이는 무시하고
활성화 적용 전의 값(logit)을 출력 레이어로 저장한다 — 기존 커스텀 형식이
출력층에 활성화를 두지 않는 것과 동일한 관례다.
"""

import sys
import os
import numpy as np
import onnx
from onnx import numpy_helper

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from CustomBinary import write_custom

_PASSTHROUGH_OPS = {"Identity", "Reshape", "Flatten", "Squeeze", "Unsqueeze", "Cast"}
_ACTIVATION_OPS = {"Relu", "Sigmoid", "Tanh", "LeakyRelu", "Elu", "Softmax", "HardSigmoid"}
_INPUT_PREPROCESS_OPS = {"Clip", "Max", "Min", "Add", "Sub", "Mul", "Div"}


def _split_const_and_data(node, current_tensor, initializers, op_name):
    a_name, b_name = node.input[0], node.input[1]
    if a_name == current_tensor and b_name in initializers:
        return initializers[b_name], False  # (const, const_is_first_operand)
    if b_name == current_tensor and a_name in initializers:
        return initializers[a_name], True
    raise ValueError(
        f"{op_name} 노드 '{node.name}'의 입력 중 하나는 현재 텐서('{current_tensor}'), "
        f"다른 하나는 initializer 상수여야 합니다 (입력: {list(node.input)})"
    )


def parse_onnx(filepath):
    model = onnx.load(filepath)
    graph = model.graph

    initializers = {init.name: numpy_helper.to_array(init) for init in graph.initializer}

    consumers = {}
    for n in graph.node:
        for inp in n.input:
            consumers.setdefault(inp, []).append(n)

    # 구형 ONNX는 initializer를 graph.input에도 중복 등록할 수 있다.
    # 실제 데이터 입력에서는 initializer와 이름이 같은 항목을 제외한다.
    data_inputs = [value for value in graph.input if value.name not in initializers]
    if len(data_inputs) != 1:
        raise ValueError(
            f"실제 입력 텐서가 정확히 1개여야 합니다 "
            f"(현재: {[value.name for value in data_inputs]})"
        )
    if len(graph.output) != 1:
        raise ValueError(f"출력 텐서가 정확히 1개여야 합니다 (현재: {[o.name for o in graph.output]})")

    current = data_inputs[0].name
    output_name = graph.output[0].name

    layer_sizes = None
    weights = []
    biases = []

    pending_W = None  # (out, in)
    pending_b = None  # (out,)
    started = False   # 첫 아핀 연산(Gemm/MatMul)을 만났는지

    def flush(reason):
        nonlocal pending_W, pending_b, layer_sizes
        if pending_W is None:
            return
        n_in, n_out = pending_W.shape[1], pending_W.shape[0]
        if layer_sizes is None:
            layer_sizes = [n_in]
        elif layer_sizes[-1] != n_in:
            raise ValueError(
                f"레이어 입력 크기({n_in})가 이전 레이어 출력 크기({layer_sizes[-1]})와 다릅니다 ({reason})"
            )
        layer_sizes.append(n_out)
        weights.append(pending_W)
        biases.append(pending_b)
        pending_W, pending_b = None, None

    while current != output_name:
        outs = consumers.get(current)
        if not outs:
            raise ValueError(f"텐서 '{current}'를 소비하는 노드가 없어 출력까지 도달할 수 없습니다")
        if len(outs) != 1:
            raise ValueError(
                f"텐서 '{current}'를 소비하는 노드가 여러 개입니다 (분기 구조는 지원하지 않음): "
                f"{[n.op_type for n in outs]}"
            )
        node = outs[0]
        op = node.op_type

        if op in _PASSTHROUGH_OPS:
            current = node.output[0]
            continue

        # 커스텀 형식은 첫 번째 FC 레이어가 입력받는 텐서부터 시작한다.
        # 입력 전처리는 AutoVerify의 모델 계약에 별도로 기록하므로 여기서는 건너뛴다.
        if not started and op in _INPUT_PREPROCESS_OPS:
            if current not in node.input:
                raise ValueError(
                    f"input preprocessing node '{node.name}' does not consume "
                    f"the current tensor '{current}'"
                )
            non_current_inputs = [name for name in node.input if name and name != current]
            missing = [name for name in non_current_inputs if name not in initializers]
            if missing:
                raise ValueError(
                    f"input preprocessing node '{node.name}' has non-constant inputs: {missing}"
                )
            current = node.output[0]
            continue

        if op in ("Gemm", "MatMul"):
            if op == "Gemm":
                attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
                if bool(attrs.get("transA", 0)):
                    raise ValueError(f"transA=1인 Gemm은 지원하지 않습니다 ({node.name})")
                if node.input[0] != current:
                    raise ValueError(f"Gemm 노드 '{node.name}'의 첫 입력이 현재 텐서가 아닙니다")
                w_name = node.input[1]
                if w_name not in initializers:
                    raise ValueError(f"Gemm 노드 '{node.name}'의 가중치 '{w_name}'가 initializer에 없습니다")
                K = initializers[w_name].astype(np.float64)
                layer_W = K if bool(attrs.get("transB", 0)) else K.T
                c_name = node.input[2] if len(node.input) > 2 else None
                layer_b = initializers[c_name].astype(np.float64).reshape(-1) if c_name in initializers else np.zeros(layer_W.shape[0])
            else:  # MatMul (바이어스 없음)
                K, const_is_first = _split_const_and_data(node, current, initializers, "MatMul")
                K = K.astype(np.float64)
                layer_W = K if const_is_first else K.T
                layer_b = np.zeros(layer_W.shape[0])

            if pending_W is None:
                pending_W, pending_b = layer_W, layer_b
            else:
                if layer_W.shape[1] != pending_W.shape[0]:
                    raise ValueError(
                        f"'{node.name}' 이전 아핀 연산 출력 크기({pending_W.shape[0]})와 "
                        f"이 연산의 입력 크기({layer_W.shape[1]})가 다릅니다"
                    )
                pending_W = layer_W @ pending_W
                pending_b = layer_W @ pending_b + layer_b
            started = True
            current = node.output[0]
            continue

        if op in ("Add", "Sub", "Mul") and started and pending_W is not None:
            const, const_is_first = _split_const_and_data(node, current, initializers, op)
            v = const.astype(np.float64).reshape(-1)
            if v.shape[0] != pending_W.shape[0]:
                raise ValueError(f"'{node.name}'의 상수 크기({v.shape[0]})가 레이어 출력 크기({pending_W.shape[0]})와 다릅니다")
            if op == "Add":
                pending_b = pending_b + v
            elif op == "Sub":
                if const_is_first:  # v - x
                    pending_W = -pending_W
                    pending_b = v - pending_b
                else:  # x - v
                    pending_b = pending_b - v
            else:  # Mul
                pending_W = v[:, None] * pending_W
                pending_b = v * pending_b
            current = node.output[0]
            continue

        if op in _ACTIVATION_OPS:
            if not started:
                raise ValueError(f"첫 FC 레이어를 만나기 전에 활성화 노드 '{node.name}'({op})를 만났습니다")
            is_last_before_output = (node.output[0] == output_name)
            flush(reason=f"'{node.name}' 앞")
            if op != "Relu":
                if is_last_before_output:
                    print(f"참고: 마지막 활성화 '{op}'는 무시하고 그 이전 logit 값을 출력 레이어로 저장합니다")
                else:
                    print(f"경고: 은닉층 활성화 '{op}'는 지원하지 않는 형식입니다 (ReLU만 지원) — 무시하고 진행합니다")
            current = node.output[0]
            continue

        raise ValueError(f"지원하지 않는 연산 '{op}' (노드: {node.name})를 만났습니다")

    flush(reason="그래프 출력")

    if not weights:
        raise ValueError("그래프에서 FC 레이어를 하나도 추출하지 못했습니다")

    return layer_sizes, weights, biases


def convert(input_path, output_path=None):
    if output_path is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        custom_dir = os.path.join(project_root, "Custom")
        os.makedirs(custom_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(custom_dir, base + "_custom.bin")

    print(f"파싱 중: {input_path}")
    layer_sizes, weights, biases = parse_onnx(input_path)

    print(f"레이어 구조: {layer_sizes}")
    print(f"가중치 레이어 수: {len(weights)}")

    write_custom(layer_sizes, weights, biases, output_path)
    print(f"변환 완료: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python OnnxToCustom.py <input.onnx> [output.bin]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) >= 3 else None
    convert(input_path, output_path)
