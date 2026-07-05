"""
.nnet → ONNX 변환기

nnet의 입력 정규화(min/max clip, mean/range 정규화)를 ONNX 그래프에
Clip + Sub + Div 노드로 포함시키고, 이후 Gemm(+Relu) 레이어를 이어붙인다.
출력층에는 ReLU를 적용하지 않는다 (nnet 표준과 동일).
"""

import sys
import os
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper


def parse_nnet(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    idx = 0
    while idx < len(lines) and lines[idx].strip().startswith("//"):
        idx += 1

    arch = [int(x) for x in lines[idx].strip().rstrip(',').split(',')]
    num_layers = arch[0]
    idx += 1

    layer_sizes = [int(x) for x in lines[idx].strip().rstrip(',').split(',')]
    idx += 1

    idx += 1  # deprecated 플래그 줄

    input_mins = [float(x) for x in lines[idx].strip().rstrip(',').split(',')]
    idx += 1
    input_maxes = [float(x) for x in lines[idx].strip().rstrip(',').split(',')]
    idx += 1
    means = [float(x) for x in lines[idx].strip().rstrip(',').split(',')]
    idx += 1
    ranges = [float(x) for x in lines[idx].strip().rstrip(',').split(',')]
    idx += 1

    weights = []
    biases = []

    for i in range(num_layers):
        rows = layer_sizes[i + 1]
        cols = layer_sizes[i]

        W = []
        for _ in range(rows):
            row = [float(x) for x in lines[idx].strip().rstrip(',').split(',')]
            W.append(row)
            idx += 1
        weights.append(np.array(W, dtype=np.float32))

        b = []
        for _ in range(rows):
            b.append(float(lines[idx].strip().rstrip(',')))
            idx += 1
        biases.append(np.array(b, dtype=np.float32))

    return layer_sizes, weights, biases, input_mins, input_maxes, means, ranges


def build_onnx(layer_sizes, weights, biases, input_mins, input_maxes, means, ranges, model_name="nnet_model"):
    input_size = layer_sizes[0]
    output_size = layer_sizes[-1]
    num_layers = len(weights)

    nodes = []
    initializers = []

    input_name = "X"
    graph_input = helper.make_tensor_value_info(input_name, TensorProto.FLOAT, [1, input_size])

    # 입력 clip: [min, max]
    min_t = numpy_helper.from_array(np.array(input_mins[:input_size], dtype=np.float32), "input_min")
    max_t = numpy_helper.from_array(np.array(input_maxes[:input_size], dtype=np.float32), "input_max")
    initializers += [min_t, max_t]
    nodes.append(helper.make_node("Max", [input_name, "input_min"], ["clipped_min"]))
    nodes.append(helper.make_node("Min", ["clipped_min", "input_max"], ["clipped"]))

    # 정규화: (x - mean) / range
    mean_t = numpy_helper.from_array(np.array(means[:input_size], dtype=np.float32), "input_mean")
    range_t = numpy_helper.from_array(np.array(ranges[:input_size], dtype=np.float32), "input_range")
    initializers += [mean_t, range_t]
    nodes.append(helper.make_node("Sub", ["clipped", "input_mean"], ["centered"]))
    nodes.append(helper.make_node("Div", ["centered", "input_range"], ["normalized"]))

    prev_output = "normalized"

    for i in range(num_layers):
        W = weights[i]
        b = biases[i]

        # Gemm은 (X @ W^T + b) 형태이므로 nnet의 W(rows=out, cols=in)를 그대로 B로 사용하고 transB=1
        w_name = f"W{i}"
        b_name = f"B{i}"
        initializers.append(numpy_helper.from_array(W, w_name))
        initializers.append(numpy_helper.from_array(b, b_name))

        gemm_out = f"gemm{i}"
        nodes.append(helper.make_node(
            "Gemm", [prev_output, w_name, b_name], [gemm_out],
            alpha=1.0, beta=1.0, transA=0, transB=1
        ))

        if i < num_layers - 1:
            relu_out = f"relu{i}"
            nodes.append(helper.make_node("Relu", [gemm_out], [relu_out]))
            prev_output = relu_out
        else:
            prev_output = gemm_out

    output_name = "Y"
    nodes.append(helper.make_node("Identity", [prev_output], [output_name]))

    graph_output = helper.make_tensor_value_info(output_name, TensorProto.FLOAT, [1, output_size])

    graph = helper.make_graph(
        nodes, model_name, [graph_input], [graph_output], initializer=initializers
    )

    model = helper.make_model(graph, producer_name="NnetToOnnx")
    model.opset_import[0].version = 13
    onnx.checker.check_model(model)
    return model


def convert(input_path, output_path=None):
    if output_path is None:
        script_dir = os.path.dirname(os.path.abspath(input_path))
        base = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(script_dir, base + ".onnx")

    print(f"파싱 중: {input_path}")
    layer_sizes, weights, biases, input_mins, input_maxes, means, ranges = parse_nnet(input_path)

    print(f"레이어 구조: {layer_sizes}")

    model = build_onnx(layer_sizes, weights, biases, input_mins, input_maxes, means, ranges,
                        model_name=os.path.splitext(os.path.basename(input_path))[0])

    onnx.save(model, output_path)
    print(f"변환 완료: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python NnetToOnnx.py <input.nnet> [output.onnx]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) >= 3 else None
    convert(input_path, output_path)
