"""
.nnet → 커스텀 형식 변환기

커스텀 형식:
  Line 1        : m  (가중치 레이어 수)
  Lines 2..m+2  : 각 레이어의 노드 수 n_0, n_1, ..., n_m  (한 줄에 하나)
  이후 m개 블록 :
      n_{i+1} 행 x n_i 열의 가중치 행렬  (행 단위로 공백 구분)
      n_{i+1} 개의 바이어스 값  (한 줄에 공백 구분)
"""

import sys
import os


def parse_nnet(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # 1. 주석 줄 스킵 (//)
    idx = 0
    while idx < len(lines) and lines[idx].strip().startswith("//"):
        idx += 1

    # 2. 아키텍처 요약 줄: numLayers, inputSize, outputSize, maxLayerSize
    arch = [int(x) for x in lines[idx].strip().rstrip(',').split(',')]
    num_layers = arch[0]
    idx += 1

    # 3. 레이어 크기 줄: input, layer1, layer2, ..., output
    layer_sizes = [int(x) for x in lines[idx].strip().rstrip(',').split(',')]
    idx += 1

    # 4. deprecated 플래그 줄 스킵
    idx += 1

    # 5. 정규화 줄 4개 스킵 (min, max, mean, range)
    idx += 4

    # 6. 가중치/바이어스 파싱
    weights = []
    biases = []

    for i in range(num_layers):
        rows = layer_sizes[i + 1]
        cols = layer_sizes[i]

        # 가중치 행렬
        W = []
        for _ in range(rows):
            row = [float(x) for x in lines[idx].strip().rstrip(',').split(',')]
            W.append(row)
            idx += 1
        weights.append(W)

        # 바이어스 벡터 (nnet 형식: 한 줄에 값 하나씩)
        b = []
        for _ in range(rows):
            b.append(float(lines[idx].strip().rstrip(',')))
            idx += 1
        biases.append(b)

    return layer_sizes, weights, biases


def write_custom(layer_sizes, weights, biases, filepath):
    m = len(weights)  # 가중치 레이어 수

    with open(filepath, 'w') as f:
        # Line 1: 레이어 수
        f.write(f"{m}\n")

        # Lines 2..m+2: 각 레이어 노드 수
        for size in layer_sizes:
            f.write(f"{size}\n")

        # m개 블록: 가중치 행렬 + 바이어스
        for W, b in zip(weights, biases):
            for row in W:
                f.write(" ".join(f"{v:.6g}" for v in row) + "\n")
            f.write(" ".join(f"{v:.6g}" for v in b) + "\n")


def convert(input_path, output_path=None):
    if output_path is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        custom_dir = os.path.join(project_root, "Custom")
        os.makedirs(custom_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(custom_dir, base + "_custom.txt")

    print(f"파싱 중: {input_path}")
    layer_sizes, weights, biases = parse_nnet(input_path)

    print(f"레이어 구조: {layer_sizes}")
    print(f"가중치 레이어 수: {len(weights)}")

    write_custom(layer_sizes, weights, biases, output_path)
    print(f"변환 완료: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python nnet_to_custom.py <input.nnet> [output.txt]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) >= 3 else None
    convert(input_path, output_path)