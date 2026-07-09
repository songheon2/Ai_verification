"""
커스텀 신경망 포맷 - 바이너리 read/write 공용 모듈

FormatConverters/NnetToCustom.py, FormatConverters/OnnxToCustom.py(writer)와
GenericNNEncoding.py(reader)가 공유하는 파일 포맷 정의.

포맷 (리틀엔디안 고정):
  magic   : 4 bytes   b'NNCB'
  version : int32      (=1)
  m       : int32      (가중치 레이어 수)
  sizes   : int32[m+1] (레이어별 노드 수 n_0..n_m)
  이후 m개 블록:
      W_i : float64[n_out, n_in]  (row-major)
      b_i : float64[n_out]

값은 float64로 통일 저장한다. .nnet(텍스트, float64)과 .onnx(주로 float32)
어느 쪽에서 온 값이든 float64에 정확히 담기므로, 기존 텍스트 커스텀 포맷
(유효숫자 6자리, "%.6g")에서 발생하던 반올림 오차 없이 원본 정밀도를 보존한다.
"""

import struct
import numpy as np

MAGIC = b"NNCB"
VERSION = 1


def write_custom(layer_sizes, weights, biases, filepath):
    m = len(weights)
    with open(filepath, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<i", VERSION))
        f.write(struct.pack("<i", m))
        f.write(struct.pack(f"<{m + 1}i", *layer_sizes))
        for W, b in zip(weights, biases):
            np.asarray(W, dtype="<f8").tofile(f)
            np.asarray(b, dtype="<f8").tofile(f)


def read_custom(filepath):
    with open(filepath, "rb") as f:
        magic = f.read(4)
        if magic != MAGIC:
            raise ValueError(f"'{filepath}'는 유효한 커스텀 바이너리 파일이 아닙니다 (magic 불일치)")

        version, = struct.unpack("<i", f.read(4))
        if version != VERSION:
            raise ValueError(f"지원하지 않는 커스텀 바이너리 버전입니다: {version}")

        m, = struct.unpack("<i", f.read(4))
        sizes = list(struct.unpack(f"<{m + 1}i", f.read(4 * (m + 1))))

        weights = []
        biases = []
        for i in range(m):
            n_in, n_out = sizes[i], sizes[i + 1]
            W = np.fromfile(f, dtype="<f8", count=n_out * n_in).reshape(n_out, n_in)
            b = np.fromfile(f, dtype="<f8", count=n_out)
            weights.append(W.tolist())
            biases.append(b.tolist())

    return sizes, weights, biases
