"""스모크 테스트용 초소형 y = ReLU(x) ONNX 모델을 생성한다."""

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


OUTPUT_PATH = Path(__file__).resolve().parent / "relu_smoke.onnx"


def create_model(output_path: Path = OUTPUT_PATH) -> None:
    """입력·은닉·출력이 각각 하나인 FC-ReLU-FC 모델을 저장한다."""

    model_input = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1])
    model_output = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 1])

    initializers = [
        numpy_helper.from_array(np.array([[1.0]], dtype=np.float32), "W1"),
        numpy_helper.from_array(np.array([0.0], dtype=np.float32), "B1"),
        numpy_helper.from_array(np.array([[1.0]], dtype=np.float32), "W2"),
        numpy_helper.from_array(np.array([0.0], dtype=np.float32), "B2"),
    ]
    nodes = [
        helper.make_node("Gemm", ["X", "W1", "B1"], ["H"], name="fc1"),
        helper.make_node("Relu", ["H"], ["R"], name="relu1"),
        helper.make_node("Gemm", ["R", "W2", "B2"], ["Y"], name="fc2"),
    ]
    graph = helper.make_graph(
        nodes,
        "relu_smoke_graph",
        [model_input],
        [model_output],
        initializers,
    )
    model = helper.make_model(
        graph,
        producer_name="ai_verification_smoke_test",
        opset_imports=[helper.make_opsetid("", 13)],
    )
    onnx.checker.check_model(model)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output_path)
    print(f"생성 완료: {output_path}")
    print("모델 수식: Y_0 = ReLU(X_0)")


if __name__ == "__main__":
    create_model()
