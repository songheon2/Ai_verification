"""자동 검증 CLI를 위한 모델 구조 검사와 안전성 점검 기능."""

from __future__ import annotations

from collections import Counter
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import asdict, dataclass
from io import StringIO
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

AI_VERIFICATION_DIR = Path(__file__).resolve().parent.parent
if str(AI_VERIFICATION_DIR) not in sys.path:
    sys.path.insert(0, str(AI_VERIFICATION_DIR))

from GenericNNEncoding import NNModel, load_nn_model


_PASSTHROUGH_OPS = {"Identity", "Reshape", "Flatten", "Squeeze", "Unsqueeze", "Cast"}
_AFFINE_START_OPS = {"Gemm", "MatMul"}
_AFFINE_FOLD_OPS = {"Add", "Sub", "Mul"}
_ACTIVATION_OPS = {"Relu", "Sigmoid", "Tanh", "LeakyRelu", "Elu", "Softmax", "HardSigmoid"}
_INPUT_PREPROCESS_OPS = {"Clip", "Max", "Min", "Add", "Sub", "Mul", "Div"}


@dataclass
class ModelInfo:
    source_path: str
    source_format: str
    layer_sizes: List[int]
    operation_counts: Dict[str, int]
    output_activation_removed: Optional[str] = None
    input_preprocessing: Optional[Dict[str, Any]] = None
    warnings: List[str] = None

    def __post_init__(self) -> None:
        if self.warnings is None:
            self.warnings = []

    @property
    def input_size(self) -> int:
        return self.layer_sizes[0]

    @property
    def output_size(self) -> int:
        return self.layer_sizes[-1]

    @property
    def relu_count(self) -> int:
        return sum(self.layer_sizes[1:-1])

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result.update(
            input_size=self.input_size,
            output_size=self.output_size,
            relu_count=self.relu_count,
        )
        return result


def _import_onnx():
    try:
        import onnx  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "ONNX support needs the 'onnx' package. Install it with: pip install onnx"
        ) from exc
    return onnx


def _validate_linear_onnx_graph(
    graph: Any, initializers: Mapping[str, Sequence[float]]
) -> Tuple[List[str], Optional[str], List[str], List[Dict[str, Any]]]:
    """GenericNNEncoding이 정확하게 표현할 수 있는 제한된 구조인지 검사한다."""

    # 구형 ONNX는 가중치 initializer를 graph.input에도 중복 등록할 수 있다.
    # initializer와 중복되지 않는 텐서만 실제 외부 입력으로 취급한다.
    data_inputs = [value for value in graph.input if value.name not in initializers]
    if len(data_inputs) != 1:
        raise ValueError(
            "the current converter requires exactly one data input tensor, "
            f"got {len(data_inputs)}: {[value.name for value in data_inputs]}"
        )
    if len(graph.output) != 1:
        raise ValueError(
            f"the current converter requires exactly one output tensor, got {len(graph.output)}"
        )

    consumers: Dict[str, List[Any]] = {}
    for node in graph.node:
        for input_name in node.input:
            consumers.setdefault(input_name, []).append(node)

    current = data_inputs[0].name
    output_name = graph.output[0].name
    operations: List[str] = []
    warnings: List[str] = []
    output_activation: Optional[str] = None
    preprocessing_steps: List[Dict[str, Any]] = []
    pending_affine = False
    saw_affine = False
    visited = set()

    def reaches_output_through_passthrough(tensor_name: str) -> bool:
        probe = tensor_name
        seen = set()
        while probe != output_name:
            next_nodes = consumers.get(probe, [])
            if len(next_nodes) != 1:
                return False
            next_node = next_nodes[0]
            if id(next_node) in seen or next_node.op_type not in _PASSTHROUGH_OPS:
                return False
            seen.add(id(next_node))
            probe = next_node.output[0]
        return True

    while current != output_name:
        nodes = consumers.get(current, [])
        if not nodes:
            raise ValueError(
                f"tensor {current!r} has no consumer on the path to output {output_name!r}"
            )
        if len(nodes) != 1:
            kinds = [node.op_type for node in nodes]
            raise ValueError(
                f"branched graphs are not supported: tensor {current!r} feeds {kinds}"
            )
        node = nodes[0]
        identity = id(node)
        if identity in visited:
            raise ValueError("cycle detected in ONNX graph")
        visited.add(identity)

        operation = node.op_type
        operations.append(operation)
        is_graph_output = reaches_output_through_passthrough(node.output[0])

        if operation in _PASSTHROUGH_OPS:
            current = node.output[0]
            continue

        if operation in _AFFINE_START_OPS:
            pending_affine = True
            saw_affine = True
            current = node.output[0]
            continue

        if not saw_affine and operation in _INPUT_PREPROCESS_OPS:
            if current not in node.input:
                raise ValueError(
                    f"input preprocessing node {node.name!r} does not consume {current!r}"
                )
            constant_inputs = []
            for input_index, input_name in enumerate(node.input):
                if not input_name or input_name == current:
                    continue
                if input_name not in initializers:
                    raise ValueError(
                        f"input preprocessing {operation} has non-constant input "
                        f"{input_name!r}"
                    )
                constant_inputs.append(
                    {
                        "index": input_index,
                        "name": input_name,
                        "values": list(initializers[input_name]),
                    }
                )
            expected_constants = 2 if operation == "Clip" else 1
            if len(constant_inputs) != expected_constants:
                raise ValueError(
                    f"input preprocessing {operation} needs {expected_constants} constant "
                    f"operand(s), got {len(constant_inputs)}"
                )
            preprocessing_steps.append(
                {
                    "op": operation,
                    "data_input_index": list(node.input).index(current),
                    "constants": constant_inputs,
                }
            )
            current = node.output[0]
            continue

        if operation in _AFFINE_FOLD_OPS:
            if not pending_affine:
                raise ValueError(
                    f"{operation} before/without an FC affine block is not supported; "
                    "move input preprocessing to input.normalization in the spec"
                )
            current = node.output[0]
            continue

        if operation in _ACTIVATION_OPS:
            if not pending_affine:
                raise ValueError(
                    f"activation {operation} is not immediately after an affine block"
                )
            pending_affine = False
            if is_graph_output:
                if operation not in {"Sigmoid", "Softmax"}:
                    raise ValueError(
                        f"final activation {operation} cannot be represented by the current "
                        "custom model; only removable Sigmoid/Softmax decisions are supported"
                    )
                output_activation = operation
                warnings.append(
                    f"final {operation} is removed; thresholds and margins are interpreted "
                    "in pre-activation/logit space"
                )
            elif operation != "Relu":
                raise ValueError(
                    f"hidden activation {operation} is unsupported; hidden layers must use ReLU"
                )
            current = node.output[0]
            continue

        if operation == "Div":
            raise ValueError(
                "Div is not supported by OnnxToCustom; describe affine input normalization "
                "with input.normalization in the verification spec"
            )
        raise ValueError(f"unsupported ONNX operation {operation!r} (node {node.name!r})")

    if not saw_affine:
        raise ValueError("no Gemm/MatMul layer was found")
    return operations, output_activation, warnings, preprocessing_steps


def _broadcast(values: Sequence[float], size: int, field: str) -> List[float]:
    numbers = [float(value) for value in values]
    if len(numbers) == 1:
        return numbers * size
    if len(numbers) != size:
        raise ValueError(f"{field} length {len(numbers)} != model input size {size}")
    return numbers


def _interpret_input_preprocessing(
    steps: Sequence[Mapping[str, Any]], input_size: int
) -> Optional[Dict[str, Any]]:
    if not steps:
        return None

    # 원시 입력 공간의 선택적 클리핑 이후 성분별 아핀 변환 y = a*x + b를 추적한다.
    a = [1.0] * input_size
    b = [0.0] * input_size
    lower: List[Optional[float]] = [None] * input_size
    upper: List[Optional[float]] = [None] * input_size
    affine_started = False

    for step_index, step in enumerate(steps):
        operation = str(step["op"])
        constants = list(step["constants"])
        if operation in {"Clip", "Max", "Min"}:
            if affine_started:
                raise ValueError(
                    f"nonlinear input clipping at preprocessing step {step_index} occurs "
                    "after an affine transform and cannot be normalized safely"
                )
            if operation == "Clip":
                min_values = _broadcast(
                    constants[0]["values"], input_size, "Clip min"
                )
                max_values = _broadcast(
                    constants[1]["values"], input_size, "Clip max"
                )
                lower = [
                    value if old is None else max(old, value)
                    for old, value in zip(lower, min_values)
                ]
                upper = [
                    value if old is None else min(old, value)
                    for old, value in zip(upper, max_values)
                ]
            else:
                values = _broadcast(
                    constants[0]["values"], input_size, f"{operation} constant"
                )
                if operation == "Max":
                    lower = [
                        value if old is None else max(old, value)
                        for old, value in zip(lower, values)
                    ]
                else:
                    upper = [
                        value if old is None else min(old, value)
                        for old, value in zip(upper, values)
                    ]
            continue

        affine_started = True
        values = _broadcast(
            constants[0]["values"], input_size, f"{operation} constant"
        )
        data_input_index = int(step["data_input_index"])
        if operation == "Add":
            b = [offset + value for offset, value in zip(b, values)]
        elif operation == "Sub":
            if data_input_index == 0:
                b = [offset - value for offset, value in zip(b, values)]
            else:
                a = [-coefficient for coefficient in a]
                b = [value - offset for value, offset in zip(values, b)]
        elif operation == "Mul":
            a = [coefficient * value for coefficient, value in zip(a, values)]
            b = [offset * value for offset, value in zip(b, values)]
        elif operation == "Div":
            if data_input_index != 0:
                raise ValueError("constant/data input division is not an affine normalization")
            if any(value == 0 for value in values):
                raise ValueError("input preprocessing Div contains a zero divisor")
            a = [coefficient / value for coefficient, value in zip(a, values)]
            b = [offset / value for offset, value in zip(b, values)]
        else:
            raise ValueError(f"unsupported preprocessing operation {operation}")

    for index, (lo, hi) in enumerate(zip(lower, upper)):
        if lo is not None and hi is not None and lo > hi:
            raise ValueError(f"empty ONNX Clip range at input index {index}")
    if any(coefficient == 0 for coefficient in a):
        raise ValueError("input preprocessing collapses an input dimension to a constant")

    preprocessing: Dict[str, Any] = {
        "space": "raw",
        "domain": {"lower": lower, "upper": upper},
    }
    if any(coefficient != 1.0 for coefficient in a) or any(offset != 0.0 for offset in b):
        scale = [1.0 / coefficient for coefficient in a]
        mean = [-offset / coefficient for offset, coefficient in zip(b, a)]
        preprocessing["normalization"] = {"mean": mean, "scale": scale}
    return preprocessing


def inspect_custom(path: str) -> ModelInfo:
    model = load_nn_model(path)
    operation_counts = {
        "Affine": model.num_layers,
        "Relu": sum(model.layer_sizes[1:-1]),
    }
    return ModelInfo(
        source_path=str(Path(path).resolve()),
        source_format="custom",
        layer_sizes=list(model.layer_sizes),
        operation_counts=operation_counts,
        warnings=[
            "custom text files do not retain ONNX activation or preprocessing metadata"
        ],
    )


def inspect_onnx(path: str) -> ModelInfo:
    onnx = _import_onnx()
    from onnx import numpy_helper

    onnx_model = onnx.load(path)
    onnx.checker.check_model(onnx_model)
    initializer_values = {
        initializer.name: numpy_helper.to_array(initializer).reshape(-1).tolist()
        for initializer in onnx_model.graph.initializer
    }
    operations, output_activation, warnings, preprocessing_steps = _validate_linear_onnx_graph(
        onnx_model.graph, initializer_values
    )

    # 위에서 더 엄격한 건전성 검사를 마친 뒤 기존 파서를 재사용한다.
    from FormatConverters.OnnxToCustom import parse_onnx

    with redirect_stdout(StringIO()):
        layer_sizes, _, _ = parse_onnx(path)
    input_preprocessing = _interpret_input_preprocessing(
        preprocessing_steps, int(layer_sizes[0])
    )
    if input_preprocessing:
        warnings.append(
            "ONNX input preprocessing is represented in the generated raw-input contract"
        )
    return ModelInfo(
        source_path=str(Path(path).resolve()),
        source_format="onnx",
        layer_sizes=[int(value) for value in layer_sizes],
        operation_counts=dict(Counter(operations)),
        output_activation_removed=output_activation,
        input_preprocessing=input_preprocessing,
        warnings=warnings,
    )


def inspect_model(path: str) -> ModelInfo:
    suffix = Path(path).suffix.lower()
    if suffix == ".onnx":
        return inspect_onnx(path)
    if suffix in {".txt", ".custom"}:
        return inspect_custom(path)
    raise ValueError("model must be an .onnx, .txt, or .custom file")


def load_model_for_verification(path: str) -> Tuple[NNModel, ModelInfo]:
    info = inspect_model(path)
    if info.source_format == "custom":
        return load_nn_model(path), info

    from FormatConverters.OnnxToCustom import parse_onnx

    with redirect_stdout(StringIO()):
        layer_sizes, weights, biases = parse_onnx(path)
    model = NNModel(
        num_layers=len(weights),
        layer_sizes=[int(value) for value in layer_sizes],
        weights=[matrix.tolist() for matrix in weights],
        biases=[vector.tolist() for vector in biases],
    )
    return model, info


def validate_output_contract(info: ModelInfo, output_spec: Mapping[str, Any]) -> None:
    output_type = str(output_spec.get("type", "unspecified")).lower()
    output_space = str(output_spec.get("space", "logits")).lower()
    if output_space not in {"logits", "pre_activation", "model"}:
        raise ValueError("output.space must be logits, pre_activation, or model")

    activation = info.output_activation_removed
    if activation is None:
        return
    if output_space == "model":
        raise ValueError(
            f"the final {activation} is removed by the encoder, so output.space must be "
            "'logits' or 'pre_activation'"
        )
    if activation == "Sigmoid" and output_type not in {
        "binary",
        "multilabel",
        "multi_label",
    }:
        raise ValueError("a removed final Sigmoid is only valid for binary/multilabel decisions")
    if activation == "Softmax" and output_type != "multiclass":
        raise ValueError("a removed final Softmax is only valid for multiclass decisions")


def make_spec_template(info: ModelInfo) -> Dict[str, Any]:
    output_note = (
        f"The ONNX final {info.output_activation_removed} is removed. Use logit-space "
        "thresholds/margins."
        if info.output_activation_removed
        else "Outputs are the model's final affine values."
    )
    if info.input_preprocessing:
        input_config = deepcopy(info.input_preprocessing)
        input_config["epsilon"] = 0.0
        input_config["_note"] = (
            "Raw bounds/cases are transformed with the preprocessing extracted from ONNX."
        )
    else:
        input_config = {
            "space": "model",
            "domain": {"lower": None, "upper": None},
            "epsilon": 0.0,
            "_note": (
                "For raw inputs, set space='raw' and normalization={mean:[...], "
                "scale:[...]}. Bounds/cases will then be converted automatically."
            ),
        }

    domain = input_config.get("domain", {})
    domain_lower = domain.get("lower") if isinstance(domain, Mapping) else None
    domain_upper = domain.get("upper") if isinstance(domain, Mapping) else None
    default_center = []
    for index in range(info.input_size):
        lo = domain_lower[index] if isinstance(domain_lower, list) else None
        hi = domain_upper[index] if isinstance(domain_upper, list) else None
        if lo is not None and hi is not None:
            default_center.append((float(lo) + float(hi)) / 2.0)
        elif lo is not None:
            default_center.append(float(lo))
        elif hi is not None:
            default_center.append(float(hi))
        else:
            default_center.append(0.0)

    return {
        "schema_version": 1,
        "model_contract": {
            "input_size": info.input_size,
            "output_size": info.output_size,
            "layer_sizes": info.layer_sizes,
        },
        "input": input_config,
        "output": {
            "type": "unspecified",
            "decision": "unspecified",
            "space": "logits" if info.output_activation_removed else "model",
            "threshold": 0.0,
            "margin": 0.0,
            "_note": output_note,
        },
        "property": {
            "reference": "expected",
            "_note": "Use expected for correctness, or center_prediction for label stability.",
        },
        "cases": [
            {
                "name": "edit_me",
                "center": default_center,
                "epsilon": 0.0,
                "expected": None,
            }
        ],
        "solver": {
            "max_rounds": 1000,
            "timeout_seconds": 300.0,
            "max_relus_without_override": 50,
        },
        "_inspection": info.to_dict(),
    }


def write_spec_template(info: ModelInfo, output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(make_spec_template(info), handle, ensure_ascii=False, indent=2)
        handle.write("\n")
