"""설정 파일 기반 신경망 검증 실행기.

사용 예시
---------
  python AutoVerify.py inspect Onnx/model.onnx --output Specs/model.json
  python AutoVerify.py verify --model Onnx/model.onnx --spec Specs/model.json
  python AutoVerify.py verify --model Custom/model.txt --spec Specs/model.json --dry-run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

AI_VERIFICATION_DIR = Path(__file__).resolve().parent.parent
if str(AI_VERIFICATION_DIR) not in sys.path:
    sys.path.insert(0, str(AI_VERIFICATION_DIR))

from DPLL import AndProp, NotProp
from DPLL_T import dpll_t_detailed
from GenericNNEncoding import NNModel, encode_nn
from Automation.ModelInspector import (
    ModelInfo,
    inspect_model,
    load_model_for_verification,
    validate_output_contract,
    write_spec_template,
)
from Automation.PropertyBuilder import (
    build_postcondition,
    build_precondition,
    infer_expected_from_outputs,
    transform_values_to_model_space,
)
from Automation.VnnlibParser import LinearExpression, parse_vnnlib_file
from XOREncoding import FreshGen
from Automation.SolverStatus import SolverStatus


def load_spec(path: str) -> Dict[str, Any]:
    suffix = Path(path).suffix.lower()
    with open(path, "r", encoding="utf-8") as handle:
        if suffix == ".json":
            data = json.load(handle)
        elif suffix in {".yaml", ".yml"}:
            try:
                import yaml  # type: ignore
            except ImportError as exc:
                raise RuntimeError(
                    "YAML specs need PyYAML. Use JSON or install it with: pip install pyyaml"
                ) from exc
            data = yaml.safe_load(handle)
        else:
            raise ValueError("verification spec must be .json, .yaml, or .yml")
    if not isinstance(data, dict):
        raise ValueError("verification spec root must be an object")
    return data


def forward_model(model: NNModel, inputs: Sequence[float]) -> List[float]:
    if len(inputs) != model.layer_sizes[0]:
        raise ValueError(
            f"input length {len(inputs)} != model input size {model.layer_sizes[0]}"
        )
    values = [float(value) for value in inputs]
    for layer_index, (weights, biases) in enumerate(zip(model.weights, model.biases)):
        next_values = []
        for row, bias in zip(weights, biases):
            value = sum(float(w) * x for w, x in zip(row, values)) + float(bias)
            next_values.append(value)
        if layer_index < model.num_layers - 1:
            next_values = [max(0.0, value) for value in next_values]
        values = next_values
    return values


def _declared_space_values(
    model_values: Sequence[float], input_spec: Mapping[str, Any]
) -> List[float]:
    if str(input_spec.get("space", "model")).lower() != "raw":
        return [float(value) for value in model_values]
    normalization = input_spec.get("normalization")
    if not isinstance(normalization, Mapping):
        return [float(value) for value in model_values]
    mean = normalization.get("mean")
    scale = normalization.get("scale")
    if isinstance(mean, (int, float)):
        mean = [mean] * len(model_values)
    if isinstance(scale, (int, float)):
        scale = [scale] * len(model_values)
    if not isinstance(mean, Sequence) or isinstance(mean, (str, bytes)):
        raise ValueError("input.normalization.mean must be a number or list")
    if not isinstance(scale, Sequence) or isinstance(scale, (str, bytes)):
        raise ValueError("input.normalization.scale must be a number or list")
    if len(mean) != len(model_values) or len(scale) != len(model_values):
        raise ValueError("normalization arrays do not match model input size")
    return [
        float(value) * float(divisor) + float(mu)
        for value, mu, divisor in zip(model_values, mean, scale)
    ]


def _validate_contract(spec: Mapping[str, Any], info: ModelInfo) -> None:
    contract = spec.get("model_contract", {}) or {}
    if not isinstance(contract, Mapping):
        raise ValueError("model_contract must be an object")
    expected_input = contract.get("input_size")
    expected_output = contract.get("output_size")
    if expected_input is not None and int(expected_input) != info.input_size:
        raise ValueError(
            f"model input size {info.input_size} != spec contract {expected_input}"
        )
    if expected_output is not None and int(expected_output) != info.output_size:
        raise ValueError(
            f"model output size {info.output_size} != spec contract {expected_output}"
        )


def _input_names(input_spec: Mapping[str, Any], size: int) -> List[str]:
    names = input_spec.get("names")
    if names is None:
        return [f"x{index}" for index in range(size)]
    if not isinstance(names, Sequence) or isinstance(names, (str, bytes)):
        raise ValueError("input.names must be a list")
    result = [str(name) for name in names]
    if len(result) != size:
        raise ValueError(f"input.names length {len(result)} != model input size {size}")
    if len(set(result)) != len(result):
        raise ValueError("input.names must be unique")
    return result


def _center_expected(
    model: NNModel,
    input_spec: Mapping[str, Any],
    output_spec: Mapping[str, Any],
    case: Mapping[str, Any],
) -> Tuple[Any, List[float]]:
    center = case.get("center")
    if not isinstance(center, Sequence) or isinstance(center, (str, bytes)):
        raise ValueError("center_prediction reference requires case.center")
    model_center = transform_values_to_model_space(center, input_spec)
    outputs = forward_model(model, model_center)
    return infer_expected_from_outputs(output_spec, outputs), outputs


def _counterexample(
    assignment: Mapping[str, float],
    input_vars: Sequence[str],
    output_vars: Sequence[str],
    input_spec: Mapping[str, Any],
    model: NNModel,
) -> Dict[str, Any]:
    model_inputs: List[Optional[float]] = [
        float(assignment[name]) if name in assignment else None for name in input_vars
    ]
    solver_outputs: List[Optional[float]] = [
        float(assignment[name]) if name in assignment else None for name in output_vars
    ]
    complete_inputs = all(value is not None for value in model_inputs)
    declared_inputs = None
    recomputed_outputs = None
    if complete_inputs:
        concrete_inputs = [float(value) for value in model_inputs if value is not None]
        declared_inputs = _declared_space_values(concrete_inputs, input_spec)
        recomputed_outputs = forward_model(model, concrete_inputs)
    return {
        "input_model_space": model_inputs,
        "input_declared_space": declared_inputs,
        "solver_outputs": solver_outputs,
        "recomputed_outputs": recomputed_outputs,
    }


def run_verification(
    model_path: str,
    spec: Mapping[str, Any],
    *,
    dry_run: bool = False,
    allow_large_model: bool = False,
    debug: bool = False,
    timeout_seconds_override: Optional[float] = None,
) -> Dict[str, Any]:
    model, info = load_model_for_verification(model_path)
    _validate_contract(spec, info)

    input_spec = spec.get("input", {}) or {}
    output_spec = spec.get("output", {}) or {}
    property_spec = spec.get("property", {}) or {}
    solver_spec = spec.get("solver", {}) or {}
    cases = spec.get("cases")
    if not isinstance(input_spec, Mapping):
        raise ValueError("input must be an object")
    if not isinstance(output_spec, Mapping):
        raise ValueError("output must be an object")
    if not isinstance(property_spec, Mapping):
        raise ValueError("property must be an object")
    if not isinstance(solver_spec, Mapping):
        raise ValueError("solver must be an object")
    if not isinstance(cases, list) or not cases:
        raise ValueError("cases must be a non-empty list")

    validate_output_contract(info, output_spec)
    input_vars = _input_names(input_spec, info.input_size)
    max_rounds = int(solver_spec.get("max_rounds", 1000))
    if max_rounds <= 0:
        raise ValueError("solver.max_rounds must be positive")
    timeout_value = (
        timeout_seconds_override
        if timeout_seconds_override is not None
        else solver_spec.get("timeout_seconds", 300.0)
    )
    timeout_seconds = None if timeout_value is None else float(timeout_value)
    if timeout_seconds is not None and timeout_seconds <= 0:
        raise ValueError("solver.timeout_seconds must be positive or null")
    safe_relu_limit = int(solver_spec.get("max_relus_without_override", 50))
    if (
        not dry_run
        and not allow_large_model
        and info.relu_count > safe_relu_limit
    ):
        raise RuntimeError(
            f"model has {info.relu_count} hidden ReLUs, above the configured automatic "
            f"limit {safe_relu_limit}. Use --dry-run first, then --allow-large-model only "
            "if the expected cost is acceptable."
        )

    results: List[Dict[str, Any]] = []
    for case_index, raw_case in enumerate(cases):
        if not isinstance(raw_case, Mapping):
            raise ValueError(f"cases[{case_index}] must be an object")
        case = dict(raw_case)
        case_name = str(case.get("name", f"case_{case_index}"))
        precondition, lower, upper = build_precondition(
            input_vars, input_spec, case
        )

        generator = FreshGen(prefix=f"auto{case_index}_")
        nn_property, output_vars, _ = encode_nn(model, input_vars, generator)

        reference = str(
            case.get("reference", property_spec.get("reference", "expected"))
        ).lower()
        center_outputs = None
        if reference == "expected":
            expected = case.get("expected")
        elif reference == "center_prediction":
            expected, center_outputs = _center_expected(
                model, input_spec, output_spec, case
            )
        else:
            raise ValueError(
                f"cases[{case_index}] reference must be expected or center_prediction"
            )

        postcondition = build_postcondition(output_vars, output_spec, expected)
        negated_property = AndProp(
            precondition,
            AndProp(nn_property, NotProp(postcondition)),
        )

        result: Dict[str, Any] = {
            "name": case_name,
            "reference": reference,
            "expected": expected,
            "effective_model_bounds": {"lower": lower, "upper": upper},
            "center_outputs": center_outputs,
        }
        if dry_run:
            result["status"] = "DRY_RUN"
        else:
            solver_result = dpll_t_detailed(
                negated_property,
                max_rounds=max_rounds,
                debug=debug,
                timeout_seconds=timeout_seconds,
            )
            result["solver"] = {
                **solver_result.to_dict(),
                "max_rounds": max_rounds,
                "timeout_seconds": timeout_seconds,
            }
            if solver_result.status == SolverStatus.SAT:
                result["status"] = "COUNTEREXAMPLE"
                result["counterexample"] = _counterexample(
                    solver_result.model or {},
                    input_vars,
                    output_vars,
                    input_spec,
                    model,
                )
            elif solver_result.status == SolverStatus.UNSAT:
                result["status"] = "VERIFIED"
                result["note"] = "반례 탐색식이 UNSAT으로 확정되었습니다."
            else:
                result["status"] = "UNKNOWN"
                result["note"] = (
                    "자원 한도 때문에 SAT/UNSAT을 확정하지 못했습니다: "
                    f"{solver_result.reason}"
                )
        results.append(result)

    return {
        "model": info.to_dict(),
        "dry_run": dry_run,
        "warning": (
            "center_prediction proves label stability, not ground-truth correctness"
            if any(result["reference"] == "center_prediction" for result in results)
            else None
        ),
        "cases": results,
    }


def _vnnlib_input_spec(info: ModelInfo) -> Dict[str, Any]:
    """반례 입력을 ONNX 외부 입력 공간으로 되돌리기 위한 정보를 만든다."""

    if info.input_preprocessing:
        return dict(info.input_preprocessing)
    return {"space": "model"}


def _vector(value: Any, size: int, default: float, field: str) -> List[float]:
    if value is None:
        return [default] * size
    if isinstance(value, (int, float)):
        return [float(value)] * size
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{field} must be a number or a list")
    result = [float(item) for item in value]
    if len(result) != size:
        raise ValueError(f"{field} length {len(result)} != expected {size}")
    return result


def _vnnlib_substitutions(
    info: ModelInfo,
    input_vars: Sequence[str],
    output_vars: Sequence[str],
) -> Dict[str, Any]:
    """VNNLIB의 X/Y 변수를 내부 신경망 변수의 선형식으로 치환한다."""

    input_spec = _vnnlib_input_spec(info)
    normalization = input_spec.get("normalization")
    if isinstance(normalization, Mapping):
        mean = _vector(
            normalization.get("mean"), len(input_vars), 0.0, "normalization.mean"
        )
        scale = _vector(
            normalization.get("scale"), len(input_vars), 1.0, "normalization.scale"
        )
    else:
        mean = [0.0] * len(input_vars)
        scale = [1.0] * len(input_vars)

    substitutions: Dict[str, Any] = {}
    for index, (name, mu, divisor) in enumerate(zip(input_vars, mean, scale)):
        if divisor == 0.0:
            raise ValueError(f"normalization.scale[{index}] must be non-zero")
        # 내부 입력 z=(raw-mean)/scale이므로 VNNLIB의 raw 입력은 scale*z+mean이다.
        substitutions[f"X_{index}"] = LinearExpression({name: divisor}, mu)
    for index, name in enumerate(output_vars):
        substitutions[f"Y_{index}"] = name
    return substitutions


def run_vnnlib_verification(
    model_path: str,
    vnnlib_path: str,
    *,
    dry_run: bool = False,
    allow_large_model: bool = False,
    debug: bool = False,
    max_rounds: int = 1000,
    max_relus_without_override: int = 50,
    strict_epsilon: float = 1e-6,
    timeout_seconds: Optional[float] = 300.0,
) -> Dict[str, Any]:
    """VNNLIB이 정의한 unsafe 영역과 신경망의 교집합을 직접 탐색한다."""

    model, info = load_model_for_verification(model_path)
    if info.output_activation_removed:
        raise ValueError(
            "VNNLIB direct verification does not support a removed final activation: "
            f"{info.output_activation_removed}"
        )
    if max_rounds <= 0:
        raise ValueError("max_rounds must be positive")
    if timeout_seconds is not None and timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive or null")
    if max_relus_without_override < 0:
        raise ValueError("max_relus_without_override must be non-negative")
    if (
        not dry_run
        and not allow_large_model
        and info.relu_count > max_relus_without_override
    ):
        raise RuntimeError(
            f"model has {info.relu_count} hidden ReLUs, above the automatic limit "
            f"{max_relus_without_override}. Use --dry-run first, then "
            "--allow-large-model only if the expected cost is acceptable."
        )

    input_vars = [f"x{index}" for index in range(info.input_size)]
    generator = FreshGen(prefix="vnnlib_")
    nn_property, output_vars, _ = encode_nn(model, input_vars, generator)
    substitutions = _vnnlib_substitutions(info, input_vars, output_vars)
    document = parse_vnnlib_file(
        vnnlib_path,
        substitutions,
        strict_epsilon=strict_epsilon,
    )

    if len(document.input_variables) != info.input_size:
        raise ValueError(
            f"VNNLIB declares {len(document.input_variables)} inputs, "
            f"model expects {info.input_size}"
        )
    if len(document.output_variables) != info.output_size:
        raise ValueError(
            f"VNNLIB declares {len(document.output_variables)} outputs, "
            f"model produces {info.output_size}"
        )

    result: Dict[str, Any] = {
        "model": info.to_dict(),
        "vnnlib": {
            "path": document.source_path,
            "assertion_count": document.assertion_count,
            "input_variables": list(document.input_variables),
            "output_variables": list(document.output_variables),
        },
        "dry_run": dry_run,
    }
    if dry_run:
        result["status"] = "DRY_RUN"
        return result

    # VNNLIB assertion은 unsafe 집합을 나타내므로 SAT이면 반례가 존재한다.
    query = AndProp(nn_property, document.formula)
    solver_result = dpll_t_detailed(
        query,
        max_rounds=max_rounds,
        debug=debug,
        timeout_seconds=timeout_seconds,
    )
    result["solver"] = {
        **solver_result.to_dict(),
        "max_rounds": max_rounds,
        "timeout_seconds": timeout_seconds,
    }
    if solver_result.status == SolverStatus.SAT:
        result["status"] = "COUNTEREXAMPLE"
        result["counterexample"] = _counterexample(
            solver_result.model or {},
            input_vars,
            output_vars,
            _vnnlib_input_spec(info),
            model,
        )
    elif solver_result.status == SolverStatus.UNSAT:
        result["status"] = "VERIFIED"
        result["note"] = "VNNLIB unsafe 영역과 신경망의 교집합이 UNSAT입니다."
    else:
        result["status"] = "UNKNOWN"
        result["note"] = (
            "자원 한도 때문에 SAT/UNSAT을 확정하지 못했습니다: "
            f"{solver_result.reason}"
        )
    return result


def _print_inspection(info: ModelInfo) -> None:
    print(f"Model format : {info.source_format}")
    print(f"Layers       : {' -> '.join(map(str, info.layer_sizes))}")
    print(f"Hidden ReLUs: {info.relu_count}")
    if info.output_activation_removed:
        print(f"Final output : pre-{info.output_activation_removed} values (logits)")
    for warning in info.warnings:
        print(f"Warning      : {warning}")


def _print_verification(result: Mapping[str, Any]) -> None:
    model = result["model"]
    print(f"Model: {' -> '.join(map(str, model['layer_sizes']))}")
    if result.get("warning"):
        print(f"Warning: {result['warning']}")
    for case in result["cases"]:
        print(f"[{case['name']}] {case['status']} | expected={case['expected']}")
        counterexample = case.get("counterexample")
        if counterexample:
            print(f"  input : {counterexample['input_declared_space']}")
            print(f"  output: {counterexample['recomputed_outputs']}")
        if case.get("note"):
            print(f"  note  : {case['note']}")
        solver = case.get("solver")
        if solver:
            print(
                f"  solver: {solver['status']} | reason={solver['reason']} | "
                f"rounds={solver['rounds']} | elapsed={solver['elapsed_seconds']:.3f}s"
            )


def _print_vnnlib_verification(result: Mapping[str, Any]) -> None:
    model = result["model"]
    vnnlib = result["vnnlib"]
    print(f"Model  : {' -> '.join(map(str, model['layer_sizes']))}")
    print(f"VNNLIB : {vnnlib['path']}")
    print(f"Asserts: {vnnlib['assertion_count']}")
    print(f"Result : {result['status']}")
    counterexample = result.get("counterexample")
    if counterexample:
        print(f"  input : {counterexample['input_declared_space']}")
        print(f"  output: {counterexample['recomputed_outputs']}")
    if result.get("note"):
        print(f"  note  : {result['note']}")
    solver = result.get("solver")
    if solver:
        print(
            f"  solver: {solver['status']} | reason={solver['reason']} | "
            f"rounds={solver['rounds']} | elapsed={solver['elapsed_seconds']:.3f}s"
        )


def _write_json_result(result: Mapping[str, Any], output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect and verify supported FC+ReLU ONNX/custom models"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser(
        "inspect", help="inspect a model and optionally generate a spec template"
    )
    inspect_parser.add_argument("model")
    inspect_parser.add_argument("--output", help="write a JSON spec template")

    verify_parser = subparsers.add_parser("verify", help="run configured properties")
    verify_parser.add_argument("--model", help="ONNX or custom text model")
    verify_parser.add_argument("--spec", required=True, help="JSON or YAML property spec")
    verify_parser.add_argument(
        "--dry-run", action="store_true", help="validate and encode without solving"
    )
    verify_parser.add_argument(
        "--allow-large-model",
        action="store_true",
        help="override the configured hidden-ReLU safety limit",
    )
    verify_parser.add_argument("--debug", action="store_true")
    verify_parser.add_argument(
        "--timeout-seconds",
        type=float,
        help="override solver.timeout_seconds from the spec",
    )
    verify_parser.add_argument("--json-output", help="write detailed results as JSON")

    vnnlib_parser = subparsers.add_parser(
        "verify-vnnlib", help="run an ONNX model against a VNNLIB property"
    )
    vnnlib_parser.add_argument("--model", required=True, help="ONNX or custom model")
    vnnlib_parser.add_argument(
        "--vnnlib", required=True, help=".vnnlib or .vnnlib.gz property file"
    )
    vnnlib_parser.add_argument("--dry-run", action="store_true")
    vnnlib_parser.add_argument("--allow-large-model", action="store_true")
    vnnlib_parser.add_argument("--debug", action="store_true")
    vnnlib_parser.add_argument("--max-rounds", type=int, default=1000)
    vnnlib_parser.add_argument(
        "--max-relus-without-override", type=int, default=50
    )
    vnnlib_parser.add_argument("--strict-epsilon", type=float, default=1e-6)
    vnnlib_parser.add_argument("--timeout-seconds", type=float, default=300.0)
    vnnlib_parser.add_argument("--json-output")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        if args.command == "inspect":
            info = inspect_model(args.model)
            _print_inspection(info)
            if args.output:
                output_path = Path(args.output)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                write_spec_template(info, str(output_path))
                print(f"Spec template: {output_path}")
            return 0

        if args.command == "verify-vnnlib":
            result = run_vnnlib_verification(
                args.model,
                args.vnnlib,
                dry_run=args.dry_run,
                allow_large_model=args.allow_large_model,
                debug=args.debug,
                max_rounds=args.max_rounds,
                max_relus_without_override=args.max_relus_without_override,
                strict_epsilon=args.strict_epsilon,
                timeout_seconds=args.timeout_seconds,
            )
            _print_vnnlib_verification(result)
            if args.json_output:
                _write_json_result(result, args.json_output)
            return 0

        spec = load_spec(args.spec)
        model_path = args.model or spec.get("model")
        if not model_path:
            raise ValueError("provide --model or set model in the spec")
        result = run_verification(
            str(model_path),
            spec,
            dry_run=args.dry_run,
            allow_large_model=args.allow_large_model,
            debug=args.debug,
            timeout_seconds_override=args.timeout_seconds,
        )
        _print_verification(result)
        if args.json_output:
            _write_json_result(result, args.json_output)
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
