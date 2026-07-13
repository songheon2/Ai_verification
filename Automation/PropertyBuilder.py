"""신경망 검증 속성을 위한 재사용 가능한 pre/post 조건 생성기.

기존 솔버는 모든 선형 술어를 ``sum(c_i*x_i) >= b`` 형태로 표현한다.
이 모듈은 해당 술어만 생성하며, 신경망 인코더나 DPLL(T)/Reluplex 구현은
변경하지 않는다.
"""

from __future__ import annotations

from math import isfinite
from pathlib import Path
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

AI_VERIFICATION_DIR = Path(__file__).resolve().parent.parent
if str(AI_VERIFICATION_DIR) not in sys.path:
    sys.path.insert(0, str(AI_VERIFICATION_DIR))

from DPLL import InequProp, Prop, TrueProp
from XOREncoding import conj


def _finite_number(value: Any, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a number, got {value!r}") from exc
    if not isfinite(number):
        raise ValueError(f"{field} must be finite, got {value!r}")
    return number


def _expand(
    value: Any,
    size: int,
    field: str,
    *,
    allow_none: bool = False,
) -> List[Optional[float]]:
    if value is None:
        if allow_none:
            return [None] * size
        raise ValueError(f"{field} is required")

    if isinstance(value, (int, float)):
        values = [value] * size
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = list(value)
        if len(values) != size:
            raise ValueError(f"{field} length {len(values)} != expected {size}")
    else:
        raise ValueError(f"{field} must be a number or a list of length {size}")

    result: List[Optional[float]] = []
    for index, item in enumerate(values):
        if item is None and allow_none:
            result.append(None)
        else:
            result.append(_finite_number(item, f"{field}[{index}]"))
    return result


def _lower_max(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None:
        return b
    if b is None:
        return a
    return max(a, b)


def _upper_min(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None:
        return b
    if b is None:
        return a
    return min(a, b)


def resolve_input_bounds(
    input_spec: Mapping[str, Any],
    case: Mapping[str, Any],
    input_size: int,
) -> Tuple[List[Optional[float]], List[Optional[float]]]:
    """케이스의 입력 영역을 계산하고 모델의 유효 입력 범위와 교차시킨다.

    케이스는 ``center`` + ``epsilon`` 또는 명시적인 ``lower`` / ``upper``
    범위 중 하나를 사용할 수 있다. 반환값은 아직 ``input_spec['space']``에
    선언된 좌표계에 있으며, 정규화는 별도 단계에서 적용한다.
    """

    domain = input_spec.get("domain", {}) or {}
    if not isinstance(domain, Mapping):
        raise ValueError("input.domain must be an object")

    domain_lower = _expand(
        domain.get("lower"), input_size, "input.domain.lower", allow_none=True
    )
    domain_upper = _expand(
        domain.get("upper"), input_size, "input.domain.upper", allow_none=True
    )

    explicit_lower = case.get("lower", case.get("lower_bounds"))
    explicit_upper = case.get("upper", case.get("upper_bounds"))
    has_explicit_region = explicit_lower is not None or explicit_upper is not None
    has_center_region = case.get("center") is not None

    if has_explicit_region and has_center_region:
        raise ValueError("a case must use either center/epsilon or lower/upper, not both")

    if has_center_region:
        center = _expand(case.get("center"), input_size, "case.center")
        epsilon_value = case.get("epsilon", input_spec.get("epsilon"))
        epsilon = _expand(epsilon_value, input_size, "case.epsilon")
        region_lower = []
        region_upper = []
        for index, (c_value, eps_value) in enumerate(zip(center, epsilon)):
            assert c_value is not None and eps_value is not None
            if eps_value < 0:
                raise ValueError(f"case.epsilon[{index}] must be non-negative")
            region_lower.append(c_value - eps_value)
            region_upper.append(c_value + eps_value)
    elif has_explicit_region:
        region_lower = _expand(
            explicit_lower, input_size, "case.lower", allow_none=True
        )
        region_upper = _expand(
            explicit_upper, input_size, "case.upper", allow_none=True
        )
    else:
        # 전체 입력 범위를 검증하는 속성은 중심점 없이도 정의할 수 있다.
        region_lower = [None] * input_size
        region_upper = [None] * input_size

    lower = [_lower_max(a, b) for a, b in zip(region_lower, domain_lower)]
    upper = [_upper_min(a, b) for a, b in zip(region_upper, domain_upper)]

    if all(value is None for value in lower + upper):
        raise ValueError(
            "the input region is unbounded; provide center/epsilon, case lower/upper, "
            "or input.domain"
        )

    for index, (lo, hi) in enumerate(zip(lower, upper)):
        if lo is not None and hi is not None and lo > hi:
            raise ValueError(
                f"empty input interval at index {index}: lower {lo} > upper {hi}"
            )
    return lower, upper


def transform_bounds_to_model_space(
    lower: Sequence[Optional[float]],
    upper: Sequence[Optional[float]],
    input_spec: Mapping[str, Any],
) -> Tuple[List[Optional[float]], List[Optional[float]]]:
    """명세가 원시 입력을 사용하면 ``(raw - mean) / scale``을 적용한다."""

    space = str(input_spec.get("space", "model")).lower()
    if space not in {"model", "raw", "normalized"}:
        raise ValueError("input.space must be 'model', 'raw', or 'normalized'")

    normalization = input_spec.get("normalization")
    if space != "raw" or normalization is None:
        return list(lower), list(upper)
    if not isinstance(normalization, Mapping):
        raise ValueError("input.normalization must be an object")

    size = len(lower)
    mean = _expand(normalization.get("mean"), size, "input.normalization.mean")
    scale = _expand(normalization.get("scale"), size, "input.normalization.scale")
    transformed_lower: List[Optional[float]] = []
    transformed_upper: List[Optional[float]] = []

    for index, (lo, hi, mu, divisor) in enumerate(zip(lower, upper, mean, scale)):
        assert mu is not None and divisor is not None
        if divisor == 0:
            raise ValueError(f"input.normalization.scale[{index}] must be non-zero")
        first = None if lo is None else (lo - mu) / divisor
        second = None if hi is None else (hi - mu) / divisor
        if divisor > 0:
            transformed_lower.append(first)
            transformed_upper.append(second)
        else:
            transformed_lower.append(second)
            transformed_upper.append(first)
    return transformed_lower, transformed_upper


def transform_values_to_model_space(
    values: Sequence[float], input_spec: Mapping[str, Any]
) -> List[float]:
    numeric_values = [
        _finite_number(value, f"input value[{index}]")
        for index, value in enumerate(values)
    ]
    space = str(input_spec.get("space", "model")).lower()
    normalization = input_spec.get("normalization")
    if space != "raw" or normalization is None:
        return numeric_values

    if not isinstance(normalization, Mapping):
        raise ValueError("input.normalization must be an object")
    mean = _expand(
        normalization.get("mean"), len(values), "input.normalization.mean"
    )
    scale = _expand(
        normalization.get("scale"), len(values), "input.normalization.scale"
    )
    result = []
    for index, (value, mu, divisor) in enumerate(zip(numeric_values, mean, scale)):
        assert mu is not None and divisor is not None
        if divisor == 0:
            raise ValueError(f"input.normalization.scale[{index}] must be non-zero")
        result.append((value - mu) / divisor)
    return result


def make_box_precondition(
    input_vars: Sequence[str],
    lower: Sequence[Optional[float]],
    upper: Sequence[Optional[float]],
) -> Prop:
    if len(input_vars) != len(lower) or len(input_vars) != len(upper):
        raise ValueError("input variable and bound lengths must match")

    predicates: List[Prop] = []
    for name, lo, hi in zip(input_vars, lower, upper):
        if lo is not None:
            predicates.append(InequProp(frozenset({(name, 1.0)}), float(lo)))
        if hi is not None:
            predicates.append(InequProp(frozenset({(name, -1.0)}), -float(hi)))
    return conj(predicates) if predicates else TrueProp()


def build_precondition(
    input_vars: Sequence[str],
    input_spec: Mapping[str, Any],
    case: Mapping[str, Any],
) -> Tuple[Prop, List[Optional[float]], List[Optional[float]]]:
    lower, upper = resolve_input_bounds(input_spec, case, len(input_vars))
    model_lower, model_upper = transform_bounds_to_model_space(
        lower, upper, input_spec
    )
    return (
        make_box_precondition(input_vars, model_lower, model_upper),
        model_lower,
        model_upper,
    )


def _bit(value: Any, field: str) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)) and float(value) in {0.0, 1.0}:
        return int(value)
    if isinstance(value, str) and value.strip().lower() in {"0", "1", "false", "true"}:
        return 1 if value.strip().lower() in {"1", "true"} else 0
    raise ValueError(f"{field} must be a binary value (0 or 1), got {value!r}")


def _at_least(variable: str, threshold: float) -> Prop:
    return InequProp(frozenset({(variable, 1.0)}), float(threshold))


def _at_most(variable: str, threshold: float) -> Prop:
    return InequProp(frozenset({(variable, -1.0)}), -float(threshold))


def _binary_post(
    output_var: str,
    expected: Any,
    threshold: float,
    margin: float,
) -> Prop:
    label = _bit(expected, "expected")
    return (
        _at_least(output_var, threshold + margin)
        if label == 1
        else _at_most(output_var, threshold - margin)
    )


def _multiclass_post(
    output_vars: Sequence[str],
    expected: Any,
    decision: str,
    margin: float,
) -> Prop:
    if len(output_vars) < 2:
        raise ValueError("multiclass output requires at least two values")
    try:
        target = int(expected)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"expected class index must be an integer, got {expected!r}") from exc
    if target < 0 or target >= len(output_vars):
        raise ValueError(
            f"expected class index {target} is outside [0, {len(output_vars) - 1}]"
        )
    if decision not in {"argmax", "argmin"}:
        raise ValueError("multiclass decision must be 'argmax' or 'argmin'")

    target_var = output_vars[target]
    predicates: List[Prop] = []
    for index, other_var in enumerate(output_vars):
        if index == target:
            continue
        if decision == "argmax":
            # 목표 출력 - 다른 출력 >= 마진
            terms = frozenset({(target_var, 1.0), (other_var, -1.0)})
        else:
            # 다른 출력 - 목표 출력 >= 마진
            terms = frozenset({(other_var, 1.0), (target_var, -1.0)})
        predicates.append(InequProp(terms, margin))
    return conj(predicates)


def _multilabel_post(
    output_vars: Sequence[str],
    expected: Any,
    thresholds: Any,
    margins: Any,
) -> Prop:
    if not isinstance(expected, Sequence) or isinstance(expected, (str, bytes)):
        raise ValueError("multilabel expected must be a list of binary values")
    labels = list(expected)
    if len(labels) != len(output_vars):
        raise ValueError(
            f"multilabel expected length {len(labels)} != outputs {len(output_vars)}"
        )
    threshold_values = _expand(
        thresholds, len(output_vars), "output.threshold"
    )
    margin_values = _expand(margins, len(output_vars), "output.margin")
    predicates = []
    for index, (variable, label, threshold, margin) in enumerate(
        zip(output_vars, labels, threshold_values, margin_values)
    ):
        assert threshold is not None and margin is not None
        if margin < 0:
            raise ValueError(f"output.margin[{index}] must be non-negative")
        predicates.append(_binary_post(variable, label, threshold, margin))
    return conj(predicates)


def _regression_post(
    output_vars: Sequence[str],
    output_spec: Mapping[str, Any],
    expected: Any,
) -> Prop:
    payload: Dict[str, Any] = dict(output_spec)
    if isinstance(expected, Mapping):
        payload.update(expected)
    elif expected is not None:
        payload["target"] = expected

    size = len(output_vars)
    if payload.get("target") is not None:
        target = _expand(payload.get("target"), size, "regression.target")
        tolerance = _expand(
            payload.get("tolerance"), size, "regression.tolerance"
        )
        lower = []
        upper = []
        for index, (center, delta) in enumerate(zip(target, tolerance)):
            assert center is not None and delta is not None
            if delta < 0:
                raise ValueError(f"regression.tolerance[{index}] must be non-negative")
            lower.append(center - delta)
            upper.append(center + delta)
    else:
        lower = _expand(
            payload.get("lower", payload.get("lower_bounds")),
            size,
            "regression.lower",
            allow_none=True,
        )
        upper = _expand(
            payload.get("upper", payload.get("upper_bounds")),
            size,
            "regression.upper",
            allow_none=True,
        )
        if all(value is None for value in lower + upper):
            raise ValueError(
                "regression postcondition needs target/tolerance or lower/upper bounds"
            )
    return make_box_precondition(output_vars, lower, upper)


def build_postcondition(
    output_vars: Sequence[str],
    output_spec: Mapping[str, Any],
    expected: Any,
) -> Prop:
    """이진·다중 클래스·다중 라벨·회귀 출력의 post 조건을 생성한다."""

    output_type = str(output_spec.get("type", "unspecified")).lower()
    if output_type in {"unspecified", "required", ""}:
        raise ValueError(
            "output.type must be binary, multiclass, multilabel, or regression"
        )

    if output_type == "binary":
        margin = _finite_number(output_spec.get("margin", 0.0), "output.margin")
        if margin < 0:
            raise ValueError("output.margin must be non-negative")
        if len(output_vars) != 1:
            raise ValueError(f"binary output requires 1 value, model has {len(output_vars)}")
        threshold = _finite_number(
            output_spec.get("threshold", 0.0), "output.threshold"
        )
        return _binary_post(output_vars[0], expected, threshold, margin)

    if output_type == "multiclass":
        margin = _finite_number(output_spec.get("margin", 0.0), "output.margin")
        if margin < 0:
            raise ValueError("output.margin must be non-negative")
        decision = str(output_spec.get("decision", "argmax")).lower()
        return _multiclass_post(output_vars, expected, decision, margin)

    if output_type in {"multilabel", "multi_label"}:
        return _multilabel_post(
            output_vars,
            expected,
            output_spec.get("threshold", 0.0),
            output_spec.get("margin", 0.0),
        )

    if output_type == "regression":
        return _regression_post(output_vars, output_spec, expected)

    raise ValueError(f"unsupported output.type: {output_type!r}")


def infer_expected_from_outputs(
    output_spec: Mapping[str, Any], output_values: Sequence[float]
) -> Any:
    """중심점 기준 라벨을 계산한다. 이 값은 실제 정답임을 보장하지 않는다."""

    output_type = str(output_spec.get("type", "unspecified")).lower()
    values = [float(value) for value in output_values]
    if output_type == "binary":
        if len(values) != 1:
            raise ValueError("binary center prediction requires exactly one output")
        threshold = _finite_number(
            output_spec.get("threshold", 0.0), "output.threshold"
        )
        return int(values[0] >= threshold)
    if output_type == "multiclass":
        decision = str(output_spec.get("decision", "argmax")).lower()
        if decision == "argmax":
            return max(range(len(values)), key=values.__getitem__)
        if decision == "argmin":
            return min(range(len(values)), key=values.__getitem__)
        raise ValueError("multiclass decision must be 'argmax' or 'argmin'")
    if output_type in {"multilabel", "multi_label"}:
        thresholds = _expand(
            output_spec.get("threshold", 0.0), len(values), "output.threshold"
        )
        return [int(value >= threshold) for value, threshold in zip(values, thresholds)]
    if output_type == "regression":
        tolerance = output_spec.get("tolerance")
        if tolerance is None:
            raise ValueError(
                "center-referenced regression requires output.tolerance"
            )
        return {"target": values, "tolerance": tolerance}
    raise ValueError(
        "output.type must be set before a center prediction can be inferred"
    )
