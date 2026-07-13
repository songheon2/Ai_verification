"""VNNLIB의 선형 입력·출력 제약을 기존 Prop 구조로 변환한다."""

from __future__ import annotations

from dataclasses import dataclass
import gzip
from math import isfinite
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Mapping, Sequence, Tuple, Union

AI_VERIFICATION_DIR = Path(__file__).resolve().parent.parent
if str(AI_VERIFICATION_DIR) not in sys.path:
    sys.path.insert(0, str(AI_VERIFICATION_DIR))

from DPLL import (
    AndProp,
    FalseProp,
    ImplProp,
    InequProp,
    NotProp,
    OrProp,
    Prop,
    TrueProp,
)
from XOREncoding import conj, disj


class VnnlibParseError(ValueError):
    """지원하지 않거나 잘못된 VNNLIB 문법을 발견했을 때 발생한다."""


@dataclass(frozen=True)
class LinearExpression:
    """상수항을 포함하는 선형식 ``sum(coeffs[var] * var) + constant``."""

    coeffs: Mapping[str, float]
    constant: float = 0.0

    @staticmethod
    def variable(name: str) -> "LinearExpression":
        return LinearExpression({name: 1.0}, 0.0)

    @staticmethod
    def number(value: float) -> "LinearExpression":
        return LinearExpression({}, float(value))

    def plus(self, other: "LinearExpression") -> "LinearExpression":
        coeffs = dict(self.coeffs)
        for name, coefficient in other.coeffs.items():
            coeffs[name] = coeffs.get(name, 0.0) + coefficient
            if coeffs[name] == 0.0:
                del coeffs[name]
        return LinearExpression(coeffs, self.constant + other.constant)

    def scaled(self, factor: float) -> "LinearExpression":
        return LinearExpression(
            {
                name: coefficient * factor
                for name, coefficient in self.coeffs.items()
                if coefficient * factor != 0.0
            },
            self.constant * factor,
        )

    def minus(self, other: "LinearExpression") -> "LinearExpression":
        return self.plus(other.scaled(-1.0))


Substitution = Union[str, LinearExpression]


@dataclass(frozen=True)
class VnnlibDocument:
    """파싱된 VNNLIB 제약식과 선언 정보를 보관한다."""

    formula: Prop
    assertion_count: int
    declared_variables: Tuple[str, ...]
    input_variables: Tuple[str, ...]
    output_variables: Tuple[str, ...]
    source_path: str = ""


def _strip_comments(text: str) -> str:
    return "\n".join(line.split(";", 1)[0] for line in text.splitlines())


def _parse_sexpressions(text: str) -> List[Any]:
    tokens = re.findall(r"\(|\)|[^\s()]+", _strip_comments(text))
    position = 0

    def parse_one() -> Any:
        nonlocal position
        if position >= len(tokens):
            raise VnnlibParseError("S-expression이 끝나기 전에 입력이 종료되었습니다")
        token = tokens[position]
        position += 1
        if token == "(":
            result = []
            while True:
                if position >= len(tokens):
                    raise VnnlibParseError("닫는 괄호가 없습니다")
                if tokens[position] == ")":
                    position += 1
                    return result
                result.append(parse_one())
        if token == ")":
            raise VnnlibParseError("여는 괄호 없이 닫는 괄호가 나왔습니다")
        return token

    forms = []
    while position < len(tokens):
        forms.append(parse_one())
    return forms


def _number(token: Any) -> float | None:
    if not isinstance(token, str):
        return None
    try:
        value = float(token)
    except ValueError:
        return None
    if not isfinite(value):
        raise VnnlibParseError(f"유한하지 않은 숫자는 지원하지 않습니다: {token}")
    return value


class _Converter:
    def __init__(
        self,
        substitutions: Mapping[str, Substitution],
        strict_epsilon: float,
    ) -> None:
        self.substitutions = substitutions
        self.strict_epsilon = float(strict_epsilon)
        if self.strict_epsilon <= 0:
            raise ValueError("strict_epsilon은 양수여야 합니다")

    def arithmetic(self, node: Any) -> LinearExpression:
        numeric = _number(node)
        if numeric is not None:
            return LinearExpression.number(numeric)

        if isinstance(node, str):
            if node not in self.substitutions:
                raise VnnlibParseError(f"매핑되지 않은 변수입니다: {node}")
            substitution = self.substitutions[node]
            if isinstance(substitution, str):
                return LinearExpression.variable(substitution)
            return LinearExpression(dict(substitution.coeffs), substitution.constant)

        if not isinstance(node, list) or not node:
            raise VnnlibParseError(f"잘못된 산술식입니다: {node!r}")
        operator = node[0]
        arguments = node[1:]

        if operator == "+":
            result = LinearExpression.number(0.0)
            for argument in arguments:
                result = result.plus(self.arithmetic(argument))
            return result

        if operator == "-":
            if not arguments:
                raise VnnlibParseError("'-' 연산에는 피연산자가 필요합니다")
            result = self.arithmetic(arguments[0])
            if len(arguments) == 1:
                return result.scaled(-1.0)
            for argument in arguments[1:]:
                result = result.minus(self.arithmetic(argument))
            return result

        if operator == "*":
            if not arguments:
                return LinearExpression.number(1.0)
            constant_factor = 1.0
            variable_expression: LinearExpression | None = None
            for argument in arguments:
                expression = self.arithmetic(argument)
                if expression.coeffs:
                    if variable_expression is not None:
                        raise VnnlibParseError("비선형 곱셈은 지원하지 않습니다")
                    variable_expression = expression
                else:
                    constant_factor *= expression.constant
            if variable_expression is None:
                return LinearExpression.number(constant_factor)
            return variable_expression.scaled(constant_factor)

        if operator == "/":
            if len(arguments) < 2:
                raise VnnlibParseError("'/' 연산에는 두 개 이상의 피연산자가 필요합니다")
            result = self.arithmetic(arguments[0])
            for argument in arguments[1:]:
                denominator = self.arithmetic(argument)
                if denominator.coeffs:
                    raise VnnlibParseError("변수로 나누는 연산은 지원하지 않습니다")
                if denominator.constant == 0.0:
                    raise VnnlibParseError("0으로 나눌 수 없습니다")
                result = result.scaled(1.0 / denominator.constant)
            return result

        raise VnnlibParseError(f"지원하지 않는 산술 연산입니다: {operator}")

    @staticmethod
    def _ge(expression: LinearExpression, minimum: float = 0.0) -> Prop:
        # sum(coeff*x) + constant >= minimum을 InequProp 형식으로 옮긴다.
        if not expression.coeffs:
            return TrueProp() if expression.constant >= minimum else FalseProp()
        return InequProp(
            coeffs=frozenset(expression.coeffs.items()),
            b=float(minimum - expression.constant),
        )

    def _comparison(self, operator: str, arguments: Sequence[Any]) -> Prop:
        if len(arguments) < 2:
            raise VnnlibParseError(f"'{operator}' 비교에는 두 항 이상이 필요합니다")
        expressions = [self.arithmetic(argument) for argument in arguments]
        predicates: List[Prop] = []

        if operator == "=":
            for left, right in zip(expressions, expressions[1:]):
                difference = left.minus(right)
                predicates.append(self._ge(difference))
                predicates.append(self._ge(difference.scaled(-1.0)))
            return conj(predicates)

        for left, right in zip(expressions, expressions[1:]):
            if operator in {">=", ">"}:
                difference = left.minus(right)
            else:
                difference = right.minus(left)
            minimum = self.strict_epsilon if operator in {"<", ">"} else 0.0
            predicates.append(self._ge(difference, minimum))
        return conj(predicates)

    def boolean(self, node: Any) -> Prop:
        if isinstance(node, str):
            lowered = node.lower()
            if lowered == "true":
                return TrueProp()
            if lowered == "false":
                return FalseProp()
            raise VnnlibParseError(f"지원하지 않는 Boolean 기호입니다: {node}")
        if not isinstance(node, list) or not node:
            raise VnnlibParseError(f"잘못된 Boolean 식입니다: {node!r}")

        operator = node[0]
        arguments = node[1:]
        if operator == "and":
            return conj([self.boolean(argument) for argument in arguments])
        if operator == "or":
            return disj([self.boolean(argument) for argument in arguments])
        if operator == "not":
            if len(arguments) != 1:
                raise VnnlibParseError("'not' 연산에는 피연산자 하나가 필요합니다")
            return NotProp(self.boolean(arguments[0]))
        if operator == "=>":
            if len(arguments) < 2:
                raise VnnlibParseError("'=>' 연산에는 두 항 이상이 필요합니다")
            result = self.boolean(arguments[-1])
            for argument in reversed(arguments[:-1]):
                result = ImplProp(self.boolean(argument), result)
            return result
        if operator in {"<=", ">=", "<", ">", "="}:
            return self._comparison(operator, arguments)
        raise VnnlibParseError(f"지원하지 않는 Boolean 연산입니다: {operator}")


def _natural_variable_key(name: str) -> Tuple[str, int]:
    match = re.fullmatch(r"([XY])_(\d+)", name)
    if match:
        return match.group(1), int(match.group(2))
    return name, -1


def parse_vnnlib_text(
    text: str,
    substitutions: Mapping[str, Substitution],
    *,
    strict_epsilon: float = 1e-6,
    source_path: str = "",
) -> VnnlibDocument:
    """VNNLIB 문자열을 하나의 conjunction Prop으로 변환한다."""

    forms = _parse_sexpressions(text)
    converter = _Converter(substitutions, strict_epsilon)
    declared: List[str] = []
    assertions: List[Prop] = []

    for form in forms:
        if not isinstance(form, list) or not form:
            raise VnnlibParseError(f"잘못된 최상위 식입니다: {form!r}")
        command = form[0]
        if command == "declare-const":
            if len(form) != 3 or str(form[2]).lower() != "real":
                raise VnnlibParseError(f"지원하지 않는 선언입니다: {form!r}")
            declared.append(str(form[1]))
        elif command == "declare-fun":
            if len(form) != 4 or form[2] != [] or str(form[3]).lower() != "real":
                raise VnnlibParseError(f"지원하지 않는 함수 선언입니다: {form!r}")
            declared.append(str(form[1]))
        elif command == "assert":
            if len(form) != 2:
                raise VnnlibParseError("assert에는 식 하나가 필요합니다")
            assertions.append(converter.boolean(form[1]))
        elif command in {"set-logic", "set-info", "check-sat", "exit"}:
            continue
        else:
            raise VnnlibParseError(f"지원하지 않는 최상위 명령입니다: {command}")

    if not assertions:
        raise VnnlibParseError("VNNLIB에 assert가 없습니다")

    input_variables = sorted(
        (name for name in declared if re.fullmatch(r"X_\d+", name)),
        key=_natural_variable_key,
    )
    output_variables = sorted(
        (name for name in declared if re.fullmatch(r"Y_\d+", name)),
        key=_natural_variable_key,
    )
    return VnnlibDocument(
        formula=conj(assertions),
        assertion_count=len(assertions),
        declared_variables=tuple(declared),
        input_variables=tuple(input_variables),
        output_variables=tuple(output_variables),
        source_path=source_path,
    )


def read_vnnlib_text(path: str) -> str:
    """일반 VNNLIB와 gzip으로 압축된 VNNLIB를 모두 읽는다."""

    source = Path(path)
    if source.suffix.lower() == ".gz":
        with gzip.open(source, "rt", encoding="utf-8") as handle:
            return handle.read()
    return source.read_text(encoding="utf-8")


def parse_vnnlib_file(
    path: str,
    substitutions: Mapping[str, Substitution],
    *,
    strict_epsilon: float = 1e-6,
) -> VnnlibDocument:
    """VNNLIB 또는 VNNLIB.GZ 파일을 Prop으로 변환한다."""

    return parse_vnnlib_text(
        read_vnnlib_text(path),
        substitutions,
        strict_epsilon=strict_epsilon,
        source_path=str(Path(path).resolve()),
    )
