"""ONNX/커스텀 완전연결(ReLU) 신경망을 이 저장소의 DPLL(T)/Reluplex로 검증하는 CLI.

세 개 서브커맨드
----------------
1) inspect — 모델 구조를 읽어서 출력하고, 필요하면 스펙 템플릿을 만든다.

     python AutoVerify.py inspect Onnx/model.onnx
     python AutoVerify.py inspect Onnx/model.onnx --output Specs/model.json

   출력 예: "Layers: 5 -> 50 -> 50 -> 5", "Hidden ReLUs: 300". --output을 주면
   그 구조에 맞는 JSON 스펙 뼈대(cases는 "edit_me" 자리표시자)를 만들어준다 —
   그 안의 cases/epsilon/expected를 직접 채운 뒤 verify에 넘기는 용도.

2) verify — JSON/YAML 스펙 파일(input/output/property/cases 정의)로 검증.
   스펙 자체 검증용 로컬 강건성/분류 조건을 여러 case로 정의할 때 쓴다.

     python AutoVerify.py verify --model Onnx/model.onnx --spec Specs/model.json
     python AutoVerify.py verify --model Custom/model.txt --spec Specs/model.json --dry-run

3) verify-vnnlib — ONNX + 표준 VNNLIB 속성 파일을 그대로 검증 (VNNCOMP
   벤치마크 형식). VNNLIB의 출력 assertion은 "찾으려는 unsafe 조건"이므로
   SAT=반례 있음(COUNTEREXAMPLE), UNSAT=안전 확정(VERIFIED)으로 뒤집어 보고한다.

     python AutoVerify.py verify-vnnlib --model model.onnx --vnnlib prop_1.vnnlib --dry-run
     python AutoVerify.py verify-vnnlib --model model.onnx --vnnlib prop_1.vnnlib.gz \
         --allow-large-model --timeout-seconds 300 --json-output Results/prop_1.json

   verify-vnnlib 전용 visualization은 --visualization-mode로 선택한다.
   realtime은 실행 중 활성 split 점을 갱신하고, feedback은 종료 후 누적
   SplitHeatmap을 저장하며, both는 둘 다 실행한다:

     python AutoVerify.py verify-vnnlib --model model.onnx --vnnlib prop_1.vnnlib.gz \
         --allow-large-model --timeout-seconds 300 \
         --visualization-mode both

   결과 JSON에도 split_summary(total_split_events, distinct_neurons)가 같이
   담긴다. 단, ReLU 분기 없이 순수 불리언 추론만으로 UNSAT이 확정되는
   인스턴스(solver.reason == BOOLEAN_UNSAT)는 Reluplex 자체가 호출되지
    않아서 split이 0건이고 히트맵 값도 모두 0이다 — 버그가 아니라 그
   인스턴스가 이론 솔버까지 갈 필요가 없었다는 뜻이다. 실제 split 이벤트를
   보려면 이론 솔버(Reluplex)까지 도달하는 인스턴스가 필요하다.

   계속 SIMPLEX_ITERATION_LIMIT으로 UNKNOWN이 나면 --simplex-max-iter를
   크게 올린다 (기본 10000):

     python AutoVerify.py verify-vnnlib --model model.onnx --vnnlib prop_1.vnnlib.gz \
         --allow-large-model --timeout-seconds 900 --simplex-max-iter 1000000

   (스모크 테스트로 실측: relu_smoke.onnx + counterexample.vnnlib를
   --simplex-max-iter 1로 강제로 돌리면 SIMPLEX_ITERATION_LIMIT/UNKNOWN이
   재현되고, 기본값(10000)으로는 정상적으로 COUNTEREXAMPLE이 나온다 —
   즉 --timeout-seconds와 별개로 반복 횟수 자체가 부족해서 UNKNOWN이 나는
   경우엔 이 옵션이 필요하다.)

   TIMEOUT이 나는데 어느 단계에서 멈춰있는지(encode_nn / VNNLIB 파싱 /
   tseitin_cnf / dpll() 재귀 탐색) 전혀 안 보일 때는 --profile-stages로
   단계별 소요시간을 찍어본다:

     python AutoVerify.py verify-vnnlib --model model.onnx --vnnlib prop_1.vnnlib.gz \
         --allow-large-model --timeout-seconds 300 --profile-stages

   dpll()은 재귀 호출이 너무 잦아서 매번 찍으면 감당이 안 되니, 기본
   1000회 호출마다 한 번씩만 "dpll(): N calls, elapsed Xs, still
   running..."을 찍는다 (DPLL_T.py의 profile_state["print_every"]를
   고치면 간격 조절 가능, CLI 옵션으로는 아직 안 뚫어놨음). 출력이
   encode_nn/파싱/tseitin_cnf까지만 나오고 dpll() 진행 로그가 한 번도 안
   찍힌 채 타임아웃 났다면 → dpll() 호출 자체는 그 간격만큼도 안 됐는데
   그 각각이 느리다는 뜻 (조합폭발이 아니라 개별 호출이 비효율적인 게 병목).

   (실측/결론: mMIMO 모델(은닉 ReLU 982개, 256->491->491->16)의 idx=0,
   eps=1 인스턴스로 확인. tseitin_cnf는 20.5s에 CNF 10660절/3554변수를
   만드는데, print_every를 1로 낮춰서 보니 dpll() 호출 #1->#2는 15.6초,
   #2->#3은 **40시간(144000초)을 줘도 안 끝남** — 즉 조합폭발(호출 수가
   많음)이 아니라 단일 호출 안에서 멈춰있는 것이다. 원인은
   `DPLL.py`의 `unit_propagation()`: 새 unit literal이 하나 확정될
   때마다 전체 CNF를 처음부터 다시 스캔하는 구조라서(watched-literal 같은
   최적화 없음), 이 신경망처럼 3단으로 깊게 이어진 회로에서는 분기 하나가
   전체 층을 타고 전파되며 "재스캔 한 번"이 수백~수천 번 반복될 수 있다.
   즉 --timeout-seconds를 아무리 늘려도 해결 안 되고, `unit_propagation()`
   자체를 최적화해야 하는 문제다.)

공통 플래그
-----------
  --dry-run              ONNX/VNNLIB 로드 + 신경망 인코딩까지만 하고 실제 솔버는 안 돌림 (배관 확인용)
  --allow-large-model     은닉 ReLU 개수가 안전 한도(기본 50, verify-vnnlib는 --max-relus-without-override로 조절)를
                           넘는 모델도 강행 실행 (ACAS Xu류는 은닉 ReLU 300개라 필수)
  --timeout-seconds N     N초 안에 SAT/UNSAT을 못 정하면 UNKNOWN으로 종료
  --max-rounds N          DPLL(T) 라운드 수 상한 (verify-vnnlib 기본 1000)
  --simplex-max-iter N    Reluplex 내부 Simplex 호출당 반복 상한 (기본 10000). 계속
                           SIMPLEX_ITERATION_LIMIT으로 UNKNOWN이 나면 크게 올릴 것
                           (예: 1000000) — 실질적인 안전장치는 --timeout-seconds이므로
                           이 값을 올릴 때는 timeout도 같이 넉넉히 잡을 것 (verify는 spec의
                           solver.simplex_max_iter를 오버라이드, 안 주면 spec/기본값 사용)
  --json-output PATH      결과 전체를 JSON으로 저장 (반례 입력/출력, solver.reason 등 포함)
  --debug                 Simplex/Reluplex 내부 tableau를 매 스텝 출력 (매우 장황함)
  --profile-stages        encode_nn/tseitin_cnf/dpll() 단계별 소요시간을 [profile]
                           접두어로 출력 (평소엔 꺼둘 것, 어디서 멈췄는지 진단할 때만 켤 것)

verify-vnnlib 전용 플래그
--------------------------
  --visualization-mode MODE       off, realtime, feedback, both 중 선택
  --realtime-log-output PATH      실시간 +/- JSONL 로그 경로
  --realtime-split-output PATH    실시간 활성 split 점 PNG 경로
  --split-heatmap-output PATH     사후 feedback 히트맵 경로 (--feedback-heatmap-output 별칭)
  --relu-feedback-output PATH     실시간 화면 형식의 전체 기간 ReLU feedback PNG 경로
  --split-heatmap-threshold N     이전 버전과의 호환용 (현재 모든 노드를 표시)
  --split-heatmap-cap N           이전 버전과의 호환용 (현재 모든 노드를 표시)
  --max-recursion N               Reluplex ReLU 분기 재귀 깊이 상한 (기본 50)

입력 파일 관련 주의사항
------------------------
- ONNX는 .onnx 확장자만 인식한다 — VNNCOMP 배포본처럼 .onnx.gz로 압축돼 있으면
  실행 전에 직접 압축을 풀어야 한다(예: `gunzip -k model.onnx.gz`). VNNLIB는
  .vnnlib.gz를 그대로 읽을 수 있어서(Automation/VnnlibParser.py) 압축 해제가 필요 없다.
- 은닉 ReLU가 많은 모델(ACAS Xu급, 300개)은 dpll()이 재귀 호출이라 Python
  기본 재귀 한도(1000)를 넘겨서 RecursionError가 날 수 있다. 이럴 때는
  sys.setrecursionlimit(100000) 등을 먼저 걸고 실행해야 한다:

    python -c "import sys, runpy; sys.setrecursionlimit(100000); \
        sys.argv=['AutoVerify.py']+sys.argv[1:]; \
        runpy.run_path('Automation/AutoVerify.py', run_name='__main__')" \
        verify-vnnlib --model model.onnx --vnnlib prop_1.vnnlib.gz --allow-large-model

  (실측: ACAS Xu 1_1 네트워크(은닉 ReLU 300개)의 공식 prop_1은 60초 타임아웃
  안에 1라운드도 못 끝냈다 — 이 저장소의 순수 Python Reluplex로는 이 규모의
  공식 VNNCOMP property가 매우 느리므로, --timeout-seconds를 넉넉히 주거나
  더 쉬운 자체 property(작은 epsilon 등)로 먼저 시도하는 것을 권장.
  반대 사례도 있다: safenlp_2024/ruarobot(은닉 ReLU 128개)의 hyperrectangle_0은
  90초는 부족했지만 900초 타임아웃에서는 124초 만에 VERIFIED로 끝났다 —
  즉 모델 크기만으로 소요 시간을 예측하기 어렵고, 급할 때는 --timeout-seconds를
  일단 넉넉하게(수백~천 단위) 주고 실제로 얼마나 걸리는지 실측하는 편이 낫다.)

결과 해석
----------
  DRY_RUN         --dry-run일 때만. 로드/파싱/인코딩 성공.
  COUNTEREXAMPLE   반례 존재 확정 (솔버 SAT)
  VERIFIED         안전 확정 (솔버 UNSAT)
  UNKNOWN          시간/라운드 한도로 결론 못 냄 — result.solver.reason 확인
                    (TIMEOUT / DPLL_T_ROUND_LIMIT / SIMPLEX_ITERATION_LIMIT /
                     RELUPLEX_RECURSION_LIMIT / RELUPLEX_REPAIR_INCONCLUSIVE /
                     NEGATED_RELU_UNSUPPORTED)

VNNCOMP 벤치마크(ACAS Xu 등)를 특정 경로 기준으로 처음부터 끝까지 돌리는
전체 walkthrough(환경 설정, 경로 등록, prop_1~6 연속 실행 등)는
Automation/VNNCOMP_RUNBOOK.md 참고.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import shutil
import time
from pathlib import Path
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

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
from visualization.SolveTrace import SolveTrace
from visualization.SplitHeatmap import split_counts_from_trace, draw_split_heatmap
from visualization.RealtimeSplitVisualization import (
    RealtimeSplitVisualizer,
    SplitEventLogger,
    draw_realtime_split_dashboard,
    read_split_events,
    split_events_from_trace,
    split_metadata_path,
)
from visualization.SolverProgress import (
    SolverProgressLogger,
    SolverProgressPanelVisualizer,
    build_solver_feedback,
    read_solver_progress,
    solver_panel_paths,
    write_solver_feedback_panels,
)
from visualization.UnifiedRealtimeDashboard import UnifiedRealtimeDashboard
from visualization.RenderProcess import REALTIME_RENDER_INTERVAL_SECONDS
from visualization.VisualizationMode import (
    VisualizationMode,
    combined_mode,
    create_visualization_run_directory,
    default_visualization_paths,
    finalize_visualization_run_directory,
)


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


def _rebase_output_paths(value: Any, old_root: Path, new_root: Path) -> Any:
    """실행 폴더 rename 뒤 result 안의 해당 경로만 새 위치로 바꾼다."""
    if isinstance(value, str):
        try:
            relative = Path(value).relative_to(old_root)
        except ValueError:
            return value
        return str(new_root / relative)
    if isinstance(value, dict):
        return {
            key: _rebase_output_paths(item, old_root, new_root)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_rebase_output_paths(item, old_root, new_root) for item in value]
    if isinstance(value, tuple):
        return tuple(_rebase_output_paths(item, old_root, new_root) for item in value)
    return value


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


class PropEvalUnsupported(Exception):
    """반례 검증기가 다룰 줄 모르는 Prop 노드를 만났다."""


def _evaluate_prop(prop: Prop, assignment: Mapping[str, float], tol: float = 0.0) -> bool:
    """구체적인 할당에서 Prop이 참인지 평가한다.

    tol > 0이면 각 부등식을 그만큼 느슨하게 본다. NotProp/ImplProp의 전건
    아래에서는 완화 방향이 뒤집히므로 -tol을 넘긴다 (느슨하게 본 것이 부정을
    통과하면서 오히려 엄격해지는 것을 막는다).
    """

    if isinstance(prop, TrueProp):
        return True
    if isinstance(prop, FalseProp):
        return False
    if isinstance(prop, AndProp):
        return _evaluate_prop(prop.p, assignment, tol) and _evaluate_prop(
            prop.q, assignment, tol
        )
    if isinstance(prop, OrProp):
        return _evaluate_prop(prop.p, assignment, tol) or _evaluate_prop(
            prop.q, assignment, tol
        )
    if isinstance(prop, NotProp):
        return not _evaluate_prop(prop.p, assignment, -tol)
    if isinstance(prop, ImplProp):
        return (not _evaluate_prop(prop.p, assignment, -tol)) or _evaluate_prop(
            prop.q, assignment, tol
        )
    if isinstance(prop, InequProp):
        total = 0.0
        for var, coeff in prop.coeffs:
            if var not in assignment:
                raise PropEvalUnsupported(f"할당에 없는 변수: {var}")
            total += float(coeff) * float(assignment[var])
        return total >= float(prop.b) - tol
    raise PropEvalUnsupported(type(prop).__name__)


def _input_box_from_formula(
    formula: Prop, input_vars: Sequence[str]
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """최상위 conjunction에서 단일 변수 부등식만 모아 입력 박스를 복원한다.

    Or/Not 아래의 제약은 무조건 성립하지 않으므로 AndProp만 타고 내려간다.
    복원하지 못한 변수는 (-inf, inf)로 남고, 클리핑에서 그냥 통과된다.
    """

    lower = {name: float("-inf") for name in input_vars}
    upper = {name: float("inf") for name in input_vars}
    stack: List[Prop] = [formula]
    while stack:
        node = stack.pop()
        if isinstance(node, AndProp):
            stack.append(node.p)
            stack.append(node.q)
            continue
        if not isinstance(node, InequProp):
            continue
        items = list(node.coeffs)
        if len(items) != 1:
            continue
        var, coeff = items[0]
        if var not in lower or float(coeff) == 0.0:
            continue
        bound = float(node.b) / float(coeff)
        if float(coeff) > 0:
            lower[var] = max(lower[var], bound)
        else:
            upper[var] = min(upper[var], bound)
    return lower, upper


def _clip_to_box(
    values: Sequence[float],
    input_vars: Sequence[str],
    box: Tuple[Mapping[str, float], Mapping[str, float]],
) -> List[float]:
    lower, upper = box
    return [
        min(upper.get(name, float("inf")), max(lower.get(name, float("-inf")), float(v)))
        for name, v in zip(input_vars, values)
    ]


def _snap_near_integers(values: Sequence[float], tol: float) -> List[float]:
    """정수에서 tol 이내인 좌표만 그 정수로 붙인다 (나머지는 그대로)."""

    snapped = []
    for v in values:
        nearest = float(round(float(v)))
        snapped.append(nearest if abs(float(v) - nearest) <= tol else float(v))
    return snapped


def _validate_counterexample(
    formula: Prop,
    model: NNModel,
    input_vars: Sequence[str],
    output_vars: Sequence[str],
    model_inputs: Sequence[float],
    *,
    snap_tol: float = 1e-6,
    accept_tol: float = 1e-6,
) -> Dict[str, Any]:
    """반례 후보를 VNNLIB 속성에 직접 대입해 검증하고, 실패하면 복구를 시도한다.

    솔버가 내놓는 해는 '신경망을 인코딩한 선형 제약 시스템 + 허용오차'를 만족할
    뿐이라, 입력만 뽑아 신경망을 다시 돌리면 속성을 어길 수 있다 (특히 반례가
    경계에 딱 붙어 margin이 0인 경우 1 ULP 차이로 뒤집힌다). 그래서 여기서
    forward 결과로 속성을 다시 검사하고, 실패하면 아래 후보들을 차례로 시도한다.
    모든 후보는 채택 전에 반드시 검증을 통과해야 하므로, 공격적인 후보(정수
    반올림 등)를 넣어도 안전하다.

    반환 dict의 status:
        valid          — 엄격하게(tol=0) 속성을 만족. repair가 어떤 후보였는지 함께 보고
        tolerance_only — accept_tol 안에서만 만족. 반례로 쓰되 경고를 남길 것
        invalid        — 어떤 후보도 만족 못 함. COUNTEREXAMPLE로 보고하면 안 된다
        unavailable    — 평가할 수 없는 Prop 노드가 있어 검증 자체를 못 함
    """

    try:
        box = _input_box_from_formula(formula, input_vars)
    except PropEvalUnsupported as exc:
        return {"status": "unavailable", "detail": str(exc)}

    base = [float(v) for v in model_inputs]
    candidates: List[Tuple[str, List[float]]] = [("as_is", base)]

    def _add(label: str, values: List[float]) -> None:
        if all(a == b for a, b in zip(values, base)) and label != "as_is":
            return
        for _, seen in candidates:
            if all(a == b for a, b in zip(values, seen)):
                return
        candidates.append((label, values))

    _add("clip", _clip_to_box(base, input_vars, box))
    _add("snap", _clip_to_box(_snap_near_integers(base, snap_tol), input_vars, box))
    _add(
        "round",
        _clip_to_box([float(round(v)) for v in base], input_vars, box),
    )

    evaluated: List[Tuple[str, List[float], List[float]]] = []
    for label, values in candidates:
        outputs = forward_model(model, values)
        evaluated.append((label, values, outputs))

    def _holds(values: Sequence[float], outputs: Sequence[float], tol: float) -> bool:
        assignment: Dict[str, float] = {}
        assignment.update({name: float(v) for name, v in zip(input_vars, values)})
        assignment.update({name: float(v) for name, v in zip(output_vars, outputs)})
        return _evaluate_prop(formula, assignment, tol)

    try:
        for label, values, outputs in evaluated:
            if _holds(values, outputs, 0.0):
                return {
                    "status": "valid",
                    "repair": label,
                    "inputs": values,
                    "outputs": outputs,
                    "candidates_tried": [c[0] for c in evaluated],
                }
        for label, values, outputs in evaluated:
            if _holds(values, outputs, accept_tol):
                return {
                    "status": "tolerance_only",
                    "repair": label,
                    "inputs": values,
                    "outputs": outputs,
                    "accept_tol": accept_tol,
                    "candidates_tried": [c[0] for c in evaluated],
                }
    except PropEvalUnsupported as exc:
        return {"status": "unavailable", "detail": str(exc)}

    return {
        "status": "invalid",
        "candidates_tried": [c[0] for c in evaluated],
    }


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
    simplex_max_iter_override: Optional[int] = None,
    max_recursion_override: Optional[int] = None,
    profile_stages: bool = False,
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
    simplex_max_iter = int(
        simplex_max_iter_override
        if simplex_max_iter_override is not None
        else solver_spec.get("simplex_max_iter", 10000)
    )
    if simplex_max_iter <= 0:
        raise ValueError("solver.simplex_max_iter must be positive")
    max_recursion = int(
        max_recursion_override
        if max_recursion_override is not None
        else solver_spec.get("max_recursion", 50)
    )
    if max_recursion <= 0:
        raise ValueError("solver.max_recursion must be positive")
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
        if profile_stages:
            _t0 = time.monotonic()
            nn_property, output_vars, _ = encode_nn(model, input_vars, generator)
            print(f"[profile] {case_name}: encode_nn: {time.monotonic() - _t0:.1f}s")
        else:
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
                simplex_max_iter=simplex_max_iter,
                max_recursion=max_recursion,
                profile_stages=profile_stages,
            )
            result["solver"] = {
                **solver_result.to_dict(),
                "max_rounds": max_rounds,
                "timeout_seconds": timeout_seconds,
                "simplex_max_iter": simplex_max_iter,
                "max_recursion": max_recursion,
            }
            if solver_result.status == SolverStatus.SAT:
                counterexample = _counterexample(
                    solver_result.model or {},
                    input_vars,
                    output_vars,
                    input_spec,
                    model,
                )
                # 반례가 되려면 "입력 영역 안에 있으면서 스펙을 어겨야" 한다.
                # nn_property는 인코딩이므로 검증에서는 forward_model로 대체한다.
                check_formula = AndProp(precondition, NotProp(postcondition))
                model_inputs = counterexample.get("input_model_space") or []
                if model_inputs and all(v is not None for v in model_inputs):
                    validation = _validate_counterexample(
                        check_formula, model, input_vars, output_vars, model_inputs
                    )
                else:
                    validation = {
                        "status": "unavailable",
                        "detail": "입력 할당이 불완전합니다",
                    }
                counterexample["validation"] = validation

                if validation["status"] in {"valid", "tolerance_only"}:
                    counterexample["input_model_space"] = validation["inputs"]
                    counterexample["input_declared_space"] = _declared_space_values(
                        validation["inputs"], input_spec
                    )
                    counterexample["recomputed_outputs"] = validation["outputs"]
                    result["status"] = "COUNTEREXAMPLE"
                    if validation.get("repair") != "as_is":
                        result["note"] = (
                            "솔버가 내놓은 점은 조건을 만족하지 않아 "
                            f"'{validation['repair']}' 복구를 적용한 뒤 검증했습니다."
                        )
                elif validation["status"] == "unavailable":
                    result["status"] = "COUNTEREXAMPLE"
                    result["note"] = (
                        "반례를 검증하지 못했습니다: "
                        f"{validation.get('detail')}. 확인되지 않은 값입니다."
                    )
                else:
                    result["status"] = "UNKNOWN"
                    result["solver"]["reason"] = "COUNTEREXAMPLE_UNVALIDATED"
                    result["note"] = (
                        "솔버는 SAT을 반환했지만 그 입력으로 신경망을 다시 돌리면 "
                        "조건을 만족하지 않습니다. 반례로 인정하지 않고 UNKNOWN으로 낮춥니다."
                    )
                result["counterexample"] = counterexample
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
    simplex_max_iter: int = 10000,
    max_recursion: int = 50,
    trace: Optional[SolveTrace] = None,
    visualization_mode: str = "off",
    realtime_visualization: bool = False,
    feedback_visualization: bool = False,
    open_realtime_view: bool = False,
    realtime_panels: Optional[Sequence[str]] = None,
    realtime_refresh_ms: int = 500,
    realtime_playback_ms: int = 0,
    realtime_window_seconds: float = 300.0,
    realtime_log_output: Optional[str] = None,
    realtime_split_output: Optional[str] = None,
    split_heatmap_output: Optional[str] = None,
    relu_feedback_output: Optional[str] = None,
    solver_progress_log_output: Optional[str] = None,
    solver_realtime_output: Optional[str] = None,
    solver_feedback_output: Optional[str] = None,
    visualization_output_dir: Optional[str] = None,
    split_heatmap_threshold: float = 1,
    split_heatmap_cap: int = 20,
    profile_stages: bool = False,
) -> Dict[str, Any]:
    """VNNLIB이 정의한 unsafe 영역과 신경망의 교집합을 직접 탐색한다.

    visualization_mode은 off/realtime/feedback/both를 지원한다. realtime은
    실행 중 활성 split 점을 갱신하고, feedback은 실행 후 SolveTrace를 집계한
    SplitHeatmap을 저장한다. both는 두 채널을 같은 solver 실행에 함께 연결한다.
    기존 split_heatmap_output만 준 호출도 feedback으로 동작한다.

    실제 실행은 시각화 모드와 무관하게 입력 사본과 result.json을 실행
    폴더에 함께 보관한다. 개별 패널 경로를 명시하면 그 경로는 우선한다.

    simplex_max_iter는 Reluplex 내부 각 Simplex 호출의 반복 상한이다.
    SIMPLEX_ITERATION_LIMIT으로 UNKNOWN이 자주 나면 크게 올릴 것 — 이때
    timeout_seconds도 같이 넉넉하게 잡아야 한다(실질적인 안전장치는 timeout).

    profile_stages=True면 encode_nn/VNNLIB 파싱/tseitin_cnf/dpll() 각 단계의
    소요시간(및 dpll()은 기본 1000회마다 진행상황)을 [profile] 접두어로 찍는다 —
    큰 모델이 어느 단계에서 멈춰있는지 진단할 때 켠다.
    """

    visualization_started_at = datetime.now().astimezone()
    requested_visualization = VisualizationMode.parse(visualization_mode)
    feedback_enabled = (
        requested_visualization.feedback_enabled
        or feedback_visualization
        or split_heatmap_output is not None
        or relu_feedback_output is not None
        or solver_feedback_output is not None
    )
    realtime_enabled = (
        requested_visualization.realtime_enabled
        or realtime_visualization
        or solver_realtime_output is not None
    )
    effective_visualization = combined_mode(
        realtime=realtime_enabled,
        feedback=feedback_enabled,
    )
    allowed_realtime_panels = {"relu", "dpll-theory", "simplex"}
    selected_realtime_panels = set(realtime_panels or allowed_realtime_panels)
    unknown_realtime_panels = selected_realtime_panels - allowed_realtime_panels
    if unknown_realtime_panels:
        names = ", ".join(sorted(unknown_realtime_panels))
        raise ValueError(f"지원하지 않는 realtime panel: {names}")
    if realtime_enabled and not selected_realtime_panels:
        raise ValueError("realtime mode에는 하나 이상의 panel이 필요합니다")
    realtime_relu_enabled = realtime_enabled and "relu" in selected_realtime_panels
    realtime_solver_panels = set()
    if realtime_enabled and "dpll-theory" in selected_realtime_panels:
        realtime_solver_panels.add("dpll_theory")
    if realtime_enabled and "simplex" in selected_realtime_panels:
        realtime_solver_panels.add("simplex")
    realtime_solver_enabled = bool(realtime_solver_panels)
    feedback_relu_enabled = feedback_enabled and "relu" in selected_realtime_panels
    feedback_solver_panels = set()
    if feedback_enabled and "dpll-theory" in selected_realtime_panels:
        feedback_solver_panels.add("dpll_theory")
    if feedback_enabled and "simplex" in selected_realtime_panels:
        feedback_solver_panels.add("simplex")
    feedback_solver_enabled = bool(feedback_solver_panels)
    # A run directory owns the inputs and result, even with visualization off
    # or with all panel paths explicitly supplied. Grouping is not a side
    # effect of which optional image happens to need a default path.
    needs_default_output = not dry_run
    run_output_directory = None
    custom_visualization_output = False
    if needs_default_output:
        if visualization_output_dir is not None:
            run_output_directory = Path(visualization_output_dir)
            run_output_directory.mkdir(parents=True, exist_ok=True)
            custom_visualization_output = True
        else:
            run_output_directory = create_visualization_run_directory(
                model_path,
                started_at=visualization_started_at,
                stable_path=realtime_enabled and open_realtime_view,
            )
    default_paths = default_visualization_paths(
        model_path,
        vnnlib_path,
        output_dir=run_output_directory,
    )
    if feedback_relu_enabled and split_heatmap_output is None:
        split_heatmap_output = str(default_paths.feedback_heatmap)
    if feedback_relu_enabled and relu_feedback_output is None:
        relu_feedback_output = str(default_paths.feedback_history)
    if realtime_relu_enabled and realtime_log_output is None:
        realtime_log_output = str(default_paths.realtime_log)
    if realtime_relu_enabled and realtime_split_output is None:
        realtime_split_output = str(default_paths.realtime_image)
    if (realtime_solver_enabled or feedback_solver_enabled) and solver_progress_log_output is None:
        solver_progress_log_output = str(default_paths.solver_progress_log)
    if realtime_solver_enabled and solver_realtime_output is None:
        solver_realtime_output = str(default_paths.solver_realtime_image)
    if feedback_solver_enabled and solver_feedback_output is None:
        solver_feedback_output = str(default_paths.solver_feedback_image)

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
    if max_recursion <= 0:
        raise ValueError("max_recursion must be positive")
    if simplex_max_iter <= 0:
        raise ValueError("simplex_max_iter must be positive")
    if max_relus_without_override < 0:
        raise ValueError("max_relus_without_override must be non-negative")
    if realtime_refresh_ms <= 0:
        raise ValueError("realtime_refresh_ms must be positive")
    if realtime_playback_ms < 0:
        raise ValueError("realtime_playback_ms must be non-negative")
    if realtime_window_seconds <= 0:
        raise ValueError("realtime_window_seconds must be positive")
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

    # Build usable initial frames before opening the dashboard.  This keeps a
    # first-ever realtime run from showing empty/broken image slots while model
    # loading and encoding are still in progress. 같은 model/property 조합은 매번
    # 같은 출력 경로를 사용하므로, 실제 인코딩을 시작하기 전에 이전 실행의 마지막
    # 프레임을 빈 상태로 교체해둔다.
    realtime_dashboard_path = None
    realtime_panel_outputs = {}
    realtime_solver_panel_paths = {}
    if realtime_enabled and not dry_run:
        if realtime_solver_enabled:
            all_solver_panel_paths = solver_panel_paths(solver_realtime_output)
            realtime_solver_panel_paths = {
                name: all_solver_panel_paths[name]
                for name in realtime_solver_panels
            }
        if realtime_relu_enabled:
            realtime_panel_outputs["relu"] = Path(realtime_split_output)
        if "dpll_theory" in realtime_solver_panels:
            realtime_panel_outputs["dpll-theory"] = realtime_solver_panel_paths[
                "dpll_theory"
            ]
        if "simplex" in realtime_solver_panels:
            realtime_panel_outputs["simplex"] = realtime_solver_panel_paths["simplex"]

        if realtime_relu_enabled:
            initial_split_visualizer = RealtimeSplitVisualizer(
                realtime_log_output,
                realtime_split_output,
                playback_interval_ms=realtime_playback_ms,
                window_seconds=realtime_window_seconds,
                model_layer_sizes=info.layer_sizes,
            )
            initial_split_logger = SplitEventLogger(
                realtime_log_output,
                update_callback=initial_split_visualizer.request_update,
                reset_log=True,
                model_layer_sizes=info.layer_sizes,
            )
            initial_split_logger.request_update()
            initial_split_visualizer.flush()

        if realtime_solver_enabled:
            initial_progress_visualizer = SolverProgressPanelVisualizer(
                solver_progress_log_output,
                solver_realtime_output,
                enabled_panels=realtime_solver_panels,
            )
            initial_progress_logger = SolverProgressLogger(
                solver_progress_log_output,
                update_callback=initial_progress_visualizer.request_update,
                reset_log=True,
                enabled_panels=realtime_solver_panels,
                simplex_detail_stride=100,
            )
            initial_progress_logger.request_update()
            initial_progress_visualizer.flush()
            initial_progress_logger.close()

        dashboard_base = (
            Path(realtime_split_output)
            if realtime_relu_enabled
            else Path(solver_realtime_output)
        )
        realtime_dashboard_path = dashboard_base.with_suffix(".html")
        UnifiedRealtimeDashboard(
            realtime_panel_outputs,
            realtime_dashboard_path,
            refresh_interval_ms=realtime_refresh_ms,
        ).write(open_browser=open_realtime_view)

    input_vars = [f"x{index}" for index in range(info.input_size)]
    generator = FreshGen(prefix="vnnlib_")

    if profile_stages:
        _t0 = time.monotonic()
        nn_property, output_vars, _ = encode_nn(model, input_vars, generator)
        print(f"[profile] encode_nn: {time.monotonic() - _t0:.1f}s")
    else:
        nn_property, output_vars, _ = encode_nn(model, input_vars, generator)

    substitutions = _vnnlib_substitutions(info, input_vars, output_vars)

    if profile_stages:
        _t0 = time.monotonic()
        document = parse_vnnlib_file(vnnlib_path, substitutions, strict_epsilon=strict_epsilon)
        print(f"[profile] parse_vnnlib_file: {time.monotonic() - _t0:.1f}s ({document.assertion_count} asserts)")
    else:
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
        "visualization": {"mode": effective_visualization.value},
    }
    if dry_run:
        result["status"] = "DRY_RUN"
        return result

    # Retain reproducible inputs alongside this run, without moving the user's
    # original model/property. Distinct names protect panel/result filenames.
    for section, source_key, supplied, archive_name in (
        ("model", "source_path", model_path, "model"),
        ("vnnlib", "path", vnnlib_path, "property"),
    ):
        source = Path(supplied).resolve()
        destination = run_output_directory / (archive_name + "".join(source.suffixes))
        if source != destination.resolve():
            if destination.exists():
                # An explicitly reused output directory must not silently
                # overwrite an archived input from another model/property.
                import filecmp
                if not filecmp.cmp(source, destination, shallow=False):
                    raise ValueError(f"Output directory contains a different input: {destination}")
            else:
                shutil.copy2(source, destination)
        result[section]["original_" + source_key] = str(source)
        result[section][source_key] = str(destination.resolve())

    if feedback_relu_enabled and trace is None:
        trace = SolveTrace()

    progress = None
    progress_visualizer = None
    progress_panels = realtime_solver_panels | feedback_solver_panels
    if progress_panels:
        progress_update_callback = None
        if realtime_solver_enabled:
            progress_visualizer = SolverProgressPanelVisualizer(
                solver_progress_log_output,
                solver_realtime_output,
                enabled_panels=realtime_solver_panels,
                refresh_interval_ms=realtime_refresh_ms,
            )
            progress_update_callback = progress_visualizer.request_update
        progress = SolverProgressLogger(
            solver_progress_log_output,
            update_callback=progress_update_callback,
            reset_log=True,
            enabled_panels=progress_panels,
            # 상세 이벤트는 그래프용 표본이다. 정확한 총 iteration/pivot 수는
            # simplex_end에 별도로 기록되므로 장기 feedback 실행에서 수백 MB의
            # JSONL을 만들 필요가 없다. realtime은 더 촘촘히 유지한다.
            simplex_detail_stride=100 if realtime_solver_enabled else 1000,
            # feedback-only는 실행 중 call duration을 그릴 필요가 없으므로
            # start/end를 완료 레코드 하나로 합친다. BOTH/REALTIME은 진행 중
            # call 표시를 위해 기존 start/end 스트림을 유지한다.
            compact_simplex_calls=(
                feedback_solver_enabled and not realtime_solver_enabled
            ),
            witness_context={
                "kind": "vnnlib_unsafe",
                "inputs": {
                    name: {"coeffs": dict(substitutions[name].coeffs),
                           "constant": substitutions[name].constant}
                    for name in document.input_variables
                },
                "outputs": {name: substitutions[name] for name in document.output_variables},
                "assertion_count": document.assertion_count,
            },
            flush_interval_seconds=(
                max(0.05, realtime_refresh_ms / 1000.0)
                if realtime_solver_enabled
                else 1.0
            ),
            flush_bytes=1024 * 1024 if not realtime_solver_enabled else 256 * 1024,
        )

    # VNNLIB assertion은 unsafe 집합을 나타내므로 SAT이면 반례가 존재한다.
    query = AndProp(nn_property, document.formula)
    try:
        solver_result = dpll_t_detailed(
            query,
            max_rounds=max_rounds,
            debug=debug,
            timeout_seconds=timeout_seconds,
            trace=trace,
            simplex_max_iter=simplex_max_iter,
            max_recursion=max_recursion,
            profile_stages=profile_stages,
            # feedback-only는 SolveTrace의 in-memory span으로 사후 그래프를 만들며
            # solving 중 split JSONL open/write/flush를 수행하지 않는다.
            split_mode="realtime" if realtime_relu_enabled else "off",
            split_log_path=realtime_log_output,
            realtime_output_path=realtime_split_output if realtime_relu_enabled else None,
            realtime_open_view=False,
            realtime_playback_ms=realtime_playback_ms,
            realtime_window_seconds=realtime_window_seconds,
            model_layer_sizes=info.layer_sizes,
            progress=progress,
        )
    finally:
        if progress is not None:
            # solver_end가 정상 경로에서 이미 닫지만, KeyboardInterrupt 같은 외부
            # 예외에서도 마지막 버퍼와 파일 핸들을 반드시 정리한다.
            progress.close()
    result["solver"] = {
        **solver_result.to_dict(),
        "max_rounds": max_rounds,
        "timeout_seconds": timeout_seconds,
        "simplex_max_iter": simplex_max_iter,
        "max_recursion": max_recursion,
    }
    if progress_visualizer is not None:
        progress_visualizer.flush()
    if realtime_enabled:
        realtime_result = {
            "dashboard_output": str(realtime_dashboard_path),
            "panel_outputs": {
                name: str(path) for name, path in realtime_panel_outputs.items()
            },
            "opened_panels": sorted(selected_realtime_panels),
            "refresh_interval_ms": realtime_refresh_ms,
            "render_interval_seconds": REALTIME_RENDER_INTERVAL_SECONDS,
            "playback_interval_ms": realtime_playback_ms,
            "window_seconds": realtime_window_seconds,
        }
        if realtime_relu_enabled:
            realtime_result.update({
                "log_output": realtime_log_output,
                "metadata_output": str(split_metadata_path(realtime_log_output)),
                "image_output": realtime_split_output,
            })
        if realtime_solver_enabled:
            realtime_result["solver_progress"] = {
                "log_output": solver_progress_log_output,
                "panel_outputs": {
                    name: str(path)
                    for name, path in realtime_solver_panel_paths.items()
                },
            }
        result["visualization"]["realtime"] = realtime_result

    if progress is not None:
        progress_feedback = build_solver_feedback(
            read_solver_progress(solver_progress_log_output)
        )
        result["solver_progress_summary"] = progress_feedback["counts"]

    if trace is not None:
        split_counts = split_counts_from_trace(trace)
        result["split_summary"] = {
            "total_split_events": sum(
                1 for ev in trace.events if ev.component == "reluplex_split"
            ),
            "distinct_neurons": len(split_counts),
        }
        if feedback_relu_enabled and split_heatmap_output is not None:
            fig, _ax = draw_split_heatmap(
                model, split_counts,
                threshold=split_heatmap_threshold, cap=split_heatmap_cap,
                title=f"Split heatmap: {Path(model_path).name}",
            )
            Path(split_heatmap_output).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(split_heatmap_output, dpi=150, bbox_inches="tight")
            import matplotlib.pyplot as plt
            plt.close(fig)
            result["split_heatmap_output"] = split_heatmap_output

        if feedback_relu_enabled and relu_feedback_output is not None:
            split_events = (
                read_split_events(realtime_log_output)
                if realtime_relu_enabled
                else split_events_from_trace(trace)
            )
            fig, _axes = draw_realtime_split_dashboard(
                realtime_log_output or "",
                events=split_events,
                full_history=True,
                model_layer_sizes=info.layer_sizes,
                title=f"Full ReLU split feedback: {Path(model_path).name}",
                round_starts=(None if realtime_relu_enabled else [
                    {"round": event.meta["round_index"],
                     "time_ns": int(event.t_start * 1_000_000_000)}
                    for event in trace.events if event.component == "dpll_round_start"
                ]),
            )
            Path(relu_feedback_output).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(relu_feedback_output, dpi=150, bbox_inches="tight")
            import matplotlib.pyplot as plt
            plt.close(fig)
            result["relu_feedback_output"] = relu_feedback_output

        if feedback_relu_enabled:
            feedback_result = result["visualization"].setdefault("feedback", {})
            if split_heatmap_output is not None:
                feedback_result["heatmap_output"] = split_heatmap_output
            if relu_feedback_output is not None:
                feedback_result["relu_history_output"] = relu_feedback_output
            if realtime_relu_enabled:
                feedback_result["relu_log_output"] = realtime_log_output
            else:
                feedback_result["relu_event_source"] = "memory"

    if feedback_solver_enabled and solver_feedback_output is not None:
        if realtime_solver_enabled:
            # Realtime and feedback use the same DPLL/Theory and Simplex panel
            # contents at solver completion.  In BOTH mode, keep the realtime
            # files as the single image set instead of rendering duplicate
            # *_feedback_*.png files.  The machine-readable feedback summary is
            # still written separately.
            solver_feedback = build_solver_feedback(
                read_solver_progress(solver_progress_log_output)
            )
            solver_feedback_panels = {
                name: realtime_solver_panel_paths[name]
                for name in sorted(feedback_solver_panels)
            }
            solver_feedback_json = Path(solver_feedback_output).with_suffix(".json")
            solver_feedback_json.parent.mkdir(parents=True, exist_ok=True)
            solver_feedback_json.write_text(
                json.dumps(
                    solver_feedback,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n",
                encoding="utf-8",
            )
            if "simplex" in feedback_solver_panels and len({
                call.get("round") for call in solver_feedback["simplex_calls"]
                if call.get("round") is not None
            }) > 1:
                # Upgrade the shared image to feedback only after solving;
                # realtime stays unannotated and no duplicate PNG is created.
                write_solver_feedback_panels(
                    solver_progress_log_output, solver_realtime_output,
                    json_path=solver_feedback_json, enabled_panels={"simplex"},
                )
        else:
            solver_feedback, solver_feedback_panels = write_solver_feedback_panels(
                solver_progress_log_output,
                solver_feedback_output,
                enabled_panels=feedback_solver_panels,
            )
        result.setdefault("visualization", {}).setdefault("feedback", {})[
            "solver_progress_panels"
        ] = {name: str(path) for name, path in solver_feedback_panels.items()}
        result["visualization"]["feedback"]["solver_progress_json"] = str(
            Path(solver_feedback_output).with_suffix(".json")
        )
        result["solver_progress_summary"] = solver_feedback["counts"]

    if solver_result.status == SolverStatus.SAT:
        input_spec = _vnnlib_input_spec(info)
        counterexample = _counterexample(
            solver_result.model or {},
            input_vars,
            output_vars,
            input_spec,
            model,
        )

        # 솔버 내부 Y와 실제 forward 결과의 괴리는 "이 반례를 믿지 말라"는 신호다.
        # (솔버의 해는 인코딩 + 허용오차를 만족할 뿐 신경망 위의 점이 아닐 수 있다)
        solver_outputs = counterexample.get("solver_outputs") or []
        recomputed = counterexample.get("recomputed_outputs") or []
        if solver_outputs and recomputed and len(solver_outputs) == len(recomputed):
            counterexample["solver_vs_recomputed_max_diff"] = max(
                abs(float(a) - float(b))
                for a, b in zip(solver_outputs, recomputed)
                if a is not None and b is not None
            )

        model_inputs = counterexample.get("input_model_space") or []
        if all(v is not None for v in model_inputs) and model_inputs:
            validation = _validate_counterexample(
                document.formula, model, input_vars, output_vars, model_inputs
            )
        else:
            validation = {"status": "unavailable", "detail": "입력 할당이 불완전합니다"}

        counterexample["validation"] = validation
        status = validation["status"]

        if status in {"valid", "tolerance_only"}:
            # 복구된 점이 원래 점보다 나으면 그것을 반례로 보고한다.
            repaired_inputs = validation["inputs"]
            counterexample["input_model_space"] = repaired_inputs
            counterexample["input_declared_space"] = _declared_space_values(
                repaired_inputs, input_spec
            )
            counterexample["recomputed_outputs"] = validation["outputs"]
            result["status"] = "COUNTEREXAMPLE"
            if validation.get("repair") != "as_is":
                result["note"] = (
                    "솔버가 내놓은 점은 속성을 만족하지 않아 "
                    f"'{validation['repair']}' 복구를 적용한 뒤 검증했습니다."
                )
            if status == "tolerance_only":
                result["note"] = (
                    (result.get("note", "") + " ").strip()
                    + f" 이 반례는 허용오차 {validation['accept_tol']:g} 안에서만 "
                    "속성을 만족합니다 — 엄격한 판정에서는 반례로 인정되지 않을 수 있습니다."
                ).strip()
        elif status == "unavailable":
            result["status"] = "COUNTEREXAMPLE"
            result["note"] = (
                "반례를 검증하지 못했습니다(속성을 평가할 수 없음): "
                f"{validation.get('detail')}. 이 반례는 확인되지 않은 값입니다."
            )
        else:
            # 어떤 복구로도 속성을 만족시키지 못했다 — 반례라고 보고하면 안 된다.
            result["status"] = "UNKNOWN"
            result["solver"]["reason"] = "COUNTEREXAMPLE_UNVALIDATED"
            result["note"] = (
                "솔버는 SAT을 반환했지만, 그 입력으로 신경망을 다시 돌리면 VNNLIB "
                "속성을 만족하지 않습니다(시도한 복구: "
                f"{', '.join(validation.get('candidates_tried', []))}). "
                "반례로 인정하지 않고 UNKNOWN으로 낮춥니다."
            )
        result["counterexample"] = counterexample
    elif solver_result.status == SolverStatus.UNSAT:
        result["status"] = "VERIFIED"
        result["note"] = "VNNLIB unsafe 영역과 신경망의 교집합이 UNSAT입니다."
    else:
        result["status"] = "UNKNOWN"
        result["note"] = (
            "자원 한도 때문에 SAT/UNSAT을 확정하지 못했습니다: "
            f"{solver_result.reason}"
        )
    if run_output_directory is not None:
        visualization_finished_at = datetime.now().astimezone()
        if custom_visualization_output:
            final_output_directory = run_output_directory
            result["visualization"]["output_directory_is_custom"] = True
        elif realtime_enabled and open_realtime_view:
            # The browser has already loaded this local HTML URL.  Renaming its
            # parent directory at solver completion invalidates both the page
            # URL and every periodically refreshed image, leaving broken-image
            # icons even though the PNG files exist under the new name.  Keep
            # the live directory stable for browser-backed realtime sessions.
            final_output_directory = run_output_directory
            result["visualization"]["output_directory_kept_for_realtime"] = True
        else:
            try:
                final_output_directory = finalize_visualization_run_directory(
                    run_output_directory,
                    model_path,
                    started_at=visualization_started_at,
                    finished_at=visualization_finished_at,
                )
            except OSError as exc:
                final_output_directory = run_output_directory
                result["visualization"]["output_directory_finalize_error"] = str(exc)
            else:
                result = _rebase_output_paths(
                    result,
                    run_output_directory,
                    final_output_directory,
                )
        result["visualization"]["output_directory"] = str(final_output_directory)
        result["visualization"]["started_at"] = visualization_started_at.isoformat()
        result["visualization"]["finished_at"] = visualization_finished_at.isoformat()
        result["result_path"] = str((final_output_directory / "result.json").resolve())
        _write_json_result(result, result["result_path"])
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
            validation = counterexample.get("validation") or {}
            if validation:
                repair = validation.get("repair")
                suffix = f" (복구: {repair})" if repair and repair != "as_is" else ""
                print(f"  검증  : {validation.get('status')}{suffix}")
        if case.get("note"):
            print(f"  note  : {case['note']}")
        solver = case.get("solver")
        if solver:
            print(
                f"  solver: {solver['status']} | reason={solver['reason']} | "
                f"rounds={solver['rounds']} | elapsed={solver['elapsed_seconds']:.3f}s"
            )
            hits = (solver.get("theory_stats") or {}).get("RELUPLEX_RECURSION_LIMIT", 0)
            if hits:
                print(
                    f"  depth : ReLU 분기 깊이 상한(max_recursion="
                    f"{solver.get('max_recursion')})에 {hits}번 막혔습니다 "
                    "- --max-recursion을 올려보세요."
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
        validation = counterexample.get("validation") or {}
        if validation:
            label = {
                "valid": "속성 만족 확인됨",
                "tolerance_only": "허용오차 안에서만 만족 (주의)",
                "invalid": "속성 불만족 — 반례 아님",
                "unavailable": "검증 불가",
            }.get(validation.get("status"), validation.get("status"))
            repair = validation.get("repair")
            suffix = f" (복구: {repair})" if repair and repair != "as_is" else ""
            print(f"  검증  : {label}{suffix}")
        diff = counterexample.get("solver_vs_recomputed_max_diff")
        if diff is not None:
            print(f"  solver Y vs forward Y 최대 차이: {diff:.3e}")
    if result.get("note"):
        print(f"  note  : {result['note']}")
    solver = result.get("solver")
    if solver:
        print(
            f"  solver: {solver['status']} | reason={solver['reason']} | "
            f"rounds={solver['rounds']} | elapsed={solver['elapsed_seconds']:.3f}s"
        )
        hits = (solver.get("theory_stats") or {}).get("RELUPLEX_RECURSION_LIMIT", 0)
        if hits:
            print(
                f"  depth : ReLU 분기 깊이 상한(max_recursion="
                f"{solver.get('max_recursion')})에 {hits}번 막혔습니다 "
                "- --max-recursion을 올려보세요."
            )
    split_summary = result.get("split_summary")
    if split_summary:
        print(
            f"  split : {split_summary['total_split_events']} events across "
            f"{split_summary['distinct_neurons']} neurons"
        )
    if result.get("split_heatmap_output"):
        print(f"  split heatmap: {result['split_heatmap_output']}")
    if result.get("relu_feedback_output"):
        print(f"  ReLU full-period feedback: {result['relu_feedback_output']}")
    visualization = result.get("visualization", {})
    if visualization.get("mode", "off") != "off":
        print(f"  visualization: {visualization['mode']}")
        if visualization.get("output_directory"):
            print(f"    output folder: {visualization['output_directory']}")
        realtime = visualization.get("realtime")
        if realtime:
            print(f"    unified dashboard: {realtime['dashboard_output']}")
            if realtime.get("log_output"):
                print(f"    realtime log  : {realtime['log_output']}")
                print(f"    model metadata: {realtime['metadata_output']}")
            for name, path in realtime.get("panel_outputs", {}).items():
                print(f"    {name} image: {path}")
        feedback = visualization.get("feedback")
        if feedback and feedback.get("solver_progress_panels"):
            for name, path in feedback["solver_progress_panels"].items():
                print(f"    {name} feedback: {path}")


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
    verify_parser.add_argument(
        "--simplex-max-iter",
        type=int,
        help="override solver.simplex_max_iter from the spec (raise this if you keep hitting "
             "SIMPLEX_ITERATION_LIMIT; raise --timeout-seconds too since that's the real backstop)",
    )
    verify_parser.add_argument(
        "--profile-stages",
        action="store_true",
        help="encode_nn/tseitin_cnf/dpll() 각 단계의 소요시간을 [profile] 접두어로 출력 "
             "(어느 단계에서 멈춰있는지 진단할 때 켤 것, 평소엔 꺼둘 것)",
    )
    verify_parser.add_argument(
        "--max-recursion",
        type=int,
        help="override solver.max_recursion from the spec (기본 50). 은닉 ReLU가 이 값보다 "
             "많으면 탐색이 끝까지 못 내려가 UNKNOWN이 된다 - recursion_limit_hits 참고",
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
    vnnlib_parser.add_argument(
        "--simplex-max-iter", type=int, default=10000,
        help="Reluplex 내부 Simplex 호출당 반복 상한. SIMPLEX_ITERATION_LIMIT으로 "
             "UNKNOWN이 자주 나면 크게 올릴 것 (예: 1000000) - 이때 --timeout-seconds도 "
             "넉넉히 잡을 것, 실질적인 안전장치는 timeout이다.",
    )
    vnnlib_parser.add_argument(
        "--max-recursion", type=int, default=50,
        help="Reluplex ReLU 분기 재귀 깊이 상한 (기본 50). 한 경로에서 split은 매번 다른 "
             "뉴런을 고정하므로 은닉 ReLU 개수보다 크게 잡을 필요는 없다. 은닉 ReLU가 이 "
             "값보다 많으면 UNKNOWN이 된다 - 결과의 recursion_limit_hits가 0이 아니면 올릴 것",
    )
    vnnlib_parser.add_argument("--json-output")
    vnnlib_parser.add_argument(
        "--visualization-mode",
        choices=[mode.value for mode in VisualizationMode],
        default=VisualizationMode.OFF.value,
        help="off: 끔, realtime: split+solver dashboard, feedback: split+solver 사후 분석, both: 둘 다",
    )
    vnnlib_parser.add_argument(
        "--visualization-output-dir",
        help=(
            "모델·VNNLIB 사본, result.json, 시각화 산출물을 함께 저장할 폴더. "
            "생략 시 visualization/outputs 아래에 시작·종료시간과 모델명으로 자동 생성"
        ),
    )
    vnnlib_parser.add_argument(
        "--realtime-panels",
        nargs="+",
        choices=("relu", "dpll-theory", "simplex"),
        help="실행·생성할 시각화 패널 선택; both에서는 feedback에도 적용 (기본: 모두)",
    )
    vnnlib_parser.add_argument(
        "--realtime-refresh-ms",
        type=int,
        default=500,
        help="실시간 브라우저 이미지 확인 주기 ms (기본: 500, 초당 최대 2회)",
    )
    vnnlib_parser.add_argument(
        "--realtime-playback-ms",
        type=int,
        default=0,
        help="ReLU +/- 이벤트를 하나씩 보여줄 간격 ms (기본 0: 실제 최신 상태)",
    )
    vnnlib_parser.add_argument(
        "--realtime-window-seconds",
        type=float,
        default=300.0,
        help="ReLU 그래프가 현재 기준으로 표시할 고정 시간 폭 (기본: 300초/5분)",
    )
    vnnlib_parser.add_argument(
        "--realtime-log-output",
        help="realtime/feedback/both의 ReLU +/- JSONL 로그 경로 (생략 시 visualization/outputs)",
    )
    vnnlib_parser.add_argument(
        "--realtime-split-output",
        help="realtime/both의 활성 split 점 PNG 경로 (생략 시 visualization/outputs)",
    )
    vnnlib_parser.add_argument(
        "--split-heatmap-output", "--feedback-heatmap-output",
        dest="split_heatmap_output",
        help="feedback/both의 실행 후 split 히트맵 PNG 경로 (기존 옵션명도 지원)",
    )
    vnnlib_parser.add_argument(
        "--relu-feedback-output",
        help="feedback/both의 전체 기간 ReLU split history PNG 경로",
    )
    vnnlib_parser.add_argument(
        "--solver-progress-log-output",
        help="DPLL/Simplex/Theory flow JSONL 경로",
    )
    vnnlib_parser.add_argument(
        "--solver-realtime-output",
        help="DPLL+Theory/Simplex 실시간 패널 base PNG 경로 (_dpll_theory/_simplex 생성)",
    )
    vnnlib_parser.add_argument(
        "--solver-feedback-output",
        help="DPLL+Theory/Simplex feedback 패널 base PNG 경로 (_dpll_theory/_simplex 생성)",
    )
    vnnlib_parser.add_argument(
        "--split-heatmap-threshold", type=float, default=1,
        help="호환용 옵션 (현재 split 횟수와 무관하게 모든 노드를 표시)",
    )
    vnnlib_parser.add_argument(
        "--split-heatmap-cap", type=int, default=20,
        help="호환용 옵션 (현재 레이어당 표시 노드 수를 제한하지 않음)",
    )
    vnnlib_parser.add_argument(
        "--profile-stages",
        action="store_true",
        help="encode_nn/VNNLIB 파싱/tseitin_cnf/dpll() 각 단계의 소요시간(dpll()은 기본 1000회마다 "
             "진행상황)을 [profile] 접두어로 출력 (큰 모델이 어느 단계에서 멈춰있는지 진단할 때 "
             "켤 것, 평소엔 꺼둘 것)",
    )
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
                simplex_max_iter=args.simplex_max_iter,
                max_recursion=args.max_recursion,
                visualization_mode=args.visualization_mode,
                open_realtime_view=args.visualization_mode in ("realtime", "both"),
                realtime_panels=args.realtime_panels,
                realtime_refresh_ms=args.realtime_refresh_ms,
                realtime_playback_ms=args.realtime_playback_ms,
                realtime_window_seconds=args.realtime_window_seconds,
                realtime_log_output=args.realtime_log_output,
                realtime_split_output=args.realtime_split_output,
                split_heatmap_output=args.split_heatmap_output,
                relu_feedback_output=args.relu_feedback_output,
                solver_progress_log_output=args.solver_progress_log_output,
                solver_realtime_output=args.solver_realtime_output,
                solver_feedback_output=args.solver_feedback_output,
                visualization_output_dir=args.visualization_output_dir,
                split_heatmap_threshold=args.split_heatmap_threshold,
                split_heatmap_cap=args.split_heatmap_cap,
                profile_stages=args.profile_stages,
            )
            _print_vnnlib_verification(result)
            if result.get("result_path"):
                print(f"Result JSON: {result['result_path']}")
            elif args.json_output:
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
            simplex_max_iter_override=args.simplex_max_iter,
            max_recursion_override=args.max_recursion,
            profile_stages=args.profile_stages,
        )
        _print_verification(result)
        if args.json_output:
            _write_json_result(result, args.json_output)
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
