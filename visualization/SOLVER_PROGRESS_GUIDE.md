# DPLL(T) / Simplex Progress Visualization Guide

## 목적

`--visualization-mode realtime|feedback|both`는 ReLU split뿐 아니라
DPLL(T) round, theory 선택, Reluplex에서 호출한 Simplex의 세부 동작도
같이 계측한다.

```text
DPLL round
  └─ selected theory atoms
       └─ Simplex call(s)
            ├─ iterations / bound violations
            ├─ entering / leaving variables
            └─ pivot rows
       └─ THEORY_SAT / THEORY_CONFLICT / THEORY_UNSAT
```

## 실시간 dashboard 구역

| 구역 | 표시 내용 |
|---|---|
| `DPLL Rounds & Theory Flow` | round 시간·결과·Simplex 호출 수와 theory atom→Simplex→반환 결과의 흐름 |
| `Simplex Internals` | 호출 시간·결과, iteration/pivot, 위반 변수, entering/leaving 변수, pivot row |
| `Realtime ReLU Split History` | 최근 30초 활성 ReLU split 수의 실시간 선 그래프 |

실시간 PNG 렌더링은 solver thread와 분리된 background worker에서 수행하며,
여러 이벤트가 빠르게 발생하면 중간 렌더 요청을 합쳐 solver 지연을 줄인다.

## JSONL 이벤트

| event | 주요 field |
|---|---|
| `round_start` | `round` |
| `theory_selection` | `round`, `atoms[{atom_id, polarity, kind, expression}]` |
| `simplex_start` | `call_id`, `round`, `depth`, `origin`, `theory_atom_ids`, row/variable 수 |
| `simplex_iteration` | `call_id`, `iteration`, `violated_var`, value/lower/upper |
| `simplex_pivot` | `call_id`, `iteration`, `entering`, `leaving`, `row` |
| `simplex_end` | `call_id`, `result`, `duration_seconds`, `iterations`, `pivots` |
| `theory_result` | `round`, `result`, conflict clause 크기 등 |
| `round_end` | `round`, `result`, `duration_seconds`, `simplex_calls` |
| `solver_end` | solver `status`, `reason`, `rounds` |

## 종료 후 feedback

feedback mode는 progress JSONL 전체를 집계해 DPLL+Theory와 Simplex의 두 PNG와
공통 JSON summary를 저장한다. JSON에는 round/call 상세, 누적 iteration/pivot 수, 상위 위반
변수, entering/leaving 변수, pivot row, theory 연결 결과가 들어 있다.

## 제어 데모

repository root에서:

```powershell
python -m visualization.solver_progress_demo --interval 0.2
```

이 데모는 round 3개, Simplex call 5개, `SAT`/`UNSAT`/`ITERATION_LIMIT`
상태와 반복 pivot 통계를 제어된 이벤트로 보여 준다. visualization
동작 확인용이며 실제 안전성 검증 결과가 아니다.

## AutoVerify 실행

```powershell
python Automation\run_large_model.py verify-vnnlib `
  --model Custom\ACASXU_experimental_v2a_1_1_custom.bin `
  --vnnlib ..\neuralsat\src\example\vnnlib\prop_1.vnnlib `
  --allow-large-model --timeout-seconds 300 `
  --visualization-mode both
```

realtime이 켜지면 하나의 HTML 안에서 DPLL+Theory, Simplex, ReLU split을
세 구역으로 표시한다. 개별 HTML은 AutoVerify에서 생성하거나 열지 않는다.
feedback이 켜지면 split heatmap, DPLL+Theory/Simplex 두 PNG, 공통 JSON이 생성된다.

## 출력 경로 옵션

| option | 용도 |
|---|---|
| `--solver-progress-log-output` | 원본 JSONL 경로 |
| `--solver-realtime-output` | solver 패널의 base PNG 경로; `_dpll_theory`, `_simplex` 파일 생성 |
| `--solver-feedback-output` | feedback 패널의 base PNG 경로; 두 PNG와 같은 stem의 JSON 생성 |

통합 realtime HTML은 `--realtime-split-output` PNG와 같은 stem으로 생성한다.

## 주의사항

- progress는 visualization을 켰 때만 수집한다.
- iteration 단위 JSONL 기록은 아예 계측하지 않는 실행보다 I/O를 늘린다.
- realtime PNG 렌더링은 background worker가 담당하지만 JSONL append 비용은
  solver 실행 시간에 포함된다.
- ReLU split이 없어도 DPLL round와 Simplex progress는 theory solver가 호출되면
  기록된다.
