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
| `Simplex Internals` | 호출 시간·결과, 호출별 pivot 수, 위반 변수, entering/leaving 변수, pivot row |
| `Realtime ReLU Split History` | 최근 5분 활성 ReLU split 수의 실시간 선 그래프 |

Simplex `Call duration`의 y축은 실제 호출 중 가장 긴 `duration_seconds`를 상한으로
사용한다. 자동 눈금 반올림 때문에 실제 데이터보다 큰 빈 시간 범위가 표시되지 않는다.
`Pivots per call`의 x축 라벨은 `Simplex call order`이며,
시간이 아니라 solver 실행 시작부터 센 Simplex 호출 순번이다. 각 x 위치에서
y값은 해당 한 호출 안에서 수행된 실제 pivot 수다.
iteration은 원본 JSONL 진단 필드에는 유지하지만 그래프에는 표시하지 않는다.

원본 progress JSONL의 모든 `simplex_start`, `simplex_iteration`, `simplex_pivot`,
`simplex_end` 이벤트에는 1부터 증가하는 `call_number`가 기록된다. 특히
`simplex_end` 한 줄에는 종료 `timestamp`, `call_number`, `result`,
`duration_seconds`, `iterations`, `pivots`가 함께 있어 몇 번째 호출이 언제
SAT/UNSAT/한도 종료됐는지 직접 확인할 수 있다.

실시간 PNG 렌더링은 solver thread와 분리된 background worker에서 수행하며,
여러 이벤트가 빠르게 발생하면 중간 렌더 요청을 합쳐 solver 지연을 줄인다.
DPLL round duration은 `round_start`부터 현재까지의 총 경과시간으로 계산하므로
해당 round에서 실행되는 Theory·Simplex·Reluplex 시간을 모두 포함한다. 진행 중
round는 새 DPLL 이벤트가 없어도 heartbeat로 다시 렌더해 duration을 갱신한다.

## JSONL 이벤트

| event | 주요 field |
|---|---|
| `round_start` | `round` |
| `theory_selection` | `round`, `atoms[{atom_id, polarity, kind, expression}]` |
| `simplex_start` | `call_id`, `call_number`, `round`, `depth`, `origin`, `theory_atom_ids`, row/variable 수 |
| `simplex_iteration` | `call_id`, `call_number`, `iteration`, `violated_var`, value/lower/upper |
| `simplex_pivot` | `call_id`, `call_number`, `iteration`, `entering`, `leaving`, `row` |
| `simplex_end` | `call_id`, `call_number`, `result`, `duration_seconds`, `iterations`, `pivots` |
| `theory_result` | `round`, `result`, conflict clause 크기 등 |
| `round_end` | `round`, `result`, `duration_seconds`, `simplex_calls` |
| `solver_end` | solver `status`, `reason`, `rounds` |

## 종료 후 feedback

feedback-only mode는 progress JSONL 전체를 집계해 DPLL+Theory와 Simplex의 두
PNG와 공통 JSON summary를 저장한다. `both` mode에서는 종료 시점의 realtime
PNG가 같은 내용을 담으므로 별도 feedback PNG를 만들지 않고 그 한 벌을 공유한다.
ReLU의 heatmap, realtime, full-history feedback은 서로 다른 그림이므로 모두
유지한다. JSON에는 round/call 상세, 누적 iteration/pivot 수, 상위 위반 변수,
entering/leaving 변수, pivot row, theory 연결 결과가 들어 있다.

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
feedback이 켜지면 ReLU split heatmap/history와 solver 공통 JSON이 생성된다.
feedback-only에서는 solver PNG 두 장도 만들고, `both`에서는 realtime solver
PNG 두 장을 그대로 공유한다.

## 출력 경로 옵션

| option | 용도 |
|---|---|
| `--solver-progress-log-output` | 원본 JSONL 경로 |
| `--solver-realtime-output` | solver 패널의 base PNG 경로; `_dpll_theory`, `_simplex` 파일 생성 |
| `--solver-feedback-output` | feedback-only 패널의 base PNG 경로; `both`에서는 realtime PNG를 공유하고 이 stem의 JSON만 생성 |

통합 realtime HTML은 ReLU 패널을 선택하면 `--realtime-split-output` PNG와 같은
stem으로 생성한다. solver 패널만 선택하면 `--solver-realtime-output`과 같은
stem으로 생성하며, 선택 패널 수와 관계없이 HTML은 하나다.

## 주의사항

- progress는 visualization을 켰 때만 수집한다.
- iteration 단위 JSONL 기록은 아예 계측하지 않는 실행보다 I/O를 늘린다.
- realtime PNG 렌더링은 background worker가 담당하지만 JSONL append 비용은
  solver 실행 시간에 포함된다.
- ReLU split이 없어도 DPLL round와 Simplex progress는 theory solver가 호출되면
  기록된다.
- ReLU split 재귀 깊이 상한은 `max(50, 모델 hidden ReLU 수 + 1)`로만
  자동 계산하며 CLI나 spec에서 수동 지정하지 않는다.
