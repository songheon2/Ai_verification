# 2026-08-10 DPLL(T) / Simplex Progress Visualization 추가 내역

## 개요

ReLU split visualization에서 사용하는 `realtime`, `feedback`, `both`
모드에 DPLL(T) round·theory atom·Simplex 내부 동작 계측을 추가했다.

```text
DPLL round
  └─ theory atom 선택
       └─ Simplex call
            ├─ iteration / bound violation
            ├─ entering / leaving variable
            └─ pivot row
       └─ theory result
```

## 수집하는 정보

### DPLL round

- 현재 round 번호
- round별 실행 시간
- round별 Simplex 호출 수
- round 종료 결과
  - `BOOLEAN_UNSAT`
  - `THEORY_TRIVIAL`
  - `THEORY_SAT`
  - `THEORY_UNSAT`
  - `THEORY_CONFLICT`
  - `TIMEOUT`
  - 기타 solver limit

### Simplex 호출

- 호출 ID
- 해당 호출이 속한 DPLL round
- Reluplex recursion depth
- 호출 위치
  - `reluplex`
  - `relu_repair`
- 연결된 theory atom ID
- tableau row 수와 variable 수
- 호출별 실행 시간
- 호출별 iteration/pivot 수
- 종료 결과
  - `SAT`
  - `UNSAT`
  - `ITERATION_LIMIT`
  - `TIMEOUT`

### Simplex 내부

- iteration 번호
- bound를 위반한 basic variable
- 위반 변수의 현재값·lower bound·upper bound
- pivot entering variable
- pivot leaving variable
- 반복적으로 pivot되는 row

### DPLL ↔ Theory 연결

- round에서 선택한 atom ID
- atom polarity
- atom 종류
  - `InequProp`
  - `ReLUProp`
- atom 표현식
- atom 선택으로 시작된 Simplex call
- Simplex 결과가 DPLL에 반환된 theory result

## Progress JSONL

모든 계측의 원본은 `*_solver_progress.jsonl`에 기록한다.

| event | 의미 |
|---|---|
| `round_start` | DPLL round 시작 |
| `theory_selection` | round의 theory atom 선택 |
| `simplex_start` | Simplex 호출 시작 |
| `simplex_iteration` | iteration과 bound 위반 변수 |
| `simplex_pivot` | entering/leaving variable과 pivot row |
| `simplex_end` | Simplex 시간·iteration·pivot·결과 |
| `theory_result` | Simplex/Reluplex에서 DPLL로 돌아간 결과 |
| `round_end` | round 시간·Simplex 호출 수·결과 |
| `solver_end` | 최종 solver status·reason·round 수 |

JSON에 표현할 수 없는 무한대 bound는 `Infinity`, `-Infinity`
문자열로 저장한다.

## 실시간 dashboard

`--visualization-mode realtime` 또는 `both`를 선택하면 통합 HTML이 자동으로 열린다.
HTML 안은 `DPLL Rounds & Theory Flow`, `Simplex Internals`,
`Realtime ReLU Split History` 세 구역으로 나누며, 세 PNG를 같은 300ms
주기로 새로고침한다. AutoVerify에서 개별 HTML은 생성하거나 열지 않는다.

dashboard는 다음 7개 패널로 구성한다.

1. `DPLL rounds`
   - round별 실행 시간
   - round 결과
   - round별 Simplex 호출 수
2. `Simplex call duration`
   - 호출별 실행 시간
   - SAT/UNSAT/limit/timeout 색상 구분
3. `Simplex work per call`
   - iteration 수
   - pivot 수
4. `Repeated bound violations`
   - 반복적으로 선택된 위반 변수
5. `Entering / leaving variable frequency`
   - entering/leaving 변수 선택 빈도
6. `Repeatedly pivoted rows`
   - pivot row 반복 빈도
7. `DPLL → theory atoms → Simplex calls → theory result`
   - round과 theory/Simplex 결과의 연결 내역

HTML은 300ms 간격으로 최신 dashboard PNG를 다시 읽는다.

## 실시간 렌더링 성능

Simplex iteration은 짧은 시간에 많은 이벤트를 만들 수 있으므로
matplotlib PNG 렌더링을 solver thread에서 직접 수행하지 않는다.

- background renderer thread 사용
- 렌더링 중 들어온 중간 요청 coalescing
- solver 종료 시 `flush()`로 최종 dashboard 저장 보장
- PNG를 임시 파일에 저장한 뒤 `os.replace()`로 교체

JSONL은 최종 feedback을 위해 iteration/pivot 이벤트를 생략하지 않는다.

## Feedback

`--visualization-mode feedback` 또는 `both`를 선택하면 종료 후 다음 파일을 생성한다.

| 파일 | 내용 |
|---|---|
| `*_solver_feedback_dpll_theory.png` | DPLL round 정보와 DPLL↔Theory↔Simplex 흐름 |
| `*_solver_feedback_simplex.png` | Simplex 호출·iteration·pivot 내부 통계 |
| `*_solver_feedback.json` | round/call 상세와 누적 상위 통계 |

feedback JSON의 주요 항목:

- `solver`
- `rounds`
- `simplex_calls`
- `counts`
- `top_violated_variables`
- `top_entering_variables`
- `top_leaving_variables`
- `top_pivot_rows`
- `theory_links`

## 수정·추가된 파일

| 파일 | 변경 내용 |
|---|---|
| `Simplex.py` | iteration, violation, pivot, 종료 상태 계측 |
| `Reluplex.py` | Simplex call에 recursion depth·origin·theory context 전달 |
| `DPLL_T.py` | round/theory 이벤트와 Simplex 연결 |
| `Automation/AutoVerify.py` | realtime/feedback logger·dashboard·결과 경로 연결 |
| `visualization/VisualizationMode.py` | solver progress 기본 출력 경로 추가 |
| `visualization/SolverProgress.py` | 공통 logger, 집계, DPLL+Theory/Simplex 패널, feedback |
| `visualization/UnifiedRealtimeDashboard.py` | DPLL+Theory/Simplex/ReLU 세 구역을 하나의 HTML로 통합 |
| `visualization/solver_progress_demo.py` | dashboard 제어 데모 |
| `visualization/tests/test_solver_progress.py` | 계측·상태·출력 테스트 |

## 실행 명령

### Dashboard 제어 데모

`C:\AI_Verification\jnunnv\Ai_verification` 루트에서:

```powershell
python -m visualization.solver_progress_demo --interval 0.2
```

제어 데모는 round 3개, Simplex call 5개, iteration/pivot, theory 연결,
`SAT`/`UNSAT`/`ITERATION_LIMIT` 상태를 확실히 표시한다. 실제
안전성 검증 결과가 아니라 visualization 확인용이다.

### ACAS Xu realtime + feedback

```powershell
python Automation\run_large_model.py verify-vnnlib --model Custom\ACASXU_experimental_v2a_1_1_custom.bin --vnnlib ..\neuralsat\src\example\vnnlib\prop_1.vnnlib --allow-large-model --timeout-seconds 300 --visualization-mode both
```

## CLI 출력 경로 옵션

| 옵션 | 의미 |
|---|---|
| `--solver-progress-log-output` | solver progress JSONL 경로 |
| `--solver-realtime-output` | 실시간 패널 base 경로; `_dpll_theory`, `_simplex` PNG 생성 |
| `--solver-feedback-output` | feedback 패널 base 경로; `_dpll_theory`, `_simplex` PNG 생성 |

통합 HTML은 `--realtime-split-output` PNG와 같은 stem으로 생성하고,
feedback JSON은 `--solver-feedback-output` base PNG와 같은 stem으로 생성한다.

## 테스트 결과

```powershell
python -m unittest Automation.tests.test_automation visualization.tests.test_split_modes visualization.tests.test_split_heatmap visualization.tests.test_solver_progress visualization.tests.test_unified_realtime_dashboard
```

```text
Ran 49 tests
OK (skipped=2)
```

확인한 항목:

- DPLL round·theory atom·Simplex call ID 연결
- Simplex SAT/UNSAT/ITERATION_LIMIT/TIMEOUT 기록
- iteration/pivot/violation/entering/leaving/pivot row 집계
- DPLL+Theory/Simplex 실시간 PNG와 단일 통합 HTML 생성
- DPLL+Theory/Simplex feedback PNG와 공통 JSON 생성
- ReLU split realtime + solver progress realtime 동시 생성
- AutoVerify realtime에서 브라우저 HTML이 하나만 생성되는지 확인
- split heatmap + solver progress feedback 동시 생성

## 제한사항

- progress는 visualization이 활성화된 실행에서만 수집한다.
- iteration 단위 JSONL append는 계측을 끄는 실행보다 I/O 비용이 있다.
- dashboard의 round/Simplex 시간은 visualization I/O가 활성화된 상태의
  wall-clock 시간이다.
- ReLU split이 0회여도 theory solver가 호출되면 DPLL/Simplex progress는
  독립적으로 기록된다.
- 제어 데모는 dashboard 확인용이며 실제 solver 결과로 해석하면 안 된다.

## 상태

변경은 현재 로컬 작업 트리에 반영되어 있으며 별도 commit은 수행하지
않았다.
