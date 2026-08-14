# 2026-08-10 ReLU Split Visualization 변경 기록

## 1. 개요

기존 ReLU split visualization을 다음 두 채널로 분리하고, 두 기능을
한 solver 실행에서 동시에 사용할 수 있게 정리했다.

| 모드 | 의미 | 주요 출력 |
|---|---|---|
| `off` | visualization 끄기 | 없음 |
| `realtime` | 실행 중 활성 ReLU split 추적 | JSONL, PNG, 자동 갱신 HTML |
| `feedback` | 실행 종료 후 누적 split 분석 | 뉴런별 split heatmap PNG |
| `both` | realtime과 feedback 동시 사용 | 위 두 채널의 모든 출력 |

CLI는 `--visualization-mode off|realtime|feedback|both` 하나로 모드를 선택한다.

## 2. 실시간 화면 변경

### 기존 방식

- 현재 활성 split을 개별 빨간 점으로 표시했다.
- PNG는 이벤트마다 덮어썼지만 일반 이미지 뷰어는 자동으로 새로고침하지
  않아 실시간 관찰이 어려웠다.

### 변경된 방식

- 작업관리자 성능 그래프처럼 최근 30초의 활성 split 수를 계단형
  선과 면적으로 표시한다.
- split `+` 이벤트가 생기면 선이 올라가고, 같은 `split_id`의 `-`
  이벤트가 생기면 선이 내려간다.
- 브라우저 live view가 자동으로 열리고 300ms 간격으로 최신 PNG를
  다시 읽는다.
- `ACTIVE`, `started`, `completed` 수치를 같이 표시한다.
- 새 검증을 시작할 때 기존 realtime JSONL을 초기화해 이전 실행의
  미완료 split이 현재 화면에 섞이지 않게 했다.
- PNG는 임시 파일에 먼저 저장한 뒤 원자적으로 교체해, 브라우저가
  작성 중인 불완전한 PNG를 읽을 가능성을 줄였다.

## 3. 그래프 축과 수치의 의미

| 표시 | 의미 |
|---|---|
| 가로축 `Last 30s` | 오른쪽 `0`이 현재, 왼쪽 `-30`이 30초 전 |
| 세로축 `Active ReLU splits` | 그 시점에 시작됐지만 아직 종료되지 않은 split 수 |
| 파란 선 상승 | 새 ReLU split 시작 |
| 파란 선 하락 | 해당 ReLU split 종료 |
| `ACTIVE` | 현재 진행 중인 split 수 |
| `started` | 현재 실행에서 시작된 누적 split 수 |
| `completed` | 현재 실행에서 종료된 누적 split 수 |

개별 split의 `layer`, `index`, 변수명, 시작·종료 시각은 JSONL에 남는다.

```json
{"timestamp":"...","time_ns":0,"split_id":"...","layer":1,"index":3,"variable":"z1_3","event":"+"}
```

## 4. 실행 흐름

```text
Reluplex split 시작/종료
        ↓
SplitEventLogger가 +/- JSONL 기록
        ↓
RealtimeSplitVisualizer가 30초 이력 PNG 교체
        ↓
HTML live view가 300ms 간격으로 PNG 재로드
        ↓
브라우저에 실시간 선 그래프 표시
```

feedback을 같이 켜면 동일한 solver 실행의 `SolveTrace`를 집계해 종료 후
뉴런별 누적 split heatmap도 추가로 저장한다.

## 5. 수정한 주요 파일

### `visualization/RealtimeSplitVisualization.py`

- `SplitEventLogger.reset_log` 추가
- `draw_realtime_split_history()` 추가
- 최근 30초 계단형 선 그래프와 누적 수치 표시
- 기존 `draw_active_split_dots()`는 호환용 별칭으로 유지
- HTML live view 생성 및 기본 브라우저 자동 열기
- 300ms 이미지 재로드
- 임시 PNG 저장 후 `os.replace()`로 교체

### `DPLL_T.py`

- `split_mode="realtime"`일 때 `SplitEventLogger`와 visualizer를 solver에 연결
- `realtime_open_view` 옵션 추가
- 실시간 실행 시 JSONL을 새 session으로 초기화
- solver의 ReLU layer/index metadata를 realtime logger에 전달

### `Automation/AutoVerify.py`

- `--visualization-mode off|realtime|feedback|both`로 CLI 모드 통합
- CLI realtime 실행 시 live view 자동 열기
- 결과에 JSONL, PNG, HTML live view 경로 추가
- feedback 모드의 누적 split heatmap 생성 유지

### `visualization/realtime_split_demo.py`

- solver의 난이도와 무관하게 live view를 확실히 볼 수 있는 제어된 데모 추가
- 활성 split 수를 일정한 간격으로 늘렸다 줄여 상승·하강 선을 표시
- `--count`, `--interval`, `--hold-seconds`, `--output-dir`, `--no-open` 지원
- 이 데모는 visualization 동작 확인용이며 solver 검증 결과가 아님

### `visualization/tests/test_split_modes.py`

- `+/-` 쌍과 활성 split 집계 테스트
- 실행 시 로그 초기화 테스트
- realtime/feedback 독립 조합 테스트
- HTML 자동 갱신 및 브라우저 1회 열기 테스트
- 다중 split 제어 데모 테스트
- 점이 아닌 연결된 선 그래프 생성 테스트

### `visualization/README.md`

- 독립 realtime/feedback 사용법
- 자동 갱신 live view와 제어 데모 명령 추가

## 6. 실행 명령

### 6.1 선 그래프를 확실히 보는 제어 데모

`C:\AI_Verification\jnunnv\Ai_verification` 루트에서 실행한다.

```powershell
python -m visualization.realtime_split_demo --count 8 --interval 1 --hold-seconds 5
```

예상 동작:

1. 브라우저 live view가 자동으로 열린다.
2. 활성 split 수가 1초마다 `0 → 8`로 상승한다.
3. 8인 상태를 5초 유지한다.
4. 활성 split 수가 1초마다 `8 → 0`으로 하강한다.

`Automation` 폴더에서 시작했다면 먼저 루트로 이동한다.

```powershell
cd ..; python -m visualization.realtime_split_demo --count 8 --interval 1 --hold-seconds 5
```

### 6.2 ACAS Xu realtime + feedback

`Ai_verification` 루트에서:

```powershell
python Automation\run_large_model.py verify-vnnlib --model Custom\ACASXU_experimental_v2a_1_1_custom.bin --vnnlib ..\neuralsat\src\example\vnnlib\prop_1.vnnlib --allow-large-model --timeout-seconds 300 --visualization-mode both
```

ACAS Xu property 1은 문제 크기와 solver 경로에 따라 split 전에 오래 걸리거나
split 0회로 종료할 수 있다. 따라서 화면 동작 확인은 위 제어 데모를
먼저 사용하는 것이 적합하다.

## 7. 출력 파일

기본 출력은 `visualization/outputs/` 아래에 생성된다.

| 파일 | 용도 |
|---|---|
| `*_realtime.jsonl` | split 시작·종료 원본 이벤트 |
| `*_realtime.png` | 최근 30초 활성 split 선 그래프 |
| `*_realtime.html` | 300ms 자동 갱신 브라우저 live view |
| `*_feedback.png` | 실행 종료 후 뉴런별 누적 split heatmap |

제어 데모는 다음 고정 이름을 사용한다.

```text
visualization/outputs/realtime_split_demo.jsonl
visualization/outputs/realtime_split_demo.png
visualization/outputs/realtime_split_demo.html
```

## 8. 검증 결과

2026-08-10에 다음 명령으로 최종 확인했다.

```powershell
python -m unittest visualization.tests.test_split_modes Automation.tests.test_automation
python -m py_compile visualization\RealtimeSplitVisualization.py visualization\realtime_split_demo.py DPLL_T.py Automation\AutoVerify.py
```

결과:

```text
Ran 49 tests
OK (skipped=2)
```

- Python 구문 검사 통과
- realtime 모드 JSONL/PNG/HTML 생성 확인
- realtime + feedback 동시 출력 확인
- 브라우저 열기가 여러 split 이벤트에서도 한 번만 호출되는지 확인
- 제어 데모의 `+,+,...,-,-,...` 이벤트와 최종 active 0 확인
- 실제 생성 PNG를 통해 최근 30초 선 그래프 상승·하강 형태 확인

## 9. 현재 제한사항

- realtime 그래프는 **전체 solver 진행률**이 아니라 **ReLU split
  활성 수**만 보여 준다.
- `encode_nn`, VNNLIB parsing, Tseitin CNF, DPLL 호출 수 같은 전체 단계 진행은
  `--profile-stages`로 별도 확인해야 한다.
- split이 하나도 없으면 선은 0으로 유지된다. 이는 visualization 오류가
  아니라 solver가 ReLU branch에 도달하지 않았거나 split 없이 결론을 낸 것이다.
- `realtime_split_demo` 이벤트는 화면 검증을 위해 제어된 이벤트이며 실제
  신경망 안전성 검증 결과로 해석하면 안 된다.
- CLI는 realtime을 켜면 live view를 자동으로 연다. Python API를 직접 호출할
  때는 `open_realtime_view=True`를 명시해야 한다.
- ONNX 모델을 사용하려면 Python 환경에 `onnx` 패키지가 필요하다. Custom
  `.bin` 모델은 해당 의존성 없이 로드할 수 있다.

## 10. 상태

위 변경은 현재 로컬 작업 트리에 반영되어 있으며, 별도 commit은 수행하지
않았다.

## 11. DPLL(T) / Simplex Progress 추가

ReLU split과 같은 realtime/feedback 모드에 다음 solver 내부 계측을
추가했다.

- DPLL round 번호, round별 실행 시간, 종료 결과
- round별 Simplex 호출 수
- Simplex 호출별 실행 시간과 `SAT` / `UNSAT` / `ITERATION_LIMIT` /
  `TIMEOUT` 결과
- Simplex iteration 수와 pivot 수
- 반복적으로 bound를 위반하는 basic variable
- entering/leaving variable 선택 빈도
- 반복적으로 pivot되는 row
- `DPLL round → theory atoms → Simplex calls → theory result` 연결

원본 이벤트는 `*_solver_progress.jsonl`에 저장하고, realtime은
7패널 dashboard PNG/HTML을, feedback은 최종 dashboard PNG와 집계 JSON을
생성한다.

### 추가된 이벤트

```text
round_start / round_end
theory_selection / theory_result
simplex_start / simplex_iteration / simplex_pivot / simplex_end
solver_end
```

### 추가된 파일

- `visualization/SolverProgress.py`: logger, JSONL parser, feedback 집계,
  realtime/feedback dashboard
- `visualization/solver_progress_demo.py`: round·Simplex·theory 제어 데모
- `visualization/SOLVER_PROGRESS_GUIDE.md`: event schema, 패널, 실행법,
  출력 설명
- `visualization/tests/test_solver_progress.py`: 연결, Simplex 상태,
  realtime/feedback 생성 테스트

### dashboard 패널

1. DPLL round 시간·결과·Simplex 호출 수
2. Simplex 호출 시간·결과
3. Simplex 호출별 iteration/pivot
4. 반복 bound 위반 변수
5. entering/leaving variable 빈도
6. pivot row 빈도
7. DPLL/theory/Simplex/result 연결 내역

### 실행 명령

dashboard 제어 데모:

```powershell
python -m visualization.solver_progress_demo --interval 0.2
```

AutoVerify realtime + feedback:

```powershell
python Automation\run_large_model.py verify-vnnlib --model Custom\ACASXU_experimental_v2a_1_1_custom.bin --vnnlib ..\neuralsat\src\example\vnnlib\prop_1.vnnlib --allow-large-model --timeout-seconds 300 --visualization-mode both
```

### 렌더링 성능

Simplex iteration은 짧은 시간에 많은 이벤트를 만들 수 있으므로,
dashboard PNG 렌더러를 background worker로 분리했다. 이벤트가 렌더링
속도보다 빠르면 중간 요청을 합쳐 처리하고, solver 종료 시 `flush()`로
최종 화면을 보장한다. JSONL은 모든 iteration/pivot 이벤트를 유지한다.

## 12. ReLU / Solver Realtime HTML 통합

기존에는 ReLU split과 DPLL/Simplex progress가 각각 HTML을 만들어 브라우저
탭이 두 개 열렸다. `visualization/UnifiedRealtimeDashboard.py`를 추가해
AutoVerify realtime은 이제 `*_realtime.html` 하나만 생성·오픈한다.

- 1구역: `DPLL Rounds & Theory Flow`—round 정보와 theory 전달 흐름
- 2구역: Simplex 호출·iteration·pivot 내부 통계
- 3구역: Realtime ReLU split history
- 세 이미지 모두 300ms 간격으로 자동 재로드
- AutoVerify에서 개별 split/solver HTML 생성 및 자동 열기 중지

solver realtime/feedback PNG는 DPLL+Theory와 Simplex 파일로 나누어 생성한다.
