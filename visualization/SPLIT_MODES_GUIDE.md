# ReLU split visualization 모드

현재 visualization은 목적에 따라 실시간과 사후 피드백의 두 형태로 구분한다.
피드백은 solver 동작을 변경하는 모드가 아니라, 실행 결과를 다음 분석에 활용하기
위한 사후 visualization을 뜻한다.

`AutoVerify.py verify-vnnlib`에서는 두 기능을 독립 플래그로 선택한다.
실시간 모드는 기본적으로 ReLU split, DPLL+Theory, Simplex 세 구역을 하나의
브라우저 HTML에서 함께 연다. `--panels relu`,
`--panels dpll-theory`, `--panels simplex`로 필요한 구역만
선택해도 HTML은 하나만 생성된다. `both` 모드에서는 같은 선택을 realtime과
feedback 양쪽에 적용한다. 선택하지 않은 패널의 logger와 PNG는 생성하지 않는다.
기본 갱신 확인 주기는 500ms이며 `--realtime-refresh-ms`로 조정할 수 있다.
`--realtime-playback-ms N`을 주면 solver는 지연시키지 않고 ReLU `+/-` 로그를
시각화에서만 N ms 간격으로 하나씩 재생한다. `0`은 최신 상태를 표시하는 기본
live 동작이다.
`--realtime-window-seconds N`은 현재를 0초로 둔 x축의 고정 시간 폭을 정한다
(기본 300초/5분). 실행 시간이 길어져도 x축은 `-N ... 0` 범위로 유지된다.

| mode | 실시간 | 사후 feedback |
|---|---:|---:|
| 옵션 생략 (기본) | X | X |
| `realtime` | O | X |
| `feedback` | X | O |
| `both` | O | O |

## 실시간 visualization

`visualization/RealtimeSplitVisualization.py`가 담당한다.

- Reluplex split 시작 직전에 `+` 이벤트 기록
- split 종료 시 `-` 이벤트 기록
- 현재 시각, `split_id`, layer/index, 변수 이름 저장
- append마다 visualizer update callback 호출
- 현재를 오른쪽 0초로 둔 고정 폭 슬라이딩 창에 활성 split 이력을 표시
- `0s: 0` 같은 점 라벨은 표시하지 않고 x축에는 고정된 시간 눈금만 표시
- `(layer, index)` 위치에 현재 활성 ReLU와 뉴런별 누적 split 횟수를 표시
- 그래프 아래에 입력·은닉·출력 레이어별 전체 뉴런 수와 총 ReLU 수를 표시
- 연속 update 요청은 background renderer에서 합쳐 solver의 PNG 저장 대기를 줄임

각 로그 줄은 JSON 객체다.

```json
{"timestamp":"2026-08-09T12:34:56.123456+09:00","time_ns":1786246496123456000,"split_id":"...-1","layer":2,"index":7,"variable":"z2_7_run","event":"+"}
```

JSONL 옆의 `*_meta.json`에는 모델의 전체 레이어 크기를 저장한다. 따라서 실행이
끝난 뒤에도 이벤트와 모델 정보를 함께 불러와 전체 과정을 다시 재생할 수 있다.

```powershell
python -m visualization.replay_split_log `
    "visualization/outputs/model_property_split.jsonl" `
    --interval-ms 250 `
    --window-seconds 300 `
    --open
```

재생 시 각 이벤트의 기록 시각을 가상 현재 시각으로 사용하므로 오래된 로그도
첫 이벤트부터 마지막 이벤트까지 고정 시간창 안에서 순서대로 표시된다.

사용 예:

```python
result = dpll_t_detailed(
    formula,
    split_mode="realtime",
    split_log_path="visualization/outputs/log.txt",
    realtime_output_path="visualization/outputs/realtime_splits.png",
)
```

조기 SAT 반환, timeout, 예외가 발생해도 `finally`에서 같은 `split_id`의 `-`를
기록한다. 따라서 열린 split 카운트가 실행 종료 후 남지 않는다.

## 사후 피드백 visualization

feedback에서는 두 종류의 ReLU 결과를 함께 만든다.

- 기존 split heatmap: 레이어/뉴런별 누적 split 횟수
- 전체 기간 split history: 실시간 ReLU dashboard와 같은 구성으로, 첫 split부터
  마지막 split까지 실제 timestamp 차이를 사용해 시간축을 자르지 않고 표시

```python
from visualization.SolveTrace import SolveTrace
from DPLL_T import dpll_t_detailed
from visualization.SplitHeatmap import split_counts_from_trace, draw_split_heatmap

trace = SolveTrace()
result = dpll_t_detailed(formula, trace=trace)
counts = split_counts_from_trace(trace)
fig, ax = draw_split_heatmap(model, counts)
fig.savefig("visualization/outputs/split_heatmap.png")
```

`AutoVerify.py verify-vnnlib`에서는 `--visualization-mode both`를 주면 동일한
solver 실행에서 두 visualization이 모두 동작한다.

```powershell
python Automation\run_large_model.py verify-vnnlib `
    --model model.onnx `
    --vnnlib prop.vnnlib.gz `
    --allow-large-model `
    --visualization-mode both
```

출력 경로를 생략하면 `visualization/outputs/시작일시~종료일시_신경망이름/`
실행별 폴더에 실시간 JSONL, 실시간 PNG, 기존 feedback heatmap과 전체 기간
history PNG를 함께 만든다. 실행 중 폴더명의 종료시각은 `running`이고 모든
feedback 생성이 끝난 뒤 실제 종료시각으로 바뀐다. 필요하면
`--realtime-log-output`, `--realtime-split-output`,
`--feedback-heatmap-output`으로 덮어쓴다. 기존 `--split-heatmap-output`도 같은
heatmap 출력 옵션으로 계속 지원한다. 전체 기간 history 경로는
`--relu-feedback-output`으로 덮어쓴다.

heatmap은 어느 뉴런에서 split이 발생했는지 누적 횟수로 보여주고, 전체 기간
history는 5분 실시간 창에서 왼쪽으로 사라졌던 과거 split까지 모두 보여준다.
결과를 보고 다음 실험이나 solver 개선 방향을 정한다는 의미에서 피드백
visualization으로 분류한다.

