# ReLU split visualization 모드

현재 visualization은 목적에 따라 실시간과 사후 피드백의 두 형태로 구분한다.
피드백은 solver 동작을 변경하는 모드가 아니라, 실행 결과를 다음 분석에 활용하기
위한 사후 visualization을 뜻한다.

`AutoVerify.py verify-vnnlib`에서는 두 기능을 독립 플래그로 선택한다.
실시간 모드는 통합 화면 하나가 아니라 ReLU split, DPLL+Theory, Simplex의
전용 브라우저 화면 세 개를 각각 연다.
`--realtime-panels relu`, `--realtime-panels dpll-theory`,
`--realtime-panels simplex`로 실행할 시각화 패널을 선택한다. `both` 모드에서는
같은 선택을 realtime과 feedback 양쪽에 적용한다. 선택하지 않은 패널의 logger,
PNG, HTML은 생성하지 않는다.
기본 갱신 확인 주기는 500ms이며 `--realtime-refresh-ms`로 조정할 수 있다.
`--realtime-playback-ms N`을 주면 solver는 지연시키지 않고 ReLU `+/-` 로그를
시각화에서만 N ms 간격으로 하나씩 재생한다. `0`은 최신 상태를 표시하는 기본
live 동작이다.

| mode | 실시간 | 사후 feedback |
|---|---:|---:|
| `off` | X | X |
| `realtime` | O | X |
| `feedback` | X | O |
| `both` | O | O |

## 실시간 visualization

`visualization/RealtimeSplitVisualization.py`가 담당한다.

- Reluplex split 시작 직전에 `+` 이벤트 기록
- split 종료 시 `-` 이벤트 기록
- 현재 시각, `split_id`, layer/index, 변수 이름 저장
- append마다 visualizer update callback 호출
- 최근 30초의 전체 활성 split 수를 시간 이력으로 표시
- `(layer, index)` 위치에 현재 활성 ReLU와 뉴런별 누적 split 횟수를 표시
- 연속 update 요청은 background renderer에서 합쳐 solver의 PNG 저장 대기를 줄임

각 로그 줄은 JSON 객체다.

```json
{"timestamp":"2026-08-09T12:34:56.123456+09:00","time_ns":1786246496123456000,"split_id":"...-1","layer":2,"index":7,"variable":"z2_7_run","event":"+"}
```

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

기존 `visualization/SolveTrace.py`와 `visualization/SplitHeatmap.py`가 담당한다. solver에 별도
feedback/Groovy 설정을 전달하지 않는다.

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

출력 경로를 생략하면 모델명과 property명을 조합해 `visualization/outputs/`에
실시간 JSONL, 실시간 PNG와 feedback PNG를 각각 만든다. 필요하면
`--realtime-log-output`, `--realtime-split-output`,
`--feedback-heatmap-output`으로 덮어쓴다. 기존 `--split-heatmap-output`도 같은
feedback 출력 옵션으로 계속 지원한다.

이 방식은 실행이 끝난 뒤 어느 뉴런에서 split이 발생했는지 누적 횟수로 보여준다.
결과를 보고 다음 실험이나 solver 개선 방향을 정한다는 의미에서 피드백
visualization으로 분류한다.

## 현재 범위

- Groovy bound tightening callback은 사용하지 않는다.
- split 횟수 Top 10을 solver에 다시 주입하지 않는다.
- 기본 DPLL/Reluplex 탐색 결과는 visualization 때문에 변경되지 않는다.
- split 전에 끝나는 실행은 heatmap 값이 0이므로, 별도로 DPLL/BCP/Simplex 단계
  visualization을 확장하는 것이 후속 과제다.
