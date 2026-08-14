# 변경사항 — 2026-08-09

## Visualization 구조 정리

visualization 관련 소스, 계측, 테스트, 문서와 기본 출력물을 루트의
`visualization/` 패키지로 통합했다.

```text
visualization/
├── SolveTrace.py
├── RealtimeSplitVisualization.py
├── SplitHeatmap.py
├── NetworkLayout.py
├── visualize_prop.py
├── VisualizeSparsity.py
├── tests/
└── outputs/
```

- `Automation/SolveTrace.py`를 `visualization/SolveTrace.py`로 이동
- 실시간 split visualizer를 Automation에서 visualization으로 이동
- 루트의 layout, heatmap, Prop/CNF, sparsity visualizer를 모두 이동
- visualization 테스트를 `visualization/tests/`로 이동
- 관련 가이드를 `visualization/`으로 이동
- 기존 DOT/PNG와 precise Prop 출력물을 `visualization/outputs/`으로 이동
- 기본 실시간 로그와 기본 PNG/DOT 출력 위치를 `visualization/outputs/`으로 통일
- 전체 solver 및 Automation import를 `visualization.*` 기준으로 갱신

## 실시간 ReLU Split Visualization

- `ReLUProp` AST에 선택적 layer/index 메타데이터 추가
- 신경망 인코딩 시 실제 ReLU layer/index 저장
- 첫 Reluplex 분기 직전에 split count 증가 및 `+` 이벤트 기록
- 분기 종료 시 `finally`에서 count 감소 및 같은 `split_id`의 `-` 이벤트 기록
- 이벤트에 시각, 나노초 시각, split ID, layer/index와 변수명 저장
- 로그 append마다 update callback 호출
- 현재 열린 split 수만큼 점을 그리는 실시간 visualizer 추가

## 피드백 Visualization 정의

기존 `SolveTrace → SplitHeatmap` 경로를 실행 후 결과를 분석하는 사후 피드백
visualization으로 정의했다. solver 탐색을 변경하는 Groovy callback, Top 10 뉴런
선정과 solver 내부 `split_mode="feedback"` 설정은 제거했다. 대신 AutoVerify의
실시간과 피드백은 `--visualization-mode off|realtime|feedback|both`로 선택한다.
`both`에서는 동일한 solver 실행에 실시간 logger와 기존 SolveTrace를 함께 연결한다.

## Reluplex 수정

ReLU 음수 phase의 `y=0` 적용이 기존 `y` bound를 덮어쓰지 않도록 수정했다.
기존 bound와 `(0,0)`의 교집합이 비면 해당 분기를 즉시 UNSAT으로 처리한다.

저장된 ACAS Xu 결과는 모두 split 0회였고, 문서화된 mMIMO 실행도 split 이전
DPLL 단계에서 정체되었으므로 이 오류 수정에 따른 기존 주요 실험 결과 변화는
확인되지 않았다.

## 검증

- visualization 및 Automation 관련 테스트 38개 통과
- 환경 의존 테스트 2개 제외
- `Automation/run_large_model.py --help` 정상 동작
- `python -m visualization.SplitHeatmap --help` 정상 동작
- `python -m visualization.NetworkLayout --help` 정상 동작
- Python compile 및 `git diff --check` 통과
