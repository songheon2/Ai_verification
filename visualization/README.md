# Visualization

solver 계측, 신경망 구조/히트맵, 명제 그래프와 sparsity 출력은 이 패키지에서
관리한다.

## 구조

```text
visualization/
├── SolveTrace.py
├── RealtimeSplitVisualization.py
├── SolverProgress.py
├── UnifiedRealtimeDashboard.py
├── solver_progress_demo.py
├── SplitHeatmap.py
├── NetworkLayout.py
├── visualize_prop.py
├── VisualizeSparsity.py
├── tests/
└── outputs/
```

- `SolveTrace.py`: DPLL(T), Simplex, Reluplex 구간 이벤트 계측
- `RealtimeSplitVisualization.py`: 실시간 split `+/-` 로그와 30초 선 그래프
- `SolverProgress.py`: DPLL과 theory 흐름을 합치고 Simplex를 분리한 패널과 feedback
- `UnifiedRealtimeDashboard.py`: DPLL+Theory/Simplex/ReLU 세 구역을 하나의 realtime HTML로 통합
- `solver_progress_demo.py`: solver progress dashboard 제어 데모
- `SplitHeatmap.py`: 실행 후 뉴런별 누적 split heatmap
- `NetworkLayout.py`: 완전연결 신경망 구조 drawing
- `visualize_prop.py`: Prop/Tseitin CNF의 DOT/PNG 생성
- `VisualizeSparsity.py`: custom 모델의 pruning sparsity 표시
- `tests/`: visualization 전용 테스트
- `outputs/`: 기본 로그, DOT, PNG 출력

## 실행

```powershell
python -m visualization.SplitHeatmap --help
python -m visualization.NetworkLayout --help
python -m visualization.VisualizeSparsity
python -m visualization.visualize_prop
```

자세한 split 모드와 기존 실험 분석은 `SPLIT_MODES_GUIDE.md`와
`SPLIT_HEATMAP_GUIDE_2026-08-03.md`를 참고한다.
DPLL/Theory/Simplex 실행 기록과 분리 dashboard 패널은
`SOLVER_PROGRESS_GUIDE.md`를 참고한다.

AutoVerify에서는 `--visualization-mode off|realtime|feedback|both`로
시각화 모드를 선택한다. `realtime` 또는 `both`이면 브라우저 live view가
자동으로 열린다. 같은 HTML에서 `DPLL Rounds & Theory Flow`,
`Simplex Internals`, `Realtime ReLU Split History` 세 구역으로 나누고
기본 500ms 간격(초당 최대 2회)으로 함께 새로고침한다.
화면 동작만 먼저 확인하려면 다음 제어된 데모를 실행한다.

```powershell
python -m visualization.realtime_split_demo
python -m visualization.solver_progress_demo
```

이 데모들은 solver 결과가 아니라 live view 확인용이다. 첫 데모는
활성 split 수의 30초 선 그래프를, 두 번째 데모는 DPLL+theory·Simplex·ReLU
세 구역 dashboard와 종료 후 feedback을 보여 준다.

```powershell
python Automation\run_large_model.py verify-vnnlib `
    --model model.onnx --vnnlib property.vnnlib.gz `
    --allow-large-model `
    --visualization-mode both
```

`realtime`은 실행 중 화면만, `feedback`은 종료 후 분석만, `both`는 둘 다 만든다.
