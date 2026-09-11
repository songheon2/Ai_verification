# Visualization 요소 요약

현재 AutoVerify 시각화는 **DPLL·Theory, Simplex, ReLU** 세 부분으로 구성된다.

## 주요 화면

| 요소 | 보여주는 내용 | 읽는 방법 |
|---|---|---|
| DPLL·Theory | 라운드별 시간·결과, Simplex 호출 수, 선택한 제약과 이론 솔버 결과의 연결 | 어느 라운드에서 시간이 오래 걸리고 충돌이 발생했는지 확인한다. |
| Simplex | 호출별 시간, 피벗 수, 호출 결과, 반복 선택된 경계 위반 변수 | 선형 제약을 맞추는 과정에서 작업이 집중되는 호출·변수를 확인한다. |
| ReLU realtime | 최근 시간 구간의 활성 분기 수와 뉴런별 분기 상태 | 빨간 농도는 누적 분기 시작 횟수, 파란 테두리는 현재 열린 분기를 나타낸다. |
| ReLU feedback heatmap | 레이어·뉴런별 누적 분기 시작 횟수 | 진할수록 해당 뉴런에서 분기가 자주 시작됐다. 뉴런의 활성값이나 오차 크기는 아니다. |
| ReLU feedback history | 실행 전체 시간에 따른 활성 분기 수 | 탐색 중 동시에 열려 있던 분기의 증가·감소를 확인한다. 누적 분기 횟수 그래프는 아니다. |

ReLU 분기 이벤트의 `+`는 분기 시작, `-`는 해당 분기의 탐색 종료를 뜻한다.

## 변수와 결과 해석

- `z`: ReLU 적용 전 값, `h`: ReLU 출력값, `ineq_slack`: 부등식을 표현하는 보조 변수다.
- Simplex의 **경계 위반**은 해당 변수값이 현재 하한·상한을 벗어난다는 뜻이다. `h = max(0, z)` 관계 위반은 별도의 ReLU 검사다.
- 반복 위반 그래프는 기록된 이벤트에서 선택된 변수의 빈도다. 상세 이벤트가 샘플링되므로 모든 위반의 정확한 총횟수는 아니다. 호출·피벗 총계는 종료 이벤트의 집계값을 사용한다.
- 개별 Simplex 호출의 SAT/UNSAT는 그 선형 문제의 결과다. 전체 신경망 검증의 최종 결과는 `result.json`에서 확인한다.
- `NUMERICAL_FAILURE`, `TIMEOUT` 등은 확정 UNSAT가 아닌 UNKNOWN 사유다. 최근 추가했던 정확 유리수 검증과 수치 복구는 제거되었다.

그래프 하단의 수치 일관성 허용오차는 다음과 같다. 이는 기존 tableau 일관성 검사의 기준이며, 현재 최종 유리수 인증은 수행하지 않는다.

```text
|actual - expected| <= 1e-9 × max(1, |actual|, |expected|)
```

## 모드와 저장 파일

| 모드 | 동작 |
|---|---|
| 옵션 생략 (기본) | 시각화하지 않는다. |
| `realtime` | 실행 중 선택한 패널을 갱신한다. |
| `feedback` | 실행 후 전체 기록을 요약한다. |
| `both` | 두 기능을 함께 사용한다. DPLL·Theory와 Simplex PNG는 공유하며 중복 저장하지 않는다. |

기본 실행 폴더에는 모델·VNNLIB 사본과 `result.json`이 함께 저장된다. 아래 파일은 선택한 모드·패널에 따라 생성된다.

- `solver_*_dpll_theory.png`, `solver_*_simplex.png`: DPLL·Theory 및 Simplex 패널.
- `relu_realtime.png`: 실시간 ReLU 화면.
- `relu_feedback_heatmap.png`, `relu_feedback_history.png`: ReLU 사후 분석.
- `solver_progress.jsonl`, `solver_feedback.json`: 솔버 이벤트와 요약 데이터.
- `relu_events.jsonl`: ReLU 실시간 이벤트 기록. feedback-only는 메모리 기록을 사용하므로 생략될 수 있다.
- `*.html`: 선택한 실시간 패널을 한 화면에 표시하는 대시보드.

## 보조 도구

- `NetworkLayout.py`: 신경망의 레이어·뉴런 연결 구조를 그린다.
- `VisualizeSparsity.py`: 가중치·바이어스가 0인지 시각화해 희소성을 확인한다.
- `visualize_prop.py`: 논리 제약식의 AST를 Graphviz 트리로 그린다.
- `SolveTrace.py`: 단계별 시간과 분기 정보를 기록한다.
- `replay_split_log.py`: 저장된 ReLU 분기 로그를 재생한다.
