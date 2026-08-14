# ACAS Xu·mMIMO Split 히트맵 실행 결과 분석

- 작성일: 2026-08-03
- 분석 대상: ACAS Xu 항공기 충돌 회피 신경망, mMIMO 안테나 선택 신경망
- 관련 코드: `visualization/SplitHeatmap.py`, `Automation/AutoVerify.py`

## 1. 분석 결론

이번에 생성한 두 이미지는 모두 실제 검증 명령의 `--split-heatmap-output`으로 만들어졌다. 그러나 두 실행 모두 기록된 ReLU split은 0회다. 따라서 이미지에 빨간 핫스팟이 없고 구조 전용 이미지와 거의 같은 모습으로 나온 것이 정상이다.

두 실행에서 split이 0회인 원인은 서로 다르다.

| 모델 | 검증 결과 | solver 진행 | split | 이미지가 옅은 이유 |
|---|---|---:|---:|---|
| ACAS Xu | `VERIFIED / UNSAT` | 2 rounds, 24.422초 | 0회 | ReLU 분기 없이 unsafe 영역이 없음을 증명함 |
| mMIMO 안테나 | `UNKNOWN / TIMEOUT` | 0 rounds, 600.187초 | 0회 | 첫 solver round를 끝내기 전에 시간 제한에 도달함 |

즉 ACAS Xu 이미지는 **분기 없이 검증에 성공한 결과**이고, 안테나 이미지는 **분기 단계까지 도달하지 못한 미완료 결과**다. 둘 다 빨간색이 없지만 의미는 같지 않다.

## 2. 히트맵을 읽는 기준

이미지는 다음 좌표계를 사용한다.

```text
가로축 = 신경망 레이어 L0, L1, L2, ...
세로축 = 해당 레이어 안의 neuron index
셀 하나 = 실제 신경망 노드 하나
셀 색상 = 검증 중 해당 ReLU에서 발생한 split 누적 횟수
```

| 색 | 해석 |
|---|---|
| 거의 흰색 | 실제 노드지만 split 0회 |
| 연한 빨간색 | 적은 횟수의 split |
| 진한 빨간색 | 반복적으로 split 대상으로 선택된 핫스팟 |
| 회색 | 해당 레이어에는 존재하지 않는 padding 영역 |

세로축은 모델에서 가장 큰 레이어에 맞춘다. 작은 레이어가 끝난 아래쪽의 회색은 노드가 누락된 것이 아니라 원래 노드가 없는 공간이다.

오른쪽 색상 막대가 `0~1`로 보이더라도 split 1회가 있었다는 뜻은 아니다. 현재 구현은 모든 값이 0일 때도 색상 축을 표시하기 위해 최대값을 1로 둔다. 실제 split 여부는 반드시 JSON의 다음 항목으로 판단해야 한다.

```text
split_summary.total_split_events
split_summary.distinct_neurons
```

## 3. ACAS Xu 항공기 신경망 이미지 분석

### 실행 대상

```text
모델: Custom/ACASXU_experimental_v2a_1_1_custom.bin
속성: ../neuralsat/src/example/vnnlib/prop_1.vnnlib
결과 이미지: Automation/Results/acasxu_prop1_split.png
결과 JSON: Automation/Results/acasxu_prop1_split.json
```

![ACAS Xu property 1 split 결과](Automation/Results/acasxu_prop1_split.png)

### 신경망 구조 때문에 나타난 형태

```text
L0    L1    L2    L3    L4    L5    L6    L7
5  →  50 →  50 →  50 →  50 →  50 →  50 →  5
```

- 레이어가 8개이므로 가로로 8개의 열이 나타난다.
- 가장 큰 레이어가 50개이므로 세로축은 neuron index 0~49를 기준으로 잡힌다.
- `L1`~`L6`은 각각 50개이므로 열 전체가 실제 노드 영역이다.
- 입력층 `L0`과 출력층 `L7`은 각각 5개뿐이다. index 0~4만 실제 노드이고 5~49는 회색 padding이다.
- split 대상이 될 수 있는 은닉 ReLU는 총 300개다.

따라서 이 이미지가 가로로 넓고 세로로 낮으며, 양 끝 열 대부분이 회색인 것은 `5→50×6→5` 구조를 정확히 반영한 결과다.

### 실제 검증 결과

```text
status: VERIFIED
solver status: UNSAT
reason: BOOLEAN_UNSAT
rounds: 2
elapsed: 24.422초
total split events: 0
distinct split neurons: 0
```

`prop_1.vnnlib`이 나타내는 unsafe 영역과 이 신경망의 가능한 동작이 만나지 않는다는 결론을 얻었다. solver는 이 결론을 내는 과정에서 어떤 ReLU도 활성/비활성 두 경우로 나눌 필요가 없었다.

그래서 `L1`~`L6`의 모든 실제 셀이 split count 0의 옅은 색으로 나온다. 이는 검증 실패가 아니며, **split 없이 UNSAT을 증명한 정상적인 검증 결과**다.

## 4. mMIMO 안테나 신경망 이미지 분석

### 실행 대상

```text
모델: ../models/onnx/Baseline mMIMO FC H hard short 80 HTHNN_LAY2_491
      RELU 20241018 PRUNED 0.93_NO_SIGMOID.onnx
속성: ../vnnlib/Baseline mMIMO FC H hard short 80 HTHNN_LAY2_491
      RELU 20241018 PRUNED 0.93_NO_SIGMOID/mmimo_top8_idx0_eps0.001.vnnlib
결과 이미지: Automation/Results/mmimo_split_check.png
결과 JSON: Automation/Results/mmimo_split_check.json
```

![mMIMO 안테나 split 확인 결과](Automation/Results/mmimo_split_check.png)

### 신경망 구조 때문에 나타난 형태

```text
L0 입력층       L1 은닉층       L2 은닉층       L3 출력층
256          → 491           → 491           → 16
```

- 레이어가 4개이므로 가로로 4개의 굵은 열이 나타난다.
- 최대 레이어가 491개이므로 세로축이 neuron index 0~490까지 매우 길다.
- `L1`, `L2`는 각각 491개의 실제 은닉 ReLU로 열 전체를 채운다.
- `L0`은 index 0~255만 실제 입력 노드이고 256~490은 회색 padding이다.
- `L3`은 index 0~15만 실제 출력 노드이고 16~490은 회색 padding이다.
- split 대상이 될 수 있는 은닉 ReLU는 총 982개다.

따라서 ACAS Xu보다 열 수는 적지만 훨씬 세로로 길게 보인다. 입력 열의 아래쪽과 출력 열 대부분이 회색인 것도 `256→491→491→16`이라는 레이어 크기 차이 때문이다.

### 실제 검증 결과

```text
status: UNKNOWN
reason: TIMEOUT
rounds: 0
elapsed: 600.187초
total split events: 0
distinct split neurons: 0
```

ONNX 인코딩과 VNNLIB 파싱, Tseitin CNF 변환은 완료됐다. 프로파일에는 CNF 변환이 약 22.6초 걸렸고 10,660개 clause와 3,554개 atom이 생성된 것으로 기록됐다. 이후 solver가 600초 동안 계산했지만 첫 round를 완료하지 못했다.

ReLU split 이벤트는 solver가 초기 계산을 마치고 위반된 ReLU를 선택한 뒤 활성/비활성 분기로 들어갈 때 기록된다. 이번 실행은 그 지점에 도달하지 못했으므로 982개의 은닉 ReLU가 모두 split count 0으로 남았다.

따라서 안테나 이미지가 옅은 것은 split 없이 문제를 해결해서가 아니다. **split 이전 계산 단계에서 타임아웃되어 관측할 이벤트가 없었기 때문**이다. 이 결과만으로 안테나 모델에 핫스팟이 없다고 결론 내릴 수 없다.

## 5. 두 이미지가 비슷하게 옅지만 의미가 다른 이유

두 이미지의 데이터 행렬은 결과적으로 모두 0으로 채워졌다.

```text
ACAS Xu: split_counts == {}  # 분기 없이 검증 성공
mMIMO  : split_counts == {}  # 분기 전에 타임아웃
```

시각화 함수는 종료 원인을 색에 넣지 않고 오직 `(layer, neuron)별 split 횟수`만 색으로 표현한다. 그래서 서로 다른 solver 결과라도 split count가 모두 0이면 같은 옅은 색으로 보인다.

차이는 이미지가 아니라 JSON과 실행 요약에서 확인해야 한다.

| 확인 항목 | ACAS Xu | mMIMO 안테나 |
|---|---|---|
| 최종 결론 | 안전 속성 검증 완료 | 결론을 내리지 못함 |
| 종료 상태 | `UNSAT` | `TIMEOUT` |
| solver rounds | 2 | 0 |
| split 0회의 해석 | 분기가 필요 없었음 | 분기까지 도달하지 못함 |

## 6. 실제 핫스팟 이미지가 나오면 달라지는 모습

실제 split이 발생하면 JSON 값이 다음 조건을 만족한다.

```text
total_split_events > 0
distinct_neurons > 0
```

그때는:

- ACAS Xu: `L1`~`L6` 중 split된 neuron index 위치가 빨갛게 나타난다.
- mMIMO: `L1`, `L2` 중 split된 neuron index 위치가 빨갛게 나타난다.
- 같은 노드가 반복 선택될수록 더 진한 빨간색이 된다.
- 입력층과 마지막 출력층은 ReLU split 대상이 아니므로 일반적으로 빨갛게 나오지 않는다.

핫스팟은 해당 검증 속성에서 solver가 자주 활성/비활성 분기를 수행한 ReLU를 뜻한다. 가중치 크기, 추론 활성값 또는 모델 중요도를 직접 뜻하지 않는다.

## 7. 최종 해석

### ACAS Xu

이미지는 정상적인 실제 검증 결과다. 구조는 `5→50×6→5`로 올바르게 표시됐고, property 1은 ReLU split 없이 UNSAT으로 증명되어 모든 노드가 옅게 나왔다.

### mMIMO 안테나

이미지 파일은 정상 생성됐지만 검증 결과는 미완료다. 구조는 `256→491→491→16`으로 올바르게 표시됐으나, solver가 첫 round 전에 타임아웃되어 split 핫스팟 데이터는 얻지 못했다.

따라서 현재 두 이미지는 모두 구조를 정확히 보여 주지만, 실제 split 핫스팟을 보여 주는 이미지는 아직 아니다. ACAS Xu는 **split이 필요 없었던 경우**, mMIMO는 **split까지 도달하지 못한 경우**라는 차이를 JSON 결과와 함께 해석해야 한다.
