# Simplex numerical failure와 변수 이름


## 그래프의 `h`, `z`, `ineq_slack`

Simplex 그래프의 `Repeated bound violations`에는 신경망 인코딩 과정에서 생성된
변수 이름이 표시될 수 있다.

| 이름 | 의미 |
|---|---|
| `z...` | ReLU 적용 전 값(pre-activation). `z = W × 이전 층 출력 + b` |
| `h...` | ReLU 적용 후 값. 은닉층에서 `h = ReLU(z) = max(0, z)` |
| `ineq_slack_...` | 부등식을 Simplex tableau에 넣기 위한 보조변수 |

부등식 `Σ(cᵢxᵢ) >= b`는 다음처럼 변환된다.

ineq_slack_k = Σ(cᵢxᵢ)
ineq_slack_k >= b

따라서 `ineq_slack`의 반복 위반은 별도의 신경망 뉴런이 문제라는 뜻이 아니라,
연결된 입력·출력 조건이나 theory 부등식이 현재 할당에서 반복해서 위반됐다는
뜻이다.

### `z`에서 bound 위반이 발생한 경우

`z`는 가중치와 bias로 계산된 ReLU 이전 값이다. Reluplex가 ReLU의 활성 상태를
나눌 때 다음 bound가 생긴다.

활성(active) 분기:   z >= 0
비활성(inactive) 분기: z <= 0

따라서 `z` 위반은 현재 tableau 할당의 `z`가 선택한 ReLU 분기와 반대 부호라는
뜻이다. 예를 들어 active 분기를 탐색 중인데 affine 식으로 계산된 `z`가 음수이면
`z`의 lower bound 위반으로 기록된다. 이는 그 뉴런이 최종적으로 잘못됐다는
판정이 아니라, 현재 할당을 피벗으로 조정해야 한다는 뜻이다.

### `h`에서 bound 위반이 발생한 경우

`h`는 ReLU 출력이므로 항상 `h >= 0`이어야 한다. inactive 분기에서는 ReLU
출력이 정확히 0이어야 하므로 bound가 더 강해진다.

공통:               h >= 0
비활성(z <= 0) 분기: h = 0
활성(z >= 0) 분기:   h = z

- `h < 0`이면 ReLU 출력의 공통 lower bound 위반이다.
- inactive 분기에서 `h != 0`이면 해당 분기의 고정 bound 위반이다.
- active 분기의 `h = z` 관계는 별도 equality row로 연결되므로, 그 관계를 
  만족시키는 과정에서 `h`, `z` 또는 equality 보조변수의 조정이 반복될 수 있다.

그래프의 `h`/`z` 위반 횟수는 **중간 Simplex 할당을 고친 횟수**에 가깝다.
그 자체로 모델의 오류나 numerical failure를 뜻하지 않는다. 피벗 후 모든 bound와
ReLU 관계가 만족되면 정상 SAT 후보가 될 수 있고, 해결할 피벗이 없으면 UNSAT
후보가 된다. 이때 tableau 일관성 검사까지 실패해야 numerical failure가 된다.
