# Ai_verification

Python으로 구현한 **DPLL(T) + Simplex + Reluplex 기반 ReLU 신경망 검증기**입니다.  
명제 논리식과 선형 부등식, ReLU 제약을 함께 다루며, XOR 신경망 예제를 통해 다음을 실험할 수 있습니다.

- Boolean + Theory 결합 검증
- ReLU 포함 선형 제약 satisfiability 확인
- XOR 네트워크의 분류 동작 검증
- 입력 perturbation에 대한 robustness 검증
- 특정 입력 구간에서 counterexample 탐색 및 샘플링 기반 검산
- 검증 대상식과 Tseitin 변환 후 CNF 식의 시각화

---

## 📌 Overview

이 프로젝트는 다음 흐름으로 동작합니다.

1. 입력 제약식을 `Prop` AST 형태로 표현
2. 이를 **NNF / Tseitin CNF**로 변환
3. `DPLL`로 Boolean abstraction을 탐색
4. 선택된 theory atom(`ineq`, `relu`)을 **Reluplex**에 전달
5. theory SAT/UNSAT 여부에 따라 blocking clause를 추가하며 반복

핵심 구성은 다음과 같습니다.

- **DPLL.py** — 논리식 AST, 파서, NNF 변환, Tseitin CNF 변환, DPLL SAT solver
- **DPLL_T.py** — Boolean solver와 theory solver를 결합하는 DPLL(T) 루프
- **Simplex.py** — 선형 산술 제약을 푸는 simplex tableau solver
- **Reluplex.py** — ReLU 제약이 포함된 실수 제약 처리
- **XOREncoding.py** — XOR 신경망을 논리식(Prop)으로 인코딩
- **Robustness.py** — L∞ 박스 perturbation에 대한 same-class robustness 검증
- **PreciseEncoding.py** — XOR 입력 구간별 counterexample 탐색, 다중 반례 수집, 샘플링 기반 검산
- **LocalRobustnessSweep.py** — epsilon을 점진적으로 증가시키며 robustness 붕괴 지점 추적
- **visualization/** — solver trace, 신경망 layout/heatmap, Prop/CNF 및 sparsity 시각화
- **Tseitin_Transformation.py** — 정수 기반 Tseitin 변환 독립 구현 (교육용)

---

## 📂 Project Structure

```text
.
├── DPLL.py
├── DPLL_T.py
├── Reluplex.py
├── Simplex.py
├── XOREncoding.py
├── Robustness.py
├── PreciseEncoding.py
├── LocalRobustnessSweep.py
├── visualization/
│   ├── SolveTrace.py
│   ├── RealtimeSplitVisualization.py
│   ├── SplitHeatmap.py
│   ├── NetworkLayout.py
│   ├── visualize_prop.py
│   ├── VisualizeSparsity.py
│   ├── tests/
│   └── outputs/
├── Tseitin_Transformation.py
└── README.md
```

---

## 📄 File Descriptions

### 🔹 `DPLL.py`

프로젝트 전체의 **논리식 인프라**를 담당하는 파일입니다.

주요 내용:
- `Prop` AST 정의
  - `TrueProp`, `FalseProp`
  - `VarProp`
  - `InequProp` — `c1*x1 + c2*x2 + ... >= b` 형태의 선형 부등식
  - `ReLUProp` — `y = relu(x)`
  - `AndProp`, `OrProp`, `NotProp`, `ImplProp`
- pretty printer (`show`)
- 식 단순화 (`simplify`)
- implication 제거 (`elim_impl`)
- NNF 변환 (`to_nnf`)
- Tseitin CNF 변환 (`tseitin_cnf`) — `(cnf, atom_map, memo)` 세 값 반환
- 순수 Python DPLL SAT solver (`dpll`)
  - Unit Propagation
  - Pure Literal Elimination
  - Backtracking
- 문자열 입력 파서 (`parse_prop`)
- CNF 절 출력 유틸 (`print_cnf_clauses`)
- 대화형 파이프라인 (`run_pipeline`) — 직접 실행 시 CLI로 동작

즉, 이 파일은 **논리식 표현, CNF 변환, SAT solving의 중심 모듈**입니다.

---

### 🔹 `DPLL_T.py`

**DPLL(T) 메인 루프**를 구현한 파일입니다.

동작 방식:
1. 입력식을 `tseitin_cnf(...)`로 Boolean CNF로 추상화
2. `dpll(...)`로 Boolean model 탐색
3. 참으로 선택된 theory atom 추출
4. 이를 `reluplex(...)`에 전달하여 theory satisfiability 검사
5. theory conflict가 발생하면 blocking clause 추가
6. SAT/UNSAT가 결정될 때까지 반복 (최대 `max_rounds=1000`)

지원 theory atom:
- `InequProp(...)` — 선형 부등식
- `ReLUProp(x, y)` — ReLU 제약

부정 처리:
- `¬ineq(...)` → 계수를 뒤집고 `-b + 1e-6`을 하한으로 설정 (strict inequality 근사)

즉, 이 파일은 **SAT solver와 theory solver를 연결하는 브리지**입니다.

---

### 🔹 `Simplex.py`

선형 제약을 해결하는 **simplex tableau solver** 구현 파일입니다.  
Dutertre & de Moura의 Algorithm 3 스타일을 따릅니다.

자료구조:
- `Row` — 기저변수 행 방정식 (`basic_var = sum(coeffs[xi] * xi)`)
- `Bound` — 변수의 하한/상한
- `SimplexTableau` — rows, bounds, assign 전체 상태

주요 함수:
- `build_tableau(row_defs, bounds)` — Tableau 구성 및 초기화
- `simplex(tableau, ...)` — 메인 simplex 루프
  - 범위 위반 기저변수 탐색
  - Bland's rule 기반 pivot 변수 선택
  - UNSAT 판정 (피벗 가능한 변수 없음)
- `_pivot(tableau, xi, xj)` — 기저/비기저 교환
- `_update_assign(tableau, xj, new_val)` — 할당값 갱신 후 기저변수 재계산

이 파일은 `Reluplex.py`가 내부적으로 사용하는 **선형 제약 해결기**입니다.

---

### 🔹 `Reluplex.py`

ReLU 제약이 포함된 실수 제약을 처리하는 **Reluplex 스타일 theory solver**입니다.

동작 방식:
1. ReLU 출력변수의 하한을 `0`으로 강제 후 simplex 실행
2. ReLU 위반 쌍 `(x, y)` 탐색 (`|y - relu(x)| > tol`)
3. local repair 시도 — `y ← relu(x)` 또는 `x ← y` 방향으로 값 조정 후 simplex 재실행
4. repair가 `branch_tau`번 이상 실패한 변수에 대해 **branching**:
   - `x >= 0` 분기: `y = x` 제약 추가
   - `x <= 0` 분기: `y = 0` 고정
5. 최대 재귀 깊이(`max_recursion=50`) 초과 시 UNKNOWN 처리

핵심 함수:
- `reluplex(row_defs, bounds, relus, ...)` — 메인 진입점
- `_check_relu_violations(assign, relus)` — 위반 탐색
- `_try_repair(tableau, x, y, direction)` — local repair 시도

즉, 선형 부분은 simplex로 풀고, ReLU 위반은 repair/branching으로 다루는 **핵심 theory solver**입니다.

---

### 🔹 `XOREncoding.py`

작은 XOR 신경망을 **검증 가능한 제약식(Prop)** 으로 인코딩하는 파일입니다.

학습된 가중치 (고정):
```
hidden1: w=[2.1247, 2.1267],   b=-2.1259
hidden2: w=[-2.1237, -2.1235], b= 2.1234
output : w=[-3.6788, -3.6766], b= 3.5451
```

유틸 함수:
- `conj(props)` / `disj(props)` — n-ary And/Or 헬퍼
- `ge_lin(terms, b)` — `sum(ci*xi) >= b` 생성
- `eq_lin(terms, b)` — `sum(ci*xi) == b` (양방향 부등식으로 인코딩)
- `FreshGen(prefix)` — 고유 변수명 생성기

XOR network encoding:
- `NN_single(x, gen)` — x 경로 단일 인코딩, `(phi, s, aux)` 반환
- `NN_dual(x, c, gen)` — x 경로 + c(고정 중심) 경로 이중 인코딩, `(phi, s_x, s_c, aux)` 반환

인코딩 특징:
- hidden layer: affine 등식 + ReLUProp
- output layer: logit 등식 (sigmoid 미포함)
- class 판단: `s >= 0 → class 1`, `s < 0 → class 0`

즉, 이 파일은 **신경망을 논리 제약식으로 바꾸는 역할**을 합니다.

---

### 🔹 `Robustness.py`

입력 중심점 주변의 작은 perturbation에 대해, 네트워크가 **같은 클래스를 유지하는지** 확인하는 robustness 검증 예제입니다.

검증 형태:
- 명세: `(precondition ∧ NN) → postcondition`
- 실제 실행: `precondition ∧ NN ∧ ¬postcondition` (반례 탐색)

주요 함수:
- `make_precondition_linf_box(x_vars, c_vals, eps, clamp_01)` — L∞ 박스 precondition 생성
  - 각 변수 `xi`에 대해 `ci - eps ≤ xi ≤ ci + eps`
  - `clamp_01=True`이면 추가로 `0 ≤ xi ≤ 1`
- `make_postcondition_same_class_by_logit(s_x, s_c)` — `(s_x ≥ 0) ↔ (s_c ≥ 0)` 생성
- `build_spec(nn_prop, pre, post)` — 함의식 구성
- `print_cex(center, sat, model)` — 반례 출력 (inputs / logits / hidden 그룹별)

실행 설정 (`__main__`):
- 중심점: `(0,0), (0,1), (1,0), (1,1)` 각각에 대해
- `eps=0.02`, `clamp_01=True`
- `NN_dual`로 x-path와 c-path 동시 인코딩

즉, 이 파일은 **robustness property를 DPLL(T)로 검증하는 예제 스크립트**입니다.

---

### 🔹 `PreciseEncoding.py`

XOR 문제를 **입력 구간별로 직접 검증**하고, 반례를 여러 개 찾으면서 동작을 분석하는 파일입니다.

입력 구간 분류 (`eps=0.1`):
- 구간0 (`zero`): `[0.0, 0.4]`
- 구간1 (`one`):  `[0.6, 1.0]`

주요 함수:
- `zero(i)` / `one(i)` — 입력 클래스 precondition
- `out_zero_logit(s)` / `out_one_logit(s)` — 출력 logit 클래스 판단
- `xor_nn_sx(x1, x2)` — 실제 신경망 forward 계산 (검산용)
- `check_xor_on_region(r1, r2, expected_out, n_samples)` — 랜덤 샘플링 검산

동작 (`__main__`):
- 4개 case `(00, 01, 10, 11)`에 대해 `NN_single`로 인코딩
- 각 case마다 `pre ∧ NN ∧ ¬post` 형태로 최대 5개 반례 탐색
- 이전 반례 주변(`delta=0.1`)을 OR 기반 blocking constraint로 제외
- UNSAT 시 10,000개 샘플링으로 경험적 검산

즉, 이 파일은 **XOR 네트워크의 정밀 반례 탐색 및 검산 스크립트**입니다.

---

### 🔹 `LocalRobustnessSweep.py`

XOR 신경망에 대해 **epsilon 값을 점진적으로 증가시키면서 local robustness가 언제 깨지는지** 추적하는 스윕 스크립트입니다.

동작 (`run_milestone_sweep`):
- `eps_values = np.arange(0.0, 0.505, 0.005)` — 0~0.5 구간을 0.005 간격으로 탐색
- XOR 4개 중심 케이스에 대해 `NN_single` + `make_precondition_linf_box` 조합으로 검증
- 각 epsilon에서 SAT(반례 존재)한 케이스 수를 집계
- SAT count가 증가하는 지점을 **milestone**으로 기록
- milestone마다 발견된 반례 입력 저장

결과 출력 (`__main__`):
- epsilon sweep 결과를 step plot으로 시각화 (matplotlib)
- milestone별 반례를 `xor_nn_sx`로 다시 계산하여 자동 검산
- 경계점 수치 오차 의심 시 입력을 `0.001` 밀어내어 오답 여부 재확인

즉, 이 파일은 **local robustness의 붕괴 시점을 스윕 기반으로 분석하고, 반례를 검산하는 실험 스크립트**입니다.

---

### 🔹 `visualization/visualize_prop.py`

검증 대상 논리식과 Tseitin 변환 후 CNF를 **Graphviz 그래프 형태로 시각화**하는 유틸리티입니다.

주요 함수:
- `prop_to_dot(prop, name)` — Prop 트리를 DOT 형식으로 변환 (노드 종류별 색상 구분)
- `cnf_to_dot(cnf, atom_map, memo, name)` — CNF 트리 + atom_map 범례를 나란히 배치
- `save_dot(dot_src, filepath)` — `visualization/outputs/precise_prop/` 하위에 `.dot` 파일 저장
- `render_dot(filepath, fmt)` — Graphviz `dot` 명령으로 PNG 렌더링
- `dump_search_phi_visualization(phi, case_name, attempt_no, ...)` — 반례 탐색 중인 phi를 tree/CNF 두 형태로 저장
- `visualize_precise_case(case_name, r1, r2, ...)` — PreciseEncoding 단일 케이스 시각화

직접 실행 시 (`__main__`):
- PreciseEncoding의 4개 case(`00, 01, 10, 11`)에 대한 neg_spec 트리/CNF 저장

노드 색상:
- AND/OR/NOT/IMPL: 연한 파랑 (ellipse)
- InequProp: 연한 초록 (box)
- ReLUProp: 연한 주황 (box)
- VarProp: 연한 노랑 (box)
- True/False: 회색 (diamond)

---

### 🔹 `Tseitin_Transformation.py`

**정수 기반 DIMACS 형식의 Tseitin 변환 독립 구현** 파일입니다.  
`DPLL.py`의 문자열 기반 구현과 별개로, 교육 목적으로 작성된 standalone 모듈입니다.

`DPLL.py`와의 차이점:
- 리터럴을 문자열이 아닌 **정수**로 표현 (`1, -2, 3, ...` DIMACS 형식)
- `IffProp` (쌍조건, `↔`) 지원
- `LinearInequProp` (다중 변수 선형 부등식 별도 타입) 지원
- 결과 출력: `print_tseitin_result(cnf, node_to_var)` — DIMACS 절, 부분식 매핑 출력

주요 함수:
- `to_nnf(prop)` — ImplProp/IffProp 제거 후 NNF 변환
- `tseitin_to_cnf(root)` — `(cnf, node_to_var)` 반환
- `print_tseitin_result(...)` — CNF, 변수 매핑, raw DIMACS 절 출력

---

## 🚀 How to Run

## 1. Requirements

기본적으로 Python 3 환경에서 실행됩니다.

사용 라이브러리:
- `numpy`
- `matplotlib`
- `tqdm`

설치 예시:

```bash
pip install numpy matplotlib tqdm
```

---

## 2. Run Each Module

### `DPLL.py`

문자열로 작성한 논리식을 파싱하고, NNF / CNF / DPLL 결과를 확인할 수 있습니다.

```bash
python DPLL.py
```

지원 문법:
- `and`, `or`, `not`, `~`, `->`
- 괄호 `( ... )`
- `true`, `false`
- `ineq(c1,x1,c2,x2,...,b)` : `c1*x1 + c2*x2 + ... >= b`
- `relu(x,y)` : `y = relu(x)`

입력 예시:
```text
(p and q) or not r
not (p -> q)
ineq(1,x,0) or p
ineq(1,x,1,y,2,z,-5)
ineq(1,x,0) or relu(x,y)
```

---

### `DPLL_T.py`

DPLL(T) 데모를 실행합니다.

```bash
python DPLL_T.py
```

포함된 예시:
- `relu(x,y)`와 선형 제약이 함께 있는 SAT 예제
- Boolean level에선 SAT지만 theory level에선 UNSAT인 예제
- `not ineq(...)` 처리 예제

직접 호출 예시:

```bash
python -c "from DPLL import parse_prop; from DPLL_T import dpll_t; p = parse_prop('ineq(1,x,1,y,5) and relu(x,y)'); print(dpll_t(p))"
```

```bash
python -c "from DPLL import parse_prop; from DPLL_T import dpll_t; p = parse_prop('ineq(1,x,0) and relu(x,y) and ineq(-1,y,1e-6)'); print(dpll_t(p))"
```

---

### `Simplex.py`

simplex solver에 대한 기본 테스트를 실행합니다.

```bash
python Simplex.py
```

포함된 예시:
- SAT 선형 제약 (s1=x+y≥0, s2=-2x+y≥2, s3=-10x+y≥-5)
- UNSAT 선형 제약 (x≥5 AND x≤3)
- 다변수 연립 제약

---

### `Reluplex.py`

Reluplex solver 자체 테스트를 실행합니다.

```bash
python Reluplex.py
```

포함된 예시:
- `x + y >= 5`, `y = relu(x)` (SAT)
- `x >= 0`, `y = relu(x)`, `y < 0` (UNSAT)
- `x + y <= 2`, `y = relu(x)` (SAT)

---

### `XOREncoding.py`

XOR 신경망 인코딩 결과를 확인합니다.

```bash
python XOREncoding.py
```

출력 내용:
- 4개 중심점 각각에 대해 `NN_dual(...)` 생성 결과
- `NN_single(...)` 생성 결과 및 logit 변수명

---

### `Robustness.py`

XOR 중심점 주변의 입력 perturbation에 대해 **same-class robustness**를 검사합니다.

```bash
python Robustness.py
```

동작:
- 중심점 `c ∈ {(0,0), (0,1), (1,0), (1,1)}`
- `eps=0.02`, `clamp_01=True`로 L∞ 박스 생성
- `NN_dual`로 x-path + c-path 동시 인코딩
- `(s_x >= 0) ↔ (s_c >= 0)` 위반하는 반례 탐색
- 반례가 있으면 counterexample 출력, 없으면 robust 판정

---

### `PreciseEncoding.py`

입력 region별 XOR 반례 탐색 및 검산을 수행합니다.

```bash
python PreciseEncoding.py
```

동작:
- `eps=0.1`, 구간: `[0.0, 0.4]`, `[0.6, 1.0]`
- 4개 case마다 `NN_single`로 인코딩 후 `pre ∧ NN ∧ ¬post` 탐색
- 최대 5개의 반례를 순차 수집 (이전 반례 주변 `delta=0.1` 제외)
- UNSAT 시 10,000개 랜덤 샘플링으로 검산

---

### `LocalRobustnessSweep.py`

epsilon sweep을 수행하여 **XOR 신경망의 local robustness가 어느 지점에서 깨지는지** 분석합니다.

```bash
python LocalRobustnessSweep.py
```

동작:
- `eps`: 0.0 ~ 0.5, 0.005 간격으로 증가
- 각 epsilon에서 4개 케이스 순차 검증
- SAT count 증가 지점을 milestone으로 저장
- 그래프(step plot)로 robustness 붕괴 패턴 시각화
- milestone별 반례를 실제 신경망 함수로 자동 검산

---

### `visualization/visualize_prop.py`

검증 대상식과 Tseitin 변환 후 CNF 식을 Graphviz DOT/PNG로 시각화합니다.

```bash
python -m visualization.visualize_prop
```

실행 결과 (`visualization/outputs/precise_prop/` 디렉터리에 저장):
- `case00/precise_search_case00_*_tree.dot/.png`
- `case00/precise_search_case00_*_cnf.dot/.png`
- (case01, case10, case11도 동일)

---

### `Tseitin_Transformation.py`

정수 기반 Tseitin 변환 결과를 확인합니다.

```bash
python Tseitin_Transformation.py
```

출력 내용:
- `¬(p ∧ q)` 예제의 NNF 변환
- 정수 리터럴 기반 CNF 절 목록
- 부분식 → SAT 변수 번호 매핑

---

## 🧠 Logic / Encoding Notes

### `InequProp`

프로젝트에서 선형 부등식은 다음 형태로 표현됩니다.

```text
c1*x1 + c2*x2 + ... >= b
```

예:
```text
ineq(1,x,1,y,2,z,-5)
```

의미:
```text
x + y + 2z >= -5
```

---

### strict inequality

현재 theory level에서는 기본적으로 `>=`만 직접 지원합니다.  
따라서 strict inequality는 보통 작은 epsilon으로 근사합니다.

예:
- `x < 0` → `-x >= 1e-6`
- `x <= 0` → `-x >= 0`

---

### ReLU atom

```text
relu(x,y)
```

는 다음 의미를 갖습니다.

```text
y = relu(x) = max(0, x)
```

---

### output class by logit

XOR 예제에서는 sigmoid를 직접 인코딩하지 않고, 출력 logit `s`의 부호를 사용합니다.

- `s >= 0` → class 1
- `s < 0`  → class 0

이는 sigmoid의 단조성 때문에 가능합니다.

---

## 🧪 Example Verification Tasks

이 저장소로 실험할 수 있는 대표적인 검증 작업:

1. **Boolean + theory satisfiability**
   - `ineq(...) and relu(...)`

2. **Theory conflict detection**
   - Boolean level SAT, theory level UNSAT

3. **XOR network correctness**
   - 입력 case별 expected output 검증

4. **Robustness around centers**
   - `(0,0), (0,1), (1,0), (1,1)` 주변 perturbation

5. **Counterexample enumeration**
   - 한 개가 아니라 여러 반례를 순차적으로 탐색

6. **Formula / CNF visualization**
   - 검증 대상식과 Tseitin 변환 결과를 그래프로 비교

---

## ⚠️ Notes / Limitations

- 이 구현은 **교육/실험 목적의 Python 구현**입니다.
- 최적화된 SAT/SMT solver와 비교하면 성능은 제한적일 수 있습니다.
- theory atom의 negation은 `DPLL_T.py`에서 `1e-6` epsilon 기반으로 근사 처리됩니다.
- strict inequality는 직접 지원하지 않으므로 작은 epsilon 기반 근사가 필요합니다.
- class 경계값(`s = 0` 근방)에서 미세한 음수 logit이 양수로 처리될 수 있어 위양성 반례가 발생할 수 있습니다.
- counterexample을 여러 개 찾기 위해 OR 기반 blocking constraint를 누적하면 탐색 공간이 조각나 속도가 크게 느려질 수 있습니다.
- `visualization/visualize_prop.py`의 PNG 렌더링은 Graphviz `dot` 명령이 설치되어 있어야 합니다. 미설치 시 `.dot` 파일만 생성됩니다.

---

## 🔍 Recommended Entry Points

처음 볼 때는 아래 순서로 실행해보는 것을 추천합니다.

1. `python DPLL.py` → 논리식 파싱 / CNF 변환 / DPLL 확인
2. `python Tseitin_Transformation.py` → 정수 기반 Tseitin 변환 구조 확인
3. `python DPLL_T.py` → Boolean + Theory 결합 흐름 확인
4. `python Simplex.py` → 선형 solver 동작 확인
5. `python Reluplex.py` → ReLU 처리 방식 확인
6. `python XOREncoding.py` → XOR 네트워크 인코딩 구조 확인
7. `python Robustness.py` → robustness 예제 실행
8. `python PreciseEncoding.py` → 입력 region별 정밀 반례 탐색 및 검산
9. `python LocalRobustnessSweep.py` → epsilon sweep 기반 robustness 붕괴 지점 분석
10. `python -m visualization.visualize_prop` → 검증 대상식과 CNF 구조 시각화

---

## 시각화의 Gemini 로그 분석

기본 사용 방식은 기존 검증 명령으로 시각화를 열고, 화면에서 선택한 항목을 분석하는 것입니다. API 키는 프로젝트 루트의 `.env`에 적습니다.

```dotenv
GEMINI_API_KEY=
```

등호 뒤에 키를 넣고 저장한 다음 **기존 검증 명령어를 그대로** 실행합니다. `.env`의 키가 우선하며, 비어 있으면 `GEMINI_API_KEY` 환경 변수를 사용합니다. 키를 바꾼 뒤에는 검증기를 다시 실행합니다. `--visualization-mode realtime` 또는 `both`이면 로컬 분석 서버를 자동 준비해 시각화 화면을 엽니다. 자동 연결 모델은 `gemini-3.5-flash-lite`이며, 검증과 별도 프로세스에서 동작합니다.

키가 없거나 형식이 잘못되었거나 서버 준비에 실패해도 검증은 중단하지 않고 기존 파일 시각화를 엽니다. 사용자 지정 대시보드를 실행 폴더 밖에 저장하는 경우에도 파일 시각화를 유지합니다. 시각화 미사용·feedback 전용·dry run에서는 분석 서버를 자동 실행하지 않습니다.

개별·통합 분석의 **모델** 드롭다운에는 API 키로 조회한 Gemini 3.5 이상 텍스트 모델이 표시됩니다. 모델별 **추론 강도**를 다음 분석 요청에 적용할 수 있습니다.

**새 검증 실행**에서 벤치마크, ONNX, VNNLIB, 엔진 및 제한 시간을 고를 수 있습니다. 로컬 파일은 `Onnx`, `Automation/SmokeTest`, `Vnnlib`, `Properties`에서, 벤치마크는 인접 `vnncomp2025_benchmarks/benchmarks`에서 읽습니다. 압축 ONNX는 임시 폴더에서 풉니다.

여러 조합을 실행하려면 각각 **대기열에 추가**한 뒤 **연속 검증 시작**을 누릅니다. 최대 50건을 등록한 순서대로 실행하며, 각 대시보드는 새 탭에서 열립니다. 모두 끝나면 최초 실행 화면에서 전송에 동의하고 **전체 실행 분석**을 눌러 판정, DPLL, Simplex 및 ReLUplex 지표를 비교할 수 있습니다. 결과 목록과 분석 답변은 `Automation/Results/verification_batches/<id>/batch.json`에 저장됩니다.

처음부터 선택 화면을 열 때는 프로젝트 폴더에서 아래 명령을 실행합니다. 신경망 검증은 Gemini API 키 없이도 시작할 수 있습니다.

```powershell
python -m Automation.VerificationLauncher
```

저장된 과거 실행을 별도로 열거나 모델·호출 상한을 바꾸고 싶은 경우에만 다음 명령을 사용합니다.

```powershell
python -m Automation.AnalyzeLogs --serve --run "시각화 실행 폴더 경로" --allow-upload --model gemini-3.5-flash-lite --max-calls 8 --open
```

- 새로 열린 대시보드의 **항목별 LLM 분석**은 기본 **전체**입니다. 함께 열린 패널만 분석 항목 목록에 표시됩니다. DPLL, Simplex, ReLU 분기 이력, **ReLU 탐색 트리**, **Mechanisms**로 범위를 좁힐 수 있습니다. Mechanisms를 선택하면 바로 옆 기법 탭에서 전체·경계 유도·충돌 분석/백점프·부동소수점·과소근사를 선택합니다. 각 화면 제목 옆의 **LLM 분석** 버튼으로 해당 항목을 바로 선택할 수도 있습니다.
- 그래프의 호출·라운드·뉴런, 탐색 트리·타임라인 행, Mechanisms 그림·발생 기록을 클릭하면 선택 항목이 네모 목록에 표시됩니다. 질문란은 직접 작성한 질문만 유지하며, 선택 정보는 분석 시 별도로 전달됩니다. 질문을 비워도 선택 항목은 반영됩니다. 클릭만으로 API를 호출하지 않으며, 동의 후 **현재까지 분석**을 눌러 전송합니다. PNG 그래프는 새 실행에서 생성한 `.selection.js` 좌표 파일이 필요합니다.
- 기본은 **중복 선택**입니다. 선택한 항목들의 로그 범위만 합쳐 함께 분석합니다. 각 항목의 `×` 또는 **선택 비우기**로 해제할 수 있고, 옵션을 끄면 마지막 선택만 남고 이후 단일 선택합니다. 동일 정보는 중복 추가하지 않습니다. 한 번에 최대 20개이며, 선택 정보의 크기 제한과 질문의 2,000자 제한은 별개입니다.
- Round를 추가로 선택할 수 있습니다. Simplex는 call ID, ReLU 분기 이력·탐색 트리·경계 유도는 Layer·Neuron으로 범위를 좁힐 수 있습니다.
- 로그 전송 동의는 처음에는 미체크이며, 한 번 체크하면 같은 대시보드 탭에서는 항목 변경·그래프 클릭·새로고침 후에도 유지됩니다. 직접 체크를 해제하면 해제 상태도 유지됩니다. **현재까지 분석**을 누를 때만 로그를 전송합니다. 모델 목록 조회에는 로그를 포함하지 않습니다.
- 버튼 아래에 이번 분석의 API 사용량·남은 요청 수를 표시합니다. 재시도도 포함되며 새 분석마다 초기화됩니다. 이는 프로젝트의 실제 무료 할당량 잔여치가 아닙니다. Google 잔여 한도는 추정하지 않고 **AI Studio 한도** 링크에서 별도로 확인하도록 구분합니다.
- 버튼을 누르면 그때까지 파일에 기록된 완전한 JSONL 행을 고정합니다. 분석 중에도 검증과 그래프는 계속 진행되지만, LLM의 모든 조회는 고정된 기록 안에서만 이루어집니다. 아직 버퍼에 있거나 쓰는 중인 행은 다음 분석에 포함됩니다.
- 분석 기준 시각·Round·call과 이후 추가된 기록 수(전체 로그 기준)를 표시합니다. 목록·기록 상태는 3초마다 로컬에서 갱신하며 API를 호출하지 않습니다. 새 기록은 자동으로 재분석하지 않습니다.
- 로그 인덱스는 추가된 부분만 읽습니다. 전송량과 별개로 `--max-log-mb`(기본 256 MiB)는 수집 범위를 제한하며, 상한·불완전한 행·손상된 행은 범위 제한으로 표시합니다. 대규모 로그의 초기 수집·로컬 스냅샷 복사에는 시간이 걸릴 수 있습니다.
- 진행 중인 Simplex 호출은 최종 시간/판정과 구분합니다. 시작 이벤트가 생략된 compact 로그에서는 진행 중 호출을 복원할 수 없습니다. 프로세스 생존 여부가 아니라 마지막으로 기록된 실행 상태를 보여줍니다.
- 선택 범위를 서버에서 실제로 필터링합니다. 배치 경계 로그도 선택한 뉴런 기록만 남기며, LLM 도구로 범위를 확장할 수 없습니다. 실행 설정·모델 정보는 배경 문맥으로 전달됩니다.
- Simplex 원인 분석에는 같은 스냅샷의 다른 완료 호출에서 계산한 비교 집계도 제공합니다. 피벗 총량과 호출 전체 시간/피벗 수를 나눠 비교하며, 다른 호출의 원시 로그는 전달하지 않습니다. 비교는 최초 최대 20,000개 완료 호출로 제한하고, 선택한 호출은 별도로 조회할 수 있습니다. 명시적으로 기록된 호출 순번만 사용하고 ID 숫자로 순번을 추정하지 않습니다.
- 관련 이벤트가 0건이어도 분석 버튼으로 LLM을 호출할 수 있습니다. 저장된 설정과 현재 구현의 참고 설명으로 원리·미적용의 의미를 설명하되, 미기록을 미적용으로 단정하거나 과거 실행이 현재 코드와 같다고 가정하지 않습니다. 이 경우도 API 사용량에 포함됩니다.
- 탐색 트리 분석에는 분기 결과·깊이·종료 기록 없는 가지·STABLE 확정 이력·소요시간이 긴 완료 가지를 제공합니다. 가지 시간은 하위 탐색을 포함하므로 합산하지 않습니다. 부모나 원인 연결이 없는 경우 임의로 추정하지 않습니다.
- 결과는 현재 선택에 맞는 것만 표시합니다. 다른 항목으로 바꾸면 이전 항목의 분석을 즉시 지웁니다. 같은 범위의 마지막 결과를 저장·복원합니다. 원시 조회 근거는 화면에 펼쳐 놓지 않고 저장된 `analysis.json`에 보존합니다.
- 분석 문장·근거는 그래프의 항목명과 R번호·Simplex 호출 ID·Layer/Neuron 위치로 설명합니다. 기존 저장 분석의 로그 키·내부 뉴런 이름·출처 표시도 화면에서는 읽기 쉬운 표기로 바꿉니다. 수치와 원본 로그/분석 파일은 변경하지 않습니다.
- API 키는 서버 프로세스에만 있습니다. 서버는 `127.0.0.1`에만 바인딩되며 임의 포트/세션 URL을 사용합니다. 표시된 URL은 공유하지 마세요. `--port`로 포트를 지정할 수 있습니다.
- 원본 HTML·로그는 수정하지 않습니다. 과거 대시보드에도 서버 응답에서 패널을 추가합니다. 새로 생성되는 HTML에는 안내 패널이 포함되지만, 파일을 직접 열어서는 분석 요청을 보낼 수 없습니다.
- 동시에 한 분석만 실행합니다. 자동 서버는 검증 종료 뒤에도 화면이 열려 있으면 유지됩니다. **화면 연결 종료**로 직접 닫거나, 모든 탭을 닫으면 마지막 요청으로부터 30분 뒤 자동 종료됩니다. 수동 서버는 `Ctrl+C`로도 종료할 수 있습니다. 진행 중 분석은 저장/시간 제한 후 종료됩니다. 일반 CLI 분석(`--serve` 없음)은 완료된 `result.json`이 필요합니다.
- 새 검증은 시작 시 `run_info.json`에 실행 ID·모델·설정을 기록하고, 종료 후 같은 ID의 `result.json`을 저장합니다. 실행 중 연결할 폴더를 고정하려면 검증 명령에 `--visualization-output-dir "visualization/outputs/my_live_run"`을 지정하세요. 한 폴더를 여러 검증이 동시에 사용하면 안 됩니다.

### 저장 로그 분석·연결 점검용 CLI

아래는 자동 연결에 필요한 절차가 아니라, 저장된 실행 비교나 오류 점검에 사용하는 보조 기능입니다. 기본 모델은 자동 연결과 동일한 `gemini-3.5-flash-lite`입니다.

검증이 끝난 실행 폴더의 `result.json`, `solver_progress.jsonl`,
`relu_events_branches.jsonl`, `relu_events.jsonl`을 읽습니다. 솔버 실행이나 원본
로그 수정은 하지 않습니다. Python 표준 라이브러리만 사용합니다.

외부 전송 없이 로컬 요약을 먼저 확인할 수 있습니다.

```powershell
python -m Automation.AnalyzeLogs --run "시각화 실행 폴더 경로" --prepare-only
```

Gemini 분석은 API 키와 명시적 전송 동의가 필요합니다. 키를 소스나 명령행 인자로
저장하지 마세요. 무료 API의 입력·응답은 Google 제품 개선에 사용될 수 있으므로
공개 벤치마크 등 전송 가능한 데이터만 선택하세요. 경로·일부 키 마스킹은 기밀정보
탐지나 완전한 익명화를 보장하지 않습니다.

```powershell
python -m Automation.AnalyzeLogs --run "시각화 실행 폴더 경로" --allow-upload --max-calls 8
```

- 계정에서 사용 가능한 다른 모델은 `--model`로 지정합니다.
- `--compare "다른 실행 폴더"`: 두 실행의 설정과 패턴을 비교합니다.
- `--question "조사할 질문"`: 분석 주제를 지정합니다.
- `--max-calls`: 최종 답변을 포함한 API 요청 상한(기본 8, 최대 30).
- `--check-api`: 로그를 읽거나 전송하지 않고 연결·키와 사용 가능한 모델 목록만 확인합니다. `--run` 없이 실행 가능합니다.
- `--check-generation`: 로그·도구 없이 `OK`라는 짧은 답변 생성만 시험합니다. 모델 목록 조회와 달리 생성 할당량을 사용합니다. `--max-calls 1`로 1회만 시험할 수 있습니다.
- `--thinking-level low|medium|high|default`: 기본 `low`로 추론 대기·토큰 사용을 줄입니다. `default`는 모델 기본값을 그대로 사용합니다. Gemini 2.5에는 대응하는 토큰 예산을 적용합니다.
- `--request-timeout`: 요청별 응답 대기 시간(기본 90초, 최대 180초).
- `--retries 0`: 기본으로 제공하는 일시적 서버 오류 재시도를 끕니다.
- `--max-log-mb`: 실행당 로그 색인 상한(기본 256 MiB). 잘림·누락·파싱 실패는 요약에 표시됩니다.
- 각 도구 응답은 최대 16,000바이트, API 요청은 최대 256 KiB입니다. 토큰 한도와는 다릅니다.
- LLM은 요약, 순위, 이벤트 필터 조회만 실행할 수 있습니다. 임의 파일 접근·SQL·명령 실행은 제공하지 않습니다.
- `analysis.txt`와 근거 조회 이력이 포함된 `analysis.json`을 `Automation/Results/llm_analysis`의 새 폴더에 저장합니다.
- HTTP 500/502/503/504만 동일 모델로 최대 1회 재시도합니다. 재시도도 `--max-calls` 상한에 포함됩니다. 사용량 초과(429)·인증 오류·네트워크/시간 초과는 재시도하지 않습니다.
- 오류가 나면 키를 가린 Google 상세 메시지를 콘솔과 결과에 표시하며, API 오류 전 조회 근거를 보존합니다. 다른 모델/유료 서비스로 자동 전환하지 않습니다.
- 이 프로그램은 무료 한도를 확인하거나 과금을 막지 않습니다. 결제 연결 여부와 모델 요금은 Google 프로젝트에서 관리해야 합니다.
- 분석은 관찰·가설이며 검증 증명이 아닙니다. 과거 로그의 미기록 설정·샘플링·수정 인과는 확정할 수 없습니다.

API 형식은 [Google generateContent 문서](https://ai.google.dev/api/generate-content),
사용 한도는 [AI Studio](https://aistudio.google.com/rate-limit), 데이터 이용은
[무료 서비스 약관](https://aistudio.google.com/docs/terms#data-use-unpaid)을 참고하세요.

---

## 외부 LP 엔진 시각화

공통 계측은 항목별 `pivot_count_available`·`iteration_count_available`·`cycle_count_available`과 `metrics_source`를 기록합니다. Python Simplex oracle은 기존 콜백을 공유하며 GLPK는 실제 반복 카운터의 호출 전후 차이를 기록합니다. 항목별 플래그가 없는 과거 기록에서만 `work_counts_available`을 따릅니다. 미수집 값은 0회가 아니라 null/그래프 공백으로 구분합니다. 피벗이 없고 반복 수만 있으면 해당 패널은 Iterations로 표시하며 다른 횟수를 피벗이라고 부르지 않습니다.

GLPK와 Z3의 우하단 `Bound violations at LP call entry`는 이전 LP 해 또는 코어 입력 할당을 현재 경계와 비교한 `call_boundary` 관측을 표시합니다. Python Simplex의 `Repeated bound violations`는 내부 반복 검사 샘플을 표시합니다. 호출 진입 시점의 위반 기록이 없으면 자료가 기록되지 않았다고 표시합니다. 새 엔진은 호출별 `simplex_metrics` 딕셔너리로 동일 형식에 참여할 수 있고, 모르는 항목은 생략합니다. 기존 저장 로그에 없는 통계는 소급 생성하지 않습니다.

Z3는 해당 산술 백엔드가 `arith-pivots`를 보고할 때 누적 카운터의 호출 전후 차이를 읽습니다. 키가 제공되지 않거나 카운터가 초기화되면 미수집으로 남기며, SMT의 다른 통계를 피벗으로 대체하지 않습니다. 통계 출처는 [Z3 원본 산술 통계 수집 코드](https://github.com/Z3Prover/z3/blob/master/src/smt/theory_lra.cpp)를 참고하세요.


## 📚 Summary

이 프로젝트는 다음을 직접 구현하고 연결합니다.

- propositional logic AST (문자열 기반, 정수 기반 각각)
- NNF / Tseitin CNF transformation
- DPLL SAT solver
- DPLL(T) loop
- simplex-based linear arithmetic solving (Dutertre & de Moura 스타일)
- Reluplex-style ReLU solving (local repair + branching)
- XOR neural network verification examples (single/dual encoding)
- robustness and counterexample search workflows
- epsilon sweep 기반 local robustness milestone 분석
- formula tree / Tseitin CNF visualization workflows

작고 명시적인 Python 구현으로,  
**ReLU 신경망 검증의 핵심 아이디어를 단계별로 실험하고 이해하기 좋은 구조**를 목표로 합니다.
