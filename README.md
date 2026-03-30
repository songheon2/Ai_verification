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

- **DPLL.py**  
  논리식 AST, 파서, NNF 변환, Tseitin CNF 변환, DPLL SAT solver

- **DPLL_T.py**  
  Boolean solver와 theory solver를 결합하는 DPLL(T) 루프

- **Simplex.py**  
  선형 산술 제약을 푸는 simplex tableau solver

- **Reluplex.py**  
  ReLU 제약이 포함된 실수 제약 처리

- **XOREncoding.py**  
  XOR 신경망을 논리식(Prop)으로 인코딩

- **Robustness.py**  
  입력 중심점 주변의 robustness 검증 예제

- **PreciseEncoding.py**  
  XOR 입력 구간별 counterexample 탐색, 다중 반례 수집, 샘플링 기반 검산

- **visualize_prop.py**  
  검증 대상 Prop 식과 Tseitin CNF 결과를 Graphviz DOT/PNG로 시각화


---

## ✨ Features

- `ineq(c1,x1,c2,x2,...,b)` 형태의 선형 부등식 지원
- `relu(x,y)` 형태의 ReLU 제약 지원
- `and`, `or`, `not`, `->`를 포함한 논리식 파싱 지원
- NNF 변환 및 Tseitin 기반 CNF 변환
- 순수 Python DPLL SAT solver
- DPLL(T) 방식의 Boolean + Theory 결합
- Simplex 기반 선형 제약 해 탐색
- Reluplex 스타일의 repair + branching
- XOR 네트워크 single / dual encoding
- robustness 및 counterexample 기반 검증 예제 제공
- Prop 트리와 Tseitin CNF 구조의 Graphviz 시각화 지원


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
├── visualize_prop.py
├── neg_spec_tree.dot
├── neg_spec_tree.png
├── neg_spec_cnf.dot
├── neg_spec_cnf.png
└── README.md

```

---

## 📄 File Descriptions

### 🔹 `DPLL.py`

프로젝트 전체의 **논리식 인프라**를 담당하는 파일입니다.

주요 내용:
- `Prop` AST 정의
  - `VarProp`
  - `InequProp`
  - `ReLUProp`
  - `AndProp`, `OrProp`, `NotProp`, `ImplProp`
- pretty printer (`show`)
- 식 단순화 (`simplify`)
- implication 제거 (`elim_impl`)
- NNF 변환 (`to_nnf`)
- Tseitin CNF 변환 (`tseitin_cnf`)
- 순수 Python DPLL SAT solver
  - Unit Propagation
  - Pure Literal Elimination
  - Backtracking
- 문자열 입력 파서 (`parse_prop`)
- CNF 절 출력 유틸

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
6. SAT/UNSAT가 결정될 때까지 반복

지원 theory atom:
- `ineq(...)`
- `relu(x,y)`

즉, 이 파일은 **SAT solver와 theory solver를 연결하는 브리지**입니다.

---

### 🔹 `Simplex.py`

선형 제약을 해결하는 **simplex tableau solver** 구현 파일입니다.

주요 내용:
- 자료구조
  - `Row`
  - `Bound`
  - `SimplexTableau`
- tableau 생성 (`build_tableau`)
- pivot 연산 (`_pivot`)
- 기저변수 값 재계산 (`_compute_basic`)
- 변수 할당 업데이트 (`_update_assign`)
- simplex 메인 알고리즘 (`simplex`)
- 디버그용 tableau 출력

이 파일은 `Reluplex.py`가 내부적으로 사용하는 **선형 제약 해결기**입니다.

---

### 🔹 `Reluplex.py`

ReLU 제약이 포함된 실수 제약을 처리하는 **Reluplex 스타일 theory solver**입니다.

주요 내용:
- `relu(v)` 정의
- 현재 할당에서 ReLU 위반 검사
- local repair 시도
- 반복 repair
- 필요 시 `x >= 0` / `x <= 0` branching
- 내부적으로 `Simplex.build_tableau(...)`, `simplex(...)` 사용

즉, 선형 부분은 simplex로 풀고, ReLU 위반은 repair/branching으로 다루는 **핵심 theory solver**입니다.

---

### 🔹 `XOREncoding.py`

작은 XOR 신경망을 **검증 가능한 제약식(Prop)** 으로 인코딩하는 파일입니다.

주요 내용:
- conjunction / disjunction helper
  - `conj`
  - `disj`
- 선형 등식/부등식 생성 helper
  - `ge_lin`
  - `eq_lin`
- fresh variable generator (`FreshGen`)
- XOR network encoding
  - `NN_single(...)`
  - `NN_dual(...)`

인코딩 특징:
- hidden layer: affine + ReLU
- output layer: logit 계산
- sigmoid는 직접 인코딩하지 않고 **logit sign**으로 class 판단

즉, 이 파일은 **신경망을 논리 제약식으로 바꾸는 역할**을 합니다.

---

### 🔹 `Robustness.py`

입력 중심점 주변의 작은 perturbation에 대해, 네트워크가 **같은 클래스를 유지하는지** 확인하는 robustness 검증 예제입니다.

주요 내용:
- L∞ 박스 형태의 precondition 생성
- same-class by logit postcondition 생성
- 전체 spec 구성
- counterexample model 출력
- XOR 중심점 `(0,0), (0,1), (1,0), (1,1)`에 대한 robustness 검증

검증 형태:
- 명세: `(precondition ∧ NN) -> postcondition`
- 실제 실행: `precondition ∧ NN ∧ ¬postcondition`

즉, 이 파일은 **robustness property를 DPLL(T)로 검증하는 예제 스크립트**입니다.

---

### 🔹 `PreciseEncoding.py`

XOR 문제를 **입력 구간별로 더 직접적으로 검증**하고, 반례를 여러 개 찾으면서 동작을 분석하는 파일입니다.

주요 내용:
- `zero(i)`, `one(i)` 입력 클래스 정의
- 출력 logit 기준 class 정의
- XOR case별 counterexample formula 생성
- 각 입력 region에 대해:
  - 반례 탐색
  - 최대 여러 개의 counterexample 수집
  - 각 반례 탐색 시간 출력
  - 추가 blocking constraint로 이전 반례 주변 제외
- 더 이상 반례가 없을 경우:
  - 랜덤 샘플링으로 실제 XOR 동작 검산

특징:
- `(x1, x2)`의 각 region을 분리해서 분석 가능
- 하나의 counterexample만 찾는 것이 아니라, **여러 반례를 순차적으로 수집**
- 마지막 UNSAT 상황에서 샘플링 기반 sanity check 수행

즉, 이 파일은 **XOR 네트워크의 정밀 디버깅/반례 탐색용 실험 스크립트**입니다.

---

### 🔹 `LocalRobustnessSweep.py`

XOR 신경망에 대해 **epsilon 값을 점진적으로 증가시키면서 local robustness가 언제 깨지는지** 추적하는 스윕 스크립트입니다.

주요 내용:
- `eps_values = np.arange(...)` 로 perturbation 반경을 일정 간격으로 증가시키며 검사
- XOR의 4개 중심 입력 케이스
  - `(0,0) -> 0`
  - `(0,1) -> 1`
  - `(1,0) -> 1`
  - `(1,1) -> 0`
  에 대해 각각 robustness 위반 여부 확인
- 각 epsilon에서 SAT(반례 존재)한 케이스 수를 집계
- 이전 epsilon 대비 SAT 개수가 증가하는 지점을 **milestone**으로 기록
- milestone마다:
  - 어떤 케이스들이 처음 깨졌는지
  - 솔버가 찾은 counterexample 입력이 무엇인지
  를 저장
- sweep 결과를 step plot으로 시각화
- sweep 후 발견된 각 반례를 실제 `xor_nn_sx(...)`로 다시 계산하여 자동 검산
- 경계점 수치 오차가 의심될 경우, 입력을 미세하게 밀어내어 오답 여부를 재확인

특징:
- 단일 epsilon 검사가 아니라 **전체 epsilon 구간에 대한 robustness profile**을 볼 수 있음
- “언제 처음 1개 케이스가 깨지고, 언제 2개/4개로 늘어나는지” 같은 **계단형 고장 패턴**을 분석 가능
- 이론적 경계와 실제 솔버가 찾는 반례 발생 시점을 함께 비교 가능

즉, 이 파일은 **local robustness의 붕괴 시점을 스윕 기반으로 분석하고, 점프 구간별 반례를 수집/검산하는 실험 스크립트**입니다.

---

### 🔹 `visualize_prop.py`

검증 대상 논리식과 Tseitin 변환 후 CNF를 **Graphviz 그래프 형태로 시각화**하는 유틸리티입니다.

주요 내용:
- `Prop` 트리를 DOT 형식으로 변환
- Tseitin 변환 후 CNF를 다시 Prop 형태로 복원하여 시각화
- theory atom 추상화 결과(`atom_map`)를 범례 형태로 함께 표시
- `.dot` 파일 저장 및 Graphviz가 설치되어 있으면 `.png` 렌더링 가능

생성 파일:
- `neg_spec_tree.dot` / `neg_spec_tree.png`
  - 변환 전 검증 대상식 트리
- `neg_spec_cnf.dot` / `neg_spec_cnf.png`
  - Tseitin 변환 후 CNF 구조와 atom_map 시각화

즉, 이 파일은 **검증식 구조와 CNF 팽창 양상을 디버깅/발표용으로 확인하는 도구**입니다.

---

## 🚀 How to Run


## 1. Requirements

기본적으로 Python 3 환경에서 실행됩니다.

사용 라이브러리:
- `numpy`

설치 예시:

```bash
pip install numpy
```

---

## 2. Run Each Module

### `DPLL.py`

문자열로 작성한 논리식을 파싱하고, NNF / CNF / DPLL 결과를 확인할 수 있습니다.

```bash
python DPLL.py
```

입력 예시:

```text
(p and q) or not r
not (p -> q)
ineq(1,x,0) or p
ineq(1,x,1,y,2,z,-5)
ineq(1,x,0) or relu(x,y)
```

지원 문법:
- `and`, `or`, `not`, `~`, `->`
- 괄호 `( ... )`
- `true`, `false`
- `ineq(c1,x1,c2,x2,...,b)` : `c1*x1 + c2*x2 + ... >= b`
- `relu(x,y)` : `y = relu(x)`

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
- SAT 선형 제약
- UNSAT 선형 제약
- 다변수 연립 제약

---

### `Reluplex.py`

Reluplex solver 자체 테스트를 실행합니다.

```bash
python Reluplex.py
```

포함된 예시:
- `x + y >= 5`, `y = relu(x)`
- `x >= 0`, `y = relu(x)`, `y < 0` (UNSAT)
- 기타 ReLU 포함 제약 예시

---

### `XOREncoding.py`

XOR 신경망 인코딩 결과를 확인합니다.

```bash
python XOREncoding.py
```

출력 내용:
- `NN_dual(...)` 생성 결과
- `NN_single(...)` 생성 결과
- 생성된 fresh variable 이름

---

### `Robustness.py`

XOR 중심점 주변의 입력 perturbation에 대해 **same-class robustness**를 검사합니다.

```bash
python Robustness.py
```

동작:
- 중심점 `c ∈ {(0,0), (0,1), (1,0), (1,1)}`
- 각 중심점 주변 입력 박스 생성
- `s_x` 와 `s_c`의 부호가 같은지 확인
- 반례가 있으면 counterexample 출력
- 없으면 해당 영역에서 robust하다고 판단

---

### `PreciseEncoding.py`

입력 region별 XOR 반례 탐색 및 검산을 수행합니다.

```bash
python PreciseEncoding.py
```

동작:
- 입력 영역을 4개 case로 분리
- 각 case마다 `pre ∧ NN ∧ ¬post` 형태의 counterexample query 생성
- `dpll_t(...)`로 반례 탐색
- 여러 반례를 찾기 위해 이전 반례 주변을 제외하는 제약 추가
- 각 반례 탐색에 걸린 시간 출력
- 더 이상 반례가 없으면 랜덤 샘플링으로 검산

---

### `visualize_prop.py`

검증 대상식과 Tseitin 변환 후 CNF 식을 Graphviz DOT/PNG로 시각화합니다.

```bash
python visualize_prop.py
```

실행 결과:
- `neg_spec_tree.dot`
- `neg_spec_tree.png` (Graphviz 설치 시 렌더링)
- `neg_spec_cnf.dot`
- `neg_spec_cnf.png` (Graphviz 설치 시 렌더링)

용도:
- 변환 전 논리식 트리 구조 확인
- Tseitin 변환 후 CNF 구조 확인
- theory atom이 어떤 Boolean atom으로 추상화되었는지 확인
- 발표 자료 / 디버깅 자료 생성

---

### `LocalRobustnessSweep.py`

epsilon sweep을 수행하여 **XOR 신경망의 local robustness가 어느 지점에서 깨지는지** 분석합니다.

```bash
python LocalRobustnessSweep.py
```

동작:
- epsilon을 일정 간격으로 증가시키며 4개 XOR 중심 케이스를 순차 검증
- 각 epsilon에서 반례가 존재하는 케이스 수(SAT count) 기록
- SAT count가 증가하는 지점을 milestone으로 저장
- 발견된 milestone별 반례 입력을 출력
- 마지막에 각 반례를 실제 네트워크 함수로 다시 계산해 자동 검산
- 결과를 그래프로 표시하여 robustness 붕괴 패턴을 시각적으로 확인

출력 예시 의미:
- `SAT 발생 수 = 0` : 아직 모든 케이스가 robust
- `SAT 발생 수 = 2` : 4개 중 2개 입력 중심 근처에서 반례 존재
- `SAT 발생 수 = 4` : 모든 중심 케이스 주변에서 robustness 붕괴

용도:
- 특정 epsilon 하나만 보는 것이 아니라, **robustness가 깨지는 임계 구간 전체를 보고 싶을 때**
- 이론적 경계와 실제 반례 발생 지점을 비교하고 싶을 때
- milestone별 counterexample을 발표/분석 자료로 정리하고 싶을 때

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
- `x < 0`  
  → `-x >= 1e-6`
- `x <= 0`  
  → `-x >= 0`

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

- `s >= 0`  → class 1
- `s < 0`   → class 0

이는 sigmoid의 단조성 때문에 가능합니다.

---

## 🧪 Example Verification Tasks

이 저장소로 실험할 수 있는 대표적인 검증 작업:

1. **Boolean + theory satisfiability**
   - `ineq(...) and relu(...)`

2. **Theory conflict detection**
   - Boolean level SAT
   - theory level UNSAT

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
- theory atom의 negation은 `DPLL_T.py`에서 근사적으로 처리됩니다.
- strict inequality는 직접 지원하지 않으므로 작은 epsilon 기반 근사가 필요합니다.
- counterexample를 여러 개 찾기 위해 OR 기반 blocking constraint를 누적하면, 마지막 UNSAT 확인이 느려질 수 있습니다.
- `PreciseEncoding.py`에서 여러 반례를 찾을수록 탐색 공간이 조각나므로 시간이 크게 증가할 수 있습니다.
- `visualize_prop.py`의 PNG 렌더링은 Graphviz `dot` 명령이 설치되어 있어야 합니다. 설치되지 않은 경우 `.dot` 파일만 생성됩니다.


---

## 🔍 Recommended Entry Points

처음 볼 때는 아래 순서로 실행해보는 것을 추천합니다.

1. `python DPLL.py`  
   → 논리식 파싱 / CNF 변환 / DPLL 확인

2. `python DPLL_T.py`  
   → Boolean + Theory 결합 흐름 확인

3. `python Simplex.py`  
   → 선형 solver 동작 확인

4. `python Reluplex.py`  
   → ReLU 처리 방식 확인

5. `python XOREncoding.py`  
   → XOR 네트워크 인코딩 구조 확인

6. `python Robustness.py`  
   → robustness 예제 실행

7. `python PreciseEncoding.py`  
   → 입력 region별 정밀 반례 탐색 및 검산

8. `python LocalRobustnessSweep.py`  
   → epsilon sweep 기반 local robustness 붕괴 지점 분석

9. `python visualize_prop.py`  
   → 검증 대상식과 CNF 구조 시각화


---

## 📚 Summary

이 프로젝트는 다음을 직접 구현하고 연결합니다.

- propositional logic AST
- NNF / Tseitin CNF transformation
- DPLL SAT solver
- DPLL(T) loop
- simplex-based linear arithmetic solving
- Reluplex-style ReLU solving
- XOR neural network verification examples
- robustness and counterexample search workflows
- **epsilon sweep 기반 local robustness milestone 분석**
- formula tree / Tseitin CNF visualization workflows


작고 명시적인 Python 구현으로,  
**ReLU 신경망 검증의 핵심 아이디어를 단계별로 실험하고 이해하기 좋은 구조**를 목표로 합니다.