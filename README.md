# Ai_verification
ReLUplex Implementation in Python
📌 Overview




📂 Project Structure

        ├── Reluplex.py

        ├── Simplex.py

        ├── Tseitin_Transformation.py

        ├── PropClassDPLL.py
        
        ├──  DPLL(T).py

        └── README.md
        

📄 File Descriptions

🔹 Reluplex.py

ReLUplex 알고리즘의 핵심 로직이 구현된 파일입니다.

ReLU 제약 처리

Active / Inactive case 분기

Simplex 호출 및 제약 해결

충돌 발생 시 branching 수행

전체 검증 흐름 제어

이 파일은 신경망 검증의 메인 실행 모듈 역할을 합니다.

🔹 Simplex.py

선형 제약을 해결하기 위한 Simplex 알고리즘 구현 파일입니다.

Simplex tableau 구성

Pivot 연산

Basic / Non-basic 변수 관리

변수의 하한(lower bound, l) / 상한(upper bound, u) 처리

Feasibility 검사

Reluplex의 기반 solver로 동작합니다.

🔹 Tseitin_Transformation.py

논리식을 CNF(Conjunctive Normal Form) 형태로 변환하는 Tseitin Transformation 구현 파일입니다.

Boolean 식을 표현할 Prop 클래스 정의

Prop 클래스로 표현한 Boolean 식을 CNF로 변환

보조 변수 도입

SAT solver 입력 형식 생성

논리 기반 제약을 SAT 문제로 변환할 때 사용됩니다.

🔹 DPLL.py

SAT 문제 해결을 위한 백트래킹 기반 탐색 알고리즘

Unit Propagation 수행

Pure Literal Elimination 처리

충돌 발생 시 backtracking 수행

CNF(Conjunctive Normal Form) 입력 처리

🔹 DPLL(T).py

DPLL(T) 메인 루프를 구현한 파일입니다. 

이 모듈은 Boolean 추상화(tseitin CNF)를 SAT로 해결한 뒤,


활성화된 이론(Theory) 원자들을 `Reluplex.reluplex` 이론 솔버에 전달하여 실수(또는 ReLU) 제약의 유효성을 검사합니다. 

이 과정에서 이론 충돌이 발생하면 해당 불리언 할당을 차단하는 clause를 CNF에 추가하고 반복합니다.

Relu 관련 원자(`relu(x,y)`)와 선형 부등식(`ineq(...)`)이 지원됩니다.

🚀 How to Run

### Tseitin_Transformation.py
main 함수의 phi 변수에 Prop클래스로 표현된 식을 할당 후 실행 시키면 출력으로
cnf형식으로 바꾼 식, 입력식, nnf형식으로 바꾼 식, 임시 변수에 할당된 값 매핑 정보, cnf 절들의 정보
순서로 알려줍니다.

예시: phi = NotProp( AndProp( VarProp( "p" ), VarProp( "q" ) ) )

python Tseitin_Transformation.py



### DPLL(T).py
간단한 사용 예:

```bash
# 데모 실행 (예: x + y >= 5 그리고 y = relu(x) 제약)
python "DPLL(T).py"

# 또는 파이썬에서 직접 호출
python -c "from DPLL import parse_prop; from DPLL(T) import dpll_t; p = parse_prop('ineq(1,x,1,y,5) and relu(x,y)'); print(dpll_t(p))"

# 상수 사용 예시
python -c "from DPLL import parse_prop; from DPLL(T) import dpll_t; p = parse_prop('ineq(1,x,0) and relu(x,y) and ineq(-1,y,1e-6)'); print(dpll_t(p))"
```

입력 문법 : DPLL(T) 알고리즘은 제약식을 PropClass 객체 형태로 전달받습니다.

parse_prop() 함수는 특정 문법 형식으로 작성된 논리식을 파싱하여, 내부에서 사용할 수 있는 PropClass 인스턴스들로 변환합니다.

parse_prop()는 다음과 같은 특정 형식으로 표현된 식을 PropClass들로 만들어줍니다.
- `ineq(c1,x1,c2,x2,...,b)` : c1*x1 + c2*x2 + ... >= b
- `relu(x,y)` : y = relu(x)
- `a and b` : a ^ b
- `ineq(-1,y,1e-6)` : y < 0

ex) parse_prop(`p and q`) =>  AndProp( VarProp( "p" ), VarProp( "q" ) )

참고: `DPLL(T).py`는 내부에서 `tseitin_cnf`/`dpll` (파일: `DPLL.py`)와
`reluplex` (파일: `Reluplex.py`)를 사용하므로 두 모듈이 함께 존재해야 합니다.


