# Ai_verification
ReLUplex Implementation in Python
📌 Overview


📂 Project Structure
.
├── Reluplex.py
├── Simplex.py
├── Tseitin_Transformation.py
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


🚀 How to Run

### Tseitin_Transformation.py
main 함수의 phi 변수에 Prop클래스로 표현된 식을 할당 후 실행 시키면 출력으로
cnf형식으로 바꾼 식, 입력식, nnf형식으로 바꾼 식, 임시 변수에 할당된 값 매핑 정보, cnf 절들의 정보
순서로 알려줍니다.

예시: phi = NotProp( AndProp( VarProp( "p" ), VarProp( "q" ) ) )

python Tseitin_Transformation.py


