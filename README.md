# Ai_verification
ReLUplex Implementation in Python
📌 Overview

이 프로젝트는 ReLU 활성화 함수를 포함한 신경망 검증 알고리즘(ReLUplex) 을 Python으로 구현한 코드입니다.

ReLUplex는 선형 제약을 해결하는 Simplex 알고리즘을 기반으로 하여, ReLU 제약을 처리하기 위해 분기(case-splitting) 전략을 추가한 확장 알고리즘입니다.

또한, 논리식을 CNF로 변환하기 위한 Tseitin Transformation 모듈도 포함되어 있습니다.

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

Boolean 식을 CNF로 변환

보조 변수 도입

SAT solver 입력 형식 생성

논리 기반 제약을 SAT 문제로 변환할 때 사용됩니다.

⚙️ Requirements

Python 3.x

(추가 라이브러리가 있다면 여기에 작성)

🚀 How to Run

예시:

python Reluplex.py


또는 별도의 메인 실행 파일이 있다면:

python main.py

🧠 Background

ReLU 기반 신경망 검증 알고리즘: ReLUplex

기반 알고리즘: Simplex

논리식 변환 기법: Tseitin Transformation

📝 Notes

ReLU 제약은 active/inactive 분기를 통해 처리됩니다.

무한 루프 방지를 위한 branching 전략이 포함될 수 있습니다.

Bound tightening 및 feasibility 검사 로직이 포함되어 있습니다.
