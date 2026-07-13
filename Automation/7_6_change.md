# 7/6 변경 내역

## 신규 파일

### `PropertyBuilder.py`

- 설정 파일을 바탕으로 precondition과 postcondition을 생성하는 기능 추가
- 입력별로 서로 다른 epsilon을 지정하는 기능 추가
- 입력의 유효 범위와 검증 범위를 자동으로 교차하는 기능 추가
- 원시 입력을 `(raw - mean) / scale` 방식으로 정규화하는 기능 추가
- 다음 출력 형식에 대한 postcondition 생성 기능 추가
  - 이진 분류
  - 다중 클래스 `argmax`/`argmin`
  - 다중 라벨
  - 회귀 출력 범위 및 목표값 허용 오차
- 중심점의 예측 결과를 기준 라벨로 계산하는 기능 추가

### `ModelInspector.py`

- ONNX 및 커스텀 텍스트 모델의 입출력 크기와 레이어 구조 검사 기능 추가
- 지원하지 않는 분기 그래프와 은닉 활성화 함수를 탐지하는 기능 추가
- 마지막 Sigmoid/Softmax 제거 여부와 logit 공간 사용 여부 검사 기능 추가
- ONNX 입력 전처리 `Clip`, `Max`, `Min`, `Add`, `Sub`, `Mul`, `Div` 분석 기능 추가
- ONNX에서 입력 범위, 평균, 정규화 scale을 자동으로 추출하는 기능 추가
- 모델 정보를 바탕으로 검증 설정 JSON 템플릿을 생성하는 기능 추가
- 은닉 ReLU 개수를 계산하는 기능 추가

### `AutoVerify.py`

- 모델 검사, 설정 생성, dry-run, 실제 검증을 수행하는 통합 CLI 추가
- ONNX를 커스텀 파일로 저장하지 않고 메모리에서 직접 인코딩하는 기능 추가
- 설정의 모델 입출력 계약과 실제 모델 크기를 비교하는 기능 추가
- `expected` 및 `center_prediction` 기준 검증 기능 추가
- SAT 반례의 입력과 출력을 원래 입력 공간으로 변환하여 출력하는 기능 추가
- 대형 모델의 실수 실행을 방지하는 ReLU 개수 제한 기능 추가
- 상세 결과를 JSON으로 저장하는 기능 추가
- `SAT`, `UNSAT`, `UNKNOWN`을 구분하는 공통 솔버 결과 구조 추가
- 시간·라운드·Simplex 반복·Reluplex 재귀/복구 한도 초과를 `UNKNOWN`으로 처리
- 확정 UNSAT만 `VERIFIED`로 표시하고 UNKNOWN 사유와 경과시간을 JSON에 기록
- JSON 없이 ONNX와 VNNLIB를 직접 결합하는 `verify-vnnlib` 명령 추가
- VNNLIB의 `X_i`, `Y_i`를 내부 신경망 입력·출력 변수에 자동 연결
- `.vnnlib.gz` 파일을 별도 압축 해제 없이 실행하는 기능 추가

### `VnnlibParser.py`

- VNNLIB S-expression 토큰화 및 파싱 기능 추가
- 선형 산술식과 `<`, `<=`, `>`, `>=`, `=` 비교식 변환 기능 추가
- `and`, `or`, `not`, implication 논리식의 `Prop` 변환 기능 추가
- VNNLIB 입력·출력 선언 개수와 신경망 크기 검증 기능 추가
- 비선형 곱셈과 변수 나눗셈을 명확히 거부하도록 처리
- 일반 VNNLIB 및 gzip 압축 VNNLIB 읽기 기능 추가

### `SolverStatus.py`

- `SAT`, `UNSAT`, `UNKNOWN` 공통 상태와 상세 결과 자료구조 추가
- 시간 제한과 내부 반복 한도를 상위 솔버로 전달하는 예외 구조 추가
- 종료 시각을 검사하고 시간 초과를 `TIMEOUT`으로 전달하는 공통 함수 추가

### `__init__.py`

- 자동화 코드를 Python 패키지 경로로 가져올 수 있도록 패키지 초기화 파일 추가

### `.gitignore`

- 자동화 실행 중 생성되는 임시 파일과 캐시가 변경 내역에 포함되지 않도록 설정

### `VNNCOMP_RUNBOOK.md`

- 실제 VNN-COMP 2025 모델 경로를 사용한 실행 절차 추가
- `.onnx.gz` 모델 압축 해제 방법 추가
- JSON 설정 없이 단일 모델 검사, VNNLIB dry-run, 실제 솔버 실행 순서 추가
- 전체 ONNX 모델의 프런트엔드 호환성을 CSV로 검사하는 명령 추가
- 프런트엔드 한계와 솔버 한계를 구분하는 기준 추가
- `setup.sh` 실행 완료 여부에 따른 ONNX 준비 절차 분리
- CMD에서 Conda 설치 확인, `crown311` 생성·활성화 및 패키지 검증 절차 추가
- `Automation` 폴더 구조와 `--timeout-seconds` 옵션을 반영한 명령으로 갱신
- `VERIFIED`, `COUNTEREXAMPLE`, `UNKNOWN` 판정과 세부 종료 사유 정리
- 제목, 목록, 표, 명령 블록을 표준 Markdown 형식으로 정리

### `VNNCOMP_EXPERIMENT_REPORT.md`

- 지금까지의 자동화 구성, Smoke 검증과 VNN-COMP 실행 과정을 하나의 보고서로 정리
- ACAS Xu prop_1~6 및 SafeNLP RuARobot prop_1~10 결과와 실행 시간 표 추가
- timeout, Simplex, Reluplex, 프런트엔드 및 수치 처리 한계 분석 추가
- theory-conflict 차단 절의 건전성 문제와 결과 해석 주의사항 기록
- 현재 결과로 말할 수 있는 범위와 개선 우선순위 정리
- 원하는 위치의 ONNX와 VNNLIB 경로를 직접 지정하는 CMD 실행 절차 추가
- `--json-output`은 수정용 설정이 아닌 실행 결과 파일임을 명시

### `Automation/tests/test_automation.py`

- 입력별 epsilon과 입력 범위 교차 테스트 추가
- raw 입력 정규화 테스트 추가
- 이진·다중 클래스·다중 라벨·회귀 postcondition 테스트 추가
- ONNX 입력 전처리 추출 테스트 추가
- XOR 모델 dry-run 테스트 추가
- 잘못된 XOR 기대 라벨에서 실제 반례가 반환되는지 확인하는 테스트 추가
- ACAS ONNX의 구조와 정규화 값 추출 테스트 추가
- 별도 예제 JSON 없이 테스트 내부 설정을 사용하도록 구성
- VNNLIB 선형 산술식, Boolean 조합, gzip 입력, 비선형식 거부 테스트 추가
- XOR 모델과 VNNLIB unsafe 조건을 결합한 실제 반례 탐색 테스트 추가

### `Automation/SmokeTest/`

- `Y_0 = ReLU(X_0)`인 1입력·1ReLU·1출력 ONNX 생성 스크립트 추가
- JSON 없이 실행할 수 있는 반례 존재·부재 VNNLIB 예제 추가
- 프런트엔드, VNNLIB 연결, SAT 반례와 UNSAT 확인을 순서대로 검사하는 절차 추가
- 원본 DPLL(T), Reluplex, Simplex의 로드 경로와 함수 연결 검사 추가
- Smoke ONNX의 노드·가중치·계산식과 두 VNNLIB의 판정 근거 문서화

## 기존 파일 수정

### 자동화 폴더 구조 정리

- 자동화 코드와 문서를 `Ai_verification/Automation/` 아래로 이동
- 실행 결과를 `Automation/Results/`, 설정을 `Automation/Specs/`로 이동
- 기존 솔버 파일은 원래 위치에 유지하고 공통 상태 모듈만 패키지 경로로 연결
- 테스트 실행 경로와 모든 CLI 문서의 경로를 새 폴더 구조로 변경

### `DPLL.py`

- DPLL 실행 과정에 선택적 종료 시각을 전달할 수 있도록 함수 인자 확장
- CNF 단순화, 단위 전파, 순수 리터럴 제거와 재귀 분기 중 시간 제한 검사 추가
- 시간 초과 시 일반적인 UNSAT 반환 대신 상위 솔버가 `UNKNOWN`으로 처리할 수
  있도록 공통 한도 초과 예외를 전달

### `DPLL_T.py`

- 기존 `(model, sat)` 반환 형식과 별도로 `dpll_t_detailed()` 상세 실행 함수 추가
- 상세 실행 결과에 `SAT`, `UNSAT`, `UNKNOWN`, 종료 사유, 라운드 수와 경과시간 기록
- Boolean UNSAT과 theory UNSAT은 확정 `UNSAT`으로 구분하고, 최대 라운드 소진은
  `DPLL_T_ROUND_LIMIT` 사유의 `UNKNOWN`으로 처리
- DPLL, Reluplex 및 Simplex까지 동일한 종료 시각을 전달하여 전체 실행 시간 제한 적용
- 기존 `dpll_t()` 호출 방식은 유지하되 `UNKNOWN` 발생 시 UNSAT처럼 반환하지 않고
  한도 초과 예외로 알리도록 변경

### `Simplex.py`

- 선택적 종료 시각과 상세 상태 보고 옵션 추가
- 각 반복에서 시간 제한을 검사하고 시간 초과를 상위 솔버로 전달
- 최대 반복 횟수 소진을 논리적 UNSAT과 구분하여
  `SIMPLEX_ITERATION_LIMIT` 사유의 `UNKNOWN`으로 처리
- 기존 직접 호출은 이전 반환 형식을 유지하도록 호환성 보존

### `Reluplex.py`

- 선택적 종료 시각과 상세 상태 보고 옵션 추가
- 초기 Simplex 실행, 지역 복구, 재귀 분기마다 시간 제한을 검사하도록 변경
- 최대 재귀 깊이 도달을 `RELUPLEX_RECURSION_LIMIT` 사유의 `UNKNOWN`으로 처리
- 지역 복구만으로 결론을 내리지 못한 경우를
  `RELUPLEX_REPAIR_INCONCLUSIVE` 사유의 `UNKNOWN`으로 처리
- 한 분기가 `UNKNOWN`이면 다른 분기에서 SAT를 찾을 수 있도록 계속 검사하되,
  최종적으로 결론이 나지 않으면 해당 `UNKNOWN` 사유를 보존
- 상세 상태 보고를 사용하지 않는 기존 직접 호출은 이전 반환 형식 유지

### `FormatConverters/OnnxToCustom.py`

- 첫 FC 레이어 이전의 입력 전처리 연산을 건너뛸 수 있도록 지원 연산 추가
- 입력 전처리의 나머지 입력이 상수 initializer인지 검사하도록 보완
- 구형 ONNX에서 initializer가 `graph.input`에 중복 등록된 경우 실제 데이터
  입력에서 제외하도록 보완
- 기존 FC/ReLU 변환 동작은 그대로 유지

### `README.md`

- Reluplex 최대 재귀 깊이 초과 결과를 UNSAT이 아닌 `UNKNOWN`으로 설명하도록 수정
