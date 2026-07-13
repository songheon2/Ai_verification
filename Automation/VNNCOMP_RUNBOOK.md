# VNN-COMP 2025 실행 명령 정리

> 기준 터미널: VS Code Command Prompt(`cmd.exe`)  
> 기준 작업 폴더: `D:\Verification_semina\reluplex`  

## 0. Conda 환경 설정

### 0.1 Conda 설치 확인

```bat
conda init cmd.exe
```

명령 실행 후 VS Code의 기존 터미널을 닫고 새 Command Prompt 터미널을 연다.

### 0.2 `crown311` 환경 생성

```bat
conda create --name crown311 python=3.11 -y
```

### 0.3 환경 활성화 및 작업 폴더 이동

```bat
conda activate crown311
cd /d D:\Verification_semina\reluplex
```

CMD를 새로 열 때마다 위 두 명령을 다시 실행한다.

## 1. 필요한 패키지 설치

Conda 환경을 처음 만들었을 때 한 번만 실행한다.

```bat
python -m pip install --upgrade pip
python -m pip install numpy onnx
```

## 2. VNN-COMP 경로 등록

```bat
set "VNN_ROOT=D:\Verification_semina\vnn_comp\vnncomp2025_benchmarks"
set "BENCHMARK_ROOT=%VNN_ROOT%\benchmarks"
```

## 3. ONNX 준비 상태 확인

압축 해제된 ONNX 개수를 확인한다.

```bat
dir /s /b "%BENCHMARK_ROOT%\*.onnx" 2>nul | find /c /v ""
```

- `setup.sh`가 정상 완료되어 개수가 1개 이상이면 4번으로 이동한다.
- 결과가 `0`이면 아래 명령으로 ACAS-Xu 모델 하나만 압축 해제한다.

```bat
wsl gzip -dk "/mnt/d/Verification_semina/vnn_comp/vnncomp2025_benchmarks/benchmarks/acasxu_2023/onnx/ACASXU_run2a_1_1_batch_2000.onnx.gz"
```

`setup.sh`가 정상 완료됐다면 위 `gzip` 명령은 실행하지 않는다.

## 4. 검사할 모델 경로 등록

```bat
set "MODEL=%BENCHMARK_ROOT%\acasxu_2023\onnx\ACASXU_run2a_1_1_batch_2000.onnx"
```

## 5. ONNX 모델 구조 검사

```bat
python Ai_verification\Automation\AutoVerify.py inspect "%MODEL%"
```

예상 출력 형태:

```text
Model format : onnx
Layers       : 5 -> 50 -> 50 -> 50 -> 50 -> 50 -> 50 -> 5
Hidden ReLUs: 300
```

`error:`가 출력되면 현재 프런트엔드가 해당 ONNX 연산 또는 그래프 구조를 지원하지 않는 것이다.

## 6. VNNLIB 속성 파일 지정

```bat
set "VNNLIB=%BENCHMARK_ROOT%\acasxu_2023\vnnlib\prop_1.vnnlib.gz"
```

`prop_2`를 지정하려면 다음 명령을 사용한다.

```bat
set "VNNLIB=%BENCHMARK_ROOT%\acasxu_2023\vnnlib\prop_2.vnnlib.gz"
```

VNNLIB는 `.vnnlib.gz` 상태로 직접 읽을 수 있으므로 압축을 풀 필요가 없다.

## 7. ONNX + VNNLIB dry-run

```bat
python Ai_verification\Automation\AutoVerify.py verify-vnnlib --model "%MODEL%" --vnnlib "%VNNLIB%" --dry-run
```

예상 출력 형태:

```text
Model  : 5 -> 50 -> 50 -> 50 -> 50 -> 50 -> 50 -> 5
Asserts: 11
Result : DRY_RUN
```

`DRY_RUN`은 ONNX 로드, VNNLIB 파싱, 변수 연결과 신경망 인코딩까지 성공했다는 의미다. 
실제 DPLL(T)/Reluplex는 실행하지 않는다.

## 8. 결과 폴더 생성

```bat
if not exist "Ai_verification\Automation\Results" mkdir "Ai_verification\Automation\Results"
```

## 9. 속성 하나만 실제 실행

ACAS-Xu 모델은 은닉 ReLU가 300개이므로 `--allow-large-model`이 필요하다.
`--timeout-seconds 300`은 솔버가 300초 안에 결론을 확정하지 못하면 `UNKNOWN`으로 종료하기 위한 제한이다.

### 9.1 `prop_1`, 최대 5,000 라운드

```bat
set "VNNLIB=%BENCHMARK_ROOT%\acasxu_2023\vnnlib\prop_1.vnnlib.gz"

python -c "import sys, runpy; sys.setrecursionlimit(100000); sys.path.insert(0, r'Ai_verification'); sys.argv=[r'Ai_verification\Automation\AutoVerify.py']+sys.argv[1:]; runpy.run_path(r'Ai_verification\Automation\AutoVerify.py', run_name='__main__')" verify-vnnlib --model "%MODEL%" --vnnlib "%VNNLIB%" --allow-large-model --max-rounds 5000 --timeout-seconds 300 --json-output "Ai_verification\Automation\Results\acasxu_prop_1_round5000.json"
```

### 9.2 `prop_2`, 최대 5,000 라운드

```bat
set "VNNLIB=%BENCHMARK_ROOT%\acasxu_2023\vnnlib\prop_2.vnnlib.gz"

python -c "import sys, runpy; sys.setrecursionlimit(100000); sys.path.insert(0, r'Ai_verification'); sys.argv=[r'Ai_verification\Automation\AutoVerify.py']+sys.argv[1:]; runpy.run_path(r'Ai_verification\Automation\AutoVerify.py', run_name='__main__')" verify-vnnlib --model "%MODEL%" --vnnlib "%VNNLIB%" --allow-large-model --max-rounds 5000 --timeout-seconds 300 --json-output "Ai_verification\Automation\Results\acasxu_prop_2_round5000.json"
```

## 10. `prop_1`부터 `prop_6`까지 연속 실행

### 10.1 최대 5,000 라운드

```bat
for /L %P in (1,1,6) do @(echo ==== prop_%P round5000 ==== & python -c "import sys, runpy; sys.setrecursionlimit(100000); sys.path.insert(0, r'Ai_verification'); sys.argv=[r'Ai_verification\Automation\AutoVerify.py']+sys.argv[1:]; runpy.run_path(r'Ai_verification\Automation\AutoVerify.py', run_name='__main__')" verify-vnnlib --model "%MODEL%" --vnnlib "%BENCHMARK_ROOT%\acasxu_2023\vnnlib\prop_%P.vnnlib.gz" --allow-large-model --max-rounds 5000 --timeout-seconds 300 --json-output "Ai_verification\Automation\Results\acasxu_prop_%P_round5000.json")
```

### 10.2 최대 10,000 라운드

```bat
for /L %P in (1,1,6) do @(echo ==== prop_%P round10000 ==== & python -c "import sys, runpy; sys.setrecursionlimit(100000); sys.path.insert(0, r'Ai_verification'); sys.argv=[r'Ai_verification\Automation\AutoVerify.py']+sys.argv[1:]; runpy.run_path(r'Ai_verification\Automation\AutoVerify.py', run_name='__main__')" verify-vnnlib --model "%MODEL%" --vnnlib "%BENCHMARK_ROOT%\acasxu_2023\vnnlib\prop_%P.vnnlib.gz" --allow-large-model --max-rounds 10000 --timeout-seconds 300 --json-output "Ai_verification\Automation\Results\acasxu_prop_%P_round10000.json")
```

> `.bat` 또는 `.cmd` 파일 안에 반복문을 넣을 때는 `%P`를 `%%P`로 변경한다.

## 11. 결과 해석 기준

| 결과 | 의미 |
|---|---|
| `DRY_RUN` | ONNX 로드, VNNLIB 파싱과 신경망 인코딩 성공 |
| `COUNTEREXAMPLE` | unsafe 조건을 만족하는 반례 발견, 솔버 상태 `SAT` |
| `VERIFIED` | 반례 탐색식이 `UNSAT`으로 확정됨 |
| `UNKNOWN` | 시간 또는 라운드·반복·재귀 한도로 `SAT/UNSAT`을 확정하지 못함 |
| `error:` | 프런트엔드 또는 솔버가 해당 입력을 처리하지 못함 |

`UNKNOWN`이면 결과 JSON의 `solver.reason`을 확인한다.

| `solver.reason` | 의미 |
|---|---|
| `TIMEOUT` | `--timeout-seconds` 시간 제한 초과 |
| `DPLL_T_ROUND_LIMIT` | `--max-rounds` 제한 초과 |
| `SIMPLEX_ITERATION_LIMIT` | Simplex 반복 제한 초과 |
| `RELUPLEX_RECURSION_LIMIT` | Reluplex 재귀 깊이 제한 초과 |
| `RELUPLEX_REPAIR_INCONCLUSIVE` | ReLU 복구 과정에서 결론을 확정하지 못함 |

## 12. 현재 구현 범위

주요 지원 구조:

- Fully Connected
- `MatMul`, `Add`, `Gemm`
- `ReLU`
- `Flatten`

현재 거부될 수 있는 구조:

- `Conv`
- `BatchNormalization`
- `MaxPool`
- 은닉층 `Sigmoid`, `Tanh`, `Softmax`
- Residual 또는 Skip connection
- 분기 그래프

## 13. 임의의 모델 직접 준비

모델은 `Ai_verification` 폴더 안에 둘 필요가 없다. 원하는 위치의 ONNX와 VNNLIB를 절대경로로 지정하면 된다. 

- ONNX: 검증할 신경망 구조와 가중치
- VNNLIB: 입력 범위와 unsafe 출력 조건

ONNX만으로는 어떤 조건을 검증해야 하는지 알 수 없으므로 VNNLIB는 필요하다.

### 13.1 파일 경로 지정

아래 두 경로만 실제 파일 위치로 바꾼다.

```bat
set "TEST_MODEL=D:\원하는경로\model.onnx"
set "TEST_VNNLIB=D:\원하는경로\property.vnnlib"
```

압축된 VNNLIB도 그대로 사용할 수 있다.

### 13.2 실제 검증 실행

소형 모델은 다음 명령으로 실행한다.
결과를 파일로 남기려면 `--json-output`을 추가한다.

```bat
python Ai_verification\Automation\AutoVerify.py verify-vnnlib --model "%TEST_MODEL%" --vnnlib "%TEST_VNNLIB%" --timeout-seconds 300 --json-output "Ai_verification\Automation\Results\my_model_result.json"
```

은닉 ReLU가 50개를 넘는 모델은 다음과 같이 실행한다.

```bat
python Ai_verification\Automation\AutoVerify.py verify-vnnlib --model "%TEST_MODEL%" --vnnlib "%TEST_VNNLIB%" --allow-large-model --timeout-seconds 300
```

## 14. 정상 작동 테스트용 모델

저장소의 Smoke 예제는 `Y_0 = ReLU(X_0)`인 작은 ONNX 모델이다. JSON 설정 없이
ONNX와 VNNLIB만으로 프런트엔드부터 원본 솔버까지 확인할 수 있다.

### 14.1 ONNX 모델 구성

`create_relu_model.py`가 다음 그래프를 `relu_smoke.onnx`로 저장한다.

```text
X ── Gemm(fc1) ──> H ── ReLU ──> R ── Gemm(fc2) ──> Y
```

| 구성 | ONNX 이름 | 값 또는 계산 |
|---|---|---|
| 입력 | `X` | 실수 입력 1개, shape `[1, 1]` |
| 첫 가중치 | `W1` | `[[1.0]]` |
| 첫 편향 | `B1` | `[0.0]` |
| 첫 FC | `fc1` | `H = X × W1 + B1 = X` |
| 은닉 활성화 | `relu1` | `R = max(0, H)` |
| 둘째 가중치 | `W2` | `[[1.0]]` |
| 둘째 편향 | `B2` | `[0.0]` |
| 출력 FC | `fc2` | `Y = R × W2 + B2 = R` |

따라서 전체 모델 계산은 다음과 같다.

```text
H = X
R = max(0, H)
Y = R
결과: Y = ReLU(X)
```

ONNX에는 FC 레이어가 2개 있으므로 레이어 구조는 `1 -> 1 -> 1`이고, 두 FC
사이에 ReLU가 하나 있으므로 은닉 ReLU 개수는 1개다. 생성 스크립트는
`onnx.checker.check_model()`로 ONNX 형식을 검사한 뒤 같은 폴더에 저장한다.

### 14.2 VNNLIB 해석 방법

두 VNNLIB 파일은 공통으로 입력과 출력을 실수 변수로 선언한다.

```lisp
(declare-const X_0 Real)
(declare-const Y_0 Real)
```

- `X_0`은 ONNX 입력 `X`에 연결된다.
- `Y_0`은 ONNX 출력 `Y`에 연결된다.
- 여러 `assert`는 모두 AND 조건으로 결합된다.
- VNNLIB의 출력 assertion은 안전 조건이 아니라 찾으려는 unsafe 조건이다.

솔버는 다음 교집합을 검사한다.

```text
신경망 식 AND VNNLIB 입력 범위 AND VNNLIB unsafe 출력 조건
```

| 솔버 결과 | 사용자 결과 | 의미 |
|---|---|---|
| `SAT` | `COUNTEREXAMPLE` | unsafe 조건을 만족하는 실제 입력이 존재함 |
| `UNSAT` | `VERIFIED` | 입력 범위 안에서 unsafe 조건을 만족하는 입력이 없음 |
| `UNKNOWN` | `UNKNOWN` | 시간이나 내부 한도 때문에 결론을 확정하지 못함 |

### 14.3 `counterexample.vnnlib` 구성

```lisp
(assert (>= X_0 -1.0))
(assert (<= X_0 1.0))
(assert (>= Y_0 0.5))
```

이를 수식으로 쓰면 다음과 같다.

```text
-1 <= X_0 <= 1
Y_0 >= 0.5
```

모델이 `Y_0 = ReLU(X_0)`이므로 `X_0 = 0.5`를 넣으면 `Y_0 = 0.5`가 된다.
모든 assertion을 만족하는 입력이 존재하므로 솔버는 `SAT`, 자동화 결과는
`COUNTEREXAMPLE`이다. 실제 실행에서도 입력 `[0.5]`, 출력 `[0.5]`가 반환된다.

해당 파일의 조건을 만족하면 반례가 되는 구성이다.

### 14.4 `verified.vnnlib` 구성

```lisp
(assert (>= X_0 -1.0))
(assert (<= X_0 0.0))
(assert (>= Y_0 0.1))
```

이를 수식으로 쓰면 다음과 같다.

```text
-1 <= X_0 <= 0
Y_0 >= 0.1
```

이 입력 범위에서는 항상 `ReLU(X_0) = 0`이므로 `Y_0 >= 0.1`을 만족할 수 없다.
따라서 솔버는 `UNSAT`, 자동화 결과는 `VERIFIED`이다.

### 14.5 모델 생성 및 경로 지정

```bat
python Ai_verification\Automation\SmokeTest\create_relu_model.py
set "SMOKE_DIR=Ai_verification\Automation\SmokeTest"
set "SMOKE_MODEL=%SMOKE_DIR%\relu_smoke.onnx"
```

### 14.6 구조 검사와 dry-run

```bat
python Ai_verification\Automation\AutoVerify.py inspect "%SMOKE_MODEL%"
python Ai_verification\Automation\AutoVerify.py verify-vnnlib --model "%SMOKE_MODEL%" --vnnlib "%SMOKE_DIR%\counterexample.vnnlib" --dry-run
```

정상이라면 구조는 `1 -> 1 -> 1`, 은닉 ReLU는 1개이며 dry-run 결과는
`DRY_RUN`이다. dry-run에서는 실제 SAT/UNSAT 판정을 실행하지 않는다.

### 14.7 반례 존재와 부재 실행

```bat
python Ai_verification\Automation\AutoVerify.py verify-vnnlib --model "%SMOKE_MODEL%" --vnnlib "%SMOKE_DIR%\counterexample.vnnlib" --timeout-seconds 30
python Ai_verification\Automation\AutoVerify.py verify-vnnlib --model "%SMOKE_MODEL%" --vnnlib "%SMOKE_DIR%\verified.vnnlib" --timeout-seconds 30
```

첫 번째 실행은 `COUNTEREXAMPLE`과 `SAT`, 두 번째 실행은 `VERIFIED`와 `UNSAT`이
정상 결과다.