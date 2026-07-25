"""AutoVerify.py를 재귀 한도를 올린 채로 실행하는 래퍼.

은닉 ReLU가 많은 모델(ACAS Xu급, 300개 등)은 dpll()의 재귀 호출이 Python
기본 재귀 한도(1000)를 넘겨서 RecursionError가 날 수 있다. 이 스크립트는
sys.setrecursionlimit(100000)을 먼저 건 다음 AutoVerify.py의 main()을 그대로
실행한다.

python -c "import sys, runpy; sys.setrecursionlimit(100000); ..." 같은 인라인
방식은 따옴표 처리가 셸(bash/cmd/PowerShell)마다 달라서 특히 PowerShell에서
깨지기 쉽다(중첩된 큰따옴표가 소실되어 NameError가 나는 식) — 그래서 인자를
평범하게 넘길 수 있는 이 파일을 대신 쓴다.

사용법
------
    python run_large_model.py inspect model.onnx
    python run_large_model.py verify-vnnlib --model model.onnx --vnnlib prop.vnnlib.gz \
        --allow-large-model --timeout-seconds 1800 --simplex-max-iter 1000000

AutoVerify.py와 완전히 동일한 서브커맨드/인자를 그대로 받는다
(inspect / verify / verify-vnnlib 전부 지원) — 앞에 이 파일 이름만 다를 뿐이다.
"""

import sys

sys.setrecursionlimit(100000)

if __name__ == "__main__":
    from AutoVerify import main

    raise SystemExit(main())
