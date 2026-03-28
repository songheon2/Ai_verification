import numpy as np
import matplotlib.pyplot as plt
import platform
from matplotlib.ticker import MultipleLocator
from tqdm import tqdm

# 기존 임포트 유지
from DPLL import AndProp, NotProp
from DPLL_T import dpll_t
from XOREncoding import FreshGen, NN_single
from Robustness import make_precondition_linf_box
from PreciseEncoding import out_zero_logit, out_one_logit

# 폰트 설정을 더 확실하게 하는 방법
if platform.system() == 'Windows':
    # 윈도우라면 맑은 고딕을 명시적으로 지정
    plt.rcParams['font.family'] = 'Malgun Gothic'
elif platform.system() == 'Darwin':
    # 맥(OSX)이라면 AppleGothic 지정
    plt.rcParams['font.family'] = 'AppleGothic'
else:
    # 리눅스 등 기타 환경
    plt.rcParams['font.family'] = 'NanumGothic'

# 마이너스 기호(-)가 깨지는 현상 방지
plt.rcParams['axes.unicode_minus'] = False

def run_advanced_sweep():
    # 1. 설정: 0.01 단위 보조 눈금을 위해 step은 0.005 유지
    eps_values = np.arange(0.0, 0.505, 0.005) 
    sat_counts = []
    
    # 첫 번째 반례를 저장할 변수
    first_cex = None

    xor_cases = [
        ((0.0, 0.0), 0), ((0.0, 1.0), 1),
        ((1.0, 0.0), 1), ((1.0, 1.0), 0)
    ]

    print("=== 정밀 Epsilon Sweep 및 반례 분석 시작 ===")
    
    for eps in tqdm(eps_values, desc="검증 진행 중"):
        current_sat = 0
        for (c1, c2), expected in xor_cases:
            fg = FreshGen(prefix=f"adv_eps{int(eps*1000)}_")
            x_vars = ("x0", "x1")
            
            # 입력 영역 설정 (clamp_01=False로 0/1 외부 탐색)
            pre = make_precondition_linf_box(x_vars, (c1, c2), eps=eps, clamp_01=False)
            nn_prop, s_var, _ = NN_single(x_vars, gen=fg)
            
            # 반례 조건 설정
            post = out_zero_logit(s_var) if expected == 0 else out_one_logit(s_var)
            phi = AndProp(pre, AndProp(nn_prop, NotProp(post)))
            
            # DPLL(T)로 반례 탐색
            model, sat = dpll_t(phi, debug=False)
            
            if sat:
                current_sat += 1
                # 최초의 반례 정보 저장
                if first_cex is None:
                    first_cex = {
                        'eps': eps,
                        'case': (c1, c2),
                        'expected': expected,
                        'x0': model.get('x0'),
                        'x1': model.get('x1')
                    }
        sat_counts.append(current_sat)

    # 2. 결과 출력
    if first_cex:
        print(f"\n[최초 반례 발견!] epsilon = {first_cex['eps']:.3f}")
        print(f"  - 케이스: {first_cex['case']} (기대값: {first_cex['expected']})")
        print(f"  - 발견된 입력: x0 = {first_cex['x0']:.6f}, x1 = {first_cex['x1']:.6f}")
    
    # 3. 시각화 개선
    plt.figure(figsize=(12, 7))
    # 계단식(steps-post)으로 그려서 정수 단위의 SAT 변화를 명확히 표현
    plt.plot(eps_values, sat_counts, marker='o', color='royalblue', 
             drawstyle='steps-post', label='반례 발생 케이스 수 (SAT)')
    
    ax = plt.gca()
    # x축 주 눈금 (0.05 단위로 숫자 표시)
    ax.xaxis.set_major_locator(MultipleLocator(0.05))
    # x축 보조 눈금 (0.01 단위로 선만 표시)
    ax.xaxis.set_minor_locator(MultipleLocator(0.01))
    
    # 그리드: 보조 눈금까지 표시하여 0.01 단위를 읽기 쉽게 함
    plt.grid(True, which='both', linestyle='--', alpha=0.4)
    plt.axvline(x=0.5, color='orange', linestyle='--', label=r'이론적 경계 ($\epsilon=0.5$)')
    
    plt.title('Epsilon 변화에 따른 XOR 신경망 강건성 분석')
    plt.xlabel(r'섭동 범위 ($\epsilon$)')
    plt.ylabel('SAT 발생 수 (4개 케이스 중)')
    plt.ylim(-0.2, 4.2)
    plt.legend()
    plt.show()

# 반례 검증용 코드
from PreciseEncoding import xor_nn_sx



if __name__ == "__main__":
    run_advanced_sweep()

    # 2. 방금 찾은 반례 직접 검증하기 (추가할 부분)
    print("\n" + "="*50)
    print("[검산] 발견된 반례 좌표를 실제 신경망에 대입")
    print("="*50)
    
    # 아까 솔버가 찾아준 좌표
    x0_cex, x1_cex = 0.230000, 1.223
    
    # PreciseEncoding.py에 있는 함수로 실제 로짓(Logit) 계산
    logit_val = xor_nn_sx(x0_cex, x1_cex)
    
    print(f"입력: x0={x0_cex}, x1={x1_cex}")
    print(f"계산된 Logit(s): {logit_val:.10f}")
    
    # Logit이 0보다 작으면 결과가 0(False), 0보다 크면 1(True)이야
    prediction = 1 if logit_val >= 0 else 0
    print(f"신경망 판단 결과: {prediction} (기대값: 1)")
    
    if prediction != 1:
        print("✅ 검증 완료: 이 지점은 실제로 신경망이 오답을 내는 '반례'가 맞네!")
    else:
        print("❓ 오잉? 결과가 맞게 나오네? 수치 정밀도를 다시 확인해봐야겠어.")