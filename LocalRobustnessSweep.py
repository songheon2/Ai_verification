import time
import numpy as np
import matplotlib.pyplot as plt
import platform
from matplotlib.ticker import MultipleLocator
from tqdm import tqdm

# 자기가 만든 파일들에서 필요한 함수들 가져오기
from DPLL import AndProp, NotProp
from DPLL_T import dpll_t
from XOREncoding import FreshGen, NN_single
from Robustness import make_precondition_linf_box
from PreciseEncoding import out_zero_logit, out_one_logit
from PreciseEncoding import xor_nn_sx  # 실제 신경망 함수 추가

# 폰트 설정
if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
elif platform.system() == 'Darwin':
    plt.rcParams['font.family'] = 'AppleGothic'
else:
    plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False

def run_milestone_sweep():
    eps_values = np.arange(0.0, 0.505, 0.005)
    sat_counts = []
    
    # 🌟 추가된 로직: SAT 개수가 점프하는 '구간(Milestone)'의 모든 반례를 저장할 딕셔너리
    # 예: {2: {'eps': 0.230, 'cexs': [반례1, 반례2]}, 4: {'eps': 0.280, 'cexs': [반례1..4]}}
    milestones = {}
    prev_sat_count = 0  # 이전 스텝의 SAT 개수를 기억

    xor_cases = [
        ((0.0, 0.0), 0), ((0.0, 1.0), 1),
        ((1.0, 0.0), 1), ((1.0, 1.0), 0)
    ]

    print("=== 구간별 다중 반례 수집 Sweep 시작 ===")
    
    for eps in tqdm(eps_values, desc="검증 진행 중"):
        current_sat = 0
        current_cexs = []  # 현재 eps에서 발견된 모든 반례를 임시 저장
        
        for (c1, c2), expected in xor_cases:
            fg = FreshGen(prefix=f"adv_eps{int(eps*1000)}_")
            x_vars = ("x0", "x1")
            
            pre = make_precondition_linf_box(x_vars, (c1, c2), eps=eps, clamp_01=False)
            nn_prop, s_var, _ = NN_single(x_vars, gen=fg)
            
            post = out_zero_logit(s_var) if expected == 0 else out_one_logit(s_var)
            phi = AndProp(pre, AndProp(nn_prop, NotProp(post)))
            
            model, sat = dpll_t(phi, debug=False)
            
            if sat:
                current_sat += 1
                # 발견된 반례를 리스트에 추가
                current_cexs.append({
                    'case': (c1, c2),
                    'expected': expected,
                    'x0': model.get('x0'),
                    'x1': model.get('x1')
                })
        
        sat_counts.append(current_sat)
        
        # 🌟 구간 점프 감지: 이전보다 SAT 개수가 늘어났다면? 그 지점을 마일스톤으로 기록!
        if current_sat > prev_sat_count:
            milestones[current_sat] = {
                'eps': eps,
                'cexs': current_cexs
            }
            prev_sat_count = current_sat  # 최고 기록 갱신

    # 그래프 그리기
    plt.figure(figsize=(12, 7))
    plt.plot(eps_values, sat_counts, marker='o', color='royalblue', 
             drawstyle='steps-post', label='반례 발생 케이스 수 (SAT)')
    
    ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(0.05))
    ax.xaxis.set_minor_locator(MultipleLocator(0.01))
    
    plt.grid(True, which='both', linestyle='--', alpha=0.4)
    plt.axvline(x=0.5, color='orange', linestyle='--', label=r'이론적 경계 ($\epsilon=0.5$)')
    
    plt.title('Epsilon 변화에 따른 XOR 신경망 강건성 분석 (구간별 반례 추적)')
    plt.xlabel(r'섭동 범위 ($\epsilon$)')
    plt.ylabel('SAT 발생 수 (4개 케이스 중)')
    plt.ylim(-0.2, 4.2)
    plt.legend()
    plt.show(block=False)
    plt.pause(1)

    return milestones

if __name__ == "__main__":
    milestones = run_milestone_sweep()

    # 2. 수집된 모든 마일스톤(계단 구간)의 반례들을 다 꺼내서 검증
    if milestones:
        print("\n\n" + "="*70)
        print("🔍 [자동 검산] 각 고장 구간(Jump)별 발견된 모든 반례 정밀 검증")
        print("="*70)
        
        # 2개 점프 구간, 4개 점프 구간 순서대로 출력
        for sat_count in sorted(milestones.keys()):
            data = milestones[sat_count]
            eps = data['eps']
            cexs = data['cexs']
            
            print(f"\n🚀 [새로운 고장 구간 진입!] 총 {sat_count}개 케이스 고장 (epsilon = {eps:.3f})")
            print("-" * 70)
            
            for i, cex in enumerate(cexs, start=1):
                c1, c2 = cex['case']
                expected = cex['expected']
                x0_cex = cex['x0']
                x1_cex = cex['x1']
                
                print(f"  [{i}/{len(cexs)}] 분석 중인 케이스: ({c1}, {c2}) -> 기대값 {expected}")
                print(f"     👉 솔버가 찾은 입력 : x0 = {x0_cex:.8f}, x1 = {x1_cex:.8f}")
                
                # 신경망 로짓 계산
                logit_val = xor_nn_sx(x0_cex, x1_cex)
                prediction = 1 if logit_val >= 0 else 0
                
                if prediction != expected:
                    print(f"     ✅ 기본 검증 완료 : Logit = {logit_val:.8f} (오답 발생 확인!)")
                else:
                    # 파이썬 수치 오차로 정답이 나와버리는 경우 -> 0.001 바깥으로 밀어내기
                    print(f"     ❓ 파이썬 수치 오차 감지 (Logit = {logit_val:.8f}). 미세 밀어내기 적용 중...")
                    
                    shift = 0.001
                    x0_pushed = x0_cex + (shift if x0_cex > c1 else -shift)
                    x1_pushed = x1_cex + (shift if x1_cex > c2 else -shift)
                    
                    logit_pushed = xor_nn_sx(x0_pushed, x1_pushed)
                    pred_pushed = 1 if logit_pushed >= 0 else 0
                    
                    if pred_pushed != expected:
                        print(f"     ✅ 밀어내기 성공  : x0={x0_pushed:.5f}, x1={x1_pushed:.5f}")
                        print(f"                       Logit = {logit_pushed:.8f} (오답 확인!)")
                    else:
                        print(f"     ❌ 밀어내기 실패  : 조금 더 강한 밀어내기가 필요할 수 있습니다.")
                print("") # 줄바꿈
    else:
        print("\n✅ 반례가 발견되지 않았습니다.")
    
    plt.show()