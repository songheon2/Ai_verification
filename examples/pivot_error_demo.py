"""피벗할 때마다 부동소수 오차가 생기고 쌓이는 것을 보여주는 예제.

피벗은 "같은 연립방정식을 다른 변수 기준으로 다시 정리"하는 것뿐이라, 수학적으로는
몇 번을 하든 원래 방정식과 똑같다. 그런데 컴퓨터는 나눗셈·덧셈마다 반올림하므로
정리된 식이 원래 방정식에서 조금씩 벗어난다.

1부: 작은 수로 한 번 나눴다가 되돌아오기만 해도 식이 틀어진다
2부: 피벗을 반복하면 오차가 쌓인다 (피벗할 계수를 어떻게 고르느냐에 따라 속도가 다르다)

실행: python examples/pivot_error_demo.py
"""
import random


# ─────────────────────────── 1부 ───────────────────────────
def part1():
    """원래 식:  s1 = eps*x + 0.7*y,   s2 = 0.3*x + 0.9*y

    피벗 1: 첫 식을 x 기준으로 정리 (eps로 나눈다)  →  x = s1/eps - (0.7/eps)*y
            둘째 식의 x에 대입                      →  s2 = (0.3/eps)*s1 + (0.9 - 0.21/eps)*y
    피벗 2: 다시 s1 기준으로 되돌린다               →  s2 = 0.3*x + 0.9*y  가 나와야 한다

    eps가 작으면 중간에 0.21/eps 같은 큰 수와 0.9를 더하게 되고, 큰 수에 묻혀
    0.9의 아랫자리가 반올림으로 사라진다. 되돌아와도 그 자리는 복구되지 않는다.
    """
    p, q, r = 0.7, 0.3, 0.9
    print("1부) 작은 수 eps로 나눴다가 되돌아오기 (y의 계수는 0.9여야 한다)")
    print(f"{'나눈 수 eps':>12} | {'되돌아온 y의 계수':>22} | {'오차':>8}")
    print("-" * 50)
    for eps in (0.3, 3e-4, 3e-8, 3e-12, 3e-16):
        a, b = 1 / eps, -p / eps        # 피벗 1: x  = a*s1 + b*y
        c, d = q * a, r + q * b         #          s2 = c*s1 + d*y
        e, f = 1 / a, -b / a            # 피벗 2: s1 = e*x + f*y
        h = c * f + d                   #          s2 = (...)*x + h*y
        print(f"{eps:>12.0e} | {h:>22.17g} | {abs(h - r):>8.1e}")
    print()


# ─────────────────────────── 2부 ───────────────────────────
M, N = 8, 8                    # 방정식 8개 (s0..s7), 변수 x0..x7
STEPS = 5000
REPORT = [1, 10, 100, 1000, 5000]


def make_problem(seed):
    # 신경망 가중치처럼 계수 크기가 여러 자릿수(1e-4 ~ 1)에 걸쳐 있게 만든다
    rng = random.Random(seed)
    return [[rng.uniform(-1, 1) * 10 ** rng.uniform(-4, 0) for _ in range(N)] for _ in range(M)]


def pivot(T, basic, nonbasic, r, c):
    """r번째 식의 왼쪽 변수와 c번째 오른쪽 변수를 맞바꿔 다시 정리한다."""
    a = T[r][c]
    row = [-v / a for v in T[r]]
    row[c] = 1.0 / a
    T[r] = row
    for i in range(M):                  # 나머지 식에 새 식을 대입
        if i != r:
            f = T[i][c]
            T[i] = [T[i][j] + f * row[j] for j in range(N)]
            T[i][c] = f * row[c]
    basic[r], nonbasic[c] = nonbasic[c], basic[r]


def error(T, basic, nonbasic, A, rng):
    """지금 식에 아무 값이나 넣어 계산한 결과가 원래 방정식을 얼마나 어기나 (상대 오차).

    식이 정확하면 어떤 값을 넣어도 원래 방정식이 딱 맞아야 한다 — 솔버의 누적 오차
    검사도 이 방식이다.
    """
    val = {v: rng.uniform(-1, 1) for v in nonbasic}
    for i in range(M):
        val[basic[i]] = sum(T[i][j] * val[nonbasic[j]] for j in range(N))
    worst = 0.0
    for i in range(M):                  # 원래 방정식: s_i = sum_j A[i][j] * x_j
        terms = [A[i][j] * val[f"x{j}"] for j in range(N)]
        worst = max(worst, abs(val[f"s{i}"] - sum(terms)) / max(1.0, sum(abs(t) for t in terms)))
    return worst


def run(choose_column, seed=0):
    A = make_problem(seed)
    T = [row[:] for row in A]
    basic = [f"s{i}" for i in range(M)]
    nonbasic = [f"x{j}" for j in range(N)]
    rng = random.Random(seed + 1)
    errs = {}
    for step in range(1, STEPS + 1):
        r = rng.randrange(M)
        pivot(T, basic, nonbasic, r, choose_column(T[r], rng))
        if step in REPORT:
            errs[step] = error(T, basic, nonbasic, A, rng)
    return errs


def largest(row, rng):          # 지금 솔버: 그 식에서 가장 큰 계수로 나눈다
    return max(range(N), key=lambda j: abs(row[j]))


def anything(row, rng):         # 계수 크기를 안 본다 (예전 Bland 규칙이 이렇다)
    return rng.choice([j for j in range(N) if abs(row[j]) > 1e-12])


def smallest(row, rng):         # 최악: 가장 작은 계수로 나눈다
    return min((j for j in range(N) if abs(row[j]) > 1e-12), key=lambda j: abs(row[j]))


def part2():
    print("2부) 피벗을 반복할 때 원래 방정식 대비 오차 (8개 식, 무작위 피벗)")
    results = [run(largest), run(anything), run(smallest)]
    print(f"{'피벗 횟수':>8} | {'큰 계수로':>10} | {'아무 계수로':>10} | {'작은 계수로':>10}")
    print("-" * 52)
    for step in REPORT:
        print(f"{step:>8} | " + " | ".join(f"{r[step]:>10.1e}" for r in results))
    print()


if __name__ == "__main__":
    part1()
    part2()
    print("참고: 1e-16 = 완벽, 1e-6 = 솔버가 원래 식으로 다시 세우는 기준")
