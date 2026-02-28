from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, List
from DPLL import Prop, TrueProp, FalseProp, VarProp, InequProp, ReLUProp, AndProp, OrProp, NotProp, ImplProp
from DPLL import parse_prop, tseitin_cnf, dpll, neg


# ============================================================
# 유틸: And를 n-ary처럼 쓰기 위한 헬퍼
# ============================================================
def conj(props: List[Prop]) -> Prop:
    if not props:
        return TrueProp()
    out = props[0]
    for p in props[1:]:
        out = AndProp(out, p)
    return out

def disj(props: List[Prop]) -> Prop:
    if not props:
        return FalseProp()
    out = props[0]
    for p in props[1:]:
        out = OrProp(out, p)
    return out

# ============================================================
# 유틸: 선형 부등식 만들기
# InequProp은 "sum(ci*xi) >= b" 만 표현 가능
#  - equality (lhs = rhs) 는 (lhs-rhs >= 0) AND (rhs-lhs >= 0) 로 인코딩
# ============================================================
def ge_lin(terms: Dict[str, float], b: float) -> Prop:
    # sum(terms[var]*var) >= b
    return InequProp(coeffs=frozenset(terms.items()), b=float(b))

def eq_lin(terms: Dict[str, float], b: float) -> Prop:
    # sum(terms[var]*var) == b
    # <=> sum(...) >= b  AND  -sum(...) >= -b
    p1 = ge_lin(terms, b)
    neg_terms = {v: -c for v, c in terms.items()}
    p2 = ge_lin(neg_terms, -b)
    return AndProp(p1, p2)

# ============================================================
# fresh variable name generator
# ============================================================
class FreshGen:
    def __init__(self, prefix: str = "t"):
        self.prefix = prefix
        self.k = 0

    def fresh(self, base: str) -> str:
        self.k += 1
        return f"{base}_{self.prefix}{self.k}"

# ============================================================
# XOR 네트워크가 완전연결이고 입력 차원도 고정(2개)라고 가정 
# NN(x,c): ReLU + (sigmoid 전) logit s 까지 제약을 Prop로 반환
#  - x = (x0_name, x1_name), c = (c0_name, c1_name)
#  - returns: (phi_nn, sx_name, sc_name, aux_vars)
# ============================================================
def NN(x: Tuple[str, str], c: Tuple[str, str], gen: FreshGen | None = None):
    """
    XOR NN with given trained weights:
      hidden1: w=[ 2.1247,  2.1267], b=-2.1259
      hidden2: w=[-2.1237, -2.1235], b= 2.1234
      out    : w=[-3.6788, -3.6766], b= 3.5451
    Activation: ReLU on hidden, (optional sigmoid after) output.
    We encode up to logit s; threshold 0.5 corresponds to s>0.

    Parameters
    ----------
    x : (x0, x1) variable names
    c : (c0, c1) variable names (can be constants you already substituted elsewhere)
    gen : fresh name generator

    Returns
    -------
    phi : Prop
        conjunction of all constraints for both paths (x-path and c-path)
    sx : str
        variable name of logit for x input
    sc : str
        variable name of logit for c input
    aux : Dict[str, List[str]]
        all intermediate variable names created (debug/inspection)
    """
    if gen is None:
        gen = FreshGen("nn")

    x0, x1 = x
    c0, c1 = c

    # ----- allocate intermediates (x path) -----
    zx1 = gen.fresh("z_x1")
    zx2 = gen.fresh("z_x2")
    hx1 = gen.fresh("h_x1")
    hx2 = gen.fresh("h_x2")
    sx  = gen.fresh("s_x")   # logit

    # affine hidden pre-activations for x
    # zx1 = 2.1247*x0 + 2.1267*x1 - 2.1259
    phi_zx1 = eq_lin({zx1: 1.0, x0: -2.1247, x1: -2.1267}, -2.1259)
    # zx2 = -2.1237*x0 - 2.1235*x1 + 2.1234
    phi_zx2 = eq_lin({zx2: 1.0, x0: 2.1237, x1: 2.1235}, 2.1234)

    # ReLU constraints
    phi_relu_x1 = ReLUProp(x=zx1, y=hx1)
    phi_relu_x2 = ReLUProp(x=zx2, y=hx2)

    # output logit: sx = -3.6788*hx1 - 3.6766*hx2 + 3.5451
    phi_sx = eq_lin({sx: 1.0, hx1: 3.6788, hx2: 3.6766}, 3.5451)

    phi_x = conj([phi_zx1, phi_zx2, phi_relu_x1, phi_relu_x2, phi_sx])

    # ----- allocate intermediates (c path) -----
    zc1 = gen.fresh("z_c1")
    zc2 = gen.fresh("z_c2")
    hc1 = gen.fresh("h_c1")
    hc2 = gen.fresh("h_c2")
    sc  = gen.fresh("s_c")   # logit

    # hidden pre-activations are constants
    zc1_val = 2.1247*c0 + 2.1267*c1 - 2.1259   # (w·c + b)
    zc2_val = -2.1237*c0 + -2.1235*c1 + 2.1234

    phi_zc1 = eq_lin({zc1: 1.0}, zc1_val)
    phi_zc2 = eq_lin({zc2: 1.0}, zc2_val)

    phi_relu_c1 = ReLUProp(x=zc1, y=hc1)
    phi_relu_c2 = ReLUProp(x=zc2, y=hc2)

    # sc is NOT a constant because hc1/hc2 depend on ReLU outputs (variables)
    phi_sc = eq_lin({sc: 1.0, hc1: 3.6788, hc2: 3.6766}, 3.5451)

    phi_c = conj([phi_zc1, phi_zc2, phi_relu_c1, phi_relu_c2, phi_sc])
    phi_nn = AndProp(phi_x, phi_c)

    # output logit for c
    phi_sc = eq_lin({sc: 1.0, hc1: 3.6788, hc2: 3.6766}, 3.5451)

    phi_c = conj([phi_zc1, phi_zc2, phi_relu_c1, phi_relu_c2, phi_sc])

    phi_nn = AndProp(phi_x, phi_c)

    aux = {
        "x_path": [zx1, zx2, hx1, hx2, sx],
        "c_path": [zc1, zc2, hc1, hc2, sc],
    }
    return phi_nn, sx, sc, aux

# ============================================================
# 테스트
# ============================================================


def main():
    # 입력 변수 이름 (네 NN 인코딩에서 쓰는 이름과 일치해야 함)
    x_vars = ("x0", "x1")

    centers = [
        (0.0, 0.0),
        (0.0, 1.0),
        (1.0, 0.0),
        (1.0, 1.0),
    ]

    for c in centers:
        print("=" * 80)
        print(f"NN constraints only | center c = {c}")

        # center마다 fresh generator를 새로 만들어야 변수 충돌 없이 깔끔
        fg = FreshGen(prefix=f"c{int(c[0])}{int(c[1])}_")

        # --- 여기만 네 NN 시그니처에 맞게 고치면 됨 ---
        # 1) 만약 NN이 (x_vars, c, fresh) 받으면:
        nn_prop = NN(x=x_vars, c=c, gen=fg)

        # 2) 만약 NN이 (x, c)만 받으면:
        # nn_prop = NN(x_vars, c)
        # ------------------------------------------------

        print(nn_prop)

        # fresh 카운터 같은 게 있으면 함께 확인
        if hasattr(fg, "counter"):
            print("Fresh counter:", fg.counter)

    print("=" * 80)
    print("Done.")

if __name__ == "__main__":
    main()