from pyexpat import model

from DPLL import Prop, TrueProp, FalseProp, VarProp, InequProp, ReLUProp, AndProp, OrProp, NotProp, ImplProp
from typing import Dict, Tuple, List
from DPLL_T import dpll_t

import re
from math import isfinite

from XOREncoding import NN, FreshGen


def make_precondition_linf_box(
    x_vars: Tuple[str, ...],
    c_vals: Tuple[float, ...],
    eps: float,
    *,
    clamp_01: bool = False,
) -> Prop:
    """
    Precondition:
      ∧_i (c_i - eps <= x_i <= c_i + eps)
    optional:
      ∧_i (0 <= x_i <= 1) if clamp_01=True
    """
    assert len(x_vars) == len(c_vals), "x_vars and c_vals must have same length"

    p: Prop = TrueProp()
    for xi, ci in zip(x_vars, c_vals):
        # lower: xi >= ci - eps
        p = AndProp(p, InequProp(coeffs=frozenset({(xi, 1.0)}), b=ci - eps))
        # upper: xi <= ci + eps
        p = AndProp(p, InequProp(coeffs=frozenset({(xi, -1.0)}), b=-(ci + eps)))

        if clamp_01:
            p = AndProp(p, InequProp(coeffs=frozenset({(xi, 1.0)}), b=0.0))
            p = AndProp(p, InequProp(coeffs=frozenset({(xi, -1.0)}), b=-1.0))

    return p

# ============================================================
# 4) Postcondition 생성: same label wrt logit threshold 0
#    (s_x >= 0) <-> (s_c >= 0)
# ============================================================

def make_postcondition_same_class_by_logit(
    s_x_var: str,
    s_c_var: str,
    *,
    threshold: float = 0.0,
) -> Prop:
    """
    Postcondition:
      (s_x >= threshold) <-> (s_c >= threshold)
    """
    sx_ge = InequProp(coeffs=frozenset({(s_x_var, 1.0)}), b=threshold)
    sc_ge = InequProp(coeffs=frozenset({(s_c_var, 1.0)}), b=threshold)
    # (s_x >= threshold) <-> (s_c >= threshold) 는 (s_x >= threshold => s_c >= threshold) AND (s_c >= threshold => s_x >= threshold) 로 표현 가능
    return AndProp(ImplProp(sx_ge, sc_ge), ImplProp(sc_ge, sx_ge))

# ============================================================
# 5) 전체 스펙: (precondition ^ NN(x,c)) => postcondition
# ============================================================

def build_spec(
    nn_prop: Prop,
    precondition: Prop,
    postcondition: Prop,
) -> Prop:
    return ImplProp(AndProp(precondition, nn_prop), postcondition)


# ===========================================================
# 출력 함수


# --- pretty printing utils ---

def _fmt(v, nd=6, sci_thresh=1e-4):
    try:
        fv = float(v)
        if not isfinite(fv):
            return str(v)

        # 진짜 -0.0만 0.0으로 정리 (값을 죽이지 않음)
        if fv == 0.0:
            fv = 0.0

        # 작은 값은 sci로 보여주기 (값 정보 보존)
        if 0.0 < abs(fv) < sci_thresh:
            return f"{fv:.{nd}e}"
        return f"{fv:.{nd}f}"
    except Exception:
        return str(v)

def filter_model(model: dict,
                 hide_prefixes=("ineq_slack", "relu_slack"),
                 keep_exact=("x0", "x1"),
                 keep_prefixes=("s_x", "s_c", "z_", "h_")):
    """Hide slacks, keep key signals."""
    out = {}
    for k, v in model.items():
        if any(k.startswith(p) for p in hide_prefixes):
            continue
        if k in keep_exact or any(k.startswith(p) for p in keep_prefixes):
            out[k] = v
    return out

def group_by_layer(filtered: dict):
    """
    Return grouped dict:
      inputs: x0,x1
      logits: s_x*, s_c*
      hidden: z_*, h_*
      other: anything else kept
    """
    groups = {"inputs": {}, "logits": {}, "hidden": {}, "other": {}}
    for k, v in filtered.items():
        if k in ("x0", "x1"):
            groups["inputs"][k] = v
        elif k.startswith("s_x") or k.startswith("s_c"):
            groups["logits"][k] = v
        elif k.startswith("z_") or k.startswith("h_"):
            groups["hidden"][k] = v
        else:
            groups["other"][k] = v
    return groups

def print_cex(center, sat, model, nd=6):
    tag = "SAT  (counterexample)" if sat else "UNSAT (robust)"
    print(f"\n=== center={center} === {tag}")
    if not sat:
        return

    fm = filter_model(model)
    g = group_by_layer(fm)

    # inputs
    if g["inputs"]:
        x0 = g["inputs"].get("x0", None)
        x1 = g["inputs"].get("x1", None)
        print(f"  x:  x0={_fmt(x0, nd)}  x1={_fmt(x1, nd)}")

    # logits (보통 각 center 접두사가 붙음)
    if g["logits"]:
        # 보기 좋게 key 정렬: s_c 먼저, s_x 다음
        for name in sorted(g["logits"].keys(), key=lambda k: (0 if k.startswith("s_c") else 1, k)):
            val = g["logits"][name]
            fv = float(val)
            # pretty + raw 동시 출력
            print(f"  {name}: {_fmt(val, nd)}   (raw={fv:+.18e})")
            #print(f"  {name}: {_fmt(g['logits'][name], nd)}")

        # 가능하면 sign flip 여부도 같이 표시 (key가 1개씩이라는 가정)
        s_c_keys = [k for k in g["logits"] if k.startswith("s_c")]
        s_x_keys = [k for k in g["logits"] if k.startswith("s_x")]
        if len(s_c_keys) == 1 and len(s_x_keys) == 1:
            sc = float(g["logits"][s_c_keys[0]])
            sx = float(g["logits"][s_x_keys[0]])
            same = (sc >= 0) == (sx >= 0)
            print(f"  same_class?(sc>=0 <-> sx>=0): {same}   "
              f"(sc>=0={sc>=0}, sx>=0={sx>=0})")

    # hidden은 너무 길 수 있으니 옵션: 상위 몇 개만
    if g["hidden"]:
        # z/h를 전부 보고 싶으면 아래 slice 제거
        keys = sorted(g["hidden"].keys())
        max_show = 12
        show_keys = keys[:max_show]
        print(f"  hidden (showing {len(show_keys)}/{len(keys)}):")
        for k in show_keys:
            print(f"    {k}: {_fmt(g['hidden'][k], nd)}")
        if len(keys) > max_show:
            print("    ...")

    if g["other"]:
        print("  other:")
        for k in sorted(g["other"].keys()):
            print(f"    {k}: {_fmt(g['other'][k], nd)}")

# ============================================================
# 6) 사용 예
# ============================================================
if __name__ == "__main__":
    
    
    # 신경망 제약 생성
    x_vars = ("x0", "x1")

    centers = [
        (0.0, 0.0),
        (0.0, 1.0),
        (1.0, 0.0),
        (1.0, 1.0),
    ]

    # 고정된  c가 있을 때 (이때 c는  (0,0), (0,1), (1,0), (1,1) 중 하나) 
    # epsilon 값만큼의 차이가있는  x (x0, x1) 를 c와 같은 분류를 해낼 수 있는가

    # 수정해야할 사항

    # make_precondition_linf_box 함수에서 eps=0.3으로 했을 때
    # c = (0,0)  또는 (1,1) 에 대해 검증할 때 
    # x의 class는 0이어야 하는데 

    # 실제 값은 음수( class : 0 )인 엄청 작은 값이 출력노드 값으로 나옴  
    # <= 표기 제한으로 0.00, 즉 양수( class :  1)로 고려 
    # <= 반례 판정


    # 현재 class 판단 방식 y >= 0 => class 1
    #                 y <= 0 => class 2

    # 생각해본 대안 :                    y > eps => class 1
    #                 y < -eps => class 0

    # 대안으로 변경시 |y| <= eps 에 있는 건 고려 안함
    # => 검증이 안된 미정 영역이 생김 => 100% 검증했다고 볼 수 없음

    for c in centers:
        # 예: XOR 중심 c=(1,1), eps=0.05, 입력변수 x0,x1
        pre = make_precondition_linf_box(("x0", "x1"), c, eps = 0.02, clamp_01=True)

        # center마다 fresh generator를 새로 만들어야 변수 충돌 없이 깔끔
        fg = FreshGen(prefix=f"c{int(c[0])}{int(c[1])}_")

        # 1) 만약 NN이 (x_vars, c, fresh) 받으면:
        NN_prop, s_x_sym, s_c_sym, aux = NN(x=x_vars, c=c, gen=fg)

        # NN이 만들어낸 최종 로짓 변수명이 예: "s_x", "s_c" 라고 가정
        post = make_postcondition_same_class_by_logit(s_x_sym, s_c_sym)

        spec = build_spec(NN_prop, pre, post)

        # 반례 찾기( pre ∧ NN ∧ ¬post )
        Neg_spec = AndProp(pre, AndProp(NN_prop, NotProp(post)))

        dpll_model, sat = dpll_t(Neg_spec, debug=False)
        print_cex(c, sat, dpll_model, nd=6)

    