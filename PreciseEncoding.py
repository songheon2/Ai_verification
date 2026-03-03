# y = xor(x1, x2)

# zero(i) : (0 <= i) ^ (i < 0.5)
# one(i) : (0.5 <= i) ^ (i <= 1)

# 증명할 4개의 명제를 or 시킨 명제 1개
# case 00 or case 01 or case 10 or case 11

# case 00: x1=0, x2=0 -> y=0

# PRE: zero(x1) and zero(x2) 
#        s = xor(x1, x2)의 출력층 활성화 함수를 넣기전 logit
# POST: s < 0

# 반례 찾기 형식으로 (PRE and not POST)

# PRE: zero(x1) and zero(x2)
#        s = xor(x1, x2)의 출력층 활성화 함수를 넣기전 logit
# POST:(s <= 0)   

from DPLL import Prop, TrueProp, FalseProp, VarProp, InequProp, ReLUProp, AndProp, OrProp, NotProp, ImplProp, tseitin_cnf
from DPLL_T import dpll_t 
from XOREncoding import NN_single, NN_dual, FreshGen, eq_lin

# small helper functions for building specs
def AND(*ps: Prop) -> Prop:
    """n-ary And fold. AND() == True."""
    cur: Prop = TrueProp()
    for p in ps:
        cur = AndProp(cur, p)
    return cur

def zero(i: str, eps_strict: float = 0.0) -> Prop:
    # zero(i) : (0 <= i) ^ (i <= 0.5)
    return AndProp(
        InequProp(frozenset([(i, 1.0)]), 0 + eps_strict),  # 0 <= i + eps_strict
        InequProp(frozenset([(i, -1.0)]), -(0.5 - eps_strict))  # i <= 0.5 - eps_strict
    )

def one(i: str, eps_strict: float = 0.0) -> Prop:
    # one(i) : (0.5 <= i) ^ (i <= 1)
    return AndProp(
        InequProp(frozenset([(i, 1.0)]), 0.5 + eps_strict),  # 0.5 <= i + eps_strict
        InequProp(frozenset([(i, -1.0)]), -(1.0 - eps_strict))  # i <= 1 - eps_strict
    )

def out_zero_logit(s: str, eps_strict: float = 0.0) -> Prop:
    # "0" 클래스: s < 0  (non-strict margin)  s <= -eps  <=>  -s >= eps
    return InequProp(frozenset([(s, -1.0)]), 0.0 + eps_strict)

def out_one_logit(s: str, eps_strict: float = 0.0) -> Prop:
    # "1" 클래스: s > 0  (non-strict margin)  s >= eps
    return InequProp(frozenset([(s, 1.0)]), 0.0 + eps_strict)

# x에 0.5가 들어오면 1.0으로 snap하는 제약 (반례 공식에서 사용)
def eq_var_const(x_var: str, c: float):
    """
    x_var == c 를 eq_lin으로 표현
    eq_lin은 (terms_dict, b)를 받는다고 가정:
      eq_lin({x_var: 1.0}, c)
    """
    return eq_lin({x_var: 1.0}, float(c))

def snap_half_to_one_prop(x_var):
    """(x == 0.5) => (x == 1.0)
       ( x == 0.5) => ( One(x) true )"""
    return ImplProp(eq_var_const(x_var, 0.5), eq_var_const(x_var, 1.0))

# 범용 case 생성 함수: cls_x1, cls_x2, cls_y는 zero/one 같은 클래스 제약 생성 함수
def cex_case(x1: str, x2: str,
             cls_x1, cls_x2, cls_out_logit,
             eps_strict: float = 0.0,
             gen: FreshGen | None = None) -> Prop:
    """
    Returns counterexample formula: PRE ∧ ¬POST
      PRE  = cls_x1(x1) ∧ cls_x2(x2) ∧ NN(x1,x2) ^ ( x =0.5 => x = 1)->y constraints
      POST = cls_out_logit(s) where s is the logit output of the NN
    """
    NNprop, s, _ = NN_single((x1, x2),gen=gen)

    pre = AND(
        cls_x1(x1, eps_strict=eps_strict),
        cls_x2(x2, eps_strict=eps_strict),
        NNprop,
    )
    post = cls_out_logit(s, eps_strict=eps_strict)
    return AND(pre, NotProp(post))

def cex_xor_all_cases(x1="x1", x2="x2", eps_strict: float = 0.0) -> Prop:
    # 각각 "이 케이스에서 output이 기대 클래스가 아니면" 반례

    # case마다 fresh generator를 새로 만들어야 변수 충돌 없이 깔끔
    fg = FreshGen(prefix=f"x00_")
    cex00 = cex_case(x1, x2, zero, zero, out_zero_logit, eps_strict=eps_strict, gen=fg)  # 0 xor 0 = 0
    fg = FreshGen(prefix=f"x01_")
    cex01 = cex_case(x1, x2, zero, one,  out_one_logit,  eps_strict=eps_strict, gen=fg)  # 0 xor 1 = 1
    fg = FreshGen(prefix=f"x10_")
    cex10 = cex_case(x1, x2, one,  zero, out_one_logit,  eps_strict=eps_strict, gen=fg)  # 1 xor 0 = 1
    fg = FreshGen(prefix=f"x11_")
    cex11 = cex_case(x1, x2, one,  one,  out_zero_logit, eps_strict=eps_strict, gen=fg)  # 1 xor 1 = 0

    # "어느 케이스든 실패하면" 반례이므로 OR로 묶는 게 맞음
    return OrProp(OrProp(cex00, cex01), OrProp(cex10, cex11))

# --- pretty printing utils ---

def _fmt(v, nd=6, sci_thresh=1e-4):
    try:
        fv = float(v)
        if not isfinite(fv):
            return str(v)

        # 진짜 -0.0만 0.0으로 정리
        if fv == 0.0:
            fv = 0.0

        if 0.0 < abs(fv) < sci_thresh:
            return f"{fv:.{nd}e}"
        return f"{fv:.{nd}f}"
    except Exception:
        return str(v)

def filter_model(model: dict,
                 hide_prefixes=("ineq_slack", "relu_slack", "not_"),
                 keep_exact=("x1", "x2"),
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
    groups:
      inputs: x1,x2
      logits: s_x*, s_c*
      hidden: z_*, h_*
      other: anything else kept
    """
    groups = {"inputs": {}, "logits": {}, "hidden": {}, "other": {}}
    for k, v in filtered.items():
        if k in ("x1", "x2"):
            groups["inputs"][k] = v
        elif k.startswith("s_x") or k.startswith("s_c"):
            groups["logits"][k] = v
        elif k.startswith("z_") or k.startswith("h_"):
            groups["hidden"][k] = v
        else:
            groups["other"][k] = v
    return groups

def print_cex(sat, model, nd=6, max_hidden=24):
    tag = "SAT  (counterexample)" if sat else "UNSAT (robust)"
    print(f"\n====== {tag}===========")
    if not sat:
        return

    fm = filter_model(model)
    g = group_by_layer(fm)

    # inputs
    if g["inputs"]:
        x1 = g["inputs"].get("x1", None)
        x2 = g["inputs"].get("x2", None)
        print(f"  x:  x1={_fmt(x1, nd)}  x2={_fmt(x2, nd)}")

    # logits
    if g["logits"]:
        for name in sorted(g["logits"].keys(), key=lambda k: (0 if k.startswith("s_c") else 1, k)):
            val = g["logits"][name]
            fv = float(val)
            print(f"  {name}: {_fmt(val, nd)}   (raw={fv:+.18e})")

        # 케이스가 여러개면 sign 요약을 케이스별로 찍기
        # 예: s_c00_5, s_c01_5 ... (s_x도 있으면 같이)
        s_c_keys = sorted([k for k in g["logits"] if k.startswith("s_c")])
        s_x_keys = sorted([k for k in g["logits"] if k.startswith("s_x")])

        if s_c_keys and s_x_keys:
            print("  case-wise same_class?  (sc>=0 <-> sx>=0):")
            # 접미사(케이스 id)를 최대한 맞춰보기: s_cXX_* vs s_xXX_*
            def case_id(k):  # "s_c00_5" -> "00", "s_x_c11_5" -> "c11" 등도 대비
                # 매우 보수적으로: s_c 다음부터 '_' 전까지
                rest = k.split("s_c", 1)[1]
                return rest.split("_", 1)[0]
            sx_by_case = {}
            for k in s_x_keys:
                rest = k.split("s_x", 1)[1]
                cid = rest.split("_", 1)[0]
                sx_by_case[cid] = k

            for kc in s_c_keys:
                cid = case_id(kc)
                if cid in sx_by_case:
                    sc = float(g["logits"][kc])
                    sx = float(g["logits"][sx_by_case[cid]])
                    same = (sc >= 0) == (sx >= 0)
                    print(f"    case {cid}: {same}   (sc={sc:+.3e}, sx={sx:+.3e})")

    # hidden
    if g["hidden"]:
        keys = sorted(g["hidden"].keys())
        show_keys = keys[:max_hidden]
        print(f"  hidden (showing {len(show_keys)}/{len(keys)}):")
        for k in show_keys:
            print(f"    {k}: {_fmt(g['hidden'][k], nd)}")
        if len(keys) > max_hidden:
            print("    ...")

    if g["other"]:
        print("  other:")
        for k in sorted(g["other"].keys()):
            print(f"    {k}: {_fmt(g['other'][k], nd)}")

# ============================================================
# 사용 예
# ============================================================
if __name__ == "__main__":
    
    # strict margin: logit을 쓰는 버전이면 1e-6 권장, 지금 y-interval 버전은 0.0~1e-6 중 선택
    # 0.0000001
    # 0.000000000000001
    eps = 0.00000001

    # 4케이스 실패 OR 전체 반례 공식 생성
    phi = cex_xor_all_cases(x1="x1", x2="x2", eps_strict=eps)
    
    print("==== XOR CEX query ====")
    print("eps_strict =", eps)
    # print("formula =", phi)

    # SAT check
    # (1) DPLL_T(phi) -> (sat: bool, model: dict) 형태라고 가정한 버전
    dpllModel, sat = dpll_t(phi)
    print_cex(sat, dpllModel)
    
