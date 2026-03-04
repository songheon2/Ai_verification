# =========================
# eps_strict 제거 버전
#  - input zero/one: eps 인자 없음
#  - output logit class: eps 인자 없음 (필요하면 별도 margin 버전으로 분리 권장)
#  - cex_case / cex_xor_all_cases: eps 인자 없음
#  - print_input_class_eval / print_cex: eps 인자 없음
# =========================

# 수정할 사항 

# 케이스 하나씩 검증하기

# 시간 측정해보기
# 1e-8

# UNSAT일 때 
# 포함안된 범위에서는 랜덤하게 선택해서 실제 신경망을 돌려보고
# 일치하는지 확인해서 경험적으로 UNSAT 판단 (1만번)


from numpy import isfinite

import random
from typing import List, Tuple

from DPLL import (
    Prop, TrueProp, FalseProp, VarProp, InequProp, ReLUProp,
    AndProp, OrProp, NotProp, ImplProp, tseitin_cnf
)
from DPLL_T import dpll_t
from XOREncoding import NN_single, NN_dual, FreshGen, eq_lin


# -------------------------
# small helper functions
# -------------------------
def AND(*ps: Prop) -> Prop:
    """n-ary And fold. AND() == True."""
    cur: Prop = TrueProp()
    for p in ps:
        cur = AndProp(cur, p)
    return cur


# -------------------------
# input classifiers (no eps)
# -------------------------
def zero(i: str) -> Prop:
    """
    zero(i) : (0 <= i) ^ (i < 0.5)

    NOTE:
      - InequProp is non-strict (>=) only.
      - So we encode (i < 0.5) approximately as (i <= 0.5).
        i <= 0.5  <=>  -i >= -0.5
      - If you must exclude i==0.5, add an explicit not-half constraint separately.
    """
    return AndProp(
        InequProp(frozenset([(i, 1.0)]), 0.0),   # i >= 0
        InequProp(frozenset([(i, -1.0)]), -0.5), # i <= 0.5
    )

def one(i: str) -> Prop:
    """
    one(i) : (0.5 <= i) ^ (i <= 1)
    """
    return AndProp(
        InequProp(frozenset([(i, 1.0)]), 0.5),   # i >= 0.5
        InequProp(frozenset([(i, -1.0)]), -1.0), # i <= 1
    )


# OPTIONAL: boundary exclusion helper (still no eps_strict plumbing; just a fixed delta parameter)
def not_half(i: str, delta: float = 1e-12) -> Prop:
    """
    Approximate i != 0.5 by excluding a tiny band:
      i <= 0.5 - delta  OR  i >= 0.5 + delta
    """
    left  = InequProp(frozenset([(i, -1.0)]), -(0.5 - delta))  # i <= 0.5 - delta
    right = InequProp(frozenset([(i,  1.0)]),  (0.5 + delta))  # i >= 0.5 + delta
    return OrProp(left, right)


# -------------------------
# output logit class (no eps)
# -------------------------
def out_zero_logit(s: str) -> Prop:
    """
    "0" class: s < 0
    Non-strict theory => encode as s <= 0  <=>  -s >= 0
    """
    return InequProp(frozenset([(s, -1.0)]), 0.0)

def out_one_logit(s: str) -> Prop:
    """
    "1" class: s > 0
    Non-strict theory => encode as s >= 0
    """
    return InequProp(frozenset([(s, 1.0)]), 0.0)


# (선택) margin이 필요하면 eps_strict라는 이름으로 전파하지 말고,
# 별도 함수로 명확히 분리하는 게 디버깅에 좋습니다.
def out_zero_logit_margin(s: str, margin: float) -> Prop:
    """s <= -margin  <=>  -s >= margin"""
    return InequProp(frozenset([(s, -1.0)]), float(margin))

def out_one_logit_margin(s: str, margin: float) -> Prop:
    """s >= margin"""
    return InequProp(frozenset([(s, 1.0)]), float(margin))


# -------------------------
# counterexample formula builders (no eps)
# -------------------------
eps = 1e-1

def cex_case(
    x1: str, x2: str,
    cls_x1, cls_x2, cls_out_logit,
    gen: FreshGen | None = None
) -> Prop:
    """
    Returns counterexample formula: PRE ∧ NN ∧ ¬POST
      PRE  = cls_x1(x1) ∧ cls_x2(x2) ∧ NNprop
      POST = cls_out_logit(s) where s is the logit output of the NN
    """
    NNprop, s, _ = NN_single((x1, x2), gen=gen)

    # x1, x2 가 0.5를 제외한 범위에서 샘플링된다고 가정 (즉, zero/one 클래스에 속한다고 가정)
    #     # x0, x1 >= 0.1 and x0, x1 <= 0.5 and x0, x1 >= 0.6 and x1, x2 <= 0.9
    # pre = AND(
    #     cls_x1(x1),
    #     cls_x2(x2),
    #     NNprop,
    #     InequProp(frozenset([(x1, 1.0)]), 0.1),   # x1 >= 0.1
    #     InequProp(frozenset([(x1, -1.0)]), -0.9), # x1 <= 0.9
    #     InequProp(frozenset([(x2, 1.0)]), 0.1),   # x2 >= 0.1
    #     InequProp(frozenset([(x2, -1.0)]), -0.9), # x2 <= 0.9
    #     InequProp(frozenset([(x1, 1.0)]), 0.6),   # x1 >= 0.6
    #     InequProp(frozenset([(x2, 1.0)]), 0.6),   # x2 >= 0.6
    #     InequProp(frozenset([(x1, -1.0)]), -0.4), # x1 <= 0.4
    #     InequProp(frozenset([(x2, -1.0)]), -0.4), # x2 <= 0.4
    # )

    # x0, x1 >= 0 + eps and x0, x1 <= 0.5-eps 
    pre = AND(
        cls_x1(x1),
        cls_x2(x2),
        NNprop,
        InequProp(frozenset([(x1, -1.0)]), -0.5 + eps), # x1 <= 0.5 - eps
        InequProp(frozenset([(x1, 1.0)]), 0.0 + eps),   # x1 >= 0.0 + eps
        InequProp(frozenset([(x2, -1.0)]), -0.5 + eps), # x2 <= 0.5 - eps
        InequProp(frozenset([(x2, 1.0)]), 0.0 + eps),   # x2 >= 0.0 + eps

        # InequProp(frozenset([(x1, -1.0)]), -1.0 + eps), # x1 <= 1.0 - eps
        # InequProp(frozenset([(x2, -1.0)]), -1.0 + eps), # x2 <= 1.0 - eps
        # InequProp(frozenset([(x1, 1.0)]), 0.5 + eps),   # x1 >= 0.5 + eps
        # InequProp(frozenset([(x2, 1.0)]), 0.5 + eps),   # x2 >= 0.5 + eps
        
        
    )
    post = cls_out_logit(s)
    return AND(pre, NotProp(post))

def cex_xor_all_cases(x1="x1", x2="x2") -> Prop:
    # case마다 fresh generator를 새로 만들어야 변수 충돌 없이 깔끔
    fg = FreshGen(prefix="x00_")
    cex00 = cex_case(x1, x2, zero, zero, out_zero_logit, gen=fg)  # 0 xor 0 = 0

    fg = FreshGen(prefix="x01_")
    cex01 = cex_case(x1, x2, zero, one,  out_one_logit,  gen=fg)  # 0 xor 1 = 1

    fg = FreshGen(prefix="x10_")
    cex10 = cex_case(x1, x2, one,  zero, out_one_logit,  gen=fg)  # 1 xor 0 = 1

    fg = FreshGen(prefix="x11_")
    cex11 = cex_case(x1, x2, one,  one,  out_zero_logit, gen=fg)  # 1 xor 1 = 0

    # "어느 케이스든 실패하면" 반례이므로 OR
    return OrProp(OrProp(cex00, cex01), OrProp(cex10, cex11))


# -------------------------
# pretty printing utils
# -------------------------
def _fmt(v, nd=6, sci_thresh=1e-4):
    try:
        fv = float(v)
        if not isfinite(fv):
            return str(v)
        if fv == 0.0:
            fv = 0.0
        if 0.0 < abs(fv) < sci_thresh:
            return f"{fv:.{nd}e}"
        return f"{fv:.{nd}f}"
    except Exception:
        return str(v)

def _fmt_with_raw(v, nd=6, sci_thresh=1e-4):
    try:
        fv = float(v)
        pretty = _fmt(fv, nd=nd, sci_thresh=sci_thresh)
        raw = f"{fv:+.18e}"
        return pretty, raw
    except Exception:
        return str(v), str(v)

def filter_model(
    model: dict,
    hide_prefixes=("ineq_slack", "relu_slack", "not_"),
    keep_exact=("x1", "x2"),
    keep_prefixes=("s_x", "s_c", "z_", "h_")
):
    out = {}
    for k, v in model.items():
        if any(k.startswith(p) for p in hide_prefixes):
            continue
        if k in keep_exact or any(k.startswith(p) for p in keep_prefixes):
            out[k] = v
    return out

def group_by_layer(filtered: dict):
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


# -------------------------
# shallow prop eval for printing
# -------------------------
def _eval_ineqprop(p: Prop, model: dict) -> bool | None:
    if not isinstance(p, InequProp):
        return None
    s = 0.0
    for (var, coeff) in p.coeffs:
        if var not in model:
            return None
        try:
            v = float(model[var])
            c = float(coeff)
        except Exception:
            return None
        if not isfinite(v) or not isfinite(c):
            return None
        s += c * v
    try:
        b = float(p.b)
    except Exception:
        return None
    if not isfinite(b):
        return None
    return (s >= b)

def _iter_child_props(obj: object):
    if not hasattr(obj, "__dict__"):
        return
    for _, val in obj.__dict__.items():
        if isinstance(val, Prop):
            yield val
        elif isinstance(val, (list, tuple)):
            for it in val:
                if isinstance(it, Prop):
                    yield it

def eval_prop_shallow(p: Prop, model: dict) -> bool | None:
    if isinstance(p, TrueProp):
        return True
    if isinstance(p, FalseProp):
        return False
    if isinstance(p, InequProp):
        return _eval_ineqprop(p, model)
    if isinstance(p, NotProp):
        child = getattr(p, "p", None)
        if isinstance(child, Prop):
            r = eval_prop_shallow(child, model)
        else:
            kids = list(_iter_child_props(p))
            r = eval_prop_shallow(kids[0], model) if kids else None
        return None if r is None else (not r)
    if isinstance(p, AndProp):
        kids = list(_iter_child_props(p))
        if not kids:
            return True
        any_unknown = False
        for k in kids:
            rk = eval_prop_shallow(k, model)
            if rk is False:
                return False
            if rk is None:
                any_unknown = True
        return None if any_unknown else True
    if isinstance(p, OrProp):
        kids = list(_iter_child_props(p))
        if not kids:
            return False
        any_unknown = False
        for k in kids:
            rk = eval_prop_shallow(k, model)
            if rk is True:
                return True
            if rk is None:
                any_unknown = True
        return None if any_unknown else False
    return None


# -------------------------
# printing (no eps)
# -------------------------
def print_input_class_eval(model: dict, x1_name="x1", x2_name="x2", cls_fns=(), nd=6):
    if x1_name not in model or x2_name not in model:
        print("  input-class: (missing x1/x2 in model)")
        return

    x1v = model[x1_name]
    x2v = model[x2_name]
    print("  input-class eval (by numeric model):")
    print(f"    {x1_name}={_fmt(x1v, nd)}")
    for (lbl, fn) in cls_fns:
        p = fn(x1_name)
        r = eval_prop_shallow(p, model)
        print(f"      {lbl}({x1_name}) = {r}")
    print(f"    {x2_name}={_fmt(x2v, nd)}")
    for (lbl, fn) in cls_fns:
        p = fn(x2_name)
        r = eval_prop_shallow(p, model)
        print(f"      {lbl}({x2_name}) = {r}")

def print_cex(
    sat, model, nd=6, max_hidden=24,
    show_input_classes=True,
    input_class_fns=(("zero", None), ("one", None))
):
    tag = "SAT  (counterexample)" if sat else "UNSAT (robust)"
    print(f"\n====== {tag}===========")
    if not sat:
        return

    fm = filter_model(model)
    g = group_by_layer(fm)

    if g["inputs"]:
        x1 = g["inputs"].get("x1", None)
        x2 = g["inputs"].get("x2", None)
        x1p, x1r = _fmt_with_raw(x1, nd=nd)
        x2p, x2r = _fmt_with_raw(x2, nd=nd)
        print(f"  x:  x1={x1p}  x2={x2p}")
        print(f"      raw: x1={x1r}  x2={x2r}")

    if show_input_classes:
        cls_fns = [(lbl, fn) for (lbl, fn) in input_class_fns if fn is not None]
        if cls_fns:
            print_input_class_eval(model, x1_name="x1", x2_name="x2", cls_fns=cls_fns, nd=nd)

    if g["logits"]:
        for name in sorted(g["logits"].keys(), key=lambda k: (0 if k.startswith("s_c") else 1, k)):
            val = g["logits"][name]
            fv = float(val)
            print(f"  {name}: {_fmt(val, nd)}   (raw={fv:+.18e})")

        s_c_keys = sorted([k for k in g["logits"] if k.startswith("s_c")])
        s_x_keys = sorted([k for k in g["logits"] if k.startswith("s_x")])
        if s_c_keys and s_x_keys:
            print("  case-wise same_class?  (sc>=0 <-> sx>=0):")
            def case_id(k):
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


def xor_nn_sx(x1: float, x2: float) -> float:
    """
    Compute sx (logit) for the given (x1, x2) using the provided XOR NN params.

    Network:
      zx1 = 2.1247*x1 + 2.1267*x2 - 2.1259
      zx2 = -2.1237*x1 - 2.1235*x2 + 2.1234
      hx1 = relu(zx1)
      hx2 = relu(zx2)
      sx  = -3.6788*hx1 - 3.6766*hx2 + 3.5451
    """
    zx1 = 2.1247 * x1 + 2.1267 * x2 - 2.1259
    zx2 = -2.1237 * x1 - 2.1235 * x2 + 2.1234

    hx1 = zx1 if zx1 > 0.0 else 0.0
    hx2 = zx2 if zx2 > 0.0 else 0.0

    sx = -3.6788 * hx1 - 3.6766 * hx2 + 3.5451
    return sx

def xor_nn(x1: float, x2: float) -> int:
    """
    Compute the XOR NN output logit sx for given inputs X1, X2.
    """
    if( x1 >= 0 and x1 < 0.5 and x2 >= 0 and x2 < 0.5):
        return 0
    elif( x1 >= 0 and x1 < 0.5 and x2 >= 0.5 and x2 <= 1):
        return 1
    elif( x1 >= 0.5 and x1 <= 1 and x2 >= 0 and x2 < 0.5):
        return 1
    elif( x1 >= 0.5 and x1 <= 1 and x2 >= 0.5 and x2 <= 1):
        return 0

def nn_label_from_sx(sx: float) -> int:
    """Classify NN output by sign of logit: sx >= 0 -> 1 else 0."""
    return 1 if sx >= 0.0 else 0

def check_xor_on_zero_zero_region(
    eps: float,
    n_samples: int = 1000,
    seed: int = 0,
    max_print: int = 10,
) -> List[Tuple[float, float, int, int, float]]:
    """
    Sample (x0,x1) uniformly from [eps, 0.5-eps]^2, compare:
      expected XOR label (0 xor 0 = 0)  vs  NN label(sign(sx)).

    Prints up to max_print mismatches and returns them:
      (x0, x1, expected_y, nn_y, sx)
    """
    random.seed(seed)

    # x1, x2의 lo, hi를 받아와야 한다.
    lo, hi = 0.0 + eps, 0.5 - eps
    if not (lo <= hi):
        raise ValueError(f"Invalid eps={eps}: interval [{lo},{hi}] is empty.")

    mismatches: List[Tuple[float, float, int, int, float]] = []
      # since both inputs are in "zero" region

    for _ in range(n_samples):
        x0 = random.uniform(lo, hi)
        x1 = random.uniform(lo, hi)

        expected_y = xor_nn(x0, x1)

        sx = xor_nn_sx(x0, x1)
        nn_y = nn_label_from_sx(sx)

        if nn_y != expected_y:
            mismatches.append((x0, x1, expected_y, nn_y, sx))
            if len(mismatches) <= max_print:
                print(
                    f"Mismatch: x0={x0:.18e}, x1={x1:.18e}, "
                    f"expected_y={expected_y}, nn_y={nn_y}, sx={sx:+.18e}"
                )
            if len(mismatches) >= max_print:
                # stop early once we've printed max_print mismatches
                # (you can remove this break if you want to continue collecting)
                break

    if not mismatches:
        print(f"No mismatches in {n_samples} samples for region [eps, 0.5-eps]^2 with eps={eps}.")
    else:
        print(f"Found {len(mismatches)} mismatches (printed up to {max_print}).")

    return mismatches

# -------------------------
# main usage
# -------------------------
if __name__ == "__main__":
    phi = cex_xor_all_cases(x1="x1", x2="x2")

    print("==== XOR CEX query ====")
    # print("formula =", phi)

    dpllModel, sat = dpll_t(phi)
    print_cex(
        sat, dpllModel,
        input_class_fns=(("zero", zero), ("one", one)),
    )

    if sat:
        x1 = float(dpllModel.get("x1", 0.0))
        x2 = float(dpllModel.get("x2", 0.0))
        sx = xor_nn_sx(x1, x2)
        print(f"\n  Evaluated sx for (x1, x2) = ({_fmt(x1)}, {_fmt(x2)}): sx = {_fmt(sx)}")
    else:
        print("x0, x1 >= 0 + eps and x0, x1 <= 0.5-eps ")
        # 1만개 를 해당 범위내에서 샘플링 후 xor_nn_sx 계산해서 
        # xor 동작을 검증한 다음에 xor 동작과 다르면 x1, x2, xor(x1,x2) 결과를 출력해줘
        # 최대 10개까지만 출력해줘
        check_xor_on_zero_zero_region(eps=eps, n_samples=10000, seed=0, max_print=10)


# 제약에 해당 실제 반례 추가해서 debugging