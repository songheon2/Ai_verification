# y = xor(x1, x2)

# zero(i) : (0 <= i) ^ (i < 0.5)
# one(i) : (0.5 <= i) ^ (i <= 1)

# “이 신경망이 XOR을 정확하게 구현하는가?”

# case 00

# (PRE and NN) => POST

# PRE: zero(x1) and zero(x2)
#        s = xor(x1, x2)의 출력층 활성화 함수를 넣기전 logit
# POST:(s < 0)   시그모이드 함수의 단조성에 의해 sigmoid(s) < 0.5

# dpll(t) 에 넣을 최종 식
# 반례 찾기 형식   
# 최종식 = not 원래식

# (PRE and NN and not POST)

# “이 신경망이 XOR과 다르게 동작하는 입력이 존재하는가?”

# SAT => 반례 존제
# UNSAT => 반례 없음





from numpy import isfinite
import time

import random
from typing import List, Tuple

from DPLL import (
    Prop, TrueProp, FalseProp, VarProp, InequProp, ReLUProp,
    AndProp, OrProp, NotProp, ImplProp, tseitin_cnf
)
from DPLL_T import dpll_t
from XOREncoding import NN_single, NN_dual, FreshGen, eq_lin
import copy

from visualize_prop import dump_search_phi_visualization
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
eps = 0.2



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

    Network: ReLu + Sigmoid
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

def check_xor_on_region(
    r1: Tuple[float, float],
    r2: Tuple[float, float],
    expected_out: int,
    n_samples: int = 1000,
    seed: int = 0,
    max_print: int = 10,
) -> List[Tuple[float, float, int, int, float]]:
    """
    Sample (x1,x2) uniformly from r1 x r2, compare:
      expected XOR label (provided by expected_out)  vs  NN label(sign(sx)).

    Prints up to max_print mismatches and returns them:
      (x1, x2, expected_y, nn_y, sx)
    """
    random.seed(seed)

    lo1, hi1 = r1
    lo2, hi2 = r2
    if not (lo1 <= hi1) or not (lo2 <= hi2):
        raise ValueError(f"Invalid region: [{lo1},{hi1}] x [{lo2},{hi2}]")

    mismatches: List[Tuple[float, float, int, int, float]] = []

    for _ in range(n_samples):
        x1 = random.uniform(lo1, hi1)
        x2 = random.uniform(lo2, hi2)

        expected_y = expected_out
        sx = xor_nn_sx(x1, x2)
        nn_y = nn_label_from_sx(sx)

        if nn_y != expected_y:
            mismatches.append((x1, x2, expected_y, nn_y, sx))
            if len(mismatches) <= max_print:
                print(
                    f"Mismatch: x1={x1:.18e}, x2={x2:.18e}, "
                    f"expected_y={expected_y}, nn_y={nn_y}, sx={sx:+.18e}"
                )
            if len(mismatches) >= max_print:
                break

    if not mismatches:
        print(f"No mismatches in {n_samples} samples for region [{lo1},{hi1}] x [{lo2},{hi2}].")
    else:
        print(f"Found {len(mismatches)} mismatches (printed up to {max_print}).")

    return mismatches

# -------------------------
# main usage
# -------------------------
if __name__ == "__main__":

    eps = 0.1  # 기존 상단에서도 정의되어 있을 수 있습니다
    
    x1 = "x1"
    x2 = "x2"

    # visualize_precise_prop 폴더에 이미지 파일 만들기
    VISUALIZE_PHI_EACH_ATTEMPT = False
    VISUALIZE_RENDER_PNG = False

    ranges = [
        (0.0, 0.5 - eps),   # 구간1: [0, 0.5 - eps]
        (0.5 + eps, 1.0)    # 구간2: [0.5 + eps, 1]
    ]
    cases = [
        (0, 0, zero, zero, out_zero_logit),   # x1 ∈ 구간1, x2 ∈ 구간1, XOR output=0
        (0, 1, zero, one,  out_one_logit),    # x1 ∈ 구간1, x2 ∈ 구간2, XOR output=1
        (1, 0, one,  zero, out_one_logit),    # x1 ∈ 구간2, x2 ∈ 구간1, XOR output=1
        (1, 1, one,  one,  out_zero_logit),   # x1 ∈ 구간2, x2 ∈ 구간2, XOR output=0
    ]

    for i, (r1_idx, r2_idx, x1cls, x2cls, outcls) in enumerate(cases):
        case_name = f"{r1_idx}{r2_idx}"
        r1 = ranges[r1_idx]
        r2 = ranges[r2_idx]

        def x1_range_prop(i):
            return AndProp(
                InequProp(frozenset([(i, 1.0)]), r1[0]),   # x1 >= lower bound
                InequProp(frozenset([(i, -1.0)]), -r1[1])  # x1 <= upper bound
            )

        def x2_range_prop(i):
            return AndProp(
                InequProp(frozenset([(i, 1.0)]), r2[0]),   # x2 >= lower bound
                InequProp(frozenset([(i, -1.0)]), -r2[1])  # x2 <= upper bound
            )

        fg = FreshGen(prefix=f"x{r1_idx}{r2_idx}_")
        NNprop, s, _ = NN_single((x1, x2), gen=fg)

        # case 00
        # PRE: zero(x1) and zero(x2)
        # pre = (x1, x2 >= 0) and (x1, x2 < 0.5 - eps) and NN
        # post = not (s < 0)

        pre = AND(
            x1_range_prop(x1),
            x2_range_prop(x2),
            NNprop
        )
        post = outcls(s)
        phi = AND(pre, NotProp(post))

        print(f"\n==== XOR CEX query for x1 in [{r1[0]}, {r1[1]}], x2 in [{r2[0]}, {r2[1]}] ====")

        counterexamples = []
        phi_current = copy.deepcopy(phi)

        # 반례 최대 5개 찾기
        while len(counterexamples) < 5:
            attempt_no = len(counterexamples) + 1

            # phi_current를 시각화
            if VISUALIZE_PHI_EACH_ATTEMPT:
                dump_search_phi_visualization(
                    phi_current,
                    case_name=case_name,
                    attempt_no=attempt_no,
                    render_png=VISUALIZE_RENDER_PNG,
                    print_alias_map=False,
                )

            t0 = time.perf_counter()
            dpllModel, sat = dpll_t(phi_current)
            elapsed = time.perf_counter() - t0

            if not sat:
                print(f"  counterexample search #{attempt_no}: UNSAT ({elapsed:.3f}s)")
                break

            print(f"  counterexample #{attempt_no} found in {elapsed:.3f}s")

            x1v = float(dpllModel.get("x1", 0.0))
            x2v = float(dpllModel.get("x2", 0.0))
            sx = xor_nn_sx(x1v, x2v)

            counterexamples.append((x1v, x2v, sx))

            # SMT Solver는 연속 구간 해가 있을 때 "존재한다"의 대표 1개만 반환합니다.
            # 반례 차단을 위해 (x1 != x1v) OR (x2 != x2v) 대신 "delta 차이 이상" 조건식 사용:
            # 예전: 거의 같은 점 여러 번 찾기 → delta 너무 작음
            #      실제 반례를 너무 많이 빼기 → delta 너무 큼

            # delta 만큼 제외
            # ex) 0 <= x <= 0.5 - eps 에서 해를 찾으면
            # |--------------------|

            # delta 만큼 제외한 구간에서 dpll_t로 다른 해를 찾기 (두번째 반례 찾기)
            # |-----| |------------|

            # 세번째 반례 찾기
            # |-----| |----| |-----|

            # 문제점
            # 만약 대표 반례 하나만 하면 출력이 빠르지만
            # 대표 반례 여러개 계산하려면 
            # 논리식에 or도 섞이게 되면서 검색 공간 분기가 커진다. => 계산이 느려짐

            delta = 1e-1

            # x1, x2 제외
            exclude_x1 = OrProp(
                InequProp(frozenset([(x1, 1.0)]), x1v + delta),
                InequProp(frozenset([(x1, -1.0)]), -(x1v - delta))
            )
            exclude_x2 = OrProp(
                InequProp(frozenset([(x2, 1.0)]), x2v + delta),
                InequProp(frozenset([(x2, -1.0)]), -(x2v - delta))
            )
            # phi = phi AND (x1 != x1v) OR (x2 != x2v)
            neq = OrProp(exclude_x1, exclude_x2)
            phi_current = AndProp(phi_current, neq)

        if counterexamples:
            for idx, (x1v, x2v, sx) in enumerate(counterexamples, 1):
                print(f"SAT (counterexample) #{idx}:")
                print(f"  x1 = {_fmt(x1v)}")
                print(f"  x2 = {_fmt(x2v)}")
                print(f"  sx = {_fmt(sx)}")
                print("  evaluated label:", nn_label_from_sx(sx))
                print()
        else:
            print(f"x1 in [{r1[0]}, {r1[1]}], x2 in [{r2[0]}, {r2[1]}] -- UNSAT")
            # 1만개 를 해당 범위내에서 샘플링 후 xor_nn_sx 계산해서 
            # xor 동작을 검증한 다음에 xor 동작과 다르면 x1, x2, xor(x1,x2) 결과를 출력

            # UNSAT일 때 진짜 반례가 없는지 확인을 위해 
            # x 범위에서 랜덤하게 샘플링하고 
            # 해당 x(x1, x2)를 실제 신경망을 돌려서
            # 일치하는지 확인해서 경험적으로 UNSAT 판단 (1만번)
            print("\n[검산] NN이 각 region에서 실제로 xor 동작을 하는지 1만개 샘플링 검증:")
            mismatches = check_xor_on_region(
                r1, r2,
                expected_out=0 if outcls is out_zero_logit else 1,
                n_samples=10000,
                max_print=10
            )


# 제약에 해당 실제 반례 추가해서 debugging
