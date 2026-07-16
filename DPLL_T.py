from time import monotonic
from typing import List, Dict, Tuple, Optional
from DPLL import parse_prop, tseitin_cnf, dpll, neg
from Reluplex import reluplex
from DPLL import InequProp, ReLUProp
from Automation.SolverStatus import (
    SolverLimitReached,
    SolverResult,
    SolverStatus,
    check_deadline,
)


def inequ_list_to_reluplex(
    ineqs: List,
    start_idx: int = 0,
) -> Tuple[List[Tuple[str, Dict[str, float]]], Dict[str, Tuple[float, float]]]:
    """
    Translate a list of InequProp objects into (row_defs, bounds) for Reluplex.

    Each inequality coeffs*x >= b is encoded by introducing a slack/basic variable s_i:
        s_i = sum(coeffs[var] * var)
    and constraint s_i >= b by setting bounds[s_i] = (b, inf)

    Returns (row_defs, bounds)
    """
    row_defs: List[Tuple[str, Dict[str, float]]] = []
    bounds: Dict[str, Tuple[float, float]] = {}

    for i, ineq in enumerate(ineqs, start=start_idx):
        sname = f"ineq_slack_{i}"
        coeffs_dict = dict(ineq.coeffs)
        row_defs.append((sname, coeffs_dict))
        bounds[sname] = (ineq.b, float("inf"))

        for v in coeffs_dict.keys():
            if v not in bounds:
                bounds[v] = (float("-inf"), float("inf"))

    return row_defs, bounds


def _dpll_t_run(
    formula,
    max_rounds: int,
    debug: bool,
    deadline: Optional[float],
) -> Tuple[Optional[Dict[str, float]], SolverStatus, str, int]:
    """
    DPLL(T) main loop.

    The conflict clause must block the signed theory literals that were checked
    by the theory solver. Blocking only positive/True atoms is unsound when a
    conflict depends on a negated theory atom such as ``not ineq(...)``.
    """
    check_deadline(deadline)
    cnf, atom_map, _memo = tseitin_cnf(formula)
    check_deadline(deadline)

    atom_to_theory = {v: k for k, v in atom_map.items()}

    for round_idx in range(max_rounds):
        check_deadline(deadline)
        model = dpll(cnf, deadline=deadline)
        if model is None:
            return None, SolverStatus.UNSAT, "BOOLEAN_UNSAT", round_idx + 1

        active_ineqs = []
        active_relus: List[Tuple[str, str]] = []
        active_theory_literals = []

        for atom, th in atom_to_theory.items():
            check_deadline(deadline)
            if atom not in model:
                continue

            if model[atom] is True:
                active_theory_literals.append(atom)
                if isinstance(th, InequProp):
                    active_ineqs.append(th)
                elif isinstance(th, ReLUProp):
                    active_relus.append((th.x, th.y))
            elif model[atom] is False:
                active_theory_literals.append(neg(atom))
                if isinstance(th, InequProp):
                    coeffs_dict = dict(th.coeffs)
                    neg_coeffs = {v: -c for v, c in coeffs_dict.items()}
                    neg_ineq = InequProp(
                        coeffs=frozenset(neg_coeffs.items()),
                        b=-th.b + 1e-6,
                    )
                    active_ineqs.append(neg_ineq)
                elif isinstance(th, ReLUProp):
                    active_relus.append((th.x, th.y))
                    active_relus.append((f"not_{th.x}", f"not_{th.y}"))

        if not active_ineqs and not active_relus:
            return {}, SolverStatus.SAT, "THEORY_TRIVIAL", round_idx + 1

        row_defs, bounds = inequ_list_to_reluplex(active_ineqs)
        for x, y in active_relus:
            if x not in bounds:
                bounds[x] = (float("-inf"), float("inf"))
            if y not in bounds:
                bounds[y] = (float("-inf"), float("inf"))

        th_model, th_sat = reluplex(
            row_defs,
            bounds,
            active_relus,
            debug=debug,
            deadline=deadline,
            report_unknown=True,
        )
        if th_sat:
            return th_model, SolverStatus.SAT, "THEORY_SAT", round_idx + 1

        if not active_theory_literals:
            return None, SolverStatus.UNSAT, "THEORY_UNSAT", round_idx + 1

        blocking_clause = [neg(lit) for lit in active_theory_literals]
        cnf.append(blocking_clause)

    return None, SolverStatus.UNKNOWN, "DPLL_T_ROUND_LIMIT", max_rounds


def dpll_t_detailed(
    formula,
    max_rounds: int = 1000,
    debug: bool = False,
    timeout_seconds: Optional[float] = None,
) -> SolverResult:
    """Run DPLL(T), distinguishing SAT, UNSAT, and UNKNOWN."""
    started_at = monotonic()
    deadline = (
        started_at + float(timeout_seconds)
        if timeout_seconds is not None
        else None
    )
    try:
        model, status, reason, rounds = _dpll_t_run(
            formula,
            max_rounds=max_rounds,
            debug=debug,
            deadline=deadline,
        )
    except SolverLimitReached as exc:
        model = None
        status = SolverStatus.UNKNOWN
        reason = exc.reason
        rounds = 0
    return SolverResult(
        status=status,
        model=model,
        reason=reason,
        rounds=rounds,
        elapsed_seconds=monotonic() - started_at,
    )


def dpll_t(
    formula,
    max_rounds: int = 1000,
    debug: bool = False,
) -> Tuple[Optional[Dict[str, float]], bool]:
    result = dpll_t_detailed(
        formula,
        max_rounds=max_rounds,
        debug=debug,
        timeout_seconds=None,
    )
    if result.status == SolverStatus.UNKNOWN:
        raise SolverLimitReached(result.reason)
    return result.model, result.status == SolverStatus.SAT


def main() -> None:
    print("DPLL(T) demo")

    print("\n" + "=" * 55)
    print("  Simple SAT example: y = ReLU(x), x >= -3, x <= 2, y >= 1")
    prop = parse_prop("relu(x,y) and ineq(1,x,-3) and ineq(-1,x,-2) and ineq(1,y,1)")
    th_model, sat = dpll_t(prop, debug=False)

    print("Result:", "SAT" if sat else "UNSAT")
    if sat:
        print("Theory model:", th_model)

    print("\n" + "=" * 55)
    print("  SAT example: x + y >= 5, y = relu(x)")
    prop = parse_prop("ineq(1,x,1,y,5) and relu(x,y)")
    th_model, sat = dpll_t(prop, debug=False)

    print("Result:", "SAT" if sat else "UNSAT")
    if sat:
        print("Theory model:", th_model)

    print("\n" + "=" * 55)
    print("  Theory UNSAT example: x >= 0, y = relu(x), y < 0")
    prop_unsat = parse_prop("ineq(1,x,0) and relu(x,y) and ineq(-1,y,1e-6)")
    th_model_unsat, sat_unsat = dpll_t(prop_unsat, debug=False)

    print("Result:", "SAT" if sat_unsat else "UNSAT")
    if sat_unsat:
        print("Theory model:", th_model_unsat)

    print("\n" + "=" * 55)
    print("  ReLU branch UNSAT example: y = ReLU(x), x <= -3, y >= 1")
    prop_unsat2 = parse_prop("relu(x,y) and ineq(-1,x,3) and ineq(1,y,1)")
    th_model_unsat2, sat_unsat2 = dpll_t(prop_unsat2, debug=False)

    print("Result:", "SAT" if sat_unsat2 else "UNSAT")
    if sat_unsat2:
        print("Theory model:", th_model_unsat2)

    print("\n" + "=" * 55)
    print("  Negated inequality example: x >= 0 and not (x <= 0)")
    prop_unsat3 = parse_prop("ineq(1,x,0) and not ineq(-1,x,0)")
    th_model_unsat3, sat_unsat3 = dpll_t(prop_unsat3, debug=True)

    print("Result:", "SAT" if sat_unsat3 else "UNSAT")
    if sat_unsat3:
        print("Theory model:", th_model_unsat3)

    print("\n" + "=" * 55)
    print(" blocking clause example: x >= 1 and x >= 0 ")
    prop_unsat4 = parse_prop('(ineq(1,x,1) and not ineq(1,x,0)) or (ineq(1,x,1) and ineq(1,x,0))')
    th_model_unsat4, sat_unsat4 = dpll_t(prop_unsat4, debug=True)

    print('Result:', 'SAT' if sat_unsat4 else 'UNSAT')
    if sat_unsat4:
        print('Theory model:', th_model_unsat4)


if __name__ == "__main__":
    main()
