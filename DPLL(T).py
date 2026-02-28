from typing import List, Dict, Tuple, Optional
from DPLL import parse_prop, to_nnf, tseitin_cnf, dpll, neg
from DPLL import Literal, CNF
from Reluplex import reluplex
from DPLL import InequProp, ReLUProp, AndProp, NotProp

def inequ_list_to_reluplex(ineqs: List, start_idx: int = 0) -> Tuple[List[Tuple[str, Dict[str, float]]], Dict[str, Tuple[float, float]]]:
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
        # ineq.coeffs is a frozenset of (var, coeff) pairs
        coeffs_dict = dict(ineq.coeffs)
        row_defs.append((sname, coeffs_dict))
        bounds[sname] = (ineq.b, float('inf'))

        # ensure variables have unbounded ranges unless already set
        for v in coeffs_dict.keys():
            if v not in bounds:
                bounds[v] = (float('-inf'), float('inf'))

    return row_defs, bounds


def dpll_t(formula, max_rounds: int = 1000, debug: bool = False) -> Tuple[Optional[Dict[str, float]], bool]:
    """
    DPLL(T) main loop.

    - Build Boolean abstraction CNF and mapping of inequalities -> atom names
    - Loop: run DPLL on CNF. If UNSAT -> overall UNSAT.
            else obtain Boolean model, collect active theory atoms, ask theory solver.
            if theory-sat -> return theory model
            else add blocking clause (negation of the active atoms) and repeat

    Args:
        formula: propositional formula with theory atoms
        max_rounds: limit on Boolean search iterations
        debug: if True, pass debug=True to the underlying simplex calls via
               reluplex (useful for tracing the theory-solving steps)
    """
    cnf, atom_map = tseitin_cnf(formula)

    # inverse map: atom name -> theory atom (InequProp or ReLUProp)
    atom_to_theory = {v: k for k, v in atom_map.items()}

    for round_idx in range(max_rounds):
        model = dpll(cnf)
        if model is None:
            return None, False

        # collect theory atoms assigned True in Boolean model
        active_ineqs = []
        active_relus: List[Tuple[str, str]] = []
        active_atoms = []
        for atom, th in atom_to_theory.items():
            if atom in model and model[atom] is True:
                active_atoms.append(atom)
                # classify theory atom
                # Inequalities -> to be converted to row_defs/bounds
                # ReLU atoms -> relus list for Reluplex
                if isinstance(th, InequProp):
                    active_ineqs.append(th)
                elif isinstance(th, ReLUProp):
                    active_relus.append((th.x, th.y))
            elif atom in model and model[atom] is False:
                # handle negated theory atoms by adding their negation to the active set
                if atom in atom_to_theory:
                    th = atom_to_theory[atom]
                    if isinstance(th, InequProp):
                        # add negation of inequality: sum(-coeffs[var]*var) >= -b
                        coeffs_dict = dict(th.coeffs)

                        neg_coeffs = {v: -c for v, c in coeffs_dict.items()}
                        neg_ineq = InequProp(coeffs=frozenset(neg_coeffs.items()), b=-th.b+1e-6)  # c * x < b
                        active_ineqs.append(neg_ineq)
                    elif isinstance(th, ReLUProp):
                        # add negation of ReLU: y != relu(x) -> (y < 0 and x >= 0) or (y >= 0 and x < 0)
                        # for simplicity, we can encode this as two separate cases in the theory solver
                        active_relus.append((th.x, th.y))  # original ReLU constraint
                        active_relus.append((f"not_{th.x}", f"not_{th.y}"))  # negated case

        # if no theory atoms are active, theory trivially sat
        if not active_ineqs and not active_relus:
            # return SAT with empty model (no reals specified)
            return {}, True

        # translate to Reluplex input
        row_defs, bounds = inequ_list_to_reluplex(active_ineqs)
        # ensure variables referenced by relus exist in bounds
        for x, y in active_relus:
            if x not in bounds:
                bounds[x] = (float('-inf'), float('inf'))
            if y not in bounds:
                bounds[y] = (float('-inf'), float('inf'))

        # forward debug flag to reluplex so that simplex prints
        print(row_defs, bounds)
        th_model, th_sat = reluplex(row_defs, bounds, active_relus, debug=debug)
        if th_sat:
            return th_model, True

        # theory conflict -> block this assignment by adding clause ¬a1 ∨ ¬a2 ∨ ...
        if not active_atoms:
            # nothing to block -> UNSAT
            return None, False

        blocking_clause = ["~" + a for a in active_atoms]
        cnf.append(blocking_clause)

    # exceeded rounds
    return None, False


def main() -> None:
    print("DPLL(T) demo")

    print("\n" + "=" * 55)
    print("  간단 예제 (SAT) : y = ReLU(x) and x ≥ -3 and and x <= 2 and y ≥ 1")
    prop = parse_prop('relu(x,y) and ineq(1,x,-3) and ineq(-1,x,-2) and ineq(1,y,1)')
    th_model, sat = dpll_t(prop, debug=False)
    
    print('Result:', 'SAT' if sat else 'UNSAT')
    if sat:
        print('Theory model:', th_model)
    
    print("\n" + "=" * 55)
    print("  이론솔버에서 제약을 추가하여 푸는 경우 (SAT) : x + y >= 5, y = relu(x)")
    prop = parse_prop('ineq(1,x,1,y,5) and relu(x,y)')
    th_model, sat = dpll_t(prop, debug=False)
    
    print('Result:', 'SAT' if sat else 'UNSAT')
    if sat:
        print('Theory model:', th_model)

    print("\n" + "=" * 55)
    print("  DPLL에선 SAT, 이론솔버에선 UNSAT인 예제 : x >= 0, y = relu(x), y < 0")
    prop_unsat = parse_prop('ineq(1,x,0) and relu(x,y) and ineq(-1,y,1e-6)')
    th_model_unsat, sat_unsat = dpll_t(prop_unsat, debug=False)
    
    print('Result:', 'SAT' if sat_unsat else 'UNSAT')
    if sat_unsat:
        print('Theory model:', th_model_unsat)

    print("\n" + "=" * 55)
    print(" 이론솔버에서 추가 제약 생성을 생성해서 푸는 경우 (UNSAT) : y = ReLU(x) and x ≤ -3 and y ≥ 1")
    prop_unsat2 = parse_prop('relu(x,y) and ineq(-1,x,3) and ineq(1,y,1)')
    th_model_unsat2, sat_unsat2 = dpll_t(prop_unsat2, debug=False)

    print('Result:', 'SAT' if sat_unsat2 else 'UNSAT')
    if sat_unsat2:
        print('Theory model:', th_model_unsat2)

    print("\n" + "=" * 55)
    print(" not IneqProp 을 반영하는지 (SAT) : x >= 0 and not (-x >= 0)")
    prop_unsat3 = parse_prop('ineq(1,x,0) and not ineq(-1,x,0)')
    th_model_unsat3, sat_unsat3 = dpll_t(prop_unsat3, debug=False)

    print('Result:', 'SAT' if sat_unsat3 else 'UNSAT')
    if sat_unsat3:
        print('Theory model:', th_model_unsat3)

if __name__ == '__main__':
    main()
