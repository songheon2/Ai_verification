from typing import List, Dict, Tuple, Optional
from DPLL import parse_prop, tseitin_cnf, dpll, neg
from DPLL import Literal, CNF
from Reluplex import reluplex


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


def dpll_t(formula, max_rounds: int = 1000) -> Tuple[Optional[Dict[str, float]], bool]:
    """
    DPLL(T) main loop.

    - Build Boolean abstraction CNF and mapping of inequalities -> atom names
    - Loop: run DPLL on CNF. If UNSAT -> overall UNSAT.
            else obtain Boolean model, collect active theory atoms, ask theory solver.
            if theory-sat -> return theory model
            else add blocking clause (negation of the active atoms) and repeat
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
                from DPLL import InequProp, ReLUProp
                if isinstance(th, InequProp):
                    active_ineqs.append(th)
                elif isinstance(th, ReLUProp):
                    active_relus.append((th.x, th.y))

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

        th_model, th_sat = reluplex(row_defs, bounds, active_relus)
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
    # simple example: x >= 0
    prop = parse_prop('ineq(1,x,1,y,5) and relu(x,y)')
    th_model, sat = dpll_t(prop)
    print('Result:', 'SAT' if sat else 'UNSAT')
    if sat:
        print('Theory model:', th_model)


if __name__ == '__main__':
    main()
