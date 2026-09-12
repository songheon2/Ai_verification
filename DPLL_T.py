from time import monotonic
from typing import Any, Callable, Iterable, List, Dict, Tuple, Optional
from DPLL import parse_prop, tseitin_cnf, dpll, neg
from Reluplex import reluplex
from DPLL import InequProp, ReLUProp
from Automation.SolverStatus import (
    SolverLimitReached,
    SolverResult,
    SolverStatus,
    check_deadline,
)
from visualization.SolveTrace import SolveTrace, span


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
        # 호출자가 InequProp을 직접 만들면서 0 계수를 명시한 경우에도
        # theory tableau는 희소하게 유지한다.
        coeffs_dict = {
            var: coefficient
            for var, coefficient in ineq.coeffs
            if coefficient != 0.0
        }
        row_defs.append((sname, coeffs_dict))
        bounds[sname] = (ineq.b, float("inf"))

        for v in coeffs_dict.keys():
            if v not in bounds:
                bounds[v] = (float("-inf"), float("inf"))

    return row_defs, bounds


def _theory_atom_description(atom, theory, polarity: bool) -> Dict[str, Any]:
    expression = repr(theory)
    if len(expression) > 500:
        expression = expression[:497] + "..."
    return {
        "atom_id": str(atom),
        "polarity": polarity,
        "kind": type(theory).__name__,
        "expression": expression,
    }


def _inequality_detail(original: InequProp, effective: InequProp) -> Dict[str, Any]:
    """Structured definition kept in memory and logged only on first violation."""
    return {
        "original": {
            "coefficients": sorted(
                ([str(var), float(coefficient)] for var, coefficient in original.coeffs),
                key=lambda item: item[0],
            ),
            "bound": float(original.b),
        },
        "effective": {
            "coefficients": sorted(
                ([str(var), float(coefficient)] for var, coefficient in effective.coeffs),
                key=lambda item: item[0],
            ),
            "bound": float(effective.b),
        },
    }


def _dpll_t_run(
    formula,
    max_rounds: int,
    debug: bool,
    deadline: Optional[float],
    trace: Optional[SolveTrace] = None,
    simplex_max_iter: int = 10000,
    max_recursion: int = 50,
    stats: Optional[Dict[str, int]] = None,
    profile_stages: bool = False,
    split_logger=None,
    progress=None,
    round_state: Optional[Dict[str, int]] = None,
) -> Tuple[Optional[Dict[str, float]], SolverStatus, str, int]:
    """
    DPLL(T) main loop.

    The conflict clause must block the signed theory literals that were checked
    by the theory solver. Blocking only positive/True atoms is unsound when a
    conflict depends on a negated theory atom such as ``not ineq(...)``.
    """
    check_deadline(deadline)
    if profile_stages:
        _t0 = monotonic()
        cnf, atom_map, _memo = tseitin_cnf(formula)
        print(
            f"[profile] tseitin_cnf: {monotonic() - _t0:.1f}s -> "
            f"{len(cnf)} clauses, {len(atom_map)} atoms"
        )
    else:
        cnf, atom_map, _memo = tseitin_cnf(formula)
    check_deadline(deadline)

    atom_to_theory = {v: k for k, v in atom_map.items()}

    profile_state = (
        {"calls": 0, "start": monotonic(), "print_every": 1000} if profile_stages else None
    )

    for round_idx in range(max_rounds):
        round_number = round_idx + 1
        with span(trace, "dpll_round_start", round_index=round_number):
            pass
        if split_logger is not None:
            split_logger.round_start(round_number)
        if round_state is not None:
            round_state["current"] = round_number
        if progress is not None:
            progress.round_start(round_number)
        check_deadline(deadline)
        model = dpll(cnf, deadline=deadline, trace=trace, profile_state=profile_state)
        if model is None:
            if progress is not None:
                progress.round_end(round_number, "BOOLEAN_UNSAT")
            return None, SolverStatus.UNSAT, "BOOLEAN_UNSAT", round_number

        active_ineqs = []
        active_relus: List[Tuple[str, str]] = []
        active_relu_metadata: Dict[Tuple[str, str], Tuple[Optional[int], Optional[int]]] = {}
        active_theory_literals = []
        selected_theory_atoms: List[Dict[str, Any]] = []

        for atom, th in atom_to_theory.items():
            check_deadline(deadline)
            if atom not in model:
                continue

            if model[atom] is True:
                active_theory_literals.append(atom)
                description = _theory_atom_description(atom, th, True)
                if isinstance(th, InequProp):
                    slack_name = f"ineq_slack_{len(active_ineqs)}"
                    description.update(
                        {
                            "constraint_id": str(atom),
                            "slack_name": slack_name,
                            "constraint_detail": _inequality_detail(th, th),
                        }
                    )
                    active_ineqs.append(th)
                elif isinstance(th, ReLUProp):
                    active_relus.append((th.x, th.y))
                    active_relu_metadata[(th.x, th.y)] = (th.layer, th.index)
                selected_theory_atoms.append(description)
            elif model[atom] is False:
                active_theory_literals.append(neg(atom))
                description = _theory_atom_description(atom, th, False)
                if isinstance(th, InequProp):
                    coeffs_dict = dict(th.coeffs)
                    neg_coeffs = {v: -c for v, c in coeffs_dict.items()}
                    neg_ineq = InequProp(
                        coeffs=frozenset(neg_coeffs.items()),
                        b=-th.b + 1e-6,
                    )
                    slack_name = f"ineq_slack_{len(active_ineqs)}"
                    description.update(
                        {
                            "constraint_id": str(atom),
                            "slack_name": slack_name,
                            "constraint_detail": _inequality_detail(th, neg_ineq),
                        }
                    )
                    active_ineqs.append(neg_ineq)
                elif isinstance(th, ReLUProp):
                    # ¬(y = ReLU(x))는 하나의 ReLU 제약으로 바꿀 수 없다.
                    # 이전 코드는 원래 ReLU와 이름만 바꾼 가짜 ReLU를 함께
                    # 추가했는데, 이는 negation을 표현하지 못하고 SAT 공식을
                    # UNSAT으로 차단할 수 있었다. 현재 theory backend가 이
                    # 비선형 complement를 정확히 지원할 때까지 UNKNOWN으로
                    # 중단해 잘못된 UNSAT 판정을 방지한다.
                    raise SolverLimitReached("NEGATED_RELU_UNSUPPORTED")
                selected_theory_atoms.append(description)

        if progress is not None:
            progress.theory_selection(round_number, selected_theory_atoms)

        if not active_ineqs and not active_relus:
            if progress is not None:
                progress.theory_result(round_number, "THEORY_TRIVIAL")
                progress.round_end(round_number, "THEORY_TRIVIAL")
            return {}, SolverStatus.SAT, "THEORY_TRIVIAL", round_number

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
            trace=trace,
            simplex_max_iter=simplex_max_iter,
            max_recursion=max_recursion,
            stats=stats,
            split_logger=split_logger,
            relu_metadata=active_relu_metadata,
            progress=progress,
            progress_context={
                "round_index": round_number,
                "theory_atom_ids": [
                    str(item["atom_id"]) for item in selected_theory_atoms
                ],
            },
        )
        if th_sat:
            if progress is not None:
                progress.theory_result(
                    round_number, "THEORY_SAT",
                    inequalities=[{"coeffs": dict(item.coeffs), "lower": item.b}
                                  for item in active_ineqs],
                    relu_count=len(active_relus),
                )
                progress.round_end(round_number, "THEORY_SAT")
            return th_model, SolverStatus.SAT, "THEORY_SAT", round_number

        if not active_theory_literals:
            if progress is not None:
                progress.theory_result(round_number, "THEORY_UNSAT")
                progress.round_end(round_number, "THEORY_UNSAT")
            return None, SolverStatus.UNSAT, "THEORY_UNSAT", round_number

        blocking_clause = [neg(lit) for lit in active_theory_literals]
        cnf.append(blocking_clause)
        if progress is not None:
            progress.theory_result(
                round_number,
                "THEORY_CONFLICT",
                blocking_clause_size=len(blocking_clause),
            )
            progress.round_end(round_number, "THEORY_CONFLICT")

    return None, SolverStatus.UNKNOWN, "DPLL_T_ROUND_LIMIT", max_rounds


def dpll_t_detailed(
    formula,
    max_rounds: int = 1000,
    debug: bool = False,
    timeout_seconds: Optional[float] = None,
    trace: Optional[SolveTrace] = None,
    simplex_max_iter: int = 10000,
    max_recursion: int = 50,
    profile_stages: bool = False,
    split_mode: str = "off",
    split_log_path: Optional[str] = None,
    split_update_callback: Optional[Callable[[], None]] = None,
    realtime_output_path: Optional[str] = None,
    realtime_open_view: bool = False,
    realtime_playback_ms: int = 0,
    realtime_window_seconds: float = 300.0,
    model_layer_sizes: Optional[Iterable[int]] = None,
    progress=None,
) -> SolverResult:
    """Run DPLL(T), distinguishing SAT, UNSAT, and UNKNOWN.

    trace를 넘기면(SolveTrace()) BCP/Simplex/Reluplex 구간 타이밍과 ReLU split
    이벤트가 그 안에 기록된다 (visualization/SolveTrace.py 참고). 안 넘기면
    계측 코드가 전혀 실행되지 않는다.

    simplex_max_iter는 Reluplex 내부 각 Simplex 호출의 반복 상한이다
    (SIMPLEX_ITERATION_LIMIT으로 UNKNOWN이 자주 나면 크게 올릴 것). timeout_seconds가
    실질적인 안전장치이므로, 이 값을 크게 올릴 때는 timeout_seconds도 넉넉히 잡아야 한다.

    profile_stages=True면 tseitin_cnf() 소요시간+CNF 크기, dpll() 재귀 호출
    진행상황(1000회마다)을 stdout에 [profile] 접두어로 찍는다 — 어느 단계에서
    멈춰있는지 진단할 때 켠다 (평소엔 꺼둘 것, 출력이 늘어남).
    """
    if split_mode not in ("off", "realtime"):
        raise ValueError("split_mode는 'off' 또는 'realtime'이어야 합니다")
    # 실시간 모드는 append-only 로그와 update callback을 명시적으로 요청한
    # 실행에서만 켠다. 사후 피드백 visualization은 기존 SolveTrace와
    # SplitHeatmap이 담당하므로 solver 실행 모드로 취급하지 않는다.
    split_logger = None
    visualizer = None
    if split_mode == "realtime":
        from visualization.RealtimeSplitVisualization import (
            DEFAULT_SPLIT_LOG_PATH,
            RealtimeSplitVisualizer,
            SplitEventLogger,
        )

        resolved_split_log_path = split_log_path or DEFAULT_SPLIT_LOG_PATH
        update_callback = split_update_callback
        if split_mode == "realtime" and realtime_output_path is not None:
            visualizer = RealtimeSplitVisualizer(
                resolved_split_log_path,
                realtime_output_path,
                live_view=realtime_open_view,
                open_live_view=realtime_open_view,
                playback_interval_ms=realtime_playback_ms,
                window_seconds=realtime_window_seconds,
                model_layer_sizes=model_layer_sizes,
            )
            if update_callback is None:
                update_callback = visualizer.request_update
            else:
                original_callback = update_callback

                def update_callback():
                    visualizer.request_update()
                    original_callback()
        split_logger = SplitEventLogger(
            resolved_split_log_path,
            update_callback=update_callback,
            reset_log=True,
            model_layer_sizes=model_layer_sizes,
        )
        # split이 한 번도 발생하지 않아도 0개 상태의 실시간 화면을 만든다.
        split_logger.request_update()

    theory_stats: Dict[str, int] = {}
    started_at = monotonic()
    round_state = {"current": 0}
    deadline = (
        started_at + float(timeout_seconds)
        if timeout_seconds is not None
        else None
    )
    try:
        try:
            model, status, reason, rounds = _dpll_t_run(
                formula,
                max_rounds=max_rounds,
                debug=debug,
                deadline=deadline,
                trace=trace,
                simplex_max_iter=simplex_max_iter,
                max_recursion=max_recursion,
                stats=theory_stats,
                profile_stages=profile_stages,
                split_logger=split_logger,
                progress=progress,
                round_state=round_state,
            )
        except SolverLimitReached as exc:
            if progress is not None:
                progress.abort_active_round(exc.reason)
            model = None
            status = SolverStatus.UNKNOWN
            reason = exc.reason
            # 이론 솔버 안에서 한도에 걸렸어도 그 작업은 현재 활성 DPLL(T)
            # 라운드에서 일어난 것이다. 0을 보고하면 몇 시간짜리 진행이
            # 결과에서 사라진다.
            rounds = int(round_state["current"])
        result = SolverResult(
            status=status,
            model=model,
            reason=reason,
            rounds=rounds,
            elapsed_seconds=monotonic() - started_at,
            theory_stats=theory_stats,
        )
        if progress is not None:
            progress.solver_end(
                result.status.value, result.reason, result.rounds, model=result.model,
            )
        return result
    finally:
        if visualizer is not None:
            visualizer.request_update()
            visualizer.flush()


def dpll_t(
    formula,
    max_rounds: int = 1000,
    debug: bool = False,
    trace: Optional[SolveTrace] = None,
    simplex_max_iter: int = 10000,
    max_recursion: int = 50,
    profile_stages: bool = False,
    split_mode: str = "off",
    split_log_path: Optional[str] = None,
    split_update_callback: Optional[Callable[[], None]] = None,
    realtime_output_path: Optional[str] = None,
    realtime_open_view: bool = False,
    realtime_playback_ms: int = 0,
    realtime_window_seconds: float = 300.0,
    model_layer_sizes: Optional[Iterable[int]] = None,
    progress=None,
) -> Tuple[Optional[Dict[str, float]], bool]:
    result = dpll_t_detailed(
        formula,
        max_rounds=max_rounds,
        debug=debug,
        timeout_seconds=None,
        trace=trace,
        simplex_max_iter=simplex_max_iter,
        max_recursion=max_recursion,
        profile_stages=profile_stages,
        split_mode=split_mode,
        split_log_path=split_log_path,
        split_update_callback=split_update_callback,
        realtime_output_path=realtime_output_path,
        realtime_open_view=realtime_open_view,
        realtime_playback_ms=realtime_playback_ms,
        realtime_window_seconds=realtime_window_seconds,
        model_layer_sizes=model_layer_sizes,
        progress=progress,
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
