"""Reluplex의 LP 풀이를 맡길 엔진: 자체 Simplex / GLPK / Z3.

reluplex(..., lp_engine="glpk")처럼 쓰면 노드마다 푸는 LP와 ReLU를 고친 뒤
다시 푸는 LP를 이 모듈의 엔진이 맡는다. 세 엔진은 문제를 같은 방식으로 받는다.

    engine.solve(rows, bounds, deadline) -> 모든 변수의 값, 또는 None(불가능)

rows는 정의식 [(head, {var: coeff})] 즉 head = Σ coeff·var 이고, bounds는 변수
(head 포함)의 경계다. 엔진 하나는 reluplex() 호출 하나 동안 LP 하나를 계속 들고
있으면서 바뀐 경계만 반영해 이어서 푼다(warm start).

정의식은 지우지 않는다
--------------------
head는 그 row에만 나오는 새 변수라서, head에 경계가 없으면 정의식은 아무것도
제약하지 않는다. 그래서 한 번 본 row는 엔진에 남겨 두고, 이번 호출의 rows에
없는 row는 경계만 풀어(자유) 둔다. 문제의 구조가 바뀌지 않으므로 직전 기저를
그대로 이어서 쓸 수 있다.

같은 식은 한 row로 합친다
-----------------------
계수가 부호만 반대인 두 row(등식 하나를 ≥ 두 개로 넣은 것)와 계수가 같은 row
(같은 ReLU의 y − x)는 엔진 row 하나로 합치고, head들의 경계는 부호를 맞춰
교집합한다. 등식이 ≥ 두 개로 들어가면 평행한 제약 두 개라 기저가 퇴화한다.

엔진별 차이
----------
simplex  자체 Simplex(Simplex.simplex)를 같은 방식으로 쓴다 — 엔진만 바꾼 비교의
         통제군. 부동소수 + 우리 안전장치(③).
glpk     GLPK(swiglpk). 부동소수 + GLPK 자체의 LU 분해·허용오차. 경계가 바뀐
         문제를 기저를 유지한 채 dual simplex로 다시 푼다. 목적함수가 0이라
         모든 기저가 쌍대 가능해서 dual simplex를 바로 시작할 수 있다.
z3       Z3의 QF_LRA. 계수와 경계를 float 값 그대로의 정확한 유리수로 넣고
         정확한 유리수 연산으로 푼다. 정의식은 바닥에 두고 경계만 push/pop한다.

경계 유도·repair·분기는 Reluplex.py가 하고, 엔진은 "이 경계에서 정의식을 만족하는
점이 있는가"만 답한다. tableau를 들여다보는 기법(피벗마다 경계 강화, tableau row로
경계 유도)은 이 방식에서는 쓰지 않는다.
"""
from fractions import Fraction
from math import isfinite
from threading import current_thread
from time import monotonic
from typing import Any, Dict, List, Optional, Tuple

from Automation.SolverStatus import SolverLimitReached, check_deadline

INF = float("inf")
ENGINES = ("simplex", "glpk", "z3")

RowDefs = List[Tuple[str, Dict[str, float]]]
Bounds = Dict[str, Tuple[float, float]]


def _scaled_tol(value: float, tol: float = 1e-9) -> float:
    return tol if not isfinite(value) else tol * max(1.0, abs(value))


class _Registry:
    """엔진이 들고 있는 row·열의 번호표. 세 엔진이 공유한다."""

    def __init__(self) -> None:
        self.key_index: Dict[Tuple[Tuple[str, float], ...], int] = {}
        self.row_terms: List[Tuple[Tuple[str, float], ...]] = []  # k -> 정규화한 계수
        # k -> 그 row를 처음 등록한 head 이름. 자체 Simplex는 기저변수를 이름순으로
        # 고르므로 row 이름이 피벗 순서를 바꾼다(1_1/φ3 루트 LP: 원래 이름 7.2초,
        # "__lp_row_k" 18.4초). 통제군이 원래 Reluplex와 같은 순서로 피벗하도록
        # 원래 head 이름을 그대로 쓴다.
        self.row_name: List[str] = []
        self.head: Dict[str, Tuple[int, int, Dict[str, float]]] = {}  # head -> (k, 부호, 원래 terms)
        self.columns: Dict[str, int] = {}          # 열 변수 -> 0부터 번호
        self.column_names: List[str] = []

    def register(self, rows: RowDefs) -> Tuple[List[int], List[str]]:
        """처음 보는 row와 열을 등록하고 (새 row 번호들, 새 열 이름들)을 돌려준다."""
        new_rows: List[int] = []
        new_cols: List[str] = []
        for head, terms in rows:
            known = self.head.get(head)
            if known is not None:
                if known[2] is not terms and known[2] != terms:
                    raise ValueError(f"같은 head에 다른 정의식이 들어왔다: {head}")
                continue
            if head in self.columns:
                raise ValueError(f"head가 다른 row의 변수로도 쓰였다: {head}")
            items = sorted((v, float(c)) for v, c in terms.items() if c != 0.0)
            sign = 1 if not items or items[0][1] > 0 else -1
            key = tuple((v, c * sign) for v, c in items)
            k = self.key_index.get(key)
            if k is None:
                k = len(self.row_terms)
                self.key_index[key] = k
                self.row_terms.append(key)
                self.row_name.append(head)
                new_rows.append(k)
                for v, _ in key:
                    if v not in self.columns:
                        if v in self.head:
                            raise ValueError(f"head가 다른 row의 변수로도 쓰였다: {v}")
                        self.columns[v] = len(self.column_names)
                        self.column_names.append(v)
                        new_cols.append(v)
            self.head[head] = (k, sign, terms)
        return new_rows, new_cols

    def row_bounds(self, rows: RowDefs, bounds: Bounds) -> Dict[int, Tuple[float, float]]:
        """이번 호출에 들어온 row만, head 경계를 엔진 row 경계로 옮겨 교집합한다."""
        out: Dict[int, Tuple[float, float]] = {}
        for head, _ in rows:
            k, sign, _terms = self.head[head]
            lo, hi = bounds.get(head, (-INF, INF))
            if sign < 0:
                lo, hi = -hi, -lo
            plo, phi = out.get(k, (-INF, INF))
            out[k] = (max(plo, lo), min(phi, hi))
        return out


def _fix_crossed(lo: float, hi: float) -> Optional[Tuple[float, float]]:
    """lo > hi면: 허용오차 안이면 가운데 한 점으로, 밖이면 None(모순)."""
    if lo <= hi:
        return lo, hi
    if lo - hi <= _scaled_tol(max(abs(lo), abs(hi))):
        mid = 0.5 * (lo + hi)
        return mid, mid
    return None


class _EngineBase:
    name = "base"

    def __init__(self) -> None:
        self.reg = _Registry()
        self.calls = 0
        self.seconds = 0.0

    def _assemble(
        self,
        rows: RowDefs,
        bounds: Bounds,
        col_values: Dict[str, float],
        row_values: Dict[int, float],
    ) -> Dict[str, float]:
        assign = dict(col_values)
        for head, _ in rows:
            k, sign, _terms = self.reg.head[head]
            assign[head] = sign * row_values[k]
        for var, (lo, hi) in bounds.items():
            if var not in assign:
                assign[var] = min(max(0.0, lo), hi)   # 어느 식에도 없는 변수
        return assign

    def _prepare(self, rows: RowDefs, bounds: Bounds):
        """경계를 정리한다. 모순이면 None."""
        row_b = self.reg.row_bounds(rows, bounds)
        for k, (lo, hi) in list(row_b.items()):
            fixed = _fix_crossed(lo, hi)
            if fixed is None:
                return None
            row_b[k] = fixed
        col_b: Dict[str, Tuple[float, float]] = {}
        for v in self.reg.column_names:
            lo, hi = bounds.get(v, (-INF, INF))
            fixed = _fix_crossed(lo, hi)
            if fixed is None:
                return None
            col_b[v] = fixed
        for var, (lo, hi) in bounds.items():
            if var not in self.reg.columns and var not in self.reg.head:
                if _fix_crossed(lo, hi) is None:
                    return None
        return row_b, col_b

    def solve(self, rows: RowDefs, bounds: Bounds,
              deadline: Optional[float] = None) -> Optional[Dict[str, float]]:
        check_deadline(deadline)
        started = monotonic()
        try:
            new_rows, new_cols = self.reg.register(rows)
            prepared = self._prepare(rows, bounds)
            if prepared is None:
                return None
            row_b, col_b = prepared
            result = self._solve(new_rows, new_cols, row_b, col_b, deadline)
            if result is None:
                return None
            col_values, row_values = result
            return self._assemble(rows, bounds, col_values, row_values)
        finally:
            self.calls += 1
            self.seconds += monotonic() - started

    def _solve(self, new_rows, new_cols, row_b, col_b, deadline):
        raise NotImplementedError


# ─────────────────────────────────────────────
#  자체 Simplex (통제군)
# ─────────────────────────────────────────────

class SimplexEngine(_EngineBase):
    """Simplex.simplex를 GLPK·Z3와 같은 방식(LP 하나를 계속 들고 경계만 바꿈)으로 쓴다."""

    name = "simplex"

    def __init__(self, simplex_kwargs: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()
        self.simplex_kwargs = dict(simplex_kwargs or {})
        self.tableau = None

    def _row_name(self, k: int) -> str:
        # tableau 안에서 row k의 값은 정규화한 식의 값이다(부호는 _Registry.head가 안다)
        return self.reg.row_name[k]

    def _solve(self, new_rows, new_cols, row_b, col_b, deadline):
        from Simplex import Bound, Row, _compute_basic, build_tableau, simplex

        names = self.reg.column_names
        if self.tableau is None:
            defs = [(self._row_name(k), dict(self.reg.row_terms[k])) for k in new_rows]
            bnds = {v: col_b[v] for v in names}
            for k in range(len(self.reg.row_terms)):
                bnds[self._row_name(k)] = row_b.get(k, (-INF, INF))
            self.tableau = build_tableau(defs, bnds)
        else:
            tab = self.tableau
            for v in new_cols:
                lo, hi = col_b[v]
                tab.bounds[v] = Bound(lower=lo, upper=hi)
                tab.assign[v] = min(max(0.0, lo), hi)
            basic_rows = {row.basic_var: row for row in tab.rows}
            for k in new_rows:
                terms: Dict[str, float] = {}
                for var, coeff in self.reg.row_terms[k]:
                    row = basic_rows.get(var)
                    if row is None:
                        terms[var] = terms.get(var, 0.0) + coeff
                    else:
                        for nv, cc in row.coeffs.items():
                            terms[nv] = terms.get(nv, 0.0) + coeff * cc
                name = self._row_name(k)
                new = Row(basic_var=name, coeffs={v: c for v, c in terms.items() if c != 0.0})
                tab.rows.append(new)
                tab.origin_rows.append((name, dict(self.reg.row_terms[k])))
                tab.bounds[name] = Bound()
                tab.assign[name] = 0.0
            # 경계를 맞추고 비기저변수를 범위 안으로 당긴 뒤 기저변수를 다시 계산
            basic = {row.basic_var for row in tab.rows}
            wanted = {v: col_b[v] for v in names}
            for k in range(len(self.reg.row_terms)):
                wanted[self._row_name(k)] = row_b.get(k, (-INF, INF))
            for var, (lo, hi) in wanted.items():
                b = tab.bounds[var]
                b.lower, b.upper = lo, hi
                if var not in basic:
                    value = tab.assign[var]
                    if value < lo:
                        tab.assign[var] = lo
                    elif value > hi:
                        tab.assign[var] = hi
            for row in tab.rows:
                tab.assign[row.basic_var] = _compute_basic(tab, row)

        sol, sat = simplex(self.tableau, deadline=deadline, **self.simplex_kwargs)
        if not sat:
            return None
        cols = {v: sol[v] for v in names}
        rows = {k: sol[self._row_name(k)] for k in range(len(self.reg.row_terms))}
        return cols, rows


# ─────────────────────────────────────────────
#  GLPK
# ─────────────────────────────────────────────

class _GlpkProblem:
    def _open_problem(self):
        import swiglpk as glp

        self.glp = glp
        self._owner_thread = current_thread()
        self.lp = glp.glp_create_prob()
        glp.glp_set_obj_dir(self.lp, glp.GLP_MIN)
        return glp

    def close(self) -> None:
        if getattr(self, "lp", None) is None:
            return
        if current_thread() is not self._owner_thread:
            raise RuntimeError("GLPK problems must be closed by their creating thread")
        lp, self.lp = self.lp, None
        self.glp.glp_delete_prob(lp)

    def __del__(self) -> None:
        # A cyclic search frame may be collected by the visualization thread.
        # GLPK's allocator uses thread-local bookkeeping.
        if current_thread() is getattr(self, "_owner_thread", None):
            self.close()


class GlpkEngine(_EngineBase, _GlpkProblem):
    """GLPK 5.0 (swiglpk) — 정의식과 경계만 받아 답하는 방식(oracle)."""

    name = "glpk"

    def __init__(self, tol_bnd: float = 1e-10, method: str = "dual") -> None:
        super().__init__()
        glp = self._open_problem()
        self.parm = _glpk_parm(glp, tol_bnd, method)
        self._row_set: Dict[int, Tuple[int, float, float]] = {}
        self._col_set: Dict[int, Tuple[int, float, float]] = {}
        self.failures = 0

    def _solve(self, new_rows, new_cols, row_b, col_b, deadline):
        glp, lp = self.glp, self.lp
        if new_cols:
            glp.glp_add_cols(lp, len(new_cols))     # 새 열은 비기저(0에 고정)로 들어온다
        if new_rows:
            first = glp.glp_add_rows(lp, len(new_rows))   # 새 row는 기저로 들어온다
            cols = self.reg.columns
            for offset, k in enumerate(new_rows):
                _set_glpk_row(glp, lp, first + offset,
                              [(cols[v] + 1, c) for v, c in self.reg.row_terms[k]])
        # 바뀐 경계만 넘긴다
        for k in range(len(self.reg.row_terms)):
            want = _glpk_bnd(glp, *row_b.get(k, (-INF, INF)))
            if self._row_set.get(k) != want:
                glp.glp_set_row_bnds(lp, k + 1, *want)
                self._row_set[k] = want
        for j, v in enumerate(self.reg.column_names):
            want = _glpk_bnd(glp, *col_b[v])
            if self._col_set.get(j) != want:
                glp.glp_set_col_bnds(lp, j + 1, *want)
                self._col_set[j] = want

        status = _glpk_run(self, deadline)
        if status is None:
            return None
        cols = {v: glp.glp_get_col_prim(lp, j + 1)
                for j, v in enumerate(self.reg.column_names)}
        rows = {k: glp.glp_get_row_prim(lp, k + 1) for k in row_b}
        for k in range(len(self.reg.row_terms)):
            rows.setdefault(k, 0.0)
        return cols, rows


def _glpk_parm(glp, tol_bnd: float, method: str):
    parm = glp.glp_smcp()
    glp.glp_init_smcp(parm)
    parm.msg_lev = glp.GLP_MSG_OFF
    parm.meth = {"dual": glp.GLP_DUALP, "primal": glp.GLP_PRIMAL, "dual-only": glp.GLP_DUAL}[method]
    parm.presolve = glp.GLP_OFF        # presolve를 켜면 기저를 이어 쓰지 못한다
    parm.tol_bnd = tol_bnd             # 우리 검산 허용오차(1e-9)보다 촘촘하게
    return parm


def _glpk_bnd(glp, lo: float, hi: float) -> Tuple[int, float, float]:
    if lo == -INF and hi == INF:
        return glp.GLP_FR, 0.0, 0.0
    if lo == -INF:
        return glp.GLP_UP, 0.0, hi
    if hi == INF:
        return glp.GLP_LO, lo, 0.0
    if lo == hi:
        return glp.GLP_FX, lo, hi
    return glp.GLP_DB, lo, hi


def _set_glpk_row(glp, lp, i: int, terms: List[Tuple[int, float]]) -> None:
    n = len(terms)
    ind = glp.intArray(n + 1)
    val = glp.doubleArray(n + 1)
    for pos, (j, c) in enumerate(terms, start=1):
        ind[pos] = j
        val[pos] = c
    glp.glp_set_mat_row(lp, i, n, ind, val)


def _glpk_run(owner, deadline: Optional[float], parm=None) -> Optional[int]:
    """glp_simplex를 한 번 부른다. 가능이면 상태, 불가능이면 None.
    기저가 특이해지는 등 실패하면 GLPK의 crash 기저에서 한 번 더 푼다."""
    glp, lp = owner.glp, owner.lp
    parm = parm or owner.parm
    for _attempt in range(2):
        if deadline is not None:
            remaining = deadline - monotonic()
            if remaining <= 0:
                raise SolverLimitReached("TIMEOUT")
            parm.tm_lim = max(1, int(remaining * 1000))
        ret = glp.glp_simplex(lp, parm)
        if ret == glp.GLP_ETMLIM:
            raise SolverLimitReached("TIMEOUT")
        if ret == 0:
            status = glp.glp_get_status(lp)
            if status in (glp.GLP_OPT, glp.GLP_FEAS):
                return status
            if status == glp.GLP_NOFEAS:
                return None
        owner.failures += 1
        glp.glp_adv_basis(lp, 0)
    raise SolverLimitReached("LP_ENGINE_FAILURE")


# ─────────────────────────────────────────────
#  Z3
# ─────────────────────────────────────────────

def _q(value: float) -> str:
    """float 값 그대로의 정확한 유리수 (Z3 numeral 문자열)."""
    n, d = float(value).as_integer_ratio()
    return f"{n}/{d}" if d != 1 else str(n)


class Z3Engine(_EngineBase):
    name = "z3"

    def __init__(self) -> None:
        super().__init__()
        import z3
        self.z3 = z3
        self.solver = z3.SolverFor("QF_LRA")
        self.col_var: List[Any] = []
        self.row_var: List[Any] = []
        self._num_cache: Dict[float, Any] = {}
        self.check_seconds = 0.0     # s.check() 자체에 쓴 시간 (나머지는 파이썬 쪽 비용)

    def _num(self, value: float):
        num = self._num_cache.get(value)
        if num is None:
            num = self.z3.RealVal(_q(value))
            if len(self._num_cache) < 200000:
                self._num_cache[value] = num
        return num

    def _solve(self, new_rows, new_cols, row_b, col_b, deadline):
        z3, s = self.z3, self.solver
        for v in new_cols:
            self.col_var.append(z3.Real(v))
        for k in new_rows:
            r = z3.Real(f"__lp_row_{k}")
            self.row_var.append(r)
            terms = self.reg.row_terms[k]
            expr = z3.Sum([self._num(c) * self.col_var[self.reg.columns[v]] for v, c in terms]) \
                if terms else z3.RealVal(0)
            s.add(r == expr)

        s.push()
        try:
            cons = []
            for j, v in enumerate(self.reg.column_names):
                self._bound_atoms(cons, self.col_var[j], *col_b[v])
            for k, (lo, hi) in row_b.items():
                self._bound_atoms(cons, self.row_var[k], lo, hi)
            s.add(cons)
            if deadline is not None:
                remaining = deadline - monotonic()
                if remaining <= 0:
                    raise SolverLimitReached("TIMEOUT")
                s.set("timeout", max(1, int(remaining * 1000)))
            started = monotonic()
            res = s.check()
            self.check_seconds += monotonic() - started
            if res == z3.unsat:
                return None
            if res != z3.sat:
                check_deadline(deadline)
                raise SolverLimitReached("LP_ENGINE_UNKNOWN")
            values = self._values(s.model())
            names = self.reg.column_names
            cols = {v: values[j] for j, v in enumerate(names)}
            # row 값은 열 값으로 다시 계산하지 않고 Z3의 정확한 값을 쓴다
            rows = {k: values[len(names) + k] for k in range(len(self.row_var))}
            return cols, rows
        finally:
            s.pop()

    def _bound_atoms(self, out: List[Any], var, lo: float, hi: float) -> None:
        if lo == hi:
            out.append(var == self._num(lo))
            return
        if lo != -INF:
            out.append(var >= self._num(lo))
        if hi != INF:
            out.append(var <= self._num(hi))

    def _values(self, m) -> List[float]:
        """열 값들, 이어서 row 값들. 래퍼 객체를 만들지 않고 C API로 바로 읽는다
        (m.eval + numerator/denominator로 읽으면 변수 950개에 약 0.1초가 든다)."""
        z3core = self.z3.z3core
        ctx = self.solver.ctx.ref()
        model = m.model
        out = []
        result = (z3core.Ast * 1)()
        for var in self.col_var + self.row_var:
            if not z3core.Z3_model_eval(ctx, model, var.as_ast(), True, result):
                raise SolverLimitReached("LP_ENGINE_FAILURE")
            text = z3core.Z3_get_numeral_string(ctx, result[0])
            num, _, den = text.partition("/")
            out.append(int(num) / int(den) if den else float(int(num)))
        return out


# ─────────────────────────────────────────────
#  코어 방식: 카츠 Reluplex의 구조 (lp_mode="core")
# ─────────────────────────────────────────────
#
# Reluplex가 자기 tableau(Simplex.SimplexTableau)를 계속 들고 있고, 그 tableau의
# "경계 위반 고치기"(Simplex.simplex 자리)만 엔진에 맡긴다. 그래서 native 경로의
# 기법이 전부 그대로 돈다: 부모 tableau 물려받기, tableau row로 경계 유도, tableau
# 위의 Update / PivotForRelu, 충돌 원인(백점프), 과소근사. 엔진 안에서 하던 일은
# 엔진마다 대응 기법으로 바꾼다.
#
#               ① 피벗마다 경계 강화          ② UNSAT의 원인          ③ 부동소수
#   native      자체 피벗마다                 막힌 row                 큰 계수 피벗·오차 복원
#   GlpkCore    GLPK를 피벗 1회씩 돌리며       dual simplex가 찾은       GLPK LU 분해, 부를 때마다
#               진입 변수를 그 표 row로        막힌 row(unbnd ray)       표를 새로 가져옴(오차 0)
#   Z3Core      피벗이 밖으로 안 보인다 →      unsat core에 든 경계     정확한 유리수
#               풀 때마다 tableau 전체 row로
#
# simplex(tableau, ...)는 Simplex.simplex와 같은 인자를 받고 같은 것을 돌려준다:
# (할당, True) 또는 (None, False), UNSAT이면 tableau.conflict에 원인 마스크.


# GLPK 표 row에서 0으로 볼 계수의 상대 임계값 (Simplex.simplex의 ZERO_REL과 같다)
TAB_ZERO_REL = 1e-12


def _uncross(lo: float, hi: float) -> Tuple[float, float]:
    """경계 강화·유도는 허용오차 안에서 lo > hi가 된 경계를 모순으로 보지 않고 남긴다.
    자체 simplex는 허용오차로 넘어가지만 GLPK는 GLP_EBOUND로 거부하므로, GLPK에
    넘길 때는 가운데 한 점으로 고정한다."""
    if lo > hi:
        mid = 0.5 * (lo + hi)
        return mid, mid
    return lo, hi


def _snap_status(glp, value: float, lo: float, hi: float, basic: bool) -> int:
    """우리 tableau의 변수 상태를 GLPK 상태로. GLPK의 비기저변수는 경계에 있어야
    하므로 경계 사이에 있으면 가까운 경계로 옮긴다(카츠 GlpkWrapper와 같다)."""
    if basic:
        return glp.GLP_BS
    if lo >= hi:
        return glp.GLP_NS
    if lo == -INF and hi == INF:
        return glp.GLP_NF
    if lo == -INF:
        return glp.GLP_NU
    if hi == INF:
        return glp.GLP_NL
    return glp.GLP_NL if value - lo <= hi - value else glp.GLP_NU


def _row_error(defs: Dict[str, Dict[str, float]], basic: str, coeffs: Dict[str, float],
               mag) -> float:
    """표 row `basic = sum(c*v)`가 원래 등식에서 벗어난 정도의 상한 (③ 대응).

    GLPK의 표 row는 LU 분해로 계산해 계수마다 반올림 오차가 있다(1_9/φ3에서 상대
    7.7e-12까지 쟀다). 원래 0이어야 할 계수가 1.7e-12로 남으면 경계 유도가 그 계수로
    나눠 오차를 1e12배 키우고, 없는 모순을 만든다(1_9/φ3이 SAT인데 UNSAT으로 판정됐다).
    그래서 row의 슬랙(원래 등식의 head)을 그 정의로 펼쳐 '0이어야 할 나머지'를 직접
    잰다: 나머지 계수 r_x에 변수 x의 크기 상한 mag(x)를 곱해 더한 값이 이 row가 경계
    안의 어느 점에서든 틀릴 수 있는 최대치다. 펼치는 계산 자체의 반올림도 더한다.
    """
    res: Dict[str, float] = {}
    size: Dict[str, float] = {}
    for v, c in [(basic, 1.0)] + [(v, -c) for v, c in coeffs.items()]:
        dv = defs.get(v)
        if dv is None:
            res[v] = res.get(v, 0.0) + c
            size[v] = size.get(v, 0.0) + abs(c)
            continue
        for x, a in dv.items():
            p = c * a
            res[x] = res.get(x, 0.0) + p
            size[x] = size.get(x, 0.0) + abs(p)
    err = 0.0
    for x, r in res.items():
        e = abs(r) + 1e-15 * size[x]
        if e:
            err += e * mag(x)
    return err


def _bound_mag(bounds):
    """변수 -> |값|의 상한. bounds는 tableau.bounds(Bound) 또는 {변수: (lo, hi)}."""
    def mag(x: str) -> float:
        b = bounds.get(x)
        if b is None:
            return INF
        lo, hi = (b.lower, b.upper) if hasattr(b, "lower") else b
        return max(abs(lo), abs(hi))
    return mag


class GlpkCore(_GlpkProblem):
    """카츠 Reluplex처럼 GLPK를 '경계 위반 고치기' 엔진으로 쓴다.

    부를 때마다 (1) tableau의 원래 등식·경계·기저를 GLPK에 넘기고, (2) GLPK가 그
    기저에서 이어서 풀며 피벗마다 진입 변수의 경계를 그 표 row로 좁히고
    (full_tighten_every 피벗마다 모든 기저변수, 논문 5절), (3) 다 풀면 GLPK의 기저와
    표(glp_eval_tab_row)와 값을 tableau로 가져온다. 표를 GLPK의 LU 분해에서 새로
    가져오므로 tableau에 쌓인 반올림 오차도 매번 사라진다(③ 오차 복원에 해당).
    카츠는 GLPK 소스를 고쳐 피벗마다 콜백을 걸었고, 여기서는 고치지 않은 GLPK를
    한 번에 피벗 1회씩(it_lim=1) 돌린다.

    GLPK 문제는 한 reluplex() 호출 동안 하나를 유지한다. 깊이 우선 탐색이라 앞 노드와
    row 목록의 앞부분이 같으므로, 다른 부분(뒤쪽)만 지우고 더한다.

    extract: 다 푼 뒤 tableau를 GLPK에 맞추는 방법.
      "full"  — 기저 row 전부를 glp_eval_tab_row로 새로 가져온다(위 (3)). row 하나에
                LU로 btran 한 번이라 row 수(ACAS Xu 약 700)만큼 C 계산이 든다.
      "basis" — 카츠 Reluplex의 기본값(_glpkExtractJustBasics). 기저 목록만 받아
                우리 tableau를 바뀐 기저만큼 피벗해 맞추고(LP 한 번에 몇 번), 값은
                비기저 값만 GLPK에서 받아 기저변수를 우리 row로 다시 계산한다. 피벗이
                쌓이므로 ③은 native와 같이 DRIFT_CHECK_EVERY 피벗마다 원래 등식 대비
                오차를 재서 DRIFT_TOL을 넘으면 "full"로 표를 새로 가져온다(카츠의
                restore에 해당; 기저는 유지되고 오차만 사라진다). 맞춘 값이 경계를
                벗어나도 "full"로 가져온다.

    row_error: 노드 경계 유도에 넘기는 tableau row에 오차 상한(_row_error)을 붙이는 때.
      "auto"   — 표 row를 GLPK에서 가져온 적이 있는 tableau(tableau.imported_rows)만.
                 "full"은 LP마다 가져오므로 늘 붙고, "basis"는 우리 피벗만 거친 row라
                 native·카츠처럼 붙이지 않는다(가져오기로 돌아간 뒤부터는 붙인다).
      "always" — 늘 붙인다(code9까지의 동작, 비교용).
    피벗마다 좁히기(_tighten)와 불가능 원인(_unsat)은 GLPK 표 row를 직접 쓰므로 늘 붙인다.
    """

    name = "glpk"
    # extract="basis"에서 이보다 작은 계수로는 피벗하지 않는다 (카츠 Reluplex의
    # NUMBERICAL_INSTABILITY_CONSTANT와 같다). 못 한 피벗은 기저가 GLPK와 하나 다르게
    # 남을 뿐이고, tableau는 여전히 맞는 tableau다.
    ADOPT_PIVOT_MIN = 1e-4

    def __init__(self, full_tighten_every: int = 1000, tol_bnd: float = 1e-10,
                 extract: str = "full", row_error: str = "auto") -> None:
        if extract not in ("full", "basis"):
            raise ValueError(f"알 수 없는 extract: {extract} (가능: full, basis)")
        if row_error not in ("auto", "always"):
            raise ValueError(f"알 수 없는 row_error: {row_error} (가능: auto, always)")
        self.extract = extract
        self.row_error = row_error
        glp = self._open_problem()
        self.parm = _glpk_parm(glp, tol_bnd, "primal")
        self.dual_parm = _glpk_parm(glp, tol_bnd, "dual-only")
        self.row_heads: List[str] = []
        self.col_index: Dict[str, int] = {}     # 변수 -> GLPK 열 번호(1부터)
        self.col_names: List[str] = []
        self.full_tighten_every = full_tighten_every
        self.failures = 0
        self.pivots = 0
        self.tightened = 0
        self.conflicts_found = 0
        self._buf_n = -1

    # ── tableau → GLPK ──
    def _sync(self, tableau) -> None:
        glp, lp = self.glp, self.lp
        heads = [h for h, _ in tableau.origin_rows]
        keep = 0
        for a, b in zip(self.row_heads, heads):
            if a != b:
                break
            keep += 1
        if keep < len(self.row_heads):
            drop = len(self.row_heads) - keep
            num = glp.intArray(drop + 1)
            for pos in range(1, drop + 1):
                num[pos] = keep + pos
            glp.glp_del_rows(lp, drop, num)
        new = tableau.origin_rows[keep:]
        new_cols = []
        for _h, terms in new:
            for v in terms:
                if v not in self.col_index:
                    self.col_index[v] = len(self.col_names) + 1
                    self.col_names.append(v)
                    new_cols.append(v)
        if new_cols:
            glp.glp_add_cols(lp, len(new_cols))
        if new:
            first = glp.glp_add_rows(lp, len(new))
            for offset, (_h, terms) in enumerate(new):
                _set_glpk_row(glp, lp, first + offset,
                              [(self.col_index[v], c) for v, c in terms.items() if c != 0.0])
        self.row_heads = heads
        self._row_pos = {h: i for i, h in enumerate(heads, start=1)}
        self._defs = dict(tableau.origin_rows)
        # 경계와 기저 상태: tableau 그대로
        basic = {row.basic_var for row in tableau.rows}
        bounds, assign = tableau.bounds, tableau.assign
        for i, h in enumerate(heads, start=1):
            lo, hi = _uncross(bounds[h].lower, bounds[h].upper)
            glp.glp_set_row_bnds(lp, i, *_glpk_bnd(glp, lo, hi))
            glp.glp_set_row_stat(lp, i, _snap_status(glp, assign[h], lo, hi, h in basic))
        for j, v in enumerate(self.col_names, start=1):
            b = bounds.get(v)
            if b is None:
                glp.glp_set_col_bnds(lp, j, glp.GLP_FR, 0.0, 0.0)
                glp.glp_set_col_stat(lp, j, glp.GLP_NF)
                continue
            lo, hi = _uncross(b.lower, b.upper)
            glp.glp_set_col_bnds(lp, j, *_glpk_bnd(glp, lo, hi))
            glp.glp_set_col_stat(lp, j, _snap_status(glp, assign.get(v, 0.0), lo, hi, v in basic))

    def _name(self, idx: int, m: int) -> str:
        return self.row_heads[idx - 1] if idx <= m else self.col_names[idx - m - 1]

    def _tab_row(self, idx: int, m: int) -> Dict[str, float]:
        """GLPK 변수 idx(기저)의 표 row를 이름으로: {비기저변수: 계수}."""
        import ctypes

        glp = self.glp
        n = len(self.col_names)
        if self._buf_n != n:
            self._ind = glp.intArray(n + 1)
            self._val = glp.doubleArray(n + 1)
            self._ind_buf = (ctypes.c_int * (n + 1)).from_address(int(self._ind.this))
            self._val_buf = (ctypes.c_double * (n + 1)).from_address(int(self._val.this))
            self._buf_n = n
        length = glp.glp_eval_tab_row(self.lp, idx, self._ind, self._val)
        heads, cols = self.row_heads, self.col_names
        pairs = list(zip(self._ind_buf[1:length + 1], self._val_buf[1:length + 1]))
        # ③ 대응: GLPK의 표 row에는 정확히 0이어야 할 계수가 1e-16 근처의 반올림 잡음으로
        # 남는다. 경계 유도는 row를 그 변수에 대해 풀 때 계수로 나누므로, 2.6e-17로
        # 나누면 오차가 1e17배가 되어 틀린 경계와 가짜 모순이 나온다(1_7/φ3이 SAT인데
        # UNSAT으로 판정됐다). 자체 simplex가 피벗마다 하는 것과 같이 row 최대 계수의
        # TAB_ZERO_REL배 이하는 0으로 지운다.
        amax = max((abs(a) for _j, a in pairs), default=0.0)
        thr = TAB_ZERO_REL * amax
        return {
            (heads[j - 1] if j <= m else cols[j - m - 1]): a
            for j, a in pairs
            if abs(a) > thr
        }

    def _push_bound(self, name: str, b) -> None:
        want = _glpk_bnd(self.glp, *_uncross(b.lower, b.upper))
        i = self._row_pos.get(name)
        if i is not None:
            self.glp.glp_set_row_bnds(self.lp, i, *want)
        else:
            self.glp.glp_set_col_bnds(self.lp, self.col_index[name], *want)

    def _tighten(self, tableau, idx: int, m: int, stats) -> Optional[bool]:
        """GLPK 기저변수 idx의 경계를 그 표 row로 좁힌다 (Simplex._tighten_from_row 그대로 —
        경계의 원인 마스크도 같이 붙인다). row의 오차(_row_error)만큼 넓혀서 좁힌다.
        좁아지면 GLPK에도 넣는다."""
        from Simplex import Row, _tighten_from_row

        name = self._name(idx, m)
        coeffs = self._tab_row(idx, m)
        err = _row_error(self._defs, name, coeffs, _bound_mag(tableau.bounds))
        r = _tighten_from_row(tableau, Row(basic_var=name, coeffs=coeffs), err=err)
        if r:
            self.tightened += 1
            if stats is not None:
                stats["PIVOT_TIGHTENED"] = stats.get("PIVOT_TIGHTENED", 0) + 1
            self._push_bound(name, tableau.bounds[name])
        return r

    def needs_row_error(self, tableau) -> bool:
        """노드 경계 유도에 이 tableau의 row를 오차 상한과 같이 넘겨야 하나 (row_error 참고)."""
        return self.row_error == "always" or tableau.imported_rows

    @staticmethod
    def rows_with_error(tableau, bounds) -> List[Tuple[str, Dict[str, float], float]]:
        """경계 유도용 (기저변수, 계수, 오차 상한) row 목록. 오차는 bounds 위에서 잰다.
        tableau.rows는 GLPK에서 가져온 뒤 repair 피벗(PivotForRelu)을 거쳤을 수 있지만,
        원래 등식으로 펼쳐 재므로 그대로 맞다."""
        defs = dict(tableau.origin_rows)
        mag = _bound_mag(bounds)
        return [(row.basic_var, row.coeffs, _row_error(defs, row.basic_var, row.coeffs, mag))
                for row in tableau.rows]

    def _basis(self, m: int) -> set:
        bhead = self.glp.glp_get_bhead
        return {bhead(self.lp, i) for i in range(1, m + 1)}

    # ── 메인: Simplex.simplex 자리 ──
    def simplex(self, tableau, max_iter: int = 10000, debug: bool = False, *,
                deadline: Optional[float] = None, report_unknown: bool = False,
                progress=None, progress_context=None, stats=None,
                tighten_bounds: bool = False, full_tighten_every: int = 0,
                numeric_safeguards: bool = True):
        glp, lp = self.glp, self.lp
        check_deadline(deadline)
        tableau.conflict = None
        self._sync(tableau)
        m = glp.glp_get_num_rows(lp)
        full_every = full_tighten_every or 0

        status = None
        if tighten_bounds:
            # 피벗 1회씩: 진입 변수(기저에 새로 들어온 변수)를 그 표 row로 좁힌다
            parm = self.parm
            saved = parm.it_lim
            parm.it_lim = 1
            try:
                basis = self._basis(m) if glp.glp_bf_exists(lp) else None
                stalls = 0
                while True:
                    if deadline is not None:
                        remaining = deadline - monotonic()
                        if remaining <= 0:
                            raise SolverLimitReached("TIMEOUT")
                        parm.tm_lim = max(1, int(remaining * 1000))
                    ret = glp.glp_simplex(lp, parm)
                    if ret == glp.GLP_ETMLIM:
                        raise SolverLimitReached("TIMEOUT")
                    if ret not in (0, glp.GLP_EITLIM):
                        break                    # 아래에서 강화 없이 한 번에 푼다
                    new = self._basis(m)
                    entering = (new - basis) if basis is not None else set()
                    basis = new
                    if entering:
                        stalls = 0
                        self.pivots += 1
                        for k in entering:
                            if self._tighten(tableau, k, m, stats) is None:
                                return self._unsat_from_tighten(stats)
                        if full_every and self.pivots % full_every == 0:
                            if stats is not None:
                                stats["SIMPLEX_FULL_TIGHTEN"] = stats.get("SIMPLEX_FULL_TIGHTEN", 0) + 1
                            for k in new:
                                if self._tighten(tableau, k, m, stats) is None:
                                    return self._unsat_from_tighten(stats)
                    else:
                        stalls += 1
                    if ret == 0:
                        s = glp.glp_get_status(lp)
                        if s in (glp.GLP_OPT, glp.GLP_FEAS):
                            status = s
                        elif s == glp.GLP_NOFEAS:
                            return self._unsat(tableau, m, deadline)
                        break
                    if stalls > 20:
                        break
            finally:
                parm.it_lim = saved
            if status is None:
                self.failures += 1
        if status is None:
            status = _glpk_run(self, deadline)
            if status is None:
                return self._unsat(tableau, m, deadline)
        started = monotonic()
        if self.extract == "basis":
            self._adopt(tableau, m, stats)
        else:
            self._import(tableau, m)
        if stats is not None:
            stats["GLPK_EXTRACT_SECONDS"] = stats.get("GLPK_EXTRACT_SECONDS", 0.0) + monotonic() - started
        return dict(tableau.assign), True

    def _unsat_from_tighten(self, stats):
        if stats is not None:
            stats["PIVOT_TIGHTEN_UNSAT"] = stats.get("PIVOT_TIGHTEN_UNSAT", 0) + 1
        return None, False             # 원인은 _tighten_from_row가 tableau.conflict에 남겼다

    def _unsat(self, tableau, m: int, deadline):
        """불가능: 원인 마스크를 tableau.conflict에 남긴다(백점프용, deps가 있을 때).

        primal이 불가능이라고 하면 dual simplex로 한 번 더 풀어 막힌 row 하나를 얻는다
        (dual은 '나갈 변수의 row에 들어올 변수가 없다'로 불가능을 판정하고,
        glp_get_unbnd_ray가 그 기저변수를 알려준다). 그 row의 비기저변수가 모두 막는
        쪽 경계에 있어도 기저변수의 경계에 못 미치는지 확인한 뒤, native와 같은
        _blocked_row_conflict로 마스크를 만든다. 확인이 안 되면 원인을 모른다(None) —
        Reluplex는 그때 경로의 분기 전부를 원인으로 본다(보수적).
        """
        from Simplex import Row, _blocked_row_conflict

        if tableau.deps is None:
            return None, False
        glp, lp = self.glp, self.lp
        if deadline is not None:
            self.dual_parm.tm_lim = max(1, int(max(0.0, deadline - monotonic()) * 1000))
        glp.glp_simplex(lp, self.dual_parm)
        k = glp.glp_get_unbnd_ray(lp)
        if k <= 0 or glp.glp_get_status(lp) != glp.GLP_NOFEAS:
            return None, False
        is_basic = (glp.glp_get_row_stat(lp, k) if k <= m
                    else glp.glp_get_col_stat(lp, k - m)) == glp.GLP_BS
        if not is_basic:
            return None, False
        name = self._name(k, m)
        coeffs = self._tab_row(k, m)
        value = glp.glp_get_row_prim(lp, k) if k <= m else glp.glp_get_col_prim(lp, k - m)
        b = tableau.bounds[name]
        going_up = value < b.lower
        if not going_up and not value > b.upper:
            return None, False
        # 막는 쪽 경계로 만든 row의 최댓값(최솟값)이 기저변수의 경계에 못 미쳐야 증거다
        reach = 0.0
        for v, a in coeffs.items():
            vb = tableau.bounds[v]
            if going_up:
                edge = vb.upper if a > 0 else vb.lower
            else:
                edge = vb.lower if a > 0 else vb.upper
            if not isfinite(edge):
                return None, False
            reach += a * edge
        tol = 1e-9 * max(1.0, abs(reach)) + _row_error(
            self._defs, name, coeffs, _bound_mag(tableau.bounds))
        if (going_up and reach >= b.lower - tol) or (not going_up and reach <= b.upper + tol):
            return None, False
        amax = max((abs(a) for a in coeffs.values()), default=0.0)
        tableau.conflict = _blocked_row_conflict(
            tableau, Row(basic_var=name, coeffs=coeffs), going_up, max(1e-12 * amax, 1e-300)
        )
        self.conflicts_found += 1
        return None, False

    # ── GLPK → tableau ──
    def _import(self, tableau, m: int) -> None:
        from Simplex import Row

        glp, lp = self.glp, self.lp
        bhead = glp.glp_get_bhead
        rows = []
        for i in range(1, m + 1):
            k = bhead(lp, i)
            rows.append(Row(basic_var=self._name(k, m), coeffs=self._tab_row(k, m)))
        tableau.rows = rows
        assign = tableau.assign
        for i, h in enumerate(self.row_heads, start=1):
            assign[h] = glp.glp_get_row_prim(lp, i)
        for j, v in enumerate(self.col_names, start=1):
            if v in assign:
                assign[v] = glp.glp_get_col_prim(lp, j)
        tableau.pivots_since_check = 0      # GLPK LU에서 새로 계산한 표라 쌓인 오차가 없다
        tableau.imported_rows = True        # 대신 GLPK 반올림 잡음 계수가 남을 수 있다

    def _adopt(self, tableau, m: int, stats) -> None:
        """extract="basis": GLPK의 기저 목록만 받아 우리 tableau를 그 기저로 피벗한다
        (카츠 Reluplex의 extractBasicVariables + adjustBasicVariables)."""
        from Simplex import (DEFAULT_TOL, DRIFT_CHECK_EVERY, DRIFT_TOL, _compute_basic,
                             _scaled_tol, _tableau_drift)

        def bump(key, n=1):
            if stats is not None:
                stats[key] = stats.get(key, 0) + n

        glp, lp = self.glp, self.lp
        bhead = glp.glp_get_bhead
        target = {self._name(bhead(lp, i), m) for i in range(1, m + 1)}
        rows = tableau.rows
        pos = {row.basic_var: k for k, row in enumerate(rows)}
        leave = set(pos) - target
        pivots = 0
        for xi in sorted(target - set(pos)):
            # 나가야 할 기저변수의 row 중 xi 계수가 가장 큰 row에서 피벗한다
            best, best_a = None, 0.0
            for xj in leave:
                a = abs(rows[pos[xj]].coeffs.get(xi, 0.0))
                if a > best_a or (a == best_a and best is not None and xj < best):
                    best, best_a = xj, a
            if best is None or best_a < self.ADOPT_PIVOT_MIN:
                bump("GLPK_ADOPT_SKIPPED")
                continue
            _pivot_touched(rows, pos[best], xi)
            pos[xi] = pos.pop(best)
            leave.discard(best)
            pivots += 1
        bump("GLPK_ADOPT_PIVOTS", pivots)
        tableau.pivots_since_check += pivots

        # ③ 피벗이 쌓였으면 원래 등식 대비 오차를 재고, 넘으면 표를 새로 가져온다
        if tableau.pivots_since_check >= DRIFT_CHECK_EVERY:
            bump("GLPK_DRIFT_CHECKS")
            if _tableau_drift(tableau) > DRIFT_TOL:
                bump("GLPK_ADOPT_RESTORES")
                self._import(tableau, m)
                return
            tableau.pivots_since_check = 0

        # 값: 우리 비기저변수는 GLPK 값, 기저변수는 우리 row로 다시 계산
        assign, basic = tableau.assign, set(pos)
        for i, h in enumerate(self.row_heads, start=1):
            if h not in basic:
                assign[h] = glp.glp_get_row_prim(lp, i)
        for j, v in enumerate(self.col_names, start=1):
            if v in assign and v not in basic:
                assign[v] = glp.glp_get_col_prim(lp, j)
        bounds = tableau.bounds
        for row in rows:
            value = _compute_basic(tableau, row)
            assign[row.basic_var] = value
            b = bounds[row.basic_var]
            if not (b.lower - _scaled_tol(b.lower, DEFAULT_TOL) <= value
                    <= b.upper + _scaled_tol(b.upper, DEFAULT_TOL)):
                # 맞춘 값이 경계 밖이다(못 한 피벗이나 반올림) → 카츠는 복원 후 실패로
                # 돌리지만, 여기서는 GLPK 해가 있으니 표 전체를 새로 가져온다
                bump("GLPK_ADOPT_FALLBACK")
                self._import(tableau, m)
                return


def _pivot_touched(rows, k: int, xi: str) -> None:
    """rows[k]의 기저변수와 비기저변수 xi를 맞바꾼다 (Simplex._pivot과 같은 계산).
    native simplex처럼 바뀐 row에서 row 최대 계수의 TAB_ZERO_REL배 이하인 잡음 계수를
    지운다 — native는 피벗마다 모든 row를 훑지만 여기서는 바뀐 row만 본다."""
    prow = rows[k]
    xj = prow.basic_var
    a = prow.coeffs[xi]
    new = {v: -c / a for v, c in prow.coeffs.items() if v != xi}
    new[xj] = 1.0 / a
    prow.basic_var, prow.coeffs = xi, new
    touched = [prow]
    for row in rows:
        if row is prow:
            continue
        factor = row.coeffs.pop(xi, None)
        if factor is None:
            continue
        coeffs = row.coeffs
        for v, c in new.items():
            coeffs[v] = coeffs.get(v, 0.0) + factor * c
        touched.append(row)
    for row in touched:
        if not row.coeffs:
            continue
        thr = TAB_ZERO_REL * max(abs(c) for c in row.coeffs.values())
        for v in [v for v, c in row.coeffs.items() if abs(c) <= thr]:
            del row.coeffs[v]


class Z3Core:
    """카츠 Reluplex 구조에서 '경계 위반 고치기'를 Z3에 맡긴다.

    Z3는 기저나 표를 밖으로 내주지 않는다. 그래서 tableau의 기저는 그대로 두고
    Z3가 찾은 점으로 값만 옮긴다(비기저변수 = Z3 값, 기저변수는 tableau row로 다시
    계산). tableau의 row는 그대로라 tableau row 유도·repair·분기는 native와 같다.
    ② UNSAT이면 경계마다 표시 변수를 달아 check(가정)으로 풀고, unsat core에 든
    경계들의 원인 마스크를 합친다. ① 피벗이 보이지 않으므로 대신 풀 때마다
    tableau의 모든 row로 기저변수 경계를 좁힌다. ③ 정확한 유리수 연산이다.
    """

    name = "z3"
    def needs_row_error(self, tableau) -> bool:
        return False            # tableau row는 native 그대로다

    def __init__(self) -> None:
        import z3
        self.z3 = z3
        self.solver = z3.SolverFor("QF_LRA")
        self.vars: Dict[str, Any] = {}
        self.defined: set = set()
        self._num_cache: Dict[float, Any] = {}
        self.counter = 0
        self.check_seconds = 0.0
        self.failures = 0
        self.conflicts_found = 0

    def _var(self, name: str):
        v = self.vars.get(name)
        if v is None:
            v = self.vars[name] = self.z3.Real(name)
        return v

    def _num(self, value: float):
        num = self._num_cache.get(value)
        if num is None:
            num = self.z3.RealVal(_q(value))
            if len(self._num_cache) < 200000:
                self._num_cache[value] = num
        return num

    def simplex(self, tableau, max_iter: int = 10000, debug: bool = False, *,
                deadline: Optional[float] = None, report_unknown: bool = False,
                progress=None, progress_context=None, stats=None,
                tighten_bounds: bool = False, full_tighten_every: int = 0,
                numeric_safeguards: bool = True):
        from Simplex import _compute_basic, _tighten_from_row

        z3, s = self.z3, self.solver
        check_deadline(deadline)
        tableau.conflict = None
        for head, terms in tableau.origin_rows:
            if head not in self.defined:
                items = [(v, c) for v, c in terms.items() if c != 0.0]
                expr = (z3.Sum([self._num(c) * self._var(v) for v, c in items])
                        if items else z3.RealVal(0))
                s.add(self._var(head) == expr)
                self.defined.add(head)
        s.push()
        try:
            lits, where = [], {}
            for var, b in tableau.bounds.items():
                if var not in tableau.assign:
                    continue
                x = self._var(var)
                for side, val in (("lo", b.lower), ("hi", b.upper)):
                    if not isfinite(val):
                        continue
                    p = z3.Bool(f"__b{self.counter}")
                    self.counter += 1
                    atom = x >= self._num(val) if side == "lo" else x <= self._num(val)
                    s.add(z3.Implies(p, atom))
                    lits.append(p)
                    where[p.get_id()] = (var, side)
            if deadline is not None:
                remaining = deadline - monotonic()
                if remaining <= 0:
                    raise SolverLimitReached("TIMEOUT")
                s.set("timeout", max(1, int(remaining * 1000)))
            started = monotonic()
            res = s.check(*lits)
            self.check_seconds += monotonic() - started
            if res == z3.unsat:
                if tableau.deps is not None:
                    deps, mask = tableau.deps, 0
                    for p in s.unsat_core():
                        var, side = where[p.get_id()]
                        mask |= deps.lo.get(var, 0) if side == "lo" else deps.hi.get(var, 0)
                    tableau.conflict = mask
                    self.conflicts_found += 1
                return None, False
            if res != z3.sat:
                check_deadline(deadline)
                self.failures += 1
                raise SolverLimitReached("LP_ENGINE_UNKNOWN")
            model = s.model()
            z3core = z3.z3core
            ctx = s.ctx.ref()
            out = (z3core.Ast * 1)()
            basic = {row.basic_var for row in tableau.rows}
            for var in tableau.assign:
                if var in basic:
                    continue
                if not z3core.Z3_model_eval(ctx, model.model, self._var(var).as_ast(), True, out):
                    raise SolverLimitReached("LP_ENGINE_FAILURE")
                num, _, den = z3core.Z3_get_numeral_string(ctx, out[0]).partition("/")
                value = int(num) / int(den) if den else float(int(num))
                b = tableau.bounds[var]
                tableau.assign[var] = min(max(value, b.lower), b.upper)
        finally:
            s.pop()
        for row in tableau.rows:
            tableau.assign[row.basic_var] = _compute_basic(tableau, row)
        if tighten_bounds:
            for row in tableau.rows:
                r = _tighten_from_row(tableau, row)
                if r is None:
                    return None, False
                if r and stats is not None:
                    stats["PIVOT_TIGHTENED"] = stats.get("PIVOT_TIGHTENED", 0) + 1
        return dict(tableau.assign), True


CORES = ("glpk", "z3")


def make_core(name: str, full_tighten_every: int = 1000, glpk_extract: str = "full",
              glpk_row_error: str = "auto"):
    if name == "glpk":
        return GlpkCore(full_tighten_every=full_tighten_every, extract=glpk_extract,
                        row_error=glpk_row_error)
    if name == "z3":
        return Z3Core()
    raise ValueError(f"코어 방식 엔진은 {', '.join(CORES)} 뿐이다: {name}")


def make_engine(name: str, simplex_kwargs: Optional[Dict[str, Any]] = None,
                glpk_kwargs: Optional[Dict[str, Any]] = None) -> _EngineBase:
    if name == "simplex":
        return SimplexEngine(simplex_kwargs)
    if name == "glpk":
        return GlpkEngine(**(glpk_kwargs or {}))
    if name == "z3":
        return Z3Engine()
    raise ValueError(f"알 수 없는 LP 엔진: {name} (가능: native, {', '.join(ENGINES)})")
