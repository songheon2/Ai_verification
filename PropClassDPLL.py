# -*- coding: utf-8 -*-
"""
Created on Mon Feb  2 17:03:24 2026

@author: a5254
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set, Union

# ============================================================
# 0) Prop AST 정의
# ============================================================

class Prop:
    pass

@dataclass(frozen=True)
class TrueProp(Prop):
    pass

@dataclass(frozen=True)
class FalseProp(Prop):
    pass

@dataclass(frozen=True)
class VarProp(Prop):
    name: str

@dataclass(frozen=True)
class InequProp(Prop):
    # c*x >= b
    c: float
    x: str
    b: float

@dataclass(frozen=True)
class AndProp(Prop):
    p: Prop
    q: Prop

@dataclass(frozen=True)
class OrProp(Prop):
    p: Prop
    q: Prop

@dataclass(frozen=True)
class NotProp(Prop):
    p: Prop

@dataclass(frozen=True)
class ImplProp(Prop):
    p: Prop
    q: Prop


# ============================================================
# 1) Pretty-print
# ============================================================

def show(prop: Prop) -> str:
    if isinstance(prop, TrueProp): return "⊤"
    if isinstance(prop, FalseProp): return "⊥"
    if isinstance(prop, VarProp): return prop.name
    if isinstance(prop, InequProp): return f"({prop.c}*{prop.x} >= {prop.b})"
    if isinstance(prop, NotProp): return f"¬{show(prop.p)}"
    if isinstance(prop, AndProp): return f"({show(prop.p)} ∧ {show(prop.q)})"
    if isinstance(prop, OrProp):  return f"({show(prop.p)} ∨ {show(prop.q)})"
    if isinstance(prop, ImplProp):return f"({show(prop.p)} → {show(prop.q)})"
    raise TypeError(prop)


# ============================================================
# 2) simplify / elim_impl / NNF
# ============================================================

def simplify(p: Prop) -> Prop:
    if isinstance(p, (VarProp, InequProp, TrueProp, FalseProp)):
        return p

    if isinstance(p, NotProp):
        inner = simplify(p.p)
        if isinstance(inner, TrueProp):  return FalseProp()
        if isinstance(inner, FalseProp): return TrueProp()
        if isinstance(inner, NotProp):   return simplify(inner.p)
        return NotProp(inner)

    if isinstance(p, AndProp):
        a, b = simplify(p.p), simplify(p.q)
        if isinstance(a, FalseProp) or isinstance(b, FalseProp): return FalseProp()
        if isinstance(a, TrueProp):  return b
        if isinstance(b, TrueProp):  return a
        return AndProp(a, b)

    if isinstance(p, OrProp):
        a, b = simplify(p.p), simplify(p.q)
        if isinstance(a, TrueProp) or isinstance(b, TrueProp): return TrueProp()
        if isinstance(a, FalseProp): return b
        if isinstance(b, FalseProp): return a
        return OrProp(a, b)

    if isinstance(p, ImplProp):
        return ImplProp(simplify(p.p), simplify(p.q))

    raise TypeError(p)


def elim_impl(p: Prop) -> Prop:
    """(p -> q) == (~p or q) 로 바꿔 ImplProp 제거"""
    if isinstance(p, (VarProp, InequProp, TrueProp, FalseProp)):
        return p
    if isinstance(p, NotProp):
        return NotProp(elim_impl(p.p))
    if isinstance(p, AndProp):
        return AndProp(elim_impl(p.p), elim_impl(p.q))
    if isinstance(p, OrProp):
        return OrProp(elim_impl(p.p), elim_impl(p.q))
    if isinstance(p, ImplProp):
        return OrProp(NotProp(elim_impl(p.p)), elim_impl(p.q))
    raise TypeError(p)


def to_nnf(p: Prop) -> Prop:
    """NNF: Not이 Var/Inequ 바로 위에만 오도록"""
    p = simplify(elim_impl(simplify(p)))

    def nnf(x: Prop) -> Prop:
        x = simplify(x)

        if isinstance(x, (VarProp, InequProp, TrueProp, FalseProp)):
            return x

        if isinstance(x, AndProp):
            return simplify(AndProp(nnf(x.p), nnf(x.q)))

        if isinstance(x, OrProp):
            return simplify(OrProp(nnf(x.p), nnf(x.q)))

        if isinstance(x, NotProp):
            a = simplify(x.p)

            if isinstance(a, (VarProp, InequProp)):
                return NotProp(a)

            if isinstance(a, TrueProp):  return FalseProp()
            if isinstance(a, FalseProp): return TrueProp()

            if isinstance(a, NotProp):
                return nnf(a.p)

            if isinstance(a, AndProp):
                return nnf(OrProp(NotProp(a.p), NotProp(a.q)))
            if isinstance(a, OrProp):
                return nnf(AndProp(NotProp(a.p), NotProp(a.q)))

            raise TypeError(a)

        raise TypeError(x)

    return nnf(p)


# ============================================================
# 3) Tseitin: NNF -> CNF
# ============================================================

Literal = str
Clause  = List[Literal]
CNF     = List[Clause]

def neg(lit: Literal) -> Literal:
    return lit[1:] if lit.startswith("~") else "~" + lit

def tseitin_cnf(formula: Prop) -> Tuple[CNF, Dict[InequProp, str]]:
    f = to_nnf(formula)

    cnf: CNF = []
    memo: Dict[Prop, str] = {}

    t_counter = 0
    def fresh_t() -> str:
        nonlocal t_counter
        t_counter += 1
        return f"t{t_counter}"

    ineq_map: Dict[InequProp, str] = {}
    a_counter = 0
    def atom_of_ineq(ineq: InequProp) -> str:
        nonlocal a_counter
        if ineq not in ineq_map:
            a_counter += 1
            ineq_map[ineq] = f"a{a_counter}"
        return ineq_map[ineq]

    def lit_of_atom(x: Prop) -> Literal:
        if isinstance(x, VarProp):
            return x.name
        if isinstance(x, InequProp):
            return atom_of_ineq(x)
        if isinstance(x, NotProp) and isinstance(x.p, VarProp):
            return "~" + x.p.name
        if isinstance(x, NotProp) and isinstance(x.p, InequProp):
            return "~" + atom_of_ineq(x.p)
        raise ValueError(f"원자 리터럴이 아님: {x}")

    def add_equiv_and(t: str, a: Literal, b: Literal):
        cnf.append([neg(t), a])
        cnf.append([neg(t), b])
        cnf.append([t, neg(a), neg(b)])

    def add_equiv_or(t: str, a: Literal, b: Literal):
        cnf.append([neg(t), a, b])
        cnf.append([t, neg(a)])
        cnf.append([t, neg(b)])

    def encode(x: Prop) -> Literal:
        x = simplify(x)

        if isinstance(x, TrueProp):
            t = fresh_t()
            cnf.append([t])
            return t
        if isinstance(x, FalseProp):
            t = fresh_t()
            cnf.append([neg(t)])
            return t

        if isinstance(x, (VarProp, InequProp)) or (isinstance(x, NotProp) and isinstance(x.p, (VarProp, InequProp))):
            return lit_of_atom(x)

        if x in memo:
            return memo[x]

        if isinstance(x, AndProp):
            t = fresh_t()
            memo[x] = t
            a = encode(x.p)
            b = encode(x.q)
            add_equiv_and(t, a, b)
            return t

        if isinstance(x, OrProp):
            t = fresh_t()
            memo[x] = t
            a = encode(x.p)
            b = encode(x.q)
            add_equiv_or(t, a, b)
            return t

        raise TypeError(f"NNF 이후 지원되지 않는 형태: {x}")

    top = encode(f)
    cnf.append([top])
    return cnf, ineq_map

def show_cnf(cnf: CNF) -> str:
    return " ∧ ".join("(" + " ∨ ".join(cl) + ")" for cl in cnf)


# ============================================================
# 4) DPLL
# ============================================================

Assignment = Dict[str, bool]

def is_neg_lit(lit: Literal) -> bool:
    return lit.startswith("~")

def var_of(lit: Literal) -> str:
    return lit[1:] if is_neg_lit(lit) else lit

def lit_value(lit: Literal, asn: Assignment) -> Optional[bool]:
    v = var_of(lit)
    if v not in asn:
        return None
    val = asn[v]
    return (not val) if is_neg_lit(lit) else val

def simplify_cnf_by_asn(cnf: CNF, asn: Assignment) -> Optional[CNF]:
    new_cnf: CNF = []
    for clause in cnf:
        sat = False
        new_clause: Clause = []
        for lit in clause:
            lv = lit_value(lit, asn)
            if lv is True:
                sat = True
                break
            elif lv is False:
                continue
            else:
                new_clause.append(lit)
        if sat:
            continue
        if len(new_clause) == 0:
            return None
        new_cnf.append(new_clause)
    return new_cnf

def apply_literal(asn: Assignment, lit: Literal) -> bool:
    v = var_of(lit)
    desired = not is_neg_lit(lit)
    if v in asn:
        return asn[v] == desired
    asn[v] = desired
    return True

def find_unit_literals(cnf: CNF) -> List[Literal]:
    return [cl[0] for cl in cnf if len(cl) == 1]

def unit_propagation(cnf: CNF, asn: Assignment) -> Optional[CNF]:
    while True:
        units = find_unit_literals(cnf)
        if not units:
            return cnf
        for u in units:
            if not apply_literal(asn, u):
                return None
        cnf2 = simplify_cnf_by_asn(cnf, asn)
        if cnf2 is None:
            return None
        cnf = cnf2

def pure_literal_elimination(cnf: CNF, asn: Assignment) -> Optional[CNF]:
    lits: Set[Literal] = set()
    for clause in cnf:
        for lit in clause:
            if var_of(lit) in asn:
                continue
            lits.add(lit)

    vars_seen = {var_of(l) for l in lits}
    pures: List[Literal] = []
    for v in vars_seen:
        pos = v in lits
        negv = ("~" + v) in lits
        if pos and not negv:
            pures.append(v)
        elif negv and not pos:
            pures.append("~" + v)

    if not pures:
        return cnf

    for pl in pures:
        apply_literal(asn, pl)

    return simplify_cnf_by_asn(cnf, asn)

def choose_branch_var(cnf: CNF, asn: Assignment) -> str:
    for clause in cnf:
        for lit in clause:
            v = var_of(lit)
            if v not in asn:
                return v
    return "__done__"

def dpll(cnf: CNF, asn: Optional[Assignment] = None) -> Optional[Assignment]:
    if asn is None:
        asn = {}

    cnf = simplify_cnf_by_asn(cnf, asn)
    if cnf is None:
        return None
    if len(cnf) == 0:
        return asn

    cnf = unit_propagation(cnf, asn)
    if cnf is None:
        return None
    if len(cnf) == 0:
        return asn

    cnf = pure_literal_elimination(cnf, asn)
    if cnf is None:
        return None
    if len(cnf) == 0:
        return asn

    v = choose_branch_var(cnf, asn)
    if v == "__done__":
        return asn

    asn1 = dict(asn)
    apply_literal(asn1, v)
    res = dpll(cnf, asn1)
    if res is not None:
        return res

    asn2 = dict(asn)
    apply_literal(asn2, "~" + v)
    return dpll(cnf, asn2)


# ============================================================
# 5) 문자열 입력 -> Prop 파서 (간단 DSL)
# ============================================================

Token = Tuple[str, str]  # (type, value)

def tokenize(s: str) -> List[Token]:
    s = s.strip()
    i = 0
    out: List[Token] = []

    def is_ident_start(ch: str) -> bool:
        return ch.isalpha() or ch == "_"
    def is_ident(ch: str) -> bool:
        return ch.isalnum() or ch == "_"

    while i < len(s):
        ch = s[i]

        if ch.isspace():
            i += 1
            continue

        if s.startswith("->", i):
            out.append(("ARROW", "->"))
            i += 2
            continue

        if ch in "() ,":
            out.append((ch, ch))
            i += 1
            continue

        if ch == "~":
            out.append(("NOT", "~"))
            i += 1
            continue

        # number (for ineq)
        if ch.isdigit() or (ch == "-" and i+1 < len(s) and s[i+1].isdigit()):
            j = i+1
            while j < len(s) and (s[j].isdigit() or s[j] == "."):
                j += 1
            out.append(("NUM", s[i:j]))
            i = j
            continue

        # identifier / keywords
        if is_ident_start(ch):
            j = i+1
            while j < len(s) and is_ident(s[j]):
                j += 1
            word = s[i:j]
            lw = word.lower()
            if lw == "and":
                out.append(("AND", "and"))
            elif lw == "or":
                out.append(("OR", "or"))
            elif lw == "not":
                out.append(("NOT", "not"))
            elif lw == "true":
                out.append(("TRUE", "true"))
            elif lw == "false":
                out.append(("FALSE", "false"))
            elif lw == "ineq":
                out.append(("INEQ", "ineq"))
            else:
                out.append(("ID", word))
            i = j
            continue

        raise ValueError(f"토큰화 실패: '{ch}' (pos {i})")

    out.append(("EOF", ""))
    return out


class Parser:
    def __init__(self, tokens: List[Token]):
        self.toks = tokens
        self.k = 0

    def peek(self) -> Token:
        return self.toks[self.k]

    def eat(self, typ: str) -> Token:
        t = self.peek()
        if t[0] != typ:
            raise ValueError(f"Expected {typ}, got {t}")
        self.k += 1
        return t

    # 문법 우선순위:
    #   implication:  A -> B
    #   or:           A or B
    #   and:          A and B
    #   not:          not A / ~A
    #   atom:         ID | true | false | (expr) | ineq(c,x,b)

    def parse(self) -> Prop:
        e = self.parse_imp()
        self.eat("EOF")
        return e

    def parse_imp(self) -> Prop:
        left = self.parse_or()
        if self.peek()[0] == "ARROW":
            self.eat("ARROW")
            right = self.parse_imp()  # right-assoc
            return ImplProp(left, right)
        return left

    def parse_or(self) -> Prop:
        left = self.parse_and()
        while self.peek()[0] == "OR":
            self.eat("OR")
            right = self.parse_and()
            left = OrProp(left, right)
        return left

    def parse_and(self) -> Prop:
        left = self.parse_not()
        while self.peek()[0] == "AND":
            self.eat("AND")
            right = self.parse_not()
            left = AndProp(left, right)
        return left

    def parse_not(self) -> Prop:
        if self.peek()[0] == "NOT":
            self.eat("NOT")
            return NotProp(self.parse_not())
        return self.parse_atom()

    def parse_atom(self) -> Prop:
        t = self.peek()

        if t[0] == "(":
            self.eat("(")
            e = self.parse_imp()
            self.eat(")")
            return e

        if t[0] == "TRUE":
            self.eat("TRUE")
            return TrueProp()

        if t[0] == "FALSE":
            self.eat("FALSE")
            return FalseProp()

        if t[0] == "ID":
            name = self.eat("ID")[1]
            return VarProp(name)

        if t[0] == "INEQ":
            self.eat("INEQ")
            self.eat("(")
            c = float(self.eat("NUM")[1])
            self.eat(",")
            x = self.eat("ID")[1]
            self.eat(",")
            b = float(self.eat("NUM")[1])
            self.eat(")")
            return InequProp(c, x, b)

        raise ValueError(f"Unexpected token: {t}")


def parse_prop(s: str) -> Prop:
    return Parser(tokenize(s)).parse()

def show_clause(cl: Clause) -> str:
    """절(OR들의 묶음)을 보기 좋게 문자열로"""
    return "(" + " ∨ ".join(cl) + ")"

def print_cnf_clauses(cnf: CNF, max_clauses: int = 200) -> None:
    """
    CNF를 절 단위로 출력.
    너무 길어질 수 있으니 기본적으로 max_clauses까지만 보여줌.
    """
    print(f"CNF 절 목록 (총 {len(cnf)}개):")
    for i, cl in enumerate(cnf, start=1):
        if i > max_clauses:
            print(f"  ... (이후 {len(cnf) - max_clauses}개 절 생략)")
            break
        print(f"  C{i:03d}: {show_clause(cl)}")


# ============================================================
# 6) 사용자 입력 -> 파이프라인 실행
# ============================================================

def run_pipeline(formula: Prop) -> None:
    print("=" * 90)
    print("입력식(Pretty) :", show(formula))

    nnf_f = to_nnf(formula)
    print("NNF           :", show(nnf_f))

    cnf, ineq_map = tseitin_cnf(formula)
    print("CNF           :", show_cnf(cnf))
    
    # 절 단위 출력 추가
    print_cnf_clauses(cnf, max_clauses=200)

    if ineq_map:
        print("Inequ 추상화 매핑(InequProp -> a_k):")
        for k, v in ineq_map.items():
            print(f"  {v} := {show(k)}")

    model = dpll(cnf)
    if model is None:
        print("DPLL 결과     : UNSAT")
    else:
        print("DPLL 결과     : SAT")
        for var in sorted(model.keys()):
            print(f"  {var} = {model[var]}")
    print()


if __name__ == "__main__":
    print("=== Spec 입력 (종료: quit / exit) ===")
    print("문법: and, or, not(or ~), ->, 괄호(), true/false, ineq(c,x,b)")
    print("예: (p and q) or not r")
    print("예: not (p -> q)")
    print("예: ineq(1, x, 0) or p")
    print("예: (ineq(1,x,-0.1) and ineq(-1,x,0.1)) -> same_class")
    print()

    while True:
        s = input("spec> ").strip()
        if not s:
            continue
        if s.lower() in ("quit", "exit"):
            break
        try:
            prop = parse_prop(s)
            run_pipeline(prop)
        except Exception as e:
            print("오류:", e)
            print("다시 입력해줘.\n")

    
