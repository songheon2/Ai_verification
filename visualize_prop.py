"""
Prop AST를 Graphviz DOT 형식으로 시각화하는 유틸리티.

사용법:
  python visualize_prop.py          # Robustness.py의 Neg_spec을 시각화
  
출력:
  neg_spec_tree.dot   (DOT 소스)
  neg_spec_tree.png   (렌더링 이미지, graphviz 설치 시)

Graphviz 설치 안내:
    Ubuntu/Debian: sudo apt install graphviz
    macOS:         brew install graphviz
    Windows:       https://graphviz.org/download/

Graphviz 설치 후, 명령줄에서 직접 렌더링 가능:
    dot -Tpng neg_spec_tree.dot -o neg_spec_tree.png
"""
from __future__ import annotations
import itertools
from pathlib import Path
from typing import Dict, List

from DPLL import (
    Prop, TrueProp, FalseProp, VarProp, InequProp, ReLUProp,
    AndProp, OrProp, NotProp, ImplProp, show,
    tseitin_cnf
)


def cnf_to_prop(cnf: List[List[str]]) -> Prop:
    """CNF(List[Clause])를 AndProp/OrProp/VarProp/NotProp 트리로 변환."""
    def lit_to_prop(lit: str) -> Prop:
        if lit.startswith("~"):
            return NotProp(VarProp(lit[1:]))
        return VarProp(lit)

    def clause_to_prop(clause: List[str]) -> Prop:
        p = lit_to_prop(clause[0])
        for lit in clause[1:]:
            p = OrProp(p, lit_to_prop(lit))
        return p

    p = clause_to_prop(cnf[0])
    for clause in cnf[1:]:
        p = AndProp(p, clause_to_prop(clause))
    return p


def _ineq_label(prop: InequProp) -> str:
    """InequProp을 읽기 좋은 문자열로 변환."""
    terms = []
    coeffs_dict = dict(prop.coeffs)
    for var in sorted(coeffs_dict.keys()):
        coeff = coeffs_dict[var]
        if coeff == 1.0:
            terms.append(var)
        elif coeff == -1.0:
            terms.append(f"-{var}")
        else:
            terms.append(f"{coeff}*{var}")
    expr = " + ".join(terms).replace("+ -", "- ")
    return f"{expr} ≥ {prop.b}"


def prop_to_dot(prop: Prop, name: str = "prop_tree") -> str:
    """
    Prop 트리를 Graphviz DOT 문자열로 변환.
    
    노드 종류별 스타일:
      - AND/OR/IMPL/NOT : 타원(ellipse), 연한 파랑
      - InequProp        : 박스, 연한 초록
      - ReLUProp         : 박스, 연한 주황
      - True/False       : 다이아몬드
      - VarProp          : 박스, 연한 노랑
    """
    _counter = itertools.count()
    lines: list[str] = []
    lines.append(f"digraph {name} {{")
    lines.append("  rankdir=TB;")
    lines.append("  node [fontname=\"Helvetica\", fontsize=11];")
    lines.append("  edge [arrowsize=0.7];")

    def _node_id() -> str:
        return f"n{next(_counter)}"

    def _escape(s: str) -> str:
        return s.replace("\\", "\\\\").replace('"', '\\"')

    def _visit(p: Prop) -> str:
        nid = _node_id()

        if isinstance(p, TrueProp):
            lines.append(f'  {nid} [label="⊤", shape=diamond, '
                         f'style=filled, fillcolor="#d3d3d3"];')
        elif isinstance(p, FalseProp):
            lines.append(f'  {nid} [label="⊥", shape=diamond, '
                         f'style=filled, fillcolor="#d3d3d3"];')
        elif isinstance(p, VarProp):
            lines.append(f'  {nid} [label="{_escape(p.name)}", shape=box, '
                         f'style=filled, fillcolor="#ffffcc"];')
        elif isinstance(p, InequProp):
            label = _escape(_ineq_label(p))
            lines.append(f'  {nid} [label="{label}", shape=box, '
                         f'style=filled, fillcolor="#ccffcc"];')
        elif isinstance(p, ReLUProp):
            lines.append(f'  {nid} [label="ReLU\\n{_escape(p.y)} = relu({_escape(p.x)})", '
                         f'shape=box, style=filled, fillcolor="#ffe0b2"];')
        elif isinstance(p, NotProp):
            lines.append(f'  {nid} [label="NOT", shape=ellipse, '
                         f'style=filled, fillcolor="#bbdefb"];')
            child = _visit(p.p)
            lines.append(f'  {nid} -> {child};')
        elif isinstance(p, AndProp):
            lines.append(f'  {nid} [label="AND", shape=ellipse, '
                         f'style=filled, fillcolor="#bbdefb"];')
            left = _visit(p.p)
            right = _visit(p.q)
            lines.append(f'  {nid} -> {left} [label="L"];')
            lines.append(f'  {nid} -> {right} [label="R"];')
        elif isinstance(p, OrProp):
            lines.append(f'  {nid} [label="OR", shape=ellipse, '
                         f'style=filled, fillcolor="#bbdefb"];')
            left = _visit(p.p)
            right = _visit(p.q)
            lines.append(f'  {nid} -> {left} [label="L"];')
            lines.append(f'  {nid} -> {right} [label="R"];')
        elif isinstance(p, ImplProp):
            lines.append(f'  {nid} [label="IMPL (→)", shape=ellipse, '
                         f'style=filled, fillcolor="#bbdefb"];')
            left = _visit(p.p)
            right = _visit(p.q)
            lines.append(f'  {nid} -> {left} [label="ante"];')
            lines.append(f'  {nid} -> {right} [label="cons"];')
        else:
            lines.append(f'  {nid} [label="{_escape(str(p))}", shape=box];')

        return nid

    _visit(prop)
    lines.append("}")
    return "\n".join(lines)


def cnf_to_dot(
    cnf: Prop,
    atom_map: Dict[Prop, str],
    memo: Dict[Prop, str] | None = None,
    name: str = "cnf_graph",
) -> str:
    """
    CNF Prop 트리와 atom_map/memo 범례를 좌우 나란히 배치하는 DOT 그래프.

    왼쪽: CNF Prop 트리 (prop_to_dot과 동일한 방식)
    오른쪽: atom_map + memo 범례 (명제 변수 → 원래 Prop 설명)
    """
    _counter = itertools.count()
    lines: list[str] = []
    lines.append(f"digraph {name} {{")
    lines.append("  rankdir=TB;")
    lines.append('  node [fontname="Helvetica", fontsize=10];')
    lines.append("  edge [arrowsize=0.6];")

    def _node_id() -> str:
        return f"n{next(_counter)}"

    def _visit(p: Prop) -> str:
        nid = _node_id()
        if isinstance(p, TrueProp):
            lines.append(f'    {nid} [label="⊤", shape=diamond, '
                         f'style=filled, fillcolor="#d3d3d3"];')
        elif isinstance(p, FalseProp):
            lines.append(f'    {nid} [label="⊥", shape=diamond, '
                         f'style=filled, fillcolor="#d3d3d3"];')
        elif isinstance(p, VarProp):
            lines.append(f'    {nid} [label="{_escape_dot(p.name)}", shape=box, '
                         f'style=filled, fillcolor="#ffffcc"];')
        elif isinstance(p, InequProp):
            label = _escape_dot(_ineq_label(p))
            lines.append(f'    {nid} [label="{label}", shape=box, '
                         f'style=filled, fillcolor="#ccffcc"];')
        elif isinstance(p, ReLUProp):
            lines.append(f'    {nid} [label="ReLU\\n{_escape_dot(p.y)} = relu({_escape_dot(p.x)})", '
                         f'shape=box, style=filled, fillcolor="#ffe0b2"];')
        elif isinstance(p, NotProp):
            lines.append(f'    {nid} [label="NOT", shape=ellipse, '
                         f'style=filled, fillcolor="#bbdefb"];')
            child = _visit(p.p)
            lines.append(f'    {nid} -> {child};')
        elif isinstance(p, AndProp):
            lines.append(f'    {nid} [label="AND", shape=ellipse, '
                         f'style=filled, fillcolor="#bbdefb"];')
            left = _visit(p.p)
            right = _visit(p.q)
            lines.append(f'    {nid} -> {left} [label="L"];')
            lines.append(f'    {nid} -> {right} [label="R"];')
        elif isinstance(p, OrProp):
            lines.append(f'    {nid} [label="OR", shape=ellipse, '
                         f'style=filled, fillcolor="#bbdefb"];')
            left = _visit(p.p)
            right = _visit(p.q)
            lines.append(f'    {nid} -> {left} [label="L"];')
            lines.append(f'    {nid} -> {right} [label="R"];')
        elif isinstance(p, ImplProp):
            lines.append(f'    {nid} [label="IMPL (→)", shape=ellipse, '
                         f'style=filled, fillcolor="#bbdefb"];')
            left = _visit(p.p)
            right = _visit(p.q)
            lines.append(f'    {nid} -> {left} [label="ante"];')
            lines.append(f'    {nid} -> {right} [label="cons"];')
        else:
            lines.append(f'    {nid} [label="{_escape_dot(str(p))}", shape=box];')
        return nid

    # --- 왼쪽: CNF 트리 ---
    lines.append('  subgraph cluster_cnf {')
    lines.append('    label="CNF";')
    lines.append('    style=dashed; color=gray;')
    _visit(cnf)
    lines.append('  }')

    # --- 오른쪽: atom_map 범례 ---
    lines.append('  subgraph cluster_atom_map {')
    lines.append('    label="atom_map";')
    lines.append('    style=dashed; color=gray;')

    abbrev = _build_abbrev(atom_map, memo)

    # atom_map 엔트리 (a 변수)
    sorted_a = sorted(atom_map.items(), key=lambda kv: kv[1])
    prev_id = None
    for prop, alias in sorted_a:
        desc = _desc_abbrev(prop, alias, abbrev)
        mid = _node_id()
        label = f"{alias}  ↔  {desc}"
        lines.append(f'    {mid} [label="{_escape_dot(label)}", shape=box, '
                     f'style=filled, fillcolor="#e8f5e9"];')
        if prev_id is not None:
            lines.append(f'    {prev_id} -> {mid} [style=invis];')
        prev_id = mid

    # memo 엔트리 (t 변수)
    if memo:
        sorted_t = sorted(memo.items(), key=lambda kv: kv[1])
        for prop, alias in sorted_t:
            mid = _node_id()
            desc = _desc_abbrev(prop, alias, abbrev)
            label = f"{alias}  ↔  {desc}"
            lines.append(f'    {mid} [label="{_escape_dot(label)}", shape=box, '
                         f'style=filled, fillcolor="#fff3e0"];')
            if prev_id is not None:
                lines.append(f'    {prev_id} -> {mid} [style=invis];')
            prev_id = mid

    lines.append('  }')

    lines.append("}")
    return "\n".join(lines)


def show_atom_map(
    atom_map: Dict[Prop, str],
    memo: Dict[Prop, str] | None = None,
) -> str:
    """atom_map과 memo를 'alias : Prop 설명' 형태로 예쁘게 출력 (축약 사용)."""
    abbrev = _build_abbrev(atom_map, memo)
    lines = []
    # a 변수: 이론 원자
    for prop, alias in sorted(atom_map.items(), key=lambda kv: kv[1]):
        desc = _desc_abbrev(prop, alias, abbrev)
        lines.append(f"  {alias} : {desc}")
    # t 변수: Tseitin 보조 변수
    if memo:
        lines.append("")
        for prop, alias in sorted(memo.items(), key=lambda kv: kv[1]):
            desc = _desc_abbrev(prop, alias, abbrev)
            lines.append(f"  {alias} : {desc}")
    return "\n".join(lines)


def _show_abbrev(prop: Prop, abbrev: Dict[Prop, str]) -> str:
    """show()와 동일하되, abbrev에 등록된 부분식은 변수 이름으로 대체."""
    if prop in abbrev:
        return abbrev[prop]
    if isinstance(prop, TrueProp): return "⊤"
    if isinstance(prop, FalseProp): return "⊥"
    if isinstance(prop, VarProp): return prop.name
    if isinstance(prop, InequProp): return f"({_ineq_label(prop)})"
    if isinstance(prop, ReLUProp): return f"relu({prop.x},{prop.y})"
    if isinstance(prop, NotProp): return f"¬{_show_abbrev(prop.p, abbrev)}"
    if isinstance(prop, AndProp):
        return f"({_show_abbrev(prop.p, abbrev)} ∧ {_show_abbrev(prop.q, abbrev)})"
    if isinstance(prop, OrProp):
        return f"({_show_abbrev(prop.p, abbrev)} ∨ {_show_abbrev(prop.q, abbrev)})"
    if isinstance(prop, ImplProp):
        return f"({_show_abbrev(prop.p, abbrev)} → {_show_abbrev(prop.q, abbrev)})"
    return str(prop)


def _build_abbrev(
    atom_map: Dict[Prop, str],
    memo: Dict[Prop, str] | None,
) -> Dict[Prop, str]:
    """memo와 atom_map의 Prop → 변수명 매핑을 통합한 dict."""
    abbrev: Dict[Prop, str] = {}
    for prop, alias in atom_map.items():
        abbrev[prop] = alias
    if memo:
        for prop, alias in memo.items():
            abbrev[prop] = alias
    return abbrev


def _desc_abbrev(
    prop: Prop,
    alias: str,
    abbrev: Dict[Prop, str],
) -> str:
    """alias 자신은 abbrev에서 제외하고 1단계만 전개한 설명을 반환."""
    saved = abbrev.pop(prop, None)
    desc = _show_abbrev(prop, abbrev)
    if saved is not None:
        abbrev[prop] = saved
    return desc


def _escape_dot(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")

OUTPUT_DIR = Path("./visualize_precise_prop")

def save_dot(dot_src: str, filepath: str = "neg_spec_tree") -> str:
    """DOT 소스를 OUTPUT_DIR 아래 .dot 파일로 저장. 경로(확장자 제외)를 반환."""
    base_path = OUTPUT_DIR / filepath
    base_path.parent.mkdir(parents=True, exist_ok=True)
    dot_path = base_path.with_suffix(".dot")

    with open(dot_path, "w", encoding="utf-8") as f:
        f.write(dot_src)

    print(f"DOT 파일 저장: {dot_path}")
    return str(base_path)



def render_dot(filepath: str = "neg_spec_tree", fmt: str = "png") -> None:
    """Graphviz가 있으면 OUTPUT_DIR 아래의 .dot를 렌더링."""
    import subprocess
    import shutil

    base_path = Path(filepath)

    # save_dot()가 이미 OUTPUT_DIR 포함 경로를 반환할 수 있으므로
    # 부모 디렉터리가 없을 때만 OUTPUT_DIR를 붙인다.
    if not base_path.is_absolute() and base_path.parent == Path("."):
        base_path = OUTPUT_DIR / base_path

    dot_path = base_path.with_suffix(".dot")
    out_path = base_path.with_suffix(f".{fmt}")

    if shutil.which("dot") is None:
        print("'dot' 명령어를 찾을 수 없습니다.")
        print("  Ubuntu/Debian: sudo apt install graphviz")
        print("  macOS:         brew install graphviz")
        print(f"설치 후 직접 실행: dot -T{fmt} {dot_path} -o {out_path}")
        return

    subprocess.run(["dot", f"-T{fmt}", str(dot_path), "-o", str(out_path)], check=True)
    print(f"이미지 저장: {out_path}")

def visualize_precise_case(
    case_name: str,
    r1: tuple[float, float],
    r2: tuple[float, float],
    cls_x1,
    cls_x2,
    outcls,
) -> None:
    """PreciseEncoding.py의 단일 케이스 반례식을 시각화."""

    x1 = "x1"
    x2 = "x2"

    def x1_range_prop(i: str) -> Prop:
        return AndProp(
            InequProp(frozenset([(i, 1.0)]), r1[0]),
            InequProp(frozenset([(i, -1.0)]), -r1[1]),
        )

    def x2_range_prop(i: str) -> Prop:
        return AndProp(
            InequProp(frozenset([(i, 1.0)]), r2[0]),
            InequProp(frozenset([(i, -1.0)]), -r2[1]),
        )

    from XOREncoding import FreshGen, NN_single
    from PreciseEncoding import AND

    fg = FreshGen(prefix=f"viz_x{case_name}_")
    NNprop, s, _ = NN_single((x1, x2), gen=fg)

    pre = AND(
        cls_x1(x1),
        cls_x2(x2),
        x1_range_prop(x1),
        x2_range_prop(x2),
        NNprop,
    )
    post = outcls(s)
    neg_spec = AND(pre, NotProp(post))

    print(f"\n{'=' * 80}")
    print(f"=== PreciseEncoding case {case_name} neg_spec ===")
    print(show(neg_spec))

    tree_base = f"precise_case{case_name}_tree"
    cnf_base = f"precise_case{case_name}_cnf"

    dot_src = prop_to_dot(neg_spec, name=f"precise_case{case_name}_neg_spec")
    tree_path = save_dot(dot_src, tree_base)
    render_dot(tree_path, fmt="png")

    cnf, atom_map, memo = tseitin_cnf(neg_spec)
    neg_spec_cnf = cnf_to_prop(cnf)

    cnf, atom_map, memo = tseitin_cnf(neg_spec)
    neg_spec_cnf = cnf_to_prop(cnf)

    print("\n=== Tseitin CNF (as Prop) ===")
    print(show(neg_spec_cnf))
    print("\n=== atom_map / memo ===")
    print(show_atom_map(atom_map, memo))

    dot_src_cnf = cnf_to_dot(
        neg_spec_cnf,
        atom_map,
        memo,
        name=f"precise_case{case_name}_neg_spec_cnf",
    )
    cnf_path = save_dot(dot_src_cnf, cnf_base)
    render_dot(cnf_path, fmt="png")

def dump_search_phi_visualization(
    phi: Prop,
    *,
    case_name: str,
    attempt_no: int,
    render_png: bool = True,
    print_alias_map: bool = False,
) -> None:
    """
    현재 dpll_t에 넣을 phi를 시각화한다.
    저장 파일 예:
      case_00/precise_search_case00_attempt1_tree.dot/.png
      case_00/precise_search_case00_attempt1_cnf.dot/.png
    """
    case_dir = f"case_{case_name}"
    base = f"precise_search_case{case_name}_attempt{attempt_no}"


    dot_src = prop_to_dot(phi, name=f"{base}_tree")
    tree_path = save_dot(dot_src, f"{case_dir}/{base}_tree")
    if render_png:
        render_dot(tree_path, fmt="png")

    cnf, atom_map, memo = tseitin_cnf(phi)
    phi_cnf = cnf_to_prop(cnf)

    dot_src_cnf = cnf_to_dot(
        phi_cnf,
        atom_map,
        memo,
        name=f"{base}_cnf",
    )
    cnf_path = save_dot(dot_src_cnf, f"{case_dir}/{base}_cnf")
    if render_png:
        render_dot(cnf_path, fmt="png")

    if print_alias_map:
        print(f"\n=== alias map for {base} ===")
        print(show_atom_map(atom_map, memo))

# ============================================================
# main: Robustness.py의 Neg_spec 하나(center=(0,0))를 시각화
# ============================================================
if __name__ == "__main__":
    from Robustness import make_precondition_linf_box, make_postcondition_same_class_by_logit
    from XOREncoding import FreshGen, NN_dual, NN_single
    from PreciseEncoding import AND, zero, out_zero_logit, one, out_one_logit

    # Robustness.py 명세 버전
    # c = (0.0, 0.0)
    # x_vars = ("x0", "x1")

    # pre = make_precondition_linf_box(x_vars, c, eps=0.02, clamp_01=True)
    # fg = FreshGen(prefix=f"c{int(c[0])}{int(c[1])}_")
    # NN_prop, s_x_sym, s_c_sym, aux = NN_dual(x=x_vars, c=c, gen=fg)
    # post = make_postcondition_same_class_by_logit(s_x_sym, s_c_sym)

    # Neg_spec = AndProp(pre, AndProp(NN_prop, NotProp(post)))
    # print(show(Neg_spec))
    # dot_src = prop_to_dot(Neg_spec, name="Neg_spec")
    # save_dot(dot_src, "neg_spec_tree")
    # # render_dot("neg_spec_tree", fmt="png")

    # cnf, atom_map, memo = tseitin_cnf(Neg_spec)
    # Neg_spec_cnf = cnf_to_prop(cnf)
    # print(show(Neg_spec_cnf))
    # print(show_atom_map(atom_map, memo))
    # dot_src_cnf = cnf_to_dot(Neg_spec_cnf, atom_map, memo, name="neg_spec_cnf")
    # save_dot(dot_src_cnf, "neg_spec_cnf")
    # # render_dot("neg_spec_cnf", fmt="png")

    # PreciseEncoding.py 명세 버전

    eps = 0.1
    ranges = [
        (0.0, 0.5 - eps),   # [0.0, 0.4]
        (0.5 + eps, 1.0),   # [0.6, 1.0]
    ]

    cases = [
        ("00", 0, 0, zero, zero, out_zero_logit),
        ("01", 0, 1, zero, one,  out_one_logit),
        ("10", 1, 0, one,  zero, out_one_logit),
        ("11", 1, 1, one,  one,  out_zero_logit),
    ]

    for case_name, r1_idx, r2_idx, cls_x1, cls_x2, outcls in cases:
        visualize_precise_case(
            case_name=case_name,
            r1=ranges[r1_idx],
            r2=ranges[r2_idx],
            cls_x1=cls_x1,
            cls_x2=cls_x2,
            outcls=outcls,
        )