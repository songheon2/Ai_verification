"""
DPLL(T) 실행 중 BCP/Simplex/Reluplex 각 구간이 언제 시작·끝났는지, ReLU split이
어느 뉴런(branch_x)에서 일어났는지 기록하는 계측(instrumentation) 계층.

Reluplex.py/DPLL.py/DPLL_T.py는 기존에도 `deadline: Optional[float] = None`
패턴으로 선택적 기능을 옵트인시켜왔다(안 넘기면 그 기능이 그냥 없는 것처럼
동작). `trace: Optional[SolveTrace] = None`도 동일한 패턴이라, trace를 안
넘기면 계측 코드가 전혀 실행되지 않아 기존 호출부와 성능에 영향이 없다.

이 모듈 자체는 이벤트를 어떻게 그릴지 전혀 모른다 — visualization/NetworkLayout.py의
node_values나 시간 프로파일 차트는 SolveTrace.events를 순회해서 만드는
별도 시각화 코드의 몫이다.

사용법 (다른 모듈에서)
----------------------
    from visualization.SolveTrace import SolveTrace, span

    trace = SolveTrace()
    dpll_t_detailed(formula, trace=trace)   # DPLL_T.py에 trace 옵션 추가 예정

    # 구간별로 걸린 시간 합계
    from collections import defaultdict
    totals = defaultdict(float)
    for ev in trace.events:
        if ev.duration is not None:
            totals[ev.component] += ev.duration

    # 특정 시점에 "현재 진행 중"인 이벤트만 보고 싶으면 t_end is None인 것만 필터
    # (완주한 트레이스에서는 보통 없지만, 타임아웃/예외로 중간에 끊긴 경우 남을 수 있음)
    still_open = [ev for ev in trace.events if ev.t_end is None]

    # ReLU split 이벤트만 (layer, neuron) 단위로 누적 카운트
    from visualization.NetworkLayout import parse_neuron_var
    from collections import Counter
    split_counts = Counter()
    for ev in trace.events:
        if ev.component == "reluplex_split" and ev.branch_x is not None:
            key = parse_neuron_var(ev.branch_x)
            if key is not None:
                split_counts[key] += 1

직접 계측 코드를 추가할 때는 span()을 쓴다 (trace가 None이면 아무 것도
안 하는 컨텍스트 매니저):

    with span(trace, "bcp"):
        cnf = unit_propagation(cnf, asn, deadline)

    with span(trace, "reluplex_split", depth=depth, branch_x=branch_x):
        r1, sat1 = _rec(bounds1, depth + 1, row_defs1)
        ...
        r2, sat2 = _rec(bounds2, depth + 1, row_defs2)
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from time import monotonic
from typing import Any, Dict, Iterator, List, Optional


@dataclass
class TraceEvent:
    """계측 이벤트 하나. t_end가 None이면 아직 끝나지 않은(현재 진행 중인) 이벤트다."""

    component: str  # "bcp" | "simplex" | "reluplex_repair" | "reluplex_split"
    t_start: float
    t_end: Optional[float] = None
    depth: Optional[int] = None
    branch_x: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> Optional[float]:
        if self.t_end is None:
            return None
        return self.t_end - self.t_start


@dataclass
class SolveTrace:
    """한 번의 DPLL(T) 실행 동안 쌓인 TraceEvent 목록."""

    events: List[TraceEvent] = field(default_factory=list)

    def enter(
        self,
        component: str,
        *,
        depth: Optional[int] = None,
        branch_x: Optional[str] = None,
        **meta: Any,
    ) -> TraceEvent:
        """이벤트를 시작 시각으로 기록하고 append한다. 나중에 exit()에 그대로 넘길 것."""
        ev = TraceEvent(component=component, t_start=monotonic(), depth=depth, branch_x=branch_x, meta=dict(meta))
        self.events.append(ev)
        return ev

    def exit(self, event: TraceEvent) -> None:
        """enter()가 반환한 이벤트에 종료 시각을 채운다."""
        event.t_end = monotonic()

    def open_events(self) -> List[TraceEvent]:
        """아직 t_end가 채워지지 않은(진행 중이거나, 타임아웃/예외로 중간에 끊긴) 이벤트들."""
        return [e for e in self.events if e.t_end is None]


@contextmanager
def span(
    trace: Optional[SolveTrace],
    component: str,
    *,
    depth: Optional[int] = None,
    branch_x: Optional[str] = None,
    **meta: Any,
) -> Iterator[Optional[TraceEvent]]:
    """trace가 None이면 아무 것도 하지 않는 컨텍스트 매니저. 블록이 예외로
    빠져나가도(타임아웃 등) finally에서 exit()이 호출되어 이벤트가 열린 채로
    남지 않는다."""
    if trace is None:
        yield None
        return
    ev = trace.enter(component, depth=depth, branch_x=branch_x, **meta)
    try:
        yield ev
    finally:
        trace.exit(ev)
