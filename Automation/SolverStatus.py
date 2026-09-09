"""솔버의 확정 결과와 자원 한도 초과를 구분하는 공통 자료구조."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from time import monotonic
from typing import Any, Dict, Optional


class SolverStatus(str, Enum):
    SAT = "SAT"
    UNSAT = "UNSAT"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class SolverResult:
    """DPLL(T) 실행 결과와 종료 사유를 함께 보관한다."""

    status: SolverStatus
    model: Optional[Dict[str, float]] = None
    reason: Optional[str] = None
    rounds: int = 0
    elapsed_seconds: float = 0.0
    # 이론 솔버가 남긴 부가 정보 (예: recursion_limit_hits).
    # UNKNOWN 사유가 TIMEOUT으로 뭉뚱그려질 때 진짜 병목을 구분하는 용도.
    theory_stats: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "reason": self.reason,
            "rounds": self.rounds,
            "elapsed_seconds": self.elapsed_seconds,
            "theory_stats": dict(self.theory_stats),
        }


class SolverLimitReached(RuntimeError):
    """시간이나 반복 한도 때문에 결론을 확정할 수 없을 때 사용한다."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def check_deadline(deadline: Optional[float]) -> None:
    """설정된 종료 시각을 지났으면 UNKNOWN으로 전파할 예외를 발생시킨다."""

    if deadline is not None and monotonic() >= deadline:
        raise SolverLimitReached("TIMEOUT")
