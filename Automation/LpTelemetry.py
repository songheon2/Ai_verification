"""Engine-independent LP metrics. Unknown counts are never invented.

Adapters may expose a per-call ``simplex_metrics`` dictionary. Built-in Python
Simplex callbacks and GLPK's iteration counter are supported without changing
their solving algorithm. Bound snapshots are observations, NOT pivot traces.
"""
from math import isfinite


METRIC_FLAGS = {"pivots": "pivot_count_available", "iterations": "iteration_count_available",
                "cycles": "cycle_count_available"}
METRIC_META = tuple(METRIC_FLAGS.values()) + (
    "work_counts_available", "lp_engine", "metrics_source", "bound_violation_scope", "model_validation")


def count(value):
    return value if type(value) is int and value >= 0 else None


def metric_available(item, field):
    flag = item.get(METRIC_FLAGS[field], item.get("work_counts_available", True))
    return flag is True and count(item.get(field, 0)) is not None


def observe_bounds(model, bounds):
    """Check only actually available values; never manufacture a zero start."""
    if not isinstance(model, dict):
        return []
    result = []
    for name, bound in bounds.items():
        value = model.get(name)
        if type(value) not in (float, int) or not isfinite(value):
            continue
        lo, hi = (bound.lower, bound.upper) if hasattr(bound, "lower") else bound
        if ((isfinite(lo) and value < lo - 1e-9 * max(1., abs(lo))) or
                (isfinite(hi) and value > hi + 1e-9 * max(1., abs(hi)))):
            result.append({"variable": str(name), "value": value,
                           "lower": lo if isfinite(lo) else None, "upper": hi if isfinite(hi) else None})
    return result


class NativeCallbacks:
    def __init__(self, logger, call_id):
        self.logger, self.call_id = logger, call_id
        self.metrics = {}
        self.cycles = 0

    def simplex_start(self, **_meta):
        return self.call_id

    def simplex_iteration(self, _call_id, iteration, **meta):
        self.logger.simplex_iteration(self.call_id, iteration, **meta)

    def simplex_pivot(self, _call_id, iteration, **meta):
        self.logger.simplex_pivot(self.call_id, iteration, **meta)

    def simplex_cycle(self, _call_id, **meta):
        self.cycles += 1
        self.logger.simplex_cycle(self.call_id, **meta)

    def simplex_end(self, _call_id, _result, *, iterations, pivots, **_meta):
        self.metrics = {"iterations": iterations, "pivots": pivots, "cycles": self.cycles}


class ExternalMetrics:
    def __init__(self, owner, logger, call_id, context):
        self.owner, self.logger, self.call_id = owner, logger, call_id
        self.native = None
        self.saved = None
        if isinstance(getattr(owner, "simplex_metrics", None), dict):
            owner.simplex_metrics = {}  # Never reuse the previous call after an early timeout.
        self.glpk_before = self._glpk_counter()
        self.z3_before = self._z3_counter()
        kwargs = getattr(owner, "simplex_kwargs", None)
        if isinstance(kwargs, dict):
            self.saved = {key: (key in kwargs, kwargs.get(key)) for key in ("progress", "progress_context")}
            self.native = NativeCallbacks(logger, call_id)
            kwargs.update(progress=self.native, progress_context=context)

    def _glpk_counter(self):
        glp, lp = getattr(self.owner, "glp", None), getattr(self.owner, "lp", None)
        if glp is not None and lp is not None:
            try:
                return count(glp.glp_get_it_cnt(lp))
            except Exception:
                pass
        return None

    def _z3_counter(self):
        if getattr(self.owner, "name", None) != "z3":
            return None
        solver = getattr(self.owner, "solver", None)
        try:
            statistics = solver.statistics()
            for key in statistics.keys():
                if key.replace("-", " ").replace("_", " ") == "arith pivots":
                    return count(statistics.get_key_value(key))
        except Exception:
            pass
        return None

    def finish(self):
        metrics, source = {}, "unavailable"
        if self.native is not None:
            metrics, source = self.native.metrics, "python-simplex-callbacks"
            kwargs = self.owner.simplex_kwargs
            for key, (present, value) in self.saved.items():
                if present:
                    kwargs[key] = value
                else:
                    kwargs.pop(key, None)
        else:
            declared = getattr(self.owner, "simplex_metrics", None)
            if isinstance(declared, dict):
                metrics = declared
                source = declared.get("source", "engine-adapter")
                if not isinstance(source, str):
                    source = "engine-adapter"
            after = self._glpk_counter()
            if self.glpk_before is not None and after is not None and after >= self.glpk_before:
                metrics = {**metrics, "iterations": after - self.glpk_before}
                source = "glpk-iteration-counter"
            z3_after = self._z3_counter()
            if z3_after is not None:
                baseline = self.z3_before if self.z3_before is not None else 0
                if z3_after >= baseline:
                    metrics = {**metrics, "pivots": z3_after - baseline}
                    source = "z3-arithmetic-pivot-counter"
        result = {field: count(metrics.get(field)) for field in METRIC_FLAGS}
        result.update({METRIC_FLAGS[field]: result[field] is not None for field in METRIC_FLAGS})
        result.update(metrics_source=source, work_counts_available=(result["pivots"] is not None and
                                                                  result["iterations"] is not None),
                      bound_violation_scope="solver_iteration" if self.native else "call_boundary")
        if "glpk_profile" in metrics:
            result["glpk_profile"] = metrics["glpk_profile"]
        return result
