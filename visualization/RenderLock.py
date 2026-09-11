"""FIFO rendering gate shared by the realtime Matplotlib workers."""
from contextlib import contextmanager
from functools import wraps
from threading import Condition


_condition = Condition()
_next_ticket = 0
_serving = 0


@contextmanager
def render_turn():
    # Matplotlib's internal lock is not fair: a busy ReLU worker can
    # repeatedly reacquire it before the solver dashboard gets to draw.
    global _next_ticket, _serving
    with _condition:
        ticket = _next_ticket
        _next_ticket += 1
        _condition.wait_for(lambda: ticket == _serving)
    try:
        yield
    finally:
        with _condition:
            _serving += 1
            _condition.notify_all()


def serialized_render(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        with render_turn():
            return function(*args, **kwargs)
    return wrapped
