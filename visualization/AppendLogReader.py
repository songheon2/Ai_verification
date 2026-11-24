"""Read complete new JSONL lines from an append-only solver log."""
from contextlib import contextmanager
import os
from pathlib import Path


class AppendLogReader:
    def __init__(self, path):
        self.path = Path(path)
        self._reset()

    def _reset(self):
        self._identity = None
        self._offset = self._line_no = 0
        self._anchor = b""

    def _new_lines(self, stream, size):
        stream.seek(self._offset)
        while self._offset < size:
            line = stream.readline(size - self._offset)
            if not line.endswith(b"\n"):
                break
            self._offset += len(line)
            self._line_no += 1
            yield self._line_no, line.decode("utf-8")

    @contextmanager
    def read(self):
        try:
            stream = self.path.open("rb")
        except FileNotFoundError:
            self._reset()
            yield True, ()
            return
        with stream:
            stat = os.fstat(stream.fileno())
            identity = (stat.st_dev, stat.st_ino)
            reset = identity != self._identity or stat.st_size < self._offset
            if not reset and self._offset:
                stream.seek(self._offset - len(self._anchor))
                reset = stream.read(len(self._anchor)) != self._anchor
            if reset:
                self._reset()
            self._identity = identity
            try:
                yield reset, self._new_lines(stream, stat.st_size)
                stream.seek(max(0, self._offset - 64))
                self._anchor = stream.read(min(64, self._offset))
            except BaseException:
                self._reset()
                raise
