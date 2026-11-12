"""Small file://-compatible data snapshots for already-open report pages."""
import json
from pathlib import Path

from visualization.AtomicFile import write_atomic_text


def snapshot_path(path):
    path = Path(path)
    return path.with_name(f"{path.stem}.snapshot.html")


def write_snapshot(path, payload, message_type):
    message = json.dumps({"type": message_type, "data": payload}, ensure_ascii=False).replace("<", "\\u003c")
    document = '<!doctype html><meta charset="utf-8"><script>parent.postMessage(' + message + ',"*");</script>'
    return write_atomic_text(snapshot_path(path), document)
