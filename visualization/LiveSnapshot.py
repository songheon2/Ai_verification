"""Small file://-compatible data snapshots for already-open report pages."""
import json
from pathlib import Path

from visualization.AtomicFile import write_atomic_text


INLINE_DATA_LIMIT = 256 * 1024


def snapshot_path(path):
    path = Path(path)
    return path.with_name(f"{path.stem}.snapshot.html")


def write_snapshot(path, payload, message_type, *, serialized_payload=None):
    data = serialized_payload if serialized_payload is not None else json.dumps(payload, ensure_ascii=False)
    message = ('{"type":' + json.dumps(message_type) + ',"data":' + data + '}').replace("<", "\\u003c")
    document = '<!doctype html><meta charset="utf-8"><script>parent.postMessage(' + message + ',"*");</script>'
    return write_atomic_text(snapshot_path(path), document)


def initial_page_data(path, payload, message_type, preview, *, live_snapshots=False, snapshot_only=False):
    """Keep large JSON out of the visible page's blocking startup script."""
    if live_snapshots:
        payload["snapshotFile"] = snapshot_path(path).name
    data = None if snapshot_only else json.dumps(payload, ensure_ascii=False)
    deferred = data is not None and len(data) > INLINE_DATA_LIMIT
    if deferred and not live_snapshots:
        payload["snapshotFile"] = snapshot_path(path).name
        data = json.dumps(payload, ensure_ascii=False)
    if live_snapshots or deferred:
        write_snapshot(path, payload, message_type, serialized_payload=data)
    if snapshot_only:
        return None
    if deferred:
        initial = preview(payload)
        initial.update(status="RUNNING", deferredData=True, snapshotFile=snapshot_path(path).name)
        initial.pop("snapshotVersion", None)
        return json.dumps(initial, ensure_ascii=False)
    return data
