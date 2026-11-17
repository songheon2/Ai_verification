"""Publish complete reports with bounded retries for Windows file locks."""
import os
from pathlib import Path
import tempfile
from time import sleep


def replace_atomic_file(source, destination):
    for attempt in range(4):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt == 3:
                raise
            sleep(0.05 * (2 ** attempt))


def write_atomic_text(path, document):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.stem}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(document)
        replace_atomic_file(temporary, path)
        return path
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
