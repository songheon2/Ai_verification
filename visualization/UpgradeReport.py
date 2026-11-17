"""Upgrade completed legacy reports without rerunning the neural verifier."""
from html import escape
import json
from pathlib import Path
import tempfile

from visualization.AtomicFile import replace_atomic_file, write_atomic_text
from visualization.LiveSnapshot import initial_page_data, snapshot_path


def _report_summary(prefix):
    decoder = json.JSONDecoder()
    position = prefix.index('"report"') + len('"report"')
    position = prefix.index('{', position) + 1
    summary = {}
    while position < len(prefix):
        try:
            position += len(prefix[position:]) - len(prefix[position:].lstrip(' \r\n,'))
            key, position = decoder.raw_decode(prefix, position)
            if key not in {'available', 'settings', 'totals'}:
                break
            position = prefix.index(':', position) + 1
            position += len(prefix[position:]) - len(prefix[position:].lstrip())
            summary[key], position = decoder.raw_decode(prefix, position)
        except (ValueError, TypeError):
            break
    return summary


def upgrade_saved_report(path):
    """Only completed runs are eligible; leave a running writer's files alone."""
    path = Path(path)
    result = json.loads((path.parent / 'result.json').read_text(encoding='utf-8-sig'))
    status = result.get('solver', {}).get('status', result.get('status', 'UNKNOWN'))
    reason = result.get('solver', {}).get('reason')
    with path.open('r', encoding='utf-8') as source:
        prefix = source.read(64 * 1024)
    if '"deferredData": true' in prefix and snapshot_path(path).exists():
        text = path.read_text(encoding='utf-8')
        position = text.index('const INITIAL_DATA = ') + len('const INITIAL_DATA = ')
        payload, _ = json.JSONDecoder().raw_decode(text, position)
        if not path.stem.endswith('_mechanisms'):
            payload['modelLayerSizes'] = result.get('model', {}).get('layer_sizes', [])
        data = json.dumps(payload, ensure_ascii=False).replace('<', '\\u003c')
        if path.stem.endswith('_mechanisms'):
            from visualization.SolverMechanisms import _HTML, _SCRIPT
            document = _HTML.replace('__SCRIPT__', _SCRIPT.replace('__DATA__', data))
        else:
            from visualization.ReluSearchTree import _HTML, _TREE_SCRIPT, _TREE_STYLES
            document = _HTML.replace('__SCRIPT__', _TREE_SCRIPT.replace('__DATA__', data)).replace('__STYLES__', _TREE_STYLES)
        return write_atomic_text(path, document.replace('__TITLE__', escape(payload['title'])))
    if not path.stem.endswith('_mechanisms'):
        from visualization.ReluSearchTree import _HTML, _TREE_SCRIPT, _TREE_STYLES, preview_tree
        text = path.read_text(encoding='utf-8')
        position = text.index('const INITIAL_DATA = ') + len('const INITIAL_DATA = ')
        payload, _ = json.JSONDecoder().raw_decode(text, position)
        payload.update(status=status, reason=reason,
                       modelLayerSizes=result.get('model', {}).get('layer_sizes', []))
        data = initial_page_data(path, payload, 'relu-search-tree-snapshot', preview_tree)
        document = _HTML.replace('__SCRIPT__', _TREE_SCRIPT.replace('__DATA__', data.replace('</', '<\\/')))
        document = document.replace('__STYLES__', _TREE_STYLES).replace('__TITLE__', escape(payload['title']))
        return write_atomic_text(path, document)

    # The mechanism payload can exceed 100 MB. Copy the JSON literal in chunks
    # rather than building another full history object graph in memory.
    from visualization.SolverMechanisms import _HTML, _SCRIPT, preview_mechanisms
    marker, ending = 'const INITIAL_DATA = ', ';\n  const SNAPSHOT'
    temporary = None
    try:
        with path.open('r', encoding='utf-8') as source, tempfile.NamedTemporaryFile(
            mode='w', encoding='utf-8', dir=path.parent, prefix='.upgrade-', suffix='.tmp', delete=False,
        ) as destination:
            temporary = Path(destination.name)
            buffer = source.read(64 * 1024)
            position = buffer.index(marker) + len(marker)
            buffer = buffer[position:]
            report = _report_summary(buffer)
            destination.write('<!doctype html><meta charset="utf-8"><script>const data=')
            while True:
                position = buffer.find(ending)
                if position >= 0:
                    metadata = {}
                    for key in ('title', 'panels'):
                        location = buffer.rfind('"' + key + '":', 0, position)
                        if location >= 0:
                            metadata[key], _ = json.JSONDecoder().raw_decode(
                                buffer[location + len(key) + 3:position].lstrip()
                            )
                    destination.write(buffer[:position].replace('<', '\\u003c'))
                    break
                if len(buffer) > len(ending):
                    destination.write(buffer[:-len(ending)].replace('<', '\\u003c'))
                    buffer = buffer[-len(ending):]
                chunk = source.read(1024 * 1024)
                if not chunk:
                    raise ValueError('No complete report payload found')
                buffer += chunk
            destination.write(';data.status=' + json.dumps(status) + ';data.snapshotFile=' + json.dumps(snapshot_path(path).name))
            destination.write(';parent.postMessage({type:"solver-mechanisms-snapshot",data:data},"*");</script>')
        replace_atomic_file(temporary, snapshot_path(path))
        payload = preview_mechanisms({'report': report, 'title': metadata.get('title', 'Reluplex 검증 기법'),
                                     'panels': metadata.get('panels', ['bounds', 'backjump', 'numerical', 'underapprox'])})
        payload.update(status='RUNNING', deferredData=True, snapshotFile=snapshot_path(path).name)
        data = json.dumps(payload, ensure_ascii=False).replace('<', '\\u003c')
        document = _HTML.replace('__TITLE__', escape(payload['title'])).replace('__SCRIPT__', _SCRIPT.replace('__DATA__', data))
        return write_atomic_text(path, document)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reports', nargs='+', type=Path)
    for report in parser.parse_args().reports:
        print(upgrade_saved_report(report))
