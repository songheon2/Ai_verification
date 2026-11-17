"""Small file://-compatible hover maps for raster ReLU neuron plots."""
import json
from bisect import bisect_right
from collections import Counter, defaultdict
from math import sqrt
from pathlib import Path

from visualization.AtomicFile import write_atomic_text

HOVER_SCRIPT = Path(__file__).with_suffix('.js').read_text(encoding='utf-8')


def write_relu_hover_map(figure, image_path):
    geometry = getattr(figure, '_relu_hover_geometry', None)
    if geometry is None:
        return
    axis, neurons, sizes, events, round_starts = geometry
    boundaries = sorted((int(item['time_ns']), int(item['round'])) for item in round_starts
                        if item.get('time_ns') is not None and item.get('round') is not None)
    stamps = [stamp for stamp, _ in boundaries]
    known_rounds = sorted({index for _, index in boundaries})
    counts = defaultdict(Counter)
    for event in events:
        if event.event != '+':
            continue
        position = bisect_right(stamps, event.time_ns) - 1
        round_index = boundaries[position][1] if position >= 0 else None
        counts[(event.layer, event.index)][round_index] += 1
    width, height = figure.bbox.width, figure.bbox.height
    points = []
    for (layer, neuron), size in zip(neurons, sizes):
        x, y = axis.transData.transform((layer, neuron))
        radius = sqrt(size) * figure.dpi / 144 + 2
        per_round = counts[(layer, neuron)]
        round_splits = [dict(round=index, count=per_round[index]) for index in known_rounds]
        if None in per_round:
            round_splits.append(dict(round=None, count=per_round[None]))
        points.append(dict(layer=layer, neuron=neuron, roundSplits=round_splits,
                           x=float(x/width), y=float(1-y/height),
                           rx=radius/width, ry=radius/height))
    image_path = Path(image_path)
    payload = json.dumps(dict(type='relu-neuron-hover', image=image_path.name, points=points))
    return write_atomic_text(image_path.with_suffix('.hover.js'), 'window.postMessage('+payload+',"*");')
