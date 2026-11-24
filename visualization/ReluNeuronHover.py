"""Small file://-compatible hover maps for raster ReLU neuron plots."""
import json
from bisect import bisect_right
from collections import Counter, defaultdict
from math import sqrt
from pathlib import Path

from visualization.AtomicFile import write_atomic_text

HOVER_SCRIPT = Path(__file__).with_suffix('.js').read_text(encoding='utf-8')


def split_round_boundaries(round_starts):
    return tuple(sorted((int(item['time_ns']), int(item['round'])) for item in round_starts
                        if item.get('time_ns') is not None and item.get('round') is not None))


def count_round_splits(events, boundaries, counts):
    stamps = [stamp for stamp, _ in boundaries]
    for event in events:
        if event.event != '+':
            continue
        position = bisect_right(stamps, event.time_ns) - 1
        round_index = boundaries[position][1] if position >= 0 else None
        counts[(event.layer, event.index)][round_index] += 1


def write_relu_hover_map(figure, image_path):
    geometry = getattr(figure, '_relu_hover_geometry', None)
    if geometry is None:
        return
    axis, neurons, markers, events, round_starts = geometry
    boundaries = split_round_boundaries(round_starts)
    known_rounds = sorted({index for _, index in boundaries})
    counts = getattr(figure, '_relu_hover_counts', None)
    if counts is None:
        counts = defaultdict(Counter)
        count_round_splits(events, boundaries, counts)
    width, height = figure.bbox.width, figure.bbox.height
    points = []
    sizes = markers.get_sizes() if markers is not None else []
    widths = markers.get_linewidths() if markers is not None else []
    for (layer, neuron), size, line_width in zip(neurons, sizes, widths):
        x, y = axis.transData.transform((layer, neuron))
        radius = (sqrt(size) + float(line_width)) * figure.dpi / 144
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
