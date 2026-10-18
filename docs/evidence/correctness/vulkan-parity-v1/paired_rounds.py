"""Module-local exact paired-round reader. No product/statistical verdicts."""
from collections import defaultdict
import math
import re
import statistics

ROUND = re.compile(r'^# D5_PAIRED_ROUND profile="([^"]+)" arm="([^"]+)" '
    r'round=(\d+) iterations=([\d.]+) seconds_per_call=([\deE+.-]+) '
    r'paired_relative=([\deE+.-]+) slots=(\d+)$', re.M)

def read(text, titles):
    rows = defaultdict(dict)
    for title, arm, number, iterations, elapsed, relative, slots in ROUND.findall(text):
        if title not in titles or arm not in ('Graph complete fragment', 'Direct complete fragment'):
            raise ValueError('unexpected paired round subject')
        number, slots = int(number), int(slots)
        iterations, elapsed, relative = float(iterations), float(elapsed), float(relative)
        if iterations != 6 or slots not in (2, 3) or 6 % slots or not 0 <= number < 24:
            raise ValueError('paired input cycles/round accounting differ')
        if not math.isfinite(elapsed) or elapsed <= 0 or not math.isfinite(relative) or relative <= 0:
            raise ValueError('invalid paired round time')
        key = (title, arm)
        if number in rows[key]: raise ValueError('duplicate paired round')
        rows[key][number] = {'seconds': elapsed, 'iterations': iterations, 'slots': slots,
            'reported_relative': relative}
    result = {}
    for title in titles:
        candidate = rows[(title, 'Graph complete fragment')]
        peer = rows[(title, 'Direct complete fragment')]
        if set(candidate) != set(range(24)) or set(peer) != set(range(24)):
            raise ValueError('missing/partial paired rounds')
        if any(candidate[i]['slots'] != peer[i]['slots'] for i in range(24)):
            raise ValueError('pair input cycles differ')
        ratios = [peer[i]['seconds'] / candidate[i]['seconds'] for i in range(24)]
        result[title] = {'paired_process_ratio': statistics.median(ratios),
            'round_ratios': ratios, 'candidate': candidate, 'peer': peer}
    return result
