"""Time-based motion prediction and one-to-one minimum-cost assignment."""
import math
import numpy as np


def box_iou(a, b):
    overlap = max(0, min(a['x2'], b['x2']) - max(a['x'], b['x'])) * max(0, min(a['y2'], b['y2']) - max(a['y'], b['y']))
    area_a = (a['x2'] - a['x']) * (a['y2'] - a['y'])
    area_b = (b['x2'] - b['x']) * (b['y2'] - b['y'])
    return overlap / max(1, area_a + area_b - overlap)


def predicted_box(track, pts_ns):
    elapsed = max(0., (pts_ns - track['last_pts_ns']) / 1e9)
    box = dict(track['box'])
    for key in ('x', 'x2'):
        box[key] += track['vx'] * elapsed
    return box


def minimum_assignment(cost):
    """Hungarian assignment, including a separate unmatched slot per row."""
    if not cost.size:
        return []
    rows, columns = cost.shape
    matrix = np.concatenate((cost, np.full((rows, rows), 2.5)), axis=1)
    n, m = matrix.shape
    u, v = np.zeros(n + 1), np.zeros(m + 1)
    p, way = np.zeros(m + 1, dtype=int), np.zeros(m + 1, dtype=int)
    for i in range(1, n + 1):
        p[0], j0 = i, 0
        best, used = np.full(m + 1, np.inf), np.zeros(m + 1, dtype=bool)
        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], np.inf, 0
            for j in range(1, m + 1):
                if used[j]:
                    continue
                value = matrix[i0 - 1, j - 1] - u[i0] - v[j]
                if value < best[j]:
                    best[j], way[j] = value, j0
                if best[j] < delta:
                    delta, j1 = best[j], j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    best[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0], j0 = p[j1], j1
            if j0 == 0:
                break
    return [(p[j] - 1, j - 1) for j in range(1, columns + 1) if p[j] and matrix[p[j] - 1, j - 1] < 2.5]


def associate(tracks, blobs, pts_ns):
    cost = np.full((len(tracks), len(blobs)), 10000.)
    for i, track in enumerate(tracks):
        predicted = predicted_box(track, pts_ns)
        px, py = (predicted['x'] + predicted['x2']) / 2, (predicted['y'] + predicted['y2']) / 2
        width, height = predicted['x2'] - predicted['x'], predicted['y2'] - predicted['y']
        scale = max(12., min(55., math.hypot(width, height) * .7))
        for j, blob in enumerate(blobs):
            cx, cy = (blob['x'] + blob['x2']) / 2, (blob['y'] + blob['y2']) / 2
            bw, bh = blob['x2'] - blob['x'], blob['y2'] - blob['y']
            distance = math.hypot(cx - px, cy - py)
            if distance > 55 or abs(cy - py) > max(8, .45 * (height + bh)):
                continue
            if not .22 <= bw / width <= 4.5 or not .4 <= bh / height <= 2.5:
                continue
            shape = abs(math.log(bw / width)) + abs(math.log(bh / height))
            cost[i, j] = distance / scale + .35 * (1 - box_iou(predicted, blob)) + .2 * shape
    return {j: tracks[i] for i, j in minimum_assignment(cost)}
