"""Проверка «можно ли это смотреть»:
1) вспышки по WCAG 2.3.1 / методике Harding: переходы относительной яркости ≥ 0.10 (тёмное состояние < 0.80)
   по всему кадру и по 9 окнам площадью 1/4 кадра; вспышка = пара противоположных переходов;
   опасно — больше 3 вспышек за любую секунду; цель — не больше 1;
2) красные вспышки: доля насыщенно-красных пикселей;
3) читаемость: контраст текста в каждом видимом пузыре (по layout.json), WCAG: ≥ 4.5:1;
4) общая яркость по секундам.

python3 check_watch.py video.mp4 layout.json
"""
import json
import subprocess
import sys

import numpy as np

VIDEO = sys.argv[1]
LAYOUT = json.load(open(sys.argv[2])) if len(sys.argv) > 2 else []
TLJ = json.load(open(sys.argv[3])) if len(sys.argv) > 3 else None
TV = (TLJ['t']['tvOff'], TLJ['t']['tvOn'] + 0.1) if TLJ else (-1, -1)
EVT = [m['t'] for m in TLJ['msgs'] if m['t'] > 0] if TLJ else []
FADE = TLJ['t']['fadeIn'][1] if TLJ else 0
BLACK = TLJ['t']['black'] if TLJ else 1e9
HITT = ([m['t'] for m in TLJ['msgs'] if m.get('hit')] + [h['t'] for h in TLJ.get('extraHits', [])]) if TLJ else []
S = 576
FPS = 30


def lin(c):
    c = c / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


proc = subprocess.Popen(['ffmpeg', '-v', 'error', '-i', VIDEO, '-vf', f'scale={S}:{S}', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'],
                        stdout=subprocess.PIPE)
lay = {round(x['t'] * 20): x['rects'] for x in LAYOUT}

regions = {'кадр': (0, 0, S, S)}
for i, oy in enumerate((0, S // 4, S // 2)):
    for j, ox in enumerate((0, S // 4, S // 2)):
        regions[f'окно{i}{j}'] = (ox, oy, ox + S // 2, oy + S // 2)
series = {k: [] for k in regions}
red_area = []
mean_lum = []
contrast = []        # (t, msg_index, age, ratio, t0)
f = 0
while True:
    buf = proc.stdout.read(S * S * 3)
    if len(buf) < S * S * 3:
        break
    a = np.frombuffer(buf, np.uint8).reshape(S, S, 3).astype(np.float32)
    L = lin(a)
    Y = L[..., 0] * 0.2126 + L[..., 1] * 0.7152 + L[..., 2] * 0.0722
    for k, (x0, y0, x1, y1) in regions.items():
        series[k].append(float(Y[y0:y1, x0:x1].mean()))
    ssum = L.sum(2) + 1e-6
    red = (L[..., 0] / ssum >= 0.8) & ((L[..., 0] - L[..., 1] - L[..., 2]) * 320 > 20)
    red_area.append(float(red.mean()))
    mean_lum.append(float(Y.mean()))
    t = f / FPS
    step = round(t * 20)
    if abs(t * 20 - step) < 1e-6 and step in lay:
        for r in lay[step]:
            if r['y'] < -0.2 * r['h'] or r['y'] + r['h'] > S:   # пузырь в основном за краем — не мерим
                continue
            x0, y0 = int(max(0, r['x'] + 5)), int(max(0, r['y'] + 5))
            x1, y1 = int(min(S, r['x'] + r['w'] - 5)), int(min(S, r['y'] + r['h'] - 5))
            if x1 - x0 < 20 or y1 - y0 < 14:
                continue
            reg = Y[y0:y1, x0:x1]
            bub, txt = np.percentile(reg, 85), np.percentile(reg, 2)
            contrast.append((t, r['i'], r['age'], (bub + 0.05) / (txt + 0.05), r['t0']))
    f += 1
proc.wait()
n = f
times = np.arange(n) / FPS


def transitions(y, thr=0.10, tol=0.006):
    """монотонные изменения яркости ≥ thr (тёмное < 0.8) — время конца каждого"""
    out = []
    start = y[0]
    ext = y[0]
    direction = 0
    for i in range(1, len(y)):
        v = y[i]
        if direction >= 0 and v >= ext:
            ext = v
            direction = 1 if v > start + tol else direction
            continue
        if direction <= 0 and v <= ext:
            ext = v
            direction = -1 if v < start - tol else direction
            continue
        # разворот?
        if abs(v - ext) > tol:
            if abs(ext - start) >= thr and min(ext, start) < 0.8:
                out.append(((i - 1) / FPS, ext - start))
            start = ext
            ext = v
            direction = 1 if v > start else -1
    if abs(ext - start) >= thr and min(ext, start) < 0.8:
        out.append(((len(y) - 1) / FPS, ext - start))
    return out


print('=== 1. ВСПЫШКИ (WCAG 2.3.1 / Harding) ===')
worst = (0, None, None)
bad = []
for k, y in series.items():
    tr = transitions(np.array(y))
    tt = np.array([x[0] for x in tr])
    for i, t0 in enumerate(tt):
        cnt = int(((tt >= t0) & (tt < t0 + 1.0)).sum())
        flashes = cnt // 2
        if flashes > worst[0]:
            worst = (flashes, k, t0)
        if flashes > 1:
            bad.append((round(float(t0), 2), k, flashes))
print(f'макс. вспышек за 1 с: {worst[0]}  (регион {worst[1]}, с {worst[2] if worst[2] is None else round(worst[2], 2)} с)')
print('ПОРОГ ОПАСНОСТИ: больше 3 в секунду;', 'ПРОВАЛ' if worst[0] > 3 else ('на грани' if worst[0] > 1 else 'ок'))
if bad:
    seen = {}
    for t0, k, fl in bad:
        key = round(t0)
        if key not in seen or seen[key][2] < fl:
            seen[key] = (t0, k, fl)
    print('секунды с >1 вспышкой:', ', '.join(f'{v[0]}с ({v[2]})' for v in sorted(seen.values())))
alltr = transitions(np.array(series['кадр']))
print(f'переходов яркости ≥0.1 по всему кадру: {len(alltr)}')
for t0, d in alltr:
    print(f'   {t0:6.2f} с  {"+" if d > 0 else ""}{d:.2f}')

print('\n=== 2. КРАСНЫЕ ВСПЫШКИ ===')
print(f'макс. доля насыщенно-красного: {max(red_area) * 100:.2f}% кадра', '(ок)' if max(red_area) < 0.05 else '(ПРОВЕРИТЬ)')

print('\n=== 3. ЧИТАЕМОСТЬ (контраст текста в пузырях, WCAG AA ≥ 4.5) ===')
if contrast:
    by_msg = {}
    for t, i, age, cr, t0 in contrast:
        settled = (t >= t0 + 0.35 and not (TV[0] <= t < TV[1]) and not any(h <= t < h + 0.25 for h in HITT)
                   and not any(e <= t < e + 0.35 for e in EVT) and FADE <= t < BLACK)
        by_msg.setdefault(i, []).append((t, age, cr, settled))
    for i in sorted(by_msg):
        rows = [r for r in by_msg[i] if r[3]]
        if not rows:
            continue
        crs = np.array([r[2] for r in rows])
        new = np.array([r[2] for r in rows if r[1] <= 2])
        low = [r for r in rows if r[2] < 4.5]
        print(f'сообщение {i:2d}: мин {crs.min():5.2f}  медиана {np.median(crs):5.2f}'
              f'  (среди 3 новых: мин {new.min() if len(new) else float("nan"):5.2f})'
              f'  <4.5: {len(low) * 0.05:.2f} с' + (f'  первое падение {low[0][0]:.2f} с (возраст {low[0][1]})' if low else ''))

print('\n=== 4. СРЕДНЯЯ ЯРКОСТЬ ПО СЕКУНДАМ (отн. яркость, 0–1) ===')
ml = np.array(mean_lum)
print(' '.join(f'{s}:{ml[s * FPS:(s + 1) * FPS].mean():.2f}' for s in range(int(n / FPS))))
