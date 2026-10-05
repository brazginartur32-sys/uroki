"""Постобработка кадров (как в референсе): наплыв из чёрного, затемнение+размытие при финальном наезде,
глитч (вспышки, сине-голубая перекраска, RGB-сдвиг, рваные полосы), обрыв в чёрное.

python3 post.py frames scene_t.json out_noaudio.mp4
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

FRAMES = Path(sys.argv[1])
T = json.load(open(sys.argv[2]))['t']
OUT = sys.argv[3]
FPS = 30
files = sorted(FRAMES.glob('*.jpg'))
W = H = 1080


def blur(a, r):
    if r <= 0.05:
        return a
    im = Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))
    return np.asarray(im.filter(ImageFilter.GaussianBlur(float(r))), dtype=np.float32) / 255


def hblur(a, r):
    """горизонтальный смаз"""
    if r < 1:
        return a
    r = int(r)
    c = np.cumsum(np.pad(a, ((0, 0), (r + 1, r), (0, 0)), mode='edge'), axis=1)
    return (c[:, 2 * r + 1:] - c[:, :-2 * r - 1]) / (2 * r + 1)


def lum(a):
    return a[..., 0] * .299 + a[..., 1] * .587 + a[..., 2] * .114


def smooth(e0, e1, x):
    k = np.clip((x - e0) / (e1 - e0), 0, 1)
    return k * k * (3 - 2 * k)


def saturate(a, s):
    L = lum(a)[..., None]
    return L + (a - L) * s


def masks(a):
    L = lum(a)
    white = smooth(0.90, 0.965, L)[..., None]      # пузыри
    dark = (1 - smooth(0.22, 0.62, L))[..., None]  # текст
    return white, dark


def recolor(a, bg, bubble, ink):
    w, d = masks(a)
    out = np.array(bg) * (1 - w) + np.array(bubble) * w
    return out * (1 - d) + np.array(ink) * d


def rgb_split(a, dx):
    dx = int(dx)
    out = a.copy()
    out[..., 0] = np.roll(a[..., 0], -dx, axis=1)
    out[..., 2] = np.roll(a[..., 2], dx, axis=1)
    return out


def slices(a, seed, n=7, maxdx=60):
    r = np.random.default_rng(seed)
    out = a.copy()
    for _ in range(n):
        y = r.integers(0, H - 40)
        h = r.integers(12, 90)
        out[y:y + h] = np.roll(out[y:y + h], int(r.integers(-maxdx, maxdx)), axis=1)
    return out


def streak(a, y0, y1, color, alpha):
    out = a.copy()
    out[y0:y1] = out[y0:y1] * (1 - alpha) + np.array(color) * alpha
    return out


BLUE = (0.10, 0.20, 0.98)
BLUE2 = (0.22, 0.42, 0.96)
CYAN = (0.72, 0.93, 0.96)
PINK = (0.86, 0.42, 0.80)


def glitch(a, k):
    """k — номер 0.05-секундного шага от начала глитча (как кадры референса при 20 fps)"""
    if k == 0:
        return blur(saturate(a, 1.6) * 0.86, 4)
    if k in (1, 3):  # белая пересвеченная вспышка, текст розовый
        return blur(recolor(a, (0.97, 0.97, 0.98), (1, 1, 1), PINK) * 0.5 + 0.5, 3)
    if k in (2, 8, 13):
        return blur(1 - (1 - a) * 0.45, 2.5)
    if k == 4:
        return blur(recolor(a, CYAN, (0.97, 0.99, 1), (0.75, 0.5, 0.85)), 2)
    if k in (5, 6, 7):
        return blur(recolor(a, (0.86, 0.96, 0.97), (1, 1, 1), (0.82, 0.62, 0.86)), 2.2)
    if k in (9, 16, 19):
        return rgb_split(blur(recolor(a, BLUE, (1, 1, 1), (0.12, 0.12, 0.16)), 1.2), 7)
    if k == 10:
        b = blur(recolor(a, CYAN, (1, 1, 1), (0.4, 0.4, 0.5)), 1.5)
        return streak(streak(b, 760, 790, (0.9, 0.3, 0.85), .6), 980, 1080, (0.9, 0.3, 0.85), .45)
    if k in (11, 21):
        return slices(rgb_split(recolor(a, BLUE, (1, 1, 1), (0.1, 0.1, 0.15)), 9), seed=k)
    if k == 12:
        b = rgb_split(blur(recolor(a, BLUE2, (0.95, 0.98, 1), (0.25, 0.25, 0.4)), 1.5), 6)
        return streak(b, 1000, 1080, (0.95, 0.25, 0.85), .7)
    if k == 14:
        return rgb_split(recolor(a, BLUE, (1, 1, 1), (0.1, 0.1, 0.14)), 5)
    if k == 15:
        return blur(recolor(a, BLUE2, (1, 1, 1), (0.2, 0.2, 0.3)), 2)
    if k == 17:
        return rgb_split(recolor(a, BLUE, (1, 1, 1), (0.08, 0.08, 0.12)), 12)
    if k == 18:
        return hblur(recolor(a, BLUE, (1, 1, 1), (0.1, 0.1, 0.15)), 28)
    if k == 20:
        return rgb_split(recolor(a, (0.12, 0.26, 1.0), (1, 1, 1), (0.05, 0.05, 0.1)), 4)
    return recolor(a, BLUE, (1, 1, 1), (0.1, 0.1, 0.15))


ff = subprocess.Popen(['ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                       # лёгкая «перезаливочная» мягкость с пересветом контуров, как у референса
                       '-vf', 'gblur=sigma=0.9,unsharp=7:7:0.9:5:5:0.0,eq=saturation=1.08:contrast=1.03',
                       '-c:v', 'libx264', '-preset', 'medium', '-crf', '16', '-pix_fmt', 'yuv420p', OUT], stdin=subprocess.PIPE)

for i, f in enumerate(files):
    t = i / FPS
    if t >= T['black']:
        a = np.zeros((H, W, 3), np.float32)
    else:
        a = np.asarray(Image.open(f).convert('RGB'), dtype=np.float32) / 255
        if t < T['fadeIn'][1]:
            a = a * (t / T['fadeIn'][1])
        if T['zoom'] <= t < T['glitch']:
            k = float(np.clip((t - T["zoom"]) / (T["glitch"] - T["zoom"]), 0, 1))
            a = blur(saturate(a, 1 + 0.6 * k) * (1 - 0.14 * k), 4.5 * k ** 1.3)
            if k > 0.6:
                a = rgb_split(a, (k - 0.6) * 10)
        elif t >= T['glitch']:
            a = glitch(a, int((t - T['glitch']) / 0.05))
    ff.stdin.write((np.clip(a, 0, 1) * 255).astype(np.uint8).tobytes())
    if i % 90 == 0:
        print(f'пост {i}/{len(files)}', flush=True)
ff.stdin.close()
ff.wait()
print('готово:', OUT)
