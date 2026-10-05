"""Постобработка кадров (по замерам референса):
- картинка 20 fps внутри 30 fps (каждый третий кадр — дубль), как у оригинала;
- наплыв из чёрного;
- при финальном наезде фон и полутона холодеют и темнеют, пузыри остаются белыми, текст темнеет, зерно;
- глитч по кадрам референса (белые/голубые вспышки с маджентовым рваным текстом, стально-синий → ультрамарин);
- тёмный кадр и обрыв в чёрное;
- «перезалив»: 576p, ореолы от шарпа, сжатие, апскейл до 1080.

python3 post.py frames timeline.json out_noaudio.mp4
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

FRAMES = Path(sys.argv[1])
TL = json.load(open(sys.argv[2]))
T = TL['t']
OUT = sys.argv[3]
FPS = 30
files = sorted(FRAMES.glob('*.jpg'))
W = H = 1080
rng = np.random.default_rng(3)


def load(i):
    i = max(0, min(len(files) - 1, i))
    return np.asarray(Image.open(files[i]).convert('RGB'), dtype=np.float32) / 255


def blur(a, r):
    if r <= 0.05:
        return a
    im = Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))
    return np.asarray(im.filter(ImageFilter.GaussianBlur(float(r))), dtype=np.float32) / 255


def unsharp(a, r=2.0, amt=0.6):
    return a + (a - blur(a, r)) * amt


def hblur(a, r):
    r = int(r)
    if r < 1:
        return a
    c = np.cumsum(np.pad(a, ((0, 0), (r + 1, r), (0, 0)), mode='edge'), axis=1)
    return (c[:, 2 * r + 1:] - c[:, :-2 * r - 1]) / (2 * r + 1)


def lum(a):
    return a[..., 0] * .299 + a[..., 1] * .587 + a[..., 2] * .114


def smooth(e0, e1, x):
    k = np.clip((x - e0) / (e1 - e0), 0, 1)
    return k * k * (3 - 2 * k)


def masks(a):
    L = lum(a)
    white = smooth(0.90, 0.955, L)[..., None]           # пузыри
    dark = (1 - smooth(0.55, 0.78, L))[..., None]       # текст и время
    return white, dark


def grain(a, s):
    return a + rng.normal(0, s, a.shape[:2])[..., None].astype(np.float32)


def bloom(a, w, amt=0.25):
    return a + blur(w.repeat(3, 2) * 0.6, 6) * amt


def fringe(a, w):
    """тонкая оранжевая кайма под пузырями"""
    below = np.roll(w, 3, axis=0)
    e = np.clip(below - w, 0, 1)
    return a * (1 - e * .7) + np.array([0.95, 0.5, 0.25]) * e * .7


def recolor(a, bg, bubble, ink, bold=False):
    w, d = masks(a)
    if bold:
        d = np.clip(blur(d.repeat(3, 2), 1.2)[..., :1] * 1.6, 0, 1)
    out = np.array(bg) * (1 - w) + np.array(bubble) * w
    out = out * (1 - d) + np.array(ink) * d
    return out, w, d


def torn_text(a, w, d, ink, seed):
    """маджентовый рваный текст: строки смазаны по горизонтали, куски выпадают"""
    r = np.random.default_rng(seed)
    dd = d.copy()
    for _ in range(14):
        y = r.integers(0, H - 20)
        h = r.integers(4, 22)
        dd[y:y + h] *= r.uniform(0, 0.4)
    dd = hblur(dd.repeat(3, 2), r.integers(8, 18))[..., :1]
    out = a * (1 - dd) + np.array(ink) * dd
    # белые горизонтальные штрихи от краёв пузырей
    streaks = hblur(w.repeat(3, 2), 30)[..., :1] - w
    out = out + np.clip(streaks, 0, 1) * 0.35
    return out


def vlines(a, seed, amt=0.05):
    r = np.random.default_rng(seed)
    cols = r.normal(0, 1, W).astype(np.float32)
    cols = np.convolve(cols, np.ones(3) / 3, mode='same')
    return a + cols[None, :, None] * amt


def band_shift(a, seed, n=6, maxdx=40):
    r = np.random.default_rng(seed)
    out = a.copy()
    for _ in range(n):
        y = r.integers(0, H - 30)
        h = r.integers(8, 70)
        out[y:y + h] = np.roll(out[y:y + h], int(r.integers(-maxdx, maxdx)), axis=1)
    return out


def rgb_split(a, dx):
    dx = int(dx)
    out = a.copy()
    out[..., 0] = np.roll(a[..., 0], -dx, axis=1)
    out[..., 2] = np.roll(a[..., 2], dx, axis=1)
    return out


def zoom_grade(a, k):
    """холодный синий фон, белые пузыри, тёмный жирный текст"""
    w, d = masks(a)
    mid = 1 - w
    fac = np.array([1 - 0.31 * k, 1 - 0.21 * k, 1 - 0.10 * k])
    out = a * (w + mid * fac)
    out = out * (1 - d * 0.6 * k)
    # чуть насыщеннее жёлтые пузыри и контраст
    L = lum(out)[..., None]
    out = L + (out - L) * (1 + 0.3 * k)
    out = (out - 0.5) * (1 + 0.17 * k) + 0.5
    out = bloom(out, w, 0.15 * k)
    return unsharp(out, 2.0, 0.5 * k), w


MAG = (0.86, 0.55, 0.84)


def g_white(a, seed, tint=(0.97, 0.97, 0.98)):
    out, w, d = recolor(a, tint, (1, 1, 1), (1, 1, 1))
    out = torn_text(out, w, d, MAG, seed)
    return blur(vlines(out, seed, 0.03), 1.5)


def g_cyan(a, bg, seed):
    out, w, d = recolor(a, bg, (0.98, 1, 1), (1, 1, 1))
    out = torn_text(out, w, d, MAG, seed)
    out = bloom(out, w, 0.3)
    return blur(vlines(out, seed, 0.04), 1.2)


def g_blue(a, bg, seed, shift=True):
    out, w, d = recolor(a, bg, (1, 1, 1), (0.04, 0.04, 0.07), bold=True)
    out = fringe(out, w)
    out = bloom(out, w, 0.3)
    if shift:
        out = band_shift(hblur(out, 3), seed)
    return unsharp(rgb_split(out, 2), 2.5, 0.8)


def c(rgb):
    return tuple(x / 255 for x in rgb)


def glitch(a, k, base_graded):
    """k — номер 20-кадрового шага от T['glitch'] (19.20, 19.25, ...)"""
    if k == 0:
        return base_graded
    if k in (1,):
        return g_white(a, 101)
    if k == 2:   # обычный кадр наезда сквозь белый засвет, смазан по горизонтали
        return hblur(base_graded * 0.55 + 0.45, 18)
    if k == 3:
        return g_white(a, 103, (0.93, 0.98, 0.98))
    if k == 4:   # стально-синий читаемый кадр
        out, w, d = recolor(a, c((120, 152, 200)), (0.98, 0.98, 0.99), (0.06, 0.06, 0.1), bold=True)
        return unsharp(bloom(out, w, .2), 2, .6)
    cy = {5: (204, 234, 236), 6: (182, 237, 235), 7: (165, 236, 236), 8: (143, 235, 236), 10: (103, 231, 237), 13: (30, 165, 235)}
    if k in cy:
        if k == 13:
            return g_blue(a, c(cy[k]), 113)
        return g_cyan(a, c(cy[k]), 100 + k)
    bl = {9: (76, 141, 200), 11: (43, 120, 191), 12: (27, 96, 193), 14: (22, 54, 187), 15: (24, 56, 232),
          16: (25, 18, 203), 17: (25, 18, 210), 19: (28, 20, 225), 20: (28, 20, 228), 21: (7, 5, 245)}
    if k in bl:
        out = g_blue(a, c(bl[k]), 200 + k, shift=k not in (9, 16))
        if k == 15:
            out = out * 0.75 + 0.25 * 0.85   # светло-серая дымка
        return out
    if k == 18:  # белая вспышка с маджентовым текстом и маджентовой полосой снизу
        out = g_white(a, 118)
        out[int(H * 0.86):] = out[int(H * 0.86):] * 0.4 + np.array(c((225, 70, 200))) * 0.6
        return out
    return g_blue(a, c((7, 5, 245)), 299)


tmp = Path(OUT).with_suffix('.hq.mp4')
ff = subprocess.Popen(['ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                       '-c:v', 'libx264', '-preset', 'medium', '-crf', '12', '-pix_fmt', 'yuv420p', str(tmp)], stdin=subprocess.PIPE)

nframes = int(round(TL['duration'] * FPS))
for i in range(nframes):
    t = i / FPS
    t20 = np.floor(t * 20 + 1e-6) / 20             # 20 fps, как у оригинала
    src = int(round(t20 * FPS))
    if t >= T['black'] + 1 / 30 - 1e-6:
        a = np.zeros((H, W, 3), np.float32)
    else:
        a = load(src)
        if t20 < T['fadeIn'][1]:
            a = a * (t20 / T['fadeIn'][1])
        kz = float(np.clip((t20 - 18.5) / (T['glitch'] - 18.5), 0, 1))
        if t20 >= 18.5 and t20 < T['glitch']:
            a, _ = zoom_grade(a, kz)
            a = grain(a, 0.012 * kz)
        elif t20 >= T['glitch']:
            base, _ = zoom_grade(a, 1.0)
            k = int(round((t20 - T['glitch']) / 0.05))
            a = glitch(a, k, base)
            a = grain(a, 0.016)
        if t >= T['black'] - 1e-6:                   # один тёмный кадр перед чёрным
            a = a * 0.08
    ff.stdin.write((np.clip(a, 0, 1) * 255).astype(np.uint8).tobytes())
    if i % 90 == 0:
        print(f'пост {i}/{nframes}', flush=True)
ff.stdin.close()
ff.wait()

# «перезалив»: 576p с ореолами от шарпа и сжатием, затем апскейл до 1080
low = Path(OUT).with_suffix('.low.mp4')
subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', str(tmp),
                '-vf', 'scale=576:576:flags=bilinear,unsharp=5:5:0.8:5:5:0.0,noise=alls=2:allf=t',
                '-c:v', 'libx264', '-preset', 'medium', '-crf', '27', '-pix_fmt', 'yuv420p', str(low)], check=True)
subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', str(low),
                '-vf', 'scale=1080:1080:flags=bicubic',
                '-c:v', 'libx264', '-preset', 'medium', '-crf', '17', '-pix_fmt', 'yuv420p',
                '-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'bt709', OUT], check=True)
tmp.unlink()
low.unlink()
print('готово:', OUT)
