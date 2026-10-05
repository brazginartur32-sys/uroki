"""Постобработка кадров (по покадровым замерам референса):
- картинка 20 fps внутри 30 fps (каденс round(t*20)), как у оригинала;
- наплыв из чёрного;
- финальный наезд: «зум пережатого видео» (кадр уменьшается до 576/z и растягивается обратно), растущее мыло,
  фон холодеет в стальной синий, исходящие пузыри желтеют, текст жирнеет, лёгкое свечение, зерно;
- глитч по кадрам референса: белые/голубые засветы с маджентовым рваным текстом, стально-синий → ультрамарин,
  вспышка с маджентовой полосой, тёмный кадр и обрыв в чёрное;
- «перезалив»: 576p, мыло + ореолы от шарпа и цветная кайма, сжатие, апскейл до 1080.

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

# масштаб наезда — та же кривая, что в index.html
ZK = [(18.15, 1.0), (18.2, 1.01), (18.5, 1.10), (18.73, 1.20), (19.0, 1.35), (19.2, 1.48), (19.4, 1.64), (19.6, 1.85), (19.8, 2.0), (20.0, 2.2), (20.3, 2.42)]


def zoom_at(t):
    if t <= ZK[0][0]:
        return 1.0
    for (a, za), (b, zb) in zip(ZK, ZK[1:]):
        if t <= b:
            return za + (zb - za) * (t - a) / (b - a)
    return ZK[-1][1]


def load(i):
    i = max(0, min(len(files) - 1, i))
    return np.asarray(Image.open(files[i]).convert('RGB'), dtype=np.float32) / 255


def to_im(a):
    return Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))


def from_im(im):
    return np.asarray(im, dtype=np.float32) / 255


def blur(a, r):
    if r <= 0.05:
        return a
    return from_im(to_im(a).filter(ImageFilter.GaussianBlur(float(r))))


def unsharp(a, r=2.0, amt=0.6):
    return a + (a - blur(a, r)) * amt


def hblur(a, r):
    r = int(r)
    if r < 1:
        return a
    c = np.cumsum(np.pad(a, ((0, 0), (r + 1, r), (0, 0)), mode='edge'), axis=1)
    return (c[:, 2 * r + 1:] - c[:, :-2 * r - 1]) / (2 * r + 1)


def vblur(a, r):
    r = int(r)
    if r < 1:
        return a
    c = np.cumsum(np.pad(a, ((r + 1, r), (0, 0), (0, 0)), mode='edge'), axis=0)
    return (c[2 * r + 1:] - c[:-2 * r - 1]) / (2 * r + 1)


def resample(a, z):
    """«зум пережатого видео»: уменьшить до 576/z и растянуть обратно"""
    n = max(200, int(576 / z))
    im = to_im(a).resize((n, n), Image.BOX).resize((W, H), Image.BILINEAR)
    return from_im(im)


def lum(a):
    return a[..., 0] * .299 + a[..., 1] * .587 + a[..., 2] * .114


def smooth(e0, e1, x):
    k = np.clip((x - e0) / (e1 - e0), 0, 1)
    return k * k * (3 - 2 * k)


def masks(a):
    L = lum(a)
    white = smooth(0.90, 0.955, L)[..., None]            # пузыри (белые и жёлтые)
    dark = (1 - smooth(0.65, 0.85, L))[..., None]        # текст и время
    sat = (a.max(2) - a.min(2))[..., None]
    yellow = white * smooth(0.05, 0.10, sat) * (a[..., 2:3] < a[..., 0:1])   # жёлтые исходящие
    return white, dark, yellow


def bolden(d, r=1.5, gain=1.8):
    return np.clip(blur(d.repeat(3, 2), r)[..., :1] * gain, 0, 1)


def screen(a, b, k):
    return 1 - (1 - a) * (1 - b * k)


def grain(a, s):
    return a + rng.normal(0, s, a.shape[:2])[..., None].astype(np.float32)


def zoom_grade(a, k, z):
    """k: 0→1 за 18.15–19.15. Фон → (145,174,208), жёлтые → (242,243,169), белые → ~249, текст жирнее, мыло."""
    a = resample(a, z)
    w, d, y = masks(a)
    bg = 1 - w
    out = a * (1 - bg * np.array([0.31, 0.21, 0.10]) * k)
    out = out * (1 - y * np.array([0.05, 0.02, 0.28]) * k)       # исходящие желтеют
    out = out * (1 - (w - y) * 0.025 * k)                          # белые чуть гаснут
    db = bolden(d)
    ink = np.array([0.18, 0.2, 0.25])
    out = out * (1 - db * 0.5 * k) + ink * db * 0.5 * k
    out = blur(out, 1.3 * k)
    out = screen(out, blur(out, 9), 0.17 * k)                      # лёгкое свечение
    return out


def recolor(a, bg, bubble, ink, bold=True):
    w, d, _ = masks(a)
    if bold:
        d = bolden(d)
    out = np.array(bg) * (1 - w) + np.array(bubble) * w
    return out * (1 - d) + np.array(ink) * d, w, d


def bubble_bottoms(w):
    """полоса сразу под пузырями (для маджентовой/тёмной полосы)"""
    ww = (w[..., 0] > 0.5).astype(np.float32)
    below = np.roll(ww, 1, axis=0)
    return np.clip(below - ww, 0, 1)


def stripe_under(out, w, color, h, alpha=1.0):
    e = bubble_bottoms(w)
    band = np.zeros_like(e)
    for dy in range(h):
        band = np.maximum(band, np.roll(e, dy, axis=0))
    band = band[..., None] * (1 - w)
    return out * (1 - band * alpha) + np.array(color) * band * alpha


MAG = (0.86, 0.55, 0.84)


def torn(d, seed, rmin=8, rmax=14):
    r = np.random.default_rng(seed)
    dd = d.copy()
    for _ in range(12):
        y = r.integers(0, H - 20)
        h = r.integers(4, 22)
        dd[y:y + h] *= r.uniform(0, 0.35)
    return hblur(dd.repeat(3, 2), r.integers(rmin, rmax))[..., :1]


def g_white(a, seed):
    """засвет почти до белого: пузыри сливаются с фоном, маджентовый залитый рваный текст"""
    w, d, _ = masks(a)
    d = bolden(d)
    out = np.full_like(a, 0.95)
    out = out * (1 - w) + 0.975 * w
    dd = torn(d, seed)
    out = out * (1 - dd * 0.8) + np.array(MAG) * dd * 0.8
    return blur(out, 2.8)


def g_light(a, bg, seed, stripe=True):
    """светлый кадр: фон цветной, пузыри размыты до ~0.94, бледный маджентовый текст, тонкая маджента под пузырём"""
    w, d, _ = masks(a)
    d = bolden(d)
    out = np.array(bg) * (1 - w) + 0.94 * w
    dd = torn(d, seed)
    out = out * (1 - dd * 0.42) + np.array(MAG) * dd * 0.42
    if stripe:
        out = stripe_under(out, w, (0.9, 0.2, 0.85), 14, 0.8)
    out = screen(out, blur(w.repeat(3, 2), 8), 0.25)
    return blur(out, 2.2)


def g_blue(a, bg, seed, smear_lines=True):
    """синий кадр: белые пузыри ~0.94, чёрный жирный мыльный текст (и время), свечение, тёмная полоса под пузырями"""
    out, w, d = recolor(a, bg, (0.94, 0.945, 0.95), (0.05, 0.05, 0.09))
    out = stripe_under(out, w, (0.08, 0.08, 0.24), 10, 0.85)
    out = screen(out, blur(w.repeat(3, 2), 10), 0.22)
    if smear_lines:
        r = np.random.default_rng(seed)
        ys = np.where(d[..., 0].mean(1) > 0.04)[0]
        for _ in range(2):
            if len(ys):
                y = int(r.choice(ys))
                h = int(r.integers(18, 44))
                y0 = max(0, y - h // 2)
                out[y0:y0 + h] = hblur(out[y0:y0 + h], int(r.integers(14, 28)))
    out = blur(out, 2.4)
    out = out + (out - blur(out, 3)) * 0.5
    return grain(out, 0.02)


def c(rgb):
    return tuple(x / 255 for x in rgb)


def glitch(a, k, base):
    """k — номер 20-кадрового шага от 19.20 (по замерам референса)"""
    if k == 0:
        return vblur(base, 6)
    if k == 1:
        return g_white(a, 101)
    if k == 2:   # обычный кадр наезда сквозь засвет, смаз по горизонтали
        return hblur(base * 0.55 + 0.45, 18)
    if k == 3:
        return g_white(a, 103)
    if k == 4:   # стально-синий читаемый кадр
        out, w, d = recolor(a, c((120, 152, 200)), (0.95, 0.95, 0.96), (0.08, 0.08, 0.12))
        return blur(screen(out, blur(w.repeat(3, 2), 9), 0.2), 2.2)
    light = {5: (204, 234, 236), 6: (182, 237, 235), 7: (182, 237, 235), 8: (143, 235, 236), 10: (103, 231, 237),
             13: (30, 165, 235), 15: (24, 56, 232)}
    if k in light:
        return g_light(a, c(light[k]), 100 + k)
    blue = {9: (76, 141, 200), 11: (43, 120, 191), 12: (27, 96, 193), 14: (22, 54, 187),
            16: (25, 18, 203), 17: (25, 18, 203), 19: (28, 20, 225), 20: (28, 20, 225), 21: (7, 5, 245)}
    if k in blue:
        return g_blue(a, c(blue[k]), 200 + k, smear_lines=k in (9, 11, 14, 17, 20))
    if k == 18:  # вспышка: верх ультрамарин, пузыри белые со смазанным бледным текстом, маджента под пузырями
        w, d, _ = masks(a)
        out = np.array(c((28, 22, 226))) * (1 - w) + 0.97 * w
        dd = torn(bolden(d), 118)
        out = out * (1 - dd * 0.5) + np.array(MAG) * dd * 0.5
        out = stripe_under(out, w, c((230, 26, 232)), 75, 1.0)
        return blur(out, 2.0)
    return g_blue(a, c((7, 5, 245)), 299)


tmp = Path(OUT).with_suffix('.hq.mp4')
ff = subprocess.Popen(['ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                       '-c:v', 'libx264', '-preset', 'medium', '-crf', '12', '-pix_fmt', 'yuv420p', str(tmp)], stdin=subprocess.PIPE)

nframes = int(round(TL['duration'] * FPS))
cache = {}
for i in range(nframes):
    t = i / FPS
    t20 = np.floor(t * 20 + 0.5 + 1e-6) / 20      # каденс 20 fps: round(t*20)
    src = int(round(t20 * FPS))
    if t >= T['black'] + 1 / 30 - 1e-6:
        a = np.zeros((H, W, 3), np.float32)
    elif src in cache:
        a = cache[src]
    else:
        a = load(src)
        if t20 < T['fadeIn'][1]:
            a = a * (t20 / T['fadeIn'][1])
        if T['zoom'] <= t20 < T['glitch']:
            kz = float(np.clip((t20 - T['zoom']) / (19.15 - T['zoom']), 0, 1))
            a = grain(zoom_grade(a, kz, zoom_at(t20)), 0.012 * kz)
        elif t20 >= T['glitch']:
            z = zoom_at(t20)
            base = zoom_grade(a, 1.0, z)
            k = int(round((t20 - T['glitch']) / 0.05))
            a = grain(glitch(resample(a, z), k, base), 0.012)
        cache = {src: a}
    out = a
    if T['black'] - 1e-6 <= t < T['black'] + 1 / 30 - 1e-6:      # один тёмный кадр перед чёрным
        out = a * 0.06
    ff.stdin.write((np.clip(out, 0, 1) * 255).astype(np.uint8).tobytes())
    if i % 90 == 0:
        print(f'пост {i}/{nframes}', flush=True)
ff.stdin.close()
ff.wait()

# «перезалив»: 576p, мыло + ореолы и цветная кайма от шарпа, сжатие, затем апскейл до 1080
low = Path(OUT).with_suffix('.low.mp4')
subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', str(tmp),
                '-vf', 'scale=576:576:flags=bilinear,gblur=sigma=0.7,unsharp=5:5:0.8:5:5:1.5,chromashift=cbv=1:crv=-1,noise=alls=2:allf=t',
                '-c:v', 'libx264', '-preset', 'medium', '-crf', '27', '-pix_fmt', 'yuv420p', str(low)], check=True)
subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', str(low),
                '-vf', 'scale=1080:1080:flags=bicubic',
                '-c:v', 'libx264', '-preset', 'medium', '-crf', '17', '-pix_fmt', 'yuv420p',
                '-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'bt709', OUT], check=True)
tmp.unlink()
low.unlink()
print('готово:', OUT)
