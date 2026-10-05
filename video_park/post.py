"""Постобработка (безопасная для светочувствительных зрителей):
- монтаж как в «Чья будете»: 20 fps внутри 30, «перезалив» в 576p;
- «Спустя час»: картинка сжимается в полосу поверх мягких помех (без чёрных провалов),
  ровный подрагивающий текст (без мигания), обратное раскрытие;
- глитч-удары в стиле референса (цветовой сдвиг голубой / маджента / синий, разрывы строк,
  расслоение каналов, рваный текст), но БЕЗ скачков яркости: средняя яркость кадра сохраняется;
- резкая темнота в конце (один переход).
Затемнение «с каждым сообщением» делается в index.html: темнеет фон и края, старые сообщения уходят в тень,
новые остаются читаемыми.

python3 post.py frames timeline.json out_noaudio.mp4
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

FRAMES = Path(sys.argv[1])
TL = json.load(open(sys.argv[2]))
T = TL['t']
OUT = sys.argv[3]
FPS = 30
W = H = 1080
files = sorted(FRAMES.glob('*.jpg'))
YY, XX = np.mgrid[0:H, 0:W].astype(np.float32)
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf'

HITS = [(m['t'], m['hit']) for m in TL['msgs'] if m.get('hit')] + [(h['t'], h['s']) for h in TL.get('extraHits', [])]
HITS.sort()
PALETTE = [(103, 231, 237), (219, 140, 213), (76, 141, 200), (28, 20, 225)]   # голубой, маджента, стальной, ультрамарин


def load(i):
    i = max(0, min(len(files) - 1, i))
    return np.asarray(Image.open(files[i]).convert('RGB'), dtype=np.float32) / 255


def blur(a, r):
    if r <= 0.05:
        return a
    im = Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))
    return np.asarray(im.filter(ImageFilter.GaussianBlur(float(r))), dtype=np.float32) / 255


def hblur(a, r):
    r = int(r)
    if r < 1:
        return a
    c = np.cumsum(np.pad(a, ((0, 0), (r + 1, r), (0, 0)), mode='edge'), axis=1)
    return (c[:, 2 * r + 1:] - c[:, :-2 * r - 1]) / (2 * r + 1)


def to_lin(a):
    a = np.clip(a, 0, 1)
    return np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4)


def to_srgb(l):
    l = np.clip(l, 0, 1)
    return np.where(l <= 0.0031308, l * 12.92, 1.055 * l ** (1 / 2.4) - 0.055)


def Ylin(l):
    return l[..., 0] * 0.2126 + l[..., 1] * 0.7152 + l[..., 2] * 0.0722


def rgb_split(a, dx):
    dx = int(round(dx))
    if dx == 0:
        return a
    out = a.copy()
    out[..., 0] = np.roll(a[..., 0], -dx, axis=1)
    out[..., 2] = np.roll(a[..., 2], dx, axis=1)
    return out


# ---------------- «Спустя час»: мягкие помехи ----------------
def tv_static(t, i):
    rng = np.random.default_rng(i * 7 + 1)
    n = rng.random((H // 4, W // 4)).astype(np.float32)
    n = np.repeat(np.repeat(n, 4, 0), 4, 1)
    n = blur(np.repeat(n[..., None], 3, 2), 1.2)[..., 0]
    g = 0.40 + (n - 0.5) * 0.40                                        # невысокий контраст
    band = np.exp(-(((YY - (t * 420) % (H + 300) + 150) / 140.0) ** 2)) * 0.06
    scan = 1 + 0.03 * np.sin(YY * np.pi / 4)
    g = (g + band) * scan
    return np.repeat(g[..., None], 3, 2) * np.array([0.95, 0.97, 1.0])


def tv_text(a, step):
    rng = np.random.default_rng(step * 13 + 5)
    img = Image.new('L', (W, H), 0)
    d = ImageDraw.Draw(img)
    f = ImageFont.truetype(FONT, 118)
    txt = 'СПУСТЯ ЧАС...'
    w = d.textlength(txt, font=f)
    jx, jy = rng.integers(-4, 5), rng.integers(-3, 4)
    d.text(((W - w) / 2 + jx, H / 2 - 66 + jy), txt, font=f, fill=255)
    m = np.asarray(img, dtype=np.float32)[..., None] / 255
    col = np.ones((H, W, 3), np.float32) * np.array([0.96, 0.96, 0.98])
    lay = rgb_split(m.repeat(3, 2), 4) * col
    if step % 9 == 4:                                                  # редкий разрыв строки
        y = int(H / 2 - 40 + rng.integers(0, 60))
        lay[y:y + 26] = np.roll(lay[y:y + 26], int(rng.choice([-18, 18])), axis=1)
    mask = np.clip(lay.max(2, keepdims=True), 0, 1)
    return a * (1 - mask * 0.9) + lay * 0.9


def squeeze(frame, bg, k):
    """кадр сжимается к центральной полосе (k=0 — целый, k=1 — исчез) поверх помех, без чёрного"""
    h = int(H * (1 - min(1.0, max(0.0, k))) ** 1.6)
    if h < 3:
        return bg
    small = np.asarray(Image.fromarray((np.clip(frame, 0, 1) * 255).astype(np.uint8)).resize((W, h)), dtype=np.float32) / 255
    out = bg.copy()
    y0 = (H - h) // 2
    out[y0:y0 + h] = small
    return out


# ---------------- глитч-удар без скачка яркости ----------------
def colorize_keep_luma(a, color, amount):
    """перекрашивает в цвет, сохраняя яркость каждого пикселя (светлые пузыри остаются светлыми)"""
    L = to_lin(a)
    Y = Ylin(L)[..., None]
    c = to_lin(np.array(color, np.float32) / 255)
    yc = float(c[0] * 0.2126 + c[1] * 0.7152 + c[2] * 0.0722)
    tint = c[None, None, :] * (Y / yc)                                 # та же яркость, другой цвет
    ratio = c / yc
    lim = np.where(ratio > 1, (1 - Y) / (np.maximum(Y, 1e-4) * (ratio - 1 + 1e-6)), 1.0)
    m = np.clip(np.minimum(lim.min(2, keepdims=True), amount), 0, amount)
    out = Y * (1 - m) + tint * m
    return to_srgb(out)


def slices(a, rng, strength):
    out = a.copy()
    for _ in range(3 + int(4 * strength)):
        y = int(rng.integers(0, H - 70))
        h = int(rng.integers(8, 64))
        dx = int(rng.integers(15, 15 + int(60 * strength) + 1)) * int(rng.choice([-1, 1]))
        out[y:y + h] = np.roll(out[y:y + h], dx, axis=1)
    return out


def torn_text(a, rng):
    L = a[..., 0] * .299 + a[..., 1] * .587 + a[..., 2] * .114
    d = (L < 0.3).astype(np.float32)
    for _ in range(10):
        y = int(rng.integers(0, H - 20))
        d[y:y + int(rng.integers(4, 18))] *= 0.2
    d = hblur(d[..., None].repeat(3, 2), 10)[..., :1]
    mag = np.array([0.86, 0.55, 0.84], np.float32)
    return a * (1 - d * 0.6) + mag * d * 0.6


def keep_mean(src, out):
    """выравнивает среднюю яркость (в линейном свете) по 9 зонам, чтобы не было вспышки"""
    Ls, Lo = to_lin(src), to_lin(out)
    res = Lo.copy()
    for y0 in range(0, H, H // 3):
        for x0 in range(0, W, W // 3):
            sl = (slice(y0, y0 + H // 3), slice(x0, x0 + W // 3))
            ys, yo = Ylin(Ls[sl]).mean(), Ylin(Lo[sl]).mean()
            if yo > 1e-4:
                res[sl] = Lo[sl] * min(1.6, ys / yo)
    return to_srgb(res)


def hit_at(t):
    """(сила, номер удара, шаг) — удар длится 3 шага по 1/20 с, начиная с появления сообщения"""
    for j, (h, s) in enumerate(HITS):
        n = int(round((t - h) * 20)) - 1
        if 0 <= n < 3:
            return s * (1.0, 0.6, 0.3)[n], j, n
    return 0.0, -1, -1


def glitch_hit(a, t):
    s, j, n = hit_at(t)
    if s <= 0:
        return a
    rng = np.random.default_rng(j * 31 + n)
    out = colorize_keep_luma(a, PALETTE[j % len(PALETTE)], 0.85 * min(1, s + 0.2))
    out = slices(out, rng, s)
    if n == 0 and s >= 0.7:
        out = torn_text(out, rng)
    out = rgb_split(out, 3 + 7 * s)
    return keep_mean(a, out)


# ---------------- главный цикл ----------------
tmp = Path(OUT).with_suffix('.hq.mp4')
ff = subprocess.Popen(['ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                       '-c:v', 'libx264', '-preset', 'medium', '-crf', '12', '-pix_fmt', 'yuv420p', str(tmp)], stdin=subprocess.PIPE)
nframes = int(round(TL['duration'] * FPS))
grain_rng = np.random.default_rng(5)
SQ = 0.3          # длительность сжатия/раскрытия картинки
cache = {}
for i in range(nframes):
    t = i / FPS
    t20 = np.floor(t * 20 + 0.5 + 1e-6) / 20
    src = int(round(t20 * FPS))
    step = int(round(t20 * 20))
    if t20 >= T['black']:
        a = np.zeros((H, W, 3), np.float32)
    elif src in cache:
        a = cache[src]
    else:
        a = load(src)
        if t20 < T['fadeIn'][1]:
            a = a * (t20 / T['fadeIn'][1])
        if T['tvOff'] <= t20 < T['tvOn']:
            st = tv_static(t20, step)
            if t20 < T['tvOff'] + SQ:                                   # картинку «засасывает» в помехи
                k = (t20 - T['tvOff']) / SQ
                a = squeeze(a, st * min(1, 0.4 + k), k)
            elif t20 >= T['tvOn'] - SQ:                                 # картинка раскрывается обратно
                k = (T['tvOn'] - t20) / SQ
                a = squeeze(load(int(round(T['tvOn'] * FPS))), st, k)
            else:
                a = tv_text(st, step)
        else:
            a = glitch_hit(a, t20)
            if t20 > T['tvOn']:
                a = a + grain_rng.normal(0, 0.008, (H, W, 1)).astype(np.float32)
        cache = {src: a}
    ff.stdin.write((np.clip(a, 0, 1) * 255).astype(np.uint8).tobytes())
    if i % 90 == 0:
        print(f'пост {i}/{nframes}', flush=True)
ff.stdin.close()
ff.wait()

# «перезалив», как в «Чья будете»
low = Path(OUT).with_suffix('.low.mp4')
subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', str(tmp),
                '-vf', 'scale=576:576:flags=bilinear,gblur=sigma=0.6,unsharp=5:5:0.7:5:5:1.2,chromashift=cbv=1:crv=-1,noise=alls=2:allf=t',
                '-c:v', 'libx264', '-preset', 'medium', '-crf', '26', '-pix_fmt', 'yuv420p', str(low)], check=True)
subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', str(low), '-vf', 'scale=1080:1080:flags=bicubic',
                '-c:v', 'libx264', '-preset', 'medium', '-crf', '17', '-pix_fmt', 'yuv420p',
                '-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'bt709', OUT], check=True)
tmp.unlink()
low.unlink()
print('готово:', OUT)
