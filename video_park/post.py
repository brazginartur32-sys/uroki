"""Постобработка: монтаж как в «Чья будете» (20 fps внутри 30, «перезалив»), плюс:
- «Спустя час»: телевизор выключается в полоску, снег-помехи с дёрганым текстом, включается обратно;
- после этого экран гаснет с каждым сообщением (яркость, цвет, виньетка), появляются блики;
- после «Я не понимаю...» экран почти тёмный, блики очень сильные;
- в конце — резкая темнота.

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

msg_times = [m['t'] for m in TL['msgs'] if m['t'] > 0]
after_tv = [t for t in msg_times if t > T['tvOn']]


def load(i):
    i = max(0, min(len(files) - 1, i))
    return np.asarray(Image.open(files[i]).convert('RGB'), dtype=np.float32) / 255


def blur(a, r):
    if r <= 0.05:
        return a
    im = Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))
    return np.asarray(im.filter(ImageFilter.GaussianBlur(float(r))), dtype=np.float32) / 255


def rgb_split(a, dx, dy=0):
    dx, dy = int(dx), int(dy)
    out = a.copy()
    out[..., 0] = np.roll(np.roll(a[..., 0], -dx, axis=1), -dy, axis=0)
    out[..., 2] = np.roll(np.roll(a[..., 2], dx, axis=1), dy, axis=0)
    return out


def lum(a):
    return a[..., 0] * .299 + a[..., 1] * .587 + a[..., 2] * .114


# ---------------- «Спустя час»: телевизионные помехи ----------------
def tv_text_layer(t, rng):
    img = Image.new('RGB', (W, H), (0, 0, 0))
    d = ImageDraw.Draw(img)
    f = ImageFont.truetype(FONT, 124)
    txt = 'СПУСТЯ ЧАС...'
    w = d.textlength(txt, font=f)
    jx = rng.integers(-14, 15) + (rng.random() < 0.15) * rng.integers(-60, 61)
    jy = rng.integers(-8, 9)
    d.text(((W - w) / 2 + jx, H / 2 - 60 + jy), txt, font=f, fill=(255, 255, 255))
    a = np.asarray(img, dtype=np.float32) / 255
    # рваные строки и сдвиг каналов
    for _ in range(rng.integers(2, 6)):
        y = rng.integers(int(H / 2 - 80), int(H / 2 + 60))
        h = rng.integers(4, 18)
        a[y:y + h] = np.roll(a[y:y + h], int(rng.integers(-50, 51)), axis=1)
    return rgb_split(a, rng.integers(3, 10))


def tv_static(t, i):
    rng = np.random.default_rng(i * 7 + 1)
    n = rng.random((H // 3, W // 3)).astype(np.float32)
    n = np.repeat(np.repeat(n, 3, 0), 3, 1)
    # горизонтальная катящаяся полоса и строки
    roll = ((YY - (t * 700) % (H + 300) + 150) / 120.0)
    band = np.exp(-roll ** 2) * 0.35
    scan = 0.85 + 0.15 * np.sin(YY * np.pi / 3)
    g = (n * 0.75 + band) * scan
    a = np.repeat(g[..., None], 3, 2) * np.array([0.92, 0.95, 1.0])
    # дёрганый текст, мерцает
    if rng.random() > 0.12:
        txt = tv_text_layer(t, rng)
        a = a * (1 - lum(txt)[..., None] * 0.85) + txt * 0.95
    # срыв синхронизации: кадр иногда «прыгает» по вертикали
    if rng.random() < 0.2:
        a = np.roll(a, int(rng.integers(-120, 120)), axis=0)
    return np.clip(a, 0, 1)


def crt_off(a, k):
    """кадр сжимается в яркую полосу, потом в точку (k: 0→1)"""
    out = np.zeros_like(a)
    if k < 0.7:
        h = max(4, int(H * (1 - k / 0.7) ** 2))
        small = np.asarray(Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8)).resize((W, h)), dtype=np.float32) / 255
        small = small * (1 + 2.5 * k) + 0.4 * k
        y0 = (H - h) // 2
        out[y0:y0 + h] = np.clip(small, 0, 1)
    else:
        r = (1 - (k - 0.7) / 0.3) * 60 + 2
        out += np.exp(-((XX - W / 2) ** 2 / (r * 8) ** 2 + (YY - H / 2) ** 2 / r ** 2))[..., None]
    return out


# ---------------- затухание и глитч-блики (как в референсе «Чья будете») ----------------
import glitchlib as G

LEVELS = [0.95, 0.91, 0.86, 0.80, 0.75, 0.70, 0.65, 0.60, 0.56, 0.52, 0.52]   # после каждого сообщения


def darkness(t):
    """яркость экрана: ступеньками на каждом сообщении, с коротким провалом в момент сообщения"""
    if t < T['tvOn']:
        return 1.0
    lv = LEVELS[0]
    for j, e in enumerate(after_tv):
        if t >= e:
            prev, nxt = LEVELS[j], LEVELS[min(j + 1, len(LEVELS) - 1)]
            p = (t - e) / 0.3
            if p < 1:
                dip = 0.5 * np.sin(np.pi * min(1, p * 2))
                lv = (prev + (nxt - prev) * min(1, p * 1.5)) * (1 - 0.3 * dip)
            else:
                lv = nxt
    return lv


def grade(a, t):
    lv = darkness(t)
    gone = float(np.clip((t - T['tvOn']) / (T['black'] - T['tvOn']), 0, 1))
    L = lum(a)[..., None]
    a = L + (a - L) * (1 - 0.6 * gone)
    a = a * np.array([1 - 0.12 * gone, 1 - 0.05 * gone, 1 + 0.04 * gone])
    a = a * lv
    rr = ((XX - W / 2) ** 2 + (YY - H * 0.55) ** 2) / (W * W)
    vig = np.exp(-rr * (0.5 + 2.5 * gone))
    return a * (0.35 + 0.65 * vig)[..., None]


# Кадры глитча референса (шаг 1/20 с). Порядок взят из конца референса.
SEQ = [1, 4, 5, 9, 3, 8, 11, 6, 14, 18, 12, 10, 16, 13, 19, 21]
BLUE_READ = [(22, 54, 187), (25, 18, 203), (43, 120, 191), (28, 20, 225), (25, 18, 203), (7, 5, 245)]

hits = after_tv[2:]                 # с «Зачем?...»
FINAL_ME = after_tv[-2]             # «Я не понимаю...»
FINAL = after_tv[-1]                # «Просто потому, что это весело»


def glitch_kind(t):
    """что показать в этом 20-кадровом шаге: None — обычный кадр; ('g', k) — кадр глитча; ('blue', i) — читаемый синий"""
    step = int(round(t * 20))
    if t >= FINAL:
        # финал как в конце референса: синий кадр с белыми пузырями и чёрным жирным текстом,
        # изредка — вспышка
        d = t - FINAL
        if d < 0.1:
            return ('g', 1)
        if step % 7 == 3:
            return ('g', [5, 18, 13][step % 3])
        return ('blue', step % len(BLUE_READ))
    if t >= FINAL_ME:
        d = t - FINAL_ME
        if d < 0.15:
            return ('g', SEQ[step % len(SEQ)])
        # между ними — то сильный глитч, то обычный тёмный кадр
        return ('g', SEQ[step % len(SEQ)]) if step % 5 in (0, 2, 3) else None
    for j, e in enumerate(hits[:-2]):
        n = 1 + j                          # с каждым сообщением глитч длиннее
        if e <= t < e + n * 0.05 + 1e-6:
            return ('g', SEQ[(int(round((t - e) * 20)) + j * 3) % len(SEQ)])
    return None


def apply_glitch(raw, t, base):
    kind = glitch_kind(t)
    if kind is None:
        return base
    if kind[0] == 'blue':
        out = G.g_blue(raw, G.c(BLUE_READ[kind[1]]), 300 + int(t * 20), smear_lines=False)
        return out
    k = kind[1]
    out = G.glitch(raw, k, G.zoom_grade(raw, 1.0, 1.0))
    # до финала глитч тоже немного притушен вместе с экраном
    if t < FINAL:
        out = out * (0.55 + 0.45 * darkness(t))
    return out


# ---------------- главный цикл ----------------
tmp = Path(OUT).with_suffix('.hq.mp4')
ff = subprocess.Popen(['ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                       '-c:v', 'libx264', '-preset', 'medium', '-crf', '12', '-pix_fmt', 'yuv420p', str(tmp)], stdin=subprocess.PIPE)
nframes = int(round(TL['duration'] * FPS))
grain_rng = np.random.default_rng(5)
cache = {}
for i in range(nframes):
    t = i / FPS
    t20 = np.floor(t * 20 + 0.5 + 1e-6) / 20
    src = int(round(t20 * FPS))
    if t20 >= T['black']:
        a = np.zeros((H, W, 3), np.float32)
    elif src in cache:
        a = cache[src]
    else:
        a = load(src)
        if t20 < T['fadeIn'][1]:
            a = a * (t20 / T['fadeIn'][1])
        if T['tvOff'] <= t20 < T['tvOff'] + 0.2:
            a = crt_off(a, (t20 - T['tvOff']) / 0.2)
        elif T['tvOff'] + 0.2 <= t20 < T['tvOn'] - 0.25:
            a = tv_static(t20, src)
        elif T['tvOn'] - 0.25 <= t20 < T['tvOn']:
            # включение: кадр раскрывается из полосы сквозь снег
            k = (t20 - (T['tvOn'] - 0.25)) / 0.25
            a = crt_off(grade(a, T['tvOn']), 1 - k) * 0.7 + tv_static(t20, src) * 0.3 * (1 - k)
        else:
            raw = a
            a = apply_glitch(raw, t20, grade(raw, t20))
            a = a + grain_rng.normal(0, 0.012, (H, W, 1)).astype(np.float32)
        cache = {src: a}
    ff.stdin.write((np.clip(a, 0, 1) * 255).astype(np.uint8).tobytes())
    if i % 90 == 0:
        print(f'пост {i}/{nframes}', flush=True)
ff.stdin.close()
ff.wait()

# «перезалив», как в «Чья будете»
low = Path(OUT).with_suffix('.low.mp4')
subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', str(tmp),
                '-vf', 'scale=576:576:flags=bilinear,gblur=sigma=0.7,unsharp=5:5:0.8:5:5:1.5,chromashift=cbv=1:crv=-1,noise=alls=2:allf=t',
                '-c:v', 'libx264', '-preset', 'medium', '-crf', '27', '-pix_fmt', 'yuv420p', str(low)], check=True)
subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', str(low), '-vf', 'scale=1080:1080:flags=bicubic',
                '-c:v', 'libx264', '-preset', 'medium', '-crf', '17', '-pix_fmt', 'yuv420p',
                '-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'bt709', OUT], check=True)
tmp.unlink()
low.unlink()
print('готово:', OUT)
