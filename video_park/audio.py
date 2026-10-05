"""Звук с нуля:
0 – «Спустя час»: лёгкая беззаботная мелодия;
«Спустя час»: выключение телевизора, шипение помех, гул 50 Гц, включение;
дальше: низкий гул, сердцебиение (ускоряется), удар на каждом сообщении, «всасывающие» свисты перед бликами;
финал: нарастающий звон, резкая тишина, тихий звон в ушах.

python3 audio.py timeline.json audio.wav
"""
import json
import sys
import wave

import numpy as np

SR = 44100
rng = np.random.default_rng(7)
tl = json.load(open(sys.argv[1]))
T = tl['t']
DUR = tl['duration']
N = int(DUR * SR)
t_all = np.arange(N) / SR
msgs = [m for m in tl['msgs'] if m['t'] > 0]


def midi(n):
    return 440.0 * 2 ** ((n - 69) / 12)


def place(buf, sig, t, gain=1.0):
    i = int(t * SR)
    if i >= len(buf):
        return
    if i < 0:
        sig, i = sig[-i:], 0
    j = min(len(buf), i + len(sig))
    buf[i:j] += sig[: j - i] * gain


def fft_filter(x, lo=None, hi=None, order=2):
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / SR)
    Hh = np.ones_like(f)
    if hi:
        Hh *= 1 / np.sqrt(1 + (f / hi) ** (2 * order))
    if lo:
        Hh *= 1 / np.sqrt(1 + (lo / np.maximum(f, 1e-3)) ** (2 * order))
    return np.fft.irfft(X * Hh, len(x))


def reverb(x, length=3.0, decay=0.9, mix=0.4, seed=0):
    r = np.random.default_rng(seed)
    L = int(length * SR)
    tt = np.arange(L) / SR
    ir = fft_filter(r.normal(0, 1, L) * np.exp(-tt / decay), hi=5000)
    ir[: int(0.015 * SR)] = 0
    ir /= np.sqrt((ir ** 2).sum())
    n = 1 << int(np.ceil(np.log2(len(x) + L)))
    wet = np.fft.irfft(np.fft.rfft(x, n) * np.fft.rfft(ir, n), n)[: len(x)]
    return x * (1 - mix) + wet * mix * 2.0


def pluck(note, dur, vel=0.5):
    f0 = midi(note)
    L = int((dur + 1.2) * SR)
    tt = np.arange(L) / SR
    s = sum(np.sin(2 * np.pi * f0 * k * tt) / k ** 1.4 * np.exp(-tt * (2 + k)) for k in range(1, 7))
    return s * np.minimum(1, tt / 0.004) * vel * 0.3


# ---------- 1. беззаботное начало (до помех) ----------
light = np.zeros(N)
BEAT = 60 / 96
prog = [[60, 64, 67, 72], [55, 59, 62, 67], [57, 60, 64, 69], [53, 57, 60, 65]]   # C G Am F
mel = [76, 79, 81, 79, 76, 74, 72, 74]
for b in range(4):
    t0 = b * 4 * BEAT - 0.3
    ch = prog[b % 4]
    for i in range(8):
        place(light, pluck(ch[i % 4] if i % 2 == 0 else ch[(i + 1) % 4], BEAT * .6, .42), t0 + i * BEAT / 2)
    place(light, pluck(ch[0] - 24, BEAT * 3.5, .6), t0)
    for i, n in enumerate(mel[b % 2 * 4: b % 2 * 4 + 4]):
        place(light, pluck(n, BEAT * .9, .5), t0 + i * BEAT)
light = reverb(fft_filter(light, hi=6000), length=2.0, decay=0.5, mix=0.25, seed=1)
light *= np.clip((T['tvOff'] - t_all) / 0.03, 0, 1) * np.clip(t_all / 0.3, 0, 1)

# ---------- 2. «Спустя час»: телевизор ----------
tv = np.zeros(N)
a0, a1 = int(T['tvOff'] * SR), int(T['tvOn'] * SR)
# выключение: «пиу» вниз
L = int(0.22 * SR)
tt = np.arange(L) / SR
tv[a0:a0 + L] += np.sin(2 * np.pi * np.cumsum(900 * np.exp(-tt * 12) + 40) / SR) * np.exp(-tt * 9) * 0.5
s0 = a0 + int(0.2 * SR)
noise = fft_filter(rng.normal(0, 1, a1 - s0), lo=400, hi=9000)
flutter = 0.75 + 0.25 * np.sin(2 * np.pi * 7 * np.arange(a1 - s0) / SR) * rng.random()
hum = 0.25 * np.sin(2 * np.pi * 50 * np.arange(a1 - s0) / SR) + 0.1 * np.sin(2 * np.pi * 100 * np.arange(a1 - s0) / SR)
tv[s0:a1] += noise / np.abs(noise).max() * 0.45 * flutter + hum * 0.4
# включение: щелчок
L = int(0.15 * SR)
tt = np.arange(L) / SR
tv[a1 - L // 3:a1 - L // 3 + L] += (rng.normal(0, 1, L) * np.exp(-tt * 40) + np.sin(2 * np.pi * 60 * tt) * np.exp(-tt * 20)) * 0.5

# ---------- 3. мрак: гул, сердцебиение, удары ----------
dark = np.zeros(N)
on = T['tvOn']
m = t_all >= on
k_end = np.clip((t_all - on) / (T['black'] - on), 0, 1)
drone = np.zeros(N)
for f, g in ((55.0, 1), (55.0 * 1.006, .8), (82.4, .5), (110 * 0.997, .3), (116.5, .18)):   # A1, E2 и диссонанс
    ph = rng.random() * 6.28
    drone += g * np.sign(np.sin(2 * np.pi * f * t_all + ph)) * 0.3 + g * np.sin(2 * np.pi * f * t_all + ph)
drone = fft_filter(drone, hi=400)
dark += drone * m * (0.35 + 0.65 * k_end) * np.clip((t_all - on) / 1.2, 0, 1)


def thump(gain=1.0):
    L = int(0.35 * SR)
    tt = np.arange(L) / SR
    return np.sin(2 * np.pi * np.cumsum(40 + 50 * np.exp(-tt * 30)) / SR) * np.exp(-tt * 10) * gain


# сердцебиение: с «Зачем?...», ускоряется 62 → 118 уд/мин
out_msgs = [x for x in msgs if x['t'] > on]
hb_start = out_msgs[2]['t'] - 0.4
tb = hb_start
while tb < T['black']:
    k = (tb - hb_start) / (T['black'] - hb_start)
    bpm = 62 + 56 * k ** 1.3
    g = 0.5 + 0.7 * k
    place(dark, thump(g), tb)
    place(dark, thump(g * 0.6), tb + 0.22 * 60 / bpm)
    tb += 60 / bpm


def impact(gain):
    L = int(1.6 * SR)
    tt = np.arange(L) / SR
    boom = np.sin(2 * np.pi * np.cumsum(30 + 90 * np.exp(-tt * 9)) / SR) * np.exp(-tt * 2.5)
    crack = fft_filter(rng.normal(0, 1, L), lo=800, hi=6000) * np.exp(-tt * 18) * 0.25
    return (boom + crack) * gain


def swell(dur, gain):
    """обратный «всасывающий» звук перед бликом"""
    L = int(dur * SR)
    tt = np.arange(L) / SR
    n = fft_filter(rng.normal(0, 1, L), lo=1500, hi=9000)
    return n * (tt / dur) ** 3 * gain * 0.6


for j, x in enumerate(out_msgs):
    stage = j / max(1, len(out_msgs) - 1)
    g = 0.25 + 0.9 * stage
    if j >= 2:
        place(dark, swell(0.35, g), x['t'] - 0.35)
    place(dark, impact(g * (1.3 if j == len(out_msgs) - 1 else 1)), x['t'])

# финальный звон-нарастание
fin0 = out_msgs[-2]['t']
L = int((T['black'] - fin0) * SR)
tt = np.arange(L) / SR
ring = np.sin(2 * np.pi * np.cumsum(1800 + 2600 * (tt / tt[-1]) ** 2) / SR) * (tt / tt[-1]) ** 2 * 0.12
place(dark, ring, fin0)

dark = reverb(dark, length=3.5, decay=1.2, mix=0.35, seed=3)

# ---------- 4. звуки интерфейса ----------
sfx = np.zeros(N)


def pop(f0, f1, dur=0.12):
    L = int(dur * SR)
    tt = np.arange(L) / SR
    return np.sin(2 * np.pi * np.cumsum(f0 + (f1 - f0) * tt / dur) / SR) * np.exp(-tt * 30) * np.minimum(1, tt / 0.003)


for x in msgs:
    late = x['t'] > on
    p = pop(900, 1500) if x['side'] == 'out' else pop(1200, 800)
    if late:
        p = pop(500, 700, .16) if x['side'] == 'out' else pop(600, 380, .16)   # ниже и глуше
    place(sfx, p, x['t'], 0.05)
for k in tl['keys']:
    if not k['send']:
        L = int(0.03 * SR)
        place(sfx, fft_filter(rng.normal(0, 1, L), lo=2000, hi=7000) * np.exp(-np.arange(L) / SR * 150), k['t'], 0.012)

# ---------- сведение ----------
def norm(x, db, a, b):
    seg = x[int(a * SR):int(b * SR)]
    return x * (10 ** (db / 20) / (np.sqrt((seg ** 2).mean()) + 1e-12))


light = norm(light, -21, 0.5, T['tvOff'] - 0.2)
tv = norm(tv, -22, T['tvOff'] + 0.3, T['tvOn'] - 0.2)
dark = norm(dark, -20, on + 1, T['black'] - 0.2)
mix = light + tv + dark + sfx
# резкая тишина в момент темноты, затем тихий звон в ушах
b0 = int(T['black'] * SR)
mix[b0:] = 0
tt = np.arange(N - b0) / SR
mix[b0:] += np.sin(2 * np.pi * 4200 * tt) * 0.012 * np.exp(-tt / 0.9) * np.minimum(1, tt / 0.15)

st = np.stack([mix, np.roll(mix, int(0.0006 * SR))], 1)
st = np.tanh(st / 0.85) * 0.85
with wave.open(sys.argv[2], 'wb') as w:
    w.setnchannels(2)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes((st * 32767).astype(np.int16).tobytes())
for s in range(0, int(DUR) + 1, 2):
    sg = st[s * SR:(s + 2) * SR, 0]
    print(f"{s:3d}s {20 * np.log10(np.sqrt((sg ** 2).mean()) + 1e-9):6.1f} dB")
